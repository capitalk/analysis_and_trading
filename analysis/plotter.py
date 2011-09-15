
import numpy as np 
import argparse
import datetime 
import os.path 

import scipy.io
import cPickle 
import pylab
import matplotlib as mpl

import dataset
import expr_lang 

parser = argparse.ArgumentParser(description='Plot market data')
parser.add_argument('filename', help='an HDF file containing processed feature data')

parser.add_argument('-f', '--features',  dest='features', nargs='*', default=[],
                   help='names of features to plot')

parser.add_argument('-x' '--x_axis', default='t', 
                    dest='x_axis', 
                    help='which feature to use as the x-axis')     

parser.add_argument('-o', '--overlay_features',  dest='overlay_features', nargs='*', default=[],
                   help='plot another set of features on a separate y-axis')
                   
parser.add_argument('-f2', '--second_features',  dest='second_features', nargs='*', default=[],
                   help='features for second plot')

#
#parser.add_argument('-x2', '--second_x_axis',  dest='second_x_axis',  default='t',
                   #help='which feature to use for x-axis of second plot')


parser.add_argument('-s', '--start_prct', dest='start_prct', default=0, help='start time as a percentage')
parser.add_argument('-e', '--end_prct', dest='end_prct', default=100, help='end time as a percentage')

#parser.add_argument('-g', '--glumpt', dest='glumpy', default=False, action='store_true', help='use glumpy to plot with OpenGL')

parser.add_argument('-i', '--info', action='store_true', 
                    dest='info',  
                   help='list all available features and timescales')
parser.add_argument('-b' '--buy_signal', action='store_true', dest='buy_signal', 
                        help='show times when algorithm should buy')
parser.add_argument('-p' '--default_plot_style', dest='default_plot_style', default='line',
                        help='plot style: line | vlines | scatter | hist | fill')

    
args = parser.parse_args()


if not os.path.exists(args.filename):
    raise RuntimeError("Couldn't find file: " + args.filename)

data = dataset.Dataset(args.filename)
evaluator = expr_lang.Evaluator() 

    
def mk_datetime(t):
    h = int(t / 3600000) % 24 
    m = int(t / 60000) % 60 
    s = int(t / 1000) % 60 
    milli = int(t % 1000 )
    return datetime.time(h,m,s,milli*1000)
    
def run_plotter(args):
        
    indices = data.get_col('idx')
    n = len(indices)
    start_idx = int((float(args.start_prct) / 100.0) * n)
    end_idx = int((float(args.end_prct) /100.0) * n)
    
    ts = data.get_col('t', start_idx, end_idx)
    if args.x_axis == 't':
        # show time in seconds instead of milliseconds
        xs = ts / 1000.0
    else:
        xs = data.get_col(args.x_axis, start_idx, end_idx)
    if args.info:
        first_time =  mk_datetime(ts[0])
        last_time  = mk_datetime(ts[-1]) 
        
       
        def print_dataset(x):
            if hasattr(data.hdf[x], 'shape'):
                print "\t", x
        print "Contents:"
        data.hdf.visit(print_dataset)
        print "Timescales = ", data.timescales
        print "Aggregators = ", sorted(data.reducers)
        print "Features = ", sorted(data.features)
        print "Num Samples = ", len(ts) 
        print "Start Time = ", first_time
        print "End Time = ", last_time 
        
    def format_timescale(timescale):
        ms = data.timescale_base ** timescale
        if ms < 1000: return str(ms) + "ms"
        else: return str(ms/1000) + "s" 
        
    def format_name(ys_f, ys_r, timescale):
        if ys_r is None or ys_r == 'mean':
            ys_name = ys_f
        else:
            ys_name= ys_r + "[" + ys_f + "]" 
        if timescale:    
            ys_name += " (" + format_timescale(timescale) + ")" 
        return ys_name
        
    def parse_feature(f): 
        lst = f.split("#") 
        
        raw_name = lst[0] 
        
        if len(lst) < 2:
            plot_style = args.default_plot_style 
        else:
            plot_style = lst[1] 
        ys = evaluator.eval_expr(raw_name, start_idx, end_idx, env = data)
        #ys_f = data.parse_feature_name(raw_name) 
        #ys_r = data.parse_reducer(raw_name)
        #timescale = data.parse_timescale(raw_name) 
        #ys_name = format_name(ys_f, ys_r, timescale)
        
        return ys, raw_name, plot_style 
    
    default_plot_colors = ['b', 'g', 'r', 'c', 'm',  'k', 'y']
    def mk_color_gen(colors = default_plot_colors): 
        # stupid python lacks lexical scope 
        color_index_wrapper = {'index':0} 
        
        def get_next_color():
            color_index = color_index_wrapper['index'] 
            c = colors[color_index]
            color_index += 1
            if color_index >= len(colors): 
                color_index = 0
            color_index_wrapper['index'] = color_index 
            return c
        return get_next_color
    
    max_num_plots = 2
    color_gens = [mk_color_gen() for i in range(max_num_plots)]
    
    
    def plot_feature_list(fs,  axes, color_gen):
        
        legendNames = []
        for f in fs:
            color = color_gen()
            # parse out the plot style 
            ys, ys_name, plot_style = parse_feature(f) 
            if plot_style == 'vlines':
                axes.vlines(xs, 0, ys, alpha=0.5, colors=color)
            elif plot_style == 'line': 
                axes.plot(xs, ys, color=color)
            elif plot_style == 'scatter':
                axes.scatter(xs, ys, color=color)
            elif plot_style == 'hist':
                axes.hist(ys, 200, color=color)
            elif plot_style == 'acorr':
                axes.acorr(ys, color=color)
            elif plot_style == 'fill':
                axes.fill(xs, ys, color=color, alpha=0.5)
            elif plot_style == 'moving_hist':
                import matplotlib.image as mpimg
                bin_edges, reduced_ts, matrix = windowed_level_hist(xs, ys)
                pylab.matshow(matrix.T, cmap = mpl.cm.spectral)
                pylab.xlabel(reduced_ts)
                bin_count = len(bin_edges)
                if bin_count < 50:
                    pylab.yticks(np.arange(bin_count), bin_edges)
                else:
                    bin_indices = np.arange(0, bin_count, bin_count/50)
                    pylab.yticks(bin_indices, bin_edges[bin_indices])
                return []
            else:
                raise RuntimeError("Unknown plot style:" + plot_style )
                
            legendNames.append(ys_name) 
        return tuple(legendNames)
    
        
    if len(args.features) > 0 or args.buy_signal:
        
        if len(args.second_features) > 0: nrows = 2
        else: nrows = 1
        
        fig = pylab.figure(1)
        axes = fig.add_subplot(nrows, 1, 1)
        axes.grid(True, alpha=0.1)
        color_gen = color_gens[0] 
        legendNames = plot_feature_list(args.features, axes, color_gen)
        
        if len(args.overlay_features) > 0:
            if len(legendNames) > 0: axes.legend(legendNames, loc='upper left')
            axes_twin = axes.twinx()
            
            legendNames2 = plot_feature_list(args.overlay_features, axes_twin, color_gen)
            if len(legendNames2) > 0:
                axes_twin.legend(legendNames2, loc='upper right')
        else:
            if len(legendNames) > 0: axes.legend(legendNames)
        axes.set_xlabel(args.x_axis)
        
        if args.buy_signal: 
            import signals
            print "Generating buy signal..." 
            vline_indicator = signals.aggressive_profit(data, start=start_idx, end=end_idx)
            vline_indices = np.nonzero(vline_indicator)[0] 
            print "Found ", len(vline_indices), " buy signals (of ", len(ts), " total timestamps)" 
            if len(vline_indices) > 0:
                vline_bottom = -1
                vline_top = 1
                ylim = axes.get_ylim()
                vline_bottom = ylim[0]
                vline_top = ylim[1] 
                axes.vlines(xs[vline_indices], vline_bottom, vline_top, alpha=0.4, color=(1.0,1.0,0,0))
                
        if len(args.second_features) > 0:
            axes2 = fig.add_subplot(nrows, 1, 2)
            axes2.grid(True, alpha=0.1)
            color_gen2 = color_gens[1]
            secondLegendNames = plot_feature_list(args.second_features, axes2, color_gen2)
            axes2.legend(secondLegendNames)
            
        title = os.path.split(args.filename)[1]
        title, ext  = os.path.splitext(title)
        pylab.title(title)
        pylab.show()
        
    

# if not given anything to do, enter interactive mode 
if args.info or len(args.features)>0:
    run_plotter(args)
else:
    s = ""
    while True:
        s = raw_input('==> ')
        s = s.strip()
        if s == 'exit' or s == 'quit':
            break
        elif s == 'help':
            print "COMMANDLINE OPTIONS:" 
            parser.print_help()
            print 
            print "INTERACTIVE HELP:"
            print "Current open file -- ", args.filename 
            print "* Type'help' to see this screen again'"
            print "* Type 'quit' to return to the command prompt'"
            print "* Type '-i' to see the datasets available in this file"
            print 
            print "To create a plot, type the '-f' keyword followed by any features you would like to see. \
In addition to features, it's also possible to plot mathematical expressions, which must be surrounded in quotes. \
For example, to plot a day's basic price movement you can type '-f bid offer'. \
If you intead want to  instead see the spread you can type '-f \"offer - bid\". \n \
The plot style of any feature can be changed by appending the '#' character and plot style name to the feature (e.g. bid/mean/60s#vlines)."
            print 
            print "To use more advanced features (such as overlaying two plots or vertically stacking plots), see commandline options above." 
        else:
            tokens = [args.filename] + parser.convert_arg_line_to_args(s)
            print tokens
            try: 
                args2 = parser.parse_args(tokens)
            except: 
                print "Parser error"
                continue
            try:
                run_plotter(args2)
            except (RuntimeError, ValueError) as e:
                print e
                continue 
