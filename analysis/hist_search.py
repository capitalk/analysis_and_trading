
import train 
import dataset 
import analysis 
import numpy as np

# assumes histograms have same bin boundaries 
# and compute similarities as sum of elementwise min of bin probabilities
def hist_sim(h1, h2):
    assert len(h1) == len(h2)
    p1 = h1 / np.sum(h1)
    p2 = h2 / np.sum(h2)
    return np.sum(np.minimum(p1, p2))


# return n equally spaced thresholds between the min and max of the input 
def gen_thresholds(x, n = 8):
    minval = np.min(x)
    maxval = np.max(x)
    r = maxval - minval 
    step = r / (n + 2)
    thresholds = np.arange(minval + step, maxval - step, step)
    return thresholds 
    

def hist_search(d, nbins = 100, nthresholds=10, longest_wait_time = 10000):
    ts = d['t/100ms']
    bids = d['bid/100ms']
    offers = d['offer/100ms']
    
    up, down = analysis.time_til_aggressive_profit(ts, bids, offers, prct = 0.0001, max_search = 60000, plot=False)
    # TODO: Also use down 
    
    inf_wait = up > longest_wait_time  
    log_wait_times = np.log10(up) 
    
    finite_wait = ~inf_wait 
    
    results = {} 
    
    best_sim = 1.0 
    best_feature = None
    best_thresh = None 
    best_times_above = None
    best_times_below = None 
    best_bin_edges = None
    for fname in train.features: 
        print "Testing", fname 
        x = d[fname]
        thresholds = gen_thresholds(x, nthresholds)
        for t in thresholds:
            above = x >= t
            below = x < t 
            
            num_above = np.sum(above)
            num_below = np.sum(below)
            
            
            num_inf_wait_above = np.sum(inf_wait[above])
            num_inf_wait_below = np.sum(inf_wait[below])
            
            prob_inf_above = num_inf_wait_above / float(num_above)
            prob_inf_below = num_inf_wait_below / float(num_below) 
            
            print "Threshold = %s, percent infinite above = %s, percent infinite below = %s" % (t,  prob_inf_above, prob_inf_below)
            finite_above = log_wait_times[finite_wait & above]
            finite_below = log_wait_times[finite_wait & below] 
            
            if (len(finite_above) > 2*nbins) and (len(finite_below) > 2*nbins):  
                h_above, bins = np.histogram(finite_above, nbins)
                h_below, bins2 = np.histogram(finite_below, bins = bins ) 
            
                # distribution over all finite waiting time bins 
                p_above = h_above / float(num_above)
                p_below = h_below / float(num_below)
            
                partial_sim = np.sum(np.minimum(p_above, p_below))
            
                
                sim = partial_sim + min(prob_inf_above, prob_inf_below) 
                results[(fname, t)] = sim 
            
                if sim < best_sim:
                    best_sim = sim 
                    best_thresh = t
                    best_feature = fname 
                    best_times_above = finite_above
                    best_times_below = finite_below
                    best_bin_edges = bins 
                print "%s >= %s : %s" % (fname, t, sim)
    mean_above = 10 ** np.mean(best_times_above)
    mean_below = 10 ** np.mean(best_times_below) 
    print "Best: %s >= %s (sim = %s, mean_above = %s, mean_below=%s)" % (best_feature, best_thresh, best_sim, mean_above, mean_below)
    import pylab
    pylab.subplot(211)
    pylab.hist(best_times_above, bins = best_bin_edges)
    pylab.title('wait times above thresh')
    pylab.subplot(212)
    pylab.hist(best_times_below, bins = best_bin_edges)
    pylab.title('wait time below thresh')
    return results, best_times_above, best_times_below  
