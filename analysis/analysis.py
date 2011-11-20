import bisect 
from bisect import bisect as find_sorted_index
import pylab
import numpy as np
import scipy 
import scipy.stats 

from array_helpers import * 
from dataset_helpers import * 

def multiscale_gradient(xs, n_scales=4, base=4):
    xs = np.array(xs, dtype='float')
    n = xs.shape[0]
    result = np.zeros( [n_scales, n] )
    window_sizes = base**np.arange(n_scales)
    csum = np.cumsum(xs)
    js = range(n_scales)
    for j, wsize in enumerate(window_sizes):
        start_offset = 2*wsize
        midpoints = csum[wsize:-wsize]
        first_vals = csum[0:-2*wsize]
        last_vals = csum[2*wsize:]
        old_sums = midpoints - first_vals 
        new_sums = last_vals - midpoints 
        result[j, start_offset:] = new_sums - old_sums 
    # to turn a difference of sums into a difference of averages, 
    # divide out the window sizes
    for j, wsize in enumerate(window_sizes):
        result[j, :] /= wsize 
    return result 
    
    
def windowed_std(xs, lag=10): 
    xs = np.atleast_1d(xs)
    results = np.zeros_like(xs)
    n = xs.shape[0] 
    ends = np.arange(n-lag) + lag 
    for i in xrange(n - lag):
        j = ends[i]
        window = xs[i:j]
        results[j] = np.std(window)
    return results 
    
def windowed_variance(xs, lag=10):
    xs = np.atleast_1d(xs)
    results = np.zeros_like(xs)
    n = xs.shape[0] 
    for i in xrange(n - lag):
        j = i+lag
        window = xs[i:j]
        results[j] = np.var(window)
    return results 
    
def delta(ys, lag=1):
    if np.rank(ys) == 2:
        return ys[lag:, :] - ys[:-lag, :]
    else:
        return ys[lag:] - ys[:-lag]


def padded_delta(ys, lag=1, side='left', prct=False):
    ds = delta(ys, lag)
    if prct:
        ds /= ys[:-lag]
    result = np.zeros_like(ys)
    r = np.rank(ys)
    if side=='left' and r == 2:
        result[lag:, :] = ds
    elif side == 'left' and r == 1:
        result[lag:] = ds 
    elif side == 'right' and r == 2: 
        result[:-lag, :] = ds
    elif side == 'right' and r == 1:
        result[:-lag] = ds 
    else:
        assert False
    return  result 
    
def counts_to_probs(xs):
    xs = np.atleast_2d(xs)
    nrows, ncols = xs.shape 
    probs = np.zeros( [nrows, ncols] )
    sums = np.sum(xs, axis=1)
    mask = sums > 0
    normalizer = np.array([sums[mask]], dtype='float').T
    probs[mask] = xs[mask] / normalizer
    return probs 

def cumulative_tick_movement_counts(xs, n):
    xs = np.atleast_1d(xs)
    deltas = xs[1:] - xs[:-1]
    ticks = np.sign(deltas)
    counts = np.zeros([2*n+1, 3], dtype='int')
    
    for i, x in enumerate(ticks[1:]):
        subseq = ticks[i-n:i]
        cumulative = np.sum(subseq)
        row = n + cumulative 
        counts[row, x + 1] += 1
    return counts 



def cumulative_tick_movement_probs(xs, n):
    counts = cumulative_tick_movement_counts(xs, n)
    return counts_to_probs(counts)

def sign_seq_counts(xs, n):
    """
    probability of sign(delta) = +1/-1 after k previous ticks in the same direction
    ----
    xs : float array, any time series
    n : int, max length of sequence 
    """
    xs = np.atleast_1d(xs)
    deltas = xs[1:] - xs[:-1] 
    ticks = np.sign(deltas)
    all_counts = {}
    DOWN_COL = 0
    SAME_COL = 1
    UP_COL = 2
    
    for past_direction in [-1, 0, 1]:
        all_counts[past_direction] = np.zeros([n, 3], dtype='int')
        
    counter = 0 
    sign = ticks[0]
    
    for x in ticks[1:]:
        if counter >= n: 
            counter = n - 1
        
        counts = all_counts[sign]
        
        if x > 0: 
            counts[counter, UP_COL] += 1
        elif x == 0:
            counts[counter, SAME_COL] += 1
        else:
            counts[counter, DOWN_COL] += 1
        
        if x == sign:
            counter += 1
        else: 
            counter = 0
            sign = x 
            
    return all_counts

    
def sign_seq_probs(xs, n):
    all_counts = sign_seq_counts(xs, n)
    all_probs = {}
    for k, counts in all_counts.items():
        all_probs[k] = counts_to_probs(counts)
         
    return all_probs
    
def plot_seq_probs(probs, title=""):
    import pylab
    # only plot the up and down probs 
    pylab.plot(probs[:, [0, 2]])
    pylab.xlabel('length of same-direction sequence')
    pylab.ylabel('prob of next tick movement')
    pylab.legend(['downward', 'upward' ])
    pylab.title(title)
    pylab.show()
    

def density_per_millisecond(t): 
    if t <= 3: return 50
    elif t <= 30: return 15
    elif t <= 100: return 8
    elif t <= 1000: return 1
    else: return 0.1 
    

# dt is in seconds
def slope_distribution(ts, xs, ys=None, dt_seconds=1.0, just_diff=False, hist=True, log_scale=False, xlabel=None, ylabel=None, title=None):
    if ys is None: ys = xs
    dt_milliseconds = int(dt_seconds*1000)
    n = len(ys)
    last_time = ts[-1] 
    last_idx = find_future_index(ts, n/2, last_time-dt_milliseconds)
    results = np.zeros(last_idx)
    for i in xrange(last_idx):
        t = ts[i] 
        j = find_future_index(ts, i, t+dt_milliseconds)
        diff = xs[i] - ys[j]
        if just_diff: results[i] = difference
        else: results[i] = diff / dt_seconds
    if hist:
        pylab.figure()
        pylab.hist(results, 100, log=log_scale)
        pylab.show()
        if xlabel: pylab.xlabel(xlabel)
        if ylabel: pylab.ylabel(ylabel)
        if title: pylab.title(title)
    return results


def windowed_level_hist(ts, xs,  win_size=1000):
    unique_xs = sorted(np.unique(xs))
    nlevels = len(unique_xs)
    numbering = {}
    counter = 0 
    for x in sorted(unique_xs):
        numbering[x] = counter
        counter += 1
    print "Computing windowed histograms..."
    n = len(ts)
    bin_cumulative_sums = np.zeros( [n, nlevels] )
    for i in xrange(n):
        x = xs[i] 
        if i > 0:
            bin_cumulative_sums[i, :] = bin_cumulative_sums[i-1, :] 
        bin = numbering[x] 
        bin_cumulative_sums[i, bin] += 1
    
    
    result= np.zeros( [n, nlevels])
    for i in xrange(n-1):
        j = i + 1
        t = ts[j] 
        end_idx = bisect.bisect_right(ts, t, 0, n)
        start_idx = bisect.bisect_left(ts, t - win_size, 0, end_idx)
        result[i, :] = bin_cumulative_sums[end_idx, :] - bin_cumulative_sums[start_idx, :]
        result[i, :] /= np.sum(result[i, :])
    return result 


def windowed_entropy(ts, xs,  win_size=1000):
    levels = windowed_level_hist(ts, xs, win_size)
    n = levels.shape[0]
    result = np.zeros(n)
    normalizer = np.log(len(bin_edges))
    for i in xrange(n):
        col = levels[i, :]
        pos_col = col[col > 0.00001]
        result[i] = np.sum(-np.log(pos_col) * pos_col)/normalizer 
    return result 


# percentage of each column explained by mode
def probability_of_mode(ts, xs, win_size =1000):
    n = len(ts)
    result= np.zeros(n)
    for i in xrange(n-1):
        j = i + 1
        t = ts[j] 
        start_idx = bisect.bisect_left(ts, t - win_size, 0, j)
        window = xs[start_idx:j]
        if len(window) < 2:
            result[i] = 1.0
        else:
            modes, counts = scipy.stats.mode(window)
            result[i] = counts[0] / float(len(window))
    return result 




def time_til_movement(ts, xs, prct = 0.0020, max_search=10000):
    n = len(xs)
    prct_up = 1 + prct 
    prct_down = 1 - prct 
    results = np.zeros(n)
    maxval = np.max(xs)
    minval = np.min(xs)
    for i in xrange(n):
        t = ts[i]
        x = xs[i]
        target_up = x * prct_up
        target_down = x * prct_down 
        
        too_high = target_up > maxval
        too_low = target_down < minval 
        if too_low and too_high:
            continue
            
        future = xs[(i+1):(i+max_search)]
        up_idx = find_first_gte(future, target_up)
        down_idx = find_find_lte(future, target_down)
        if up_idx is None and down_idx is None:
            t_future = np.inf 
        elif up_idx is None:
            t_future = ts[down_idx + i + 1]
        elif down_idx is None:
            t_future = ts[up_idx + i +1]
        else:
            idx = min(up_idx, down_idx)
            t_future = ts[idx+i+1]
        results[i] = t_future - t
    return results 

    
def time_til_aggressive_profit(ts, bids, offers, prct = 0.0001, max_search = 10000, plot=True, logscale=True):
    n = len(ts)
    
    up_times = np.zeros(n)
    down_times = np.zeros(n)
    
    prct_up = 1 + prct
    prct_down = 1 - prct 
    max_bid = np.max(bids)
    min_offer = np.min(offers)
    _inf = np.inf 
    for i in xrange(n):
        t = ts[i]
        bid = bids[i]
        offer = offers[i] 
        target_up = offer * prct_up
        target_down = bid * prct_down 
        next = i+1
        last = i + max_search 
        
        if target_up <= max_bid:
            future_bids = bids[next:last]
            above_idx = find_first_gte(future_bids, target_up)
        else:
            above_idx = None
            
        if target_down >= min_offer:
            future_offers = offers[next:last]
            below_idx = find_first_lte(future_offers, target_down)
        else: 
            below_idx = None 
            
        if above_idx is None: 
            up_times[i] = _inf
        else: 
            above_t = ts[above_idx + i + 1]
            up_times[i] = above_t - t
        
        if below_idx is None: 
            down_times[i] = _inf 
        else: 
            below_t = ts[below_idx + i + 1]
            down_times[i] = below_t - t    
         
    if plot:
        pylab.figure() 
        up2 = clean(up_times)
        if logscale: up2 = np.log10(up2)
        pylab.subplot(211); 
        pylab.hist(up2, 200); 
        pylab.title('Log waiting time for profitable upward movement')
        
        down2 = clean(down_times)
        if logscale: down2 = np.log10(down2)
        pylab.subplot(212); 
        pylab.hist(down2, 200); 
        pylab.title('Log waiting time for profitable downward movement')

    return up_times, down_times 
    
