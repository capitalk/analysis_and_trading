import numpy as np
import scipy 
import scipy.stats 

def exponential_smoothing(xs, alpha=0.75):
    xs = np.asarray(xs)
    n = xs.shape[0]
    smoothed = np.zeros(n, dtype='float')
    smoothed[0] = xs[0]
    code = """
        for (int i = 1; i < n; ++i) {
            smoothed[i] = alpha * xs[i] + (1-alpha) * smoothed[i-1]; 
        }
        """
    scipy.weave.inline(code, ['smoothed', 'xs', 'alpha', 'n'], verbose=2)
    return smoothed
    
def multiscale_exponential_smoothing(xs, n_scales = 4, base = 4):
    xs = np.asarray(xs, dtype='float')
    n = xs.shape[0]
    smoothed = np.zeros([n_scales, n], dtype='float')
    for i in xrange(n_scales):
        lag = base ** i 
        alpha = 1.0 / lag 
        smoothed[i, :] = exponential_smoothing(xs, alpha)
    return smoothed

def simple_smoothing(xs, scale):
    csum = np.cumsum(xs)
    result = np.zeros_like(xs)
    result[scale:] = csum[scale:] - csum[:-scale]
    result /= scale
    return result 

def multiscale_simple_smoothing(xs, n_scales = 4, base=4):
    xs = np.array(xs, dtype='float')
    n = xs.shape[0]
    result = np.zeros( [n_scales, n] )
    window_sizes = base**np.arange(n_scales)
    csum = np.cumsum(xs)
    for j, wsize in enumerate(window_sizes):
        first_vals = csum[:-wsize]
        last_vals = csum[wsize:]
        result[j, wsize:] = last_vals - first_vals 
    # to turn a difference of sums into a difference of averages, 
    # divide out the window sizes
    for j, wsize in enumerate(window_sizes):
        result[j, :] /= wsize 
    return result 


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

def simple_multiscale_gradients(xs, n_scales=4, base=4):
    dx = padded_delta(xs, lag=1)
    gradients = np.zeros([n_scales, dx.shape[0]], dtype='float')
    for i in xrange(n_scales):
        gradients[i, :] = simple_smoothing(dx, base ** i)
    return gradients 
    
def exponential_multiscale_gradients(xs, n_scales=4, base=4): 
    dx = padded_delta(xs, lag=1)
    gradients = np.zeros([n_scales, dx.shape[0]], dtype='float')
    for i in xrange(n_scales):
        scale = base ** i
        alpha = 1.0 / scale
        gradients[i, :] = exponential_smoothing(dx, alpha=alpha)
    return gradients 
        
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
    
