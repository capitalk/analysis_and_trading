
import numpy as np
import scipy.stats as stats

def last(x):
    return x[-1] 

def prct_zero(x):
    return np.sum(x == 0) / float(len(x))

def mean_absolute_difference(x):
    m = np.mean(x)
    return np.mean(np.abs(x - m))

def median_absolute_deviation(x):
    m = np.median(x)
    return np.median(np.abs(x - m))

def mean_and_std(x):
    m = np.mean(x)
    diff = x - m 
    s = np.mean(diff*diff)
    return m,s

def median_and_deviation(x):
    med = np.median(x)
    deviation = np.median(np.abs(x-med))
    return med, deviation 

import scipy.stats as stats 

def tmin(xs): 
    return stats.scoreatpercentile(xs,10)
def tmax(xs): 
    return stats.scoreatpercentile(xs, 90)
    
def tmean(xs):
    if len(xs) > 3: 
        return stats.tmean(xs, limits=(tmin(xs), tmax(xs)), inclusive=(True,True))
    else:
        return np.mean(xs)
            
def trange(xs):
    return tmax(xs) - tmin(xs) 
    
def iqr(x):
    return stats.scoreatpercentile(x, 75) - stats.scoreatpercentile(x, 25)

def approx_skewness(x):
    m = np.mean(x)
    nless = np.sum(x < m)
    nmore = np.sum(x > m)
    n = float(len(x))
    return (nless - nmore) / n 

# k-point slope approximation
def approx_slope(t, x, k=5):
    n = len(x)
    nslopes = (k*k - k) / 2  
    slopes = np.zeros(nslopes)
    idx = 0 
    sample_indices = xrange(0,n, n/k + 1)
    idx = 0
    for i in xrange(0,k):
        for j in xrange(k-i-1):
            first_sample_index = sample_indices[i]
            second_sample_index = sample_indices[i+j+1]
            x0 = x[first_sample_index]
            t0 = t[first_sample_index]
            x1 = x[second_sample_index]
            t1 = t[second_sample_index]
            slopes[idx] = (x1 - x0) / float(t1 - t0)
            idx += 1
    return np.median(slopes)        

def ols(t, window):
    npoints = len(window)
    # even if there are two points, we'll get noisy slopes, so ignore it 
    if npoints < 2:
        return 0.0
    else:
        # we want rate per second, not millisecond, so normalize time
        normalized_time_slice = (t - t[0])
        C = np.cov(normalized_time_slice, window, bias=1)
        c = C[0,1] # covariance between time and value 
        v = C[0,0] # variance of time 
        slope = c / v 
        return slope 
        
def ols_1000x(t, window):
    return ols(t/1000.0, window) 
    
# delta is ratio of error variances between time and feature 
def deming_regression(t, window, delta = 1):
    npoints = len(window)
    # even if there are two points, we'll get noisy slopes, so ignore it 
    if npoints < 2:
        return 0.0
    else:
        normalized_time_slice = (t - t[0]) 
        C = np.cov(normalized_time_slice, window, bias=1)
        c = C[0,1] # covariance between time and value 
        v_t = C[0, 0] # variance of time 
        v_w = C[1,1] # variance of feature 
        diff = v_w - delta * v_t
        top = diff + np.sqrt(diff*diff + 4 * delta * c * c)
        bottom = 2 * c
        slope = top / bottom
        return slope 
        
def index_weighted_sum(x):
    weights = np.arange(len(x))+1
    total = np.dot(weights, x)
    return total / float(np.sum(weights))


# get the means of multiple windows simultaneouslt
# each window runs from [start_indices[i], i] inclusively 
def quick_windowed_means(frames, start_indices):
    n = len(start_indices)
    cumsum = np.cumsum(frames)
    means = np.zeros(n)
    for (i, start_idx) in enumerate(start_indices):
        if start_idx > 0: start_val = cumsum[start_idx-1]
        else: start_val = 0
        window_size = i - start_idx + 1
        means[i] = (cumsum[i]  - start_val) / float(window_size)
    return means

def quick_windowed_min(frames, start_indices, stride=1):
    n = len(frames)
    mins = np.zeros(n)
    last_min = None
    last_idx = -1000
    for i, start_idx in enumerate(start_indices):
        # we've either haven't shifted the start or we only shifted by 1 point and 
        # it wasn't the min 
        if (start_idx == last_idx) or (start_idx - last_idx == 1 and frames[last_idx] != last_min):
            last_min = min(last_min, frames[i])
        else:
            last_min = np.min(frames[start_idx:i+1:stride])
        mins[i] = last_min 
        last_idx = start_idx 
    return mins 

def quick_windowed_max(frames, start_indices, stride=1):
    n = len(frames)
    maxes = np.zeros(n)
    last_max = None
    last_idx = -1000
    for i, start_idx in enumerate(start_indices):
        # we've either haven't shifted the start or we only shifted by 1 point and 
        # it wasn't the min 
        if (start_idx == last_idx) or (start_idx - last_idx == 1 and frames[last_idx] != last_max):
            last_max = max(last_max, frames[i])
        else:
            last_max = np.max(frames[start_idx:i+1:stride])
        maxes[i] = last_max
        last_idx = start_idx 
    return maxes
    
    
def mean_crossing_rate(x, mean = None): 
    if mean is None: mean = np.mean(x) 
    x = x - mean 
    lagged_product = x[1:] * x[:-1]
    crossing_indicator = lagged_product  < 0 
    crossing_count = np.sum(crossing_indicator)
    return crossing_count / float(len(x))
    

    
