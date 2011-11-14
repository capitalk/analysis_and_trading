
import glob 
import numpy as np

import dataset 
import dataset_helpers 
import signals 

def correlation_matrix(ys, nlags = 20):
    n = len(ys)
    best_corrs = np.zeros( [n,n], dtype='float' )
    best_lags = np.zeros( [n,n], dtype='int')
    for i, y1 in enumerate(ys):
        y1_mean = np.mean(y1)
        y1_diff = y1 - y1_mean 
        y1_std = np.std(y1)
        for j, y2 in enumerate(ys):
            y2_mean = np.mean(y2)
            y2_diff = y2 - y2_mean 
            y2_std = np.std(y2)
            normalizer = y1_std * y2_std
            best_lag = 0 
            best_corr = 0 
            for lag in np.arange(nlags)+1: 
                prod = y1_diff[:-lag] * y2_diff[lag:]
                corr = np.mean(prod) / normalizer 
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag 
            best_corrs[i,j] = best_corr
            best_lags[i, j] = best_lag 
    return best_corrs, best_lags 

def correlation_matrix_from_path(path, nlags= 20, signal = signals.prct_curr_tick_midprice_change, start_hour = 6, end_hour=20):
    files = glob.glob(path)
    assert len(files) > 0
    ys = [] 
    for filename in files:
        d = dataset.Dataset(filename)
        start_idx = dataset_helpers.hour_to_idx(d.t, start_hour)
        end_idx = dataset_helpers.hour_to_idx(d.t, end_hour)
        ys.append(signal(d, start_idx = start_idx, end_idx = end_idx))
    return correlation_matrix(ys, nlags), files 
    
    
    
