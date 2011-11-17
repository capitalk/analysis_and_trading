
import glob 
import numpy as np

import dataset 
import dataset_helpers 
import signals 

    
def correlation_matrix(ys, nlags = 100):
    n = len(ys)
    ordinal = isinstance(ys[0], int)
    best_corrs = np.zeros( [n,n], dtype='float' )
    best_lags = np.zeros( [n,n], dtype='int')
    for i, y1 in enumerate(ys):
        if ordinal:
            y1_mask = (y1 != 0)
            y1_count = np.sum(np.abs(y1_mask))
        else:
            y1_mean = np.mean(y1)
            y1_diff = y1 - y1_mean 
            y1_std = np.std(y1)
    
        for j, y2 in enumerate(ys):
            if ordinal:
                y2_mask = (y2 != 0)
                y2_count = np.sum(np.abs(y2_mask))
                normalizer = 0.5 * (y1_count + y2_count)
            else:
                y2_mean = np.mean(y2)
                y2_diff = y2 - y2_mean 
                y2_std = np.std(y2)
                normalizer = y1_std * y2_std
            best_lag = 0 
            best_corr = 0 
            for lag in np.arange(nlags)+1:
                if ordinal:
                    y1_trim = y1[:-lag]
                    y1_mask_trim = y1_mask[:-lag]
                    y2_lagged = y2[lag:]
                    y2_mask_lagged = y2_mask[lag:]
                    mask = y1_mask_trim | y2_mask_lagged 
                    n_agree = np.sum(y1_trim[mask] == y2_lagged[mask])
                    corr = n_agree / normalizer 
                else:
                    prod = y1_diff[:-lag]  * y2_diff[lag:]
                    corr = np.mean(prod) / normalizer
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag 
            best_corrs[i,j] = best_corr
            best_lags [i,j] = best_lag 
    return best_corrs, best_lags 

def correlation_matrix_from_path(path, nlags= 100, signal = signals.prct_curr_tick_midprice_change, start_hour = 4, end_hour=20):
    files = glob.glob(path)
    assert len(files) > 0
    ys = [] 
    for filename in files:
        d = dataset.Dataset(filename)
        start_idx = dataset_helpers.hour_to_idx(d.t, start_hour)
        end_idx = dataset_helpers.hour_to_idx(d.t, end_hour)
        y = signal(d, start_idx = start_idx, end_idx = end_idx)
        assert len(y) > 0 
        ys.append(y)
    return correlation_matrix(ys, nlags), files 
    
    
    
