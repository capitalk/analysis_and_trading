import numpy as np 
import glob 
from array_helpers import check_data 
from aws_helpers import * 
from expr_lang import Evaluator 
from dataset import Dataset
from bisect import bisect_left




def dataset_to_feature_matrix(d, features, start_idx=None, end_idx=None): 
    ev = Evaluator() 
    ncols = len(features)
    t = d['t'][start_idx:end_idx]
    nrows = len(t)
    print "feature matrix shape:", [nrows, ncols]
    result = np.zeros( [nrows, ncols] )
    for (idx, f) in enumerate(features):
        print "Retrieving feature ", f
        vec = ev.eval_expr(f, env = d, start_idx=start_idx, end_idx=end_idx)
        if np.any(np.isnan(vec)):
            print "Warning: NaN in", f
        elif np.any(np.isinf(vec)):
            print "Warning: inf in", f
        result[:, idx] = vec
    return result

def hour_to_idx(t, hour):
    milliseconds = hour * 3600000
    #print "finding", milliseconds, "in range", t[0], t[-1] 
    return bisect_left(t, milliseconds)

import signals 
import features 
def load_file(filename, feature_list, signal, start_hour=None, end_hour=None):
    d = Dataset(filename)
    start_idx = None if start_hour is None else hour_to_idx(d.t, start_hour)
    end_idx = None if end_hour is None else hour_to_idx(d.t, end_hour)
    
    if feature_list:
        x = dataset_to_feature_matrix(d, feature_list, start_idx=start_idx, end_idx=end_idx)
    else:
        x = None 
        
    if signal:
        y = signal(d, start_idx = start_idx, end_idx = end_idx)
    else:
        y = None
        
    return x, y 


# for some reason the builtin numpy hstack seems to freeze
# or take a very long time, so wrote my own 
def hstack(arrays):
    if len(arrays) == 0: return np.empty([0])
    else:
        arrays = map(np.atleast_1d, arrays)
        dtype = arrays[0].dtype
        n_rows = arrays[0].shape[0]
        shapes = [array.shape for array in arrays]
        col_sizes = [s[1] if len(array.shape) > 1 else 1 for s in shapes]
        total_cols = sum(col_sizes)
        result = np.zeros([n_rows, total_cols], dtype=dtype, order='F')
        col_idx = 0 
        for array in arrays:
            
            assert array.shape[0] == n_rows 
            if len(array.shape) < 2:
                array = np.atleast_2d(array).T
            n_cols = array.shape[1]
            result[:, col_idx:col_idx + n_cols] = array
            col_idx += n_cols
        return result
            

def load_files(path, feature_list = ['midprice'], signal = None, start_hour = None, end_hour = None):
    files = glob.glob(path)
    xs = []
    ys = []
    for filename in files:
        print "Loading", filename 
        x, y = load_file(
            filename, 
            feature_list = feature_list, 
            signal = signal, 
            start_hour = start_hour, 
            end_hour = end_hour) 
        if x is not None: xs.append(x) 
        if y is not None: ys.append(y)
    print "Concatentating", len(xs), "data matrices"
    x = hstack(xs)
    print "Concatentating", len(ys), "signal vectors"
    y = hstack(ys)
    return x, y, files 



def load_s3_data(files, features, signal_fn, start_hour=None, end_hour=None): 
    """Reads given filenames from s3, returns:
        - feature matrix (based on the features argument)
        - output signal (based on given signal_fn)
        - times 
        - bids
        - offers
        - currency pair"""
    print "Loading datasets..."
    datasets = load_s3_files(files) 
    print "Flattening datasets into feature matrices..." 
    
    matrices = []
    signals = []
    times = []
    bids = []
    offers = []
    currencies = [] 
    
    for d in datasets:
        print "Processing dataset", d.hdf 
        start_idx = None if start_hour is None else hour_to_idx(d.t, start_hour) 
        end_idx = None if end_hour is None else hour_to_idx(d.t, end_hour)
        print "start_idx", start_idx, "end_idx", end_idx 
        
        start_ok = start_idx is None or start_idx < len(d.t)
        end_ok = end_idx is None or (end_idx < len(d.t) and end_idx > start_idx)
        
        if start_ok and end_ok:
            mat = dataset_to_feature_matrix(d, features, start_idx=start_idx, end_idx=end_idx)
            matrices.append(mat)
            times.append(d.t[start_idx:end_idx])
        
            bids.append(d['bid/100ms'][start_idx:end_idx])
            offers.append(d['offer/100ms'][start_idx:end_idx])
            currencies.append(d.currency_pair)
        
            print "Generating output signal"
            signals.append(signal_fn(d, start_idx=start_idx, end_idx=end_idx))
        
    feature_data = np.concatenate(matrices)
    print "Checking data validity..."
    check_data(feature_data) 
    
    signal = np.concatenate(signals)
    times = np.concatenate(times)
    bids = np.concatenate(bids)
    offers = np.concatenate(offers)
    
    print "Deleting local files..." 
    for d in datasets:
        d.hdf.close()
    return feature_data, signal, times, bids, offers, currencies   
