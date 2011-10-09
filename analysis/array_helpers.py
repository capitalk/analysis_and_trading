import numpy as np 
import scipy.weave 

def find_first_gte(x,v):
    code = """
        int nx = Nx[0];
        double vf = PyFloat_AsDouble(v);
        return_val = -1; 
        for (int i = 0; i < nx; ++i) { 
            if (x[i] >= vf) {
                return_val = i;
                break;
            }
        }
        """
    idx = scipy.weave.inline(code, ['x', 'v'], verbose=2)
    if idx == -1: return None
    else: return idx 


def find_first_lte(x,v):
    code = """
        int nx = Nx[0];
        double vf = PyFloat_AsDouble(v);
        return_val = -1; 
        for (int i = 0; i < nx; ++i) { 
            if (x[i] <= vf) {
                return_val = i;
                break;
            }
        }
        """
    idx = scipy.weave.inline(code, ['x', 'v'], verbose=2)
    if idx == -1: return None
    else: return idx 

def find_first(x):
    indices = np.nonzero(x)[0] 
    if len(indices) > 0: return indices[0]
    else: return None 
    
def find_future_index(ts, curr_idx, future_time, max_per_millisecond=None):
    t = ts[curr_idx] 
    dt = future_time - t  # in milliseconds
    if max_per_millisecond is None: 
        max_per_millisecond = density_per_millisecond(dt)
    future_horizon = curr_idx + max_per_millisecond * dt + 1
    candidate_times = ts[curr_idx:future_horizon]
    last_index = curr_idx+find_sorted_index(candidate_times, future_time)
    return last_index 


def check_data(x):
    inf_mask = np.isinf(x)
    if np.any(inf_mask):
        raise RuntimeError("Found inf: " + str(np.nonzero(inf_mask)))
        
    nan_mask = np.isnan(x)
    if np.any(nan_mask):
        raise RuntimeError("Found NaN: " + str(np.nonzero(nan_mask)))
        
    same_mask = (np.std(x, 0) <= 0.0000001)
    if np.any(same_mask):
        raise RuntimeError("Column all same: " + str(np.nonzero(same_mask)))

def clean(x):
    """remove any infinities or NaN's from input vector""" 
    return x[np.logical_not(np.logical_or(np.isinf(x), np.isnan(x)))]

def remove_isolated(pred, win_size = 5):
    """remove any non-zero elements that occur in isolation"""
    pred = pred.copy()
    for i, curr in enumerate(pred):
        first_idx = max(i-win_size+1, 0)
        last_idx = i+win_size
        if (curr != 0) and np.all(pred[first_idx:i]==0) and np.all(pred[i+1:last_idx] == 0):
            pred[i] = 0
    return pred 
               
