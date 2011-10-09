from aws_helpers import * 
from expr_lang import Evaluator 
from dataset import Dataset

def dataset_to_feature_matrix(d, features): 
    ev = Evaluator() 
    ncols = len(features)
    nrows = len(d['t'])
    print "feature matrix shape:", [nrows, ncols]
    result = np.zeros( [nrows, ncols] )
    for (idx, f) in enumerate(features):
        print "Retrieving feature ", f
        vec = ev.eval_expr(f, env = d)
        if np.any(np.isnan(vec)):
            print "Warning: NaN in", f
        elif np.any(np.isinf(vec)):
            print "Warning: inf in", f
        result[:, idx] = vec
    return result


def load_s3_data(files, features, signal_fn): 
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
    matrices = [dataset_to_feature_matrix(d, features) for d in datasets] 
    feature_data = np.concatenate(matrices)
    print "Checking data validity..."
    check_data(feature_data) 
    print "Generating output signal..."
    signal = np.concatenate([signal_fn(d) for d in datasets] )
    times = np.concatenate([d['t/100ms'] for d in datasets])
    bids = np.concatenate( [d['bid/100ms'] for d in datasets])
    offers = np.concatenate( [d['offer/100ms'] for d in datasets])
    currencies = [d.currency_pair for d in datasets]
    print "Deleting local files..." 
    for d in datasets:
        d.hdf.close()
    return feature_data, signal, times, bids, offers, currencies   
