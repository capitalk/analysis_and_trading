
features = [ 
    'log offer_vol/mean/50s safe_div bid_vol/mean/50s', # log ratio of volumes 
    '(midprice/mean/5s - midprice/mean/500s) safe_div midprice/std/500s', # what's the z-score of the past 100ms of offers relative to a 500s gaussian
    '(midprice/mean/5s - midprice/mean/50s) safe_div midprice/std/50s', # what's the z-score of the past 100ms of offers relative to a 500s gaussian
    'log midprice/mean/5s % weighted_total_price/mean/5s',  # ratio between the midprice and the volume weighted average of all levels
    
    'spread/mean/5s', 
    
    'null_100ms_frame/mean/50s', # what percentage of 100ms frames have had messages arriving?
    'last_bid_digit_near_zero/mean/5s', # how close to 0 or 9 is the last digit?
    '(midprice/mean/5s - midprice/min/50s) safe_div (midprice/max/50s - midprice/min/50s)', # where in the range from min to max are we?  
    
    'offer/std/5s',  # fast standard deviation of the bids 
    'offer/std/500s', # slow standard deviation of the bids
    'log bid/std/50s safe_div offer/std/50s', # ratio between bid and offer dispersion 
    'weighted_total_price/slope/5s', # what direction has the volume weighted level average been moving? 
    'slope slope midprice/mean/50s', # acceleration of price, computed by repeated differencing (scaled by time difference between points)
    "log bid_range/mean/50s safe_div offer_range/mean/50s", # how much wider is the bidside than the offer side? 
    "abs (bid_vol/slope/5s - bid_vol/slope/500s) safe_div bid_vol/std/500s", # how far is the recent bid_volume rate of change deviating from 500s 
    "abs (offer_vol/slope/5s - offer_vol/slope/500s) safe_div offer_vol/std/500s", # how far off is recent offer volume rate of change from 500s
    "clean log bid_range/mean/5s safe_div spread/mean/50s", # how many spreads wide is the verical bid range? 
    "clean log offer_range/mean/5s safe_div spread/mean/50s", # how many spreads wide is the vertical offer range? 
    't % 86400000' # t is milliseconds since midnight, divide by milliseconds in day to normalize
    ]
import numpy as np     
import os 
import tempfile 

import boto 
import scikits.learn 
import cloud

from dataset import Dataset 
from expr_lang import Evaluator 
import signals     
import simulate
import encoder     

def load_s3_file(filename, bucket='capk-fxcm'):     
    conn = boto.connect_s3('AKIAITZSJIMPWRM54I4Q','8J9VG9WlYCOmT6tq6iyC7h1K2rOk8v+q8FehsBdv')
    bucket = conn.get_bucket(bucket)
    key = bucket.get_key(filename) 
    (fileid, local_filename) = tempfile.mkstemp()
    key.get_contents_to_filename(local_filename)
    # do some s3 magic 
    return Dataset(local_filename)
    
def dataset_to_feature_matrix(d, features=features): 
    ev = Evaluator() 
    ncols = len(features)
    nrows = len(d['t'])
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

def load_files(files, features=features, signal_fn=signals.aggressive_profit): 
    datasets = [load_s3_file(f) for f in files] 
    matrices = [dataset_to_feature_matrix(d, features) for d in datasets] 
    feature_data = np.concatenate(matrices)
    signal = np.concatenate([signal_fn(d) for d in datasets] )
    times = np.concatenate([d['t/100ms'] for d in datasets])
    bids = np.concatenate( [d['bid/100ms'] for d in datasets])
    offers = np.concatenate( [d['offer/100ms'] for d in datasets])
    currencies = [d.currency_pair for d in datasets]
    for d in datasets:
        d.hdf.close()
        os.remove(d.filename)
    return feature_data, signal, times, bids, offers, currencies   

    
    


def eval_prediction(ts, bids, offers, pred, currency_pair, cut=0.0015):

    profit_series = simulate.aggressive_with_hard_thresholds(ts, bids, offers, pred, currency_pair, max_loss_prct = cut)
    sum_profit = np.sum(profit_series)
    ntrades = np.sum(profit_series != 0)
    if ntrades > 0: profit_per_trade = sum_profit / float(ntrades)
    else: profit_per_trade = 0 
    raw_accuracy, tp, fp, tn, fn, tz, fz = signals.accuracy(y_test, pred)
    result = {
        'profit': sum_profit, 
        'ntrades': ntrades, 
        'ppt': profit_per_trade, 
        'accuracy': raw_accuracy, 
        'tp': tp, 'fp': fp, 
        'tn': tn,  'fn': fn, 
        'tz': tz, 'fz': fz
    }
    return result 

# load each file, extract features, concat them together 
def worker(params, features, train_files, test_files): 
    train_data, train_signal, _, _, _, _ = load_files(train_files) 
    test_data, test_signal, test_times, test_bids, test_offers, currencies = load_files(test_files)
    # assume all files from same currency pair 
    ccy = currencies[0]
    print "Creating encoder with k=", params['k']
    e = encoder.FeatureEncoder(train_data, whiten=False, n_centroids=params['k'])
    print "Encoding training data" 
    train_encoded = e.encode(train_data, transformtion = params['t'])
    
    svm = scikits.learn.svm.LinearSVC(C = params['c'])
    weights = {1: params['pos_weight'], 0:1, -1:params['neg_weight']}
    print "Training SVM with C=", c, "weights = ", weights 
    svm.fit(train_encoded, train_signal, class_weight = weights)
    print "Encoding test data" 
    test_encoded = e.encode(test_data, transformtion = params['t'])
    pred = svm.predict(test_encoded)
    
    result = eval_prediction(test_times, test_bids, test_offers, pred, ccy)
    return params, features, e, svm, result 

def gen_work_list(): 
    n_centroids = [None, 100, 200, 400, 800] 
    cs = [1.0, 5.0]
    cut_thresholds = [.0005, .001, .0015,  0.002]
    transformations = ['prob', 'triangle']
    class_weights = [5, 15, 30, 45] 
    worklist = [] 
    for k in n_centroids:
        for t in transformations: 
            for c in cs:
                for pos_weight in class_weights:
                    for neg_weight in class_weights:
                        params = {
                            'k': k, 
                            't': t, 
                            'c': c, 
                            'pos_weight': pos_weight, 
                            'neg_weight': neg_weight, 
                        }
                        worklist.append(params)
                        
    return worklist 

def init_cloud(): 
    cloud.config.force_serialize_debugging = False
    cloud.config.force_serialize_logging = False 
    cloud.config.commit()
    cloud.setkey(2579, "f228c0325cf687779264a0b0698b0cfe40148d65")

def param_search(train_files, test_files, features=features):
    init_cloud() 
    params = gen_work_list()
    print "Generated", len(params), "params"
    def eval_param(p): 
        return worker(p, features, train_files, test_files)
    jobids = cloud.map(eval_param, params, _fast_serialization=0, _type='c1') 
    for params, features, e, svm, result in cloud.iresult(jobids):
        print params, "=>", result 


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--train", dest="train", help="training files", nargs='*', default=[])
    parser.add_argument("--test", dest="test", help="testing files", nargs='*', default=[])
    args = parser.parse_args()
    param_search(args.train, args.test) 



