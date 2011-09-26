
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

import progressbar
import multiprocessing
import boto 
import scikits.learn 
import cloud
from collections import namedtuple

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
    bids = np.concatenate(d['bid/100ms'] for d in datasets])
    offers = np.concatenate(d['offer/100ms'] for d in datasets])
    for d in datasets:
        d.hdf.close()
        os.remove(d.filename)
    return feature_data, signal, times, bids, offers  

    
    
Params = namedtuple('Params', 'k transformation C pos_weight neg_weight')
Result = namedtuple('Result', 'profit ntrades ppt accuracy tp fp tn fn tz fz')
    
# load each file, extract features, concat them together 
def worker(params, features, train_files, test_files): 
    train_data, train_signal, train_times, train_bids, train_offers = load_files(train_files) 
    test_data, test_signal, test_times, test_bids, test_offers = load_files(test_files)
    e = encoder.FeatureEncoder(train_data, whiten=False, n_centroids=params.k)
    train_encoded = e.encode(train_data, transformtion = params.transformation)
    test_encoded = e.encode(test_data, transformtion = params.transformation)
    svm = scikits.learn.svm.LinearSVC(C = params.C)
    weights = {1: params.pos_weight, 0:1, -1:params.neg_weight}
    svm.fit(train_encoded, train_signal, class_weight = weights)
    
def test_svm(d, x_train, x_test, y_train, y_test, test_start_idx, linear=True)
    ts = d['t/100ms']
    bids = d['bid/100ms']
    offers = d['offer/100ms']
    currency_pair = d.currency_pair
    
    n_centroids = [None, 100, 200, 400, 800] 
    cs = [1.0, 5.0]
    cut_thresholds = [.0005, .001, .0015,  0.002]
    transformations = ['prob', 'triangle']
    class_weights = [5, 15, 30, 45] 
    results = {}
    profits = {} 
    best_profit = 0 
    best_desc = None 
    best_result = None 
    
    worklist = [] 
    for k in n_centroids:
        for t in transformations: 
            for c in cs:
                for pos_weight in class_weights:
                    for neg_weight in class_weights:
                        params = Params(k, t, c, pos_weight, neg_weight)
                        worklist.append(params)
                        
                        print "Training SVM with C=", c, "pos_weight=", pos_weight, "neg_weight =", neg_weight
                        svm.fit(x_train_encoded, y_train, class_weight = {1: pos_weight, 0:1, -1:neg_weight})
                        pred = svm.predict(x_test_encoded)
                        for cut in cut_thresholds:
                            
                            profit_series = simulate.aggressive_with_hard_thresholds(ts, bids, offers, pred, currency_pair, max_loss_prct = cut)
                            sum_profit = np.sum(profit_series)
                            ntrades = np.sum(profit_series != 0)
                            if ntrades > 0: profit_per_trade = sum_profit / float(ntrades)
                            else: profit_per_trade = 0 
                            raw_accuracy, tp, fp, tn, fn, tz, fz = signals.accuracy(y_test, pred)
                            result = Result(sum_profit, ntrades, profit_per_trade, raw_accuracy, tp, fp, tn, fn, tz, fz)
                            print desc
                            print result
                            print
                            if result.profit > best_profit:
                                best_profit = result.profit
                                best_result = result
                                best_desc = desc
                            results[desc] = result
                            profits[desc] = result.profit 
    
    
    
    num_class_weights = len(class_weights)
    nparams = len(n_centroids) * len(cs) * len(transformations)  * num_class_weights * num_class_weights * len(cut_thresholds)
    print "Total param combinations: ", nparams
    for k in n_centroids:
        
        
        print "Training encoder with k = ", k
    
        for t in transformations: 
            x_train_encoded = e.encode(x_train, transformation = t)
            x_test_encoded = e.encode(x_test, transformation = t) 
            for c in cs:
                if linear: 
                else: svm = scikits.learn.svm.SVC(C = c)
                for pos_weight in class_weights:
                    for neg_weight in class_weights:
                        print "Training SVM with C=", c, "pos_weight=", pos_weight, "neg_weight =", neg_weight
                        svm.fit(x_train_encoded, y_train, class_weight = {1: pos_weight, 0:1, -1:neg_weight})
                        pred = svm.predict(x_test_encoded)
                        for cut in cut_thresholds:
                            desc = Params(k, t, c, pos_weight, neg_weight, cut)
                            profit_series = simulate.aggressive_with_hard_thresholds(ts, bids, offers, pred, currency_pair, max_loss_prct = cut)
                            sum_profit = np.sum(profit_series)
                            ntrades = np.sum(profit_series != 0)
                            if ntrades > 0: profit_per_trade = sum_profit / float(ntrades)
                            else: profit_per_trade = 0 
                            raw_accuracy, tp, fp, tn, fn, tz, fz = signals.accuracy(y_test, pred)
                            result = Result(sum_profit, ntrades, profit_per_trade, raw_accuracy, tp, fp, tn, fn, tz, fz)
                            print desc
                            print result
                            print
                            if result.profit > best_profit:
                                best_profit = result.profit
                                best_result = result
                                best_desc = desc
                            results[desc] = result
                            profits[desc] = result.profit 
    print "Best over all params:"
    print best_desc
    print best_result
    return profits, results 




def test_file(filenames, learner='linsvm'):
    
    d, x_train, x_test, y_train, y_test,test_start_idx = load_file(filename)
    if learner == 'linsvm':
        return test_svm(d, x_train, x_test, y_train, y_test, test_start_idx, linear=True)
    elif learner == 'svm':
        return test_svm(d, x_train, x_test, y_train, y_test, test_start_idx, linear=False)
    else:
        return test_knn(d, x_train, x_test, y_train, y_test, test_start_idx)
    
