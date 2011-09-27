
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

AWS_ACCESS_KEY_ID = 'AKIAITZSJIMPWRM54I4Q'
AWS_SECRET_ACCESS_KEY = '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'

def get_hdf_bucket(bucket='capk-fxcm'):
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    return conn.get_bucket(bucket)


def load_s3_file(filename):     
    print "Loading", filename, "from S3"
    bucket = get_hdf_bucket()
    key = bucket.get_key(filename) 
    (fileid, local_filename) = tempfile.mkstemp(prefix=filename)
    print "Created local file:", local_filename 
    print "Copying data from S3"
    key.get_contents_to_filename(local_filename)
    # do some s3 magic 
    return local_filename 
    
    
def load_s3_files_in_parallel(filenames):
    jobids = cloud.mp.map(load_s3_file, filenames)
    local_filenames = cloud.mp.result(jobids)
    return [Dataset(f) for f in local_filenames]
    
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
    print "Loading datasets..."
    datasets = load_s3_files_in_parallel(files)
    print "Flattening datasets into feature matrices..." 
    matrices = [dataset_to_feature_matrix(d, features) for d in datasets] 
    feature_data = np.concatenate(matrices)
    signal = np.concatenate([signal_fn(d) for d in datasets] )
    times = np.concatenate([d['t/100ms'] for d in datasets])
    bids = np.concatenate( [d['bid/100ms'] for d in datasets])
    offers = np.concatenate( [d['offer/100ms'] for d in datasets])
    currencies = [d.currency_pair for d in datasets]
    print "Deleting local files..." 
    for d in datasets:
        d.hdf.close()
        os.remove(d.filename)
    return feature_data, signal, times, bids, offers, currencies   

    
def eval_prediction(ts, bids, offers, pred, actual, currency_pair, cut=0.0015):

    profit_series = simulate.aggressive_with_hard_thresholds(ts, bids, offers, pred, currency_pair, max_loss_prct = cut)
    sum_profit = np.sum(profit_series)
    ntrades = np.sum(profit_series != 0)
    if ntrades > 0: profit_per_trade = sum_profit / float(ntrades)
    else: profit_per_trade = 0 
    raw_accuracy, tp, fp, tn, fn, tz, fz = signals.accuracy(actual, pred)
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
    print "Loading training data..."
    train_data, train_signal, train_times, train_bids, train_offers, _ = load_files(train_files) 
    print "X_train:", train_data[10:13, :], "..."
    print "Y_train:", train_signal[10:50], "..."
    
    print "Loading testing data..."
    test_data, test_signal, test_times, test_bids, test_offers, currencies = load_files(test_files)
    print "X_test:", test_data[10, :]
    print "Y_test:", test_signal[10:40]
    
    # assume all files from same currency pair 
    ccy = currencies[0]
    print "Creating encoder with k=", params['k']
    e = encoder.FeatureEncoder(train_data, whiten=False, n_centroids=params['k'])
    
    print "Encoding training data" 
    train_encoded = e.encode(train_data, transformation = params['t'])
    print train_data.shape, "=>", train_encoded.shape
    del train_data 
    
    print "Encoding test data" 
    test_encoded = e.encode(test_data, transformation = params['t'])
    print test_data.shape, "=>", test_encoded.shape
    del test_data
    
    
    # sometimes we get a list of weights and sometimes we get just one weight
    # this code is written to work in either case 
    pos_weights = [params['pos_weight']] if 'pos_weight' in params else params['pos_weights']
    neg_weights = [params['neg_weight']] if 'neg_weight' in params else params['neg_weights']
    alphas = [params['alpha']] if 'alpha' in params else params['alphas']
    loss = params['loss']
    penalty = params['penalty']
    
    # scramble the training the set order and split it into a half-training set
    # and a validation set to search for best hyper-parameters like 
    # alpha and class weights 
    n = train_encoded.shape[0]
    print "Creating validation set (size", (n/2), ")"
    
    p = np.random.permutation(n)
    half_train_indices = p[:(n/2)]
    validation_indices = p[(n/2):]
    half_train = train_encoded[half_train_indices, :] 
    half_signal = train_signal[half_train_indices]
    validation_set = train_encoded[validation_indices, :]
    validation_signal = train_signal[validation_indices] 
    validation_times = train_times[validation_indices]
    validation_bids = train_bids[validation_indices]
    validation_offers = train_offers[validation_indices] 
    
    best_model = None 
    best_accuracy = -1
    best_weights = None 
    
    print "Searching for best hyper-parameters" 
    for pos_weight in pos_weights: 
        for neg_weight in neg_weights: 
            for alpha in alphas: 
                model = scikits.learn.linear_model.SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, shuffle=True, n_jobs=-1)
                weights = {0:1, -1:neg_weight, 1: pos_weight}
                print "Training SVM with weights = ", weights, 'alpha=',alpha
                #svm = scikits.learn.svm.LinearSVC(C = params['c'])
            
                model.fit(half_train, half_signal, class_weight = weights)
                pred = model.predict(validation_set)
                result = eval_prediction(validation_times, validation_bids, validation_offers, pred, validation_signal, ccy)
                print result
            
                if result['accuracy'] > best_accuracy: 
                    best_model = scikits.learn.linear_model.SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, shuffle=True, n_jobs=-1)
                    best_accuracy = result['accuracy'] 
                    best_weights = weights 
                    
    print "Fitting full model"
    best_model.fit(train_encoded, train_signal, class_weight=best_weights)
    
    print "Evaluating full model"
    
    pred = best_model.predict(test_encoded)
    result = eval_prediction(test_times, test_bids, test_offers, pred, test_signal, ccy)
    print best_model 
    print best_model.coef_
    print e
    print features
    print result 
    
    return params, features, e, best_weights, best_model, result

def gen_work_list(): 
    n_centroids = [None, 50, 100] 
    cs = [1.0, 5.0]
    #cut_thresholds = [.0005, .001, .0015,  0.002]
    transformations = ['triangle', 'thresh', 'prob']
    losses = ['hinge', 'modified_huber']
    penalties = ['l2', 'l1', 'elasticnet']
    alphas = [0.001, 0.01]
    
    class_weights = [5, 15, 30] 
    worklist = [] 
    for k in n_centroids:
        ts = transformations if k is not None else [None] 
        for t in ts: 
            #for c in cs:
            for loss in losses:
                for p in penalties:
                    params = {
                        'penalty': p, 
                        'loss': loss, 
                        'k': k, 
                        't': t, 
                        'alphas': alphas, 
                #        'c': c, 
                        'pos_weights': class_weights, 
                        'neg_weights': class_weights,
                    }
                    worklist.append(params)
    return worklist 

def init_cloud(): 
    cloud.config.force_serialize_debugging = False
    cloud.config.force_serialize_logging = False 
    cloud.config.commit()
    cloud.setkey(2579, "f228c0325cf687779264a0b0698b0cfe40148d65")

def param_search(train_files, test_files, features=features, debug=False):
    print "Features:", features 
    print "Training files:", train_files
    print "Testing files:", test_files 
    
    def eval_param(p): 
        return worker(p, features, train_files, test_files)
    
    if debug: 
        params = [{'k':None, 't':None, 'alpha':0.01, 'pos_weight':15, 'neg_weight':15}]
        jobids = cloud.mp.map(eval_param, params, _fast_serialization=2)
        for params, features, e, svm, result in cloud.mp.iresult(jobids):
            print params, "=>", result 
    else: 
        init_cloud() 
        params = gen_work_list()
        jobids = cloud.map(eval_param, params, _fast_serialization=2, _type='m1') 
        
        print "Launched", len(params), "jobs, waiting for results..."
        for params, features, e, weights, svm, result in cloud.iresult(jobids):
            if result:
                print "[Input] k=", params['k'], 't:', params['t']
                print svm
                print result 
                


def print_s3_hdf_files(): 
    bucket = get_hdf_bucket()
    filenames = [k.name for k in bucket.get_all_keys() if k.name.endswith('hdf')]
    print "\n".join(filenames )

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--train", dest="train", help="training files", nargs='*', default=[])
    parser.add_argument("--test", dest="test", help="testing files", nargs='*', default=[])
    parser.add_argument("--debug", action='store_true', default=False, dest='debug')
    args = parser.parse_args()
    if args.train == [] or args.test == []: print_s3_hdf_files()
    else: param_search(args.train, args.test, debug=args.debug) 



