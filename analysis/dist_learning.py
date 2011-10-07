
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
    "bid_range/mean/50s - offer_range/mean/50s", # how much wider is the bidside than the offer side? 
    "abs (bid_vol/slope/5s - bid_vol/slope/500s) safe_div bid_vol/std/500s", # how far is the recent bid_volume rate of change deviating from 500s 
    "abs (offer_vol/slope/5s - offer_vol/slope/500s) safe_div offer_vol/std/500s", # how far off is recent offer volume rate of change from 500s
    "clean log bid_range/mean/5s safe_div spread/mean/50s", # how many spreads wide is the verical bid range? 
    "clean log offer_range/mean/5s safe_div spread/mean/50s", # how many spreads wide is the vertical offer range? 
    't % 86400000' # t is milliseconds since midnight, divide by milliseconds in day to normalize
    ]

# pairwise product features: '33 = 1st & last' '73 = 3rd & 6th' 

import numpy as np     
import os 
import tempfile 

import boto 
import sklearn 
import sklearn.linear_model 
import cloud


import simulate
#import signals     
import encoder     
import sgd_cascade
import balanced_ensemble
from dataset import Dataset 
from expr_lang import Evaluator 
#from analysis import check_data 

AWS_ACCESS_KEY_ID = 'AKIAITZSJIMPWRM54I4Q' 
AWS_SECRET_ACCESS_KEY = '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' 

def aggressive_profit(data, max_hold_frames = 80, num_profitable_frames = 2, target_prct=0.0001, start=None, end=None):
    ts = data['t/100ms'][start:end]
    bids = data['bid/100ms'][start:end]
    offers = data['offer/100ms'][start:end]
    n = len(ts) 
    signal = np.zeros(n)
    for (idx, start_offer) in enumerate(offers):
        if idx < n - max_hold_frames - 1:
            bid_window = bids[idx+1:idx+max_hold_frames+1]
            target_up = (1 + target_prct) * start_offer
            profit_up =  np.sum(bid_window >= target_up) > num_profitable_frames 
            
            start_bid = bids[idx]
            target_down = (1 - target_prct) * start_bid
            offer_window = offers[idx+1:idx+max_hold_frames + 1]
            profit_down = np.sum(offer_window <= target_down) > num_profitable_frames
            
            if profit_up and not profit_down:
                signal[idx] = 1
            elif profit_down and not profit_up:
                signal[idx] = -1
    return signal 

def future_change(ys, horizon = 100):
    n = len(ys)
    signal = np.zeros(n)
    for i in xrange(n-1):
        future_delta = ys[i+1:i+horizon] - ys[i] 
        max_val = np.max(future_delta)
        min_val = np.min(future_delta)
        if max_val > -min_val: signal[i] = max_val 
        else: signal[i] = min_val
    return signal 

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

                                
def accuracy(y_test, pred): 
    pred_zero = pred == 0
    num_zero = np.sum(pred_zero)
    
    pred_pos = pred > 0
    num_pos = np.sum(pred_pos) 
    
    pred_neg = pred < 0
    num_neg = np.sum(pred_neg) 
    num_nonzero = num_pos + num_neg 
    
    y_test_pos = y_test > 0 
    y_test_neg = y_test < 0
    y_test_zero = y_test == 0
    tp = np.sum(y_test_pos & pred_pos)
    tz = np.sum(y_test_zero & pred_zero)
    tn = np.sum(y_test_neg & pred_neg)
    
    fp = num_pos - tp 
    fn = num_neg - tn 
    fz = num_zero - tz 
    total = float(tp + fp + tn + fn)
    if total > 0: accuracy = (tp + tn) / total 
    else: accuracy = 0.0 
    return accuracy, tp, fp, tn, fn, tz, fz
        
def get_hdf_bucket(bucket='capk-fxcm'):
    # just in case
    import socket
    socket.setdefaulttimeout(None)
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    return conn.get_bucket(bucket)

cache_dir = '/s3_cache'
def load_s3_file(filename, max_failures=2):     
    print "Loading", filename, "from S3"
    cache_name = cache_dir + "/" + filename 
    if os.path.exists(cache_name): 
        print "Found", filename, "in environment's local cache, returning", cache_name 
        return cache_name 
    max_failures = 3 
    fail_count = 0 
    got_file = False 
    while fail_count < max_failures and not got_file:
        bucket = get_hdf_bucket()
        key = bucket.get_key(filename) 
        if key is None: raise RuntimeError("file not found: " + filename)
        (fileid, local_filename) = tempfile.mkstemp(prefix=filename)
        print "Created local file:", local_filename 
        print "Copying data from S3"
        try: 
            key.get_contents_to_filename(local_filename)
            got_file = True 
        except: 
            print "Download from S3 failed" 
            fail_count += 1
            os.remove(local_filename) 
            if fail_count >= max_failures: raise 
            else: print "...trying again..."
    return local_filename 
    
def load_s3_files(filenames):
    return [Dataset(load_s3_file(remote_filename)) for remote_filename in filenames]
    
def load_s3_files_in_parallel(filenames):
    jobids = cloud.mp.map(load_s3_file, filenames)
    local_filenames = cloud.mp.result(jobids)
    return [Dataset(f) for f in local_filenames]
    
def dataset_to_feature_matrix(d, features=features): 
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

def load_files(files, features=features, signal_fn=aggressive_profit): 
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
#        os.remove(d.filename)
    return feature_data, signal, times, bids, offers, currencies   

    
def eval_prediction(ts, bids, offers, pred, actual, currency_pair, cut=0.0015):

    profit_series = simulate.aggressive_with_hard_thresholds(ts, bids, offers, pred, currency_pair, max_loss_prct = cut, max_hold_time=30000)
    #profit_series, _, _ = simulate.aggressive(ts, bids, offers, pred, currency_pair)
    sum_profit = np.sum(profit_series)
    ntrades = np.sum(profit_series != 0)
    if ntrades > 0: profit_per_trade = sum_profit / float(ntrades)
    else: profit_per_trade = 0 
    
    raw_accuracy, tp, fp, tn, fn, tz, fz = accuracy(actual, pred)
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


def confusion(pred, actual):
    """Given a predicted binary label vector and a ground truth returns a tuple containing the counts for:
    true positives, false positives, true negatives, false negatives """
    
    # implemented using indexing instead of boolean mask operations
    # since the positive labels are expecetd to be sparse 
    n = len(pred)
    pred_nz_mask = (pred != 0)
    pred_nz_indices = np.nonzero(pred_nz_mask)[0]
        
    total_nz = len(pred_nz_indices)
    tp = sum(pred[pred_nz_indices] == actual[pred_nz_indices])
    fp = total_nz - tp 
    actual_nz_mask = (actual != 0)
    actual_nz_indices = np.nonzero(actual_nz_mask)[0]
        
    fn = sum(1 - pred[actual_nz_indices])
    tn = n - tp - fp - fn 
    return tp, fp, tn, fn 
        
def eval_all_thresholds(times, bids, offers, target_sign, target_probs, actual, ccy):
    best_score = -100
    best_precision = 0
    best_recall = 0
    
    best_thresh = 0 
    best_pred = (target_probs >= 0).astype('int')
    
    precisions = []
    recalls = []
    f_scores = []
    
    thresholds = .2 + np.arange(80) / 100.0
    
    print "Evaluating threshold"
    for t in thresholds:
        pred = target_sign * (target_probs >= t).astype('int') 
        
        # compute confusion matrix with masks instead of 
        tp, fp, tn, fn = confusion(pred, actual) 
        total_pos = tp + fp
        
        if total_pos > 0: precision = tp / float(total_pos)
        else: precision = 0
        
        recall = tp / float(tp + fn)
        beta = 0.5
        
        if precision > 0 and recall > 0:
            f_score = (1 + beta**2) * (precision * recall) / ((beta**2) * precision + recall)
        else:
            f_score = 0.0 
        print "Threshold:", t, "precision =", precision, "recall =", recall, "f_score =", f_score 
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)
        
        if f_score > best_score: 
            best_score = f_score 
            best_precision = precision 
            best_recall = recall 
            best_thresh = t
            best_pred = pred
            
    detailed_result = eval_prediction(times, bids, offers, best_pred, actual, ccy)
    return {
        'best_thresh': best_thresh, 
        'best_precision': best_precision, 
        'best_recall': best_recall, 
        'best_score': best_score, 
        'ppt': detailed_result['ppt'], 
        'ntrades': detailed_result['ntrades'], 
        'all_thresholds': list(thresholds), 
        'all_precisions': precisions, 
        'all_recalls': recalls,
    }
        
    
# load each file, extract features, concat them together 
def worker(params, features, train_files, test_files): 
    general_params, encoder_params, ensemble_params, model_params, train_params   = params 
    print "General params:", general_params 
    print "Encoder params:", encoder_params
    print "Ensemble params:", ensemble_params
    print "Model params:", model_params
    print "Train params:", train_params 
    
    print "Loading training data..."
    train_data, train_signal, train_times, train_bids, train_offers, currencies = load_files(train_files) 
    ntrain = train_data.shape[0] 
    if 'target' in general_params: target = general_params['target']
    else: target = None
    if target: train_signal = (train_signal == target).astype('int')
            
    # assume all files from same currency pair 
    ccy = currencies[0]
    
    e = encoder.FeatureEncoder(**encoder_params)
    print "Encoding training data..." 
    train_data = e.fit_transform(train_data)
    print "Encoded shape:", train_data.shape 
    print "train_data[500]", train_data[500, :] 
    
    
    model = balanced_ensemble.Ensemble(model_params=model_params, **ensemble_params)
    
    if 'class_weight' in train_params: model.fit(train_data, train_signal, class_weight=train_params['class_weight'])
    else: model.fit(train_data, train_signal)
    
    del train_data
    del train_signal 
    
    print "Reminder, here were the params:", params 
    
    print "Loading testing data..."
    test_data, test_signal, test_times, test_bids, test_offers, _ = load_files(test_files)
    
    print "Encoding test data" 
    test_data = e.transform(test_data, in_place=True)
    
    print "test_data[500] =", test_data[500, :]
    
                    
    print "Evaluating full model"
    #pred = svm.predict(test_encoded)
    pred, probs = model.predict(test_data, return_probs=True) 
    
    
    if target:
        test_signal = target * (test_signal == target).astype('int')
        target_index = model.classes.index(1)
        target_probs = probs[:,target_index]
        result = eval_all_thresholds(test_times, test_bids, test_offers, target, target_probs, test_signal, ccy)
    else: result = eval_prediction(test_times, test_bids, test_offers, pred, test_signal, ccy)
    
    print features
    print '[model]'
    print model
    print '[encoder]'
    print e.mean_
    print e.std_
    print '[result]' 
    print "precisions:", result['all_precisions']
    print "recalls:",  result['all_recalls']
    print 'threshold:', result['best_thresh']
    print 'precision:', result['best_precision']
    print 'recall:', result['best_recall']
    print 'f-score:', result['best_score']
    print 'ntrades:', result['ntrades']
    print 'ppt:', result['ppt']
    # have to clear sample weights since SGDClassifier stupidly keeps them 
    # after training 
    #model.sample_weight = [] 
    return {'params':params, 'result': result, 'encoder': e, 'model': model}
    
def cartesian_product(options):
    import itertools
    combinations = [x for x in apply(itertools.product, options.values())]
    return [dict(zip(options.keys(), p)) for p in combinations]

def prune(dicts, condition):
    result = []
    for d in dicts:
        if not condition(d):
            result.append(d)
    return result 
        
    
def gen_work_list(): 

    targets = [-1]
    oversampling_factors = [0]
    
    
    class_weights = [1] 
    
    alphas = [0.00001]
    Cs = [.01, 0.1, 1.0]
    

    possible_encoder_params = {
        'dictionary_type': [None, 'kmeans', 'sparse'],
        'dictionary_size': [10, 20, 60],
        'pca_type': [None, 'whiten'], 
        'compute_pairwise_products': [False], 
        'binning': [False, True]
    }
    all_encoder_params = cartesian_product(possible_encoder_params)
    all_encoder_params = prune(all_encoder_params, lambda d: d['dictionary_type'] is None and d['dictionary_size'] != 10)
    
    possible_ensemble_params = {
        'balanced_bagging': [True], 
        'num_classifiers': [20], #[100, 200]
        'num_random_features': [0.5],
        'base_classifier': ['sgd'], 
        'neutral_weight': [5], 
        'model_weighting': ['logistic'],
    }
    all_ensemble_params =  cartesian_product(possible_ensemble_params)
    
    
    worklist = [] 
    for target in targets:
        for smote_factor in oversampling_factors:
            general_params = {
                'oversampling_factor': smote_factor, 
                'target': target
            }
            for encoder_params in all_encoder_params:
                for ensemble_params in all_ensemble_params:
                    for cw in class_weights:    
                        train_params = { 'class_weight': {0:1, 1:cw} }
                        if ensemble_params['base_classifier'] == 'sgd':
                            all_model_params = [{'alpha': alpha} for alpha in alphas]
                        else: 
                            all_model_params = [{ 'C': c} for c in Cs]
                    for model_params in all_model_params:
                        param_tuple =  (general_params, encoder_params, ensemble_params, model_params, train_params)
                        worklist.append (param_tuple)
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
    
    params = gen_work_list()
    if debug: 
        print "[Debug mode]"
        result_list = map(eval_param, params[:1])
        for params, features, e, svm, result in result_list:
            print params, "=>", result 
    else: 
        init_cloud() 
        label = ", ".join(train_files)
        jobids = cloud.map(eval_param, params, _fast_serialization=2, _type='m1', _label=label, _env='param_search') 
        results = [] 
        print "Launched", len(params), "jobs, waiting for results..."
        for x in cloud.iresult(jobids):
            if x is not None:
                results.append(x)
                print x['params']
                print x['model']
                r = x['result']
                print 'Result:  precision =', r['best_precision'], 'recall =', r['best_recall'], 'ppt =', r['ppt'], 'ntrades =', r['ntrades']
                print "---" 
                
        def cmp(x,y):
            return int(np.sign(x['result']['best_score'] - y['result']['best_score']))
        
        results.sort(cmp=cmp)
        
        #accs = [x['result']['accuracy'] for x in results]
        #ppts = [x['result']['ppt'] for x in results]
        #print ppts 
        #print accs
        
        print "Best:"
        for item in results[-5:]:
            print item['params']
            r = item['result']
            print [(k, r[k]) for k in sorted(r.keys())]

def print_s3_hdf_files(): 
    bucket = get_hdf_bucket()
    filenames = [k.name for k in bucket.get_all_keys() if k.name.endswith('hdf')]
    print "\n".join(filenames )

def make_filenames(ecn, ccy, dates): 
    ecn = ecn.upper()
    ccy = ccy.upper().replace('/', '') 
    dates = [d.replace('/', '_') for d in dates]
    return [ecn +"_" + ccy + "_" + d + ".hdf" for d in dates] 

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecn", dest="ecn", default="fxcm")
    parser.add_argument("--ccy", dest="ccy", help="currency pair (e.g. USDJPY)")
    parser.add_argument("--train", dest="train", help="training dates", nargs='*', default=[])
    parser.add_argument("--train_files", dest="train_files", help="training files", nargs='*', default=[])
    
    parser.add_argument("--test", dest="test", help="testing dates", nargs='*', default=[])
    parser.add_argument("--test_files", dest="test_files", help="testing files", nargs='*', default=[])
    parser.add_argument("--debug", action='store_true', default=False, dest='debug')
    #todo: make this actually work 
    parser.add_argument("--output", dest="output", default=None, help="output file for model and encoder")
    # todo: allow either parameter sweep or manually specify learning params  ie --thresh 0.9 0.95 
    args = parser.parse_args()
    if args.train == [] or args.test == []: print_s3_hdf_files()
    else: 
        training_files = make_filenames(args.ecn, args.ccy, args.train) + args.train_files
        testing_files = make_filenames(args.ecn, args.ccy, args.test) + args.test_files 
        param_search(training_files, testing_files, debug=args.debug) 
