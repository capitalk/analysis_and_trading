
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
            

import numpy as np     
import os 
import tempfile 

import boto 
import scikits.learn 
import cloud


import simulate
import signals     
import encoder     
import sgd_cascade
import balanced_ensemble
from dataset import Dataset 
from expr_lang import Evaluator 
from analysis import check_data 

AWS_ACCESS_KEY_ID = 'AKIAITZSJIMPWRM54I4Q' 
AWS_SECRET_ACCESS_KEY = '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' 

def get_hdf_bucket(bucket='capk-fxcm'):
    # just in case
    import socket
    socket.setdefaulttimeout(None)
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    return conn.get_bucket(bucket)


def load_s3_file(filename, max_failures=2):     
    print "Loading", filename, "from S3"
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

def load_files(files, features=features, signal_fn=signals.aggressive_profit): 
    print "Loading datasets..."
    datasets = load_s3_files(files) 
    print "Flattening datasets into feature matrices..." 
    matrices = [dataset_to_feature_matrix(d, features) for d in datasets] 
    feature_data = np.concatenate(matrices)
    check_data(feature_data) 
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

    profit_series = simulate.aggressive_with_hard_thresholds(ts, bids, offers, pred, currency_pair, max_loss_prct = cut, max_hold_time=60000)
    #profit_series, _, _ = simulate.aggressive(ts, bids, offers, pred, currency_pair)
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
    print params 
    print "Loading training data..."
    train_data, train_signal, train_times, train_bids, train_offers, currencies = load_files(train_files) 
    if 'target' in params: train_signal = (train_signal == params['target']).astype('int')
    # assume all files from same currency pair 
    ccy = currencies[0]
    k = params['k']
    whiten = params['whiten']
    prod = params['pairwise_products']
    print "Creating encoder with k=", k, 'whiten=', whiten
    e = encoder.FeatureEncoder(train_data, whiten=whiten, n_centroids=k, products=prod)
    print "mean: ", e.mean_
    print "std: ", e.std_ 
    print "centroids: ", e.centroids 
    print "pca :", e.pca 
    
    print "Encoding training data" 
    train_encoded = e.encode(train_data, transformation = params['t'], in_place=True, unit_norm=params['unit_norm'])
    print train_encoded[0, :]
    print train_encoded[500, :] 
    del train_data
    
    
    # sometimes we get a list of weights and sometimes we get just one weight
    # this code is written to work in either case 
    #class_weights = [params['class_weight']] if 'class_weight' in params else params['class_weights']
    
    #alphas = [params['alpha']] if 'alpha' in params else params['alphas']
    #loss = params['loss']
    #penalty = params['penalty']
    # scramble the training the set order and split it into a half-training set
    # and a validation set to search for best hyper-parameters like 
    # alpha and class weights 
    ntrain = train_encoded.shape[0] 
    
    model_params = {}
    if 'loss' in params: model_params['loss'] = params['loss']
    if 'C' in params: model_params['C'] = params['C']
    if 'penalty' in params: model_params['penalty']  = params['penalty']
    if 'shuffle' in params: model_params['shuffle'] = params['shuffle']
    if 'neutral_weight' in params: model_params['neutral_weight'] = params['neutral_weight']
    #def mk_model(alpha):
    #    return balanced_ensemble.Ensemble(num_classifiers=params['num_classifiers'],  **model_params) #alpha=alpha,
    def mk_model():
        return balanced_ensemble.Ensemble(num_classifiers=params['num_classifiers'], **model_params)
    #best_weights = None 
    #best_alpha = None 
    #best_value = -10000 #{'accuracy': -1, 'ppt': -10000}
    
    #if len(alphas) == 1:
    #    best_alpha = alphas[0]
    #else:
    #    print "Searching for best hyper-parameters" 
    #    nsubset = min(ntrain/2, 200000)
    #    print "Creating validation set (size = ", nsubset, ")"
        
    #    p = np.random.permutation(ntrain)
    #    half_train_indices = p[:nsubset]
    #    validation_indices = p[nsubset:(2*nsubset)]
    #    half_train = train_encoded[half_train_indices, :] 
    #    half_signal = train_signal[half_train_indices]
        
    #    validation_set = train_encoded[validation_indices, :]
    #    validation_signal = train_signal[validation_indices] 
        
    #    for alpha in alphas: 
    #        model = mk_model(alpha)
    #        #weights = {0:1, -1:neg_weight, 1: pos_weight}
    #        print "Training SVM with alpha=",alpha #weights = 
    
    #        model.fit(half_train, half_signal)
            # pred = model.predict(validation_set) 
    #        pred = model.predict(validation_set) #multiclass_output(model, validation_set)
            # accuracy, tp, fp, etc...
    #        result = signals.accuracy(validation_signal, pred)
    #        print result
                
    #        curr_value = result[0]
    #        if best_value < curr_value:
    #            best_value = curr_value  
    #            best_alpha = alpha 
        
        # clear some space 
    #    del half_train
    #    del half_signal
    #    del validation_indices
    #    del p 
    #    del validation_set
    #    del validation_signal 
        
    #print "Fitting full model, alpha=", best_alpha
    
    model = mk_model()
    model.fit(train_encoded, train_signal)
    del train_encoded
    del train_signal 
    
    print "Loading testing data..."
    test_data, test_signal, test_times, test_bids, test_offers, _ = load_files(test_files)
    if 'target' in params: test_signal = (test_signal == params['target']).astype('int')
    
    print "Encoding test data" 
    test_encoded = e.encode(test_data, transformation = params['t'], in_place=True, unit_norm=params['unit_norm'])
    print test_encoded[0, :]
    print test_encoded[500, :]
    del test_data 
                    
    print "Evaluating full model"
    #pred = svm.predict(test_encoded)
    pred, probs = model.predict(test_encoded, return_probs=True) 
    print "Probabilities: ", probs[:100]
    
    print "predictions: ", pred[1:50] 
    result = eval_prediction(test_times, test_bids, test_offers, pred, test_signal, ccy)
    
    print features
    print '[model]'
    print model
    print '[encoder]'
    print e.mean_
    print e.std_
    print e.centroids
    print '[result]' 
    print result 
    # have to clear sample weights since SGDClassifier stupidly keeps them 
    # after training 
    #model.sample_weight = [] 
    return  params, result, e, model
    
def gen_work_list(): 

#    n_centroids = [None, 25, 50, 100] 
    #cut_thresholds = [.0005, .001, .0015,  0.002]
    #transformations = ['triangle', 'thresh']
    transformations = ['triangle'] 
    n_centroids = [None, 15, 30]  
    #targets = [1, -1]
    unit_norm = [False] #, True] 
    pairwise_products = [ False, True] 
    whiten = [True, False]
    neutral_weights = [1, 4, 8]
    num_classifiers = [8, 32, 128]
    cs = [0.1, 1.0, 10.0]
    #losses = ['hinge']# , 'modified_huber']
    #penalties = ['l2']#, 'l1']#, 'l1'] #'elasticnet'] #, 'l1', 'elasticnet']
    
    #alphas = [0.00001] #10.0 ** np.arange(-7, -3)
    
    #cascade_length = [3,4,5]
    #class_weights = [8, 16] #, 32]
    worklist = [] 
    for prod in pairwise_products: 
        for k in n_centroids:
            ts = transformations if k is not None else [None] 
            for t in ts: 
                for w in whiten: 
                    for u in unit_norm:
                        for n in num_classifiers:
                            for c in cs:
                                #for target in targets:
                                for neutral_weight in neutral_weights:
                                    params = {
                                #        'target': target, 
                                        'k': k, 
                                        't': t, 
                                        'whiten': w, 
                                        'pairwise_products': prod,
                                        'unit_norm': u, 
                                        'num_classifiers': n,
                                        'C': c, 
                                        'neutral_weight': neutral_weight,
                                    }
                                    worklist.append(params)

                        #for loss in losses:
                        #for p in penalties:
    return worklist 

def init_cloud(): 
    cloud.config.force_serialize_debugging = False
    cloud.config.force_serialize_logging = False 
    cloud.config.commit()
    cloud.setkey(2579, "f228c0325cf687779264a0b0698b0cfe40148d65")

#def print_params(params):
#    print "[Input] k:", params['k'], 'whiten:', params['whiten'], 't:', params['t'], 'loss:', params['loss'], 'penalty:', params['penalty'], 'prod:', params['pairwise_products']

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
        jobids = cloud.map(eval_param, params, _fast_serialization=2, _type='m1', _label=label) 
        results = [] 
        print "Launched", len(params), "jobs, waiting for results..."
        for params, result, e, model in cloud.iresult(jobids):
            results.append({'params':params, 'result': result, 'encoder': e, 'model': model})
            print params
            print model
            print result 
            
                
        def cmp(x,y):
            return int(np.sign(x['result']['accuracy'] - y['result']['accuracy']))
        
        results.sort(cmp=cmp)
        
        accs = [x['result']['accuracy'] for x in results]
        ppts = [x['result']['ppt'] for x in results]
        print accs
        print ppts 
        
        print "Best:"
        for item in results[-10:]:
            print item['params']
            print item['result']
        
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
    args = parser.parse_args()
    if args.train == [] or args.test == []: print_s3_hdf_files()
    else: 
        training_files = make_filenames(args.ecn, args.ccy, args.train) + args.train_files
        testing_files = make_filenames(args.ecn, args.ccy, args.test) + args.test_files 
        param_search(training_files, testing_files, debug=args.debug) 
