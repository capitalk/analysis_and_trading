
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
    ]
import numpy as np     
from dataset import Dataset 
from expr_lang import Evaluator 

def get_features(filename, features=features):
    d = Dataset(filename)
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
    return result, d 


def split_data(x, y, ntrain=None):
    if ntrain is None:
        total = len(y)
        ntrain = int(total * 0.5 )
    # skip first few samples to avoid artifacts like zeros in std, etc... 
    start_at = 100 
    split = ntrain + start_at 
    x_train = x[start_at:split, :]
    x_test = x[split:, :] 
    y_train = y[start_at:split]
    y_test = y[split:] 
    return x_train, x_test, y_train, y_test, split

import signals     
def load_file(filename, ntrain=None, signal=signals.aggressive_profit):
    x,d = get_features(filename)
    print "Computing target signal..." 
    y = signal(d)
    x_train, x_test, y_train, y_test, split_idx = split_data(x,y, ntrain=ntrain)
    return  d, x_train, x_test, y_train, y_test, split_idx 

import sklearn 
import simulate
def eval_lin_classifier(mk_model, d, x_train,x_test, y_train, y_test, test_start_idx, penalty='l2'):
    cs = np.logspace(-2, 1, 4)
    best_profit = None
    best_c = None
    for c in cs:
        model = mk_model(C = c, penalty=penalty)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        profit_series = simulate.simulate_profits_dataset(d, pred, start_index = test_start_idx)
        profit_series_cumsum = np.cumsum(profit_series)
        print "c = ", c, "profit = ", profit_series_cumsum[-1]
        if best_profit is None or profit_series_cumsum[-1] > best_profit[-1]:
            best_profit = profit_series_cumsum
            best_c = c
    print "Best result", best_profit, "(for c =", best_c, ")"
    return best_profit, best_c 
    
def eval_logreg(d, x_train, x_test, y_train, y_test, test_start_idx):
    model = scikits.learn.linear_model.LogisticRegression
    return eval_lin_classifier(model, d, x_train, x_test, y_train, y_test, test_start_idx)
    
def eval_linsvm(d, x_train, x_test, y_train, y_test, test_start_idx):
    model = scikits.learn.svm.LinearSVC
    return eval_lin_classifier(model, d, x_train, x_test, y_train, y_test, test_start_idx)

import encoder     
import scikits.learn 

import cloud
from collections import namedtuple
import progressbar
import multiprocessing

def test_svm(d, x_train, x_test, y_train, y_test, test_start_idx, linear=True):
    
    ts = d['t/100ms'][test_start_idx:]
    bids = d['bid/100ms'][test_start_idx:]
    offers = d['offer/100ms'][test_start_idx:]
    currency_pair = d.currency_pair
    
    n_centroids = [None,  200, 400] 
    cs = [1.0, 5.0]
    cut_thresholds = [.0005, .001, .0015,  0.002]
    transformations = ['prob', 'triangle']
    class_weights = [15, 30, 45] 
    results = {}
    profits = {} 
    best_profit = 0 
    best_desc = None 
    best_result = None 
    
    Params = namedtuple('Params', 'k transformation C pos_weight neg_weight cut_thresh')
    Result = namedtuple('Result', 'profit ntrades ppt accuracy tp fp tn fn tz fz')
    num_class_weights = len(class_weights)
    nparams = len(n_centroids) * len(cs) * len(transformations)  * num_class_weights * num_class_weights * len(cut_thresholds)
    print "Total param combinations: ", nparams
    for k in n_centroids:
        print "Training encoder with k = ", k
        e = encoder.FeatureEncoder(x_train, whiten=False, n_centroids=k)
        for t in transformations: 
            x_train_encoded = e.encode(x_train, transformation = t)
            x_test_encoded = e.encode(x_test, transformation = t) 
            for c in cs:
                if linear: svm = scikits.learn.svm.LinearSVC(C = c)
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



def test_knn(d, x_train, x_test, y_train, y_test, test_start_idx):
    
    ts = d['t/100ms'][test_start_idx:]
    bids = d['bid/100ms'][test_start_idx:]
    offers = d['offer/100ms'][test_start_idx:]
    currency_pair = d.currency_pair
    
    n_centroids = [None,  200, 400] 
    ks = [3, 7, 21, 49] 
    cut_thresholds = [.0015,  0.002]
    transformations = ['prob', 'triangle']
    results = {}
    profits = {} 
    best_profit = 0 
    best_desc = None 
    best_result = None 
    
    Params = namedtuple('Params', 'dict_size transformation k cut_thresh')
    Result = namedtuple('Result', 'profit ntrades ppt accuracy tp fp tn fn tz fz')

    nparams = len(n_centroids) * len(ks) * len(transformations)  * len(cut_thresholds)
    print "Total param combinations: ", nparams
    for nc in n_centroids:
        print "Training encoder with dict_size = ", nc
        e = encoder.FeatureEncoder(x_train, whiten=False, n_centroids=nc)
        for t in transformations:
            x_train_encoded = e.encode(x_train, transformation = t)
            x_test_encoded = e.encode(x_test, transformation = t) 
            for k in ks:
                model = scikits.learn.neighbors.NeighborsClassifier(n_neighbors = k, algorithm='ball')
                print "Using knn classifier with k = ", k 
                model.fit(x_train_encoded, y_train)
                pred = model.predict(x_test_encoded)
                for cut in cut_thresholds:
                    desc = Params(nc, t, k, cut)
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

# TODO: test regression 

#def test_knn(d, x_train, x_test, y_train, y_test, test_start_idx):
    
    #ts = d['t/100ms'][test_start_idx:]
    #bids = d['bid/100ms'][test_start_idx:]
    #offers = d['offer/100ms'][test_start_idx:]
    #currency_pair = d.currency_pair
    
    #n_centroids = [None,  200, 400] 
    #ks = [3, 7, 21, 49] 
    #cut_thresholds = [.0015,  0.002]
    #transformations = ['prob', 'triangle']
    #results = {}
    #profits = {} 
    #best_profit = 0 
    #best_desc = None 
    #best_result = None 
    
    #Params = namedtuple('Params', 'dict_size transformation k cut_thresh')
    #Result = namedtuple('Result', 'profit ntrades ppt accuracy tp fp tn fn tz fz')

    #nparams = len(n_centroids) * len(ks) * len(transformations)  * len(cut_thresholds)
    #print "Total param combinations: ", nparams
    #for nc in n_centroids:
        #print "Training encoder with dict_size = ", nc
        #e = encoder.FeatureEncoder(x_train, whiten=False, n_centroids=nc)
        #for t in transformations:
            #x_train_encoded = e.encode(x_train, transformation = t)
            #x_test_encoded = e.encode(x_test, transformation = t) 
            #for k in ks:
                #model = scikits.learn.neighbors.NeighborsClassifier(n_neighbors = k, algorithm='ball')
                #print "Using knn classifier with k = ", k 
                #model.fit(x_train_encoded, y_train)
                #pred = model.predict(x_test_encoded)
                #for cut in cut_thresholds:
                    #desc = Params(nc, t, k, cut)
                    #profit_series = simulate.aggressive_with_hard_thresholds(ts, bids, offers, pred, currency_pair, max_loss_prct = cut)
                    #sum_profit = np.sum(profit_series)
                    #ntrades = np.sum(profit_series != 0)
                    #if ntrades > 0: profit_per_trade = sum_profit / float(ntrades)
                    #else: profit_per_trade = 0 
                    #raw_accuracy, tp, fp, tn, fn, tz, fz = signals.accuracy(y_test, pred)
                    #result = Result(sum_profit, ntrades, profit_per_trade, raw_accuracy, tp, fp, tn, fn, tz, fz)
                    #print desc
                    #print result
                    #print
                    #if result.profit > best_profit:
                        #best_profit = result.profit
                        #best_result = result
                        #best_desc = desc
                    #results[desc] = result
                    #profits[desc] = result.profit 
    #print "Best over all params:"
    #print best_desc
    #print best_result
    #return profits, results 



def test_file(filename, learner='linsvm'):
    d, x_train, x_test, y_train, y_test,test_start_idx = load_file(filename)
    if learner == 'linsvm':
        return test_svm(d, x_train, x_test, y_train, y_test, test_start_idx, linear=True)
    elif learner == 'svm':
        return test_svm(d, x_train, x_test, y_train, y_test, test_start_idx, linear=False)
    else:
        return test_knn(d, x_train, x_test, y_train, y_test, test_start_idx)
    
