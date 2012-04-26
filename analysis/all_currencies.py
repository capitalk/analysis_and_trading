
import numpy as np 
from treelearn import ClusteredRegression
from sklearn.linear_model import LinearRegression    
import filter 
import features 
import signals 


def powerset(seq):
    if isinstance(seq, set):
        seq = list(seq)
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item


def make_pairs(xs):
    return [ (x,y) for x in xs for y in xs]
    
def maximum_clique(unique_names, pairs):
    best_subset = []
    for subset in powerset(unique_names):
        if len(subset) > len(best_subset):
            good = True 
            for (ccy_a, ccy_b) in make_pairs(subset):
                if ccy_a != ccy_b and (ccy_a, ccy_b) not in pairs:
                    good = False
                    break
            if good:
                best_subset = subset 
    return best_subset 
            
import glob
from dataset import Dataset 
from dataset_helpers import hour_to_idx

def load_pairwise_tensor(path, fn, reciprocal_fn, diagonal, start_hour = 1, end_hour = 20, expect_clique = None):
    currencies = set([])
    vectors = {}
    nticks = None
    all_files = glob.glob(path)
    assert len(all_files) > 0
    for filename in all_files:
        d = Dataset(filename)
        start_idx = hour_to_idx(d.t, start_hour)
        end_idx = hour_to_idx(d.t, end_hour)
        nticks = end_idx - start_idx 
        ccy_a, ccy_b = d.currency_pair
        currencies.add(ccy_a)
        currencies.add(ccy_b)
        vectors[ (ccy_a, ccy_b) ] = fn(d)[start_idx:end_idx] 
        vectors[ (ccy_b, ccy_a) ] = reciprocal_fn(d)[start_idx:end_idx]

    clique = list(maximum_clique(currencies, vectors))
    
    n = len(clique)
    result = np.zeros( [n,n, nticks], dtype='float')
    print 'tensor', result.shape
    for i in xrange(n):
        ccy_a = clique[i] 
        for j in xrange(n):
            if i == j:
                result[i,j,:] = diagonal
            else: 
                ccy_b = clique[j]
                result[i,j, :] = vectors[ (ccy_a, ccy_b) ] 

    if expect_clique is not None:
        assert set(clique) == set(expect_clique)
        
        permuted_result = np.zeros_like(result)
        for i in xrange(n):
            ccy_a = clique[i]
            pi = expect_clique.index(ccy_a)
            for j in xrange(n):
                ccy_b = clique[j]
                pj = expect_clique.index(ccy_b)
                permuted_result[pi,pj,:] = result[i,j,:]
                
        result = permuted_result
        clique = expect_clique 
    return clique, result, currencies, vectors 
    

def load_pairwise_rates_from_path(path, start_hour=1, end_hour=20, expect_clique=None):
    fn1 = lambda d: d['bid/100ms']
    fn2 = lambda d: 1.0/d['offer/100ms']
    clique, data, _, _ = \
      load_pairwise_tensor(path, fn1, fn2, 1.0, start_hour, end_hour, expect_clique = expect_clique)
    return clique, data 

def load_pairwise_message_counts_from_path(path, start_hour = 1, end_hour = 20):
    fn = lambda d: d['message_count/100ms']
    return load_pairwise_tensor(path, fn, fn, 0, start_hour, end_hour)
    
import scipy.stats 
def ccy_value_hmean(rate_matrix_series):
    """
    input = k * k * t array, where k is the number of currencies and 
    t is the number of timesteps, return the harmonic mean of each row
    per each timestep
    """
    return scipy.stats.mstats.hmean(rate_matrix_series, axis=1)
    
def foreach_matrix(f, ms):
    """Apply function f to each 2D matrix, iterating over 3rd dim"""
    return np.array([f(ms[:, :, i]) for i in xrange(ms.shape[2])]).T

def abs_first_eigenvector(x):
    _, V = np.linalg.eig(x)
    return np.abs(V[:, 0]) # abs since it might be scaled by an imaginary number 
    
def ccy_value_eig (rate_matrix_series):
    """
    input = k * k * t array where k is the number of currencies
    and t is the number of timesteps. 
    return the scaled singular vector for each timestep
    """
    return foreach_matrix(abs_first_eigenvector, rate_matrix_series)
    
def normalized_first_singular_vector(x):
     U,_, _ = np.linalg.svd(x)
     u0 = U[:, 0]
     return  u0 / np.sum(u0)
     
def ccy_value_svd(rate_matrix_series):
    return foreach_matrix(normalized_first_singular_vector, rate_matrix_series)
    

def percent_max_eigenvalue(rate_matrix):
    eigs = np.abs(np.linalg.eigvals(rate_matrix))
    return np.max(eigs) / np.sum(eigs)

def percent_max_eigenvalues(rate_matrix_series):
    return foreach_matrix(percent_max_eigenvalue, rate_matrix_series)
    
def inconsistency(m):
    m_squared = np.dot(m,m)
    diff = m_squared - m.shape[0]*m
    eigs = np.linalg.eigvals(diff)
    return np.mean(np.abs(eigs))

def inconsistencies(rate_matrix_series):
    return foreach_matrix(inconsistency, rate_matrix_series)

def make_ideal_rate_matrix(values):
    n = values.shape[0]
    mat = np.zeros([n,n] )
    for i in xrange(n):
        for j in xrange(n):
            mat[i,j] = values[i] / values[j]
    return mat 

def make_ideal_rate_series(value_series):
    """give a time series of value vectors, 
        create idealized (noise-free) rate matrices"""
    n_ccy, n_timesteps = value_series.shape
    result = np.zeros( [n_ccy, n_ccy, n_timesteps])
    for i in xrange(n_ccy):
        for j in xrange(n_ccy):
            result[i, j, :] = value_series[i, :] / value_series[j, :] 
    return result 

def difference_from_ideal_rate(rates, values):
    ideal_rates = make_ideal_rate_series(values)
    differences = np.zeros_like(rates)
    n = rates.shape[0]
    for i in xrange(n):
        for j in xrange(n):
            midprice = 0.5*rates[i,j, :] + 0.5 / rates[j, i, :]
            differences[i,j,:] = midprice - ideal_rates[i,j, :]
    return differences 



   
def load_clique_values_from_path(path, start_hour=1, end_hour=20, expect_clique=None):
    print "Searching for maximum clique..."
    clique, rates = load_pairwise_rates_from_path(path, start_hour, end_hour, expect_clique)
    print rates.shape
    print "Found", clique, " ( length = ", len(clique), ")"
    print "Computing currency values..." 
    values = ccy_value_eig(rates)
    return values, clique, rates 

def returns(values, lag):
    present = values[:, :-lag]
    future = values[:, lag:]
    return np.log(future / present)

def present_and_future_returns(values, past_lag, future_lag=None): 
    if future_lag is None:
        future_lag = past_lag
    past_returns = np.log(  values[:, past_lag:] / values[:, :-past_lag])
    # truncate past_returns 
    past_returns = past_returns[:, :-future_lag]
    future_returns = np.log ( values[:, (past_lag+future_lag):] / values[:, past_lag:-future_lag])
    return past_returns, future_returns


# like a geometric mean but also works for negative numbers
def normalized_product(vi, vj): 
    return np.sign(vi) * np.sign(vj) * np.sqrt(np.abs(vi) * np.abs(vj))

def transform_pairwise(x, fn = normalized_product, diagonal=False):
    d,n = x.shape
    new_features = []
    for i in range(d):
        vi = x[i, :]
        new_features.append(vi)
        for j in range(d):
            if i < j or (diagonal and i == j):
                vj = x[j, :]
                new_features.append(fn(vi, vj))
    return np.array(new_features)
    

import sklearn 
import sklearn.linear_model
import sklearn.tree 
import sklearn.svm 



import sys

from scipy import stats
import scipy.sparse

class InputEncoder:
    def __init__(self, lag1, lag2, future_offset, thresh_percentile, binning=False, pairwise_products=False):
        self.lag1 = lag1
        self.lag2 = lag2 
        self.future_offset = future_offset
        self.thresh_percentile = thresh_percentile
        self.binning = binning
        self.pairwise_products = pairwise_products
    
    def _one_day_returns(self, v):
        returns1 = np.log(v[:, self.lag1:] / v[:, :-self.lag1])
        returns2 = np.log(v[:, self.lag2:] / v[:, :-self.lag2])
        # align to make returns the same length
        if self.lag1 < self.lag2:
            returns1 = returns1[:, (self.lag2 - self.lag1):]
        else:
            returns2 = returns2[:, (self.lag1 - self.lag2):]
        # truncate past so it aligns with vector of future returns 
        returns1 = returns1[:, :-self.future_offset]
        returns2 = returns2[:, :-self.future_offset]
        return np.vstack([returns1, returns2])
        
        
    def _all_day_returns(self, vs):
        
        return np.hstack([self._one_day_returns(v) for v in vs])
        
    def transform(self, vs, fit=False):
        returns = self._all_day_returns(vs)

        if fit:
            self.bottom_threshes = []
            self.top_threshes = [] 
        n_base_features = returns.shape[0]
        n_samples = returns.shape[1]
        
        if self.binning:
            features = np.zeros( (2*n_base_features, n_samples), dtype='bool')
        else: 
            features = np.zeros( (n_base_features, n_samples), dtype='int')
            
        for i in xrange(n_base_features):
            row = returns[i, :]
            if fit:
                bottom_thresh = stats.scoreatpercentile(row[row < 0], self.thresh_percentile)
                self.bottom_threshes.append(bottom_thresh)
                top_thresh = stats.scoreatpercentile(row[row > 0], 100 - self.thresh_percentile)
                self.top_threshes.append(top_thresh)
            else:
                top_thresh = self.top_threshes[i]
                bottom_thresh = self.bottom_threshes[i]
            if self.binning:
                features[2*i, :] = row < bottom_thresh
                features[2*i+1, :] = row > top_thresh
            else:
                features[i, :] = -1 * (row < bottom_thresh) + (row > top_thresh)
                
        if self.pairwise_products:
            base_features = features
            n = features.shape[0]
            n_total_features = (n * n + n) / 2  
            features = np.zeros( (n_total_features, n_samples), dtype='int')
            idx = 0
            for i in xrange(n):
                for j in xrange(n):
                    if i <= j:
                        features[idx,:] = base_features[i,:] * base_features[j, :]
                        idx = idx + 1
            assert idx == n_total_features
        return features.astype('float')
    


#assumes 1d output 

class OutputEncoder:
    def __init__(self, future_offset, past_lag, thresh_percentile):
        self.future_offset = future_offset
        self.past_lag = past_lag 
        self.thresh_percentile = thresh_percentile 
        self.bottom_thresh = None
        self.top_thresh = None

    def _one_day_future_returns(self, v ):
        return np.log ( v[(self.past_lag+self.future_offset):] / v[self.past_lag:-self.future_offset])
    
    def _all_day_future_returns(self, vs ):
        return np.hstack([self._one_day_future_returns(v) for v in vs])
    
    def transform(self, vs, fit=False):
        returns = self._all_day_future_returns(vs) 
        result = np.zeros(returns.shape, dtype='int')
        if fit:
            self.bottom_thresh = stats.scoreatpercentile(returns[returns < 0], self.thresh_percentile)
            self.top_thresh = stats.scoreatpercentile(returns[returns > 0], 100 - self.thresh_percentile)
        result[returns < self.bottom_thresh] = -1 
        result[returns > self.top_thresh] = 1 
        return result 


    
def f_score(precision, recall, beta=1.0):
    top = precision * recall
    beta_squared = beta * beta 
    bottom = beta_squared * precision + recall 
    return (1 + beta_squared) * (top / bottom)

def eval_prediction(test_y, pred, beta=1.0):
    zero = test_y == 0
    nonzero = ~zero
    pred_zero = pred == 0
    pred_nonzero = ~pred_zero

    correct = (pred == test_y)
    incorrect = ~correct
    
    total = len(pred)
    tp = np.sum(nonzero & correct)
    tn = np.sum(zero & correct)
    fp = np.sum(pred_nonzero & incorrect)
    fn = total - tp - tn - fp 
    
    if (tp > 0) and (tn > 0):
        specificity = tn / float(tn + fp)
        precision = tp / float(tp + fp)
        recall = tp / float (tp + fn)
        score = f_score(precision, recall, beta)
    else:
        specificity = 0
        precision = 0
        recall = 0
        score = 0
    return score, precision, recall, specificity

def probs_to_features(up, down, zero, up_v_down, up_v_zero, down_v_zero):
    not_down = 1 - down 
    not_zero = 1 - zero
    not_up = 1 - up

    return np.vstack([
        up, 
        down, 
        zero, 
        up * not_down * not_zero, 
        down * not_up * not_zero, 
        zero * not_up * not_down, 
        up_v_down, 
        up_v_zero, 
        down_v_zero,
        not_up * down_v_zero,  # p(DOWN or ZERO) * p(DOWN | DOWN OR ZERO)
        not_down * up_v_zero, # p (UP or ZERO) * p(UP | UP or ZERO)
        not_zero * up_v_down, # p(DOWN or UP) * p(DOWN | DOWN OR UP)
        
    ])
    

from collections import OrderedDict, namedtuple 
import math 
import copy
import sklearn.ensemble
import sklearn.decomposition

def param_search(training_days, testing_days, 
        predict_idx = 0, 
        target_precision = 0.6, 
        percentiles=[15, 20, 25, 30], 
        lags=[100, 200,  300, 400, 500, 600, 700], beta = 2.0, 
        alphas = [ 0.000001], 
        etas = [0.01],
        penalties = ['l2', ], 
        losses=[ 'hinge'],
        hidden_layer_thresholds = [None] ):
    Params = namedtuple('Params', 
        ('long_lag', 'short_lag', 'future_lag',  \
        'input_threshold_percentile', 
        'output_threshold_percentile', 
        'pairwise_products', 
        'binning', 
        'pca_components', 
        'use_hidden_layer', 
        'hidden_layer_threshold', 
        'final_regression', 
        'use_corrector', 
        'loss', 'penalty', 
        'target_updates', 
        'eta0', 'alpha'))
    Result = namedtuple('Result', 
        ('score', 
        'precision', 
        'recall', 
        'specificity',
        'train_score', 
        'train_precision', 
        'train_recall', 
        'train_specificity', 
        'all_precisions', 'all_recalls', 
        'y', 
        'ypred', 
        'input_encoder', 
        'output_encoder', 
        'pca', 
        'models', 
        'combiner', 
        'rejector', 
        'corrector', 
        'threshold'))
        
    best_params = None
    best_result = Result(*[0 for _ in Result._fields])

    all_scores = {}
    for binning in [False, True]:
        # don't allow both binning and pairwise products
        for pairwise_products in ([False] if binning else [False, True]):
            for long_lag in reversed(lags):
                for short_lag in [l for l in reversed(lags) if l < long_lag]:
                    for future_lag in [l for l in lags if l >= short_lag and l < long_lag]:
                        for input_threshold_percentile in percentiles:
                            input_encoder = InputEncoder(
                                    lag1 = long_lag, 
                                    lag2 = short_lag, 
                                    future_offset = future_lag, 
                                    thresh_percentile = input_threshold_percentile, 
                                    binning = binning, 
                                    pairwise_products = pairwise_products)
                                    
                                    
                            encoded_train_x = input_encoder.transform(training_days, fit=True)
                            for pca_components in [8, None]:
                                if pca_components is not None:
                                    pca = sklearn.decomposition.RandomizedPCA(pca_components)
                                    train_x = pca.fit_transform(encoded_train_x.T).T
                                else:
                                    pca = None
                                    train_x = encoded_train_x

                                # to avoid trivial predictions at least make the future percentile greater than the number of 
                                # ticks into the future we're looking
                                for output_threshold_percentile in [p for p in percentiles if p >= input_threshold_percentile and p <= future_lag]:
                                    print 
                                    print " --- lag1 =", long_lag, \
                                        " | lag2 =", short_lag, \
                                        " | future =", future_lag, \
                                        " | products =", pairwise_products, \
                                        " | binning =", binning, \
                                        " | input_threshold =", input_threshold_percentile, \
                                        " | output_threshold =", output_threshold_percentile, \
                                        " ---"
                                    print 
                            
                                    output_encoder = OutputEncoder(future_offset = future_lag, past_lag = long_lag, thresh_percentile = output_threshold_percentile)
                                    
                                    train_y = output_encoder.transform([day[predict_idx, :] for day in training_days], fit=True)
                                    print "Training output stats: ", \
                                        "down prct =", np.exp(output_encoder.bottom_thresh) -1,  \
                                        "up prct =", np.exp(output_encoder.top_thresh) - 1, \
                                        "count(0) = ", np.sum(train_y == 0), \
                                        "count(1) = ", np.sum(train_y == 1), \
                                        "count(-1) = ", np.sum(train_y == -1)
                                    sys.stdout.flush()
                                    
                                    encoded_test_x = input_encoder.transform(testing_days, fit=False)
                                    test_y = output_encoder.transform([day[predict_idx, :] for day in testing_days], fit=False)
                                    
                                    print "Testing output stats: count(0) = ", np.sum(test_y == 0), "count(1) = ", np.sum(test_y == 1), "count(-1) = ", np.sum(test_y == -1)
                                    print 
                                    sys.stdout.flush()
                                    
                                    if pca_components is None:
                                        test_x = encoded_test_x
                                    else:
                                        test_x = pca.transform(encoded_test_x.T).T
                                        
                                    for loss in losses:
                                        for penalty in penalties:
                                            for alpha in alphas:
                                                for eta0 in etas:
                                                    for target_updates in [1000000]:
                                                        n_samples = train_x.shape[1]
                                                        
                                                
                                                        
                                                        # simplifying assumption: 
                                                        # use same model params for predictor and
                                                        # filters 
                                                        def mk_model(loss = loss, penalty=penalty, n_samples=n_samples, regression = False):
                                                            if regression:
                                                                constructor = sklearn.linear_model.SGDRegressor
                                                            else:
                                                                constructor = sklearn.linear_model.SGDClassifier
                                                            n_iter = int(math.ceil(float(target_updates) / n_samples))
                                                            return constructor (
                                                                penalty= penalty, 
                                                                loss = loss, 
                                                                shuffle = True, 
                                                                alpha = alpha,
                                                                eta0 = eta0,  
                                                                n_iter = n_iter)
                                                            
                                                        for use_hidden_layer in [False, True]:
                                                            for hidden_layer_threshold in \
                                                                hidden_layer_thresholds if use_hidden_layer else [None]:
                                                            
                                                                
                                                                models = {}
                                                                if use_hidden_layer:
                                                                    train_probs = {}
                                                                    test_probs = {}
                                                                    labels = [1, -1, 0]
                                                                    ys = {}
                                                                    
                                                                    for l in labels:
                                                                        model = mk_model('log')
                                                                        ys[l] = (train_y == l) 
                                                                        model.fit(train_x.T, ys[l])
                                                                        models[l] = model
                                                                        train_probs[l] = model.predict_proba(train_x.T)
                                                                        test_probs[l] = model.predict_proba(test_x.T)
                                                                    
                                                                    for i,l1 in enumerate(labels):
                                                                        for j,l2 in enumerate(labels):
                                                                            if i < j:
                                                                                mask = ys[l1] | ys[l2]
                                                                                data = train_x[:, mask].T
                                                                                model = mk_model('log', n_samples = data.shape[0])
                                                                                
                                                                                model.fit(data, ys[l1][mask])
                                                                                
                                                                                models[ (l1, l2) ] = model 
                                                                                train_pred = model.predict_proba(train_x.T)
                                                                                train_probs[ (l1, l2) ] = train_pred
                                                                                train_probs[ (l2, l1) ] = 1 - train_pred
                                                                                
                                                                                test_pred = model.predict_proba(test_x.T)
                                                                                test_probs[ (l1, l2) ] = test_pred 
                                                                                test_probs[ (l2, l1) ] = 1 - test_pred 
                                                                            
                                                                    #probs_to_features(up, down, zero, up_v_down, up_v_zero, down_v_zero):
                                                                    def mk_second_layer_features(ps):
                                                                        if hidden_layer_threshold is not None:
                                                                            h = hidden_layer_threshold
                                                                            return probs_to_features(
                                                                                ps[1] > h, ps[-1] > h, ps[0] > h,
                                                                                ps[(1,-1)] > h, ps[(1,0)] > h, ps[(-1, 0)] > h
                                                                            )    
                                                                        else:
                                                                            return probs_to_features(
                                                                                ps[1], ps[-1], ps[0],
                                                                                ps[(1,-1)], ps[(1,0)], ps[(-1, 0)]
                                                                            )                                                                        
                                                                    train2 = mk_second_layer_features(train_probs)
                                                                    test2 = mk_second_layer_features(test_probs)
                                                                else:
                                                                    train2 = train_x
                                                                    test2 = test_x 
                                                                
                                                                rejector = mk_model('log')
                                                                rejector.fit(train2.T, train_y == 0)
                                                                
                                                                train_reject_signal = rejector.predict_proba(train2.T)
                                                                test_reject_signal = rejector.predict_proba(test2.T)
                                                                
                                                                for final_regression in [False, True]:
                                                                    
                                                                    if final_regression:
                                                                        combiner = mk_model(loss = 'squared_loss', regression = True)
                                                                    else:
                                                                        combiner = mk_model() 
                                                                        
                                                                    combiner.fit(train2.T, train_y)
                                                                    
                                                                    train_pred = combiner.predict(train2.T)        
                                                                    raw_pred = combiner.predict(test2.T)
                                                                    
                                                                    if final_regression:
                                                                        train_pred = np.sign(train_pred)
                                                                        raw_pred = np.sign(raw_pred)
                                                                    
                                                                    
                                                                    for use_corrector in [False]:
                                                                        if use_corrector:
                                                                            wrong = train_pred != train_y
                                                                            n_wrong = np.sum(wrong)
                                                                            print "Num. wrong on training set:", n_wrong, "/", len(wrong)
                                                                            #corrector = sklearn.ensemble.GradientBoostingClassifier(min_samples_split = 100, min_samples_leaf = 10, subsample=0.5)
                                                                            corrector = mk_model(n_samples = n_wrong)
                                                                            #corrector = sklearn.ensemble.RandomForestClassifier(max_depth=10, min_split=100)
                                                                            corrector.fit(train2[:, wrong].T, train_y[wrong])
                                                                            up_idx = raw_pred == 1
                                                                            corrector_output = corrector.predict(test2.T) 
                                                                            raw_pred[up_idx] *=  (corrector_output[up_idx] != -1)
                                                                            down_idx = raw_pred == -1
                                                                            raw_pred[down_idx] *= (corrector_output[down_idx] != 1)
                                                                        else:
                                                                            corrector = None
                                                                        params = Params(
                                                                            long_lag, short_lag, future_lag,
                                                                            input_threshold_percentile, 
                                                                            output_threshold_percentile,
                                                                            pairwise_products, binning, 
                                                                            pca_components, 
                                                                            use_hidden_layer,
                                                                            hidden_layer_threshold, 
                                                                            final_regression, 
                                                                            use_corrector, 
                                                                            loss, penalty, 
                                                                            target_updates, 
                                                                            eta0, alpha)
                                                            
                                                                        print params 
                                                                        sys.stdout.flush()
                                                                    
                                                                            
                                                                        score, precision, recall, specificity = (0,0,0,0)
                                                                        precisions = []
                                                                        recalls = []
                                                                        threshold = None
                                                                        for candidate_threshold in np.unique(train_reject_signal)[1:]:
                                                                            candidate_pred =  train_pred * (train_reject_signal < candidate_threshold)
                                                                            cscore, cp, cr, cspec = eval_prediction(train_y, candidate_pred, beta)
                                                                            precisions.append(cp)
                                                                            recalls.append(cr)
                                                                            print candidate_threshold, cp, cr 
                                                                            if cp >= target_precision:
                                                                                if cscore > score:
                                                                                    threshold = candidate_threshold
                                                                                    score, precision, recall, specificity = cscore, cp, cr, cspec
                                                                            elif cp < 0.9 * target_precision:
                                                                                break 
                                                                
                                                                        pred =  raw_pred * (test_reject_signal < threshold)
                                                                        test_score, test_prec, test_recall, test_spec = \
                                                                            eval_prediction(test_y, pred, beta)
                                                                        
                                                                        all_scores[params] = score    
                                                                        if score > 0:
                                                                            print "Predicted output: count(0)=%d, count(1)=%d, count(-1)=%d, filtered=%d" %  \
                                                                                (np.sum(pred == 0), np.sum(pred == 1), np.sum(pred == -1), np.sum(pred != raw_pred) )
                                                                            
                                                                            print "Train Score =", score, "precision =", precision, "recall =", recall 
                                                                            print "Test score =", test_score, "precision =", test_prec, "recall =", test_recall 
                                                                        else:
                                                                            print "Score = 0"
                                                                        sys.stdout.flush()
                                                                
                                                            
                                                                        if recall > best_result.recall:
                                                                            result = Result(
                                                                                test_score, test_prec, test_recall, test_spec, 
                                                                                score, precision, recall, specificity, 
                                                                                np.array(precisions),  
                                                                                np.array(recalls), 
                                                                                test_y, 
                                                                                pred, 
                                                                                input_encoder = input_encoder, 
                                                                                output_encoder = output_encoder, 
                                                                                pca = pca,
                                                                                models = models,
                                                                                combiner= combiner,
                                                                                rejector = rejector,
                                                                                corrector = corrector, 
                                                                                threshold = threshold)
                                                                            best_params = params
                                                                            best_result = result 
                                                                        print 
                                print "***"
                                print "Best params:", best_params
                                print 
                                print "Best result:", best_result 
                                print 
                                print "Best precision:", best_result.precision
                                print "Best recall:", best_result.recall
                                print "***"
                                sys.stdout.flush()
    return best_params, best_result, all_scores
                                    
                            
                            
            
         
    
        
        
        
