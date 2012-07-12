
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
        print "===> ", filename
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
    def __init__(self, 
            long_lag, 
            short_lag, 
            future_offset, 
            keep_original = False, 
            long_percentile = None, 
            short_percentile = None,
            normalize = False, 
            multiply_by = 1, 
            log_returns = False, 
            binning=False, 
            pairwise_products=False, 
            pca_components=None):
        if long_lag < short_lag:
            long_lag, short_lag = short_lag, long_lag 
        self.keep_original = keep_original 
        self.long_lag = long_lag
        self.short_lag = short_lag 
        self.future_offset = future_offset
        self.long_percentile = long_percentile
        self.short_percentile = short_percentile
        self.normalize = normalize 
        self.multiply_by = multiply_by 
        self.log_returns = log_returns 
        self.binning = binning
        self.pairwise_products = pairwise_products
        
        if pca_components is None:
            self.pca = None
        else:
            self.pca = sklearn.decomposition.RandomizedPCA(pca_components)
    
    def _one_day_returns(self, v):
        long_ratio = v[:, self.long_lag:] / v[:, :-self.long_lag]
        short_ratio = v[:, self.short_lag:] / v[:, :-self.short_lag]
        
        if self.log_returns:
            long_returns = np.log(long_ratio)
            short_returns = np.log(short_ratio)
        else:
            long_returns = long_ratio - 1.0
            short_returns = short_ratio - 1.0 
            
        # align to make returns the same length
        short_returns = short_returns[:, (self.long_lag - self.short_lag):]
            
        # truncate past so it aligns with vector of future returns 
        long_returns = long_returns[:, :-self.future_offset]
        short_returns = short_returns[:, :-self.future_offset]
        return np.vstack([long_returns, short_returns])
        
        
    def _all_day_returns(self, vs):        
        return np.hstack([self._one_day_returns(v) for v in vs])

    def _transform(self, vs, fit=False, final_type='float'):
        if not isinstance(vs, list): 
            vs = [vs]
            
        returns = self._all_day_returns(vs)
        n_original_features = returns.shape[0] / 2
        
        if fit and self.long_percentile:
            self.bottom_threshes = []
            self.top_threshes = [] 
            
        n_base_features = returns.shape[0]
        n_samples = returns.shape[1]
        
        if self.long_percentile and self.short_percentile:
            features = np.zeros( (n_base_features, n_samples), dtype='int')
            for i in xrange(n_base_features):
                row = returns[i, :]
                if fit:
                    p = self.long_percentile if i < n_original_features else self.short_percentile
                    
                    bottom_thresh = stats.scoreatpercentile(row[row < 0], p)
                    self.bottom_threshes.append(bottom_thresh)
                    
                    top_thresh = stats.scoreatpercentile(row[row > 0], 100 - p)
                    self.top_threshes.append(top_thresh)
                else:
                    top_thresh = self.top_threshes[i]
                    bottom_thresh = self.bottom_threshes[i]
                features[i, :] = -1 * (row < bottom_thresh) + (row > top_thresh)
        else:
            features = returns

        if self.pairwise_products:
            base_features = features
            n = features.shape[0]
            n_total_features = (n * n + n) / 2  
            features = np.zeros( (n_total_features, n_samples), dtype=features.dtype)
            idx = 0
            for i in xrange(n):
                for j in xrange(n):
                    if i <= j:
                        features[idx,:] = base_features[i,:] * base_features[j, :]
                        idx = idx + 1
            assert idx == n_total_features
        
        features *= self.multiply_by 
        
        if self.pca and fit:
            features = self.pca.fit_transform(features.T).T
        elif self.pca:
            features = self.pca.transform(features.T).T
        
        if self.binning:
            nrows = features.shape[0]
            binned_features = np.zeros( (2*nrows, n_samples), dtype=features.dtype)
            for i in xrange(nrows):
                row = features[i, :]
                pos = row > 0
                binned_features[2*i, pos] = row[pos]
                binned_features[2*i+1, ~pos] = -1*row[~pos]
            features = binned_features
          
        return features.astype(final_type)
    
    def transform(self, vs, final_type='float'):
        return self._transform(vs, fit=False, final_type=final_type)
    
    def fit_transform(self, vs, final_type='float'):
        return self._transform(vs, fit=True, final_type=final_type)
    


#assumes 1d output 

class OutputEncoder:
    def __init__(self, future_offset, past_lag, thresh = None, thresh_is_percentile = False, log_returns=False):
        self.future_offset = future_offset
        self.past_lag = past_lag 
        self.thresh = thresh
        self.thresh_is_percentile = thresh_is_percentile
        self.log_returns = log_returns
        
    def _one_day_future_returns(self, v ):
        ratio = v[self.future_offset:] / v[:-self.future_offset]
        # truncate data to line up with inputs
        ratio = ratio[self.past_lag:]
        if self.log_returns:
            return np.log(ratio)
        else:
            return ratio - 1.0 
    
    def _all_day_future_returns(self, vs ):
        all_returns = []
        for v in vs:
            r = self._one_day_future_returns(v)
            all_returns.append(r)
        return np.hstack(all_returns)
    
    def _transform(self, vs, fit=False):
        if not isinstance(vs, list): 
            vs = [vs]
        returns = self._all_day_future_returns(vs) 
        result = np.zeros(returns.shape, dtype='int')
        if fit and self.thresh_is_percentile:
            self.bottom_thresh = stats.scoreatpercentile(returns[returns < 0], self.thresh)
            self.top_thresh = stats.scoreatpercentile(returns[returns > 0], 100 - self.thresh)
        if self.thresh_is_percentile:
            result[returns < self.bottom_thresh] = -1 
            result[returns > self.top_thresh] = 1 
        else:
            result[returns < -self.thresh] = -1 
            result[returns > self.thresh] = 1
        return result
        
    def transform(self, vs):
        return self._transform(vs, fit=False)
        
    def fit_transform(self, vs):
        return self._transform(vs, fit=True)
        
class Ensemble():
    def __init__(self):
        self.classifiers = []
        self.rejectors = []
        self.weights = []
        self.encoders = []
                
    def add(self, encoder, classifier, rejector, weight = 1.0):
        self.encoders.append(encoder)
        self.classifiers.append(classifier)
        self.rejectors.append(rejector)
        self.weights.append(weight)
    
    # how to combine days!? They all have initial periods of
    # watching passively
    def predict(self, days):
        n_days = len(days)
        samples_per_day = np.array([day.shape[1] for day in days])
        n_samples = np.sum(samples_per_day)
        print n_samples_per_day 
        
        counts = np.zeros( (n_samples,3), dtype='int' )
        reject_score = np.zeros(n_samples, dtype='float')
        print n_samples 
        for i,c  in enumerate(self.classifiers):
            encoder = self.encoders[i]
            x_encoded = encoder.transform(days).T
            print "x", x_encoded.shape
            w = self.weights[i]
            pred = np.sign(c.predict(x_encoded))
            r = self.rejectors[i]
            reject_prob = r.predict_proba(x_encoded)
            base_idx = 0
            all_indices = np.arange(n_samples)
            down = pred == -1
            zero = pred == 0
            up = pred == 1 
            for n in samples_per_day:
                end_idx = base_idx + n
                lhs_mask = (all_indices >= base_idx) & (all_indices < end_idx)
                rhs_mask = lhs_mask.copy()
                longest_lag = max(encoder.lag1, encoder.lag2)
                rhs_mask[:longest_lag] = 0
                counts[lhs_mask & down, 0] += w
                counts[lhs_mask & zero, 1] += w
                counts[lhs_mask & up, 2] += w
                reject_score += w * reject_prob
                base_idx += n 
            
        normalizer = sum(self.weights)
        return counts / normalizer, reject_score / normalizer
            
    
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


import os
def get_immediate_subdirectories(d):
    return sorted(
            [name for name in os.listdir(d)
                if os.path.isdir(os.path.join(d, name))])

def load_all_days(base_dir, start_hour=3, end_hour=7):
    values = []
    names = []
    first_clique = None
    for s in get_immediate_subdirectories(base_dir):
        feature_dir = os.path.join(base_dir, s, 'features')
        if os.path.exists(feature_dir) and os.path.isdir(feature_dir):
            wildcard = os.path.join(feature_dir, '*.hdf')
            try:
                print "Loading", wildcard 
                v, c, _ = load_clique_values_from_path(wildcard, start_hour, end_hour, first_clique)
                if v.shape[0] != len(c):
                    raise RuntimeError('Unexpected number of currencies (expected %d, got %d)' % (len(c), v.shape[0]))
                if v.shape[1] == 0:
                    raise RuntimeError('Empty dataset')
                first_clique = c
                names.append(s)
                values.append(v)
            except: 
                print "Failed", sys.exc_info()
    return values, names

            
def eval_thresholds(y, pred, reject_signal, beta, target_precision):
        score, precision, recall, specificity = (0,0,0,0)
        precisions = []
        recalls = []
        threshold = None
        for candidate_threshold in np.unique(reject_signal)[1:]:
            candidate_pred =  pred * (reject_signal < candidate_threshold)
            cscore, cp, cr, cspec = eval_prediction(y, candidate_pred, beta)
            precisions.append(cp)
            recalls.append(cr)
            print candidate_threshold, cp, cr 
            if cp >= target_precision:
                if cscore > score:
                    threshold = candidate_threshold
                    score, precision, recall, specificity = cscore, cp, cr, cspec
            elif cp < 0.9 * target_precision:
                break 
        return score, precision, recall, specificity, threshold,  precisions, recalls

import random
from confident_density_thresholding import ClusterThresholding
def param_search(
        days, 
        verbose = False, 
        target_precision = 0.75,
        n_training = 9, 
        ccy = 0,
        multiply_by = [10000 , 1], 
        long_lags = [400, 300, 200],
        short_lags = [100, 200, 300], 
        pca_components = [ None, 10],
        log_returns = [True], 
        n_iters = [10, 20], 
        all_restarts = [True, False], 
        gmm = [False, True],
        n_clusters = [8, 16], 
        output_targets = [0.0001, 0.0002], 
        binning = [ False, True] ):
    n_days = len(days)
    
    Params = namedtuple('Params', 
        ('long_lag', 'short_lag', 'future_offset',  \
        'pca_components',
        'log_returns', 
        'binning', 
        'multiply_by', 
        'output_target', 
        'n_iter', 
        'all_restarts', 
        'n_clusters',
        'gmm', 
        ))
    #Result = namedtuple('Result', 
    #    ('score', 
    #    'precision', 
    #    'recall'))
    
    best_score = None
    best_precision = None
    best_recall = None
    best_params = None
    
    def print_best():
        print 
        print "*** Best params:", best_params
        print "    score =", best_score, "precision =", best_precision, "recall =", best_recall 
        print 
    for g in gmm:
        for long_lag in reversed(long_lags):
            for short_lag in reversed([l for l in short_lags if l < long_lag]):
                for future_offset in reversed([l for l in short_lags if l >= short_lag and l <= long_lag]):
                    for output_target in output_targets:
                        for pca in pca_components:
                            for lr in log_returns:
                                for b in binning:
                                    for m in multiply_by:
                                        for ni in n_iters:
                                            for ar in all_restarts:
                                                for nc in n_clusters:                                                
                                                    params = Params(long_lag, short_lag, future_offset, pca, lr, b, m, output_target, ni, ar, nc, g)
                                                    print params 
                                                    precisions = []
                                                    recalls = []
                                                    scores = []
                                                    failed = False
                                                    for start_idx in xrange(n_days - n_training - 1):
                                                        ie = InputEncoder(
                                                            long_lag = long_lag, 
                                                            short_lag = short_lag, 
                                                            future_offset = future_offset, 
                                                            binning = b, 
                                                            multiply_by= m, 
                                                            log_returns = lr, 
                                                            pca_components = pca)
                                                        oe = OutputEncoder(
                                                            future_offset = future_offset, 
                                                            past_lag = long_lag, 
                                                            thresh = output_target, 
                                                            log_returns=False)
                                                        test_idx = start_idx + n_training
                                                        test_days = [days[test_idx]]
                                                        training_indices =   range(start_idx, start_idx+n_training)
                                                        training_days = [ days[i] for i in training_indices]
                                                        x_train = ie.fit_transform(training_days)
                                                        x_test = ie.transform(test_days)
                                                        y_train = oe.fit_transform([d[ccy, :] for d in training_days])
                                                        y_test = oe.transform([d[ccy, :] for d in test_days])
                                                        
                                                        print "\t - iter %d: " % start_idx
                                                        print "\t   " \
                                                            "training permutation =", training_indices, \
                                                            "train shape=", x_train.shape, \
                                                            "test day=", test_idx, \
                                                            "test_shape=", x_test.shape
                                                        print "\t   train nz = %d, test nz = %d" % (np.sum(y_train !=0), np.sum(y_test != 0))
                                                        # ignore targets which never occur 
                                                        if np.sum(y_train != 0) == 0 or np.sum(y_test != 0) == 0:
                                                            print "Skipping", params
                                                            failed = True
                                                            break 
                                                        model = ClusterThresholding(target_precision=target_precision,  
                                                            n_clusters = nc, 
                                                            neutral_class = 0, 
                                                            n_iters = ni, 
                                                            n_restarts = ni if ar else ni / 2, 
                                                            gmm = g, 
                                                            verbose = verbose)
                                                        try:
                                                            train_score = model.fit(x_train.T, y_train)
                                                        except: 
                                                            print "Failed to converge!"
                                                            failed = True
                                                            break
                                                        y_pred = model.predict(x_test.T)
                                                        score, prec, recall, _ = eval_prediction(y_test, y_pred, beta=0.25)
                                                        scores.append(score)
                                                        recalls.append(recall)
                                                        precisions.append(prec)
                                                        print "\t   pred nz = %d, train_score = %s | score = %s, precision = %s, recall = %s" % \
                                                            ( np.sum(y_pred != 0), train_score, score, prec, recall)
                                                        print 
                                                    if not failed:
                                                        avg_precision = np.mean(precisions)
                                                        avg_recall = np.mean(recalls)
                                                        avg_score = np.mean(scores)
                                                        print "Average precision = %s, recall = %s" % (avg_precision, avg_recall)
                                                        if best_score is None or avg_score > best_score:
                                                            best_score = avg_score 
                                                            best_precision = avg_precision 
                                                            best_recall = avg_recall 
                                                            best_params = params
                                                    print "---"
                    print_best()
    print_best()
    return best_params, best_score, best_recall, best_precision
        
