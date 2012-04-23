
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

def eval_results(y, pred):
    mad = np.mean(np.abs(y-pred))
    mad_ratio = mad/ np.mean( np.abs(y) )
    prct_same_sign = np.mean(np.sign(y) == np.sign(pred))
    return mad, mad_ratio, prct_same_sign

import treelearn 
def eval_returns_regression(values, past_lag, future_lag = None, predict_idx=0, train_prct=0.5, pairwise_fn = None, values_are_features=False):
    if future_lag is None:
        future_lag = past_lag
    
    xtrain, xtest, ytrain, ytest = \
      make_returns_dataset(values, past_lag, future_lag, predict_idx, train_prct, pairwise_fn = pairwise_fn, values_are_features = values_are_features)
    
    avg_output = np.mean(np.abs(ytrain))
    avg_input = np.mean(np.abs(xtrain))
    n_features = xtrain.shape[0]
    model = sklearn.ensemble.ExtraTreesRegressor(100)
    #model = sklearn.linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=8, copy_X=False)
    #model = sklearn.svm.SVR(kernel='linear', epsilon=0.001 * avg_output, gamma=avg_input/n_features, scale_C = True)
    #model = sklearn.tree.DecisionTreeRegressor(max_depth=20, min_split=7)
    #model = sklearn.linear_model.LinearRegression(copy_X=True)
    #model = sklearn.linear_model.Ridge(alpha=avg_output)
    
    model.fit(xtrain.T, ytrain)
    
    #model = treelearn.train_clustered_regression_ensemble(xtrain.T, ytrain, num_models=100, k=25, bagging_percent=0.75, feature_subset_percent=1.0)
    #model = treelearn.train_random_forest(xtrain.T, ytrain)
    #model = treelearn.train_clustered_ols(xtrain.T, ytrain)
    
    
    pred = model.predict(xtest.T)
    mad, mr, p = eval_results(ytest, pred)
    return mad, mr, p, ytest, pred 

import sys

from scipy import stats

class InputEncoder:
    def __init__(self, lag1, lag2, future_offset, thresh_percentile, pairwise_products=False):
        self.lag1 = lag1
        self.lag2 = lag2 
        self.future_offset = future_offset
        self.thresh_percentile = thresh_percentile
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
        
        features = np.zeros( (2*n_base_features, n_samples), dtype='float')
        for i in xrange(n_base_features):
            row = returns[i, :]
            if fit:
                bottom_thresh = stats.scoreatpercentile(row, self.thresh_percentile)
                self.bottom_threshes.append(bottom_thresh)
                top_thresh = stats.scoreatpercentile(row, 100 - self.thresh_percentile)
                self.top_threshes.append(top_thresh)
            else:
                top_thresh = self.top_threshes[i]
                bottom_thresh = self.bottom_threshes[i]
            features[2*i, :] = row < bottom_thresh
            features[2*i+1, :] = row > top_thresh
        if self.pairwise_products:
            return transform_pairwise(features, fn = np.multiply)
        else:
            return features 
    

#def inputs_from_values(v, lag1, lag2, future_offset, thresh_percentile, pairwise_products=False):
    

#assumes 1d output 
def _one_day_future_returns(v, future_offset, past_lag ):
    return np.log ( v[(past_lag+future_offset):] / v[past_lag:-future_offset])
    
def _all_day_future_returns(vs, future_offset, past_lag ):
    return np.hstack([_one_day_future_returns(v, future_offset, past_lag) for v in vs])

def outputs_from_values(vs, future_offset, past_lag, thresh_percentile):
    returns = _all_day_future_returns(vs, future_offset, past_lag) 
    result = np.zeros(returns.shape, dtype='int')
    bottom_thresh = stats.scoreatpercentile(returns, thresh_percentile)
    top_thresh = stats.scoreatpercentile(returns, 100 - thresh_percentile)
    result[returns < bottom_thresh] = -1 
    result[returns > top_thresh] = 1 
    return result 


    
def f_score(precision, recall, beta=1.0):
    top = precision * recall
    beta_squared = beta * beta 
    bottom = beta_squared * precision + recall 
    return (1 + beta_squared) * (top / bottom)


from collections import OrderedDict, namedtuple 

def param_search(training_days, testing_days, predict_idx = 0, \
        percentiles=[5,10,15], 
        lags=[25, 50, 100, 200, 400], beta = 0.5, Cs =[1, 10]):
    Params = namedtuple('Params', 
        ('long_lag', 'short_lag', 'future_lag',  \
        'input_threshold_percentile', 'output_threshold_percentile', 
        'pairwise_products', 'c'))
    Result = namedtuple('Result', 
        ('score', 'precision', 'recall', 
        'sensitivity', 'specificity',
        'y', 'ypred', 'model'))
    best_params = None
    best_result = None
    best_score = 0
    for c in Cs:
        for pairwise_products in [False, True]:
            for long_lag in lags:
                for short_lag in [l for l in lags if l < long_lag]:
                    for future_lag in [short_lag]: # [l for l in lags if l <= long_lag]:
                        print 
                        print "--- C = ", c, " | long_lag =", long_lag, " | short_lag =", short_lag, " | future_lag =", future_lag, " | pairwise_products = ", pairwise_products, " ---"
                        print 
                        for input_threshold_percentile in percentiles:
                            for output_threshold_percentile in [input_threshold_percentile]: #[p for p in percentiles if p >= input_threshold_percentile]:
                                
                                params = Params(long_lag, short_lag, future_lag,
                                    input_threshold_percentile, 
                                    output_threshold_percentile,
                                    pairwise_products, c)
                                print params 
                                sys.stdout.flush()
                                e = InputEncoder(
                                        lag1 = long_lag, 
                                        lag2 = short_lag, 
                                        future_offset = future_lag, 
                                        thresh_percentile = input_threshold_percentile, 
                                        pairwise_products = pairwise_products)

                                def make_outputs(days):
                                    return outputs_from_values([day[predict_idx, :] for day in days],
                                            future_offset = future_lag,
                                            past_lag = long_lag, 
                                            thresh_percentile = output_threshold_percentile)
                                            
                                train_x = e.transform(training_days, fit=True)
                                train_y = make_outputs(training_days)
                                
                                print "Training output stats: count(0) = ", np.sum(train_y == 0), "count(1) = ", np.sum(train_y == 1), "count(-1) = ", np.sum(train_y == -1)
                                sys.stdout.flush()
                                
                                test_x = e.transform(testing_days, fit=False)
                                test_y = make_outputs(testing_days)
                                
                                print "Testing output stats: count(0) = ", np.sum(test_y == 0), "count(1) = ", np.sum(test_y == 1), "count(-1) = ", np.sum(test_y == -1)
                                sys.stdout.flush()
                                
                                
                                print "...training model..."
                                sys.stdout.flush()
                                
                                model = sklearn.linear_model.LogisticRegression(penalty='l1', C=c)
                                model.fit(train_x.T, train_y)
                                pred = model.predict(test_x.T)
                                
                                print "Predicted output stats: count(0) = ", np.sum(pred == 0), "count(1) = ", np.sum(pred == 1), "count(-1) = ", np.sum(pred == -1)
                                sys.stdout.flush()
                                
                                nonzero = test_y != 0
                                zero = test_y == 0
                                pred_nonzero = pred != 0
                                pred_zero = pred == 0
                                
                                total = len(pred)
                                tp = np.sum(nonzero & pred_nonzero)
                                tn = np.sum(zero & pred_zero)
                                fp = np.sum(pred_nonzero & zero)
                                fn = total - (tp + tn + fp)
                                
                                sensitivity = tp / float(tp + fn)
                                specificity = tn / float(tn + fp)
                                precision = tp / float(tp + fp)
                                recall = tp / float (tp + fn)
                                score = f_score(precision, recall, beta)
                                
                                    
                                result = Result(score, precision, recall, sensitivity, specificity, test_y, pred, model)
                                            
                                print "Score =", score, "precision =", precision, "recall =", recall 
                                print 
                                sys.stdout.flush()
          
                                if score > best_score:
                                    best_score = score
                                    best_params = params
                                    best_result = result 
    print "Best score:", best_score
    print "Best params:", best_params
    print "Best result:", best_result 
    return best_params, best_result 
                                    
                            
                            
            
         
    
        
        
        
