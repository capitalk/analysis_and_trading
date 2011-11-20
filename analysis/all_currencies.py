
import numpy as np 
from treelearn import ClusteredRegression
from sklearn.linear_model import LinearRegression    

def best_regression_lags(x, min_lag = 3, 
                            n_lags = 10,
                            num_clusters = 10, 
                            multiplier = 10000,
                            min_move_size = 1.5, 
                            train_prct = 0.65):
    lags = np.arange(n_lags)+min_lag 
    n_rows, n_cols = x.shape 
    train_idx = int(train_prct * n_rows)
    n_test = n_rows - train_idx 
    
    init_dict = { 'score': 0 }
    best = [init_dict] * n_cols 
    
    
    for past_offset in lags:
        past = x[:-past_offset, :]
        present = x[past_offset:, :] 
        past_delta_prct = (present - past) / past 
        inputs = np.zeros(x.shape)
        inputs[past_offset:,:] = past_delta_prct * multiplier 
        input_train = inputs[:train_idx, :]
        input_test = inputs[train_idx:, :] 
        for future_offset in lags:
            
            print "past_offset = ", past_offset, " future_offset=", future_offset
            present = x[:-future_offset, :]
            future = x[future_offset:, :]
            future_delta_prct = (future - present) / present
            outputs = np.zeros(x.shape)
            outputs[:-future_offset, :] = future_delta_prct * multiplier
            output_train = outputs[:train_idx, :]
            output_test = outputs[train_idx:, :] 
            for ccy_idx in xrange(n_cols):
                if num_clusters == 1:
                    model = LinearRegression()
                else:
                    model = ClusteredRegression(num_clusters)
                target = output_train[:, ccy_idx]
                model.fit(input_train, target)
                pred = model.predict(input_test)
                actual = output_test[:, ccy_idx]
                actual_big_moves = np.abs(actual) > min_move_size
                num_big_moves = np.sum(actual_big_moves)
                if num_big_moves > 0:
                    pred_big_moves = np.abs(pred) > 0.5 * min_move_size
                    same_sign = np.sign(actual) == np.sign(pred)
                    correct = actual_big_moves & pred_big_moves & same_sign 
                    score = np.sum(correct, dtype='float') / num_big_moves
                    mean_abs_err = np.mean(np.abs(pred - actual))
                    print "         currency ", ccy_idx, "score = ",  score, "mae = ", mean_abs_err 
                    if score > best[ccy_idx]['score']:
                        best[ccy_idx] = { 
                            'score':score, 
                            'mean_abs_err': mean_abs_err,  
                            'model': model, 
                            'past_offset':past_offset, 
                            'future_offset':future_offset, 
                        }
    return best 
            


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


def pairs(xs):
    return [ (x,y) for x in xs for y in xs]
    
def maximum_clique(unique_names, rates):
    best_subset = []
    for subset in powerset(unique_names):
        if len(subset) > len(best_subset):
            good = True 
            for (ccy_a, ccy_b) in pairs(subset):
                if ccy_a != ccy_b and (ccy_a, ccy_b) not in rates:
                    good = False
                    break
            if good:
                best_subset = subset 
    return best_subset 
            
import glob
from dataset import Dataset 
from dataset_helpers import hour_to_idx

def pairwise_rates_from_path(path, start_hour=18, end_hour=20):
    currencies = set([])
    rates = {}
    nticks = None
    for filename in glob.glob(path):
        d = Dataset(filename)
        start_idx = hour_to_idx(d.t, start_hour)
        end_idx = hour_to_idx(d.t, end_hour)
        nticks = end_idx - start_idx 
        bids = d['bid'][start_idx:end_idx]
        offers = d['offer'] [start_idx:end_idx]
        ccy_a, ccy_b = d.currency_pair
        currencies.add(ccy_a)
        currencies.add(ccy_b)
        rates[ (ccy_a, ccy_b) ] = bids 
        rates[ (ccy_b, ccy_a) ] = 1.0 / offers

    clique = list(maximum_clique(currencies, rates))
    n = len(clique)
    clique_rates = np.zeros( [n,n, nticks], dtype='float')
    for i in xrange(n):
        ccy_a = clique[i] 
        for j in xrange(n):
            if i == j:
                clique_rates[i,j,:] = 1.0
            else: 
                ccy_b = clique[j]
                clique_rates[i,j, :] = rates[ (ccy_a, ccy_b) ] 
    return clique, clique_rates, currencies, rates 

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
    return np.array([f(ms[:, :, i]) for i in xrange(ms.shape[2])])

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
