
import numpy as np 
from treelearn import ClusteredRegression
from sklearn.linear_model import LinearRegression    
import filter 
import features 
import signals 

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

def load_pairwise_tensor(path, fn, reciprocal_fn, diagonal, start_hour = 1, end_hour = 20):
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
    for i in xrange(n):
        ccy_a = clique[i] 
        for j in xrange(n):
            if i == j:
                result[i,j,:] = diagonal
            else: 
                ccy_b = clique[j]
                result[i,j, :] = vectors[ (ccy_a, ccy_b) ] 
    return clique, result, currencies, vectors 
    

def load_pairwise_rates_from_path(path, start_hour=1, end_hour=20):
    fn1 = lambda d: d['bid/100ms']
    fn2 = lambda d: 1.0/d['offer/100ms']
    clique, data, _, _ = \
      load_pairwise_tensor(path, fn1, fn2, 1.0, start_hour, end_hour)
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



def load_pairwise_features_from_path(
      path, 
      signal = signals.bid_offer_cross, 
      start_hour=1, end_hour=20):
    print "Searching for maximum clique"
    clique, clique_rates = \
      load_pairwise_rates_from_path(path, start_hour, end_hour)
    
    clique_size = len(clique)
    print "Found clique of size", clique_size, ":",  clique 
    
    n_scales = 4
    n_pair_features = 3 
    n_pairs = (clique_size-1) * (clique_size-2)
    n_features = n_scales * (clique_size + n_pairs * n_pair_features)
    print "Computing", n_features, "features for", \
      n_pairs, "currency pairs over", n_scales, "time scales"
    feature_list = []
    multiscale_feature_list = [] 
    
    # add currency value gradients to features 
    print "Computing currency values from principal eigenvectors of rate matrices (with shape", clique_rates.shape, ")"
    ccy_values = ccy_value_eig(clique_rates)
    
    for i in xrange(clique_size):       
        gradients = \
          filter.multiscale_exponential_gradients(ccy_values[i, :], n_scales = n_scales)
        feature_list.append(gradients[0, :])
        for scale in xrange(n_scales):
            multiscale_feature_list.append(gradients[scale, :])
    
    # compute difference from ideal rates 
    pair_counter = 0
    for i in xrange(clique_size):
        for j in np.arange(clique_size-i-1)+i+1:
            ideal_midprice = ccy_values[i, :] / ccy_values[j, :]
            midprice = 0.5*clique_rates[i,j,:] + 0.5/clique_rates[j,i,:]

            diff = midprice - ideal_midprice
            feature_list.append(diff)
            smoothed = \
              filter.multiscale_exponential_smoothing(diff, n_scales = n_scales)
            for scale in xrange(n_scales):
                multiscale_feature_list.append(smoothed[scale, :])
            
    
    
    signals = []
    for filename in glob.glob(path):
        d = Dataset(filename)
        ccy_a, ccy_b = d.currency_pair 
        if ccy_a in clique and ccy_b in clique:
            start_idx = hour_to_idx(d.t, start_hour)
            end_idx = hour_to_idx(d.t, end_hour)
            
            print 
            print "Getting features for", d.currency_pair 
            print 
            print "Bid side slope"
            bss = features.bid_side_slope(d, start_idx, end_idx)
            feature_list.append(bss)
            smoothed = filter.multiscale_exponential_smoothing(bss, n_scales = n_scales)
            for scale in xrange(n_scales):
                multiscale_feature_list.append(smoothed[scale, :])
            
            print "Offer side slope"
            oss = features.offer_side_slope(d, start_idx, end_idx)
            feature_list.append(oss)
            smoothed = filter.multiscale_exponential_smoothing(oss, n_scales = n_scales)
            for scale in xrange(n_scales):
                multiscale_feature_list.append(smoothed[scale, :])
            
            
            print "Message count"
            msgs = d['message_count/100ms'][start_idx:end_idx]
            feature_list.append(msgs)
            smoothed_message_counts = filter.multiscale_exponential_smoothing(msgs, n_scales = n_scales)
            for scale in xrange(n_scales):
                multiscale_feature_list.append(smoothed_message_counts[scale, :])
            
            print "Computing output signal for", d.currency_pair  
            y = signal(d, start_idx = start_idx, end_idx = end_idx)
            signals.append(y)
    # assuming d, start_idx, end_idx are still bound 
    print "Time" 
    t = d['t'][start_idx:end_idx] / (3600.0 * 1000 * 24)
    feature_list.append(t)
    multiscale_feature_list.append(t)
            
    print 
    print "Concatenating results"
    simple_features = np.array(feature_list).T
    multiscale_features = np.array(multiscale_feature_list).T
    signals = np.array(signals, dtype='int')
    return simple_features, multiscale_features, signals 
   
def load_clique_values_from_path(path, start_hour=1, end_hour=20):
	print "Searching for maximum clique..."
	clique, rates = load_pairwise_rates_from_path(path, start_hour, end_hour)
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

def transform_pairwise(x, fn = normalized_product):
	d,n = x.shape
	new_features = []
	for i in range(d):
		vi = x[i, :]
		new_features.append(vi)
		for j in range(d):
			if i <= j:
				vj = x[j, :]
				new_features.append(fn(vi, vj))
	return np.array(new_features)
	
def make_returns_dataset(values, past_lag, future_lag = None, predict_idx=0, train_prct = 0.65, pairwise_fn=None, values_are_features=False):
	if future_lag is None:
		future_lag = past_lag
	x, ys = \
	  present_and_future_returns(values, past_lag, future_lag)
	if pairwise_fn is not None:
		x = transform_pairwise(x, pairwise_fn)
	if values_are_features:
		x = np.vstack( [values[:, past_lag:-future_lag], x] ) 
	y = ys[predict_idx, :]
	n = len(y)
	ntrain = int(train_prct * n)
	ntest = n - ntrain 
	xtrain = x[:, :ntrain]
	xtest = x[:, ntrain:]
	ytrain = y[:ntrain]
	ytest = y[ntrain:]
	return xtrain, xtest, ytrain, ytest 

import sklearn 
import sklearn.linear_model
import sklearn.tree 

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
	#model = sklearn.tree.DecisionTreeRegressor(max_depth=20, min_split=7)
	#model = sklearn.linear_model.LinearRegression(copy_X=False)
	model = sklearn.linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=8, copy_X=False)
	model.fit(xtrain.T, ytrain)
	
	#model = treelearn.train_clustered_regression_ensemble(xtrain.T, ytrain, num_models=100, k=25, bagging_percent=0.75, feature_subset_percent=0.75)
	
	
	pred = model.predict(xtest.T)
	mad, mr, p = eval_results(ytest, pred)
	return mad, mr, p, ytest, pred 

import sys

def param_search(values, predict_idx = 0, values_are_features = False, pairwise_fn = None, dataset_start_hour=1, dataset_end_hour=20):
	
	# data I was using only went up to 20th hour, assume each slice
	# is 3 hours long 
	last_hour = dataset_end_hour - dataset_start_hour 
	dur_hours = 2
	ticks_per_second = 10 
	start_hours = np.arange(last_hour - dur_hours)
	lags = ticks_per_second * np.array([10, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600])
	nlags = len(lags)
	same_sign_results = []
	mad_ratio_results = []
	best_score = 0
	best_data = None 
	
	for start_hour in start_hours:
		# 1 + the hour since we're assume 
		real_start_hour = dataset_start_hour + start_hour
		print "---- Start Hour:", real_start_hour, "---"
		if best_data: print "Best so far:", best_score, best_data 
		end_hour = start_hour + dur_hours
		multiplier = ticks_per_second * 60 * 60
		start_tick = start_hour * multiplier
		end_tick = end_hour * multiplier 
		slice_values = values[:, start_tick:end_tick]
		score_result = np.zeros(nlags )
		mr_result = np.zeros(nlags)
		
 		for i, lag in enumerate(lags):
			mad, mr, prct_same_sign, y, pred = \
			  eval_returns_regression(slice_values, lag, predict_idx=predict_idx, pairwise_fn = pairwise_fn, values_are_features=values_are_features)
			print "Lag =", lag /10, \
				"| mean change =", np.mean(y), \
				"| mean predicted =", np.mean(pred), \
				"| mad =", mad, \
				"| mad_ratio =", mr, \
				"| same_sign =", prct_same_sign

			sys.stdout.flush()
			score_result[i] = prct_same_sign
			mr_result[i] = mr
			score = prct_same_sign**2 * (1.0/mr)
			if best_score < score:
				best_score = score
				best_data = { 
					'start_hour': real_start_hour, 
					'lag': lag / 10, 
					'mad': mad, 
					'mad_ratio': mr, 
					'prct_same_sign': prct_same_sign, 
					'y_test': y, 
					'y_pred': pred, 
				}
		same_sign_results.append(score_result)	
		mad_ratio_results.append(mr_result)
	print "Best overall:", best_score, best_data 
	return best_score, best_data, same_sign_results, mad_ratio_results
	
	
	
	
        
        
        
