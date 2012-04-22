
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
	

def single_day_param_search(values, predict_idx = 0, values_are_features = False, pairwise_fn = None, dataset_start_hour=1, dataset_end_hour=20):
	
	# data I was using only went up to 20th hour, assume each slice
	# is 3 hours long 
	last_hour = dataset_end_hour - dataset_start_hour 
	dur_hours = 3
	ticks_per_second = 10 
	start_hours = np.arange(last_hour - dur_hours)
	lags = ticks_per_second * np.array([5, 10, 20, 40, 60, 80, 100, 150, 200])
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
	
