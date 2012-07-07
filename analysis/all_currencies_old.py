
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


def param_search(training_days, testing_days, 
        predict_idx = 0, 
        target_precision = 0.6, 
        input_percentiles=[None], #[ 5, 10, 15, 20, 25], 
        output_percentiles=[5, 10],[# 15],
        long_lags = [100, 200, 300, 400, 600],
        short_lags= [75, 100, 200, 300, 400, 500], 
        beta = 2.0, 
        alphas = [ 0.000001], 
        possible_pca_components = [None, 8, 16], 
        possible_pairwise_products = [False, True],
        possible_binning = [False, True],
        etas = [0.01],
        penalties = ['l2'], 
        losses=[ 'hinge'],
        hidden_layer_thresholds = [None],
        possible_final_regression = [False, True] ):
    Params = namedtuple('Params', 
        ('long_lag', 'short_lag', 'future_lag',  \
        'long_input_threshold_percentile', 
        'short_input_threshold_percentile', 
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
        'models', 
        'combiner', 
        'rejector', 
        'corrector', 
        'threshold'))
        
    best_params = None
    best_result = Result(*[0 for _ in Result._fields])

    all_scores = {}
    ensemble = Ensemble()
    for binning in possible_binning:
        # don't allow both binning and pairwise products
        for pairwise_products in ([False] if binning else possible_pairwise_products):
            for long_lag in reversed(long_lags):
                for long_percentile in input_percentiles:
                    for short_lag in [l for l in reversed(short_lags) if l < long_lag]:
                        for short_percentile in input_percentiles:
                            for future_lag in [l for l in short_lags if l >= short_lag and l < long_lag]:
                                for pca_components in possible_pca_components:
                                    input_encoder = InputEncoder(
                                        lag1 = long_lag, 
                                        lag2 = short_lag, 
                                        future_offset = future_lag, 
                                        percentile1 = long_percentile, 
                                        percentile2 = short_percentile, 
                                        binning = binning, 
                                        pairwise_products = pairwise_products, 
                                        pca_components = pca_components)
                                        
                                    train_x = input_encoder.transform(training_days, fit=True)
                                
                                    # to avoid trivial predictions at least make the future percentile greater than the number of 
                                    # ticks into the future we're looking
                                    for output_threshold_percentile in \
                                     [p for p in output_percentiles if p >= short_percentile]:
                                        print 
                                        print " --- lag1 =", long_lag, \
                                            " | lag2 =", short_lag, \
                                            " | future =", future_lag, \
                                            " | long_percentile =", long_percentile, \
                                            " | short_percentile =", short_percentile, \
                                            " | output_percentile =", output_threshold_percentile, \
                                            " | products =", pairwise_products, \
                                            " | binning =", binning, \
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
                                        
                                        test_x = input_encoder.transform(testing_days, fit=False)
                                        test_y = output_encoder.transform([day[predict_idx, :] for day in testing_days], fit=False)
                                        
                                        print "Testing output stats: count(0) = ", np.sum(test_y == 0), "count(1) = ", np.sum(test_y == 1), "count(-1) = ", np.sum(test_y == -1)
                                        print 
                                        sys.stdout.flush()
                                        
                                            
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
                                                                    
                                                                    for final_regression in possible_final_regression:
                                                                        
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
                                                                                long_percentile,
                                                                                short_percentile, 
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
                                                                        
                                                                    
                                                                            score, precision, recall, specificity, threshold, precisions, recalls  = \
                                                                                eval_thresholds(train_y, train_pred, train_reject_signal, beta, target_precision)
                                                                            
                                                                            pred =  raw_pred * (test_reject_signal < threshold)
                                                                            test_score, test_prec, test_recall, test_spec = \
                                                                                eval_prediction(test_y, pred, beta)
                                                                            
                                                                            all_scores[params] = score    
                                                                            
                                                                            if score > 0:
                                                                                ensemble.add(input_encoder, combiner, rejector, recall)
                                                                                
                                                                                print "Train Score =", score, "precision =", precision, "recall =", recall 
                                                                                print "Test score =", test_score, "precision =", test_prec, "recall =", test_recall 
                                                                                print "Predicted output: count(0)=%d, count(1)=%d, count(-1)=%d, filtered=%d" %  \
                                                                                    (np.sum(pred == 0), np.sum(pred == 1), np.sum(pred == -1), np.sum(pred != raw_pred) )
                                                                                
                                                                            else:
                                                                                print "Score = 0"
                                                                            sys.stdout.flush()
                                                                    
                                                                
                                                                            if recall > best_result.train_recall:
                                                                                result = Result(
                                                                                    test_score, test_prec, test_recall, test_spec, 
                                                                                    score, precision, recall, specificity, 
                                                                                    np.array(precisions),  
                                                                                    np.array(recalls), 
                                                                                    test_y, 
                                                                                    pred, 
                                                                                    input_encoder = input_encoder, 
                                                                                    output_encoder = output_encoder, 
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
                                print "Best precision:", best_result.train_precision
                                print "Best recall:", best_result.train_recall
                                print "***"
                                sys.stdout.flush()
    
    probs, reject_scores = ensemble.predict(testing_days)
    pred = np.argmax(probs, 1) - 1
    #score, precision, recall, specificity, threshold, precisions, recalls  = \
    #    eval_thresholds(train_y, pred, combined_rejects, target_precision)
    return ensemble, pred, probs, reject_scores, best_params, best_result, all_scores
                                    
                            
                            
            
         
    
        
        
        

