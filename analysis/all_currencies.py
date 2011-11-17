
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
            
            
    
    
