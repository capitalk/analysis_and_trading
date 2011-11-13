import numpy as np 
import simulate 
import sklearn.metrics 


def eval_regression(y, pred): 
    result = { 
        'mae': np.mean(np.abs(y - pred))
        'mse': sklearn.metrics.mean_square_error(y, pred), 
        'prob_same_sign': np.sum(np.sign(y) == np.sign(pred)) / float(len(y))
    }
    return result
        

def three_class_precision(y_test, pred): 
    pred_zero = pred == 0
    num_zero = np.sum(pred_zero)
    
    pred_pos = pred > 0
    num_pos = np.sum(pred_pos) 
    
    pred_neg = pred < 0
    num_neg = np.sum(pred_neg) 
    num_nonzero = num_pos + num_neg 
    
    y_test_pos = y_test > 0 
    y_test_neg = y_test < 0
    y_test_zero = y_test == 0
    tp = np.sum(y_test_pos & pred_pos)
    tz = np.sum(y_test_zero & pred_zero)
    tn = np.sum(y_test_neg & pred_neg)
    
    fp = num_pos - tp 
    fn = num_neg - tn 
    fz = num_zero - tz 
    total = float(tp + fp + tn + fn)
    if total > 0: precision = (tp + tn) / total 
    else: precision = 0.0 
    return precision, tp, fp, tn, fn, tz, fz
        
    
def eval_prediction(ts, bids, offers, pred, actual, currency_pair, cut=0.0015):

    profit_series = simulate.aggressive_with_hard_thresholds(ts, bids, offers, pred, currency_pair, max_loss_prct = cut, max_hold_time=30000)
    #profit_series, _, _ = simulate.aggressive(ts, bids, offers, pred, currency_pair)
    sum_profit = np.sum(profit_series)
    ntrades = np.sum(profit_series != 0)
    if ntrades > 0: profit_per_trade = sum_profit / float(ntrades)
    else: profit_per_trade = 0 
    
    precision, tp, fp, tn, fn, tz, fz = three_class_precision(actual, pred)
    result = {
        'profit': sum_profit, 
        'ntrades': ntrades, 
        'ppt': profit_per_trade, 
        'precision': precision,
        'tp': tp, 'fp': fp, 
        'tn': tn,  'fn': fn, 
        'tz': tz, 'fz': fz
    }
    return result 


def confusion(pred, actual):
    """Given a predicted binary label vector and a ground truth returns a tuple containing the counts for:
    true positives, false positives, true negatives, false negatives """
    
    # implemented using indexing instead of boolean mask operations
    # since the positive labels are expecetd to be sparse 
    n = len(pred)
    pred_nz_mask = (pred != 0)
    pred_nz_indices = np.nonzero(pred_nz_mask)[0]
        
    total_nz = len(pred_nz_indices)
    tp = sum(pred[pred_nz_indices] == actual[pred_nz_indices])
    fp = total_nz - tp 
    actual_nz_mask = (actual != 0)
    actual_nz_indices = np.nonzero(actual_nz_mask)[0]
        
    fn = sum(1 - pred[actual_nz_indices])
    tn = n - tp - fp - fn 
    return tp, fp, tn, fn 

def f_score(tp, fp, tn, fn, beta=0.25):
    total_pos = tp + fp
    if total_pos > 0: precision = tp / float(total_pos)
    else: precision = 0
        
    recall = tp / float(tp + fn)
    if precision > 0 and recall > 0:
        score = (1 + beta**2) * (precision * recall) / ((beta**2) * precision + recall)
    else:
        score = 0.0 
    return score, precision, recall 
        
def eval_all_thresholds(times, bids, offers, target_sign, target_probs, actual, ccy):
    best_score = -100
    best_precision = 0
    best_recall = 0
    
    best_thresh = 0 
    best_pred = (target_probs >= 0).astype('int')
    
    precisions = []
    recalls = []
    f_scores = []
    
    thresholds = .2 + np.arange(80) / 100.0
    
    print "Evaluating threshold"
    for t in thresholds:
        pred = target_sign * (target_probs >= t).astype('int') 
        
        # compute confusion matrix with masks instead of 
        tp, fp, tn, fn = confusion(pred, actual) 
        score, precision, recall = f_score(tp, fp, tn, fn)
        
        print "Threshold:", t, "precision =", precision, "recall =", recall, "f_score =", score 
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(score)
        
        if score > best_score: 
            best_score = score 
            best_precision = precision 
            best_recall = recall 
            best_thresh = t
            best_pred = pred
            
    detailed_result = eval_prediction(times, bids, offers, best_pred, actual, ccy)
    return {
        'best_thresh': best_thresh, 
        'best_precision': best_precision, 
        'best_recall': best_recall, 
        'best_score': best_score, 
        'ppt': detailed_result['ppt'], 
        'ntrades': detailed_result['ntrades'], 
        'all_thresholds': list(thresholds), 
        'all_precisions': precisions, 
        'all_recalls': recalls,
    }
