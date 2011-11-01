import math 
import numpy as np
import analysis
import bisect 
import scipy 


def aggressive_profit(data, max_hold_frames = 80, num_profitable_frames = 2, target_prct=0.0001, start=None, end=None):
    ts = data['t/100ms'][start:end]
    bids = data['bid/100ms'][start:end]
    offers = data['offer/100ms'][start:end]
    n = len(ts) 
    signal = np.zeros(n)
    for (idx, start_offer) in enumerate(offers):
        if idx < n - max_hold_frames - 1:
            bid_window = bids[idx+1:idx+max_hold_frames+1]
            target_up = (1 + target_prct) * start_offer
            profit_up =  np.sum(bid_window >= target_up) > num_profitable_frames 
            
            start_bid = bids[idx]
            target_down = (1 - target_prct) * start_bid
            offer_window = offers[idx+1:idx+max_hold_frames + 1]
            profit_down = np.sum(offer_window <= target_down) > num_profitable_frames
            
            if profit_up and not profit_down:
                signal[idx] = 1
            elif profit_down and not profit_up:
                signal[idx] = -1
    return signal 


def bid_offer_cross(data):
    return aggressive_profit(data, target_prct = 0.0)

def future_change(ys, short_horizon = 3, long_horizon=50):
    n = len(ys)
    if n <= long_horizon:
        print "Signal shorter than long_horizon"
        return np.zeros(n)
    num_nonzero = n - long_horizon
    signal = np.zeros(num_nonzero)
    for i in xrange(num_nonzero):
        i_short = i + short_horizon 
        curr = np.mean(ys[i:i_short])
        future = np.mean(ys[i_short : i + long_horizon])
        future_delta = future - curr 
        signal[i] = future_delta 
    return np.concatenate([signal, np.zeros(long_horizon)])


def future_window_statistic(ts, ys, fn, window_size = 15000):
    n = len(ts)
    results = np.zeros(n)
    for idx in xrange(n):
        t = ts[idx] 
        future_time_limit = t + window_size
        last_possible_index = min(idx + window_size, n)
        end_idx = bisect.bisect_right(ts, future_time_limit, idx, last_possible_index)
        time_slice = ts[idx:end_idx] 
        window = ys[idx:end_idx]
        curr_result = fn(time_slice, window)
        results[idx] = curr_result 
    return results

def future_log_ratio(d, window_size = 15000):
    ts = d['t']
    ys = d['midprice/100ms']
    def fn(ts, window):
        return np.log(np.mean(window) / window[0])
    return future_window_statistic(ts, ys, fn, window_size)

def future_log_ratio_exceeds_thresh(d, window_size = 15000, thresh=0.0001):
    lr = future_log_ratio(d, window_size)
    return np.sign(lr) * (np.abs(lr) > 0.0001)
    
def percent_bid_above_offer(d,  window_size = 15000):
    offers = d['offer/100ms']
    bids = d['bid/100ms']
    ts = d['t']
    n = len(ts)
    results = np.zeros(n)    
    for idx in xrange(n):
        t = ts[idx]
        future_time_limit = t + window_size
        last_possible_index = min(idx + window_size, n)
        end_idx = bisect.bisect_right(ts, future_time_limit, idx, last_possible_index)
        bid_window = bids[idx:end_idx]
        curr_offer = offers[idx] 
        results[idx] = np.sum(bid_window > curr_offer) / float(len(bid_window))
    return results



def percent_offer_below_bid(d,  window_size = 15000):
    offers = d['offer/100ms']
    bids = d['bid/100ms']
    ts = d['t']
    n = len(ts)
    results = np.zeros(n)    
    for idx in xrange(n):
        t = ts[idx]
        future_time_limit = t + window_size
        last_possible_index = min(idx + window_size, n)
        end_idx = bisect.bisect_right(ts, future_time_limit, idx, last_possible_index)
        offer_window = bids[idx:end_idx]
        curr_bid = offers[idx] 
        results[idx] = np.sum(offer_window < curr_bid) / float(len(offer_window))
    return results


def percent_above_midprice(d,  window_size = 15000):
    ys = d['midprice/mean/100ms']
    ts = d['t']    
    def fn(time_slice, window):
        return np.sum(window > window[0]) / float(len(window))
    return future_window_statistic(ts, ys, fn, window_size)

def percent_above_midprice_plus_spread(d,  window_size = 15000):
    ys = d['midprice/100ms']
    spreads = d['spread/mean/60s']
    ts = d['t']
    n = len(ys)
    results = np.zeros(n)    
    for idx in xrange(n):
        t = ts[idx]
        future_time_limit = t + window_size
        last_possible_index = min(idx + window_size, n)
        end_idx = bisect.bisect_right(ts, future_time_limit, idx, last_possible_index)
        window = ys[idx:end_idx]
        results[idx] = np.sum(window > (window[0] + spreads[idx]))/ float(len(window))
    return results

def percent_below_midprice(d,  window_size = 15000):
    ys = d['midprice/100ms']
    ts = d['t/100ms']
    
    def fn(time_slice, window):
        return np.sum(window < window[0]) / float(len(window))
    return future_window_statistic(ts, ys, fn, window_size)


def percent_below_midprice_minus_spread(d,  window_size = 15000):
    ys = d['midprice/100ms']
    ts = d['t']
    spreads = d['spread/mean/60s']
    n = len(ys)
    results = np.zeros(n)
    for idx in xrange(n):
        t = ts[idx]
        future_time_limit = t + window_size
        last_possible_index = min(idx + window_size, n)
        end_idx = bisect.bisect_right(ts, future_time_limit, idx, last_possible_index)
        window = ys[idx:end_idx]
        results[idx] = np.sum(window < (window[0] - spreads[idx]))/ float(len(window))
    return results

    
def future_slopes(ts, ys, window_size = 15000):
    def slope(time_slice, window):
        if len(time_slice) < 3: 
            return 0.0
        else: 
            normalized_time_slice = (time_slice - time_slice[0]) / 1000.0
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(normalized_time_slice, window)
            return slope 
            
    return future_window_statistic(ts, ys, slope, window_size)
    
def future_movements_scaled_by_spread(ts, ys, spreads, window_size = 15000):
    def movement(ts, ys):
        if len(ys) < 5: 
            return 0.0
        else:
            maxval = np.max(ys)
            minval = np.min(ys)
            y = ys[0] 
            increase = maxval - y 
            abs_decrease = y - minval 
            if increase > abs_decrease: return increase 
            else: return -abs_decrease 
    changes = future_window_statistic(ts, ys, movement, window_size)
    movements = changes / spreads 
    return movements 

def future_midprice_movements_scaled_by_spread(d, window_size = 15000):
    ts = d['t/100ms']
    ys = d['midprice/100ms']
    spreads = d['spread/mean/60s']
    return future_movements_scaled_by_spread(ts, ys, spreads, window_size) 


def net_movements(ts, ys, window_size = 15000):
    def fn(ts, ys):
        if len(ys) < 5: 
            return 0.0
        else:
            top = np.max(ys)
            bottom = np.min(ys)
            y = ys[0] 
            return (top + bottom)/2 -  y
    return future_window_statistic(ts, ys, fn, window_size)

def net_midprice_movements(d, window_size = 15000): 
    ts = d['t']
    ys = d['midprice/mean/5s']
    return net_movements(ts, ys, window_size)
    
def future_bid_slopes(dataset, window_size = 15000):
    bids = dataset['bid/100ms']
    ts = dataset['t']
    return future_slopes(ts, bids, window_size)

    
def future_midprice_slopes(dataset, window_size = 15000):
    ts = dataset['t']
    ys = dataset['midprice/100ms']
    return future_slopes(ts, ys, window_size)

    
def future_offer_slopes(dataset, window_size =15000):
    ts = dataset['t']
    ys = dataset['offer/100ms']
    return future_slopes(ts, ys, window_size)


def aggressive_profit(data, max_hold_frames = 80, num_profitable_frames = 2, target_prct=0.0001, start=None, end=None):
    
    ts = data['t/100ms'][start:end]
    bids = data['bid/100ms'][start:end]
    offers = data['offer/100ms'][start:end]
    n = len(ts) 
    signal = np.zeros(n)
    for (idx, start_offer) in enumerate(offers):
        if idx < n - max_hold_frames - 1:
            bid_window = bids[idx+1:idx+max_hold_frames+1]
            target_up = (1 + target_prct) * start_offer
            profit_up =  np.sum(bid_window >= target_up) > num_profitable_frames 
            
            start_bid = bids[idx]
            target_down = (1 - target_prct) * start_bid
            offer_window = offers[idx+1:idx+max_hold_frames + 1]
            profit_down = np.sum(offer_window <= target_down) > num_profitable_frames
            
            if profit_up and not profit_down:
                signal[idx] = 1
            elif profit_down and not profit_up:
                signal[idx] = -1
    return signal 


def bid_exceeds_offer(data, max_hold_frames=80, num_profitable_frames = 1, target_prct=.0002, start=None, end=None):
    target_prct = 1 + target_prct
    ts = data['t/100ms'][start:end]
    bids = data['bid/100ms'][start:end]
    offers = data['offer/100ms'][start:end]
    n = len(ts)
    signal = np.zeros(n)
    for (idx, start_offer) in enumerate(offers):
        if idx < n - max_hold_frames - 1:
            window = bids[idx+1:idx+max_hold_frames+1]
            target_price = target_prct * start_offer
            signal[idx] = np.sum(window > target_price) > num_profitable_frames 
    return signal 

# wait 1.5 seconds, check whether following 250ms are profitable
def min_bid_exceeds_offer_single_timeframe(data, wait_time=1500, hold_time=250, target_prct=1.0001):
    ts = data['t']
    bids = data['bid/100ms']
    offers = data['offer/100ms']
    
    signal = np.zeros(len(ts))
    max_per_millisecond = analysis.density_per_millisecond(hold_time)
    last_time = wait_time+hold_time
    find_index = analysis.find_future_index 
    for idx in xrange(len(ts)):
        t = ts[idx]
        start_offer = offers[idx]
        start_hold_idx = find_index(ts, idx, t+wait_time, max_per_millisecond=max_per_millisecond)
        # prefilter 
        if bids[start_hold_idx-1] > start_offer:
            target_price = target_prct * start_offer
            end_hold_idx = find_index(ts, idx, t+last_time, max_per_millisecond=max_per_millisecond)
            future_bids = bids[start_hold_idx:end_hold_idx]
            if len(future_bids) > 0 and np.min(future_bids) > target_price: 
                signal[idx] = 1
    return signal 
        


    
# current offer = offers averaged over network delay window
# signal when future bid price exceeds current offer without first
# dropping more than specified percentage
def bid_exceeds_offer_before_drop(data, max_drop = .00001, min_profit=0.00005, max_time = 10000, network_delay=30):
    ts = data.get_col('t')
    
    bids = data.get_col('bid/100ms')
    offers = data.get_col('offer/100ms')
    
    signal = np.zeros(len(ts))
    max_per_millisecond = analysis.density_per_millisecond(max_time-network_delay)
    find_index = analysis.find_future_index
    for idx in xrange(len(ts)):
        t = ts[idx]
        start_bid = bids[idx] 
        
        # average prices over the time window in which a buy message may reach 
        # the server
        find_index(ts, idx, t+network_delay, max_per_millisecond=max_per_millisecond)
        delayed_offers = offers[idx:last_delay_index] 
        avg_delayed_offer = np.mean(delayed_offers) 
        
        last_future_index = find_index(ts, idx, max_time, max_per_millisecond=max_per_millisecond)
        future_bids = bids[last_delay_index:last_future_index]
        
        sale_price_prct = (future_bids - avg_delayed_offer) / avg_delayed_offer
        profit_index = find_first (sale_price_prct >= min_profit)
        if profit_index:
            neg_bid_prct = (future_bids - start_bid) / start_bid 
            quit_index = find_first( neg_bid_prct <=  -max_drop)
            if quit_index is None or quit_index > profit_index: 
                signal[idx] = 1
        
    return signal

