import math 
import numpy as np
import analysis
import scipy 

def aggressive_profit(data, max_hold_frames = 100, num_profitable_frames = 2, target_prct=0.0001, start_idx=None, end_idx=None):
    ts = data['t/100ms'][start_idx:end_idx]
    bids = data['bid/100ms'][start_idx:end_idx]
    offers = data['offer/100ms'][start_idx:end_idx]
    n = len(ts) 
    signal = np.zeros(n)
    for (idx, start_offer) in enumerate(offers):
        if idx < n - max_hold_frames - 1:
            bid_window = bids[idx+1:idx+max_hold_frames+1]
            target_up = (1 + target_prct) * start_offer
            profit_up =  np.sum(bid_window >= target_up) >= num_profitable_frames 
            
            start_bid = bids[idx]
            target_down = (1 - target_prct) * start_bid
            offer_window = offers[idx+1:idx+max_hold_frames + 1]
            profit_down = np.sum(offer_window <= target_down) >= num_profitable_frames
            
            if profit_up and not profit_down:
                signal[idx] = 1
            elif profit_down and not profit_up:
                signal[idx] = -1
    return signal 


def bid_offer_cross(data, start_idx = None, end_idx = None):
    return aggressive_profit(data, target_prct = 0.0, start_idx = start_idx, end_idx = end_idx)


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

def prct_future_midprice_change(data, start_idx = None, end_idx = None):
    midprice = data['midprice'][start_idx:end_idx]
    change = future_change(midprice)
    return change / midprice 


def delta(ys):
    return ys[1:] - ys[:-1]

def prct_next_tick_midprice_change(data, start_idx = None, end_idx = None): 
    midprice = data['midprice'][start_idx:end_idx]
    change = delta(midprice)
    return np.concatenate([change, [0]]) / midprice
    

def prct_curr_tick_midprice_change(data, start_idx = None, end_idx = None): 
    midprice = data['midprice'][start_idx:end_idx]
    change = delta(midprice)
    return np.concatenate([[0], change]) / midprice
