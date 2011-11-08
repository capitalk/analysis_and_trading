import math 
import numpy as np
import analysis
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

def prct_future_midprice_change(data):
    midprice = data['midprice']
    change = future_change(midprice)
    return change / midprice 
