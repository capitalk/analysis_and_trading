import numpy as np
import logging as log
import time


def momentum1(dataset):
    """
    momentum1
    Simple momentum strategy:
        if slope of 1s midprice > slope of 500s midprice then BUY
        if slope of 1s midprice < slope of 500s midprice then SELL
    """
    BUY = +1
    SELL = -1
    
    ts = dataset['t']
    n = len(ts)
    signal = np.zeros(n)

    bid_mean_1s = dataset['bid/mean/1s']
    offer_mean_1s = dataset['offer/mean/1s']
    mid_slope_1s = dataset['midprice/slope/1s']
    mid_slope_500s = dataset['midprice/slope/500s']
    offer_range_100ms = dataset['offer_range/100ms'] 
    bid_range_100ms = dataset['bid_range/100ms'] 
    spread = dataset['spread/100ms'] 
    crossed_books = [ s < 0 for s in spread]
    num_crossed_books = np.count_nonzero(crossed_books)
    
    for idx in xrange(n):
        cur_mid_slope_500s =  mid_slope_500s[idx]  
        cur_mid_slope_1s =  mid_slope_1s[idx]  
        

        if cur_mid_slope_1s > cur_mid_slope_500s:# and cur_mid_slope_1s > 0 and cur_mid_slope_500s > 0:
            signal[idx] = BUY 

        elif cur_mid_slope_1s < cur_mid_slope_500s:# and cur_mid_slope_1s < 0 and cur_mid_slope_500s < 0:
            signal[idx]  = SELL

    
    mean_spread = np.mean(spread)
    mean_range = np.mean(bid_range_100ms) 
    return signal, mean_spread, mean_range

def momentum2(dataset):
    """
    momentum2
    Simple momentum strategy:
        if slope of 1s midprice > slope of 500s midprice then BUY
        if slope of 1s midprice < slope of 500s midprice then SELL
        dont' buy/sell the same price twice
    """


    BUY = +1
    SELL = -1
    
    ts = dataset['t']
    n = len(ts)
    signal = np.zeros(n)

    bid_mean_1s = dataset['bid/mean/1s']
    offer_mean_1s = dataset['offer/mean/1s']
    mid_slope_1s = dataset['midprice/slope/1s']
    mid_slope_500s = dataset['midprice/slope/500s']
    offer_range_100ms = dataset['offer_range/100ms'] 
    bid_range_100ms = dataset['bid_range/100ms'] 
    spread = dataset['spread/100ms'] 
    crossed_books = [ s < 0 for s in spread]
    last_bid = round(bid_mean_1s[0], 5)
    last_offer = round(offer_mean_1s[0], 5)
    
    for idx in xrange(n):
        cur_mid_slope_500s =  mid_slope_500s[idx]  
        cur_mid_slope_1s =  mid_slope_1s[idx]  
        curr_bid = round(bid_mean_1s[idx], 5)
        curr_offer = round(offer_mean_1s[idx], 5)
        
        #print cur_mid_slope_1s, ' ', cur_mid_slope_500s

        if cur_mid_slope_1s > cur_mid_slope_500s and last_offer != curr_offer:# and cur_mid_slope_1s > 0 and cur_mid_slope_500s > 0:
            print "last_offer: ", last_offer, "curr_offer: ", curr_offer
            signal[idx] = BUY 

        elif cur_mid_slope_1s < cur_mid_slope_500s and last_bid != curr_bid:# and cur_mid_slope_1s < 0 and cur_mid_slope_500s < 0:
            print "last_bid: ", last_bid, "curr_bid: ", curr_bid
            signal[idx]  = SELL

        last_bid = curr_bid
        last_offer = curr_offer

    mean_spread = np.mean(spread)
    mean_range = np.mean(bid_range_100ms) 
    return signal, mean_spread, mean_range

def active1(dataset):
    
    position = 0
    position_price = 0
    BUY = +1
    SELL = -1

    ts = dataset['t']

    n = len(ts)
    pnl = np.zeros(n)
    signal = np.zeros(n)

    bid_mean_1s = dataset['bid/mean/1s']
    bid = dataset['bid']
    offer = dataset['offer']
    #offer_mean_1s = dataset['offer/mean/1s']
    #mid_slope_1s = dataset['midprice/slope/1s']
    bid_range = dataset['bid_range/100ms']
    offer_range = dataset['offer_range/100ms']
    spread = dataset['spread/100ms']
    mean_spread = np.mean(spread)
    mean_range = np.mean(bid_range)
    nstep = 5
    i = 0 
    sr = 0
    nsr = 0

    for idx in xrange(n):
        i += 1
        curr_bid = bid[idx]
        curr_offer = offer[idx]
         
        if i == nstep:
            range_pct_change  = (bid_range[idx]-bid_range[idx-nstep])/bid_range[idx-nstep]
            spread_pct_change = (spread[idx]-spread[idx-nstep])/spread[idx-nstep]
            assert i == nstep
            #norm_range = bid_range[idx]/bid_range[idx-nstep]
            #norm_spread = spread[idx]/spread[idx-nstep]
            #assert spread_pct_change != 0
            #spread_adjusted_range = range_pct_change/spread_pct_change 
            #sr += spread_adjusted_range
            #assert sr < np.inf
            #nsr += 1
            #mean_spread_adjusted_range = sr/nsr

            bid_move_pct = (curr_bid-bid[idx-nstep])/bid[idx-nstep]
            offer_move_pct = (curr_offer-offer[idx-nstep])/offer[idx-nstep]

            #print idx, ":", "i(", i, ")", "sr(", sr, ")", "nsr(", nsr, ")", "range_pct_change(", range_pct_change, ")", " spread_pct_change(", spread_pct_change, ")", " bid move pct(", bid_move_pct, ")", " offer move pct(", offer_move_pct, ")"
        
            if range_pct_change > 0 and spread_pct_change > 0:
                    if bid_move_pct < 0:
                        signal[idx] = SELL
                    if bid_move_pct > 0:
                        signal[idx] = BUY

            elif range_pct_change < 0 and spread_pct_change < 0:
                    pass
                
            i = 0
        
        last_bid = curr_bid
        last_offer = curr_offer    

    return signal, mean_spread, mean_range

def to_fit(array):
    V = [[x] for x in array]
    return V

def pct_change(x,y):
    pct_change_x = (x-x[0]/V[0])
    return pct_change_x

