import numpy as np 
import online_features 
import online_aggregator

def approx_eq(x,y):
    return abs(x-y)<0.00001

bids = np.array([1.5, 1, 0.5])
bid_sizes = np.array([10,20,30])
offers = np.array([2.0, 2.5, 3.0])
offer_sizes = np.array([10, 20, 30])

def mk_agg():
    agg = online_aggregator.OnlineAggregator(window_sizes={"1s": 10, "5s": 50}, frame_size = 100)
    agg.add_feature("bid", online_features.best_bid)
    return agg 

def test_millisecond_update():
    agg = mk_agg()
    agg.millisecond_update(0, bids, bid_sizes, offers, offer_sizes)
    agg.millisecond_update(50, 1+bids, bid_sizes, 1+offers, offer_sizes)
    agg.millisecond_update(90, 2+bids, bid_sizes, 2+offers, offer_sizes)
    # bid average = (50 * 1.5 + 40 * 2.5 + 10* 3.5) / 100 == 2.1 
    frame = agg.aggregate_frame(100)
    print frame
    assert abs(frame['bid'] - 2.1) < 0.00001

def test_stats(): 
    agg = mk_agg()
    agg.millisecond_update(0, bids, bid_sizes, offers, offer_sizes)
    agg.millisecond_update(50, 1+bids, bid_sizes, 1+offers, offer_sizes)
    agg.millisecond_update(90, 2+bids, bid_sizes, 2+offers, offer_sizes)
    agg.aggregate_frame(100)
    
    agg.millisecond_update(110, bids, bid_sizes, offers, offer_sizes)
    agg.millisecond_update(150, 2+bids, bid_sizes, 2+offers, offer_sizes)
    agg.millisecond_update(180, 3+bids, bid_sizes, 3+offers, offer_sizes)
    # first bid frame is 2.1
    # second frame = (3.5 * 10 + 40 * 1.5 + 30 * 3.5 + 20 * 4.5) / 100 == 2.9 
    # so mean should be (2.1 + 2.9) / 2 == 2.5
    # variance == (.4^2 + .4^2)/ (n-1) == .32 
    agg.aggregate_frame(200) 
    stats_1s = agg.stats[('bid', 10)]
    stats_5s = agg.stats[('bid', 50)]
    assert approx_eq(stats_1s.mean(), 2.5)
    assert approx_eq(stats_5s.mean(), 2.5)
    assert approx_eq(stats_1s.var(), .32)
    assert approx_eq(stats_5s.var(), .32)
    
    
    
    
