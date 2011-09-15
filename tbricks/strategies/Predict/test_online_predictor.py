import numpy as np 

import online_features 
import online_predictor

feature_exprs = ['(bid/mean/5s + offer/mean/5s) % 2', '(bid/mean/5s - bid/mean/50s) % bid/std/50s'] 
class DumbModel():
    def predict(self, x):
        y = np.sum(x)
        if abs(y) < 0.5: return 0
        else: return np.sign(y)
class DumbEncoder():
    def encode(self, x):
        n = len(x)
        y = np.zeros(2*n)
        y[0:n] = x
        y[n:] = x ** 2
        return y 

def mk_predictor(): 
    dumb_model = DumbModel() 
    dumb_encoder = DumbEncoder() 
    window_sizes={'5s': 50, '50s': 500}
    raw_features={'bid': online_features.best_bid, 'offer': online_features.best_offer}
    return online_predictor.OnlinePredictor(
            feature_exprs, 
            window_sizes=window_sizes, 
            raw_features=raw_features, 
            model=dumb_model, 
            encoder=dumb_encoder,
            min_frames_before_prediction = 2)
    
def test_predictor(): 
    p = mk_predictor()

    bids = np.array([1.5, 1, 0.5])
    bid_sizes = np.array([10,20,30])
    offers = np.array([2.0, 2.5, 3.0])
    offer_sizes = np.array([10, 20, 30])

    p.tick(1, bids, bid_sizes, offers, offer_sizes)
    p.tick(20, 1+bids, bid_sizes, 1+offers, offer_sizes)
    p.tick(30, 2+bids, bid_sizes, 2+offers, offer_sizes)
    p.aggregate_frame() 
    # ramp-up period-- need at least two frames before we can estimate var/std
    assert p.raw_prediction() == 0 
    
    p.tick(110, 3+bids, bid_sizes, 3+offers, offer_sizes)
    p.aggregate_frame(200)
    assert p.raw_prediction() == 1 
