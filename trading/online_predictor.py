import numpy as np 

import online_aggregator
# python is annoying about relative imports 
import sys; 
sys.path.append('../analysis')
import expr_lang 

import online_features 

def mk_agg(raw_features, window_sizes):
    agg = online_aggregator.OnlineAggregator(window_sizes=window_sizes)
    for feature_name, feature_fn in raw_features.items():
        agg.add_feature(feature_name, feature_fn)
    return agg 
        

class OnlinePredictor():
    def __init__(self, 
            feature_exprs, 
            model,
            encoder=None,
            window_sizes={'5s': 50, '50s': 500}, 
            raw_features={'bid': online_features.best_bid, 'offer': online_features.best_offer}, 
            min_frames_before_prediction=2):
        
        self.raw_features = raw_features
        self.window_sizes = window_sizes 
        self.agg = mk_agg(raw_features, window_sizes) 
        
        self.feature_exprs = feature_exprs 
        # a list of expr_lang functions which map a symbol environment to the
        # evaluated feature expression 
        self.compiled_features = [expr_lang.compile_expr(expr) for expr in feature_exprs]
        
        # the set of symbols that need to be present in an environment so that 
        # all the feature expressions can be evaluated 
        self.feature_symbols = expr_lang.symbol_set(feature_exprs)
        self.encoder = encoder 
        self.model = model 
        self.min_frames_before_prediction = min_frames_before_prediction
        self.longest_window = np.max(window_sizes.values())
    
    
    
    def __getstate__(self): 
        d = self.__dict__.copy() 
        # remove features since they are functions which can't be pickled 
        del d['compiled_features']
        # don't pickle aggregator, just recreate it when unpickling
        del d['agg']
        return d 
        
    def __setstate__(self, state):
        self.__dict__ = state  
        # reconstruct feature expr functions, since they weren't pickled 
        self.compiled_features = [expr_lang.compile_expr(expr) for expr in self.feature_exprs]
        # reconstruct the aggregator which wasn't pickled 
        self.agg = mk_agg(state['raw_features'], state['window_sizes'])

    def tick(self, t, bids, bv, offers, ov):
        self.agg.millisecond_update(t, bids, bv, offers, ov)
    
    def aggregate_frame(self, curr_time): 
        self.agg.aggregate_frame(curr_time) 
    
    def get_feature_vector(self):
        env = self.agg.build_env(self.feature_symbols ) 
        n = len(self.feature_exprs)
        vec = np.zeros(n)
        for i, compiled_feature in enumerate(self.compiled_features):
            vec[i] = compiled_feature(env) 
        if self.encoder:
            vec = self.encoder.encode(vec)
        return vec 
    
    def frame_count(self):
        count = len(self.agg.frame_buffers[self.longest_window])
        return count 
        
    def raw_prediction(self):
        if self.frame_count() >= self.min_frames_before_prediction:
            vec = self.get_feature_vector()
            return self.model.predict(vec) 
        else:
            return 0 
        
    def filtered_prediction(self): 
        pass 
        # TODO: keep a voting prediction buffer 
    
