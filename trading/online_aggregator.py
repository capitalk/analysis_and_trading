import collections 
import numpy as np 
import incremental_stats 



class OnlineAggregator:
    # timescales = pairs of names and number of frames 
    def __init__(self, frame_size = 100, window_sizes = {"1s": 10, "5s": 50, "50s": 500, "500s": 5000} ):
        self.raw_feature_times = [] 
        self.raw_feature_buffers = {}
        
        self.feature_fns = {}
        self.stats = {} 
        
        self.frame_size = frame_size
        
        self.num_recent_updates = 0 
        self.window_sizes = window_sizes
        
        self.last_frame = None
        
        self.frame_buffers = {}
        for w in window_sizes.values():
            self.frame_buffers[w] = collections.deque()
            
        self.last_frame = None 
        
        # maps 'feature/stat/window_name' string to function which looks up that 
        # value 
        self.stat_lookups = {} 
    
    def add_feature(self, fName, fn):
        self.feature_fns[fName] = fn
        self.raw_feature_buffers[fName] = [] 
        for wName, w in self.window_sizes.items():
            k = (fName, w)
            stats_obj = incremental_stats.OnlineMeanVar() 
            self.stats[k] = stats_obj 
            # build a dictionary of aggregated feature names mapping to anonymous
            # functions which will look up the value associated with each name 
            for stat_name in ['mean', 'var', 'std']: 
                str_key = fName + '/' + stat_name + '/' + wName 
                self.stat_lookups[str_key] = getattr(stats_obj, stat_name)
            
    
    def millisecond_update(self, time, bid_prices, bid_sizes, offer_prices, offer_sizes):
        #print "time", time 
        #print "bid prices", bid_prices
        #print "bid sizes", bid_sizes
        #print "offer prices", offer_prices
        #print "offer sizes", offer_sizes
        
        self.raw_feature_times.append(time) 
        for name, fn in self.feature_fns.items():
        
            v = fn(bid_prices, bid_sizes, offer_prices, offer_sizes)
            self.raw_feature_buffers[name].append(v)
        self.num_recent_updates += 1
        
    def update_windowed_stats(self, arriving_frame): 
        #print "Called update_windowed_stats with frame:", arriving_frame
        
        for window_name, w in self.window_sizes.items():
            frame_buffer = self.frame_buffers[w] 
            # first shorten the frame buffer to be one less than max size
            while len(frame_buffer) >= w:
                departing_frame = frame_buffer.popleft()
                # update stats for all data in departing frame 
                for feature_name, v in departing_frame.items(): 
                    k = (feature_name, w)
                    self.stats[k].remove(v)
            # now add new data to frame buffer 
            frame_buffer.append(arriving_frame) 
            # update stats for all values in new frame 
            for feature_name, v in arriving_frame.items(): 
                k = (feature_name, w)
                self.stats[k].add(v)
        #print self.frame_buffers 
    def aggregate_frame(self, curr_time):
        #print "Called aggregate_frames at ", curr_time, "num recent updates=", self.num_recent_updates
        if self.num_recent_updates == 0 and self.last_frame is None:
            raise RuntimeError("nothing to aggregate") 
        elif self.num_recent_updates == 0: 
            self.update_windowed_stats(self.last_frame) 
        else: 
            start_time = curr_time - self.frame_size 
            times_vec = np.concatenate([ self.raw_feature_times, [curr_time]])
            if self.raw_feature_times[0] < start_time:
                times_vec[0] = start_time 
            
            weights = np.diff(times_vec)
            weight_sum = np.sum(weights)
            frame = {} 
            new_raw_features = {}
            for name, raw_vec in self.raw_feature_buffers.items():
                dot = np.dot(weights, raw_vec)
                frame[name] = dot / weight_sum 
                new_raw_features[name] = [raw_vec[-1]]
            
            # TODO: change raw_features to raw_feature_buffers 
            self.raw_feature_buffers = new_raw_features 
            self.raw_feature_times = [self.raw_feature_times[-1]]
            self.last_frame = frame 
            self.update_windowed_stats(frame) 
            
        self.num_recent_updates = 0 
        return self.last_frame 
        
    def build_env(self, keys=None):
        env = {}
        if keys is None:
            keys = self.stat_lookups.keys()
        for k in keys:
            fn = self.stat_lookups[k]
            v = fn() 
            
            env[k] = v
        return env 
