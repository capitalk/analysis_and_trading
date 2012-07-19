import gc, bisect 
import math
import numpy as np
import cloud
import h5py 

import progressbar
import buildBook 
import aggregators


# grouping sparse messages into frames whose time range is determined
# by the variable scale (defaults to 1ms)
# NOTE: ASSUMING THE SECOND ARGUMENT ONLY CONTAINS VALUES FOUND IN THE FIRST
def make_frame_indices(times, unique_times, scale=1):
    num_unique_times = len(unique_times)
    num_raw_times = len(times)
    window_start_indices = np.zeros(num_unique_times)
    search_start = 0 
    for i, t in enumerate(unique_times):
        search_t = t - scale + 1
        first_index = bisect.bisect_left(times, search_t, search_start, num_raw_times)
        window_start_indices[i] = first_index 
        search_start = first_index 
    window_end_indices = np.concatenate( [window_start_indices[1:], [num_raw_times]])
    return window_start_indices, window_end_indices  

    

# given a list of books, return a dictionary of feature vectors 
def features_from_books(books, feature_fns, feature_uses_prev_orderbook, show_progress=False, output=True):
    result = {}
    if output: print "Extracting features...."
    
    # scan for orderbook with non-empty bids and offers
    validBooks = [ book for book in books if book.bids and book.offers]
    # generator expression to count all invalid books 
    
    
    numInvalid = len(books) - len(validBooks)
    if numInvalid > 0 and output:
        print "Dropping", numInvalid, "invalid order books"

    prevBook = validBooks[0]
    validBooks = validBooks[1:]
    n = len(validBooks)
    
    nfeatures = len(feature_fns)
    
    if show_progress: progress = progressbar.ProgressBar(nfeatures).start()
    for (featurenum, (name, fn)) in enumerate(feature_fns.items()): 
        if name != 'midprice' and name != 'spread':
            # in the future we should probably track feature types, 
            # but for now assume everything except time is a float 
            if name != 't': timeseries = np.zeros(n)
            else: timeseries = np.zeros(n, dtype='int')
            if feature_uses_prev_orderbook[name]:
                for (i,book) in enumerate(validBooks):
                    book.compute_order_flow_stats()
                    timeseries[i] = fn(prevBook, book)
                    prevBook = book 
            else:
                for (i, book) in enumerate(validBooks):
                    book.compute_order_flow_stats()
                    timeseries[i] = fn(book)
            result[name] = timeseries
        if show_progress: progress.update(featurenum)
    if show_progress: progress.finish()
    if 'spread' in feature_fns:
        result['spread'] = result['offer'] - result['bid'] 
    if 'midprice' in feature_fns:
        result['midprice'] = (result['offer'] + result['bid']) / 2
    if output: print 
    return result   

        
def features_from_file(f, feature_fns, feature_uses_prev_orderbook, debug=False, max_books=None, show_progress=False, output=True):
    header, books = buildBook.read_books_from_file(f, debug, end=max_books)
    if max_books: books = books[:max_books]
    features = features_from_books(books, feature_fns, feature_uses_prev_orderbook, show_progress=show_progress, output=output)
    return header, features 

def aggregate_1ms_frames(features, frame_reducers, output=True): 
    
    times = features['t']
    unique_times = np.unique(times)
    
    unique_times.sort()
    num_unique_times = len(unique_times)
    if output: print "Found",  len(times), "timestamps, of which", num_unique_times, "are unique"
    
    if output: print "Computing 1ms frame indices..." 
    window_starts, window_ends = make_frame_indices(times, unique_times)
    t_diff = np.concatenate([[0], np.diff(unique_times)])
    frames_1ms = {'t': unique_times,  'time_since_last_message': t_diff}
    
    if output: print "Aggregating 1ms frames..." 
    nreducers = len(frame_reducers)
    if output: progress = progressbar.ProgressBar(nreducers).start()
    counter = 0 
    gc.disable()
    for name, fn in frame_reducers.items():
        if name != 't':
            raw = features[name] 
            result = np.zeros(num_unique_times)
            for i in xrange(num_unique_times):
                start_idx = window_starts[i] 
                end_idx = window_ends[i] 
                curr_slice = raw[start_idx:end_idx] 
                result[i] = fn(curr_slice)
            frames_1ms[name] = result
        counter += 1
        if output: progress.update(counter)
    gc.enable()
    if output: progress.finish() 
    if output: print 
    return frames_1ms 
    
    

#gzip, lzf, or None
compression = 'lzf' 

def add_col(hdf, name, data):
    #print "Adding column", name, " shape = ", data.shape
  parts = name.split('/')
  dirpath = '/'.join(parts[:-1])
  if len(dirpath) > 0 and dirpath not in hdf:
    hdf.create_group(dirpath)
  hdf.create_dataset(name, data=data, dtype=data.dtype, compression=compression, chunks=True)

class FeaturePipeline:
    
    def __init__(self):
        self.feature_fns = {} 
        self.feature_uses_prev_orderbook = {} 

        # maps feature name to list of (name,fn) pairs
        self.feature_uses_reducers = {
            'time_since_last_message': True, 
            '1ms_frame_count': True, 
            'null_100ms_frame': True, 
        }
        #maps feature name to single (name,fn) pair
        self.frame_reducers_1ms = {}
        self.sum_100ms_feature = {} 
            

    def add_feature(self, name, fn, 
            use_prev_orderbook=False, 
            use_window_reducers = True, 
            frame_reducer_1ms=aggregators.last, 
            sum_100ms = False):
        self.feature_fns[name] = fn
        self.frame_reducers_1ms[name] = frame_reducer_1ms
        self.feature_uses_prev_orderbook[name] = use_prev_orderbook
        self.feature_uses_reducers[name] = use_window_reducers 
        self.sum_100ms_feature[name] = sum_100ms 
         

    def aggregate_100ms_frames(self, frames_1ms, output=True): 
        milliseconds = frames_1ms['t']
        start_millisecond = milliseconds[0]
        end_millisecond = milliseconds[-1] 
        round_start = int(math.ceil(start_millisecond / 100.0) * 100)
        round_end = int(math.ceil(end_millisecond / 100.0) * 100)
        
        if output: print "Generating 100ms frame indices..." 
        frame_times = np.arange(round_start, round_end+1, 100)    
        n = len(frame_times)
        start_indices = np.zeros(n)
        end_indices = np.zeros(n)
        empty_frames = np.zeros(n, dtype='bool')
        for (i, end_t) in enumerate(frame_times):
            start_t = end_t - 100 
            # start_indices exclude time (t - 100)
            start_idx = bisect.bisect_right(milliseconds, start_t)
            if milliseconds[start_idx] > end_t:
                empty_frames[i] = True 
            # start indices exclude time (t - 100)
            start_indices[i] = start_idx
            # end indices include time t
            end_idx = bisect.bisect_right(milliseconds, end_t, start_idx) 
            end_indices[i] = end_idx
        
        
        features_100ms = {'t': frame_times, 'null_100ms_frame': empty_frames}
        
        print "Aggregating 100ms frames..." 
        
        for fName, vec_1ms in frames_1ms.items(): 
            if output: print "  ", fName 
            # time, and counts get compute separately from normal features 
            if fName != 't' and fName != 'time_since_last_message': 
                if self.sum_100ms_feature[fName]:
                    features_100ms[fName] = aggregators.sum_100ms(vec_1ms, start_indices, end_indices, frame_times)
                else:
                    features_100ms[fName] = aggregators.time_weighted_average_100ms(vec_1ms, start_indices, end_indices, milliseconds, frame_times, empty_frames)
        
        print "Computing time between messages and 1ms frame counts..."         
        # compute the time since the last message for all frames, even null ones 
        # this is done by looking at the millisecond timestamp of the message before
        # the frame's end. for null frames this will reach back to a time > 100ms.
        # Also, compute the total message count over the frame, which 
        # is simply summed over the individual 1ms counts 
        time_since_last_message = np.zeros(n)
        small_frame_count = np.zeros(n)
        for i in xrange(n):
            start_idx = start_indices[i]
            end_idx = end_indices[i] 
            time_since_last_message[i] = frame_times[i] - milliseconds[end_indices[i] - 1]
            small_frame_count[i] = end_idx - start_idx + 1 
        features_100ms['time_since_last_message'] = time_since_last_message 
        features_100ms['1ms_frame_count'] = small_frame_count 

        return features_100ms        
    
    
    def dict_to_hdf(self, d, path, ccy):
      hdf = h5py.File(path, 'w')
      hdf.attrs['features'] = self.feature_fns.keys()
      hdf.attrs['ccy'] = ccy
       
      for name, vec in d.items(): 
        add_col(hdf, name, vec)
      
      # if program quits before this flag is added, ok to overwrite 
      # file in the future
      hdf.attrs['finished'] = True 
      hdf.close()
      
            
    def run(self, input_filename, 
            output_filename_1ms, 
            output_filename_100ms,
            max_books = None):
        
        header, raw_features = features_from_file(input_filename,  
          self.feature_fns, 
          self.feature_uses_prev_orderbook, 
          max_books=max_books, 
          show_progress=True)
        
        frames_1ms = aggregate_1ms_frames(raw_features, self.frame_reducers_1ms)
        del raw_features 
        self.dict_to_hdf(frames_1ms, output_filename_1ms, header['ccy'])
        
        frames_100ms = self.aggregate_100ms_frames(frames_1ms)
        del frames_1ms 
        self.dict_to_hdf(frames_100ms, output_filename_100ms, header['ccy'] )
       
        
