import os, gc, zlib, bisect 
import math, scipy, scipy.stats, scipy.signal
import numpy as np
import cloud
import h5py 
import cPickle

import progressbar
import buildBook 
import aggregators


def rolling_window(a, window):
    """Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>"""
    if window < 1:
       raise ValueError, "`window` must be at least 1."
    if window > a.shape[-1]:
       raise ValueError, "`window` is too long."
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

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
                    timeseries[i] = fn(prevBook, book)
                    prevBook = book 
            else:
                for (i, book) in enumerate(validBooks):
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

        
def features_from_lines(lines, feature_fns, feature_uses_prev_orderbook, debug=False, max_books=None, show_progress=False, output=True):
    books = buildBook.books_from_lines(lines, debug, end=max_books, drop_out_of_order=True)
    if max_books: books = books[:max_books]
    return features_from_books(books, feature_fns, feature_uses_prev_orderbook, show_progress=show_progress, output=output)

def aggregate_1ms_frames(features, frame_reducers, output=True): 
    
    times = features['t']
    unique_times = np.unique(times)
    
    unique_times.sort()
    num_unique_times = len(unique_times)
    if output: print "Found",  len(times), "timestamps, of which", num_unique_times, "are unique"
    
    if output: print "Computing 1ms frame indices..." 
    window_starts, window_ends = make_frame_indices(times, unique_times)
    t_diff = np.concatenate([[0], np.diff(unique_times)])
    frames_1ms = {'t': unique_times, 'message_count': window_ends - window_starts , 'time_since_last_message': t_diff}
    
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
    
def aggregate_100ms_worker(feature_1ms, feature_name, start_indices, end_indices, milliseconds, frame_times, empty_frames):
    n = len(frame_times)
    
    feature_100ms = np.zeros(n) 
    for (i, end_t) in enumerate(frame_times): 
        # empty frames get same value as previous frame 
        if empty_frames[i]:
            feature_100ms[i] = feature_100ms[i-1]
        else:
            start_idx = start_indices[i] 
            start_t = end_t - 100 
            end_idx = end_indices[i] 
                
            time_slice = milliseconds[start_idx:end_idx]
            time_slice_from_zero = time_slice - start_t 
                
            if (time_slice_from_zero[0] == 1) or (i == 0):
                # weights are time between frame arrivals
                weights = np.diff(np.concatenate( [time_slice_from_zero, [101]] )) 
                slice_1ms = feature_1ms[start_idx:end_idx]
            else:
                # if there's no frame on the first 1ms, then we need to 
                # reach back to the last 1ms in the previous 100ms 
                weights = np.diff(np.concatenate( [[1], time_slice_from_zero, [101]] ))
                slice_1ms = feature_1ms[start_idx-1:end_idx] 
            feature_100ms[i] = np.dot(weights, slice_1ms) / np.sum(weights)
    return {'feature_100ms': feature_100ms, 'name': feature_name}
    
def aggregate_100ms_frames(frames_1ms, output=True): 
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
    last_nonempty_ms = np.zeros(n) 
    for (i, end_t) in enumerate(frame_times):
        start_t = end_t - 100 
        # start_indices exclude time (t - 100)
        start_idx = bisect.bisect_right(milliseconds, start_t)
        if milliseconds[start_idx] > end_t:
            empty_frames[i] = True 
            last_nonempty_ms = milliseconds[start_idx-1] 
        # start indices exclude time (t - 100)
        start_indices[i] = start_idx
        # end indices include time t
        end_idx = bisect.bisect_right(milliseconds, end_t, start_idx) 
        end_indices[i] = end_idx
    
    
    if output: print "Launching jobs to aggregate 100ms frames..."
    features_100ms = {'t': frame_times, 'null_100ms_frame': empty_frames}
    jobids = []  
    for fName, vec_1ms in frames_1ms.items(): 
        # time, and counts get compute separately from normal features 
        if fName != 't' and fName != 'time_since_last_message' and fName != 'message_count': 
            jobid = cloud.mp.call(aggregate_100ms_worker, vec_1ms, fName, start_indices, end_indices, milliseconds, frame_times, empty_frames, _fast_serialization=2)
            jobids.append(jobid)
    
    
    print "Computing frame and message counts..."         
    # compute the time since the last message for all frames, even null ones 
    # this is done by looking at the millisecond timestamp of the message before
    # the frame's end. for null frames this will reach back to a time > 100ms.
    # Also, compute the total message count over the frame, which 
    # is simply summed over the individual 1ms counts 
    time_since_last_message = np.zeros(n)
    message_counts_1ms = frames_1ms['message_count']
    message_counts = np.zeros(n)
    small_frame_counts = np.zeros(n)
    for i in xrange(n):
        start_idx = start_indices[i]
        end_idx = end_indices[i] 
        time_since_last_message[i] = frame_times[i] - milliseconds[end_indices[i] - 1]
        message_counts[i] = np.sum(message_counts_1ms[start_idx:end_idx])
        small_frame_counts[i] = end_idx - start_idx 
        
    features_100ms['time_since_last_message'] = time_since_last_message 
    features_100ms['message_count'] = message_counts 
    features_100ms['1ms_frame_count'] = small_frame_counts 
    
    if output: print "Collecting 100ms aggregate results..." 
    if output: progress = progressbar.ProgressBar(len(jobids)).start()
    gc.disable()
    for (counter, result) in enumerate(cloud.mp.iresult(jobids)):
        data = result['feature_100ms']
        name = result['name'] 
        features_100ms[name] = data
        if output: progress.update(counter)
    gc.enable() 
    cloud.mp.delete(jobids)
    if output: progress.finish()
    if output: print 
    return features_100ms
    
def aggregate_window_worker(work):
    fName = work['feature_name']
    rName = work['reducer_name']
    reducer = work['reducer']
    scale = work['scale']
    # stride used as an optimization for min/max to avoid redundant computation
    
    scale_name = work['scale_name']
    frames = work['frames']
    n = len(frames)
    uses_time = work['reducer_uses_time']
    if uses_time: t = work['t']
    if 'time_stride' in work: index_stride = work['time_stride']  / 100 
    else: index_stride = 1
    
    nframes_in_window = scale / 100 
    window_start_indices = np.maximum(0, np.arange(n) - nframes_in_window)
    
    if reducer == np.mean:
        aggregated = aggregators.quick_windowed_means(frames, window_start_indices)
    elif reducer == np.max:
        aggregated = aggregators.quick_windowed_max(frames, window_start_indices, index_stride)
    elif reducer == np.min:
        aggregated = aggregators.quick_windowed_min(frames, window_start_indices, index_stride)
    else:
        aggregated = np.zeros(n)
        if uses_time: 
            for (i, start_idx) in enumerate(window_start_indices):
                end_idx = i+1
                curr_slice = frames[start_idx:end_idx] 
                time_slice = t[start_idx:end_idx]
                aggregated[i] = reducer(time_slice, curr_slice)
        else:
            for (i, start_idx) in enumerate(window_start_indices):
                end_idx = i+1
                curr_slice = frames[start_idx:end_idx] 
                aggregated[i] = reducer(curr_slice)
    result = { 
        'feature_name': fName, 
        'reducer_name': rName, 
        'scale_name': scale_name, 
        'aggregated': aggregated, 
    }
    return result 

#gzip, lzf, or None
compression = 'lzf' 

def add_col(hdf, name, data):
    #print "Adding column", name, " shape = ", data.shape
    parts = name.split('/')
    dirpath = '/'.join(parts[:-1])
    if len(dirpath) > 0 and dirpath not in hdf:
        #print "Creating HDF path", dirpath
        hdf.create_group(dirpath)
    
    hdf.create_dataset(name, data=data, dtype=data.dtype, compression=compression, chunks=True)

default_timescales = [ ("5s", 1000), ("50s", 10000)]
class FeaturePipeline:
    
    def __init__(self, timescales = default_timescales):
        self.feature_fns = {} 
        self.feature_uses_prev_orderbook = {} 
        self.timescales = timescales
        # maps feature name to list of (name,fn) pairs
        self.feature_uses_reducers = {
            'time_since_last_message': True, 
            'message_count': True, 
            '1ms_frame_count': True, 
            'null_100ms_frame': True, 
        }
        #maps feature name to single (name,fn) pair
        self.frame_reducers_1ms = {}
        #maps reducer name to fn 
        self.reducers = {}
        self.reducer_uses_time = {} 
        
            

    def add_feature(self, name, fn, 
            use_prev_orderbook=False, 
            use_window_reducers = True, 
            frame_reducer_1ms=aggregators.last):
        self.feature_fns[name] = fn
        self.frame_reducers_1ms[name] = frame_reducer_1ms
        self.feature_uses_prev_orderbook[name] = use_prev_orderbook
        self.feature_uses_reducers[name] = use_window_reducers 
            
    def add_reducer(self, name, fn, uses_time=False):
        self.reducers[name] = fn
        self.reducer_uses_time[name] = uses_time
    
    def compute_raw_features(self, inputFile, max_books=None):
        if max_books:
            print "Parsing order books and generating raw feature vectors..."
            return features_from_lines(inputFile, self.feature_fns, self.feature_uses_prev_orderbook, max_books=max_books, show_progress=True)
        else: 
            jobids = []
            worklist = []
            print "Launching jobs to parse order books and generate raw features..." 
            def mk_features(lines):
                return features_from_lines(lines, self.feature_fns, self.feature_uses_prev_orderbook, max_books=max_books, show_progress=False)
                
            i = 0
            for linegroup in buildBook.line_group_generator(inputFile):
                i += 1
                label = 'Orderbook input group ' + str(i)
                jobid = cloud.mp.call(mk_features, linegroup, _fast_serialization=2)#, _high_cpu=True, _high_mem=True, _label=label)
                jobids.append(jobid)
            
            print "Collecting raw features..."
            progress = progressbar.ProgressBar(len(jobids)).start()
            ncompleted = 0 
            # collect results into an HDF file
            # where concat is more efficient due to chunked storage
            # and we don't risk running out of memory 
            import tempfile 
            temp_handle, temp_name = tempfile.mkstemp()
            if os.path.exists(temp_name):
                os.remove(temp_name)
            temp = h5py.File(temp_name, 'w')
            for result in cloud.mp.iresult(jobids):
                for name, vec in result.items():
                    if name in  temp:
                        curr = temp[name] 
                        oldsize = len(curr)
                        newsize = oldsize+len(vec)
                        curr.resize((newsize,))
                        curr[oldsize:newsize] = vec
                    else: 
                        temp.create_dataset(name, data=vec, maxshape=None, chunks=True)
                ncompleted += 1
                progress.update(ncompleted)
            cloud.mp.delete(jobids)
            progress.finish() 
            features = {}
            print "Reallocating feature vectors into contiguous memory..." 
            for name, dataset in temp.items(): 
                # extract a contiguous numpy vector, convert names from unicode to ascii
                features[str(name)] = dataset[...]
            temp.close()
            os.remove(temp_name) 
            return features 
        
        
    def aggregate_windows(self, raw_features, hdf):
        t = raw_features['t']
        n = len(t)
        
        message_counts_per_big_frame = raw_features['message_count']
        msg_count_cumulative_sums = np.cumsum(message_counts_per_big_frame) 
        
        small_frame_counts_per_big_frame = raw_features['1ms_frame_count']
        frame_count_cumulative_sums = np.cumsum(small_frame_counts_per_big_frame)
        
        null_100ms_frames = raw_features['null_100ms_frame']
        null_cumulative_sums = np.cumsum(null_100ms_frames)
        
        
        binary_reducers = [("mean", np.mean)]
        last_time_scale = 100 
        # keep these for min and max 
        minmax_results = {}
        gc.disable() 
        for scale_name, scale in self.timescales:
            print "Generating jobs for", scale_name, "window aggregation..."
            worklist = []
            for fName, frames in raw_features.items():
                if fName != 't' and self.feature_uses_reducers[fName]:
                    if frames.dtype == 'bool': reducers = binary_reducers
                    else: reducers = self.reducers.items()
                    for rName, reducer in reducers:
                        uses_time = self.reducer_uses_time[rName]
                        work = { 
                            'reducer': reducer,
                            'feature_name': fName, 
                            'reducer_name': rName, 
                            'scale': scale, 
                            'scale_name': scale_name, 
                            'reducer_uses_time': uses_time, 
                        }
                        
                        fr_name = fName + '/' + rName
                        if (reducer == np.min or reducer == np.max) and fr_name in minmax_results:
                            work['frames'] = minmax_results[fr_name] 
                            work['time_stride'] = last_time_scale 
                        else: 
                            work['frames'] = frames
                        if uses_time: work['t'] = t
                        worklist.append(work)
            jobids = cloud.mp.map(aggregate_window_worker, worklist, _fast_serialization=2)# _label='aggregate_windows', _high_cpu=True, _high_mem=True)
            
            #compute counts while the workers do their thing 
            print "Computing message and frame counts..." 
            window_message_counts = np.zeros(n)
            window_small_frame_counts = np.zeros(n)
            window_big_frame_counts = np.zeros(n)
            window_len = scale / 100 
            for i in xrange(n):
                start_idx = max(i - window_len, 0)
                
                if start_idx > 0: 
                    prev_idx = start_idx - 1
                    msg_count_cumsum_start = msg_count_cumulative_sums[prev_idx]
                    small_frame_count_cumsum_start = frame_count_cumulative_sums[prev_idx]
                    null_cumsum_start = null_cumulative_sums[prev_idx]
                else: 
                    msg_count_cumsum_start = 0 
                    null_cumsum_start = 0
                    small_frame_count_cumsum_start = 0 
                window_message_counts[i] = msg_count_cumulative_sums[i] - msg_count_cumsum_start
                window_small_frame_counts[i] = frame_count_cumulative_sums[i] - small_frame_count_cumsum_start 
                null_sum = null_cumulative_sums[i] - null_cumsum_start 
                window_big_frame_counts[i] = window_len - null_sum 
            add_col(hdf, "message_count/" + scale_name, window_message_counts)
            add_col(hdf, "1ms_frame_count/" + scale_name, window_small_frame_counts)
            add_col(hdf, '100ms_frame_count/' + scale_name, window_big_frame_counts)
        
            
            print "Collecting", len(jobids), "window results for", scale_name, "..." 
            progress = progressbar.ProgressBar(len(jobids)).start()
            for (ncompleted, result) in enumerate(cloud.mp.iresult(jobids)):
                fName = result['feature_name']
                rName = result['reducer_name']
                scale_name = result['scale_name']
                aggregated = result['aggregated']
                fr = fName + '/' + rName
                col_name = fr + '/' + scale_name
                add_col(hdf, col_name, aggregated)
                # as an optimization, min and max recycle their previous outputs
                if rName == 'min' or rName == 'max': minmax_results[fr] = aggregated 
                progress.update(ncompleted)
            cloud.mp.delete(jobids)
            progress.finish()
            last_time_scale = scale
            print 
        gc.enable() 
    
    def run(self, inputFile, outputFilename, max_books = None):
        hdf = h5py.File(outputFilename,'w')
        hdf.attrs['timescales'] =  [pair[0] for pair in self.timescales]
        hdf.attrs['features'] = self.feature_fns.keys()
        hdf.attrs['reducers'] = self.reducers.keys()
        raw_features = self.compute_raw_features(inputFile, max_books)
        frames_1ms = aggregate_1ms_frames(raw_features, self.frame_reducers_1ms)
        del raw_features 
        for name, vec in frames_1ms.items(): add_col(hdf, name + "/1ms", vec)
        
        frames_100ms = aggregate_100ms_frames(frames_1ms)
        del frames_1ms 
        for name, vec in frames_100ms.items(): add_col(hdf, name + "/100ms", vec)
        self.aggregate_windows(frames_100ms, hdf)
        
        # if program quits before this flag is added, ok to overwrite 
        # file in the future 
        hdf.attrs['finished'] = True 
        hdf.close()
    
