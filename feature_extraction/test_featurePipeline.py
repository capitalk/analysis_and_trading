import featurePipeline 
import numpy as np 
import unittest 
from orderBook import * 

def vec_approx_eq(x,y):
    return np.mean(np.abs(x-y)) <= 0.000001
    
def vec_eq(x,y):
    return np.all(x ==y)
    
class TestFeaturePipeline(unittest.TestCase): 
    def test_window_indices_1ms_frames(self):
        ts1 = np.array( [1,1,1,3, 11,11,13])
        ts2 = np.array( [1,3,11,13] )
        start_indices, end_indices = featurePipeline.make_past_window_indices(ts1, ts2)
        expected_start_indices = np.array([0, 3, 4, 6])
        self.assertTrue(vec_eq(start_indices, expected_start_indices))
        expected_end_indices = np.array([3, 4, 6, 7])
        self.assertTrue(vec_eq(expected_end_indices, end_indices))
        
    def test_window_indices_boundaries(self):
        ts1 = [1, 100, 101, 200 ]
        ts2 = [100, 200]
        start_indices, end_indices = featurePipeline.make_past_window_indices(ts1, ts2, scale=100)
        print start_indices 
        expected_start_indices = np.array( [0, 2] )
        self.assertTrue(vec_eq(start_indices, expected_start_indices))
        expected_end_indices = np.array( [2, 4] )
        self.assertTrue(vec_eq(end_indices, expected_end_indices))

    def test_missing_window_indices(self):
        ts1 = [1, 100, 101, 200, 501, 502 ]
        ts2 = [300, 700]
        start_indices, end_indices = featurePipeline.make_past_window_indices(ts1, ts2, scale=100)
        self.assertEqual(start_indices[0], 4)
        self.assertEqual(start_indices[1], 6)
        
        
    def test_window_indices_100ms_frames(self):
        ts_1ms = np.array( [1, 2, 101, 102] )
        ts_100ms = np.array( [100, 200])
        start_indices, end_indices = featurePipeline.make_past_window_indices(ts_1ms, ts_100ms, scale=100)
        print start_indices, end_indices 
        
    def test_aggregate_window_worker(self):
        fName = 'zoggle'
        rName = 'mean'
        reducer = np.mean
        frames = np.array([1, 1, 2, 3])
        sName = '1ms'
        window_starts = np.array([0, 1, 1, 1])
        t = np.array([1, 2, 3, 4])
        work = {
            'feature_name': fName, 'reducer_name': rName, 'reducer': reducer, 
            'window_starts': window_starts, 'scale_name': sName, 
            'frames': frames, 't': t, 'reducer_uses_time': False,
        }
        result = featurePipeline.aggregate_window_worker(work)
        self.assertEqual(result['feature_name'], work['feature_name'])
        self.assertEqual(result['reducer_name'], work['reducer_name'])
        self.assertEqual(result['scale_name'], work['scale_name'])
        agg = result['aggregated']
        expected = np.array([1, 1, 1.5,2])
        self.assertTrue(vec_approx_eq(agg, expected))
        
    def test_features_from_books(self): 
        
        import orderBookConstants as obc 
        bid1 = Order(side=obc.BID, level=0, price=1.41, size=100)
        offer1 = Order(side=obc.OFFER, level=0, price=1.42, size=200)
        ob = OB()
        ob.add_bid(bid1)
        ob.add_offer(offer1)
        
        orderbooks = [ob, ob]
        import features 
        feature_fns = {'midprice': features.midprice, 'offer': features.best_offer, 'bid': features.best_bid}
        feature_uses_prev_orderbook = {'midprice': False, 'offer': False, 'bid':False}
        features = featurePipeline.features_from_books(orderbooks, feature_fns, feature_uses_prev_orderbook)
        
        self.assertIn('bid', features)
        self.assertIn('offer', features)
        self.assertIn('midprice', features)
        print "Best bid", features['bid']
        print "Best offer", features['offer']
        print "Midprice", features['midprice']
        # we expect the first orderbook to be skipped!
        self.assertTrue(vec_approx_eq(np.array([1.41]), features['bid']))
        self.assertTrue(vec_approx_eq(np.array([1.42]), features['offer']))
        self.assertTrue(vec_approx_eq(np.array([1.415]), features['midprice']))

    def test_aggregate_100ms_worker(self): 
        milliseconds = np.array([30,  32, 101, 107, 303, 400])
        midprice = np.array([1.1, 1.1,  1.2, 1.3, 1.4, 1.6])
        frame_times = np.array([100, 200, 300, 400])
        empty_frames = np.array([False, False, True, False])
        start_indices = np.array([0, 2, 4, 4])
        end_indices = np.array([2, 4, 4, 6])
        result = featurePipeline.aggregate_100ms_worker(midprice, 'midprice', start_indices, end_indices, milliseconds, frame_times, empty_frames)
        self.assertIn('name', result)
        self.assertEqual(result['name'], 'midprice')
        expected_100ms_midprice = np.array([1.1, 1.294, 1.294, 1.4])
        self.assertIn('feature_100ms', result)
        self.assertTrue(vec_approx_eq(result['feature_100ms'], expected_100ms_midprice))
        
    def test_aggregate_100ms_frames(self): 
        
        t = np.array([30,  32,   101, 107, 303, 400, 500])
        m = np.array([1.1, 1.1,  1.2, 1.3, 1.4, 1.6, 1.6])
        message_count_1ms = np.array([1,2,1,2,5,1, 2])
        frames_1ms = { 't': t, 'midprice': m, 'message_count': message_count_1ms}
        import cloud
        cloud.config.num_procs = 2
        cloud.config.max_transmit_data = 57000000
        cloud.config.serialize_logging = False
        cloud.config.commit()
        
        frames_100ms = featurePipeline.aggregate_100ms_frames(frames_1ms, cloud = cloud.mp, output=False)
        print "Received 100ms features:", frames_100ms
        expected_100ms_t = np.array([100, 200, 300, 400, 500])
        self.assertIn('t', frames_100ms)
        self.assertTrue(vec_eq(frames_100ms['t'], expected_100ms_t))
        
        self.assertIn('time_since_last_message', frames_100ms)
        time_since_last_message = np.array([68, 93, 193, 0, 0])
        self.assertTrue(vec_eq(frames_100ms['time_since_last_message'], time_since_last_message))
        
        self.assertIn('null_100ms_frame', frames_100ms)
        null_frames = np.array([False, False, True, False, False])
        self.assertTrue(vec_eq(frames_100ms['null_100ms_frame'], null_frames))
        
        self.assertIn('midprice', frames_100ms)
        # first frame midprice = (2/70) * 1.1 + (68/70) * 1.1 == 1.1
        # second frame midprice = (6/100) * 1.2 + (94/100) * 1.3 == 1.294
        # third frame midprice = (2/100) * 1.3 + (97/100) * 1.4 + 1/100 * 1.5 == 1.4
        expected_100ms_midprice = np.array([1.1, 1.294, 1.294, 1.4, 1.6])
        print "Received midprice: " , frames_100ms['midprice'], " Expected:", expected_100ms_midprice 
        self.assertTrue(vec_approx_eq(frames_100ms['midprice'], expected_100ms_midprice))
        
        self.assertIn('message_count', frames_100ms)
        message_count_100ms = np.array([3, 3, 0, 6, 2])
        self.assertTrue (vec_eq (frames_100ms['message_count'], message_count_100ms))
        
