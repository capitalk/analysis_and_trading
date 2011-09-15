import aggregators 
import numpy as np 
import unittest 


class TestAggregators(unittest.TestCase):
    frames = np.array([1,1,4,2,1,2])
    start_indices = np.array([0,0,0,0,1,1])
    expected_means = np.array([1,1,2,2,2,2])
    expected_maxes = np.array([1,1,4,4,4,4])
    expected_mins = np.array([ 1,1,1,1,1,1])
        
    def test_quick_windowed_means(self):
        computed_means = aggregators.quick_windowed_means(self.frames, self.start_indices)
        self.assertTrue(np.all(self.expected_means ==computed_means))
    
    def test_quick_windowed_max(self):
        
        computed_max = aggregators.quick_windowed_max(self.frames, self.start_indices)
        print computed_max
        self.assertTrue(np.all(self.expected_maxes ==computed_max))
    
    def test_quick_windowed_min(self): 
        computed_mins = aggregators.quick_windowed_min(self.frames, self.start_indices)
        print computed_mins
        self.assertTrue(np.all(self.expected_mins ==computed_mins))
    
        
    def test_index_weighted_sum(self):
        x = np.array([2,0,2,4])
        # expected result = (2*1 + 0*2 + 2*3 + 4*4)/10 = 2.4
        result = aggregators.index_weighted_sum(x)
        self.assertEqual(result , 2.4)
        
    def test_ols(self):
        t = np.array([1,2,3,4,5,6])
        actual_slope = 2
        y = actual_slope * t + 1
        computed_slope = aggregators.ols(t, y)
        self.assertEqual(actual_slope, computed_slope)

    def test_mean_cross_rate(self): 
        result = aggregators.mean_crossing_rate(self.frames)
        # mean of frames is 1.8333, so expecting 3 crossings of 6 frames
        self.assertApproxEqual(result, 0.5)
        
