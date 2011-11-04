import analysis
import numpy as np 
import unittest 

class TestAnalysis(unittest.TestCase):                            
    def test_lte(self):
        x = np.array([0,-1,2,3.1,-1.2, -1.6,8.1,6])
        idx = analysis.find_first_lte(x, -1.5)
        self.assertEqual(idx, 5)
        
    def test_gte(self): 
        x = np.array([0,-1,2,3.1,-1.2, -1.6,8.1, 6])
        idx = analysis.find_first_gte(x, 8.1)
        self.assertEqual(idx, 6) 

    def test_sign_seq_counts(self):
        xs = [1,2,3,2,1]
        # deltas = [1, 1, -1, -1]
        n = 2
        counts = analysis.sign_seq_counts(xs, n)
        print counts 
        assert counts[1][0, 0] == 1
        assert counts[1][1, 0] == 0
        assert counts[1][1, 2] == 1
        assert counts[-1][0, 2] == 1
        assert counts[-1][1, 2] == 0

