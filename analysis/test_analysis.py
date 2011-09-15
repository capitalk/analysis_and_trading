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
