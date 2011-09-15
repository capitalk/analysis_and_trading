import features 
import numpy as np 
import unittest 


class TestAggregators(unittest.TestCase):

    def test_digit_close_to_wrap(self):
        self.assertEqual(features.digit_close_to_wrap(5), 0)
        self.assertEqual(features.digit_close_to_wrap(0), 1.0)
        self.assertEqual(features.digit_close_to_wrap(4), 1.0/5.0)

    def test_nth_digit(self):
        n = 3.14159
        d1 = features.nth_digit(n, 1)
        print "First digit of", n, " = ", d1
        self.assertEqual(d1, 1)
        d2 = features.nth_digit(n, 2)
        print "Second digit of", n, " = ", d2
        self.assertEqual(d2, 4)
        d3 = features.nth_digit(n, 3)
        print "Third digit of", n, " = ", d3
        self.assertEqual(d3, 1)
        d4 = features.nth_digit(n, 4)
        print "Fourth digit of", n, " = ", d4
        self.assertEqual(d4, 5)
        d5 = features.nth_digit(n, 5)
        print "Fifth digit of", n, " = ", d5
        self.assertEqual(d5, 9)
        self.assertEqual(features.nth_digit(31.2, 1), 2)
    
    def test_nth_digit_negative(self):
        n = -3.14159
        d1 = features.nth_digit(n, 1)
        print "First digit of", n, " = ", d1
        self.assertEqual(d1, 1)
        d2 = features.nth_digit(n, 2)
        print "Second digit of", n, " = ", d2
        self.assertEqual(d2, 4)
        d3 = features.nth_digit(n, 3)
        print "Third digit of", n, " = ", d3
        self.assertEqual(d3, 1)
        d4 = features.nth_digit(n, 4)
        print "Fourth digit of", n, " = ", d4
        self.assertEqual(d4, 5)
        d5 = features.nth_digit(n, 5)
        print "Fifth digit of", n, " = ", d5
    
        
    def test_nth_digit_tail(self):
        n = 3.14159
        t1 = features.nth_digit_tail(n, 1)
        self.assertAlmostEqual(t1, 1.4159)
        t2 = features.nth_digit_tail(n, 2)
        self.assertAlmostEqual(t2, 4.159)

    def test_volume_weighted_overall(self):
        import orderBookConstants as obc
        from orderBook import * 
        bid1 = Order(price= 1.0, size=90)
        offer1 = Order(price = 2.0, size=30)  
        ob = OB()
        ob.add_bid(bid1)
        ob.add_offer(offer1)
        vwp = features.volume_weighted_overall_price(ob)
        self.assertAlmostEqual(vwp, 1.25)
        
        
