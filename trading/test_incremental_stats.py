
import numpy as np 
import incremental_stats
xs = 2.0 ** np.arange(15) 

def test_mean():
    expected_means = expected_means = np.cumsum(xs) / (np.arange(len(xs))+1.0)
    mv = incremental_stats.OnlineMeanVar()
    # ascending 
    for (i,x) in enumerate(xs):
        mv.add(x)
        assert mv.mean() == expected_means[i]
    n = len(xs) 
    for (i,x) in enumerate(reversed(xs)):
        mv.remove(x)
        if i < n - 1:
            expected = expected_means[n - i - 2]
        else:
            # once we've removed all the data, no mean can be computed 
            expected = None 
        print "i:", i, "x:", x, "Current mean", mv.mean, "Expected mean:", expected
        if expected is None:
            assert mv.var() is None
        else:
            assert mv.mean() == expected 
    

def test_var():
    mv = incremental_stats.OnlineMeanVar()
    n = len(xs)
    expected_vars = np.zeros(n)
    for i in np.arange(n):
        if i > 0:
            sub_xs = xs[:i+1]
            m = np.mean(sub_xs)
            diff = sub_xs - m
            expected_vars[i] = np.sum(diff * diff) / i
            
    # ascending 
    for (i,x) in enumerate(xs):
        mv.add(x)
        expected = expected_vars[i]
        print "Incrementing variance i:", i, "x:", x, "Current var", mv.var, "Expected var:", expected
        assert abs(mv.var() - expected) < 0.000001
    
    for (i,x) in enumerate(reversed(xs)):
        mv.remove(x)
        if i < n - 1:
            expected = expected_vars[n - i - 2]
        else:
            # once we've removed all the data, no var can be computed 
            expected = None 
        print "Decrementing variance i:", i, "x:", x, "Current var", mv.var, "Expected var:", expected
        if expected is None:
            assert mv.var() is None
        else:
            assert abs(mv.var() - expected) < 0.000001 
    
