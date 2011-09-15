import numpy as np
import signals

def test_clean():
    x = np.array([1,0,0,1,0,0])
    expected_x3 = np.array([0,0,0,0,0,0])
    actual_x3 = signals.clean_signal(x, win_size=3)
    print "Window size 3", actual_x3
    assert np.all(expected_x3 == actual_x3)
    actual_x4 = signals.clean_signal(x, win_size = 4)
    print "Window size 4", actual_x4 
    assert np.all(x == actual_x4)
    
