import expr_lang 
import numpy as np 

def test_tokenize(): 
    res = expr_lang.tokenize("x+y") 
    expected = ["x", "+", "y"] 
    print "Expected:", expected, "Result:", res 
    assert res == expected 
    res2 = expr_lang.tokenize("bid/mean/5s >=3.234")
    expected2 = ["bid/mean/5s", ">=", "3.234"] 
    print "Expected:", expected2, "Result:", res2 
    assert res2 == expected2
    res3 = expr_lang.tokenize("(log2 x) + (log2 y)")
    expected3 = ['(', 'log2', 'x', ')', '+', '(', 'log2', 'y', ')']
    print "Expected:", expected3, "Result:", res3
    assert res3 == expected3 

def test_add():
    ev = expr_lang.Evaluator()
    res = ev.eval_expr("1+2") 
    assert res == 3 

def test_env():
    ev = expr_lang.Evaluator()
    env = {'x': np.array([1,2,3]), 'y': np.array([4,5,6])}
    res = ev.eval_expr("x+y", env=env) 
    print res 
    assert np.all(res == np.array([5,7,9]))
    # test slicing 
    res2 = ev.eval_expr("x+y", start_idx=1, end_idx=2, env=env)
    print res2
    assert np.all(res2 == np.array([7])) 

def test_compiler_simple():
    code = expr_lang.compile_expr('3+4')
    assert code({}) == 7 
    
def test_compiler_env():
    env = {'x':3, 'y':4}
    code = expr_lang.compile_expr('x+y')
    res1 = code(env)
    print "Expected 7, got", res1 
    assert res1 == 7
    env['x'] = 4
    res2 = code(env)
    print "Expected 8, got", res2 
    assert res2 == 8

def test_compiler_nested():
    code = expr_lang.compile_expr('(10+4)%2')
    res = code({})
    print "Received: ", res
    assert res == 7 
    
def test_compile_array(): 
    code = expr_lang.compile_expr('(log2 x) + (log2 y)')
    env = {'x': np.array([2, 4]), 'y': np.array([4, 8])}
    res = code(env)
    expected = np.array([3,5])
    print "Expected:", expected, "Received:", res 
    assert np.all(res == expected)
    
def test_symbol_set():
    s = '(log2 x) + (log2 y) + 0.5 * 3 - x/y/z'
    symbols = expr_lang.symbol_set(s)
    expected = set(['x', 'y', 'x/y/z'])
    print "Expected:", expected, "Result:", symbols 
    assert symbols == expected 
