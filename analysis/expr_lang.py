import numpy as np 
import scipy
import scipy.signal 
import scipy.stats

def clean(x):
    top = scipy.stats.scoreatpercentile(x, 99)
    bottom = scipy.stats.scoreatpercentile(x, 1)    
    outliers = np.abs(x) > 2 * top 
    if np.any(outliers):
        x = x.copy()
        x[outliers & (x < 0)] = bottom
        x[outliers & (x > 0)] = top 
    return x

def diff(x):
    return np.concatenate([[0], np.diff(x)])
    
def slope(x):
    return diff(x) / 1000.0

def medfilt(x, winsize):
    return scipy.signal.medfilt(x, winsize)

def avg(x,y):
    return (x+y)/2
            
def replace_inf(x,y):
    inf = np.isinf(x)
    if np.any(inf):
        x = x.copy()
        x[inf] = y 
    return x
    
def replace_nan(x,y):
    nan = np.isnan(x)
    if np.any(nan):
        x = x.copy()
        x[nan] = y 
    return x


def replace_inf_or_nan(x,y):
    return replace_inf(replace_nan(x,y), y)

# ratio between two quantities, in the presence of zeros and weird extrema
# if a ratio is bad (too small, too big, NaN, or inf) set it to 1.0 
def safe_div(x,y):
    x_zero = x == 0
    y_zero = y == 0
    if np.any(x_zero) or np.any(y_zero):
        both_zero = x_zero & y_zero 
        x = x.copy()
        x[both_zero] = 1.0
        y = y.copy()
        y[both_zero] = 1.0
        y[y_zero] = x[y_zero] 
    z = x / y
    return clean(z)


def add(x,y):
    return x + y 
    
def sub(x,y):
    return x - y
    
def mult(x,y):
    return x * y 
    
def div(x,y):
    return x / y 

def gt(x,y):
    return x > y
    
def gte(x,y):
    return x >= y
    
def lt(x,y):
    return x < y
    
def lte(x,y):
    return x <= y
    
def eq(x,y):
    return x == y 

def select_bigger_abs(x,y):
    return np.where(np.abs(x) > np.abs(y), x, y)
    
def select_smaller_abs(x,y):
    return np.where(np.abs(x) < np.abs(y), x, y)

def when(x,y):
    return x[y]
    
binops = { 
    '+': add, #np.add,
    '-': sub, #np.subtract, 
    '*': mult, #np.multiply, 
    '%': div, #np.divide,
    '>': gt,
    '>=': gte,
    '<': lt, 
    '<=': lte, 
    '=': eq,
    'mod': np.mod,
    'avg': avg, 
    'min': np.minimum, 
    'max': np.maximum, 
    'replace_inf': replace_inf,
    'replace_nan': replace_nan, 
    'replace_inf_or_nan': replace_inf_or_nan, 
    'safe_div': safe_div,
    'select_bigger_abs': select_bigger_abs,
    'select_smaller_abs': select_smaller_abs, 
    'when': when, 
    'medfilt': medfilt, 
}


    
unops = { 
    'diff': diff, 
    'slope': slope, 
    'log': np.log, 
    'log10': np.log10, 
    'log2': np.log2, 
    'sin': np.sin, 
    'cos': np.cos, 
    'tan': np.tan, 
    'std': np.std, 
    'mean': np.mean, 
    'abs': np.abs, 
    'clean': clean,
    
}
def tokenize(s):
    # remove quotes and strip whitespace
    s = s.replace('"', '')
    s = s.strip()
    
    tokens = []
    curr = ''
    
    special = ['(', ')', '+', '-', '*', '%', '>', '>=', '<', '<=', '=']
    for c in s:
        if c == ' ' or c == '\t': 
            if curr != '': tokens.append(curr)
            curr = '' 
        elif c in special:
            if curr + c in special: 
                curr += c
            else: 
                if curr != '': tokens.append(curr)
                curr = ''
                tokens.append(c) 
        else:
            if curr in special:
                if curr != '': tokens.append(curr)
                curr = c 
            else: 
                curr += c 
    if curr != '': tokens.append(curr)
    return tokens 


def mk_const_fn(const):
    def fn(env):
        return const
    return fn 
    
def mk_var_fn(name):
    def fn(env):
        return env[name]
    return fn 
    
def mk_unop_fn(unop_name, arg): 
    unop = unops[unop_name] 
    def fn(env):
        arg_val = arg(env)
        return unop(arg_val)
    return fn 
    
def mk_binop_fn(binop_name, left, right): 
    binop = binops[binop_name] 
    def fn(env):
        left_val = left(env)
        right_val = right(env)
        return binop(left_val, right_val)
    return fn 
    
def compile_tokens(tokens):
    
    # a stack of 0-ary function used as arguments to the mk_xyz functions above
    curr_value_stack = [] 
    curr_waiting_binops = []
    
    old_value_stacks = []
    old_waiting_binops = [] 
    
    # reversed since we evaluate right to left 
    for token in reversed(tokens): 
        #print idx, ":",  token, " binop stack:", curr_waiting_binops
        if token in unops:
            assert len(curr_value_stack) >= 1 
            arg = curr_value_stack.pop()
            future_unop_result = mk_unop_fn(token, arg)
            curr_value_stack.append(future_unop_result)
        elif token in binops:
            assert len(curr_value_stack) >= 1
            curr_waiting_binops.append(token)
            
        elif token == ')':
            old_value_stacks.append(curr_value_stack)
            old_waiting_binops.append(curr_waiting_binops)
            curr_value_stack = []
            curr_waiting_binops = [] 
            
        elif token == '(':
            assert len(curr_value_stack) == 1
            v = curr_value_stack.pop()
            assert len(old_value_stacks) >= 1
            curr_value_stack = old_value_stacks.pop()
            assert len(curr_waiting_binops) == 0
            assert len(old_waiting_binops) >= 1
            curr_waiting_binops = old_waiting_binops.pop()
            if len(curr_waiting_binops) > 0:
                binop_name = curr_waiting_binops.pop()
                assert len(curr_value_stack) >= 1
                rightArg = curr_value_stack.pop()
                future_result = mk_binop_fn(binop_name, v, rightArg)
                curr_value_stack.append(future_result) 
            else:
                curr_value_stack.append(v) 
        else:
            try:
                const = float(token)
                arg = mk_const_fn(const) 
            except: 
                arg = mk_var_fn(token)
            if len(curr_waiting_binops) > 0:
                assert len(curr_value_stack) > 0
                binop_name = curr_waiting_binops.pop()
                rightArg = curr_value_stack.pop() 
                future_result = mk_binop_fn(binop_name, arg, rightArg)
                curr_value_stack.append(future_result)
            else:
                curr_value_stack.append(arg)
    assert old_waiting_binops == []
    assert old_value_stacks == []
    assert curr_waiting_binops == []
    assert len(curr_value_stack) == 1
    return curr_value_stack.pop() 
    
# returns a 0-argument function which re-evaluates the expression
# every time it's called. The env stays fixed but the values of its mappings
# can change between calls (since it's assumed to be a mutable dict) 
def compile_expr(expr):
    tokens = tokenize(expr)
    return compile_tokens(tokens)

# helps the dataset's field expression evaluator keep track of evaluation state 
class EvalHelper():
    def __init__(self):
        self.value_stack = []
        self.code_stack = [] 
        self.saved_contexts = [] 

    def _try_running_code(self):
        if self.code_stack != []: 
            (fn, arity) = self.code_stack[-1] 
            if len(self.value_stack) >= arity:
                args = []
                for i in xrange(arity):
                    args.append(self.value_stack.pop())
                #print "Calling ", fn, "with ", args 
                result = fn(*args)
                self.value_stack.append(result)
                self.code_stack.pop() 
        
    def enter_context(self):
        self.saved_contexts.append([self.value_stack, self.code_stack])
        self.value_stack = []
        self.code_stack = []
    
    # last value in a context is returned and old value/code stacks restored
    def exit_context(self):
        v = self.value_stack[-1]
        (old_values, old_code) = self.saved_contexts.pop()
        self.value_stack = old_values
        self.code_stack = old_code
        return v 
        
    def push_value(self, v): 
        self.value_stack.append(v)
        self._try_running_code()
    
    def pop_value(self):
        return self.value_stack.pop()
    
    def has_values(self):
        return self.value_stack != []
        
    def push_code(self, c, arity):
        self.code_stack.append([c,arity])
        self._try_running_code()

# parses either one string or a list of strings
# returns a set of symbols excluding all special operators and binops 
def symbol_set(strings): 
    if type(strings) == str: strings = [strings] 
    symbols  = set([]) 
    for s in strings:
        tokens = tokenize(s)
        for t in tokens:
            if t not in unops and t not in binops and t != '(' and t != ')':
                try:
                    f = float(t)
                except:
                    # only add non-numeric tokens 
                    symbols.add(t)
    return symbols
    
# the difference between the compiler and evaluator is that the evaluator 
# allows you to apply the same start_idx, end_idx to every symbol
# implicitly assuming that they are all arrays. 
# eventually these should be merged. 
class Evaluator(): 
    def __init__(self):
        pass
    
    # given a column name or constant, return its value
    def col_or_num(self, s, start_idx=None, end_idx=None, env=None):
        try:
            return float(s)
        except:
            if env is None:
                raise RuntimeError("Environment is None, can't evaluate: " + s)
            else:
                try:
                    v = env[s] 
                    if (start_idx is None) and (end_idx is None):
                        return v
                    else:
                        return v[start_idx:end_idx] 
                except:
                    raise RuntimeError("Unrecognized expression: " + s +" (expected number or column name)")
            
    def eval_token_list(self, tokens, last_token=None, start_idx=None, end_idx=None, env=None):
        
        # evaluate in right-to-left APL style 
        if last_token is None:
            idx = len(tokens) - 1 
        else:
            idx = last_token 
        state = EvalHelper()
        while idx >= 0:
            token = tokens[idx]
            if token in unops:
                if not state.has_values():
                    raise RuntimeError("[eval] Unary operator " + token + " missing argument")
                else:
                    state.push_code(unops[token], 1)
            elif token in binops:
                if idx == 0:
                    raise RuntimeError("[eval] Binary operator " + token + " missing left argument")
                elif not state.has_values():
                    raise RuntimeError("[eval] Binary operator " + token + " missing right argument")
                else:
                    state.push_code(binops[token], 2)
            elif token == ')':
                state.enter_context()
            elif token == '(':
                v = state.exit_context()
                state.push_value(v)
            else:
                v = self.col_or_num(token, start_idx, end_idx, env=env)
                state.push_value(v)
            idx -= 1
        return state.pop_value()
        
    # given an array expression consisting of column names, constant numbers
    # and mathematical operators, evaluate the expression and return its result
    def eval_expr(self, expr, start_idx = None, end_idx = None, env=None):
        tokens = tokenize(expr)
        return self.eval_token_list(tokens, start_idx=start_idx, end_idx=end_idx, env=env)
    
