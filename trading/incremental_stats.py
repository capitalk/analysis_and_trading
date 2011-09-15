import collections 
import math 

class OnlineMeanVar:
    def __init__(self):
        self.n = 0 
        self.mean_ = None
        self.var_ = None 
        #self.minval = None
        #self.maxval = None 
    
    def mean(self):
        return self.mean_
        
    def var(self): 
        return self.var_
        
    def std(self):
        v = self.var_ 
        if v is not None: return math.sqrt(v)
        else: return None 
        

    def add(self, x):
        n = self.n 
        
        if n < 1:
            self.mean_ = float(x)
            self.var_ = 0.0
        elif n == 1:
            old_mean = self.mean_ 
            self.mean_ += x
            self.mean_ /= 2
            self.var_ = (x-old_mean) * (x-self.mean_)
            
        else:
            
            old_mean = self.mean_ 
            self.mean_ *= n 
            self.mean_ += x
            self.mean_ /= (n+1) 
            
            self.var_ *= (n-1)
            self.var_ += (x-old_mean) * (x - self.mean_)
            self.var_ /= n 
        
        self.n += 1
        
        
    def remove(self, x):
        n = self.n 
        if n <= 1:
            self.mean_ = None
            self.var_ = None
        elif n == 2:
            self.mean_ *= 2 
            self.mean_ -= x 
            self.var_ = 0.0 
        else:
            old_mean = self.mean_ 
            self.mean_ *= n 
            self.mean_ -= x
            self.mean_ /= (n-1) 
            
            self.var_ *= (n-1)
            self.var_ -= (x-old_mean) * (x - self.mean_)
            self.var_ /= (n-2)
        self.n -= 1 
