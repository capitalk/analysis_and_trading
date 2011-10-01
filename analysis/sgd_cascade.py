import numpy as np 
import scikits.learn
import scikits.learn.svm as svm 
import scikits.learn.linear_model as lin 

# rather than relying on the usual one-vs-all highest score wins output, 
# instead pos/neg have to both beat their opposite signal as well as neutral 
def multiclass_output(model, X, neutral_class=0): 
    n = X.shape[0]
    scores = np.dot(X, model.coef_.T)
    classes = list(model.classes)
    neutral_index = classes.index(neutral_class)
    outputs = np.zeros(n, dtype='int')
    
    pos_index = classes.index(1)
    pos_scores = scores[:, pos_index]
    pos_mask = pos_scores > 0
    
    neg_index = classes.index(-1)
    neg_scores = scores[:, neg_index]
    neg_mask = neg_scores > 0    
    
    neutral_scores = scores[:, neutral_index]
    not_neutral = neutral_scores < 0
    outputs[pos_mask & ~neg_mask & not_neutral] = 1
    outputs[neg_mask & ~pos_mask & not_neutral] = -1 
    #outputs[(pos_scores > 0) & (pos_scores > neg_scores)] = 1
    #outputs[(neg_scores > neutral_scores) & (neg_scores > pos_scores)] = -1
    return outputs 
    
    
class Cascade:
    def __init__(self, num_classifiers=3,  class_weight='auto', error_multiplier = 2.0, neutral_class = 0,  **keywords):
        self.sgd_keywords = keywords 
        self.num_classifiers = num_classifiers
        self.neutral_class = neutral_class 
        self.error_multiplier = error_multiplier
        self.class_weight = class_weight 

        self.models = [] 

    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, state): 
        self.__dict__ = state 
    
    def __str__(self):
        return "Cascade(" + str(self.models) + ")"
        
    def fit(self, X,Y): 
        self.classes = np.unique(Y)
        
        n = X.shape[0]
        n_iter = int(np.ceil(10**6 / float(n)))
        sample_weight = np.ones(n)
        #neutral = Y == self.neutral_class 
        if self.class_weight == 'auto':
            weights = 'auto'
        else: 
            weights = dict([ (c, self.class_weight) for c in self.classes])
            weights[self.neutral_class] = 1
        for i in xrange(self.num_classifiers):
            print "Cascade level", i 
            model = lin.SGDClassifier(n_iter=n_iter, **self.sgd_keywords)
            
            model.fit(X,Y, class_weight=weights, sample_weight = sample_weight)
            
            # clear this field since scikits.learn doesn't for some reason 
            model.sample_weight = [] 
            self.models.append(model)
            pred = model.predict(X)
            wrong = (pred != Y)
            sample_weight[wrong] *= self.error_multiplier
            sample_weight[~wrong] /= self.error_multiplier
            
            
    def predict(self, X):
        n = X.shape[0]
        result = np.ones(n) * self.neutral_class  
        prev_predictions = result.copy() 
        # start with all rows and sequentially discard those that get
        # predicted as the neutral class at any point in the pipeline 
        idx = np.arange(n)
        for model in self.models:
            #if set(self.classes) == set([1,0,-1]):
            #    pred =  multiclass_output(model, X[idx, :]) #model.predict(X[idx, :])
            #else:
            #    pred = model.predict(X[idx, :])
            pred = model.predict(X[idx, :])
            not_neutral = pred != self.neutral_class
            consistent = (prev_predictions[idx] == pred) | (prev_predictions[idx] == self.neutral_class)
            prev_predictions[idx] = pred 
            keep = not_neutral & consistent
            idx = idx[keep]
        result[idx] = pred[keep]
        return result 
