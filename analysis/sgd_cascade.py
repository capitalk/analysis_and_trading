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
    pos_index = classes.index(1)
    neg_index = classes.index(-1)
    outputs = np.zeros(n, dtype='int')
    pos_scores = scores[:, pos_index]
    neutral_scores = scores[:, neutral_index]
    neg_scores = scores[:, neg_index]
    pos_mask = pos_scores > 0
    neg_mask = neg_scores > 0
    neutral_mask = neutral_scores > 0
    outputs[pos_mask & ~neg_mask & ~neutral_mask] = 1
    outputs[neg_mask & ~pos_mask & ~neutral_mask] = -1 
    #outputs[(pos_scores > 0) & (pos_scores > neg_scores)] = 1
    #outputs[(neg_scores > neutral_scores) & (neg_scores > pos_scores)] = -1
    return outputs 
    
    
class Cascade:
    def __init__(self, neutral_class = 0, num_classifiers=3, **keywords):
        self.sgd_keywords = keywords 
        self.num_classifiers = num_classifiers
        self.neutral_class = neutral_class 
        self.models = [] 

    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, state): 
        self.__dict__ = state 
    
    def __str__(self):
        return "Cascade(" + str(self.models) + ")"
        
    def fit(self, X,Y): 
        self.classes = np.unique(Y)
        not_neutral = Y != self.neutral_class
        for i in xrange(self.num_classifiers):
            # stop training once there are no more instances of the neutral class
            if np.sum(Y == self.neutral_class) == 0: break
            
            print "Cascade level", i 
            n = X.shape[0]
            if n > 20000:
                n_iter = int(np.ceil(10**6 / float(n)))
                model = lin.SGDClassifier(n_iter=n_iter, **self.sgd_keywords)
            elif n > 10000:
                model = svm.LinearSVC()
            else:
                model = svm.SVC(kernel='poly', degree=2)
                
            model.fit(X,Y, class_weight="auto")
            print "Automatic class weights=", model.class_weight 
            # clear this field since scikits.learn doesn't for some reason 
            
            model.sample_weight = [] 
            self.models.append(model)
            pred = model.predict(X)
            wrong = (pred != Y)
            keep_idx = not_neutral | wrong 
            X = X[keep_idx, :] 
            Y = Y[keep_idx] 
            not_neutral = not_neutral[keep_idx]
            
            
    def predict(self, X):
        n = X.shape[0]
        result = np.ones(n) * self.neutral_class  
        # start with all rows and sequentially discard those that get
        # predicted as the neutral class at any point in the pipeline 
        idx = np.arange(n)
        for model in self.models:
            pred = model.predict(X[idx, :]) #multiclass_output(model, X[idx, :]) #
            idx = idx[pred != self.neutral_class]
        result[idx] = pred[pred != self.neutral_class]  
        return result 
