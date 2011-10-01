import numpy as np
import scikits.learn
import scikits.learn.linear_model as lin 

# create a bagged ensemble with rebalanced classes 
class Ensemble:
    def __init__(self, bag_prct=0.75, num_classifiers = 100, **model_keywords):
        self.models = [] 
        self.classes = [] 
        self.bag_prct = bag_prct 
        self.nmodels = num_classifiers
        self.model_keywords = model_keywords
        
    # each bootstrap sample consists of 75% of the rarest class and
    # equal poritions of all other classes 
    def fit(self, X,Y):
        self.classes = np.unique(Y)
        nclasses = len(self.classes)
        class_slices = []
        class_outputs = []
        for c in self.classes:
            mask = (Y==c)
            class_slices.append(X[mask, :])
            class_outputs.append(Y[mask, :])
        
        min_size = np.min([x.shape[0] for x in class_slices])
        print "min_size=", min_size
        class_bag_size = int(min_size * self.bag_prct)
        print "slice_size=", class_bag_size
        total_bag_size = nclasses * class_bag_size 
        print "total_bag_size=", total_bag_size
        n_iter = min(int(np.ceil(10**6 / float(total_bag_size))), 50)
        
        for i in xrange(self.nmodels):
            input_list = []
            output_list = []
            for x, y in zip(class_slices, class_outputs):
                idx = np.random.permutation(x.shape[0])[:class_bag_size]
                input_list.append(x[idx, :])
                output_list.append(y[idx])
            inputs = np.concatenate(input_list)
            outputs = np.concatenate(output_list)
            
            model = lin.SGDClassifier(n_iter=n_iter, **self.model_keywords)
            model.fit(inputs, outputs)
            # bug in scikits.learn keeps around sample weights after training,
            # making the serialization too bloated for network transfer 
            if hasattr(model, 'sample_weight'): model.sample_weight = [] 
            self.models.append(model)
        
    def predict(self, X, return_probs=False):
        cs = self.classes 
        nclasses = len(cs)
        nrows = X.shape[0]
        votes = np.zeros( [nrows, nclasses], dtype='int')
        indices = np.arange(nrows) 
        for model in self.models:
            y = model.predict(X)
            votes += np.sum([y == c for c in cs], 1)
        majority = cs[np.argmax(votes, 1)]
        if return_probs: return majority, votes / np.sum(votes, dtype='float')
        else: return majority 
        
    def __setstate__(self, state):
        self.__dict__ = state
    
    def __getstate__(self):
        return self.__dict__
    
    def __str__(self):
        return "Ensemble: num_classifiers=" + str(self.nmodels) + ", bag_prct=" + str(self.bag_prct)
        
