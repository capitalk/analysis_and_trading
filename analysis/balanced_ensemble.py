import math 
import numpy as np
import scikits.learn
import scikits.learn.linear_model as lin 
import scikits.learn.svm as svm 

# create a bagged ensemble with rebalanced classes 
class Ensemble:
    # weighting = 'uniform' | 'accuracy' 
    # nfeatures = percent | 'sqrt' | 'log'
    # thresh = percent of votes required for a non-zero class 
    def __init__(self, balanced_bagging=False, bag_prct=0.75, base_classifier='sgd', num_random_features='sqrt', num_classifiers = 100, weighting='f-score', thresh=0.6, recall_importance=0.25, neutral_weight=1, **model_keywords):
        self.models = [] 
        self.weighting = weighting
        self.model_weights = None
        self.model_features = [] 
        self.classes = [] 
        self.balanced_bagging = balanced_bagging
        self.bag_prct = bag_prct 
        self.base_classifier = base_classifier
        self.nmodels = num_classifiers
        self.num_random_features = num_random_features
        self.model_keywords = model_keywords
        #self.prior_probabilities = None 
        self.neutral_weight = neutral_weight
        self.recall_importance = recall_importance 
        self.thresh = thresh 
        
    # each bootstrap sample consists of 75% of the rarest class and
    # equal poritions of all other classes 
    def fit(self, X, Y, class_weight=None):
        self.classes = np.unique(Y)
        nclasses = len(self.classes)
        total_nrows = X.shape[0] 
        nfeatures = X.shape[1]
        
        if self.balanced_bagging:
            class_slices = []
            class_outputs = []
            for c in self.classes:
                mask = (Y==c)
                class_slices.append(X[mask, :])
                class_outputs.append(Y[mask, :])
            min_size = np.min([x.shape[0] for x in class_slices])
            class_bag_size = int(min_size * self.bag_prct)
            total_bag_size = nclasses * class_bag_size 
        else:
            total_bag_size = total_nrows
        
        
        print "total_bag_size=", total_bag_size
        
        if self.num_random_features == 'sqrt':
            features_per_model = int(math.ceil(math.sqrt(nfeatures)))
        elif self.num_random_features == 'log':
            features_per_model = int(math.ceil(math.log(nfeatures, 2)))
        else:
            features_per_model = int(math.ceil(nfeatures * self.num_random_features))
        print "Features per model:", features_per_model
        
        if class_weight is None: 
            class_weight = {}
            for c in self.classes:
                class_weight[c] = 1.0
            class_weight[0] = self.neutral_weight 
        print "[Class Weights]", class_weight 
        
        f_scores = [] 
        
        for i in xrange(self.nmodels):
            print "Training model #" + str(i)
            feature_indices = np.random.permutation(nfeatures)[:features_per_model]
            print "Features:", feature_indices 
            self.model_features.append(feature_indices)
            
            if self.balanced_bagging:
                input_list = []
                output_list = []
                for x, y in zip(class_slices, class_outputs):
                    row_indices = np.random.permutation(x.shape[0])[:class_bag_size]
                    row_slice = x[row_indices, :] 
                    input_list.append(row_slice[:, feature_indices])
                    output_list.append(y[row_indices])
                
                inputs = np.concatenate(input_list)
                outputs = np.concatenate(output_list)
            else:
                inputs = X[:, feature_indices]
                outputs = Y
            
            
            if self.base_classifier == 'sgd':
                print "Input shape:", inputs.shape
                n_iter = int(np.ceil(10**6 / float(inputs.shape[0])))
                print "Num iters: ", n_iter
                model = lin.SGDClassifier(n_iter=n_iter, shuffle=True, **self.model_keywords)
            else:
                model = svm.LinearSVC(**self.model_keywords) # svm.SVC(kernel='poly', degree=2)
            model.fit(inputs, outputs, class_weight=class_weight)
            print model 
            print model.coef_
            # bug in scikits.learn keeps around sample weights after training,
            # making the serialization too bloated for network transfer 
            if hasattr(model, 'sample_weight'): model.sample_weight = [] 
            self.models.append(model)
            
            # remember the balanced accuracy for each model 
            pred = model.predict(inputs)
            print "outputs[100:150]", outputs[100:150]
            print "pred[100:150]", pred[100:150]
            
            # compure F-score for model weighting and user feedback 
            actual_not_zero = (outputs != 0)
            actual_not_zero_count = np.sum(actual_not_zero)
            print "  Actual NNZ: ", actual_not_zero_count
            
            pred_not_zero = (pred != 0)
            pred_not_zero_count = np.sum(pred_not_zero)
            print "  Predicted NNZ:", pred_not_zero_count
            
            correct = (outputs == pred)
            correct_not_zero = np.sum(correct & actual_not_zero, dtype='float') 
            print "   Correct NNZ:", correct_not_zero 
            
            precision = correct_not_zero / float(pred_not_zero_count)
            print "  Precision:", precision 
            
            recall = correct_not_zero / float(actual_not_zero_count)
            print "  Recall:", recall 
            
            if precision > 0 or recall > 0:
                beta_squared = self.recall_importance ** 2
                denom = beta_squared * precision + recall 
                f_score = (1+beta_squared)* (precision * recall) / denom
            else: f_score = 0.0
            print "  F-score:", f_score
            f_scores.append(f_score)

        f_scores = np.array(f_scores)
        sum_f_scores = np.sum(f_scores)
        if sum_f_scores == 0: raise RuntimeError("all these classifiers are terrible")
        self.model_weights = f_scores / sum_f_scores
        
    def predict(self, X, return_probs=False):
        cs = self.classes 
        nclasses = len(cs)
        nrows = X.shape[0]
        votes = np.zeros( [nrows, nclasses], dtype='float')
        indices = np.arange(nrows) 
        
        for i in xrange(self.nmodels):
            model = self.models[i]
            weight = self.model_weights[i]
            feature_indices = self.model_features[i]
            y = model.predict(X[:, feature_indices])
            curr_votes = weight * np.array([y == c for c in cs]).T    
            votes += curr_votes
        
        majority = cs[np.argmax(votes, 1)]
        max_vals = np.max(votes, 1)
        majority[max_vals < self.thresh] = 0
        if return_probs: 
            probs = votes / np.array([np.sum(votes, 1, dtype='float')]).T
            return majority, probs
        else: return majority 
        
    def __setstate__(self, state):
        self.__dict__ = state
    
    def __getstate__(self):
        return self.__dict__
    
    def __str__(self):
        return "Ensemble: nmodels = " + str(self.nmodels)
        
