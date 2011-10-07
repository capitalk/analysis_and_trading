import math 
import numpy as np
import sklearn.linear_model as lin 
import sklearn.svm as svm 


# create a bagged ensemble with rebalanced classes 
class Ensemble:
    # nfeatures = percent | 'sqrt' | 'log'
    # thresh = percent of votes required for a non-zero class 
    def __init__(self, balanced_bagging=False, bag_prct=0.85, base_classifier='sgd', num_random_features=0.5, num_classifiers = 100, model_weighting='logistic', thresh=0.75, recall_importance=0.25, neutral_weight=4, model_params={}):
        self.models = [] 
        self.model_weighting = model_weighting
        self.model_scores = None
        self.model_features = [] 
        self.classes = [] 
        self.balanced_bagging = balanced_bagging
        self.bag_prct = bag_prct 
        self.base_classifier = base_classifier
        self.nmodels = num_classifiers
        self.num_random_features = num_random_features
        self.model_keywords = model_params
        #self.prior_probabilities = None 
        self.neutral_weight = neutral_weight
        self.recall_importance = recall_importance 
        self.thresh = thresh 
    
    def transform_to_classifer_space(self, X):
        nmodels = len(self.models)
        nrows = X.shape[0]
        X2 = np.zeros( (nrows, nmodels) )
        nnz = 0 
        print "Transforming to classifier space.." 
        for i, model in enumerate(self.models):
            feature_indices = self.model_features[i]
            y = model.predict(X[:, feature_indices])
            nnz += np.sum(y != 0) 
            X2[:, i] = y
        total_elts = nrows * nmodels
        print "Transformed sparsity = ", nnz, "/", (total_elts), "[", (nnz / float(total_elts)), "]"
        return X2     
    
    # each bootstrap sample consists of 75% of the rarest class and
    # equal poritions of all other classes 
    def fit(self, X, Y, class_weight=None):
        self.classes = list(np.unique(Y))
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
            print "Balanced bagging, min class size =", class_bag_size 
            total_bag_size = (nclasses -1)* class_bag_size  + (self.neutral_weight * class_bag_size)
        else:
            total_bag_size = total_nrows
        
        
        print "total_bag_size = ", total_bag_size
        
        if self.num_random_features == 'sqrt':
            features_per_model = int(math.ceil(math.sqrt(nfeatures)))
        elif self.num_random_features == 'log':
            features_per_model = int(math.ceil(math.log(nfeatures, 2)))
        else:
            features_per_model = int(math.ceil(nfeatures * self.num_random_features))
        print "Features per model:", features_per_model
        
        if class_weight is None: class_weight = 'auto'
        print "[Class Weights]", class_weight 
        
        f_scores = [] 
        
        for i in xrange(self.nmodels):
            print "Training model #" + str(i)
            feature_indices = np.random.permutation(nfeatures)[:features_per_model]
            print "  Features:", feature_indices 
       
            
            if self.balanced_bagging:
                input_list = []
                output_list = []
                for i, c in enumerate(self.classes):
                    x = class_slices[i]
                    y = class_outputs[i]
                    n = self.neutral_weight * class_bag_size if c == 0 else class_bag_size
                    row_indices = np.random.permutation(x.shape[0])[:n]
                    row_slice = x[row_indices, :] 
                    input_list.append(row_slice[:, feature_indices])
                    output_list.append(y[row_indices])
                
                inputs = np.concatenate(input_list)
                outputs = np.concatenate(output_list)
            else:
                inputs = X[:, feature_indices]
                outputs = Y
            
            
            if self.base_classifier == 'sgd':
                print "  Input shape:", inputs.shape
                n_iter = int(np.ceil(10**6 / float(inputs.shape[0])))
                print "  Num iters: ", n_iter
                model = lin.SGDClassifier(n_iter=n_iter, shuffle=True, **self.model_keywords)
            elif self.base_classifier == 'logistic': 
                model = lin.LogisticRegression(**self.model_keywords)
            elif self.base_classifier == 'nu-svm':
                model = svm.NuSVC(nu=0.1, kernel='linear')
            else:
                model = svm.LinearSVC(**self.model_keywords) # svm.SVC(kernel='poly', degree=2)
            model.fit(inputs, outputs, class_weight=class_weight)
            print model 
            #print model.coef_
            # bug in scikits.learn keeps around sample weights after training,
            # making the serialization too bloated for network transfer 
            if hasattr(model, 'sample_weight'): model.sample_weight = [] 
            
            # remember the balanced accuracy for each model 
            pred = model.predict(inputs)
            #print "outputs[100:150]", outputs[100:150]
            #print "pred[100:150]", pred[100:150]
            
            # compure F-score for model weighting and user feedback 
            actual_not_zero = (outputs != 0)
            actual_not_zero_count = np.sum(actual_not_zero)
            
            
            pred_not_zero = (pred != 0)
            pred_not_zero_count = np.sum(pred_not_zero)
            
            correct = (outputs == pred)
            correct_not_zero = np.sum(correct & actual_not_zero, dtype='float') 
            
            print "   Correct NNZ:", correct_not_zero,  "Actual NNZ: ", actual_not_zero_count, "Predicted NNZ:", pred_not_zero_count

            if pred_not_zero_count > 0: precision = correct_not_zero / float(pred_not_zero_count)
            else: precision = 0.0
            
            if actual_not_zero_count > 0: recall = correct_not_zero / float(actual_not_zero_count)
            else: recall = 0.0
            
            if precision > 0 and recall > 0:
                beta_squared = self.recall_importance ** 2
                denom = beta_squared * precision + recall 
                f_score = (1+beta_squared)* (precision * recall) / denom
            else: f_score = 0.0
            
            print "  Precision:", precision , "Recall:", recall, "F-score:", f_score            
            
            if f_score > 0: 
                self.model_features.append(feature_indices)
                f_scores.append(f_score)
                self.models.append(model)

            
        f_scores = np.array(f_scores)
        sum_f_scores = np.sum(f_scores)
        if sum_f_scores == 0: 
            print "!!!! All classifiers are terrible  !!!!"
            self.model_scores = f_scores
        else:
            self.model_scores = f_scores / sum_f_scores
            # estimate how good each feature is 
            counts = np.zeros(nfeatures)
            feature_scores = np.zeros(nfeatures)
            
            for f, indices in zip(f_scores, self.model_features):
                counts[indices] += 1
                feature_scores[indices] += f
            feature_scores /= counts 
            print "Average feature scores:", feature_scores 
            #sorted in ascending order
            sort_indices = np.argsort(feature_scores)
            print "Best 5 features:", sort_indices[-5:]
            
        if self.model_weighting == 'logistic':
            X2 = self.transform_to_classifer_space(X)
            print "Training logistic regression on top of ensemble outputs..."
            self.gating_classifier = lin.LogisticRegression()
            self.gating_classifier.fit(X2, Y)
        else:
            self.gating_classifier = None 
            

    def predict(self, X, return_probs=False):
        cs = np.array(self.classes)
        nclasses = len(cs)
        nrows = X.shape[0]
        
        # if we have a classifier on top of the ensemble use it
        if self.gating_classifier is not None:
            X2 = self.transform_to_classifer_space(X)
            probs = self.gating_classifier.predict_proba(X2)
        # otherwise weight each model by f_score 
        else:
            votes = np.zeros( [nrows, nclasses], dtype='float')
            for i, model in enumerate(self.models):
                weight = self.model_scores[i]
                feature_indices = self.model_features[i]
                y = model.predict(X[:, feature_indices])
                curr_votes = weight * np.array([y == c for c in cs]).T    
                votes += curr_votes
                probs = votes / np.array([np.sum(votes, 1, dtype='float')]).T
        majority = cs[np.argmax(probs, 1)]
        
        # set any probabilities below threshold to neutral class 
        max_probs = np.max(probs, 1)
        majority[max_probs < self.thresh] = 0
        if return_probs: 
            return majority, probs
        else: 
            return majority 
        
    def __setstate__(self, state):
        self.__dict__ = state
    
    def __getstate__(self):
        return self.__dict__
    
    def __str__(self):
        return "Ensemble: " + str({
            'nmodels': len(self.models), 
            'balanced_bagging': self.balanced_bagging, 
            'bag_prct': self.bag_prct, 
            'base_classifier': self.base_classifier, 
            'num_random_features': self.num_random_features, 
            'thresh': self.thresh, 
            'recall_importance': self.recall_importance, 
            'neutral_weight': self.neutral_weight, 
        })
        
