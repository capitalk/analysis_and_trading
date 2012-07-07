import numpy as np 
import sklearn.mixture 
import sklearn.cluster 

class ClusterThresholding:
    def __init__(self, 
            target_precision=0.7,  
            n_clusters = 5, 
            neutral_class = 0, 
            n_iters = 10, 
            n_restarts = 3, 
            verbose = False,
            gmm = False, 
            minibatch = True, # use approximate k-means
            covariance_type = 'full',
            cooling = 1.0):
        self.neutral_class = neutral_class
        self.target_precision = target_precision
        self.n_iters = n_iters 
        self.n_restarts = n_restarts
        self.gmm = gmm 
        self.minibatch = minibatch
        self.n_clusters = n_clusters
        self.covariance_type = covariance_type
        self.cooling = cooling 
        self.verbose = verbose 
        self.models = []
        self.thresholds = [] 
        self.classes = None
        self.active_classes = None


    def _mk_model(self):
        if self.gmm:
            return sklearn.mixture.GMM(n_components=self.n_clusters, cvtype = self.covariance_type)
        elif self.minibatch:
            return sklearn.cluster.MiniBatchKMeans(k = self.n_clusters)
        else:
            return sklearn.cluster.KMeans(k = self.n_clusters)

    def _scores(self, model, X):
        if self.gmm:
            return sklearn.mixture.lmvnpdf(X, model.means, model.covars, self.covariance_type)
        else:
            # assume it's k-means distances 
            return -model.transform(X)

    def _restart(self, model, X_slice):
        if self.gmm:
            model.fit(X_slice, init_params = 'wc')
        else:
            model = sklearn.cluster.KMeans(k = self.n_clusters, init = model.cluster_centers_) 
            model.fit(X_slice)
        return model

    def fit(self, X, Y):
        cs = np.unique(Y)
        self.classes = cs
        self.active_classes = [c for c in cs if c != self.neutral_class]
        n_classes = len(self.classes)
        n_samples, n_dims = X.shape 
        # boolean mask which remove samples from param estimation 
        mask = np.ones(n_samples, dtype='bool')
        
        for curr_iter in xrange(self.n_iters):
            progress = float(curr_iter+1) / self.n_iters
            cutoff = self.cooling ** ( (1-progress)**2) * self.target_precision 
            init_fresh = curr_iter == 0 or (curr_iter < self.n_restarts)
            if self.verbose: 
                print "\t\t*** Iter %d / %d (target precision = %s, init fresh = %s) ***" % \
                    (curr_iter +1, self.n_iters, cutoff, init_fresh)
            recalls = []
                
            for i,c in enumerate(self.active_classes):
                
                class_mask = (Y == c) 
                n_class = np.sum(class_mask)
                
                X_slice = X[class_mask & mask, :]
                n_slice = X_slice.shape[0]
                
                if n_slice == 0:
                    raise RuntimeError("Failed to converge")
                
                if init_fresh:
                    model = self._mk_model()
                    model.fit(X_slice)
                else: 
                    old_model = self.models[i]
                    model = self._restart(old_model, X_slice)
                    
                
                # descending log probs
                thresholds = []
                scores = self._scores(model, X)
                overall_kept = np.zeros(n_samples, dtype='int')
                for cluster_id in xrange(self.n_clusters):
                    cluster_scores = scores[:, cluster_id]
                    precision = None
                    threshold = None
                    recall = 0
                    pred_pos = None
                    sorted_scores = np.sort(cluster_scores)
                    rev_sorted_scores = sorted_scores[::-1]
                    n_scores = len(rev_sorted_scores)
                    score_idx = 0
                    try_doubling = True 
                    while score_idx < n_scores:
                        t = rev_sorted_scores[score_idx]
                        pred_pos = cluster_scores >= t
                        n_correct = np.sum(pred_pos & class_mask)
                        p = float(n_correct) / np.sum(pred_pos)
                        #print score_idx, t, p, try_doubling 
                        # as soon as we get to desired precision, stop'
                        failed = p < cutoff 
                        # if my nearest neighbor is wrong just give up
                        if failed and score_idx == 0:
                            break 
                        elif failed:
                            if try_doubling:
                                try_doubling = False
                            else:
                                break
                        
                        precision = p
                        threshold = t
                        if try_doubling:
                            score_idx = score_idx * 2 + 1
                        else:
                            score_idx -= 1
                    thresholds.append(threshold)
                    if threshold is not None: 
                        correct = (cluster_scores >= threshold) & class_mask
                        recall = float(np.sum(correct)) / n_class
                        overall_kept += (cluster_scores >= threshold)
                    
                    if self.verbose:
                        print "\t\t\tClass %d, cluster %d: threshold = %s, precision = %s, recall = %s" % (c, cluster_id, threshold, precision, recall)
                    
                    
                    keep = overall_kept > 0
                    mask[class_mask & keep] = 1
                    mask[class_mask & ~keep] = 0
                    if curr_iter == 0:
                        self.models.append(model)
                        self.thresholds.append(thresholds)
                    else:
                        self.models[i] = model 
                        self.thresholds[i] = thresholds
                overall_recall = np.sum(overall_kept > 0, dtype='float') / n_class
                recalls.append(overall_recall)
                if self.verbose:
                    print "\t\tOverall recall for class",c, "=", overall_recall  
            hmean = len(recalls) / np.sum(1.0 / np.array(recalls))
            if self.verbose:
                print "\t\tMean recall:", hmean 
        return hmean 
                    
    def predict(self, X):
        n_samples = X.shape[0]
        counts = np.zeros( (n_samples, len(self.active_classes)), dtype='int')
        for i,c in enumerate(self.active_classes):
            model = self.models[i]
            scores = self._scores(model, X)
            thresholds = self.thresholds[i]
            for cluster_id in xrange(self.n_clusters):
                cluster_scores = scores[:, cluster_id]
                t = thresholds[cluster_id] 
                if t is not None: 
                    counts[:, i] += (cluster_scores >= t)
        nonzero_counts = counts > 0
        sum_rows = np.sum(nonzero_counts, axis=1)
        multiple_matches = sum_rows > 1
        n_multiple =  np.sum(multiple_matches)
        if self.verbose or n_multiple > 0:
            print "\t\t\tSamples matching multiple classes:", n_multiple
        pred = np.ones(n_samples, dtype='int') * self.neutral_class
        for i, c in enumerate(self.active_classes):
            pred[counts[:, i] > 0] = c
        pred[multiple_matches] = self.neutral_class
        return pred 
            
                
        
#from sklearn.linear_model import LogisticRegression 
                
#class LogisticThresholding:
    #def __init__(self, 
            #target_precision=0.7,  
            #n_clusters = 5, 
            #neutral_class = 0, 
            #n_iters = 10, 
            #verbose = False,
            #cooling = 1.0):
        #self.neutral_class = neutral_class
        #self.target_precision = target_precision
        #self.n_iters = n_iters 
        #self.n_clusters = n_clusters
        #self.cooling = cooling 
        #self.verbose = verbose 
        
        #self.models = {}
        #self.thresholds = {}
        #self.cluster_assignments = {}
        #self.classes = None
        #self.active_classes = None


    #def _mk_model(self):
        #return LogisticRegression(scale_C = True)      

    #def _scores(self, model, X):
        #return model.predict_proba(X)
        
    
    #def fit(self, X, Y):
        #cs = np.unique(Y)
        #self.classes = cs
        #self.active_classes = [c for c in cs if c != self.neutral_class]
        #n_classes = len(self.classes)
        #n_samples, n_dims = X.shape 
        ## boolean mask which remove samples from param estimation 
        #mask = np.ones(n_samples, dtype='bool')
        
        #for curr_iter in xrange(self.n_iters):
            #progress = float(curr_iter+1) / self.n_iters
            #cutoff = self.cooling ** ( (1-progress)**2) * self.target_precision 
            #init_fresh = curr_iter == 0 or (curr_iter < self.n_restarts)
            #if self.verbose: 
                #print "\t\t*** Iter %d / %d (target precision = %s, init fresh = %s) ***" % \
                    #(curr_iter +1, self.n_iters, cutoff, init_fresh)
            #recalls = []
                
            #for i,c in enumerate(self.active_classes):
                
                #class_mask = (Y == c) 
                #n_class = np.sum(class_mask)
                #X_slice = X[class_mask & mask, :]
                #n_slice = X_slice.shape[0]
        
                #if n_slice == 0:
                    #raise RuntimeError("Failed to converge")
                
                #if init_fresh:
                    #model = self._mk_model()
                    #model.fit(X_slice)
                #else: 
                    #old_model = self.models[i]
                    #model = self._restart(old_model, X_slice)
                    
                
                ## descending log probs
                #thresholds = []
                #scores = self._scores(model, X)
                #overall_kept = np.zeros(n_samples, dtype='int')
                #for cluster_id in xrange(self.n_clusters):
                    #cluster_scores = scores[:, cluster_id]
                    #precision = None
                    #recall = None
                    #threshold = None
                    
                    #pred_pos = None
                    #for counter, t in enumerate(np.sort(cluster_scores)[::-1]):
                        #pred_pos = cluster_scores >= t
                        #n_pred_pos = np.sum(pred_pos)
                        #correct = pred_pos & class_mask
                        #n_correct = np.sum(correct)
                        #p = float(n_correct) / n_pred_pos
                        #r = float(n_correct) / n_class
                        
                        
                        ## as soon as we get to desired precision, stop
                        #if p < cutoff and counter > 10: 
                            #break
                        #elif p >= cutoff:
                            #precision = p
                            #recall = r 
                            #threshold = t 
                    #thresholds.append(threshold)
                    
                    #if threshold is not None: 
                        #overall_kept += (cluster_scores >= threshold)
                    
                    #if self.verbose:
                        #print "\t\t\tClass %d, cluster %d: threshold = %s, precision = %s, recall = %s" % (c, cluster_id, threshold, precision, recall)
                    
                    
                    #keep = overall_kept > 0
                    #mask[class_mask & keep] = 1
                    #mask[class_mask & ~keep] = 0
                    #if curr_iter == 0:
                        #self.models.append(model)
                        #self.thresholds.append(thresholds)
                    #else:
                        #self.models[i] = model 
                        #self.thresholds[i] = thresholds
                #overall_recall = np.sum(overall_kept > 0, dtype='float') / n_class
                #recalls.append(overall_recall)
                #if self.verbose:
                    #print "\t\tOverall recall for class",c, "=", overall_recall  
            #hmean = len(recalls) / np.sum(1.0 / np.array(recalls))
            #if self.verbose:
                #print "\t\tMean recall:", hmean 
        #return hmean 
                    
    #def predict(self, X):
        #n_samples = X.shape[0]
        #counts = np.zeros( (n_samples, len(self.active_classes)), dtype='int')
        #for i,c in enumerate(self.active_classes):
            #model = self.models[i]
            #scores = self._scores(model, X)
            #thresholds = self.thresholds[i]
            #for cluster_id in xrange(self.n_clusters):
                #cluster_scores = scores[:, cluster_id]
                #t = thresholds[cluster_id] 
                #if t is not None: 
                    #counts[:, i] += (cluster_scores >= t)
        #nonzero_counts = counts > 0
        #sum_rows = np.sum(nonzero_counts, axis=1)
        #multiple_matches = sum_rows > 1
        #n_multiple =  np.sum(multiple_matches)
        #if self.verbose or n_multiple > 0:
            #print "\t\t\tSamples matching multiple classes:", n_multiple
        #pred = np.ones(n_samples, dtype='int') * self.neutral_class
        #for i, c in enumerate(self.active_classes):
            #pred[counts[:, i] > 0] = c
        #pred[multiple_matches] = self.neutral_class
        #return pred 
            
    
