import numpy as np 
import sklearn.mixture 
class ConfidentClassifier:
	def __init__(self, 
			target_precision=0.7,  
			n_mixture_components = 5, 
			neutral_class = 0, 
			covariance_type = 'full',
			n_iters = 10):
		self.neutral_class = neutral_class
		self.target_precision = target_precision
		self.n_iters = n_iters 
		self.n_mixture_components = n_mixture_components
		self.covariance_type = covariance_type
		self.models = []
		self.thresholds = [] 
		self.classes = None
		self.active_classes = None
		
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
			cutoff = 0.5 ** ( (1-progress)**2) * self.target_precision 
			print "*** Iter %d (discounted precision = %s) ***" % (curr_iter +1, cutoff)
			
			for i,c in enumerate(self.active_classes):
				model = sklearn.mixture.GMM(
					n_components=self.n_mixture_components, 
					cvtype = self.covariance_type)
				
				class_mask = (Y == c) 
				X_slice = X[class_mask & mask, :]
				
				model.fit(X_slice)
				
				
				#precisions = densities.copy()
				#recalls = densities.copy()
				# descending log probs
				thresholds = []
				scores = sklearn.mixture.lmvnpdf(X, model.means, model.covars, self.covariance_type)
				overall_kept = np.zeros(n_samples, dtype='int')
				for cluster_id in xrange(self.n_mixture_components):
					cluster_scores = scores[:, cluster_id]
					precision = None
					recall = None
					threshold = None
					for j, t in enumerate(reversed(np.sort(cluster_scores))):
						pred_pos = cluster_scores >= t
						n_pred_pos = np.sum(pred_pos)
						correct = pred_pos & class_mask
						n_correct = np.sum(correct)
						p = float(n_correct) / n_pred_pos
						r = float(n_correct) / np.sum(class_mask)
						
						if p < cutoff and j > 20: 
							break
						elif p >= cutoff:
							overall_kept += pred_pos
							precision = p
							recall = r 
							threshold = t 
							
					thresholds.append(threshold)
					print "Class %d, cluster %d: threshold = %s, precision = %s, recall = %s" % (c, cluster_id, threshold, precision, recall)
					
					keep = overall_kept > 0
					mask[class_mask & keep] = 1
					mask[class_mask & ~keep] = 0
					if curr_iter == 0:
						self.models.append(model)
						self.thresholds.append(thresholds)
					else:
						self.models[i] = model 
						self.thresholds[i] = thresholds
					
	def predict(self, X):
		n_samples = X.shape[0]
		counts = np.zeros( (n_samples, len(self.active_classes)), dtype='int')
		for i,c in enumerate(self.active_classes):
			model = self.models[i]
			scores = sklearn.mixture.lmvnpdf(X, model.means, model.covars, self.covariance_type)
			thresholds = self.thresholds[i]
			for cluster_id in xrange(self.n_mixture_components):
				cluster_scores = scores[:, cluster_id]
				t = thresholds[cluster_id] 
				if t is not None: 
					counts[:, i] += (cluster_scores >= t)
		nonzero_counts = counts > 0
		sum_rows = np.sum(nonzero_counts, axis=1)
		multiple_matches = sum_rows > 1
		print "Samples matching multiple classes:", np.sum(multiple_matches)
		pred = np.ones(n_samples, dtype='int') * self.neutral_class
		for i, c in enumerate(self.active_classes):
			pred[counts[:, i] > 0] = c
		pred[multiple_matches] = self.neutral_class
		return pred 
			
				
		
				
			
		
		
