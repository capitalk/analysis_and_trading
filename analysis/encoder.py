import numpy as np
import scipy 
import scikits.learn
from scikits.learn import * 
from analysis import check_data 

def mean(X):
    return np.mean(X, axis=0)

def std(X, centered=False):
    if centered:
        return np.sqrt(np.mean(X*X, axis=0))
    else:
        return np.std(X, axis=0)

def pairwise_products(Z):
    m = Z.shape[0] 
    n = Z.shape[1]
    # pairwise products and original data 
    prods = np.zeros( [m, n*n + n] )
    prods[:, 0:n] = Z
    for i in xrange(n):
        for j in xrange(n):
            prods[:, i*n+j] = Z[:, i] * Z[:, j]
    return prods
        
# modifies X
def normalize(X, Xmean=None, Xstd=None):
    if Xmean is None: Xmean = mean(X)
    X_centered =  X - Xmean
    if Xstd is None: Xstd = std(X_centered, centered=True)
    X_normalized = X_centered  / Xstd
    return X_normalized

class FeatureEncoder():
    def __getstate__(self): 
        return {
            'mean': self.mean_, 
            'std': self.std_, 
            'centroids': self.centroids, 
            'pca': self.pca
        }
    def __setstate__(self, state):
        self.mean_ = state['mean']
        self.std_ = state['std']
        self.centroids = state['centroids']
        self.pca = state['pca']
  
    # if ncentroids = None, then don't cluster inputs
    def __init__(self, X_train,  n_centroids=None, whiten=False):
        #if products: 
        #    X_train = pairwise_products(X_train)
            
        nrows = X_train.shape[0]
        nfeatures = X_train.shape[1]
        
        # When normalizing the training data, 
        # save the mean and std vectors so we can normalize 
        # test data using same params.
        # Similarly, save the PCA matrix and centroids so that
        # test data can be whitened and encoded. 
        self.mean_ = mean(X_train)
        X_train_centered = X_train - self.mean_
        self.std_ = std(X_train_centered, centered=True)
        if whiten: 
            X_train_normalized = X_train_centered / self.std_
            self.pca = scikits.learn.decomposition.RandomizedPCA(whiten=True, n_components=nfeatures)
            self.pca.fit(X_train_normalized)
        else:
            self.pca = None 
            
        if n_centroids is None:
            self.centroids = None 
        else:
            if whiten:
                cluster_inputs = self.pca.transform(X_train_normalized)
            else:
                cluster_inputs = X_train_centered / self.std_
            
            cluster_restarts = 3
            cluster_iters = 50
            n_random_indices = min(100000, nrows)
            # k-means too slow, pull out a subset of the data 
            if nrows > n_random_indices:
                indices = np.arange(nrows)
                np.random.shuffle(indices)
                random_index_subset = indices[0:n_random_indices]
                cluster_inputs = cluster_inputs[random_index_subset, :] 
            (self.centroids, label, intertia) = scikits.learn.cluster.k_means(cluster_inputs, n_centroids, max_iter=cluster_iters, n_init=cluster_restarts)
            
        
    # Two possible final steps:
    # 1) "The Importance of Encoding Versus Training with Sparse Coding and Vector Quantization" 
    #    Split the negative and positive inner products. f_i = max(D_i `dot` x, 0), f_i+j = max(-D_i `dot` x, 0)
    # -or-
    # 2) "An Analysis of Single-Layer Networks in Unsupervised Feature Learning"
    #    Triangle activation: max(mean_dist - dist[i], 0)

    def encode(self, X, transformation='triangle', alpha=0.5, validate=True):
        Z = normalize(X, self.mean_, self.std_)
        if self.pca:
            Z = self.pca.transform(Z)
        if self.centroids is not None:
            # dist from centroid, with ~50% set to zero 
            if transformation == 'triangle':
                dists = scipy.spatial.distance.cdist(Z, self.centroids)
                if validate: check_data(dists)
                mean_dists = np.mean(dists, axis=1, dtype='float')
                mean_dists_col = np.array([mean_dists]).T
                #only "greater than average" distances allowed to be active
                Z = np.maximum(mean_dists_col - dists , 0)
            # probability distribution over centroids 
            elif transformation == 'prob':
                dists = scipy.spatial.distance.cdist(Z, self.centroids, 'sqeuclidean')
                if validate: check_data(dists) 
                sims = np.exp(-dists)
                del dists
                row_sums = np.sum(sims, axis = 1)

                Z = sims / np.array([row_sums]).T
            # thresholded inner product with centroids 
            elif transformation == 'thresh':
                inner_products = np.dot(Z, self.centroids.T)
                if validate: check_data(inner_products)
                Z = np.maximum(inner_products - alpha, 0)
        return Z
