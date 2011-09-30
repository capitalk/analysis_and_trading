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
    
    upper_triangle_count = (n*n - n) / 2 
    result = np.zeros( [m, upper_triangle_count + 2*n ] )

    result[:, 0:n] = Z
    col = n
    for i in xrange(n):
        x = Z[:, i] 
        for j in xrange(0,i+1):
            y = Z[:, j]
            result[:, col] = x * y 
            col += 1
    return result
        
# modifies X
def normalize(X, Xmean=None, Xstd=None, in_place = False):
    if in_place: 
        X -= Xmean
        X /= Xstd
        return X
    else: return (X - Xmean) / Xstd
        
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
    def __init__(self, X_train,  n_centroids=None, whiten=False, products=False):
        self.compute_pairwise_products = products
        if products: X_train = pairwise_products(X_train)
            
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
        X_train_centered /= self.std_
        if whiten or n_centroids is not None: 
            n_random_indices = min(500000, nrows)
            print "[encoder] Reducing size from", nrows, "to", n_random_indices 
            # k-means and PCA are too slow, pull out a subset of the data 
            if nrows > n_random_indices:
                indices = np.arange(nrows)
                np.random.shuffle(indices)
                random_index_subset = indices[0:n_random_indices]
                X_train_centered = X_train_centered[random_index_subset, :] 
                
        if whiten: 
            self.pca = scikits.learn.decomposition.RandomizedPCA(whiten=True, n_components=min(50, nfeatures))
            self.pca.fit(X_train_centered)
        else:
            self.pca = None 
            
        if n_centroids is None:
            self.centroids = None 
        else:
            cluster_inputs = self.pca.transform(X_train_centered) if whiten else X_train_centered 
            cluster_restarts = 3
            cluster_iters = 50
            (self.centroids, label, intertia) = scikits.learn.cluster.k_means(cluster_inputs, n_centroids, max_iter=cluster_iters, n_init=cluster_restarts)
            
        
    # Two possible final steps:
    # 1) "The Importance of Encoding Versus Training with Sparse Coding and Vector Quantization" 
    #    Split the negative and positive inner products. f_i = max(D_i `dot` x, 0), f_i+j = max(-D_i `dot` x, 0)
    # -or-
    # 2) "An Analysis of Single-Layer Networks in Unsupervised Feature Learning"
    #    Triangle activation: max(mean_dist - dist[i], 0)

    def encode(self, X, transformation='triangle', alpha=0.5, validate=True, in_place=False, unit_norm=True):
        if self.compute_pairwise_products: 
            oldshape = X.shape
            X = pairwise_products(X)
            print "Pairwise products", oldshape, "=>", X.shape
            
        Z = normalize(X, self.mean_, self.std_, in_place=in_place)
        if self.pca: Z = self.pca.transform(Z)
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
        if unit_norm:
            norms = np.apply_along_axis(np.linalg.norm, arr=Z, axis=1)
            Z /= np.array([norms]).T
        return Z
