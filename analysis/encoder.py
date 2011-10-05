import numpy as np
import scipy 
#import scikits.learn
#from scikits.learn import * 
import sklearn 
import sklearn.cluster 
import sklearn.decomposition

from analysis import check_data 

def mean(X):
    return np.mean(X, axis=0)

def std(X, centered=False):
    if centered:
        return np.sqrt(np.mean(X*X, axis=0))
    else:
        return np.std(X, axis=0)

# computes upper triangle of column products
# excludes original data and diagonal (original data squared) 
def pairwise_products(Z):
    m = Z.shape[0] 
    n = Z.shape[1]
    upper_triangle_count = (n*n - n) / 2 
    result = np.zeros( [m, upper_triangle_count ] ) # + 2*n
    col = 0
    for i in xrange(n):
        x = Z[:, i] 
        for j in xrange(0,i):
            y = Z[:, j]
            result[:, col] = x * y 
            col += 1
    return result

def triangle(dists):
    """triangle activation function, only "greater than average" distances allowed to be active"""
    mean_dists = np.mean(dists, axis=1, dtype='float')
    mean_dists_col = np.array([mean_dists]).T
    return np.maximum(mean_dists_col - dists , 0)
    

# modifies X
def normalize(X, Xmean=None, Xstd=None, in_place = False):
    if in_place: 
        X -= Xmean
        X /= Xstd
        return X
    else: return (X - Xmean) / Xstd

def unit_norm_rows(X, in_place=False):
    norms = np.apply_along_axis(np.linalg.norm, arr=X, axis=1)
    col = np.array([norms]).T
    if in_place: X /= col
    else: X = X / col
    return X 
    
def bin_negatives(X):
    nrows = X.shape[0]
    ncols = X.shape[1]
    result = np.zeros([nrows, 2*ncols])
    for colidx in xrange(ncols):
        col = X[:, colidx]
        is_neg = col < 0
        result[:, colidx] = col
        result[is_neg, colidx] = 0
        result[:, ncols+colidx] = -1*col
        result[~is_neg, ncols+colidx] = 0
    return result 
    
class FeatureEncoder():
    def __getstate__(self): 
        return self.__dict__
        
    def __setstate__(self, state):
        self.__dict__ = state 
  
    def __init__(self, dictionary_type=None, dictionary_size = 25, pca_type=None, pca_size = 25, compute_pairwise_products=False, binning=False, unit_norm=False):
        """Options:
            dictionary_type = None | 'kmeans' | 'sparse' 
            dictionary_size = None | int 
            pca_type = None | 'sparse' | 'whiten' 
            compute_pairwise_products = False | True
            binning = False | True
        """
        self.mean_ = None
        self.std_ = None 

        
        self.dictionary_type = dictionary_type
        self.dictionary_size = dictionary_size 
        self.dictionary = None 
        
        self.pca_type = pca_type
        self.pca_size = pca_size 
        self.pca = None 
        
        self.compute_pairwise_products = compute_pairwise_products
        self.binning = binning 
        self.unit_norm = unit_norm 
    
    def fit_transform(self, X, in_place=True):
        nrows = X.shape[0]
        # When normalizing the training data, 
        # save the mean and std vectors so we can normalize 
        # test data using same params.
        # Similarly, save the PCA matrix and centroids so that
        # test data can be whitened and encoded. 
        self.mean_ = mean(X)
        if in_place: X -= self.mean_
        else: X = X - self.mean_
        
        self.std_ = std(X, centered=True)
        X /= self.std_
        
        if self.compute_pairwise_products: X = pairwise_products(X)
            
        nfeatures = X.shape[0] 
        if self.pca_type is None:
            self.pca = None 
        elif self.pca_type == 'whiten': 
            print "Performing randomized PCA..."
            self.pca = sklearn.decomposition.RandomizedPCA(whiten=True, n_components=min(self.pca_size, nfeatures))
            # since we normalize ourselves, avoid redundant work 
            self.pca.mean_ = None 
            X = self.pca.fit_transform(X)
        elif self.pca_type == 'sparse':
            print "Performing sparse PCA..."
            self.pca = sklearn.decomposition.MiniBatchSparsePCA(min(self.pca_size, nfeatures))
            X = self.pca.fit_transform(X)
        else:
            raise RuntimeError("Unknown PCA type: " + self.pca_type)    
        if self.dictionary_type is None:
            self.dictionary = None
        elif self.dictionary_type == 'kmeans':
            print "Running k-means..."
            self.dictionary = sklearn.cluster.MiniBatchKMeans( self.dictionary_size, init='k-means++')
            # bug in sklearn doesn't clear this field
            self.dictionary.labels_  = None 
            self.dictionary.fit(X)
            dists = self.dictionary.transform(X)
            X = triangle(dists)
        elif self.dictionary_type == 'sparse':
            self.dictionary = sklearn.decomposition.MiniBatchDictionaryLearning(n_atoms=self.dictionary_size, fit_algorithm='cd', transform_algorithm='threshold', transform_alpha=0.25 )
            X = self.dictionary.fit_transform(X)
        else: 
            raise RuntimeError("Unknown dictionary type: " + self.dictionary_type)
            
        if self.unit_norm: X = unit_norm_rows(X, in_place=in_place)
        if self.binning: X = bin_negatives(X)
        return X
        
    # Two possible final steps:
    # 1) "The Importance of Encoding Versus Training with Sparse Coding and Vector Quantization" 
    #    Split the negative and positive inner products. f_i = max(D_i `dot` x, 0), f_i+j = max(-D_i `dot` x, 0)
    # -or-
    # 2) "An Analysis of Single-Layer Networks in Unsupervised Feature Learning"
    #    Triangle activation: max(mean_dist - dist[i], 0)

    def transform(self, X, validate=False, in_place=True):
        X = normalize(X, self.mean_, self.std_, in_place=in_place)
        
        if self.compute_pairwise_products: 
            old_shape = X.shape 
            X = pairwise_products(X)
            print "Pairwise products", old_shape, "=>", X.shape
            
        if self.pca is None: 
            print "[encoder] No PCA matrix found"
        else:
            old_shape = X.shape 
            X = self.pca.transform(X)
            print "PCA: ", old_shape, "=>", X.shape 
            
        if self.dictionary is None:
            print "[encoder] No feature dictionary"
        elif self.dictionary_type == 'kmeans':
            # triangle dist from centroid, with ~50% set to zero 
            print "Computing distances..."
            dists = self.dictionary.transform(X)
            if validate: check_data(dists)
            X = triangle(dists)
            print "Transform by feature dictionary, new shape:", X.shape
        else:
            print "Transforming to dictionary feature space..." 
            X = self.dictionary.transform(X)
        
        if self.unit_norm: X = unit_norm_rows(X, in_place=in_place)

        if self.binning:
            old_shape = X.shape 
            X = bin_negatives(X)
            print "Bin negatives, ", old_shape, "=>", X.shape             
        return X
