import sklearn.neighbors 

def oversample(X, Y, factor = 5, majority_class = 0):
    classes = np.unique(Y)
    x_list = []
    nn = sklearn.neighbors.NearestNeighbors(algorithm='brute')
    for c in classes:
        rows = X[Y == c, :] 
        nn.fit(rows)
        neighbors = 

        
