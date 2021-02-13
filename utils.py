from sklearn.cluster import KMeans
import numpy as np

class SpectralClustering:
    ''' Minimal implementation of Spectral clustering with normalized laplacian and precomputed similarity matrix '''
    def __init__(self,k):
        self.k = k
    
    def fit_predict(self,X):
        D = np.diag(np.sum(X,axis=1))
        L =  D - X
        Dinv_sqrt = np.sqrt(np.linalg.pinv(D))
        L = np.eye(X.shape[0]) - Dinv_sqrt @ X @ Dinv_sqrt

        # EigenDecomposition
        eig_val,eig_vec = np.linalg.eig(L)

        # Take eigenvector corresponding to k smallest eigenvalues
        Xnew = eig_vec[:,np.argsort(eig_val)[:2]].real

        k = KMeans(3)
        y_hat = k.fit_predict(Xnew)

        return y_hat
    
    def __repr__(self):
        return str(self.__class__.__name__)
    
    def __str__(self):
        return self.__repr__()