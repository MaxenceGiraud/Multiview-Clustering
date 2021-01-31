import numpy as np
import tensorly as tl
import scipy.linalg as la
from sklearn.cluster import KMeans
from .base_multiview import BaseMultiview

class MC_FR_OI(BaseMultiview):
    ''' Multi-view Clustering based on optimization integration of the Frobenius-norm objective function

    ref : Multi-View Partitioning via Tensor Methods. (2013) Xinhai Liu, Shuiwang Ji, Wolfgang Glänzel, and Bart De Moor, Fellow, IEEE

    Parameters
    -----------
    m : int,
        Number of clusters
    method : str,
        Method used, either mlsdv or hooi    
    predict_method : class,
        Method to compute the clusters after transformation of the data. Default to Kmeans from sklearn.
    max_iter : int,
        Maximum number of iteration of HOOI algorithm. Defaults to 200. Ignore if method is not hooi.
    '''
    def __init__(self,m,method='mlsvd',predict_method=KMeans,max_iter=200):
        self.max_iter = max_iter
        self.method = method 
        assert method in ['mlsvd','hooi'], "Method name isn't correct"
        super().__init__(m=m,predict_method=predict_method)
    
    def _transform_mlsvd(self,A):
        # Unfold the tensor
        A_unfold1 = tl.base.unfold(A,0)

        # Compute truncated SVD
        U = la.svd(A_unfold1)[0][:,:self.m]

        # U is already normalized 
        return U
    
    def _transform_hooi(self,A):
        # Unfold the tensor
        A_unfold1 = tl.base.unfold(A,0)
        A_unfold2 = tl.base.unfold(A,1)
        # A_unfold3 = tl.base.unfold(A,2)

        # Compute truncated MLSVD
        U = la.svd(A_unfold1,full_matrices=False)[0][:,:self.m]
        V = la.svd(A_unfold2,full_matrices=False)[0][:,:self.m]
        W = np.eye(A.shape[2])

        ## HOOI
        for _ in range(self.max_iter):
            U_old = U

            U = la.svd(A_unfold1 @ (la.kron(V,W)))[0][:,:self.m]
            V = la.svd(A_unfold2 @ (la.kron(U_old,W)))[0][:,:self.m]

            if la.norm(U-U_old) < 1e-12 :
                break
                
        # Normalize rows of U
        U = 1/ la.norm(U,axis=1).reshape(-1,1) * U 
        U = U [:,:self.m] # Truncate

        return U
    
    def fit_predict(self,S):
        ''' 
        S : list of similarity matrices
        '''

        # Contruct Similarity Tensor
        A = self._construct_similarity_tensor(S)

        if self.method == "mlsvd" :
            U = self._transform_mlsvd(A)
        elif self.method == "hooi":
            U = self._transform_hooi(A)

        # Predict clusters
        return self.predict_method(self.m).fit_predict(U)