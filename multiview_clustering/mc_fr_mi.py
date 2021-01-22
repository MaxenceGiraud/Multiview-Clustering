import numpy as np
import tensorly as tl
import scipy.linalg as la
from sklearn.cluster import KMeans
from .base_multiview import BaseMultiview

class MC_FR_MI(BaseMultiview):
    ''' Multi-view clustering by matrix integration in the Frobenius-norm objective function

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
    def __init__(self,m,method='direct',predict_method=KMeans,max_iter=200):
        self.max_iter = max_iter
        self.method = method 
        assert method in ['direct','full'], "Method name isn't correct"
        super().__init__(m=m,predict_method=predict_method)
    
    def _transform_hooi_direct(self,A):
        # Unfold the tensor
        A_unfold1 = tl.base.unfold(A,0)
        A_unfold2 = tl.base.unfold(A,1)
        A_unfold3 = tl.base.unfold(A,2)


        # Compute MLSVD to init values of U,V,W
        U = la.svd(A_unfold1,full_matrices=False)[0][:,:self.m]
        V = la.svd(A_unfold2,full_matrices=False)[0][:,:self.m]
        W = la.svd(A_unfold3,full_matrices=False)[0][:,0].reshape(-1,1)

        # print(W.shape)

        ## HOOI
        for _ in range(self.max_iter):
            U_old = U

            U = la.svd(A_unfold1 @ (la.kron(V,W)),full_matrices=False)[0][:,:self.m]
            V = la.svd(A_unfold2 @ (la.kron(W,U)),full_matrices=False)[0][:,:self.m]
            W = la.svd(A_unfold3 @ (la.kron(U,V)),full_matrices=False)[0][:,0].reshape(-1,1)

            # print(la.norm(U-U_old))
            if la.norm(U-U_old) < 1e-12 :
                break
        print(la.norm(U-U_old))
        # Normalize rows of U
        U = 1/ la.norm(U,axis=1).reshape(-1,1) * U 
        U = U [:,:self.m] # Truncate
        print(f"norm : {la.norm(U)}")

        return U
    
    def _transform_hooi(self,A):
        # Unfold the tensor
        A_unfold1 = tl.base.unfold(A,0)
        A_unfold3 = tl.base.unfold(A,2)


        # Compute MLSVD to init U
        U = la.svd(A_unfold1,full_matrices=False)[0][:,:self.m]


        ## HOOI
        for _ in range(self.max_iter):
            U_old = U

            W = la.svd(A_unfold3 @ (la.kron(U,U)))[0][:,0]

            # New integration Matrix
            S = np.sum(W * A,axis=2)

            eig_val,eig_vec = la.eig(S) 
            U = eig_vec[:,np.argsort(eig_val)[::-1][:self.m]]  # Take m eigenvec corresponding to m biggest eigen values
            
            # print(la.norm(U-U_old))
            if la.norm(U-U_old) < 1e-12 :
                break
                
        # Normalize rows of U
        U = 1/ la.norm(U,axis=1).reshape(-1,1) * U 
        U = U [:,:self.m] # Truncate

        return abs(U)
    
    def fit_predict(self,S):
        ''' 
        S : list of similarity matrices
        '''

        # Contruct Similarity Tensor
        A = self._construct_similarity_tensor(S)

        if self.method == "direct" :
            U = self._transform_hooi_direct(A)
        elif self.method == "full":
            U = self._transform_hooi(A)

        # Predict clusters
        return self.predict_method(self.m).fit_predict(U)