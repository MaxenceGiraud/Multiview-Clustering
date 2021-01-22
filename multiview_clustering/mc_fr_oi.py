import numpy as np
import tensorly as tl
import scipy.linalg as la
from sklearn.cluster import KMeans
from .base_multiview import BaseMultiview

class MC_FR_OI_MLSVD(BaseMultiview):
    ''' Multi-view Clustering based on optimization integration of the Frobenius-norm objective function using ML-SVD to solve the problem

    ref : Multi-View Partitioning via Tensor Methods. (2013) Xinhai Liu, Shuiwang Ji, Wolfgang Glänzel, and Bart De Moor, Fellow, IEEE

    Parameters
    -----------
    m : int,
        Number of clusters
    predict_method : class,
        Method to compute the clusters after transformation of the data. Default to Kmeans from sklearn.
    '''
    
    def fit_predict(self,S):
        ''' 
        S : list of similarity matrices
        '''

        # Contruct Similarity Tensor
        A = self._construct_similarity_tensor(S)

        # Unfold the tensor
        A_unfold1 = tl.base.unfold(A,0)

        # Compute truncated SVD
        U = la.svd(A_unfold1)[0][:,:self.m]

        # U is already normalized 

        # Predict clusters
        return self.predict_method(self.m).fit_predict(U)