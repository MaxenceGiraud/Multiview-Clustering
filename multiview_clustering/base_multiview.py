from sklearn.cluster import KMeans
import numpy as np

class BaseMultiview:
    def __init__(self,m,predict_method=KMeans):
        self.m = m
        self.predict_method = predict_method

    
    def _construct_similarity_tensor(self,S):
        # Contruct Similarity Tensor
        A = np.zeros((*S[0].shape,len(S)))
        for i in range(len(S)) :
            A[:,:,i] = S[i]

        return A
    
    def __repr__(self):
        if hasattr(self,'method'):
            add = ' ' + str(self.method)
        else :
            add = ''
        return str(self.__class__.__name__)+ add
    
    def __str__(self):
        return self.__repr__()