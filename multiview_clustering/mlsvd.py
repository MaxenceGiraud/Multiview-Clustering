import tensorly as tl
import scipy.linalg as la

def MLSVD(A,truncated=False):
    ''' Multi-Linear Singular Value Decomposition

    Parameters
    ----------
    A : Tensor of order n,
        Tensor to decompose
    truncated False or int,
        Whether to compute truncated SVD. Defaults to False.
    Yields
    -------
    [U1,U2,...,Un] : list of matrices,
        List of mode n singular vectors of A

    '''
    U = []
    if truncated is False :
        truncated = -1

    for o in range(len(A.shape)):
        unfolded = tl.unfold(A,o)
        U.append(la.svd(unfolded)[0][:,:truncated])
    
    return U