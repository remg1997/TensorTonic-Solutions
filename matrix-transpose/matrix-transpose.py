import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    num_rows = A.shape[0]
    num_cols = A.shape[1]
    B = np.zeros((num_cols,num_rows))
    for i in range(num_rows):
        B[:,i] = A[i,:]

    
    return B