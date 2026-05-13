import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x, y = np.array(x), np.array(y)
    return float(np.sqrt(np.sum((x-y)**2)))