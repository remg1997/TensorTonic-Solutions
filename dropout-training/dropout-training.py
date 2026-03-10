import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)
    if rng is None:
        rng = np.random.default_rng()
    randoms = rng.random(x.shape)
    mask = (randoms < 1-p).astype(x.dtype)
    scaling_factor = 1/(1-p)
    final = mask*x*scaling_factor
    return final, mask*scaling_factor