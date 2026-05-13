import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    loss = 0
    num = len(y_true)
    for i, j in zip(y_true, y_pred):
        loss += -1*np.log(j[i])
    return loss/num
