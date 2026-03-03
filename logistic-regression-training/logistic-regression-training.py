import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    num_samples = len(y)
    w = np.zeros((X.shape[-1]))
    b = 0.0

    for step in range(steps):
        forward_prop = np.dot(X,w) + b
        probs = _sigmoid(forward_prop)
        diff = probs-y
        dW = (1/num_samples)*(np.dot(np.transpose(X), diff))
        dB = (1/num_samples)*np.sum(diff)
        w-= lr*dW
        b-=lr*dB

    return (w,b)
    