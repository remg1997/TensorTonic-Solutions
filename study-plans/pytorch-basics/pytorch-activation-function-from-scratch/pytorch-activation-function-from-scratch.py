import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x)
    if method == "relu":
        return torch.clamp(x, min=0).tolist()
    elif method == "sigmoid":
        return 1/(1+(torch.exp(-1*x)))
    elif method == "tanh":
        posex = torch.exp(x)
        negex = torch.exp(-1*x)
        return (posex - negex)/(posex + negex)
    else:
        return torch.where(x>0, x, 0.01*x)
        