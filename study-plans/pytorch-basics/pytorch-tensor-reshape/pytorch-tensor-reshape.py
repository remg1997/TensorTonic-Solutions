import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x= torch.tensor(x, dtype=torch.float16)
    if op == "flatten":
        return torch.flatten(x)
    elif op == "squeeze":
        return torch.squeeze(x)
    elif op == "unsqueeze":
        return torch.unsqueeze(x)
    else:
        return torch.transpose(x, 1, 0)