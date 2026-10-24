import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    c = torch.max(t, dim=dim, keepdim = True).values
    u = torch.exp(t-c)
    v = torch.sum(torch.exp(t-c), dim=dim, keepdim=True)
    return u/v
    
