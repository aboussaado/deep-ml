import torch

def make_tensor():
    """Return a 2x3 float32 tensor [[1, 2, 3], [4, 5, 6]]."""
    data = [list(range(1,4)),list(range(4,7))]
    return torch.tensor(data, dtype = torch.float32)
