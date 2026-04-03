import torch


def detectNaN(x: torch.Tensor):
    """Detect whether there is NaN in the tensor. If have, return true."""
    return torch.isnan(x).sum()
