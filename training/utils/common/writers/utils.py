from typing import Tuple
import torch


def normalize(x: torch.Tensor, value_range: Tuple[float, float] = (0.0, 1.0)):
    """
    normalize x to (0, 1)
    """
    return (x.clamp(*value_range) - value_range[0]) / (value_range[1] - value_range[0])
