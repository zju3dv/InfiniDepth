"""
Average accumulator for logging.
"""

from typing import Dict
import torch
from torch.distributed import ReduceOp, all_reduce, is_initialized


class AverageAccumulator:
    """
    Accumulate average values over multiple iterations.
    This is intended for loss logging purposes.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the accumulator.
        """
        self.sum = {}
        self.num = {}

    @torch.no_grad()
    def add(self, **kwargs: Dict[str, torch.Tensor]):
        """
        Add value to the accumulator.
        """
        for k, v in kwargs.items():
            assert torch.is_tensor(v)
            self.sum[k] = self.sum.get(k, 0) + v
            self.num[k] = self.num.get(k, 0) + 1

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Get accumulated average.
        """
        return {k: self.sum[k] / self.num[k] for k in self.sum.keys()}

    def get_and_reset(self) -> Dict[str, torch.Tensor]:
        """
        Get accumulated average and reset.
        """
        val = self.get()
        self.reset()
        return val


class DistributedAverageAccumulator(AverageAccumulator):
    """
    Accumulate average values over multiple iterations and over all GPUs.
    This is intended for loss logging purposes.
    The distributed accumulator must be instantiated on all GPU ranks.
    The method "get" and "get_and_reset" must be invoked on all GPU ranks.
    """

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Get accumulated average from all ranks.
        """
        # Instead of all_reduce on values individually,
        # we merge values to a big tensor and only reduce once together.
        result = super().get().items()
        tensor = torch.cat([v.view(-1) for _, v in result])
        if is_initialized():
            all_reduce(tensor, op=ReduceOp.AVG)
        # Split back to original shapes.
        tensor = tensor.split([v.numel() for _, v in result])
        result = {k: t.reshape_as(v) for t, (k, v) in zip(tensor, result)}
        return result
