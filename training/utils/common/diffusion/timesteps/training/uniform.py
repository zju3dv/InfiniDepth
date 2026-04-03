from typing import Sequence
import torch

from ..base import TrainingTimesteps


class UniformTrainingTimesteps(TrainingTimesteps):
    """
    Uniform sampling of timesteps in [0, T].
    """

    def sample(
        self,
        size: Sequence[int],
        device: torch.device = "cpu",
    ) -> torch.Tensor:
        t = torch.rand(size=size, device=device).mul_(self.T)
        return t if self.is_continuous() else t.round().int()
