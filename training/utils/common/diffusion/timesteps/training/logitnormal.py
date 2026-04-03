from typing import Sequence, Union
import torch
from torch.distributions import LogisticNormal

from ..base import TrainingTimesteps


class LogitNormalTrainingTimesteps(TrainingTimesteps):
    """
    Logit-Normal sampling of timesteps in [0, T].
    """

    def __init__(self, T: Union[int, float], loc: float, scale: float):
        super().__init__(T)
        self.dist = LogisticNormal(loc, scale)

    def sample(
        self,
        size: Sequence[int],
        device: torch.device = "cpu",
    ) -> torch.Tensor:
        t = self.dist.sample(size)[..., 0].to(device).mul_(self.T)
        return t if self.is_continuous() else t.round().int()
