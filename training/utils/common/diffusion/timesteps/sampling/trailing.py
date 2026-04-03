import torch

from ...types import SamplingDirection
from ..base import SamplingTimesteps


class UniformTrailingSamplingTimesteps(SamplingTimesteps):
    """
    Uniform trailing sampling timesteps.
    Defined in (https://arxiv.org/abs/2305.08891)
    """

    def __init__(
        self,
        T: int,
        steps: int,
        device: torch.device = "cpu",
    ):
        if isinstance(T, float):
            timesteps = torch.arange(T, 0, -T / steps, device=device)
        else:
            timesteps = torch.arange(T, -1, -(T + 1) / steps, device=device).round().int()

        super().__init__(T=T, timesteps=timesteps, direction=SamplingDirection.backward)
