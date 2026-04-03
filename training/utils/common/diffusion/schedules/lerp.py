"""
Linear interpolation schedule (lerp).
"""

from typing import Union
import torch

from .base import Schedule


class LinearInterpolationSchedule(Schedule):
    """
    Linear interpolation schedule (lerp) is proposed by flow matching and rectified flow.
    It leads to straighter probability flow theoretically. It is also used by Stable Diffusion 3.
    <https://arxiv.org/abs/2209.03003>
    <https://arxiv.org/abs/2210.02747>

        x_t = (1 - t) * x_0 + t * x_T

    Can be either continuous or discrete.
    """

    def __init__(self, T: Union[int, float] = 1.0):
        self._T = T

    @property
    def T(self) -> Union[int, float]:
        return self._T

    def A(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - (t / self.T)

    def B(self, t: torch.Tensor) -> torch.Tensor:
        return t / self.T

    # ----------------------------------------------------

    def isnr(self, snr: torch.Tensor) -> torch.Tensor:
        t = self.T / (1 + snr**0.5)
        t = t if self.is_continuous() else t.round().int()
        return t
