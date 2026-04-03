"""
Discrete variance preserving schedule (vp).
"""

import torch
from typing_extensions import Self

from .base import Schedule


class DiscreteVariancePreservingSchedule(Schedule):
    """
    Discrete variance preserving schedule (vp) is originally proposed in DDPM.
    It is also widely used by Stable Diffusion.

        x_t = sqrt(alphas_cumprod[t]) * x_0 + sqrt(1 - alphas_cumprod[t]) * x_T

    The total number of steps T is implicitly defined by len(alphas_cumprod).
    The device and dtype is implicitly defined by alphas_cumprod.device and alphas_cumprod.dtype
    """

    def __init__(self, alphas_cumprod: torch.Tensor):
        assert torch.is_tensor(alphas_cumprod)
        assert alphas_cumprod.ndim == 1
        self.alphas_cumprod = alphas_cumprod.detach().clone()

    @property
    def T(self) -> int:
        return len(self.alphas_cumprod) - 1

    def A(self, t: torch.Tensor) -> torch.Tensor:
        return self.alphas_cumprod[t] ** 0.5

    def B(self, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.alphas_cumprod[t]) ** 0.5

    # ----------------------------------------------------

    def to_zsnr(self) -> Self:
        """
        Return a new schedule with zero terminal SNR.
        Common Diffusion Noise Schedules and Sample Steps are Flawed, algorithm 1
        <https://arxiv.org/pdf/2305.08891.pdf>
        """
        alphas_cumprod_sqrt = self.alphas_cumprod**0.5
        # Store old values.
        alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
        alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (
            alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T
        )
        # Convert back to alphas_cumprod
        alphas_cumprod = alphas_cumprod_sqrt**2
        return DiscreteVariancePreservingSchedule(alphas_cumprod)

    def shift_snr(self, factor: float) -> Self:
        """
        Return a new schedule with shifted snr.
        Simple diffusion: End-to-end diffusion for high resolution images
        <https://arxiv.org/pdf/2301.11093.pdf>
        """
        snr = (self.alphas_cumprod) / (1 - self.alphas_cumprod)
        snr *= factor
        alphas_cumprod = snr / (1 + snr)
        return DiscreteVariancePreservingSchedule(alphas_cumprod)

    # ----------------------------------------------------

    @staticmethod
    def from_betas(betas: torch.Tensor) -> Self:
        """
        Create from betas.
        """
        alphas = 1 - betas
        alphas_cumprod = alphas.cumprod(0)
        return DiscreteVariancePreservingSchedule(alphas_cumprod)

    @staticmethod
    def from_preset(
        name: str,
        steps: int = 1000,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Self:
        if name == "squared_linear":
            betas = (
                torch.linspace(0.00085**0.5, 0.012**0.5, steps, device=device, dtype=dtype) ** 2
            )
        elif name == "linear":
            betas = torch.linspace(0.00085, 0.012, steps, device=device, dtype=dtype)
        else:
            raise NotImplementedError
        return DiscreteVariancePreservingSchedule.from_betas(betas)
