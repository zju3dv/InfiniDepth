from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from .projection import get_world_rays
from .types import Gaussians


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float = 1e-10
    gaussian_scale_max: float = 5.0
    sh_degree: int = 2


class GaussianAdapter(nn.Module):
    def __init__(self, cfg: GaussianAdapterCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("sh_mask", torch.ones((self.d_sh,), dtype=torch.float32), persistent=False)
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * (0.25**degree)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh

    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        coordinates_xy: torch.Tensor,
        depths: torch.Tensor,
        opacities: torch.Tensor,
        raw_gaussians: torch.Tensor,
    ) -> Gaussians:
        """Build world-space gaussians from per-point raw parameters.

        image: [B, 3, H, W]
        extrinsics: [B, 4, 4] camera-to-world
        intrinsics: [B, 3, 3] in pixel units
        coordinates_xy: [B, N, 2] pixel-space (x, y)
        depths: [B, N]
        opacities: [B, N]
        raw_gaussians: [B, N, 7 + 3*d_sh]
        """
        b, _, h, w = image.shape
        scales_raw, rotations_raw, sh_raw = torch.split(raw_gaussians, [3, 4, 3 * self.d_sh], dim=-1)
        scales = torch.clamp(
            F.softplus(scales_raw - 4.0),
            min=self.cfg.gaussian_scale_min,
            max=self.cfg.gaussian_scale_max,
        )
        rotations = rotations_raw / (torch.norm(rotations_raw, dim=-1, keepdim=True) + 1e-8)

        harmonics = sh_raw.view(b, -1, 3, self.d_sh) * self.sh_mask.view(1, 1, 1, -1)

        image = rearrange(image, "b c h w -> b (h w) c")
        harmonics[..., 0] = harmonics[..., 0] + rgb_to_sh(image)

        origins, directions = get_world_rays(
            coordinates_xy,
            extrinsics,
            intrinsics,
        )
        means = origins + directions * depths.unsqueeze(-1)

        return Gaussians(
            means=means,
            harmonics=harmonics,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            covariances=None,
        )
