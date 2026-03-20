from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Gaussians:
    means: torch.Tensor          # [B, N, 3]
    harmonics: torch.Tensor      # [B, N, 3, d_sh]
    opacities: torch.Tensor      # [B, N]
    scales: torch.Tensor         # [B, N, 3]
    rotations: torch.Tensor      # [B, N, 4]
    covariances: Optional[torch.Tensor] = None
