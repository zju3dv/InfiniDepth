from typing import Any, Optional, Tuple
import numpy as np
import torch
from torch import nn


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    patch_size = 14

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None, image_pe_method: str = "patch") -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

        self.image_pe_method = image_pe_method
        if self.image_pe_method == "image":
            # self.patch_embed = nn.Conv2d(num_pos_feats*2, num_pos_feats*2, kernel_size=self.patch_size, stride=self.patch_size)
            self.patch_embed = nn.Sequential(
                nn.Conv2d(num_pos_feats * 2, num_pos_feats // 2, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(
                    num_pos_feats // 2,
                    num_pos_feats * 2,
                    kernel_size=self.patch_size // 2,
                    stride=self.patch_size // 2,
                ),
            )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_encoding(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))  # HxWx2 -> HxWxC
        return pe.permute(2, 0, 1)  # C x H x W

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:

        if self.image_pe_method == "patch":
            return self.forward_encoding(size)
        elif self.image_pe_method == "image":
            pe_encoding = self.forward_encoding(size)
            pe_encoding_high = self.forward_encoding((size[0] * self.patch_size, size[1] * self.patch_size))
            return pe_encoding + self.patch_embed(pe_encoding_high[None])[0]

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
