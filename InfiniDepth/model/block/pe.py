import torch
import torch.nn as nn
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Dict

acc_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else (torch.float16 if torch.cuda.is_available() else torch.float32)
)

POS_EMB_REGISTRY = {}

def register_pos_emb(name):
    def decorator(cls):
        POS_EMB_REGISTRY[name.lower()] = cls
        return cls
    return decorator

        
@register_pos_emb("dct")
class DctPositionEmbedding(nn.Module):
    """
    Only supports 2D separable DCT encoding for query coordinates coords: [B, N, 2]:
      Phi(x,y)[fx,fy] = cos(pi * fx * x) * cos(pi * fy * y) * 1/(1+fx*fy)
    Convention: coords should be in the range [0, 1].
    """
    def __init__(self, max_freqs: int = 8):
        super().__init__()
        self.max_freqs = max_freqs

        freqs = torch.arange(max_freqs).float()              # [F] -> 0..F-1
        fx = freqs.view(-1, 1)                               # [F,1]
        fy = freqs.view(1, -1)                               # [1,F]
        coeffs = (1.0 + fx * fy) ** -1                       # [F,F]

        self.register_buffer("_freqs_1d", freqs, persistent=False)
        self.register_buffer("_coeffs_2d", coeffs, persistent=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [B, N, 2], value range should be [0,1]
        return: [B, N, F^2]
        """
        assert coords.dim() == 3 and coords.size(-1) == 2, "coords must be [B, N, 2]"
        B, N, _ = coords.shape
        device, dtype = coords.device, coords.dtype

        freqs = self._freqs_1d.to(device=device, dtype=dtype)    # [F]
        coeffs = self._coeffs_2d.to(device=device, dtype=dtype)  # [F,F]
        F = freqs.numel() # frequency dimension = max_freqs

        x = coords[..., 0:1]                                     # [B,N,1]
        y = coords[..., 1:2]                                     # [B,N,1]
        dct_x = torch.cos(math.pi * x * freqs.view(1, 1, F))     # [B,N,F]
        dct_y = torch.cos(math.pi * y * freqs.view(1, 1, F))     # [B,N,F]

        out = dct_x.unsqueeze(-1) * dct_y.unsqueeze(-2)          # [B,N,F,F]
        out = out * coeffs.view(1, 1, F, F)                      # [B,N,F,F]
        dct_emb = out.reshape(B, N, F * F)                       # [B,N,F^2]

        return dct_emb                                           # [B,N,F^2]


@register_pos_emb("random")
class RandomPositionEmbedding(nn.Module):
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
        coords = 2 * coords - 1  # [0,1] --> [-1,1], equivalent to align_corners=False after this transform
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
    

@register_pos_emb("rope")
class RoPEPositionEmbedding(nn.Module):
    """2D Rotary Position Embedding with support for continuous coordinates.

    For each coordinate p (can be float), directly compute θ = p * inv_freq, then derive cos/sin.
    """
    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        # Cache the inv_freq vector: key = feature_dim
        self._inv_freq_cache: Dict[int, torch.Tensor] = {}

    def _get_inv_freq(self, dim: int, device: torch.device, dtype: torch.dtype):
        """
        Computes frequency components for rotary embeddings.
        Returns an inv_freq vector of length dim/2, in the form 1 / base_freq^(2i/d)
        """
        if dim not in self._inv_freq_cache:
            # Use frequencies on even dimensions only
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            self._inv_freq_cache[dim] = inv_freq.to(dtype)
        return self._inv_freq_cache[dim]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Rotation: split [u0, u1, u2, u3,...] into two halves and concatenate (-v, u)."""
        D = x.shape[-1]
        x1, x2 = x[..., : D//2], x[..., D//2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope_continuous(
        self, 
        x: torch.Tensor,          # [B, N, d_half]
        pos: torch.Tensor,        # [B, N] floating-point coordinates
        inv_freq: torch.Tensor    # [d_half]
    ) -> torch.Tensor:
        # 1) Compute angles: [B, N, d_half] = outer(pos, inv_freq)
        #    pos.unsqueeze(-1): [B, N, 1], inv_freq.unsqueeze(0): [1, d_half]
        angles = pos.unsqueeze(-1) * inv_freq.unsqueeze(0)
        # 2) Duplicate to double dimension: [B, N, d_half*2]
        angles = torch.cat([angles, angles], dim=-1)

        # 3) Compute cos/sin and expand to [B, N, D]
        cos = angles.cos()
        sin = angles.sin()

        # 4) Apply rotation
        return x * cos + self._rotate_features(x) * sin

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        tokens:    [B, N, dim]   
        positions: [B, N, 2]    continuous coords: (y,x)
        """
        B, N, D = tokens.shape
        assert D % 2 == 0, "Feature dimension must be even"

        assert positions.shape == (B, N, 2), "positions must be [B, N, 2]"

        # Allocate half of the features to each direction
        d_half = D // 2

        # Get the inv_freq vector
        inv_freq = self._get_inv_freq(d_half, tokens.device, tokens.dtype)  # [d_half]
        # Split feature dimension into first and second halves
        tok_v, tok_h = tokens[..., :d_half], tokens[..., d_half:]

        # Apply RoPE separately on y and x directions, positions[0]--> y, positions[1]--> x
        out_v = self._apply_1d_rope_continuous(tok_v, positions[..., 0], inv_freq)
        out_h = self._apply_1d_rope_continuous(tok_h, positions[..., 1], inv_freq)

        return torch.cat([out_v, out_h], dim=-1)


def build_pos_emb(pos_emb_type="nerf", **kwargs):
    pos_emb_type = pos_emb_type.lower()
    if pos_emb_type not in POS_EMB_REGISTRY:
        raise ValueError(f"Unknown pos_emb_type: {pos_emb_type}")
    return POS_EMB_REGISTRY[pos_emb_type](**kwargs)






