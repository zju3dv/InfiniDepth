"""
Geometry utility functions for PyTorch tensors.
Adapted from MoGe project for sampled point-based operations.
"""
from typing import *
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_mean(
    x: torch.Tensor, 
    w: torch.Tensor = None, 
    dim: Union[int, Tuple[int, ...]] = None, 
    keepdim: bool = False, 
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute weighted mean of tensor x with weights w.
    
    Args:
        x: Input tensor
        w: Weight tensor (same shape as x or broadcastable)
        dim: Dimension(s) to reduce
        keepdim: Whether to keep the reduced dimensions
        eps: Small epsilon for numerical stability
        
    Returns:
        Weighted mean of x
    """
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)


def harmonic_mean(
    x: torch.Tensor, 
    w: torch.Tensor = None, 
    dim: Union[int, Tuple[int, ...]] = None, 
    keepdim: bool = False, 
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute harmonic mean of tensor x with optional weights w.
    
    Args:
        x: Input tensor
        w: Weight tensor (same shape as x or broadcastable)
        dim: Dimension(s) to reduce
        keepdim: Whether to keep the reduced dimensions
        eps: Small epsilon for numerical stability
        
    Returns:
        Harmonic mean of x
    """
    if w is None:
        return x.add(eps).reciprocal().mean(dim=dim, keepdim=keepdim).reciprocal()
    else:
        w = w.to(x.dtype)
        return weighted_mean(x.add(eps).reciprocal(), w, dim=dim, keepdim=keepdim, eps=eps).add(eps).reciprocal()


def geometric_mean(
    x: torch.Tensor, 
    w: torch.Tensor = None, 
    dim: Union[int, Tuple[int, ...]] = None, 
    keepdim: bool = False, 
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute geometric mean of tensor x with optional weights w.
    
    Args:
        x: Input tensor
        w: Weight tensor (same shape as x or broadcastable)
        dim: Dimension(s) to reduce
        keepdim: Whether to keep the reduced dimensions
        eps: Small epsilon for numerical stability
        
    Returns:
        Geometric mean of x
    """
    if w is None:
        return x.add(eps).log().mean(dim=dim).exp()
    else:
        w = w.to(x.dtype)
        return weighted_mean(x.add(eps).log(), w, dim=dim, keepdim=keepdim, eps=eps).exp()


def normalized_view_plane_uv(
    width: int, 
    height: int, 
    aspect_ratio: float = None, 
    dtype: torch.dtype = None, 
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate normalized UV coordinates for the view plane.
    UV with left-top corner as (-width / diagonal, -height / diagonal) 
    and right-bottom corner as (width / diagonal, height / diagonal).
    
    Args:
        width: Image width
        height: Image height
        aspect_ratio: Aspect ratio (width/height), computed from width/height if None
        dtype: Output dtype
        device: Output device
        
    Returns:
        UV coordinates tensor of shape (H, W, 2)
    """
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def focal_to_fov(focal: torch.Tensor) -> torch.Tensor:
    """Convert focal length to field of view (in radians)."""
    return 2 * torch.atan(0.5 / focal)


def fov_to_focal(fov: torch.Tensor) -> torch.Tensor:
    """Convert field of view (in radians) to focal length."""
    return 0.5 / torch.tan(fov / 2)


def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute angle difference between two 3D vectors.
    
    Args:
        v1: First vector tensor of shape (..., 3)
        v2: Second vector tensor of shape (..., 3)
        eps: Small epsilon for numerical stability
        
    Returns:
        Angle difference in radians
    """
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))


def coord_to_pixel_indices(
    coord: torch.Tensor,
    height: int,
    width: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert normalized coordinates [-1, 1] to pixel indices.
    
    Args:
        coord: Normalized coordinates tensor of shape (..., 2) where coord[..., 0] is y and coord[..., 1] is x
        height: Image height
        width: Image width
        
    Returns:
        Tuple of (y_indices, x_indices) tensors
    """
    # coord is in range [-1, 1], convert to pixel indices
    y = ((coord[..., 0] + 1) * height / 2 - 0.5).long().clamp(0, height - 1)
    x = ((coord[..., 1] + 1) * width / 2 - 0.5).long().clamp(0, width - 1)
    return y, x


def pixel_indices_to_coord(
    y: torch.Tensor,
    x: torch.Tensor,
    height: int,
    width: int
) -> torch.Tensor:
    """
    Convert pixel indices to normalized coordinates [-1, 1].
    
    Args:
        y: Y indices tensor
        x: X indices tensor
        height: Image height
        width: Image width
        
    Returns:
        Normalized coordinates tensor of shape (..., 2)
    """
    y_norm = (2.0 * (y.float() + 0.5) / height) - 1.0
    x_norm = (2.0 * (x.float() + 0.5) / width) - 1.0
    return torch.stack([y_norm, x_norm], dim=-1)


def sample_points_from_depth_map(
    depth: torch.Tensor,
    coord: torch.Tensor,
    intrinsics: torch.Tensor = None,
    focal: torch.Tensor = None,
    height: int = None,
    width: int = None
) -> torch.Tensor:
    """
    Sample 3D points from depth map at given normalized coordinates.
    
    Args:
        depth: Depth values tensor of shape (B, N, 1) or (B, N)
        coord: Normalized coordinates tensor of shape (B, N, 2)
        intrinsics: Camera intrinsics tensor of shape (B, 3, 3), optional
        focal: Focal length tensor of shape (B,), optional (used if intrinsics is None)
        height: Image height (required if using focal)
        width: Image width (required if using focal)
        
    Returns:
        3D points tensor of shape (B, N, 3)
    """
    if depth.dim() == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)  # (B, N)
    
    B, N = depth.shape
    
    if intrinsics is not None:
        # Use intrinsics matrix
        fx = intrinsics[:, 0, 0]  # (B,)
        fy = intrinsics[:, 1, 1]  # (B,)
        cx = intrinsics[:, 0, 2]  # (B,)
        cy = intrinsics[:, 1, 2]  # (B,)
        
        # Convert normalized coords to pixel coords
        y_pixel = (coord[..., 0] + 1) * height / 2 - 0.5  # (B, N)
        x_pixel = (coord[..., 1] + 1) * width / 2 - 0.5   # (B, N)
        
        # Back-project to 3D
        x_3d = (x_pixel - cx[:, None]) * depth / fx[:, None]
        y_3d = (y_pixel - cy[:, None]) * depth / fy[:, None]
        z_3d = depth
        
    else:
        # Use normalized view plane coordinates
        assert focal is not None and height is not None and width is not None
        
        diagonal = (height ** 2 + width ** 2) ** 0.5
        aspect_ratio = width / height
        
        span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
        span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5
        
        # coord is in [-1, 1], convert to normalized view plane coordinates
        u = coord[..., 1] * span_x  # x direction
        v = coord[..., 0] * span_y  # y direction
        
        x_3d = u * depth / focal[:, None]
        y_3d = v * depth / focal[:, None]
        z_3d = depth
    
    return torch.stack([x_3d, y_3d, z_3d], dim=-1)


def compute_points_from_depth_and_coord(
    depth: torch.Tensor,
    coord: torch.Tensor,
    focal: torch.Tensor,
    height: int,
    width: int
) -> torch.Tensor:
    """
    Compute 3D points from sampled depth values and normalized coordinates.
    
    This is the primary function for converting sampled depth values to 3D points
    for use in MoGe-style losses.
    
    Args:
        depth: Sampled depth values of shape (B, N, 1) or (B, N)
        coord: Normalized coordinates of shape (B, N, 2), where coord[..., 0] is y, coord[..., 1] is x
        focal: Focal length of shape (B,), relative to half diagonal
        height: Original image height
        width: Original image width
        
    Returns:
        3D points of shape (B, N, 3)
    """
    if depth.dim() == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)  # (B, N)
    
    device = depth.device
    dtype = depth.dtype
    B, N = depth.shape
    
    # Compute view plane span
    diagonal = (height ** 2 + width ** 2) ** 0.5
    aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5
    
    # coord[..., 0] is y (in [-1, 1]), coord[..., 1] is x (in [-1, 1])
    # Convert to view plane coordinates
    u = coord[..., 1] * span_x  # x direction
    v = coord[..., 0] * span_y  # y direction
    
    # Compute 3D points: (u/f * z, v/f * z, z)
    # focal is relative to half diagonal, so we need to scale appropriately
    x_3d = u * depth / focal[:, None]
    y_3d = v * depth / focal[:, None]
    z_3d = depth
    
    return torch.stack([x_3d, y_3d, z_3d], dim=-1)
