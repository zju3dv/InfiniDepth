"""
MoGe-style losses adapted for sampled patch-based depth estimation.

This module provides losses that work with sampled points from PatchSampleQueryPairs,
adapting MoGe's affine-invariant global and local losses to work with sparse samples
instead of dense depth maps.

NOTE: These losses work directly with depth values, without projecting to 3D point clouds.
"""
from typing import Tuple, Dict, Any, Optional, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from training.utils.moge_utils.geometry_torch import (
    weighted_mean,
    harmonic_mean,
    geometric_mean,
)
from training.utils.moge_utils.alignment import (
    align,
    align_depth_scale,
    align_depth_affine,
    split_batch_fwd,
)


def _smooth(err: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    """Apply Huber-like smoothing to error."""
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def _compute_weight(
    gt: torch.Tensor,
    mask: torch.Tensor,
    geometry_type: str = 'disparity',
    clamp_factor: float = 10.0,
) -> torch.Tensor:
    """
    Compute weight for alignment and loss computation.
    
    For depth: weight = 1/depth (inverse depth weighting, far points have lower weight)
    For disparity: weight = disparity (disparity itself is 1/depth, so just use it directly)
    
    Args:
        gt: Ground truth values of shape (..., N)
        mask: Validity mask of shape (..., N)
        geometry_type: 'depth' or 'disparity'
        clamp_factor: Factor to clamp max weight relative to mean
        
    Returns:
        weight: Weight tensor of shape (..., N)
    """
    if geometry_type == 'depth':
        # Inverse depth weighting: w = 1/z
        # Far points (large depth) get lower weight
        weight = mask.float() / gt.clamp_min(1e-2)
    elif geometry_type == 'disparity':
        # Disparity is already 1/depth, use it directly as weight
        # Far points (small disparity) get lower weight
        weight = mask.float() * gt.clamp_min(1e-5)
    else:
        raise ValueError(f"Unknown geometry_type: {geometry_type}. Expected 'depth' or 'disparity'.")
    
    # Clamp to avoid extreme weights
    weight_mean = weighted_mean(weight, mask, dim=-1, keepdim=True)
    weight = weight.clamp_max(clamp_factor * weight_mean)
    
    return weight


def _subsample_for_alignment(
    pred: torch.Tensor,
    gt: torch.Tensor, 
    mask: torch.Tensor,
    weight: torch.Tensor,
    max_samples: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized subsampling of points for alignment to avoid OOM.
    
    Similar to MoGe's mask_aware_nearest_resize, but for sparse sampled points.
    Uses random scores + topk for fully vectorized random sampling across all batches.
    
    Works for both:
    - Global loss: (B, N) where B is batch size
    - Local loss: (num_patches, patch_pixel_count) 
    
    Args:
        pred: Predicted depth of shape (B, N)
        gt: Ground truth depth of shape (B, N)
        mask: Validity mask of shape (B, N)
        weight: Weight tensor of shape (B, N)
        max_samples: Maximum number of samples per batch/patch for alignment
        
    Returns:
        pred_lr: Subsampled predicted depth of shape (B, max_samples)
        gt_lr: Subsampled ground truth depth of shape (B, max_samples)
        weight_lr: Subsampled weight of shape (B, max_samples)
    """
    B, N = pred.shape
    device = pred.device
    
    if N <= max_samples:
        return pred, gt, weight
    
    # Vectorized random sampling for all batches/patches
    # Generate random scores, set invalid points to -inf so they won't be selected
    rand_scores = torch.rand(B, N, device=device)
    rand_scores = torch.where(mask, rand_scores, torch.tensor(-float('inf'), device=device, dtype=rand_scores.dtype))
    
    # Get top-k indices for each batch/patch (random sampling via random scores + topk)
    _, indices = rand_scores.topk(max_samples, dim=-1)  # (B, max_samples)
    
    # Gather values using the sampled indices
    pred_lr = torch.gather(pred, dim=-1, index=indices)
    gt_lr = torch.gather(gt, dim=-1, index=indices)
    weight_lr = torch.gather(weight, dim=-1, index=indices)
    
    return pred_lr, gt_lr, weight_lr


class MoGeSampledGlobalLoss(nn.Module):
    """
    MoGe-style affine-invariant global loss for sampled depth values.
    
    This loss aligns predicted depth to ground truth depth using scale and shift (affine),
    then computes the weighted L1 loss on aligned depth values directly.
    
    Note: MoGe uses align_points_scale_z_shift for 3D points, which for depth-only case
    is equivalent to affine alignment (scale + shift).
    
    Works with sampled points from PatchSampleQueryPairs or RapidSampleQueryPairs.
    """
    
    def __init__(
        self,
        beta: float = 0.0,
        trunc: float = 1.0,
        sparsity_aware: bool = False,
        align_max_samples: int = 4096,
        geometry_type: str = 'depth',
    ):
        """
        Args:
            beta: Huber smoothing parameter (0 for L1)
            trunc: Truncation threshold for robust alignment
            sparsity_aware: Whether to apply sparsity-aware reweighting
            align_max_samples: Maximum number of samples for alignment (to avoid OOM).
                              Set to 0 or negative to disable subsampling.
            geometry_type: 'depth' or 'disparity'. Controls weight computation:
                          - 'depth': weight = 1/depth (far points have lower weight)
                          - 'disparity': weight = disparity (same effect, disparity is 1/depth)
        """
        super().__init__()
        self.beta = beta
        self.trunc = trunc
        self.sparsity_aware = sparsity_aware
        self.align_max_samples = align_max_samples
        self.geometry_type = geometry_type
    
    def forward(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
        """
        Compute the global affine-invariant loss.
        
        Args:
            pred_depth: Predicted depth values of shape (B, N, 1) or (B, N)
            gt_depth: Ground truth depth values of shape (B, N, 1) or (B, N)
            mask: Validity mask of shape (B, N)
            
        Returns:
            loss: Scalar loss tensor
            misc: Dictionary of metrics
            scale: Estimated scale factor of shape (B,)
        """
        device = pred_depth.device
        dtype = pred_depth.dtype
        
        # Ensure correct shapes
        if pred_depth.dim() == 3 and pred_depth.shape[-1] == 1:
            pred_depth = pred_depth.squeeze(-1)
        if gt_depth.dim() == 3 and gt_depth.shape[-1] == 1:
            gt_depth = gt_depth.squeeze(-1)
        
        B, N = pred_depth.shape
        
        # Compute weight based on geometry type
        weight = _compute_weight(gt_depth, mask, self.geometry_type)
        
        # Subsample for alignment to avoid OOM (similar to MoGe's mask_aware_nearest_resize)
        if self.align_max_samples > 0 and N > self.align_max_samples:
            pred_lr, gt_lr, weight_lr = _subsample_for_alignment(
                pred_depth, gt_depth, mask, weight, self.align_max_samples
            )
        else:
            pred_lr, gt_lr, weight_lr = pred_depth, gt_depth, weight
        
        # Align depth using affine (scale + shift) on subsampled points
        scale, shift = align_depth_affine(pred_lr, gt_lr, weight_lr, trunc=self.trunc)
        
        # Apply alignment
        valid = scale > 0
        pred_depth_aligned = scale[..., None] * pred_depth + shift[..., None]
        
        # Compute loss weight (recompute with valid mask)
        loss_weight = _compute_weight(gt_depth, valid[..., None] & mask, self.geometry_type)
        
        loss = _smooth(
            (pred_depth_aligned - gt_depth).abs() * loss_weight, 
            beta=self.beta
        ).mean(dim=-1)
        
        if self.sparsity_aware:
            sparsity = mask.float().mean(dim=-1)
            loss = loss / (sparsity + 1e-7)
        
        # Compute metrics
        with torch.no_grad():
            err = (pred_depth_aligned.detach() - gt_depth).abs() / gt_depth.clamp_min(1e-5)
            misc = {
                'truncated_error': weighted_mean(err.clamp_max(1.0), mask).item(),
                'delta': weighted_mean((err < 0.1).float(), mask).item(),  # 10% threshold
            }
        
        return loss.mean(), misc, scale.detach()


class MoGeSampledLocalLoss(nn.Module):
    """
    MoGe-style affine-invariant local loss for sampled patches.
    
    This loss operates on patches sampled by PatchSampleQueryPairs, aligning
    each patch independently using affine transformation (scale + shift),
    equivalent to MoGe's align_points_scale_xyz_shift for depth-only case.
    """
    
    def __init__(
        self,
        beta: float = 0.0,
        trunc: float = 1.0,
        sparsity_aware: bool = False,
        min_valid_ratio: float = 0.3,
    ):
        """
        Args:
            beta: Huber smoothing parameter
            trunc: Truncation threshold for robust alignment
            sparsity_aware: Whether to apply sparsity-aware reweighting
            min_valid_ratio: Minimum ratio of valid points in a patch
        """
        super().__init__()
        self.beta = beta
        self.trunc = trunc
        self.sparsity_aware = sparsity_aware
        self.min_valid_ratio = min_valid_ratio
    
    def forward(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        mask: torch.Tensor,
        patch_info: Dict[str, Any],
        global_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the local affine-invariant loss on patches.
        
        Args:
            pred_depth: Predicted depth values of shape (B, N, 1) or (B, N)
            gt_depth: Ground truth depth values of shape (B, N, 1) or (B, N)
            mask: Validity mask of shape (B, N)
            patch_info: Dictionary containing patch structure information
            global_scale: Global scale from global loss, used for filtering bad patches
            
        Returns:
            loss: Scalar loss tensor
            misc: Dictionary of metrics
        """
        device = pred_depth.device
        dtype = pred_depth.dtype
        
        # Ensure correct shapes
        if pred_depth.dim() == 3 and pred_depth.shape[-1] == 1:
            pred_depth = pred_depth.squeeze(-1)
        if gt_depth.dim() == 3 and gt_depth.shape[-1] == 1:
            gt_depth = gt_depth.squeeze(-1)
        
        B, N = pred_depth.shape
        
        # Get patch info
        num_patches = patch_info.get('num_patches', 0)
        patch_pixel_count = patch_info.get('patch_pixel_count', 0)
        
        if isinstance(num_patches, torch.Tensor):
            num_patches = int(num_patches[0].item())
        else:
            num_patches = int(num_patches)
        if isinstance(patch_pixel_count, torch.Tensor):
            patch_pixel_count = int(patch_pixel_count[0].item())
        else:
            patch_pixel_count = int(patch_pixel_count)
        
        if num_patches == 0 or patch_pixel_count == 0:
            return torch.tensor(0.0, device=device, dtype=dtype), {}
        
        # Reshape to patches: (B, num_patches, patch_pixels)
        pred_patches = pred_depth.view(B, num_patches, patch_pixel_count)
        gt_patches = gt_depth.view(B, num_patches, patch_pixel_count)
        mask_patches = mask.view(B, num_patches, patch_pixel_count)
        
        # Compute patch-wise loss
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_patches = 0
        total_err = 0.0
        total_delta = 0.0
        
        for b in range(B):
            batch_global_scale = global_scale[b] if global_scale is not None else None
            
            for p in range(num_patches):
                patch_pred = pred_patches[b, p]  # (patch_pixels,)
                patch_gt = gt_patches[b, p]      # (patch_pixels,)
                patch_mask = mask_patches[b, p]   # (patch_pixels,)
                
                # Skip patches with too few valid points
                valid_ratio = patch_mask.float().mean()
                if valid_ratio < self.min_valid_ratio:
                    continue
                
                # Compute local mean depth for normalization
                patch_mean_depth = patch_gt[patch_mask].mean() if patch_mask.any() else 1.0
                
                # Compute weight
                weight = patch_mask.float() / patch_gt.clamp_min(1e-5)
                
                # Align patch depth using affine (scale + shift)
                local_scale, local_shift = align_depth_affine(
                    patch_pred.unsqueeze(0),
                    patch_gt.unsqueeze(0),
                    weight.unsqueeze(0),
                    trunc=self.trunc
                )
                local_scale = local_scale.squeeze(0)
                local_shift = local_shift.squeeze(0)
                
                # Filter bad patches based on global scale
                if batch_global_scale is not None and batch_global_scale > 0:
                    scale_ratio = local_scale / batch_global_scale
                    if scale_ratio < 0.1 or scale_ratio > 10.0:
                        continue
                
                if local_scale <= 0:
                    continue
                
                # Apply alignment
                aligned_pred = local_scale * patch_pred + local_shift
                
                # Compute loss
                patch_weight = patch_mask.float() / patch_gt.clamp_min(0.1 * patch_mean_depth)
                patch_loss = _smooth(
                    (aligned_pred - patch_gt).abs() * patch_weight,
                    beta=self.beta
                ).mean()
                
                total_loss = total_loss + patch_loss
                total_patches += 1
                
                # Compute metrics
                with torch.no_grad():
                    err = (aligned_pred.detach() - patch_gt).abs() / patch_gt.clamp_min(1e-5)
                    total_err += weighted_mean(err.clamp_max(1), patch_mask).item()
                    total_delta += weighted_mean((err < 0.1).float(), patch_mask).item()
        
        if total_patches > 0:
            loss = total_loss / total_patches
            misc = {
                'truncated_error': total_err / total_patches,
                'delta': total_delta / total_patches,
                'valid_patches': total_patches,
            }
        else:
            loss = torch.tensor(0.0, device=device, dtype=dtype)
            misc = {'valid_patches': 0}
        
        return loss, misc


class MoGeSampledLocalLossVectorized(nn.Module):
    """
    Vectorized MoGe-style local loss for sampled patches.
    
    This is a more efficient implementation that processes all patches in parallel,
    using affine alignment (scale + shift) equivalent to MoGe's align_points_scale_xyz_shift.
    """
    
    def __init__(
        self,
        beta: float = 0.0,
        trunc: float = 1.0,
        sparsity_aware: bool = False,
        min_valid_points: int = 32,
        max_patches_per_batch: int = 64,
        align_max_samples_per_patch: int = 1024,
        geometry_type: str = 'depth',
    ):
        """
        Args:
            beta: Huber smoothing parameter
            trunc: Truncation threshold for robust alignment
            sparsity_aware: Whether to apply sparsity-aware reweighting
            min_valid_points: Minimum number of valid points in a patch (MoGe uses 32)
            max_patches_per_batch: Maximum patches to process at once for alignment (to avoid OOM)
            align_max_samples_per_patch: Maximum samples per patch for alignment.
                                        MoGe uses align_resolution=32 -> 32*32=1024.
                                        Set to 0 to disable subsampling.
            geometry_type: 'depth' or 'disparity'. Controls weight computation.
        """
        super().__init__()
        self.beta = beta
        self.trunc = trunc
        self.sparsity_aware = sparsity_aware
        self.min_valid_points = min_valid_points
        self.max_patches_per_batch = max_patches_per_batch
        self.align_max_samples_per_patch = align_max_samples_per_patch
        self.geometry_type = geometry_type
    
    def forward(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        mask: torch.Tensor,
        patch_info: Dict[str, Any],
        global_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Vectorized computation of local loss.
        """
        device = pred_depth.device
        dtype = pred_depth.dtype
        
        if pred_depth.dim() == 3 and pred_depth.shape[-1] == 1:
            pred_depth = pred_depth.squeeze(-1)
        if gt_depth.dim() == 3 and gt_depth.shape[-1] == 1:
            gt_depth = gt_depth.squeeze(-1)
        
        B, N = pred_depth.shape
        
        num_patches = patch_info.get('num_patches', 0)
        patch_pixel_count = patch_info.get('patch_pixel_count', 0)
        
        if isinstance(num_patches, torch.Tensor):
            num_patches = int(num_patches[0].item())
        else:
            num_patches = int(num_patches)
        if isinstance(patch_pixel_count, torch.Tensor):
            patch_pixel_count = int(patch_pixel_count[0].item())
        else:
            patch_pixel_count = int(patch_pixel_count)
        
        if num_patches == 0 or patch_pixel_count == 0:
            return torch.tensor(0.0, device=device, dtype=dtype), {}
        
        # Reshape: (B, num_patches, patch_pixels)
        pred_patches = pred_depth.view(B, num_patches, patch_pixel_count)
        gt_patches = gt_depth.view(B, num_patches, patch_pixel_count)
        mask_patches = mask.view(B, num_patches, patch_pixel_count)
        
        # Flatten batch and patches: (B * num_patches, patch_pixels)
        total_patches = B * num_patches
        pred_flat = pred_patches.view(total_patches, patch_pixel_count)
        gt_flat = gt_patches.view(total_patches, patch_pixel_count)
        mask_flat = mask_patches.view(total_patches, patch_pixel_count)
        
        # Compute weight for alignment based on geometry type
        weight_for_align = _compute_weight(gt_flat, mask_flat, self.geometry_type)
        
        # Subsample within each patch for alignment (similar to MoGe's mask_aware_nearest_resize)
        if self.align_max_samples_per_patch > 0 and patch_pixel_count > self.align_max_samples_per_patch:
            pred_lr, gt_lr, weight_lr = _subsample_for_alignment(
                pred_flat, gt_flat, mask_flat, weight_for_align, self.align_max_samples_per_patch
            )
        else:
            pred_lr, gt_lr, weight_lr = pred_flat, gt_flat, weight_for_align
        
        # Align patches using affine (scale + shift), with batch splitting to avoid OOM
        if total_patches > self.max_patches_per_batch:
            local_scale, local_shift = split_batch_fwd(
                align_depth_affine, 
                self.max_patches_per_batch, 
                pred_lr, gt_lr, weight_lr, self.trunc
            )
        else:
            local_scale, local_shift = align_depth_affine(pred_lr, gt_lr, weight_lr, trunc=self.trunc)
        
        # Filter invalid patches
        valid_count = mask_flat.sum(dim=-1)
        patch_valid = (local_scale > 0) & (valid_count >= self.min_valid_points)
        
        if global_scale is not None:
            global_scale_expanded = global_scale[:, None].expand(B, num_patches).reshape(total_patches)
            scale_ratio = local_scale / global_scale_expanded.clamp_min(1e-7)
            patch_valid = patch_valid & (scale_ratio > 0.1) & (scale_ratio < 10.0) & (global_scale_expanded > 0)
        
        if not patch_valid.any():
            return torch.tensor(0.0, device=device, dtype=dtype), {'valid_patches': 0}
        
        # Apply alignment (scale + shift)
        aligned_pred = local_scale[:, None] * pred_flat + local_shift[:, None]
        
        # Compute loss weight based on geometry type
        patch_weight = _compute_weight(gt_flat, mask_flat, self.geometry_type)
        
        err_raw = (aligned_pred - gt_flat).abs() * patch_weight
        err_smooth = _smooth(err_raw, beta=self.beta)
        
        # Mean over patch pixels
        patch_loss = err_smooth.mean(dim=-1)  # (total_patches,)
        
        # Only sum valid patches
        total_loss = (patch_loss * patch_valid.float()).sum()
        num_valid = patch_valid.sum()
        
        if num_valid > 0:
            loss = total_loss / num_valid
        else:
            loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        # Metrics
        with torch.no_grad():
            err = (aligned_pred.detach() - gt_flat).abs() / gt_flat.clamp_min(1e-5)
            valid_err = err[patch_valid]
            valid_mask = mask_flat[patch_valid]
            
            if valid_err.numel() > 0:
                misc = {
                    'truncated_error': weighted_mean(valid_err.clamp_max(1), valid_mask).item(),
                    'delta': weighted_mean((valid_err < 0.1).float(), valid_mask).item(),
                    'valid_patches': num_valid.item(),
                }
            else:
                misc = {'valid_patches': 0}
        
        return loss, misc


class MoGeSampledGradientLoss(nn.Module):
    """
    Gradient consistency loss for sampled patches.
    
    Computes the difference between predicted and ground truth depth gradients
    within each patch. This is equivalent to a simplified normal loss without
    3D projection.
    """
    
    def __init__(
        self,
        beta: float = 0.0,
    ):
        """
        Args:
            beta: Huber smoothing parameter
        """
        super().__init__()
        self.beta = beta
    
    def forward(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        mask: torch.Tensor,
        patch_info: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute gradient loss on patches.
        
        The gradients are computed from depth differences within each patch.
        """
        device = pred_depth.device
        dtype = pred_depth.dtype
        
        if pred_depth.dim() == 3 and pred_depth.shape[-1] == 1:
            pred_depth = pred_depth.squeeze(-1)
        if gt_depth.dim() == 3 and gt_depth.shape[-1] == 1:
            gt_depth = gt_depth.squeeze(-1)
        
        B, N = pred_depth.shape
        
        num_patches = patch_info.get('num_patches', 0)
        patch_pixel_count = patch_info.get('patch_pixel_count', 0)
        
        if isinstance(num_patches, torch.Tensor):
            num_patches = int(num_patches[0].item())
        else:
            num_patches = int(num_patches)
        if isinstance(patch_pixel_count, torch.Tensor):
            patch_pixel_count = int(patch_pixel_count[0].item())
        else:
            patch_pixel_count = int(patch_pixel_count)
        
        patch_size = int(math.sqrt(patch_pixel_count))
        
        if num_patches == 0 or patch_size < 2:
            return torch.tensor(0.0, device=device, dtype=dtype), {}
        
        # Reshape to patches: (B, num_patches, patch_size, patch_size)
        pred_patches = pred_depth.view(B, num_patches, patch_size, patch_size)
        gt_patches = gt_depth.view(B, num_patches, patch_size, patch_size)
        mask_patches = mask.view(B, num_patches, patch_size, patch_size)
        
        # Normalize depths for scale-invariance
        mask_sum = mask_patches.float().sum(dim=(-2, -1), keepdim=True).clamp_min(1)
        pred_mean = (pred_patches * mask_patches.float()).sum(dim=(-2, -1), keepdim=True) / mask_sum
        gt_mean = (gt_patches * mask_patches.float()).sum(dim=(-2, -1), keepdim=True) / mask_sum
        
        # Scale pred to match gt mean
        scale = gt_mean / pred_mean.clamp_min(1e-5)
        pred_patches_scaled = pred_patches * scale
        
        # Compute horizontal and vertical gradients
        pred_dx = pred_patches_scaled[:, :, :-1, :] - pred_patches_scaled[:, :, 1:, :]
        pred_dy = pred_patches_scaled[:, :, :, :-1] - pred_patches_scaled[:, :, :, 1:]
        
        gt_dx = gt_patches[:, :, :-1, :] - gt_patches[:, :, 1:, :]
        gt_dy = gt_patches[:, :, :, :-1] - gt_patches[:, :, :, 1:]
        
        mask_dx = mask_patches[:, :, :-1, :] & mask_patches[:, :, 1:, :]
        mask_dy = mask_patches[:, :, :, :-1] & mask_patches[:, :, :, 1:]
        
        # Compute gradient error
        loss_dx = _smooth((pred_dx - gt_dx).abs(), beta=self.beta)
        loss_dy = _smooth((pred_dy - gt_dy).abs(), beta=self.beta)
        
        # Combine losses
        if mask_dx.any():
            loss_x = weighted_mean(loss_dx, mask_dx, dim=(-3, -2, -1)).mean()
        else:
            loss_x = torch.tensor(0.0, device=device, dtype=dtype)
        
        if mask_dy.any():
            loss_y = weighted_mean(loss_dy, mask_dy, dim=(-3, -2, -1)).mean()
        else:
            loss_y = torch.tensor(0.0, device=device, dtype=dtype)
        
        loss = (loss_x + loss_y) / 2
        
        misc = {
            'grad_x_loss': loss_x.item() if isinstance(loss_x, torch.Tensor) else loss_x,
            'grad_y_loss': loss_y.item() if isinstance(loss_y, torch.Tensor) else loss_y,
        }
        
        return loss, misc


class MoGeSampledEdgeLoss(nn.Module):
    """
    Edge consistency loss for sampled patches.
    
    Ensures that depth edges in prediction match ground truth edges.
    This is computed using the ratio of depth differences.
    """
    
    def __init__(
        self,
        beta: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
    
    def forward(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        mask: torch.Tensor,
        patch_info: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute edge loss on patches using depth ratios.
        """
        device = pred_depth.device
        dtype = pred_depth.dtype
        
        if pred_depth.dim() == 3 and pred_depth.shape[-1] == 1:
            pred_depth = pred_depth.squeeze(-1)
        if gt_depth.dim() == 3 and gt_depth.shape[-1] == 1:
            gt_depth = gt_depth.squeeze(-1)
        
        B, N = pred_depth.shape
        
        num_patches = patch_info.get('num_patches', 0)
        patch_pixel_count = patch_info.get('patch_pixel_count', 0)
        
        if isinstance(num_patches, torch.Tensor):
            num_patches = int(num_patches[0].item())
        else:
            num_patches = int(num_patches)
        if isinstance(patch_pixel_count, torch.Tensor):
            patch_pixel_count = int(patch_pixel_count[0].item())
        else:
            patch_pixel_count = int(patch_pixel_count)
        
        patch_size = int(math.sqrt(patch_pixel_count))
        
        if num_patches == 0 or patch_size < 2:
            return torch.tensor(0.0, device=device, dtype=dtype), {}
        
        # Reshape to patches
        pred_patches = pred_depth.view(B, num_patches, patch_size, patch_size)
        gt_patches = gt_depth.view(B, num_patches, patch_size, patch_size)
        mask_patches = mask.view(B, num_patches, patch_size, patch_size)
        
        # Compute depth ratios (scale-invariant)
        pred_ratio_x = pred_patches[:, :, :-1, :] / pred_patches[:, :, 1:, :].clamp_min(1e-5)
        pred_ratio_y = pred_patches[:, :, :, :-1] / pred_patches[:, :, :, 1:].clamp_min(1e-5)
        
        gt_ratio_x = gt_patches[:, :, :-1, :] / gt_patches[:, :, 1:, :].clamp_min(1e-5)
        gt_ratio_y = gt_patches[:, :, :, :-1] / gt_patches[:, :, :, 1:].clamp_min(1e-5)
        
        mask_dx = mask_patches[:, :, :-1, :] & mask_patches[:, :, 1:, :]
        mask_dy = mask_patches[:, :, :, :-1] & mask_patches[:, :, :, 1:]
        
        # Compute ratio error (log scale for symmetry)
        loss_dx = _smooth((pred_ratio_x.log() - gt_ratio_x.log()).abs(), beta=self.beta)
        loss_dy = _smooth((pred_ratio_y.log() - gt_ratio_y.log()).abs(), beta=self.beta)
        
        # Combine losses
        if mask_dx.any():
            loss_x = weighted_mean(loss_dx, mask_dx, dim=(-3, -2, -1)).mean()
        else:
            loss_x = torch.tensor(0.0, device=device, dtype=dtype)
        
        if mask_dy.any():
            loss_y = weighted_mean(loss_dy, mask_dy, dim=(-3, -2, -1)).mean()
        else:
            loss_y = torch.tensor(0.0, device=device, dtype=dtype)
        
        loss = (loss_x + loss_y) / 2
        
        misc = {}
        
        return loss, misc


class MoGeCombinedSampledLoss(nn.Module):
    """
    Combined MoGe-style loss for sampled patches.
    
    Combines global, local, gradient, edge losses with configurable weights.
    All losses work directly on depth values without 3D projection.

    """
    
    def __init__(
        self,
        global_weight: float = 1.0,
        local_weight: float = 0.5,
        gradient_weight: float = 0.1,
        edge_weight: float = 0.1,
        beta: float = 0.0,
        trunc: float = 1.0,
        geometry_type: str = 'disparity',
    ):
        """
        Args:
            global_weight: Weight for global loss
            local_weight: Weight for local loss
            gradient_weight: Weight for gradient loss
            edge_weight: Weight for edge loss
            beta: Huber smoothing parameter
            trunc: Truncation threshold
            geometry_type: 'depth' or 'disparity'. Controls weight computation for global and local losses.
        """
        super().__init__()
        self.global_weight = global_weight
        self.local_weight = local_weight
        self.gradient_weight = gradient_weight
        self.edge_weight = edge_weight
        
        self.global_loss = MoGeSampledGlobalLoss(beta=beta, trunc=trunc, geometry_type=geometry_type)
        self.local_loss = MoGeSampledLocalLossVectorized(beta=beta, trunc=trunc, geometry_type=geometry_type)
        self.gradient_loss = MoGeSampledGradientLoss(beta=beta) if gradient_weight > 0 else None
        self.edge_loss = MoGeSampledEdgeLoss(beta=beta) if edge_weight > 0 else None
    
    def forward(
        self,
        pred_global: torch.Tensor,
        gt_global: torch.Tensor,
        mask_global: torch.Tensor,
        pred_local: torch.Tensor,
        gt_local: torch.Tensor,
        mask_local: torch.Tensor,
        patch_info: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute combined loss with separate global and local inputs.
        
        Args:
            pred_global: Predicted depth for global loss, shape (B, N_global, 1) or (B, N_global)
            gt_global: Ground truth depth for global loss
            mask_global: Validity mask for global loss
            pred_local: Predicted depth for local loss, shape (B, N_local, 1) or (B, N_local)
                       If None, uses pred_global for local loss as well
            gt_local: Ground truth depth for local loss
                     If None, uses gt_global for local loss as well
            mask_local: Validity mask for local loss
                       If None, uses mask_global for local loss as well
            patch_info: Dictionary containing patch structure information (required for local loss)
            
        Returns:
            total_loss: Combined loss tensor
            all_misc: Dictionary of metrics from each loss component
        """
        device = pred_global.device
        dtype = pred_global.dtype
        
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        all_misc = {}
        global_scale = None
        
        # Global loss (on global samples)
        if self.global_weight > 0:
            g_loss, g_misc, global_scale = self.global_loss(
                pred_global, gt_global, mask_global
            )
            total_loss = total_loss + self.global_weight * g_loss
            all_misc['global'] = g_misc
        
        # Local loss (on local/patch samples)
        if self.local_weight > 0 and patch_info is not None:
            l_loss, l_misc = self.local_loss(
                pred_local, gt_local, mask_local, patch_info, global_scale
            )
            total_loss = total_loss + self.local_weight * l_loss
            all_misc['local'] = l_misc
        
        # Gradient loss (on local/patch samples)
        if self.gradient_weight > 0 and self.gradient_loss is not None and patch_info is not None:
            grad_loss, grad_misc = self.gradient_loss(
                pred_local, gt_local, mask_local, patch_info
            )
            total_loss = total_loss + self.gradient_weight * grad_loss
            all_misc['gradient'] = grad_misc
        
        # Edge loss (on local/patch samples)
        if self.edge_weight > 0 and self.edge_loss is not None and patch_info is not None:
            e_loss, e_misc = self.edge_loss(
                pred_local, gt_local, mask_local, patch_info
            )
            total_loss = total_loss + self.edge_weight * e_loss
            all_misc['edge'] = e_misc
        
        return total_loss, all_misc
