import math
import os
from typing import Optional

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from InfiniDepth.gs import Gaussians
from InfiniDepth.gs.projection import homogenize_points, transform_cam2world, unproject


def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.clamp(torch.norm(v), min=eps)


def _look_at_c2w(position: torch.Tensor, target: torch.Tensor, up_hint: torch.Tensor) -> torch.Tensor:
    forward = _safe_normalize(target - position)
    # Camera basis is stored as [right, up, forward]. Using cross(forward, up)
    # flips the x-axis and produces a horizontally mirrored render. Keep the
    # original up hint and derive a right-handed basis instead.
    right = torch.cross(up_hint, forward, dim=0)
    if torch.norm(right) < 1e-6:
        right = torch.cross(
            torch.tensor([1.0, 0.0, 0.0], device=position.device, dtype=position.dtype),
            forward,
            dim=0,
        )
    right = _safe_normalize(right)
    up = _safe_normalize(torch.cross(forward, right, dim=0))

    c2w = torch.eye(4, device=position.device, dtype=position.dtype)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    return c2w


def _build_orbit_poses(
    base_c2w: torch.Tensor,
    target: torch.Tensor,
    num_frames: int,
    radius: float,
    vertical: float,
    forward_amp: float,
) -> list[torch.Tensor]:
    base_pos = base_c2w[:3, 3]
    right = base_c2w[:3, 0]
    up = base_c2w[:3, 1]
    forward = base_c2w[:3, 2]

    poses: list[torch.Tensor] = []
    n = max(2, int(num_frames))
    for i in range(n):
        theta = 2.0 * math.pi * float(i) / float(n)
        offset = (
            right * (radius * math.sin(theta))
            + up * (vertical * math.sin(2.0 * theta))
            + forward * (forward_amp * 0.5 * (1.0 - math.cos(theta)))
        )
        pos = base_pos + offset
        poses.append(_look_at_c2w(pos, target, up))
    return poses


def _build_swing_poses(
    base_c2w: torch.Tensor,
    num_frames: int,
    radius: float,
    forward_amp: float,
) -> list[torch.Tensor]:
    base_pos = base_c2w[:3, 3]
    right = base_c2w[:3, 0]
    forward = base_c2w[:3, 2]

    key_offsets = [
        torch.zeros(3, device=base_pos.device, dtype=base_pos.dtype),
        -right * radius,
        right * radius,
        forward * forward_amp,
        torch.zeros(3, device=base_pos.device, dtype=base_pos.dtype),
    ]

    poses: list[torch.Tensor] = []
    seg_frames = max(1, int(num_frames) // (len(key_offsets) - 1))
    for seg in range(len(key_offsets) - 1):
        p0 = base_pos + key_offsets[seg]
        p1 = base_pos + key_offsets[seg + 1]
        for i in range(seg_frames):
            alpha = 1.0 if seg_frames == 1 else float(i) / float(seg_frames - 1)
            pos = (1.0 - alpha) * p0 + alpha * p1
            pose = base_c2w.clone()
            pose[:3, 3] = pos
            if seg > 0 and i == 0:
                continue
            poses.append(pose)
    return poses


def _scale_intrinsics_for_render(
    intrinsics: torch.Tensor,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> torch.Tensor:
    scaled = intrinsics.clone()
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    scaled[0, 0] *= sx
    scaled[1, 1] *= sy
    scaled[0, 2] *= sx
    scaled[1, 2] *= sy
    return scaled


def _resolve_video_render_size(render_h: int, render_w: int) -> tuple[int, int]:
    """Make render size compatible with libx264/yuv420p encoding."""
    safe_h = max(2, int(render_h) - (int(render_h) % 2))
    safe_w = max(2, int(render_w) - (int(render_w) % 2))
    return safe_h, safe_w


def _render_gaussian_frame(
    rasterization_fn,
    means: torch.Tensor,
    harmonics: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    render_h: int,
    render_w: int,
    bg_color: tuple[float, float, float],
) -> np.ndarray:
    xyzs = means.unsqueeze(0).float()  # [1, N, 3]
    opacitys = opacities.unsqueeze(0).float()  # [1, N]
    rotations_b = rotations.unsqueeze(0).float()  # [1, N, 4]
    scales_b = scales.unsqueeze(0).float()  # [1, N, 3]

    # [N, 3, d_sh] -> [1, N, d_sh, 3]
    features = harmonics.unsqueeze(0).permute(0, 1, 3, 2).contiguous().float()
    d_sh = features.shape[-2]
    sh_degree = int(round(math.sqrt(float(d_sh)) - 1.0))

    w2c = torch.linalg.inv(c2w).unsqueeze(0).unsqueeze(0).float()  # [1, 1, 4, 4]
    Ks = intrinsics.unsqueeze(0).unsqueeze(0).float()  # [1, 1, 3, 3]
    backgrounds = torch.tensor(bg_color, dtype=torch.float32, device=xyzs.device).view(1, 1, 3)

    rendering, _, _ = rasterization_fn(
        xyzs,
        rotations_b,
        scales_b,
        opacitys,
        features,
        w2c,
        Ks,
        render_w,
        render_h,
        sh_degree=sh_degree,
        render_mode="RGB+D",
        packed=False,
        backgrounds=backgrounds,
        covars=None,
        eps2d=1e-8,
    )

    rgb = rendering[0, 0, :, :, :3].clamp(0.0, 1.0)
    return (rgb * 255.0).to(torch.uint8).cpu().numpy()


def _render_novel_video(
    means: torch.Tensor,
    harmonics: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    base_c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    render_h: int,
    render_w: int,
    video_path: str,
    trajectory: str,
    num_frames: int,
    fps: int,
    radius: float,
    vertical: float,
    forward_amp: float,
    bg_color: tuple[float, float, float],
) -> None:
    try:
        from gsplat import rasterization as rasterization_fn
    except ImportError as exc:
        raise RuntimeError("Novel-view rendering requires gsplat. Please install gsplat first.") from exc

    target = means.mean(dim=0)
    if trajectory == "swing":
        poses = _build_swing_poses(base_c2w, num_frames, radius, forward_amp)
    else:
        poses = _build_orbit_poses(base_c2w, target, num_frames, radius, vertical, forward_amp)

    video_dir = os.path.dirname(video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    try:
        with imageio.get_writer(
            video_path,
            fps=float(max(1, fps)),
            codec="libx264",
            macro_block_size=1,
        ) as writer:
            for pose in poses:
                frame_rgb = _render_gaussian_frame(
                    rasterization_fn=rasterization_fn,
                    means=means,
                    harmonics=harmonics,
                    opacities=opacities,
                    scales=scales,
                    rotations=rotations,
                    c2w=pose,
                    intrinsics=intrinsics,
                    render_h=render_h,
                    render_w=render_w,
                    bg_color=bg_color,
                )
                writer.append_data(frame_rgb)
    except Exception as exc:
        raise RuntimeError(f"Failed to write video with imageio: {video_path}") from exc


def _build_sparse_uniform_gaussians(
    dense_gaussians,
    query_3d_uniform_coord: torch.Tensor,
    pred_depth_3d: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    h: int,
    w: int,
) -> Gaussians:
    """Convert dense pixel gaussians to sparse 3d-uniform gaussians.
    """
    if dense_gaussians.means.shape[0] != 1:
        raise ValueError("Current strict-aligned sparse interpolation only supports batch size 1.")

    sparse_coords_normalized = query_3d_uniform_coord[0]  # [N,2], [y,x]
    sparse_depths = pred_depth_3d[0]  # [N,1]

    # Convert normalized coordinates to pixel coordinates
    p_y = ((sparse_coords_normalized[:, 0] + 1.0) * (h / 2.0)) - 0.5
    p_x = ((sparse_coords_normalized[:, 1] + 1.0) * (w / 2.0)) - 0.5
    xy_coords = torch.stack([p_x, p_y], dim=-1)  # [N,2], [x,y]

    depth_values = sparse_depths.squeeze(-1)
    camera_points = unproject(xy_coords.unsqueeze(0), depth_values.unsqueeze(0), intrinsics)[0]
    camera_points_hom = homogenize_points(camera_points)
    world_points = transform_cam2world(camera_points_hom.unsqueeze(0), extrinsics)[0]
    sparse_pts_world = world_points[..., :3]  # [N,3]

    grid = sparse_coords_normalized[:, [1, 0]].unsqueeze(0).unsqueeze(0)  # [1,1,N,2]

    def sample_attribute(attr):
        if attr.dim() == 2:
            attr_spatial = attr.view(1, 1, h, w)
            sampled = F.grid_sample(attr_spatial, grid, mode="bilinear", align_corners=False)
            return sampled.squeeze(0).squeeze(0)
        if attr.dim() == 3:
            d = attr.shape[-1]
            attr_spatial = attr.view(1, h, w, d).permute(0, 3, 1, 2)
            sampled = F.grid_sample(attr_spatial, grid, mode="bilinear", align_corners=False)
            return sampled.squeeze(2).permute(0, 2, 1)
        if attr.dim() == 4:
            d1, d2 = attr.shape[-2:]
            attr_flat = attr.view(1, h, w, d1 * d2).permute(0, 3, 1, 2)
            sampled = F.grid_sample(attr_flat, grid, mode="bilinear", align_corners=False)
            return sampled.squeeze(2).permute(0, 2, 1).view(1, -1, d1, d2)
        raise ValueError(f"Unsupported attribute dimension: {attr.dim()}")

    sparse_harmonics = sample_attribute(dense_gaussians.harmonics)
    sparse_opacities = sample_attribute(dense_gaussians.opacities)
    sparse_scales = sample_attribute(dense_gaussians.scales)
    sparse_rotations = sample_attribute(dense_gaussians.rotations)
    sparse_rotations = sparse_rotations / (torch.norm(sparse_rotations, dim=-1, keepdim=True) + 1e-8)

    return Gaussians(
        means=sparse_pts_world.unsqueeze(0),
        covariances=None,
        harmonics=sparse_harmonics,
        opacities=sparse_opacities,
        scales=sparse_scales,
        rotations=sparse_rotations,
    )
