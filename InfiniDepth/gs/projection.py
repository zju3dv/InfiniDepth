import torch


def homogenize_points(points: torch.Tensor) -> torch.Tensor:
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_cam2world(homogeneous: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    return torch.matmul(extrinsics, homogeneous.unsqueeze(-1)).squeeze(-1)


def unproject(coordinates_xy: torch.Tensor, z: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Unproject pixel-space xy to camera space using z depth.

    coordinates_xy: [B, N, 2] in pixel coordinates (x, y)
    z: [B, N]
    intrinsics: [B, 3, 3] in pixel units
    """
    coordinates_h = homogenize_points(coordinates_xy)  # [B, N, 3]
    intr_inv = torch.linalg.inv(intrinsics)            # [B, 3, 3]
    rays = torch.matmul(intr_inv.unsqueeze(1), coordinates_h.unsqueeze(-1)).squeeze(-1)
    return rays * z.unsqueeze(-1)


def get_world_rays(
    coordinates_xy: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-space ray origins and directions.

    coordinates_xy: [B, N, 2] in pixel coordinates (x, y)
    extrinsics: [B, 4, 4] camera-to-world
    intrinsics: [B, 3, 3] pixel intrinsics
    """
    ones = torch.ones_like(coordinates_xy[..., 0])
    directions_cam = unproject(coordinates_xy, ones, intrinsics)
    directions_cam = directions_cam / torch.clamp(directions_cam[..., 2:], min=1e-6)
    directions_world = transform_cam2world(homogenize_vectors(directions_cam), extrinsics)[..., :3]
    origins_world = extrinsics[:, None, :3, 3].expand_as(directions_world)
    return origins_world, directions_world


def sample_image_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Return pixel center coordinates with shape [H*W, 2], order (x, y)."""
    ys = torch.arange(h, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(w, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
