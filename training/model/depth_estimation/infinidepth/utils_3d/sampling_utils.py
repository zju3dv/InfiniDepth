import torch
import numpy as np
import torch.nn.functional as F
import math
import typing
from scipy.spatial import cKDTree
from tqdm.auto import tqdm
from typing import Optional
import matplotlib.pyplot as plt

from .vis_utils import visualize_normal


def _scale_intrinsics(fx, fy, cx, cy, org_h, org_w, h, w):
    sx, sy = w / float(org_w), h / float(org_h)
    return fx * sx, fy * sy, cx * sx, cy * sy


def depth_to_normal(depth, K):
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    X = (j - cx) * depth / fx
    Y = (i - cy) * depth / fy
    Z = depth
    points = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)

    dzdx = points[:, 1:, :] - points[:, :-1, :]
    dzdy = points[1:, :, :] - points[:-1, :, :]

    dzdx = np.pad(dzdx, ((0,0),(0,1),(0,0)), mode='edge')
    dzdy = np.pad(dzdy, ((0,1),(0,0),(0,0)), mode='edge')

    normal = np.cross(dzdx, dzdy)
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= norm + 1e-8

    return torch.from_numpy(normal)


def make_2d_uniform_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    query_coords = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        query_coords = query_coords.view(-1, query_coords.shape[-1])
    return query_coords


# ----------------- method 1 (zplane) -------------------
def make_3d_uniform_coord_zplane(grid_depth, 
                          fx, fy, cx, cy,
                          num_samples=500000,
                          rng_seed=None,
                          method='deterministic',  # 'deterministic' or 'stochastic'
                          min_per_nonzero_cell=0,  # 0 or 1
                          device=None):
    """
    Pure torch implementation of area-based importance sampling.

    Parameters
    ----------
    grid_depth : torch.Tensor, shape (1,1,H,W)
        Depth at grid centers.
    fx, fy, cx, cy : float
        Intrinsics.
    num_samples : int >= 0
    rng_seed : int or None
    method : str
    min_per_nonzero_cell : int
    device : torch.device or None
        If None, use grid_depth.device.

    Returns
    -------
    samples_norm : torch.Tensor, shape (M,2), dtype=torch.float32
        Values in [-1,1], align_corners=False, order (x_norm, y_norm).
    """
    if device is None:
        device = grid_depth.device

    # check inputs
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if method not in ('deterministic', 'stochastic'):
        raise ValueError("method must be 'deterministic' or 'stochastic'")
    if min_per_nonzero_cell not in (0, 1):
        raise ValueError("min_per_nonzero_cell must be 0 or 1")

    # RNG
    g = torch.Generator(device=device)
    if rng_seed is not None:
        g.manual_seed(rng_seed)

    _, _, H, W = grid_depth.shape
    depth = grid_depth[0, 0].to(dtype=torch.float32, device=device)

    if num_samples == 0:
        return torch.zeros((0, 2), dtype=torch.float32, device=device)

    # pixel centers u=0..W-1, v=0..H-1
    u_coords = torch.arange(W, dtype=torch.float32, device=device)
    v_coords = torch.arange(H, dtype=torch.float32, device=device)
    uu, vv = torch.meshgrid(v_coords, u_coords, indexing='ij')  # (H,W)

    # corners
    u_tl = uu - 0.5; v_tl = vv - 0.5
    u_tr = uu + 0.5; v_tr = vv - 0.5
    u_br = uu + 0.5; v_br = vv + 0.5
    u_bl = uu - 0.5; v_bl = vv + 0.5

    # invalid depth
    invalid_mask = torch.logical_or(~torch.isfinite(depth), depth <= 0)

    # helper for 3D corners
    def compute_3d_corners(u_corner, v_corner, z):
        x = (u_corner - cx) / fx * z
        y = (v_corner - cy) / fy * z
        return torch.stack([x, y, z], dim=-1)

    p_tl = compute_3d_corners(u_tl, v_tl, depth)
    p_tr = compute_3d_corners(u_tr, v_tr, depth)
    p_br = compute_3d_corners(u_br, v_br, depth)
    p_bl = compute_3d_corners(u_bl, v_bl, depth)

    def tri_area(a, b, c):
        ab = b - a
        ac = c - a
        cross = torch.cross(ab, ac, dim=-1)
        return 0.5 * torch.norm(cross, dim=-1)

    area1 = tri_area(p_tl, p_tr, p_br)
    area2 = tri_area(p_tl, p_br, p_bl)
    areas = area1 + area2
    areas[invalid_mask] = 0.0
    areas = torch.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)

    areas_flat = areas.reshape(-1)
    areas_sum = areas_flat.sum()
    n_cells = H * W

    if areas_sum <= 0:
        probs = torch.full((n_cells,), 1.0 / n_cells, dtype=torch.float32, device=device)
    else:
        probs = areas_flat / areas_sum

    per_cell = torch.zeros(n_cells, dtype=torch.int32, device=device)

    if num_samples <= n_cells:
        if method == 'deterministic':
            idxs = torch.argsort(-probs)[:num_samples]
            per_cell[idxs] = 1
        else:
            if probs.sum() <= 0:
                idxs = torch.arange(num_samples, device=device)
            else:
                idxs = torch.multinomial(probs, num_samples, replacement=False, generator=g)
            per_cell[idxs] = 1
    else:
        if min_per_nonzero_cell == 1 and torch.any(areas_flat > 0):
            nonzero_idxs = torch.nonzero(areas_flat > 0, as_tuple=False).view(-1)
            per_cell[nonzero_idxs] = 1
            remaining = num_samples - per_cell.sum()
            if remaining > 0:
                if probs.sum() <= 0:
                    base = remaining // n_cells
                    per_cell += base
                    rem = remaining - base * n_cells
                    if rem > 0:
                        per_cell[:rem] += 1
                else:
                    if method == 'stochastic':
                        add = torch.multinomial(probs, remaining, replacement=True, generator=g)
                        per_cell.scatter_add_(0, add, torch.ones_like(add, dtype=torch.int32))
                    else:
                        raw = probs * remaining
                        floors = torch.floor(raw).to(torch.int32)
                        per_cell += floors
                        rem = remaining - floors.sum()
                        if rem > 0:
                            frac = raw - floors
                            idxs = torch.argsort(-frac)
                            per_cell[idxs[:rem]] += 1
        else:
            if method == 'stochastic':
                add = torch.multinomial(probs, num_samples, replacement=True, generator=g)
                per_cell.scatter_add_(0, add, torch.ones_like(add, dtype=torch.int32))
            else:
                raw = probs * num_samples
                floors = torch.floor(raw).to(torch.int32)
                per_cell = floors.clone()
                rem = num_samples - per_cell.sum()
                if rem > 0:
                    frac = raw - floors
                    noise = torch.rand(frac.size(), generator=g, device=device) * 1e-12
                    frac_noise = frac + noise
                    idxs = torch.argsort(-frac_noise)
                    per_cell[idxs[:rem]] += 1

    # Sanity check sum
    total_assigned = int(per_cell.sum().item())
    if total_assigned != num_samples:
        diff = num_samples - total_assigned
        if diff > 0:
            idxs = torch.argsort(-probs)
            per_cell[idxs[:diff]] += 1
        else:
            idxs = torch.argsort(probs)
            i = 0
            while per_cell.sum() > num_samples and i < len(idxs):
                idx = idxs[i]
                if per_cell[idx] > 0:
                    remove = min(int(per_cell[idx]), int(per_cell.sum() - num_samples))
                    per_cell[idx] -= remove
                i += 1

    # Expand per_cell
    cell_indices = torch.nonzero(per_cell > 0, as_tuple=False).view(-1)
    counts = per_cell[cell_indices]
    if counts.sum() == 0:
        return torch.zeros((0, 2), dtype=torch.float32, device=device)

    v_idx = (cell_indices // W).to(torch.int32)
    u_idx = (cell_indices % W).to(torch.int32)

    u_centers_rep = u_coords[u_idx].repeat_interleave(counts)
    v_centers_rep = v_coords[v_idx].repeat_interleave(counts)

    offsets_u = torch.rand(u_centers_rep.shape[0], generator=g, device=device) - 0.5
    offsets_v = torch.rand(v_centers_rep.shape[0], generator=g, device=device) - 0.5

    samples_u = u_centers_rep + offsets_u
    samples_v = v_centers_rep + offsets_v

    samples_norm_x = ((samples_u + 0.5) / W) * 2.0 - 1.0
    samples_norm_y = ((samples_v + 0.5) / H) * 2.0 - 1.0
    samples_norm = torch.stack([samples_norm_y, samples_norm_x], dim=-1).to(torch.float32)

    return samples_norm


# ----------------- method 2 (triangle_area) -------------------
def _depth_to_vertices(D, fx, fy, cx, cy):
    """D: (h,w) depth in meters (or any consistent unit). Return V: (h,w,3)."""
    h, w = D.shape
    js, is_ = np.meshgrid(np.arange(w), np.arange(h))
    Z = D
    X = (js - cx) / fx * Z
    Y = (is_ - cy) / fy * Z
    V = np.stack([X, Y, Z], axis=-1)
    return V


def _build_faces(h, w):
    """Return faces: (2*(h-1)*(w-1), 3) of flattened vertex indices."""
    idx = np.arange(h*w).reshape(h, w)
    f1 = np.stack([idx[:-1,:-1], idx[1:,:-1], idx[:-1,1:]], axis=-1).reshape(-1,3)
    f2 = np.stack([idx[1:,1:], idx[:-1,1:], idx[1:,:-1]], axis=-1).reshape(-1,3)
    return np.vstack([f1, f2])


def _prune_faces(Vflat, faces, depth_ratio=1.05, max_edge=None):
    """Prune triangles that likely cross depth discontinuities or are too large."""
    A, B, C = Vflat[faces[:,0]], Vflat[faces[:,1]], Vflat[faces[:,2]]
    zA, zB, zC = A[:,2], B[:,2], C[:,2]
    zmin = np.minimum(np.minimum(zA, zB), zC)
    zmax = np.maximum(np.maximum(zA, zB), zC)
    keep = (zmin > 0) & (zmax / np.maximum(zmin, 1e-9) < depth_ratio)

    if max_edge is not None:
        e0 = np.linalg.norm(B - A, axis=1)
        e1 = np.linalg.norm(C - B, axis=1)
        e2 = np.linalg.norm(A - C, axis=1)
        keep &= (np.maximum(np.maximum(e0, e1), e2) < max_edge)
    return faces[keep]


def _faces_to_ij(faces, h, w):
    """For each vertex index -> (i,j)."""
    i = faces // w
    j = faces %  w
    return i, j


def make_3d_uniform_coord_triangle(
    depth_hw, fx, fy, cx, cy, N, coord_norm="minus_one_to_one",
    depth_ratio=1.05, max_edge=None, rng=None
):
    """
    depth_hw: (h,w) numpy array (predicted depth on the resized image grid)
    returns: coords (N,2) numpy in the same 2D system as your model expects
             coord_norm: 'minus_one_to_one' -> [-1,1]  ;  'zero_one' -> [0,1]
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w = depth_hw.shape
    V = _depth_to_vertices(depth_hw, fx, fy, cx, cy)  # (h,w,3)
    Vflat = V.reshape(-1, 3)
    faces = _build_faces(h, w)

    faces = _prune_faces(Vflat, faces, depth_ratio=depth_ratio, max_edge=max_edge)
    if len(faces) == 0:
        raise RuntimeError("All faces pruned; relax 'depth_ratio' or 'max_edge'.")

    A = Vflat[faces[:,0]]; B = Vflat[faces[:,1]]; C = Vflat[faces[:,2]]
    areas = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    areas = np.clip(areas, 0, None)
    total_area = areas.sum()
    if not np.isfinite(total_area) or total_area <= 0:
        raise RuntimeError("Invalid total area; check depth values.")
    probs = areas / total_area

    tri_idx = rng.choice(len(faces), size=N, p=probs)
    f = faces[tri_idx]

    u = rng.random(N); v = rng.random(N)
    mask = (u + v > 1.0); u[mask] = 1 - u[mask]; v[mask] = 1 - v[mask]
    w0 = 1.0 - u - v; w1 = u; w2 = v

    fi, fj = _faces_to_ij(f, h, w)  # each (N,3)
    i0, i1, i2 = fi[:,0], fi[:,1], fi[:,2]
    j0, j1, j2 = fj[:,0], fj[:,1], fj[:,2]

    i_s = w0 * i0 + w1 * i1 + w2 * i2
    j_s = w0 * j0 + w1 * j1 + w2 * j2
    
    # normalize (align_corners=False)
    x = 2.0 * ((j_s + 0.5) / w) - 1.0
    y = 2.0 * ((i_s + 0.5) / h) - 1.0

    coords = np.stack([y, x], axis=-1).astype(np.float32)  # (N,2)
    return coords


# ------------ method 3 (delta area)------------
def backproject_depth_to_V(depth_hw: torch.Tensor, fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    h, w = depth_hw.shape
    device = depth_hw.device
    i = torch.arange(h, device=device).float()
    j = torch.arange(w, device=device).float()
    ii, jj = torch.meshgrid(i, j, indexing='ij')

    Z = depth_hw
    X = (jj - cx) / fx * Z
    Y = (ii - cy) / fy * Z
    V = torch.stack([X, Y, Z], dim=-1)  # (h,w,3)
    return V


def area_element_from_V(V: torch.Tensor, depth_hw: torch.Tensor) -> torch.Tensor:
    h, w, _ = V.shape
    device = V.device
    Vt = V.permute(2, 0, 1).unsqueeze(0)  # [1,3,h,w]

    kx = torch.tensor([[-0.5, 0.0, 0.5]], device=device).view(1,1,1,3)
    ky = torch.tensor([[-0.5],[0.0],[0.5]], device=device).view(1,1,3,1)
    weight_x = kx.repeat(3,1,1,1)  # (3,1,1,3)
    weight_y = ky.repeat(3,1,1,1)  # (3,1,3,1)

    dV_dj = F.conv2d(Vt, weight_x, padding=(0,1), groups=3)  # [1,3,h,w]
    dV_di = F.conv2d(Vt, weight_y, padding=(1,0), groups=3)  # [1,3,h,w]

    dV_dj = dV_dj.squeeze(0).permute(1,2,0)  # (h,w,3)
    dV_di = dV_di.squeeze(0).permute(1,2,0)  # (h,w,3)

    cross = torch.cross(dV_di, dV_dj, dim=-1)  # (h,w,3)
    area_w = torch.linalg.norm(cross, dim=-1)  # (h,w)

    valid = torch.isfinite(depth_hw) & (depth_hw > 0)
    area_w = torch.where(valid, area_w, torch.zeros_like(area_w))
    return area_w


def make_3d_uniform_coord_delta(
    depth_hw: torch.Tensor, fx: float, fy: float, cx: float, cy: float,
    N: int
) -> torch.Tensor:
    device = depth_hw.device
    h, w = depth_hw.shape

    V = backproject_depth_to_V(depth_hw, fx, fy, cx, cy)        # (h,w,3)
    area_w = area_element_from_V(V, depth_hw)                   # (h,w)

    w_flat = area_w.reshape(-1)
    w_sum = w_flat.sum()
    if not torch.isfinite(w_sum) or w_sum <= 0:
        raise RuntimeError("面积权重总和无效；请检查深度或内参。")

    idx = torch.multinomial(w_flat, num_samples=N, replacement=True)  # (N,)
    ii = (idx // w).float()
    jj = (idx %  w).float()
    di = torch.rand(N, device=device) - 0.5
    dj = torch.rand(N, device=device) - 0.5
    i_s = ii + di
    j_s = jj + dj

    # normalize (align_corners=False)
    x = 2.0 * ((j_s + 0.5) / w) - 1.0
    y = 2.0 * ((i_s + 0.5) / h) - 1.0
    query_coord = torch.stack([y, x], dim=-1).unsqueeze(0)  # [1,N,2]
    return query_coord


# ------------ method 4 (surface_poisson)------------
def bilinear_sample_V(V: torch.Tensor, query_coord: torch.Tensor, align_corners: bool = False) -> torch.Tensor:
    Vchw = V.permute(2,0,1).unsqueeze(0)   # [1,3,h,w]
    grid = query_coord.unsqueeze(2)         # [1,N,1,2]
    P = F.grid_sample(Vchw, grid, mode='bilinear', padding_mode='border',
                      align_corners=align_corners)  # [1,3,N,1]
    P = P.squeeze(0).squeeze(-1).permute(1,0)       # (N,3)
    return P


def estimate_total_area(area_w: torch.Tensor) -> float:
    return float(area_w.sum().item())


def radius_from_area(total_area: float, N: int, alpha: float = 0.9) -> float:
    r = math.sqrt((2.0 / math.sqrt(3.0)) * (total_area / max(N,1)))
    return r * alpha


def poisson_disk_select_fixed_candidates(P: np.ndarray, r: float) -> np.ndarray:
    if len(P) == 0:
        return np.array([], dtype=np.int64)

    a = r / math.sqrt(3.0)
    mins = P.min(axis=0)
    inv_a = 1.0 / max(a, 1e-12)

    def cell_idx(pt):
        return tuple(((pt - mins) * inv_a).astype(np.int64))

    grid = {}  # dict[(ix,iy,iz)] -> list of indices kept in this cell
    keep = []
    r2 = r * r

    order = np.random.permutation(len(P))
    for idx in order:
        p = P[idx]
        cx, cy, cz = cell_idx(p)

        ok = True
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    cell = (cx+dx, cy+dy, cz+dz)
                    if cell not in grid: 
                        continue
                    for j in grid[cell]:
                        q = P[j]
                        if np.dot(p-q, p-q) < r2:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            keep.append(idx)
            grid.setdefault((cx,cy,cz), []).append(idx)
    return np.array(keep, dtype=np.int64)


def make_3d_uniform_coord_surface_poisson(
    depth_hw: torch.Tensor, fx: float, fy: float, cx: float, cy: float,
    N: int, oversample: float = 4.0, align_corners: bool = False,
    alpha_radius: float = 0.9
):
    device = depth_hw.device
    h, w = depth_hw.shape

    V = backproject_depth_to_V(depth_hw, fx, fy, cx, cy)         # (h,w,3)
    area_w = area_element_from_V(V, depth_hw)                    # (h,w)
    total_area = estimate_total_area(area_w)

    M = max(int(oversample * N), N+1)

    cand_coord = make_3d_uniform_coord_delta(
        depth_hw, fx, fy, cx, cy, N=M
    )  # [1,M,2]

    P = bilinear_sample_V(V, cand_coord, align_corners=align_corners)  # (M,3) torch
    P_np = P.detach().cpu().numpy()

    r = radius_from_area(total_area, N, alpha=alpha_radius)

    keep_np = poisson_disk_select_fixed_candidates(P_np, r)  # (K,)
    if keep_np.size == 0:
        keep_np = np.arange(min(N, M), dtype=np.int64)
    if keep_np.size > N:
        keep_np = keep_np[:N]

    keep = torch.from_numpy(keep_np).to(device)
    query_coord = cand_coord[:, keep, :]  # [1,K,2]
    return query_coord


def make_3d_uniform_coord_quadtree(
    model, image, prompt_depth,
    fx: float, fy: float, cx: float, cy: float,
    distance_threshold: float = 0.05,
    min_cell_size: Optional[float] = 1.0,
    max_points: Optional[int] = None,
    show_progress: bool = True,
    cache_tolerance: float = 1e-6,
):
    device = image.device
    _, _, H, W = image.shape
    coord_cache = {}

    def get_cached_depth(coords_norm: torch.Tensor) -> torch.Tensor:
        keys = [tuple((c / cache_tolerance).round().cpu().numpy()) for c in coords_norm]
        need_query_idx = [i for i, k in enumerate(keys) if k not in coord_cache]
        depths = torch.zeros(len(coords_norm), device=device)
        if need_query_idx:
            batch = coords_norm[need_query_idx].unsqueeze(0)
            with torch.no_grad():
                d, _ = model.inference(
                    image=image,
                    query_coord=batch,
                    prompt_depth=prompt_depth,
                )
            d = d.squeeze(0).squeeze(-1)
            for idx, val in zip(need_query_idx, d):
                coord_cache[keys[idx]] = val.item()
        for i, k in enumerate(keys):
            depths[i] = coord_cache[k]
        return depths

    def pixel_to_3d(i_pix: torch.Tensor, j_pix: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        x = (j_pix - cx) / fx * depth
        y = (i_pix - cy) / fy * depth
        return torch.stack([x, y, depth], dim=-1)

    centers_i = torch.tensor([H/2], device=device, dtype=torch.float32)
    centers_j = torch.tensor([W/2], device=device, dtype=torch.float32)
    h_sizes = torch.tensor([H], device=device, dtype=torch.float32)
    w_sizes = torch.tensor([W], device=device, dtype=torch.float32)

    all_depths, all_coords = [], []

    pbar = tqdm(total=max_points if max_points else 0, desc="QuadTree Sampling", dynamic_ncols=True) if show_progress else None

    while len(centers_i) > 0:
        x_norm = 2.0 * ((centers_j + 0.5) / W) - 1.0
        y_norm = 2.0 * ((centers_i + 0.5) / H) - 1.0
        coords_norm = torch.stack([y_norm, x_norm], dim=-1)

        depths = get_cached_depth(coords_norm)
        valid_mask = torch.isfinite(depths) & (depths > 0)
        if valid_mask.sum() == 0:
            break
        centers_i = centers_i[valid_mask]
        centers_j = centers_j[valid_mask]
        h_sizes = h_sizes[valid_mask]
        w_sizes = w_sizes[valid_mask]
        depths = depths[valid_mask]
        coords_norm = coords_norm[valid_mask]

        points_3d = pixel_to_3d(centers_i, centers_j, depths)

        all_depths.append(depths)
        all_coords.append(coords_norm)
        if pbar is not None:
            pbar.update(len(depths))

        half_h = h_sizes / 4
        half_w = w_sizes / 4
        offsets = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]], device=device, dtype=torch.float32)

        child_centers_i = (centers_i.unsqueeze(1) + offsets[:,0].unsqueeze(0) * half_h.unsqueeze(1)).reshape(-1)
        child_centers_j = (centers_j.unsqueeze(1) + offsets[:,1].unsqueeze(0) * half_w.unsqueeze(1)).reshape(-1)
        child_h_sizes = (h_sizes / 2).repeat_interleave(4)
        child_w_sizes = (w_sizes / 2).repeat_interleave(4)

        child_x_norm = 2.0 * ((child_centers_j + 0.5) / W) - 1.0
        child_y_norm = 2.0 * ((child_centers_i + 0.5) / H) - 1.0
        child_coords_norm = torch.stack([child_y_norm, child_x_norm], dim=-1)
        child_depths = get_cached_depth(child_coords_norm)
        valid_mask_child = torch.isfinite(child_depths) & (child_depths > 0)
        if valid_mask_child.sum() == 0:
            centers_i = torch.tensor([], device=device)
            centers_j = torch.tensor([], device=device)
            h_sizes = torch.tensor([], device=device)
            w_sizes = torch.tensor([], device=device)
            continue
        child_centers_i = child_centers_i[valid_mask_child]
        child_centers_j = child_centers_j[valid_mask_child]
        child_h_sizes = child_h_sizes[valid_mask_child]
        child_w_sizes = child_w_sizes[valid_mask_child]
        child_depths = child_depths[valid_mask_child]

        child_points_3d = pixel_to_3d(child_centers_i, child_centers_j, child_depths)

        num_parent = len(centers_i)
        child_points_3d_reshaped = child_points_3d.view(num_parent, 4, 3)
        spreads = torch.norm(child_points_3d_reshaped - points_3d.unsqueeze(1), dim=-1).max(dim=1).values

        subdivide_mask = spreads > distance_threshold
        if min_cell_size is not None:
            subdivide_mask &= (h_sizes > min_cell_size) | (w_sizes > min_cell_size)

        if subdivide_mask.sum() > 0:
            keep_parent_indices = subdivide_mask.nonzero(as_tuple=True)[0]
            centers_i = child_centers_i.view(num_parent, 4)[keep_parent_indices].reshape(-1)
            centers_j = child_centers_j.view(num_parent, 4)[keep_parent_indices].reshape(-1)
            h_sizes = child_h_sizes.view(num_parent, 4)[keep_parent_indices].reshape(-1)
            w_sizes = child_w_sizes.view(num_parent, 4)[keep_parent_indices].reshape(-1)
        else:
            centers_i = torch.tensor([], device=device)
            centers_j = torch.tensor([], device=device)
            h_sizes = torch.tensor([], device=device)
            w_sizes = torch.tensor([], device=device)

        if max_points is not None and sum(len(d) for d in all_depths) >= max_points:
            break

    if pbar is not None:
        pbar.close()

    if len(all_depths) == 0:
        return torch.zeros((1,0,1), device=device)

    all_coords = torch.cat(all_coords, dim=0)
    all_depths = torch.cat(all_depths, dim=0).unsqueeze(-1)

    return all_coords.unsqueeze(0), all_depths.unsqueeze(0)


def make_3d_uniform_coord_quadtree_random(
    model, image, prompt_depth,
    fx: float, fy: float, cx: float, cy: float,
    distance_threshold: float = 0.05,
    min_cell_size: Optional[float] = 1.0,
    max_points: Optional[int] = None,
    show_progress: bool = True,
    cache_tolerance: float = 1e-6,
    jitter_ratio: float = 0.25,
):
    device = image.device
    _, _, H, W = image.shape
    coord_cache = {}

    def get_cached_depth(coords_norm: torch.Tensor) -> torch.Tensor:
        # coords_norm: [N, 2] (y_norm, x_norm)
        keys = [tuple((c / cache_tolerance).round().cpu().numpy()) for c in coords_norm]
        need_query_idx = [i for i, k in enumerate(keys) if k not in coord_cache]
        depths = torch.zeros(len(coords_norm), device=device)
        if need_query_idx:
            batch = coords_norm[need_query_idx].unsqueeze(0)  # [1, M, 2]
            with torch.no_grad():
                d, _ = model.inference(
                    image=image,
                    query_coord=batch,
                    prompt_depth=prompt_depth,
                )
            d = d.squeeze(0).squeeze(-1)  # [M]
            for idx, val in zip(need_query_idx, d):
                coord_cache[keys[idx]] = val.item()
        for i, k in enumerate(keys):
            depths[i] = coord_cache[k]
        return depths

    def pixel_to_3d(i_pix: torch.Tensor, j_pix: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        # i_pix: [N] (row), j_pix: [N] (col), depth: [N]
        x = (j_pix - cx) / fx * depth
        y = (i_pix - cy) / fy * depth
        return torch.stack([x, y, depth], dim=-1)  # [N, 3]

    centers_i = torch.tensor([float(H) / 2.0], device=device, dtype=torch.float32)
    centers_j = torch.tensor([float(W) / 2.0], device=device, dtype=torch.float32)
    h_sizes = torch.tensor([float(H)], device=device, dtype=torch.float32)
    w_sizes = torch.tensor([float(W)], device=device, dtype=torch.float32)

    all_depths, all_coords = [], []

    pbar = tqdm(total=max_points if max_points else 0,
                desc="QuadTree Sampling", dynamic_ncols=True) if show_progress else None

    while len(centers_i) > 0:
        x_norm = 2.0 * ((centers_j + 0.5) / W) - 1.0
        y_norm = 2.0 * ((centers_i + 0.5) / H) - 1.0
        coords_norm = torch.stack([y_norm, x_norm], dim=-1)  # [P, 2]

        depths = get_cached_depth(coords_norm)  # [P]
        valid_mask = torch.isfinite(depths) & (depths > 0)
        if valid_mask.sum() == 0:
            break

        centers_i = centers_i[valid_mask]
        centers_j = centers_j[valid_mask]
        h_sizes = h_sizes[valid_mask]
        w_sizes = w_sizes[valid_mask]
        depths = depths[valid_mask]
        coords_norm = coords_norm[valid_mask]

        points_3d = pixel_to_3d(centers_i, centers_j, depths)  # [P,3]

        all_depths.append(depths)
        all_coords.append(coords_norm)
        if pbar is not None:
            pbar.update(len(depths))

        half_h = h_sizes / 4.0
        half_w = w_sizes / 4.0
        offsets = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
                               device=device, dtype=torch.float32)  # [4,2]

        base_child_i = centers_i.unsqueeze(1) + offsets[:, 0].unsqueeze(0) * half_h.unsqueeze(1)  # [P,4]
        base_child_j = centers_j.unsqueeze(1) + offsets[:, 1].unsqueeze(0) * half_w.unsqueeze(1)  # [P,4]

        jitter_i = (torch.rand_like(base_child_i) - 0.5) * 2.0 * (jitter_ratio * half_h.unsqueeze(1))  # [P,4]
        jitter_j = (torch.rand_like(base_child_j) - 0.5) * 2.0 * (jitter_ratio * half_w.unsqueeze(1))  # [P,4]

        child_centers_i = (base_child_i + jitter_i).reshape(-1)  # [P*4]
        child_centers_j = (base_child_j + jitter_j).reshape(-1)  # [P*4]

        child_centers_i = child_centers_i.clamp(0.0, float(H) - 1.0)
        child_centers_j = child_centers_j.clamp(0.0, float(W) - 1.0)

        child_h_sizes = (h_sizes / 2.0).repeat_interleave(4)  # [P*4]
        child_w_sizes = (w_sizes / 2.0).repeat_interleave(4)  # [P*4]

        child_x_norm = 2.0 * ((child_centers_j + 0.5) / W) - 1.0
        child_y_norm = 2.0 * ((child_centers_i + 0.5) / H) - 1.0
        child_coords_norm = torch.stack([child_y_norm, child_x_norm], dim=-1)  # [P*4, 2]

        child_depths = get_cached_depth(child_coords_norm)  # [P*4]
        P = centers_i.shape[0]
        child_depths = child_depths.view(P, 4)  # [P,4]
        child_centers_i = child_centers_i.view(P, 4)
        child_centers_j = child_centers_j.view(P, 4)
        child_h_sizes = child_h_sizes.view(P, 4)
        child_w_sizes = child_w_sizes.view(P, 4)

        flat_child_pts = pixel_to_3d(child_centers_i.reshape(-1), child_centers_j.reshape(-1), child_depths.reshape(-1))
        child_points_3d = flat_child_pts.view(P, 4, 3)  # [P,4,3]

        valid_child_mask = torch.isfinite(child_depths) & (child_depths > 0)  # [P,4]
        valid_counts = valid_child_mask.sum(dim=1)  # [P]

        mask_f = valid_child_mask.unsqueeze(-1).to(child_points_3d.dtype)  # [P,4,1]
        sum_valid = mask_f.sum(dim=1)  # [P,1]
        centroid = (child_points_3d * mask_f).sum(dim=1) / sum_valid.clamp(min=1.0)  # [P,3]

        centered = (child_points_3d - centroid.unsqueeze(1)) * mask_f  # [P,4,3]

        # X: [P,4,3] -> want cov: [P,3,3]
        X = centered  # [P,4,3], already masked zeros for invalid
        # compute cov = X^T @ X
        cov = torch.matmul(X.transpose(1, 2), X)  # [P,3,3]
        denom = valid_counts.clamp(min=1).to(cov.dtype).unsqueeze(-1).unsqueeze(-1)
        cov = cov / denom  # [P,3,3]

        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)  # eigvals: [P,3], eigvecs: [P,3,3]
            normals = eigvecs[:, :, 0]
        except RuntimeError:
            normals = torch.zeros((P, 3), device=device, dtype=child_points_3d.dtype)
            for pi in range(P):
                valid_idx = valid_child_mask[pi].nonzero(as_tuple=True)[0]
                if valid_idx.numel() >= 3:
                    pts = child_points_3d[pi, valid_idx]  # [k,3]
                    c = pts.mean(dim=0, keepdim=True)
                    U, S, Vt = torch.linalg.svd(pts - c)  # Vt: [3,3]
                    normals[pi] = Vt[-1]
                else:
                    normals[pi] = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=child_points_3d.dtype)

        norm_norms = torch.linalg.norm(normals, dim=1, keepdim=True).clamp(min=1e-8)
        normals_unit = normals / norm_norms  # [P,3]

        # residuals: [P,4]
        diffs = child_points_3d - centroid.unsqueeze(1)  # [P,4,3]
        residuals = torch.abs(torch.matmul(diffs, normals_unit.unsqueeze(-1)).squeeze(-1))  # [P,4]

        residuals_masked = residuals.clone()
        residuals_masked[~valid_child_mask] = 0.0
        max_plane_res = residuals_masked.max(dim=1).values  # [P]

        # compute fallback distances
        parent_pts = points_3d  # [P,3]
        dists = torch.linalg.norm(child_points_3d - parent_pts.unsqueeze(1), dim=-1)  # [P,4]
        dists_masked = dists.clone()
        dists_masked[~valid_child_mask] = 0.0
        max_child_parent_dist = dists_masked.max(dim=1).values  # [P]

        use_plane_mask = valid_counts >= 3
        spreads = torch.where(use_plane_mask, max_plane_res, max_child_parent_dist)  # [P]

        subdivide_mask = spreads > distance_threshold
        if min_cell_size is not None:
            subdivide_mask &= (h_sizes > min_cell_size) | (w_sizes > min_cell_size)

        if subdivide_mask.sum() > 0:
            keep_parent_indices = subdivide_mask.nonzero(as_tuple=True)[0]

            # child_centers_i/j currently shaped [P,4]
            next_centers_i = child_centers_i[keep_parent_indices].reshape(-1)
            next_centers_j = child_centers_j[keep_parent_indices].reshape(-1)
            next_h_sizes = child_h_sizes[keep_parent_indices].reshape(-1)
            next_w_sizes = child_w_sizes[keep_parent_indices].reshape(-1)

            centers_i = next_centers_i
            centers_j = next_centers_j
            h_sizes = next_h_sizes
            w_sizes = next_w_sizes
        else:
            centers_i = torch.tensor([], device=device)
            centers_j = torch.tensor([], device=device)
            h_sizes = torch.tensor([], device=device)
            w_sizes = torch.tensor([], device=device)

        if max_points is not None and sum(len(d) for d in all_depths) >= max_points:
            break

    if pbar is not None:
        pbar.close()

    if len(all_depths) == 0:
        return torch.zeros((1, 0, 2), device=device), torch.zeros((1, 0, 1), device=device)

    all_coords = torch.cat(all_coords, dim=0)  # [N, 2]
    all_depths = torch.cat(all_depths, dim=0).unsqueeze(-1)  # [N,1]

    return all_coords.unsqueeze(0), all_depths.unsqueeze(0)


def make_3d_uniform_coord_autograd(
    model,                 
    image,
    prompt, 
    K,                    
    H, W,                 
    N,                   
    eps=1e-6,              
    w_min=1e-6, w_max=1e6,
    vis_normal=True,
    normal_save_path=None,
    chunk_size=20000       
):
    device = image.device
    dtype  = image.dtype
    K_inv  = torch.inverse(K).to(device=device, dtype=dtype)

    flat_yx = make_2d_uniform_coord(shape=(H, W), flatten=True).unsqueeze(0).to(device=device, dtype=dtype) 
    grid_yx = flat_yx.reshape(H, W, 2) 
    grid_y, grid_x = grid_yx[..., 0], grid_yx[..., 1] 

    u = ((grid_x + 1) * W - 1) / 2.0  
    v = ((grid_y + 1) * H - 1) / 2.0   

    with torch.no_grad():
        z_full, _ = model.inference(image=image, query_coord=flat_yx, prompt_depth=prompt)  
        z_full = z_full.reshape(H, W) 
        # depth --> 3d points
        xy1_full = torch.stack([u, v, torch.ones_like(u, dtype=dtype, device=device)], dim=-1) 
        dir_cam_full = torch.einsum("ij,...j->...i", K_inv, xy1_full)                        
        X_full = z_full[..., None] * dir_cam_full                                           

    Npix = H * W
    n_out = torch.empty((Npix, 3), device=device, dtype=dtype)
    Hf, Wf = float(H), float(W)
    for s in range(0, Npix, chunk_size):
        e = min(s + chunk_size, Npix)
        q_chunk = flat_yx[:, s:e, :].detach().clone().requires_grad_(True)  # (1,M,2) (y_n, x_n)
        y_n, x_n = q_chunk[..., 0], q_chunk[..., 1]        # (1,M)
        u_pix = ((x_n + 1) * W - 1) / 2.0                  # (1,M)
        v_pix = ((y_n + 1) * H - 1) / 2.0                  # (1,M)
        xy1   = torch.stack([u_pix, v_pix, torch.ones_like(u_pix)], dim=-1)  # (1,M,3)
        dir_cam = torch.einsum("ij,bmj->bmi", K_inv, xy1)  # (1,M,3)

        z_chunk, _ = model.inference(image=image, query_coord=q_chunk, prompt_depth=prompt)  # (1,M,1)
        z_chunk = z_chunk.reshape(1, -1)                   # (1,M)
        X_chunk = z_chunk[..., None] * dir_cam             # (1,M,3)
        grads = []
        for c in range(3):
            g = torch.ones_like(X_chunk[:, :, c])          # (1,M)
            grad_c = torch.autograd.grad(
                outputs=X_chunk[:, :, c],   # (1,M)
                inputs=q_chunk,             # (1,M,2)  (y_n, x_n)
                grad_outputs=g,             # (1,M)
                create_graph=False,
                retain_graph=True,
                only_inputs=True
            )[0][0]  # (M,2)
            grads.append(grad_c.unsqueeze(1))     # (M,1,2)
        grad_full = torch.cat(grads, dim=1)       # (M,3,2)
        dX_du = grad_full[:, :, 1] * (2.0 / Wf)   # (M,3) ∂/∂u
        dX_dv = grad_full[:, :, 0] * (2.0 / Hf)   # (M,3) ∂/∂v
        
        n_cross = torch.cross(dX_du, dX_dv, dim=-1)    # (M,3)
        n_chunk = n_cross / (n_cross.norm(dim=-1, keepdim=True) + 1e-6)
        n_out[s:e] = n_chunk

        del q_chunk, y_n, x_n, u_pix, v_pix, xy1, dir_cam, z_chunk, X_chunk, grads, grad_full, dX_dv, dX_du, n_cross, n_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n = n_out.view(H, W, 3)  

    X_norm = torch.norm(X_full, dim=-1, keepdim=True) + eps
    v_dir  = -X_full / X_norm                                           # (H,W,3)
    cos_theta = torch.abs(torch.sum(n * v_dir, dim=-1)).clamp_min(eps)  # (H,W)

    if vis_normal and (normal_save_path is not None):
        try:
            visualize_normal(n, normal_save_path)
        except Exception:
            pass
        try:
            diff_normal = depth_to_normal(z_full.detach().cpu().numpy(), K.detach().cpu().numpy())
            visualize_normal(diff_normal, normal_save_path.replace('.png', '_diff_normal.png'))
        except Exception:
            pass

    z_sq = z_full ** 2
    w = z_sq / cos_theta   # (H,W)
    w_clamped = torch.clamp(w, w_min, w_max)
    p = w_clamped / torch.sum(w_clamped)

    flat_p = p.reshape(-1)
    cell_indices = torch.multinomial(flat_p, num_samples=N, replacement=True)  # (N,)

    y_idx = (cell_indices // W)
    x_idx = (cell_indices %  W)

    x_centers = (x_idx.float() + 0.5)
    y_centers = (y_idx.float() + 0.5)
    offsets_x = torch.rand_like(x_centers, dtype=dtype) - 0.5
    offsets_y = torch.rand_like(y_centers, dtype=dtype) - 0.5
    samples_x = x_centers + offsets_x
    samples_y = y_centers + offsets_y

    samples_norm_x = ((samples_x + 0.5) / W) * 2.0 - 1.0  # x
    samples_norm_y = ((samples_y + 0.5) / H) * 2.0 - 1.0  # y
    xy_samples = torch.stack([samples_norm_y, samples_norm_x], dim=-1)  # (N,2) → (y_n, x_n)

    return xy_samples



if __name__ == "__main__":
    H, W = 504, 672

    depth = torch.linspace(1.0, 4.0, steps=H*W, dtype=torch.float32).reshape(H, W)
    depth[0, 0] = float('nan')
    depth[1, 1] = 0.0
    depth[2, 2] = float('inf')

    grid_depth = depth.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    fx = fy = 50.0
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    samples_norm = make_3d_uniform_coord_zplane(
        grid_depth,
        fx, fy, cx, cy,
        num_samples=500000,
        rng_seed=123,
        method='deterministic',
        min_per_nonzero_cell=0,
        device='cpu'
    )

    print("samples_norm.shape:", samples_norm.shape)
    print("first 8 normalized coords (x,y):\n", samples_norm[:8])

    px_x = ((samples_norm[:, 0] + 1.0) * (W / 2.0)) - 0.5
    px_y = ((samples_norm[:, 1] + 1.0) * (H / 2.0)) - 0.5
    px_back = torch.stack([px_x, px_y], dim=-1)
    print("reconstructed pixels (first 8):\n", px_back[:8])

    assert torch.all(px_back[:, 0] >= -0.5 - 1e-6) and torch.all(px_back[:, 0] <= W - 0.5 + 1e-6)
    assert torch.all(px_back[:, 1] >= -0.5 - 1e-6) and torch.all(px_back[:, 1] <= H - 0.5 + 1e-6)
    print("sanity checks passed")


SAMPLING_METHODS = {
    "2d_uniform": make_2d_uniform_coord,
    "zplane": make_3d_uniform_coord_zplane,
    "triangle_area": make_3d_uniform_coord_triangle,
    "delta_area": make_3d_uniform_coord_delta,
    "surface_poisson": make_3d_uniform_coord_surface_poisson,
    "quadtree": make_3d_uniform_coord_quadtree,
    "quadtree_random": make_3d_uniform_coord_quadtree_random,
}
