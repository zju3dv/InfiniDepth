import torch
import numpy as np
import typing
from .vis_utils import visualize_normal
from typing import Optional


def sample_by_equal_mass_inverse_cdf(p_map, N):
    flat_p = p_map.reshape(-1)
    cdf = torch.cumsum(flat_p, dim=0)
    cdf = cdf / cdf[-1]
    q = (torch.arange(N, device=flat_p.device, dtype=flat_p.dtype) + 0.5) / N
    idx = torch.searchsorted(cdf, q, right=True).clamp_max(flat_p.numel()-1)
    return idx 


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


def _depth_to_vertices(D, fx, fy, cx, cy):
    """
    D: (h,w) torch.Tensor depth in meters
    Return V: (h,w,3) torch.Tensor
    """
    h, w = D.shape
    device = D.device
    js = torch.arange(w, device=device, dtype=torch.float32)
    is_ = torch.arange(h, device=device, dtype=torch.float32)
    jj, ii = torch.meshgrid(js, is_, indexing='xy')  # jj: (h,w), ii: (h,w)
    
    Z = D
    X = (jj - cx) / fx * Z
    Y = (ii - cy) / fy * Z
    V = torch.stack([X, Y, Z], dim=-1)  # (h,w,3)
    return V


def _build_faces(h, w, device):
    """Return faces: (2*(h-1)*(w-1), 3) of flattened vertex indices as torch.Tensor."""
    idx = torch.arange(h * w, device=device).reshape(h, w)
    f1 = torch.stack([idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:]], dim=-1).reshape(-1, 3)
    f2 = torch.stack([idx[1:, 1:], idx[:-1, 1:], idx[1:, :-1]], dim=-1).reshape(-1, 3)
    return torch.cat([f1, f2], dim=0)


def _prune_faces_by_mask_and_edge(Vflat, faces, sky_mask_flat=None, max_edge=None):
    """Prune triangles that touch sky-mask vertices or exceed the maximum edge length."""
    keep = torch.ones(len(faces), dtype=torch.bool, device=faces.device)

    if sky_mask_flat is not None:
        keep &= ~sky_mask_flat[faces].any(dim=1)

    if max_edge is not None:
        A = Vflat[faces[:, 0]]
        B = Vflat[faces[:, 1]]
        C = Vflat[faces[:, 2]]
        e0 = torch.norm(B - A, dim=1)
        e1 = torch.norm(C - B, dim=1)
        e2 = torch.norm(A - C, dim=1)
        keep &= (torch.max(torch.max(e0, e1), e2) < max_edge)

    return faces[keep]

def _prune_faces(Vflat, faces, depth_ratio=1.05, max_edge=None, depth_ratio_far=1.10):
    """
    Prune triangles that likely cross depth discontinuities or are too large.
    All operations in PyTorch.
    """
    A = Vflat[faces[:, 0]]  # (N, 3)
    B = Vflat[faces[:, 1]]
    C = Vflat[faces[:, 2]]
    
    zA, zB, zC = A[:, 2], B[:, 2], C[:, 2]
    zmin = torch.min(torch.min(zA, zB), zC)
    zmax = torch.max(torch.max(zA, zB), zC)

    zmean = (zA + zB + zC) / 3.0
    
    log_z = torch.log10(zmean.clamp(min=1.0))  # log10(1)=0, log10(10)=1, log10(100)=2
    alpha = torch.clamp(log_z / 2.0, 0.0, 1.0)  # 0 @ 1m, 0.5 @ 10m, 1.0 @ 100m
    adaptive_ratio = depth_ratio + (depth_ratio_far - depth_ratio) * alpha
    
    keep = (zmin > 0) & (zmax / torch.clamp(zmin, min=1e-9) < adaptive_ratio)
    
    if max_edge is not None:
        e0 = torch.norm(B - A, dim=1)
        e1 = torch.norm(C - B, dim=1)
        e2 = torch.norm(A - C, dim=1)
        keep &= (torch.max(torch.max(e0, e1), e2) < max_edge)
    
    return faces[keep]

def _faces_to_ij(faces, h, w):
    """For each vertex index -> (i,j). Returns torch.Tensor."""
    i = faces // w
    j = faces % w
    return i, j


def make_3d_uniform_coord_triangle(
    depth_hw: torch.Tensor, 
    fx: float, 
    fy: float, 
    cx: float, 
    cy: float, 
    N: int, 
    coord_norm: str = "minus_one_to_one",
    sample_filter_mode: typing.Literal["none", "max_depth", "sky_mask"] = "max_depth",
    depth_ratio: float = 1.05, 
    sky_mask_hw: Optional[torch.Tensor] = None,
    max_edge: Optional[float] = None,
    max_depth_margin: float = 0.9,
    deterministic: bool = True,
) -> torch.Tensor:
    """
    Triangle-based area-weighted sampling for depth maps (PyTorch version).
    
    Args:
        depth_hw: (h,w) torch.Tensor predicted depth on the resized image grid
        fx, fy, cx, cy: Camera intrinsics
        N: Number of samples to generate
        coord_norm: 'minus_one_to_one' -> [-1,1] (default); 'zero_one' -> [0,1]
        sample_filter_mode: Sample-level filter policy. One of:
                            'none' -> only depth-discontinuity pruning
                            'max_depth' -> depth-discontinuity pruning + max-depth pruning
                            'sky_mask' -> depth-discontinuity pruning + sky-mask pruning
        depth_ratio: Max ratio for pruning discontinuous triangles
        sky_mask_hw: Optional boolean mask with shape (h, w). When provided, any triangle
                     touching sky vertices is removed when sample_filter_mode='sky_mask'
        max_edge: Max edge length for pruning large triangles
        max_depth_margin: Multiplier for auto-computed max depth when sample_filter_mode='max_depth'
        deterministic: If True, use deterministic sampling (centroid of each triangle).
                       This produces more regular/structured point clouds.
        
    Returns:
        coords: (N, 2) torch.Tensor in normalized coordinates [y, x]
    """
    device = depth_hw.device
    h, w = depth_hw.shape

    V = _depth_to_vertices(depth_hw, fx, fy, cx, cy)  # (h,w,3)
    Vflat = V.reshape(-1, 3)
    faces = _build_faces(h, w, device)

    faces = _prune_faces(Vflat, faces, depth_ratio=depth_ratio, max_edge=max_edge)

    if sample_filter_mode == "sky_mask":
        if sky_mask_hw is None:
            raise ValueError("`sky_mask_hw` is required when `sample_filter_mode='sky_mask'`.")
        if sky_mask_hw.shape != depth_hw.shape:
            raise ValueError(
                f"sky_mask_hw shape {tuple(sky_mask_hw.shape)} must match depth_hw shape {tuple(depth_hw.shape)}."
            )
        sky_mask_flat = sky_mask_hw.to(device=device, dtype=torch.bool).reshape(-1)
        faces = faces[~sky_mask_flat[faces].any(dim=1)]
    elif sample_filter_mode == "max_depth":
        A_temp = Vflat[faces[:, 0]]
        B_temp = Vflat[faces[:, 1]]
        C_temp = Vflat[faces[:, 2]]
        zA, zB, zC = A_temp[:, 2], B_temp[:, 2], C_temp[:, 2]
        zmax_per_tri = torch.max(torch.max(zA, zB), zC)
        depth_mask = zmax_per_tri < 90
        faces = faces[depth_mask]
    elif sample_filter_mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported sample_filter_mode: {sample_filter_mode}")

    if len(faces) == 0:
        if sample_filter_mode == "sky_mask":
            raise RuntimeError("All faces pruned; relax 'sky_mask_hw' coverage, 'depth_ratio', or 'max_edge'.")
        if sample_filter_mode == "max_depth":
            raise RuntimeError("All faces pruned; relax 'depth_ratio', 'max_edge', or 'max_depth_margin'.")
        raise RuntimeError("All faces pruned; relax 'depth_ratio' or 'max_edge'.")

    A = Vflat[faces[:, 0]]
    B = Vflat[faces[:, 1]]
    C = Vflat[faces[:, 2]]
    cross_product = torch.cross(B - A, C - A, dim=-1)
    areas = 0.5 * torch.norm(cross_product, dim=-1)
    areas = torch.clamp(areas, min=0.0)
    total_area = areas.sum()
    if not torch.isfinite(total_area) or total_area <= 0:
        raise RuntimeError("Invalid total area; check depth values.")
    probs = areas / total_area

    num_faces = len(faces)
    
    if deterministic:
        
        fi_all, fj_all = _faces_to_ij(faces, h, w)  # each (num_faces, 3)
        fi_all = fi_all.float()
        fj_all = fj_all.float()
        
        i_centroids = (fi_all[:, 0] + fi_all[:, 1] + fi_all[:, 2]) / 3.0
        j_centroids = (fj_all[:, 0] + fj_all[:, 1] + fj_all[:, 2]) / 3.0
        
        if N <= num_faces:
            top_indices = torch.argsort(-areas)[:N]
            i_s = i_centroids[top_indices]
            j_s = j_centroids[top_indices]
        else:
            base_i = i_centroids  # (num_faces,)
            base_j = j_centroids  # (num_faces,)
            
            remaining_points = N - num_faces
            
            if remaining_points <= 0:
                i_s = base_i
                j_s = base_j
            else:
                sqrt_areas = torch.sqrt(areas)
                sqrt_probs = sqrt_areas / sqrt_areas.sum()
                
                extra_raw = sqrt_probs * remaining_points
                extra_counts = torch.floor(extra_raw).to(torch.int64)
                
                still_remaining = remaining_points - extra_counts.sum().item()
                if still_remaining > 0:
                    frac = extra_raw - extra_counts.float()
                    top_frac_idx = torch.argsort(-frac)[:int(still_remaining)]
                    extra_counts[top_frac_idx] += 1
                
                max_subdiv = 50
                bary_list = []
                for n in range(1, max_subdiv + 1):
                    for a in range(n + 1):
                        for b in range(n + 1 - a):
                            c = n - a - b
                            w0, w1, w2 = a / n, b / n, c / n
                            if abs(w0 - 1/3) < 0.01 and abs(w1 - 1/3) < 0.01:
                                continue
                            bary_list.append((w0, w1, w2))
                            if len(bary_list) >= 2000:
                                break
                        if len(bary_list) >= 2000:
                            break
                    if len(bary_list) >= 2000:
                        break
                
                bary_template = torch.tensor(bary_list, dtype=torch.float32, device=device)  # (M, 3)
                max_extra_per_tri = len(bary_list)
                
                has_extra = extra_counts > 0
                extra_tri_indices = torch.nonzero(has_extra, as_tuple=True)[0]
                extra_tri_counts = extra_counts[has_extra]
                
                if len(extra_tri_indices) > 0:
                    tri_expanded = extra_tri_indices.repeat_interleave(extra_tri_counts)  # (total_extra,)
                    
                    total_extra = int(extra_tri_counts.sum().item())
                    cumsum = extra_tri_counts.cumsum(0)
                    offsets = torch.zeros_like(cumsum)
                    offsets[1:] = cumsum[:-1]
                    offsets_expanded = offsets.repeat_interleave(extra_tri_counts)
                    global_idx = torch.arange(total_extra, device=device)
                    point_local_idx = global_idx - offsets_expanded
                    
                    point_local_idx = point_local_idx.clamp(max=max_extra_per_tri - 1)
                    
                    bary_w = bary_template[point_local_idx]  # (total_extra, 3)
                    w0, w1, w2 = bary_w[:, 0], bary_w[:, 1], bary_w[:, 2]
                    
                    i0 = fi_all[tri_expanded, 0]
                    i1 = fi_all[tri_expanded, 1]
                    i2 = fi_all[tri_expanded, 2]
                    j0 = fj_all[tri_expanded, 0]
                    j1 = fj_all[tri_expanded, 1]
                    j2 = fj_all[tri_expanded, 2]
                    
                    extra_i = w0 * i0 + w1 * i1 + w2 * i2
                    extra_j = w0 * j0 + w1 * j1 + w2 * j2
                    
                    i_s = torch.cat([base_i, extra_i])
                    j_s = torch.cat([base_j, extra_j])
                else:
                    i_s = base_i
                    j_s = base_j
        
    else:
        tri_idx = torch.multinomial(probs, num_samples=N, replacement=True)  # (N,)
        f = faces[tri_idx]  # (N, 3)

        u = torch.rand(N, device=device)
        v = torch.rand(N, device=device)
        mask = (u + v > 1.0)
        u[mask] = 1.0 - u[mask]
        v[mask] = 1.0 - v[mask]
        
        w0 = 1.0 - u - v  # (N,)
        w1 = u
        w2 = v

        fi, fj = _faces_to_ij(f, h, w)  # each (N,3)
        i0, i1, i2 = fi[:, 0], fi[:, 1], fi[:, 2]
        j0, j1, j2 = fj[:, 0], fj[:, 1], fj[:, 2]

        i_s = w0 * i0.float() + w1 * i1.float() + w2 * i2.float()  # (N,)
        j_s = w0 * j0.float() + w1 * j1.float() + w2 * j2.float()  # (N,)
    
    # normalize (align_corners=False)
    if coord_norm == "minus_one_to_one":
        x = 2.0 * ((j_s + 0.5) / w) - 1.0
        y = 2.0 * ((i_s + 0.5) / h) - 1.0
    elif coord_norm == "zero_one":
        x = (j_s + 0.5) / w
        y = (i_s + 0.5) / h
    else:
        x = 2.0 * ((j_s + 0.5) / w) - 1.0
        y = 2.0 * ((i_s + 0.5) / h) - 1.0

    coords = torch.stack([y, x], dim=-1)  # (N, 2) [y, x] format
    return coords


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
        q_chunk = flat_yx[:, s:e, :].detach().clone().requires_grad_(True) 
        y_n, x_n = q_chunk[..., 0], q_chunk[..., 1]     
        u_pix = ((x_n + 1) * W - 1) / 2.0                 
        v_pix = ((y_n + 1) * H - 1) / 2.0                 
        xy1   = torch.stack([u_pix, v_pix, torch.ones_like(u_pix)], dim=-1)  
        dir_cam = torch.einsum("ij,bmj->bmi", K_inv, xy1) 

        z_chunk, _ = model.inference(image=image, query_coord=q_chunk, prompt_depth=prompt) 
        z_chunk = z_chunk.reshape(1, -1)                  
        X_chunk = z_chunk[..., None] * dir_cam             
        grads = []
        for c in range(3):
            g = torch.ones_like(X_chunk[:, :, c])         
            grad_c = torch.autograd.grad(
                outputs=X_chunk[:, :, c],  
                inputs=q_chunk,             
                grad_outputs=g,            
                create_graph=False,        
                retain_graph=True,         
                only_inputs=True
            )[0][0]  # (M,2)
            grads.append(grad_c.unsqueeze(1))    
        grad_full = torch.cat(grads, dim=1)       
        dX_du = grad_full[:, :, 1] * (2.0 / Wf)   
        dX_dv = grad_full[:, :, 0] * (2.0 / Hf)  
        
        n_cross = torch.cross(dX_du, dX_dv, dim=-1)   
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

    cell_indices = sample_by_equal_mass_inverse_cdf(p, N)   # (N,)

    y_idx = (cell_indices // W)  # row
    x_idx = (cell_indices %  W)  # column

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



SAMPLING_METHODS = {
    "2d_uniform": make_2d_uniform_coord,
    "3d_uniform":  make_3d_uniform_coord_autograd,
    "3d_uniform_triangle": make_3d_uniform_coord_triangle,
}
