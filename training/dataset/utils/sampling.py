import torch 
import os
import random
import cv2
import numpy as np
import math
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Sequence, Tuple
import matplotlib


def pixel_sampling(depth, disparity, mask, disparity_mask):
    h, w = depth.shape

    def make_coord_seq(n, v0=-1, v1=1):
        r = (v1 - v0) / (2 * n)
        return v0 + r + (2 * r) * np.arange(n, dtype=np.float32)

    ys = make_coord_seq(h)
    xs = make_coord_seq(w)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')  # shape [H, W]
    coord = np.stack([yy, xx], axis=-1).reshape(-1, 2)  # yy,xx --> H, W

    depth = depth.reshape(-1, 1)
    disparity = disparity.reshape(-1, 1)
    mask = mask.reshape(-1)
    disparity_mask = disparity_mask.reshape(-1)

    coord_depth = coord.copy()
    depth = depth.copy()
    coord_disparity = coord.copy()
    disparity = disparity.copy()

    return coord_depth, depth, coord_disparity, disparity


sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
lap_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32).view(1,1,3,3)

def _sobel_grad_mag(x: torch.Tensor) -> torch.Tensor:
    # reflect padding
    x_pad = F.pad(x, (1,1,1,1), mode="reflect")
    gx = F.conv2d(x_pad, sobel_x.to(x.device))
    gy = F.conv2d(x_pad, sobel_y.to(x.device))
    return torch.sqrt(gx**2 + gy**2)

# --- Laplacian ---
def _laplacian_abs(x: torch.Tensor) -> torch.Tensor:
    x_pad = F.pad(x, (1,1,1,1), mode="reflect")
    l = F.conv2d(x_pad, lap_kernel.to(x.device))
    return l.abs()

# --- Gaussian blur ---
def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    ksize = max(3, int(2*round(3*sigma)+1))
    ax = torch.arange(ksize, device=x.device) - ksize//2
    kernel = torch.exp(-0.5 * (ax.float()/sigma)**2)
    kernel = kernel / kernel.sum()
    k2d = (kernel[:,None] * kernel[None,:]).to(x.dtype)
    k2d = k2d.view(1,1,ksize,ksize)

    pad = ksize // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x_pad, k2d)

def _normalize01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xmin = x.min()
    xmax = x.max()
    return (x - xmin) / (xmax - xmin + eps)


# --- Jet colormap ---
def depth_to_jet(depth: torch.Tensor) -> torch.Tensor:
    """depth: [H,W] in [0,1]"""
    c = torch.zeros((3, *depth.shape), device=depth.device, dtype=depth.dtype)
    four_value = 4 * depth.unsqueeze(0)  # [1,H,W]
    c[0] = (four_value - 1.5).clamp(0, 1)                # R
    c[1] = (1.5 - (four_value - 2).abs()).clamp(0, 1)    # G
    c[2] = (2.5 - four_value).clamp(0, 1)                # B
    return c

# --- Draw black points on RGB ---
def draw_points(rgb01: torch.Tensor, coords: torch.Tensor, radius: int=2) -> torch.Tensor:
    _, H, W = rgb01.shape
    marker = torch.zeros((1,1,H,W), dtype=rgb01.dtype, device=rgb01.device)
    y = coords[:,0].long().clamp(0, H-1)
    x = coords[:,1].long().clamp(0, W-1)
    marker[0,0, y, x] = 1.0
    if radius > 0:
        marker = F.max_pool2d(marker, kernel_size=2*radius+1, stride=1, padding=radius)
    rgb01 = rgb01 * (1 - marker.squeeze(0))
    return rgb01.clamp(0,1)


def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral"):
    """
    Colorize depth maps.
    """
    depth = depth_map.copy()
    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth + 1e-6)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, 0:3]  # value from 0 to 1

    return img_colored_np


def visualize_and_save(depth, coords, heatmap, save_dir="vis"):
    H, W = depth.shape

    depth_color = colorize_depth_maps(depth.cpu().numpy(), depth.min().item(), depth.max().item(), cmap="Spectral") # [H,W,3]

    heatmap_img = cv2.applyColorMap((heatmap.cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_img = heatmap_img[..., ::-1] / 255.0   # BGR->RGB

    mask = torch.zeros((H,W), dtype=torch.float32, device=depth.device)
    y = coords[:,0].long().clamp(0,H-1)
    x = coords[:,1].long().clamp(0,W-1)
    mask[y,x] = 1.0
    mask = F.max_pool2d(mask.unsqueeze(0).unsqueeze(0),
                        kernel_size=3, stride=1, padding=1).squeeze()
    gray_img = mask.cpu().numpy()[...,None].repeat(3,axis=2)

    sampled_depth = (depth * mask).cpu().numpy()
    sampled_depth_color = colorize_depth_maps(sampled_depth, depth.min().item(), depth.max().item(), cmap="Spectral")

    vis = np.concatenate([
        (depth_color*255).astype(np.uint8),
        (heatmap_img*255).astype(np.uint8),
        (gray_img*255).astype(np.uint8),
        (sampled_depth_color*255).astype(np.uint8)
    ], axis=1)

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, "debug.png"), vis[..., ::-1])  # RGB->BGR


def importance_sampling(
    depth: torch.Tensor,                     # [H,W]
    disparity: Optional[torch.Tensor],       
    mask: Optional[torch.Tensor],            
    disparity_mask: Optional[torch.Tensor],  
    n: int,
    dd_combine: str = "max",        # "max" | "sum"
    smooth_ksize: int = 0,
    temperature: float = 1.0,
    debug: bool = False,
):
    def _to_tensor(x, name: str):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            return x.float()
        else:
            raise TypeError(f"{name} must be np.ndarray or torch.Tensor, got {type(x)}")

    depth = _to_tensor(depth, "depth")
    disparity = _to_tensor(disparity, "disparity")
    mask = _to_tensor(mask, "mask")
    disparity_mask = _to_tensor(disparity_mask, "disparity_mask")

    H, W = depth.shape

    if mask is None:
        valid = torch.ones_like(depth, dtype=torch.bool)
    else:
        valid = mask > 0
    if disparity_mask is not None:
        valid = valid & (disparity_mask > 0)

    def geom_energy(img_hw: torch.Tensor,
                    geom_scales=(0.0, 1.2, 2.4, 4.8),
                    grad_weight=0.65,
                    lap_weight=0.35,
                    percentile=0.95) -> torch.Tensor:
        img = img_hw[None,None]                   # [1,1,H,W]
        g_acc = torch.zeros_like(img)
        l_acc = torch.zeros_like(img)

        for s in geom_scales:
            xs = _gaussian_blur(img, float(s)) if s > 0 else img
            g = _sobel_grad_mag(xs)
            l = _laplacian_abs(xs)
            g_acc = torch.maximum(g_acc, g)
            l_acc = torch.maximum(l_acc, l)

        qg = torch.quantile(g_acc.view(-1), q=percentile).clamp_min(1e-6)
        ql = torch.quantile(l_acc.view(-1), q=percentile).clamp_min(1e-6)
        g_acc = (g_acc / qg).clamp(0,1)
        l_acc = (l_acc / ql).clamp(0,1)

        return (grad_weight * g_acc + lap_weight * l_acc).squeeze(0).squeeze(0)  # [H,W]

    def combine_energy(e_depth, e_disp, mode: str = 'max'):
        if mode == "depth":
            return e_depth
        if mode == "disparity":
            return e_disp
        if mode == "sum":
            return e_depth + e_disp
        if mode == 'max':
            return torch.maximum(e_depth, e_disp)
    
    e_depth = geom_energy(depth)
    if disparity is not None:
        e_disp = geom_energy(disparity)
        score = combine_energy(e_depth, e_disp, dd_combine)
    else:
        score = e_depth

    score = score.masked_fill(~valid, 0.0)        # [H,W]

    if smooth_ksize and smooth_ksize > 1:
        pad = smooth_ksize // 2
        score = F.avg_pool2d(score[None,None], kernel_size=smooth_ksize, stride=1,
                              padding=pad, count_include_pad=False).squeeze(0).squeeze(0)

    if temperature != 1.0:
        score = torch.pow(score, 1.0 / temperature)

    probs = score.view(-1)
    sum_w = probs.sum()
    valid_num = int(valid.sum().item())
    if sum_w <= 0:
        probs = valid.view(-1).float()
        probs = probs / probs.sum().clamp_min(1.0)
    else:
        probs = probs / sum_w

    replacement = True if valid_num == 0 else (n > valid_num)
    idx = torch.multinomial(probs, n, replacement=replacement)   # [n]
    y = idx // W
    x = idx %  W
    coords = torch.stack([y, x], dim=-1).long()                  # [n,2]

    if debug:
        depth01 = _normalize01(depth)                            # [H,W]
        score01 = _normalize01(score)                            # [H,W]
        visualize_and_save(depth01, coords, score01)
    
    y_norm = 2.0 * (y.float() + 0.5) / H - 1.0
    x_norm = 2.0 * (x.float() + 0.5) / W - 1.0
    coords_depth = torch.stack([y_norm, x_norm], dim=-1).cpu().numpy()
    coords_disparity = coords_depth.copy()

    depth_flat = depth[y, x].cpu().numpy().reshape(-1, 1)
    disparity_flat = disparity[y, x].cpu().numpy().reshape(-1, 1) if disparity is not None else None

    return coords_depth, depth_flat, coords_disparity, disparity_flat


def run_test(H=768, W=1024, n=10000, device="cuda"):
    depth = torch.rand(H, W, device=device)
    disparity = torch.rand(H, W, device=device)
    mask = torch.ones(H, W, device=device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
    ) as prof:
        with torch.profiler.record_function("importance_sampling_run"):
            coords_depth, depth_flat, coords_disp, disp_flat = importance_sampling(
                depth=depth,
                disparity=disparity,
                mask=mask,
                disparity_mask=None,
                n=n,
                dd_combine="max",
                smooth_ksize=5,
                temperature=1.0,
                debug=False,
            )
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    run_test(H=768, W=1024, n=20000, device="cuda")
