import os
from typing import Optional

import cv2
import matplotlib
import numpy as np


def depth_to_jet(depth):
    """depth: [H,W] in [0,1], return [H,W,3] in [0,1]"""
    H, W = depth.shape
    c = np.zeros((H, W, 3), dtype=np.float32)
    four_value = 4 * depth
    c[..., 0] = np.clip(four_value - 1.5, 0, 1)
    c[..., 1] = np.clip(1.5 - np.abs(four_value - 2), 0, 1)
    c[..., 2] = np.clip(2.5 - four_value, 0, 1)
    return c


def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral"):
    """
    Colorize depth maps.
    """
    depth = depth_map.copy()
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth + 1e-6)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, 0:3]
    return img_colored_np


def visualize_and_save(depth: np.ndarray, coords: np.ndarray, heatmap: np.ndarray, save_dir="vis"):
    """
    depth: [H,W], normalized to [0,1]
    coords: [N,2] (y,x) pixel coordinates
    heatmap: [H,W], normalized to [0,1]
    """
    H, W = depth.shape

    depth_color = colorize_depth_maps(depth, float(depth.min()), float(depth.max()), cmap="Spectral")

    heatmap_img = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_img = heatmap_img[..., ::-1].astype(np.float32) / 255.0

    mask = np.zeros((H, W), dtype=np.float32)
    y = np.clip(coords[:, 0].astype(int), 0, H - 1)
    x = np.clip(coords[:, 1].astype(int), 0, W - 1)
    mask[y, x] = 1.0

    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1).astype(np.float32)
    gray_img = np.repeat(mask_dilated[..., None], 3, axis=2)

    sampled_depth = depth * mask_dilated
    sampled_depth_color = colorize_depth_maps(sampled_depth, float(depth.min()), float(depth.max()), cmap="Spectral")

    vis = np.concatenate(
        [
            (depth_color * 255).astype(np.uint8),
            (heatmap_img * 255).astype(np.uint8),
            (gray_img * 255).astype(np.uint8),
            (sampled_depth_color * 255).astype(np.uint8),
        ],
        axis=1,
    )

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, "debug.png"), vis[..., ::-1])


def _as_numpy_f32(x, name: str) -> Optional[np.ndarray]:
    """Accepts numpy array or torch tensor-like; returns np.float32 or None."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    raise TypeError(f"{name} must be np.ndarray or torch.Tensor-like, got {type(x)}")


def _normalize01(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m, M = float(np.min(img)), float(np.max(img))
    if M - m < eps:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - m) / (M - m)).astype(np.float32)


def _sobel_grad_mag(img: np.ndarray) -> np.ndarray:
    """Gradient magnitude via Sobel (3x3)."""
    gx = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    return np.hypot(gx, gy)


def _laplacian_abs(img: np.ndarray) -> np.ndarray:
    """4-neighborhood Laplacian to match [[0,1,0],[1,-4,1],[0,1,0]]."""
    kernel = np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    l = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_REFLECT_101)
    return np.abs(l)


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)


def _avg_pool2d_same_exclude_pad(img: np.ndarray, ksize: int) -> np.ndarray:
    """Average pooling with stride=1 and count_include_pad=False."""
    if ksize <= 1:
        return img
    H, W = img.shape
    r = ksize // 2
    S = np.cumsum(np.cumsum(img, axis=0), axis=1)
    S = np.pad(S, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    ones = np.ones_like(img, dtype=np.float32)
    C = np.cumsum(np.cumsum(ones, axis=0), axis=1)
    C = np.pad(C, ((1, 0), (1, 0)), mode="constant", constant_values=0)

    ys = np.arange(H)
    xs = np.arange(W)
    y0 = np.maximum(ys - r, 0)
    y1 = np.minimum(ys + r + 1, H)
    x0 = np.maximum(xs - r, 0)
    x1 = np.minimum(xs + r + 1, W)

    sum_win = (
        S[y1[:, None], x1[None, :]]
        - S[y0[:, None], x1[None, :]]
        - S[y1[:, None], x0[None, :]]
        + S[y0[:, None], x0[None, :]]
    )
    cnt_win = (
        C[y1[:, None], x1[None, :]]
        - C[y0[:, None], x1[None, :]]
        - C[y1[:, None], x0[None, :]]
        + C[y0[:, None], x0[None, :]]
    )
    return (sum_win / np.maximum(cnt_win, 1.0)).astype(np.float32)


def _geom_energy(
    img_hw: np.ndarray,
    geom_scales=(0.6, 1.2),
    grad_weight=0.0,
    lap_weight=1.0,
    percentile=0.98,
    eps=1e-6,
) -> np.ndarray:
    """
    img_hw: [H,W] float32 single-channel
    return: [H,W] float32 energy map in [0,1]
    """
    del eps
    img = img_hw.astype(np.float32, copy=False)
    H, W = img.shape
    g_acc = np.zeros((H, W), dtype=np.float32)
    l_acc = np.zeros((H, W), dtype=np.float32)

    for s in geom_scales:
        xs = _gaussian_blur(img, float(s)) if s > 0 else img
        g = _sobel_grad_mag(xs)
        l = _laplacian_abs(xs)
        np.maximum(g_acc, g, out=g_acc)
        np.maximum(l_acc, l, out=l_acc)

    qg = float(np.quantile(g_acc, percentile))
    ql = float(np.quantile(l_acc, percentile))
    qg = max(qg, 1e-6)
    ql = max(ql, 1e-6)
    g_acc = np.clip(g_acc / qg, 0.0, 1.0)
    l_acc = np.clip(l_acc / ql, 0.0, 1.0)

    return (grad_weight * g_acc + lap_weight * l_acc).astype(np.float32)


def _combine_energy(e_depth: np.ndarray, e_disp: Optional[np.ndarray], mode: str) -> np.ndarray:
    if e_disp is None:
        return e_depth
    if mode == "depth":
        return e_depth
    if mode == "disparity":
        return e_disp
    if mode == "sum":
        return e_depth + e_disp
    return np.maximum(e_depth, e_disp)


def importance_sampling(
    depth: np.ndarray,
    disparity: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    disparity_mask: Optional[np.ndarray] = None,
    n: int = 1024,
    dd_combine: str = "max",
    smooth_ksize: int = 0,
    temperature: float = 0.8,
    debug: bool = False,
):
    """
    Multi-scale geometric importance sampling (NumPy + OpenCV).
    """
    depth = _as_numpy_f32(depth, "depth")
    disparity = _as_numpy_f32(disparity, "disparity")
    mask = _as_numpy_f32(mask, "mask")
    disparity_mask = _as_numpy_f32(disparity_mask, "disparity_mask")

    H, W = depth.shape

    if mask is None:
        valid = np.ones((H, W), dtype=bool)
    else:
        valid = mask > 0
    if disparity_mask is not None:
        valid &= disparity_mask > 0

    e_depth = _geom_energy(depth)
    e_disp = _geom_energy(disparity) if disparity is not None else None
    score = _combine_energy(e_depth, e_disp, dd_combine)
    score = np.where(valid, score, 0.0).astype(np.float32)

    if isinstance(smooth_ksize, int) and smooth_ksize > 1:
        score = _avg_pool2d_same_exclude_pad(score, smooth_ksize)

    if temperature != 1.0:
        score = np.power(np.clip(score, 0.0, None), 1.0 / float(temperature))

    probs = score.reshape(-1)
    sum_w = float(probs.sum())
    valid_num = int(valid.sum())

    if sum_w <= 0.0:
        if valid_num > 0:
            probs = valid.reshape(-1).astype(np.float32)
            probs /= float(probs.sum())
        else:
            probs = np.full(H * W, 1.0 / (H * W), dtype=np.float32)
    else:
        probs = (probs / sum_w).astype(np.float32)

    replacement = True if valid_num == 0 else (n > valid_num)
    idx = np.random.choice(H * W, size=int(n), replace=replacement, p=probs)
    y = (idx // W).astype(np.int64)
    x = (idx % W).astype(np.int64)

    if debug:
        depth01 = _normalize01(depth)
        score01 = _normalize01(score)
        coords = np.stack([y, x], axis=-1)
        visualize_and_save(depth01, coords, score01)

    y_norm = 2.0 * (y.astype(np.float32) + 0.5) / float(H) - 1.0
    x_norm = 2.0 * (x.astype(np.float32) + 0.5) / float(W) - 1.0
    coords_depth = np.stack([y_norm, x_norm], axis=-1)
    coords_disparity = coords_depth.copy()

    depth_flat = depth[y, x].reshape(-1, 1)
    disparity_flat = disparity[y, x].reshape(-1, 1) if disparity is not None else None

    return coords_depth, depth_flat, coords_disparity, disparity_flat


def pixel_sampling(depth, disparity, mask, disparity_mask):
    """Convert depth and disparity maps to pixel samples (NumPy version)."""
    h, w = depth.shape

    def make_coord_seq(n, v0=-1, v1=1):
        r = (v1 - v0) / (2 * n)
        return v0 + r + (2 * r) * np.arange(n, dtype=np.float32)

    ys = make_coord_seq(h)
    xs = make_coord_seq(w)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    coord = np.stack([yy, xx], axis=-1).reshape(-1, 2)

    depth = depth.reshape(-1, 1)
    disparity = disparity.reshape(-1, 1)
    mask = mask.reshape(-1)
    disparity_mask = disparity_mask.reshape(-1)

    coord_depth = coord.copy()
    depth = depth.copy()
    coord_disparity = coord.copy()
    disparity = disparity.copy()

    return coord_depth, depth, coord_disparity, disparity

