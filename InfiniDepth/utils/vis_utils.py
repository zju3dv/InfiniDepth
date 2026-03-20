import matplotlib
import numpy as np
import torch
import imageio
from PIL import Image
import copy
import cv2
import numpy as np
import onnxruntime


def visualize_normal(n, save_path="normal_map.png"):
    n_vis = (n + 1) / 2.0
    n_vis = n_vis.clamp(0, 1) 
    n_vis = n_vis.detach().cpu().numpy()
    
    # Save as uint8 PNG
    imageio.imwrite(save_path, (n_vis * 255).astype("uint8"))


def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
    assert len(depth_map.shape) >= 2, "Invalid dimension"
    input_depth_map = depth_map

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()

    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return (img_colored_np[0].transpose(1, 2, 0) * 255.0).astype(np.uint8)


def clip_outliers_by_percentile(depthmap: torch.Tensor,
                                lower_q: float = 0.01,
                                upper_q: float = 0.99,
                                min_positive: float = 1e-8):
    """
    depthmap: torch tensor of shape [1,1,H,W] or [H,W]
    Returns: clipped depthmap and (lo, hi)
    """
    if depthmap.ndim == 4:
        d = depthmap
    elif depthmap.ndim == 2:
        d = depthmap.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("depthmap shape must be [1,1,H,W] or [H,W]")

    valid = torch.isfinite(d) & (d > 0)
    if valid.sum() < 10:
        return depthmap, (None, None)

    vals = d[valid]
    lo = torch.quantile(vals, lower_q)
    hi = torch.quantile(vals, upper_q)
    lo = torch.clamp(lo, min=min_positive)
    hi = torch.clamp(hi, min=lo + 1e-8)

    d_clamped = torch.clamp(d, min=lo.item(), max=hi.item())
    return d_clamped.view_as(depthmap), (lo.item(), hi.item())


def build_sky_model(model_path="checkpoints/skyseg.onnx"):
    return onnxruntime.InferenceSession(model_path)


def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        np.ndarray: Segmentation mask
    """

    # Pre process:Resize, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image.cpu().numpy().squeeze().transpose(1,2,0), dsize=(input_size[0], input_size[1]))
    x = np.array(resize_image, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result


def blender_img_and_sky_mask(image_bgr: np.ndarray, sky_mask_gray: np.ndarray, alpha: float = 0.3):
    if sky_mask_gray.shape[:2] != image_bgr.shape[:2]:
        sky_mask_gray = cv2.resize(sky_mask_gray, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    if len(sky_mask_gray.shape) == 2:
        sky_mask_gray = cv2.cvtColor(sky_mask_gray, cv2.COLOR_GRAY2BGR)

    img_f = image_bgr.astype(np.float32)
    mask_f = sky_mask_gray.astype(np.float32)

    blended = cv2.addWeighted(img_f, 1.0, mask_f, alpha, 0)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


def save_sky_mask_gray(sky_mask_gray: np.ndarray, save_path: str = "debug_mask.png"):
    if sky_mask_gray.dtype != np.uint8:
        sky_mask_gray = (np.clip(sky_mask_gray, 0, 1) * 255).astype(np.uint8)
    _, sky_binary = cv2.threshold(sky_mask_gray, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite("sky_mask_binary.png", sky_binary)      


def build_edge_band(img_bgr: np.ndarray, band_px=2):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0,0), 1.2)
    edges = cv2.Canny(gray, 50, 120, L2gradient=True)    # You can replace 50/120 with percentile-based thresholds
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*band_px+1, 2*band_px+1))
    band = cv2.dilate(edges, kernel, iterations=1) > 0
    return band.astype(np.bool_)
