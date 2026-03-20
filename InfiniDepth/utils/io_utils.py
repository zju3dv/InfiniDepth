import torch
import imageio
import h5py
import open3d as o3d
import torch.nn.functional as F
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # Must be set before importing cv2
import cv2
import numpy as np
from typing import Any, Optional, Tuple
from PIL import Image, ImageOps
from scipy import ndimage
from .vis_utils import colorize_depth_maps
from .logger import Log


def filter_depth_noise_numpy(depth: np.ndarray,
                              depth_mask: np.ndarray,
                              std_threshold: float = 0.2,
                              median_threshold: float = 0.2,
                              gradient_threshold: float = 0.2,
                              min_neighbors: int = 7,
                              bilateral_d: int = 5,
                              bilateral_sigma_color: float = 0.5,
                              bilateral_sigma_space: float = 5.0) -> tuple:
    depth_np = depth.copy()
    mask_np = depth_mask.astype(bool).copy()

    valid_depths = depth_np[mask_np]

    if len(valid_depths) == 0:
        print("[Warning] No valid depth values found!")
        return depth, depth_mask

    mean_depth = np.mean(valid_depths)
    std_depth = np.std(valid_depths)
    lower_bound = mean_depth - std_threshold * std_depth
    upper_bound = mean_depth + std_threshold * std_depth

    outlier_mask = (depth_np < lower_bound) | (depth_np > upper_bound)
    mask_np[outlier_mask] = False
    depth_np[outlier_mask] = 0

    print(f"[Filter] Step 1 - Statistical outlier: removed {np.sum(outlier_mask)} points")
    print(f"[Filter] Valid depth range: [{lower_bound:.2f}m, {upper_bound:.2f}m]")

    if np.sum(mask_np) > 0:
        depth_for_median = depth_np.copy()
        depth_for_median[~mask_np] = 0

        median_filtered = ndimage.median_filter(depth_for_median, size=5)

        valid_median = median_filtered[mask_np]
        valid_original = depth_np[mask_np]

        relative_diff = np.abs(valid_original - valid_median) / (valid_median + 1e-6)

        local_consistency_mask = mask_np.copy()
        local_consistency_mask[mask_np] = relative_diff < median_threshold

        removed_inconsistent = np.sum(mask_np) - np.sum(local_consistency_mask)
        mask_np = local_consistency_mask
        depth_np[~mask_np] = 0

        print(f"[Filter] Step 2 - Local consistency: removed {removed_inconsistent} points")

    if np.sum(mask_np) > 0:
        depth_for_grad = depth_np.copy()
        depth_for_grad[~mask_np] = 0

        dx = ndimage.sobel(depth_for_grad, axis=1)
        dy = ndimage.sobel(depth_for_grad, axis=0)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)

        normalized_gradient = gradient_magnitude / (depth_np + 1e-6)

        high_gradient_mask = (normalized_gradient > gradient_threshold) & mask_np
        mask_np[high_gradient_mask] = False
        depth_np[high_gradient_mask] = 0

        print(f"[Filter] Step 3 - Gradient filter: removed {np.sum(high_gradient_mask)} points")

    if np.sum(mask_np) > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_count = ndimage.convolve(mask_np.astype(np.uint8), kernel, mode='constant')

        isolated_mask = (neighbor_count < min_neighbors) & mask_np
        mask_np[isolated_mask] = False
        depth_np[isolated_mask] = 0

        print(f"[Filter] Step 4 - Isolated points: removed {np.sum(isolated_mask)} points")

    if np.sum(mask_np) > 0:
        depth_for_bilateral = depth_np.copy().astype(np.float32)
        if np.sum(mask_np) > 100: 
            bilateral_filtered = cv2.bilateralFilter(
                depth_for_bilateral,
                d=bilateral_d,
                sigmaColor=bilateral_sigma_color,
                sigmaSpace=bilateral_sigma_space
            )
            depth_np[mask_np] = bilateral_filtered[mask_np]

            print(f"[Filter] Step 5 - Bilateral filtering applied")

    remaining_points = np.sum(mask_np)
    original_points = np.sum(depth_mask > 0)
    removed_ratio = (original_points - remaining_points) / max(original_points, 1) * 100

    print(f"[Filter] Summary: {original_points} -> {remaining_points} points ({removed_ratio:.1f}% removed)")

    return depth_np, mask_np.astype(np.float32)


def load_image(image_path: str,
               tar_size: Tuple[int, int] = (504, 672),
               ) -> torch.Tensor: 
    '''
    Load image and resize to target size.
    Args:
        image_path: Path to input image.
        tar_size: Target size (h, w).
    Returns:
        image: Image tensor with shape (1, 3, h, w).
    '''
    with Image.open(image_path) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        image = np.asarray(pil_image)
    h, w = image.shape[:2]
    org_img = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image = cv2.resize(image, tar_size[::-1], interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return org_img, image, (h, w)


def _to_single_channel_depth(depth: np.ndarray, depth_path: str) -> np.ndarray:
    depth = np.asarray(depth)
    depth = np.squeeze(depth)
    if depth.ndim == 3 and depth.shape[0] in (1, 3) and depth.shape[2] not in (1, 3):
        # Handle CHW arrays saved from tensor pipelines.
        depth = np.moveaxis(depth, 0, -1)
    if depth.ndim == 2:
        return depth.astype(np.float32)
    if depth.ndim == 3:
        if depth.shape[2] == 1:
            return depth[:, :, 0].astype(np.float32)
        if np.issubdtype(depth.dtype, np.integer) and depth.shape[2] >= 3:
            ch0 = depth[:, :, 0]
            ch1 = depth[:, :, 1]
            ch2 = depth[:, :, 2]
            if np.array_equal(ch0, ch1) and np.array_equal(ch1, ch2):
                return ch0.astype(np.float32)
            return (
                ch0.astype(np.float32) * (256.0 ** 2)
                + ch1.astype(np.float32) * 256.0
                + ch2.astype(np.float32)
            )
        return depth[:, :, 0].astype(np.float32)
    raise ValueError(f"Unsupported depth shape {depth.shape} for file: {depth_path}")


def _load_depth_from_png(depth_path: str) -> np.ndarray:
    raw = imageio.imread(depth_path)
    depth = _to_single_channel_depth(raw, depth_path)
    if np.issubdtype(raw.dtype, np.integer):
        # Common PNG depth encodings are in millimeters.
        if float(np.nanmax(depth)) > 255.0:
            depth = depth / 1000.0
    return depth.astype(np.float32)


def _load_depth_from_npz(depth_path: str) -> np.ndarray:
    with np.load(depth_path, allow_pickle=True) as npz_data:
        if len(npz_data.files) == 0:
            raise ValueError(f"Empty npz depth file: {depth_path}")
        preferred_keys = ("depth", "data", "depth_map", "arr_0")
        key = next((k for k in preferred_keys if k in npz_data.files), npz_data.files[0])
        depth = np.asarray(npz_data[key])
    return _to_single_channel_depth(depth, depth_path)


def _find_h5_dataset(node: h5py.Group) -> Optional[h5py.Dataset]:
    preferred_keys = ("dataset", "depth", "data")
    for key in preferred_keys:
        if key in node and isinstance(node[key], h5py.Dataset):
            return node[key]
    for key in node.keys():
        child = node[key]
        if isinstance(child, h5py.Dataset):
            return child
    for key in node.keys():
        child = node[key]
        if isinstance(child, h5py.Group):
            found = _find_h5_dataset(child)
            if found is not None:
                return found
    return None


def _load_depth_from_h5(depth_path: str) -> np.ndarray:
    with h5py.File(depth_path, "r") as f:
        dataset = _find_h5_dataset(f)
        if dataset is None:
            raise ValueError(f"No dataset found in h5 file: {depth_path}")
        depth = np.asarray(dataset)
    return _to_single_channel_depth(depth, depth_path)


def _decode_depth_dict(depth_obj: dict[str, Any], depth_path: str) -> np.ndarray:
    if "mask" in depth_obj and "value" in depth_obj:
        mask = np.asarray(depth_obj["mask"]).astype(bool)
        value = np.asarray(depth_obj["value"], dtype=np.float32)
        depth = np.zeros(mask.shape, dtype=np.float32)
        if value.shape == mask.shape:
            depth[mask] = value[mask]
        else:
            value_flat = value.reshape(-1)
            valid_count = int(mask.sum())
            if value_flat.size < valid_count:
                raise ValueError(
                    f"Depth dict value count ({value_flat.size}) smaller than mask valid count ({valid_count}) "
                    f"for file: {depth_path}"
                )
            depth[mask] = value_flat[:valid_count]
        return depth

    preferred_keys = ("depth", "data", "depth_map", "arr_0", "z")
    for key in preferred_keys:
        if key in depth_obj:
            return _to_single_channel_depth(np.asarray(depth_obj[key]), depth_path)

    if len(depth_obj) == 1:
        only_key = next(iter(depth_obj))
        return _to_single_channel_depth(np.asarray(depth_obj[only_key]), depth_path)

    raise ValueError(f"Unsupported npy depth dict keys {list(depth_obj.keys())} in file: {depth_path}")


def _load_depth_from_npy(depth_path: str) -> np.ndarray:
    loaded = np.load(depth_path, allow_pickle=True)
    obj = loaded

    # np.save(dict) + allow_pickle=True is loaded as a 0-d object array.
    if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
        obj = loaded.item()

    if isinstance(obj, dict):
        depth = _decode_depth_dict(obj, depth_path)
    else:
        depth = _to_single_channel_depth(np.asarray(obj), depth_path)
 
    return depth.astype(np.float32)


def _load_depth_from_exr(depth_path: str) -> np.ndarray:
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"Failed to read EXR depth file: {depth_path}")
    return _to_single_channel_depth(depth, depth_path)


def _read_depth_array(depth_path: str) -> np.ndarray:
    ext = os.path.splitext(depth_path)[1].lower()
    if ext == ".png":
        return _load_depth_from_png(depth_path)
    if ext == ".npz":
        return _load_depth_from_npz(depth_path)
    if ext in (".hdf5", ".h5"):
        return _load_depth_from_h5(depth_path)
    if ext == ".npy":
        return _load_depth_from_npy(depth_path)
    if ext == ".exr":
        return _load_depth_from_exr(depth_path)
    raise ValueError(f"Invalid depth_path extension `{ext}`: {depth_path}")


def read_depth_array(depth_path: str) -> np.ndarray:
    """Public wrapper for loading a raw depth array from supported file formats."""
    return _read_depth_array(depth_path)


def load_depth(depth_path: str,
               tar_size: Tuple[int, int] = (504, 672),
               num_samples: int = 1500,
               min_prompt: int = 1,
               max_prompt: int = 100,
               enable_noise_filter: bool = False,
               filter_std_threshold: float = 0.8,
               filter_median_threshold: float = 0.5,
               filter_gradient_threshold: float = 0.5,
               filter_min_neighbors: int = 5,
               filter_bilateral_d: int = 7,
               filter_bilateral_sigma_color: float = 1.0,
               filter_bilateral_sigma_space: float = 10.0) -> torch.Tensor:
    '''
    Load depth map and optionally apply noise filtering.

    Args:
        depth_path: Path to depth file
        tar_size: Target size (h, w)
        num_samples: Number of samples for sparse depth prompt
        min_prompt: Minimum depth value for valid prompt
        max_prompt: Maximum depth value for valid prompt
        enable_noise_filter: Whether to apply noise filtering (default: True)
        filter_*: Noise filter parameters (see filter_depth_noise_numpy for details)

    Returns:
        depth: Full depth map (1, 1, H, W)
        sample_depth: Sampled sparse depth (1, 1, H, W)
        depth_mask: Valid depth mask (1, 1, H, W)
    '''
    depth = _read_depth_array(depth_path=depth_path)
    depth = cv2.resize(depth, tar_size[::-1], interpolation=cv2.INTER_NEAREST)

    if enable_noise_filter:
        print("\n=== Applying strict depth noise filtering ===")
        initial_mask = ((depth > min_prompt) & (depth < max_prompt)).astype(np.float32)
        depth, filtered_mask = filter_depth_noise_numpy(
            depth=depth,
            depth_mask=initial_mask,
            std_threshold=filter_std_threshold,
            median_threshold=filter_median_threshold,
            gradient_threshold=filter_gradient_threshold,
            min_neighbors=filter_min_neighbors,
            bilateral_d=filter_bilateral_d,
            bilateral_sigma_color=filter_bilateral_sigma_color,
            bilateral_sigma_space=filter_bilateral_sigma_space
        )
        print("=== Depth noise filtering completed ===\n")
        depth_mask = filtered_mask
    else:
        print("[Info] Skipping depth noise filtering")
        depth_mask = ((depth > min_prompt) & (depth < max_prompt)).astype(np.float32)

    # ---------- Valid Depth Prompt ----------
    valid_depth = depth * depth_mask 
    if (valid_depth > 0.1).sum() > num_samples:
        height, width = depth.shape
        sample_depth = valid_depth.reshape(-1)
        nonzero_index = np.array(list(np.nonzero(sample_depth>0.1))).squeeze()
        index = np.random.permutation(nonzero_index)[:num_samples]
        sample_mask = np.ones_like(sample_depth)
        sample_mask[index] = 0.
        sample_depth[sample_mask.astype(bool)] = 0.
        sample_depth = sample_depth.reshape(height, width)
    else:
        sample_depth = valid_depth

    depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
    sample_depth = torch.from_numpy(sample_depth).unsqueeze(0).unsqueeze(0).float()
    depth_mask = torch.from_numpy(depth_mask).unsqueeze(0).unsqueeze(0)
   
    return depth, sample_depth, depth_mask


def depth_to_disparity(depth: torch.Tensor) -> torch.Tensor:
    disp = depth.clone()
    valid = disp > 0
    disp[valid] = 1.0 / disp[valid]
    return disp


def plot_depth(image: torch.Tensor, depth: torch.Tensor, output_path: str) -> None:
    depth_min, depth_max = depth.min().item(), depth.max().item()
    vis_img = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    if vis_img.shape[-2:] != depth.shape[-2:]:
        vis_img = cv2.resize(vis_img, (depth.shape[-1], depth.shape[-2]), interpolation=cv2.INTER_AREA)

    vis_depth = colorize_depth_maps(depth[0, 0].detach().cpu().numpy(), min_depth=depth_min, max_depth=depth_max, cmap='Spectral')
    vis_img = vis_depth
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.imwrite(output_path, vis_img)


def depth2pcd(
    sampled_coord,  
    sampled_depth,  
    rgb_image,     
    ixt,           
    depth_min=None,
    depth_max=None,
    ext=None,
    clip_box=None,
    ret_mask=False
):
    device = sampled_coord.device if torch.is_tensor(sampled_coord) else torch.device("cpu")

    if rgb_image.ndim == 3 and rgb_image.shape[0] == 3:  
        rgb_image = rgb_image.unsqueeze(0) 
    elif rgb_image.ndim == 3 and rgb_image.shape[2] == 3: 
        rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0) 
    else:
        raise ValueError("rgb_image shape not supported")

    rgb_image = rgb_image.float().to(device)
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image / 255.0

    grid = sampled_coord.unsqueeze(0).unsqueeze(0) 
    sampled_rgb = torch.nn.functional.grid_sample(
        rgb_image, grid.flip(-1), mode='bilinear', align_corners=False  # grid --> (x,y) / (w,h)
    ) 
    sampled_rgb = sampled_rgb.squeeze(0).squeeze(1).permute(1, 0) 

    p_x = ((sampled_coord[:, 1] + 1) * (rgb_image.shape[3] / 2.0)) - 0.5  # w --> x
    p_y = ((sampled_coord[:, 0] + 1) * (rgb_image.shape[2] / 2.0)) - 0.5  # h --> y
    ones = torch.ones_like(p_x)

    cam_coords = torch.stack([p_x, p_y, ones], dim=1) * sampled_depth.unsqueeze(1)
    cam_coords = cam_coords @ torch.inverse(torch.from_numpy(ixt).float().to(device)).T

    if ext is not None:
        cam_coords = torch.cat([cam_coords, torch.ones((cam_coords.shape[0], 1), device=device)], dim=1)
        cam_coords = cam_coords @ torch.inverse(torch.from_numpy(ext).float().to(device)).T

    mask = torch.ones(cam_coords.shape[0], dtype=torch.bool, device=device)
    if depth_min is not None:
        mask &= sampled_depth >= depth_min
    if depth_max is not None:
        mask &= sampled_depth <= depth_max

    if clip_box is not None:
        assert len(clip_box) == 6
        if clip_box[0] is not None: mask &= cam_coords[:, 0] <= clip_box[0]
        if clip_box[1] is not None: mask &= cam_coords[:, 1] <= clip_box[1]
        if clip_box[2] is not None: mask &= cam_coords[:, 2] <= clip_box[2]
        if clip_box[3] is not None: mask &= cam_coords[:, 0] >= clip_box[3]
        if clip_box[4] is not None: mask &= cam_coords[:, 1] >= clip_box[4]
        if clip_box[5] is not None: mask &= cam_coords[:, 2] >= clip_box[5]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cam_coords[mask, :3].cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(sampled_rgb[mask].cpu().numpy())

    if ret_mask:
        return pcd, mask
    return pcd


def save_sampled_point_clouds(
    sampled_coord,
    sampled_depth,
    rgb_image,
    fx,
    fy,
    cx,
    cy,
    output_path,
    ixt=None,
    filter_flying_points=True,
    nb_neighbors=30,
    std_ratio=2.0,
    **kwargs
):
    if ixt is None:
        try:
            ixt = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        except Exception as e:
            raise ValueError("Either provide ixt or valid fx, fy, cx, cy to build it.") from e
    pcd = depth2pcd(sampled_coord, sampled_depth, rgb_image, ixt, **kwargs)
    if filter_flying_points:
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd = pcd.select_by_index(ind)
    success = o3d.io.write_point_cloud(output_path, pcd)
    if success:
        Log.info(f"Save point cloud to {output_path}")
    else:
        Log.error(f"Failed to save point cloud to {output_path}")


def filter_depth_noise(depth: torch.Tensor,
                       depth_mask: torch.Tensor,
                       std_threshold: float = 2.5,
                       median_threshold: float = 0.3,
                       gradient_threshold: float = 0.5,
                       min_neighbors: int = 5,
                       bilateral_d: int = 5,
                       bilateral_sigma_color: float = 0.5,
                       bilateral_sigma_space: float = 5.0) -> tuple:

    device = depth.device
    depth_np = depth.squeeze().cpu().numpy()
    mask_np = depth_mask.squeeze().cpu().numpy().astype(bool)

    valid_depths = depth_np[mask_np]

    if len(valid_depths) == 0:
        print("[Warning] No valid depth values found!")
        return depth, depth_mask

    mean_depth = np.mean(valid_depths)
    std_depth = np.std(valid_depths)
    lower_bound = mean_depth - std_threshold * std_depth
    upper_bound = mean_depth + std_threshold * std_depth

    outlier_mask = (depth_np < lower_bound) | (depth_np > upper_bound)
    mask_np[outlier_mask] = False
    depth_np[outlier_mask] = 0

    print(f"[Filter] Step 1 - Statistical outlier: removed {np.sum(outlier_mask)} points")
    print(f"[Filter] Valid depth range: [{lower_bound:.2f}m, {upper_bound:.2f}m]")

    if np.sum(mask_np) > 0:
        depth_for_median = depth_np.copy()
        depth_for_median[~mask_np] = 0

        median_filtered = ndimage.median_filter(depth_for_median, size=5)

        valid_median = median_filtered[mask_np]
        valid_original = depth_np[mask_np]

        relative_diff = np.abs(valid_original - valid_median) / (valid_median + 1e-6)

        local_consistency_mask = mask_np.copy()
        local_consistency_mask[mask_np] = relative_diff < median_threshold

        removed_inconsistent = np.sum(mask_np) - np.sum(local_consistency_mask)
        mask_np = local_consistency_mask
        depth_np[~mask_np] = 0

        print(f"[Filter] Step 2 - Local consistency: removed {removed_inconsistent} points")

    if np.sum(mask_np) > 0:
        depth_for_grad = depth_np.copy()
        depth_for_grad[~mask_np] = 0

        dx = ndimage.sobel(depth_for_grad, axis=1)
        dy = ndimage.sobel(depth_for_grad, axis=0)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)

        normalized_gradient = gradient_magnitude / (depth_np + 1e-6)

        high_gradient_mask = (normalized_gradient > gradient_threshold) & mask_np
        mask_np[high_gradient_mask] = False
        depth_np[high_gradient_mask] = 0

        print(f"[Filter] Step 3 - Gradient filter: removed {np.sum(high_gradient_mask)} points")

    if np.sum(mask_np) > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_count = ndimage.convolve(mask_np.astype(np.uint8), kernel, mode='constant')

        isolated_mask = (neighbor_count < min_neighbors) & mask_np
        mask_np[isolated_mask] = False
        depth_np[isolated_mask] = 0

        print(f"[Filter] Step 4 - Isolated points: removed {np.sum(isolated_mask)} points")

    if np.sum(mask_np) > 0:
        depth_for_bilateral = depth_np.copy().astype(np.float32)
        if np.sum(mask_np) > 100: 
            bilateral_filtered = cv2.bilateralFilter(
                depth_for_bilateral,
                d=bilateral_d,
                sigmaColor=bilateral_sigma_color,
                sigmaSpace=bilateral_sigma_space
            )

            depth_np[mask_np] = bilateral_filtered[mask_np]

            print(f"[Filter] Step 5 - Bilateral filtering applied")

    filtered_depth = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float().to(device)
    filtered_mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(device)

    remaining_points = np.sum(mask_np)
    original_points = np.sum(depth_mask.squeeze().cpu().numpy() > 0)
    removed_ratio = (original_points - remaining_points) / max(original_points, 1) * 100

    print(f"[Filter] Summary: {original_points} -> {remaining_points} points ({removed_ratio:.1f}% removed)")

    return filtered_depth, filtered_mask
