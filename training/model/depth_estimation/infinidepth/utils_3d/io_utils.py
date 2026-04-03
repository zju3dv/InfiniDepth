from typing import Tuple
import torch
import imageio
import cv2
import numpy as np
import os
import h5py
import open3d as o3d
import torch.nn.functional as F
from training.utils.logger import Log

from .vis_utils import colorize_depth_maps

def load_image(image_path: str,
               dataset_name: str = "ETH3D",
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
    image = imageio.imread(image_path)[:, :, :3]
    if dataset_name == "nyu":
        image = image[45:-45, 60:-60]
    h, w = image.shape[:2]
    org_img = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image = cv2.resize(image, tar_size[::-1], interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return org_img, image, (h, w)

def load_depth(depth_path: str,
               dataset_name: str,
               tar_size: Tuple[int, int] = (504, 672),
               num_samples: int = 1500) -> torch.Tensor:
    '''
    depth is in mm and stored in 16-bit PNG
    '''
    if depth_path.endswith(".png"):
        depth = imageio.imread(depth_path)
        depth = (depth / 1000.).astype(np.float32)
    elif depth_path.endswith(".npz"):
        depth = np.load(depth_path)["data"]
    elif depth_path.endswith(".hdf5"):
        depth = h5py.File(depth_path)["dataset"]
        depth = np.asarray(depth)
    elif depth_path.endswith(".npy"):
        if dataset_name == "waymo":
            depth_dict = np.load(depth_path, allow_pickle=True).item()
            depth = np.zeros_like(depth_dict['mask']).astype(np.float32)
            depth[depth_dict['mask']] = depth_dict['value']
        else:
            depth = np.load(depth_path)
    else:
        raise ValueError(f"Invalid depth_path: {depth_path}")

    if dataset_name == "nyu":
        depth = depth[45:-45, 60:-60]

    depth = cv2.resize(depth, tar_size[::-1], interpolation=cv2.INTER_NEAREST)
    if (depth > 0.1).sum() > num_samples:
        height, width = depth.shape
        sample_depth = depth.reshape(-1)
        nonzero_index = np.array(list(np.nonzero(sample_depth>0.1))).squeeze()
        index = np.random.permutation(nonzero_index)[:num_samples]
        sample_mask = np.ones_like(sample_depth)
        sample_mask[index] = 0.
        sample_depth[sample_mask.astype(bool)] = 0.
        sample_depth = sample_depth.reshape(height, width)
    else:
        sample_depth = depth

    depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
    sample_depth = torch.from_numpy(sample_depth).unsqueeze(0).unsqueeze(0).float()
    
    return depth, sample_depth

def plot_depth(image: torch.Tensor, depth: torch.Tensor, prompt_depth: torch.Tensor, output_path: str) -> None:
    depth_min, depth_max = depth.min().item(), depth.max().item()
    vis_img = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    if vis_img.shape[-2:] != depth.shape[-2:]:
        vis_img = cv2.resize(vis_img, (depth.shape[-1], depth.shape[-2]), interpolation=cv2.INTER_AREA)
    prompt_depth = prompt_depth[0, 0].detach().cpu().numpy()
    prompt_depth = cv2.resize(prompt_depth, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    prompt_depth = cv2.resize(prompt_depth, (vis_img.shape[1], vis_img.shape[0]), interpolation=cv2.INTER_AREA)
    vis_prompt_depth = colorize_depth_maps(prompt_depth, min_depth=depth_min, max_depth=depth_max, cmap='Spectral')
    vis_depth = colorize_depth_maps(depth[0, 0].detach().cpu().numpy(), min_depth=depth_min, max_depth=depth_max, cmap='Spectral')
    vis_img = np.concatenate([vis_img, vis_depth], axis=1)
    # vis_img = np.concatenate([vis_img, vis_prompt_depth], axis=1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.imwrite(output_path, vis_img)


def depth2pcd(
    sampled_coord,
    sampled_depth,  # (N,)
    rgb_image,
    ixt,
    depth_min=None,
    depth_max=None,
    ext=None,
    clip_box=None,
    ret_mask=False
):
    device = sampled_coord.device if torch.is_tensor(sampled_coord) else torch.device("cpu")

    if rgb_image.ndim == 3 and rgb_image.shape[0] == 3:  # (3,H,W)
        rgb_image = rgb_image.unsqueeze(0)  # (1,3,H,W)
    elif rgb_image.ndim == 3 and rgb_image.shape[2] == 3:  # (H,W,3)
        rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    else:
        raise ValueError("rgb_image shape not supported")

    rgb_image = rgb_image.float().to(device)
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image / 255.0

    grid = sampled_coord.unsqueeze(0).unsqueeze(0)  # (1,1,N,2)
    sampled_rgb = torch.nn.functional.grid_sample(
        rgb_image, grid.flip(-1), mode='bilinear', align_corners=False  # grid --> (x,y) / (w,h)
    )  # (1,3,1,N)
    sampled_rgb = sampled_rgb.squeeze(0).squeeze(1).permute(1, 0)  # (N,3)

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
    **kwargs
):
    if ixt is None:
        try:
            ixt = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        except Exception as e:
            raise ValueError("Either provide ixt or valid fx, fy, cx, cy to build it.") from e
    pcd = depth2pcd(sampled_coord, sampled_depth, rgb_image, ixt, **kwargs)
    success = o3d.io.write_point_cloud(output_path, pcd)
    if success:
        Log.info(f"Save point cloud to {output_path}")
    else:
        Log.error(f"Failed to save point cloud to {output_path}")
