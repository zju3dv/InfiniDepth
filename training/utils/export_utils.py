import os
from enum import Enum, auto

import torch
import trimesh
import numpy as np
from .parallel_utils import save_image, save_image_async
from .vis_utils import visualize_depth

__all__ = [
    'export_pointcloud', # export_pointcloud
    'export_image', # export_image
    'ImageType', # ImageType
]


class ImageType(Enum):
    '''
    Enum class for different image types:
    - RGB: RGB image normalized to [0, 1]
    - RGB_NORM: RGB image normalized by COCO mean and std
    - DEPTH: depth image
    '''
    RGB = auto() 
    RGB_NORM = auto()
    MASK = auto()
    DEPTH = auto()

def export_pointcloud(sample,
                      color=None,
                      name='0000.ply',
                      tag='none',
                      save_dir='./debug',
                      async_save=False):
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().cpu().numpy()
        if color is not None and isinstance(color, torch.Tensor):
            color = color.detach().cpu().numpy()
            if color.dtype == np.uint8:
                color = (color / 255.).astype(np.float32)
    
    sample = sample.reshape(-1, 3)
    if color is not None:
        color = color.reshape(-1, 3)
    pointcloud = trimesh.PointCloud(sample, color=color)
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    pointcloud.export(os.path.join(save_dir, name.replace(name[-4:], '_' + tag + name[-4:])))

def export_image(sample, 
                 name='0000.jpg', 
                 tag='none', 
                 save_dir='./debug',
                 image_type=ImageType.RGB_NORM,
                 async_save=True):
    '''
    Visualize and save an image sample to disk.

    Args:
        sample: Input image tensor or numpy array
        name: Output filename
        tag: Tag to append to filename
        save_dir: Directory to save output
        image_type: Type of image (RGB, RGB_NORM, or DEPTH)
        async_save: Whether to save asynchronously

    The function handles:
    1. Converting tensor to numpy if needed
    2. Transposing to HWC format if needed 
    3. Processing based on image type (normalization etc)
    4. Saving to disk
    '''
    # to numpy
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().cpu().numpy()
    
    # to HWC
    if len(sample.shape) == 3:
        if sample.shape[0] == 3 or sample.shape[0] == 1:
            sample = sample.transpose(1, 2, 0)

    if image_type == ImageType.DEPTH:
        save_img = visualize_depth(sample)
    elif image_type == ImageType.RGB:
        save_img = (sample * 255.).astype(np.uint8)
    elif image_type == ImageType.MASK:
        if sample.max() <= (1.0 + 1e-6):
            save_img = (sample * 255.).astype(np.uint8)
        else:
            save_img = sample.astype(np.uint8)
        save_img = save_img[..., 0]
    elif image_type == ImageType.RGB_NORM:
        mean = np.array([0.485, 0.456, 0.406])[None, None, :]
        std = np.array([0.229, 0.224, 0.225])[None, None, :]
        save_img = (sample * std + mean) * 255.
        save_img = save_img.astype(np.uint8)
    else:
        raise ValueError(f"Invalid image type: {image_type}")
    
    save_path = os.path.join(save_dir, name.replace(name[-4:], '_' + tag + name[-4:]))

    if async_save: save_image_async(save_img, save_path)
    else: save_image(save_img, save_path)