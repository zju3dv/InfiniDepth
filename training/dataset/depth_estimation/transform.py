from configparser import Interpolation
import os
from pickle import NONE
import random

from click import argument
import cv2
import numpy as np
from sympy import use
import torch
import math
import torch.nn.functional as F
from omegaconf.listconfig import ListConfig
from training.utils.depth_utils import GenerateSpotMask, bilateralFilter, interp_depth_rgb, norm_coord
from training.utils.logger import Log
from training.utils.sampling_np import importance_sampling, pixel_sampling

EPS = 1e-4


def _safe_quantile(values, q):
    values = np.asarray(values, dtype=np.float32)
    return np.quantile(values, q)


class Rotate:
    """Prepare sample for usage as network input."""

    def __init__(self):
        pass

    def __str__(self):
        return "Rotate"

    def __repr__(self):
        return "Rotate"

    def __call__(self, sample):
        assert "direction" in sample, "No direction in sample"
        if sample["direction"] == "Up":
            pass
        else:
            if sample["direction"] == "Left":
                rotate_option = cv2.ROTATE_90_CLOCKWISE
            elif sample["direction"] == "Right":
                rotate_option = cv2.ROTATE_90_COUNTERCLOCKWISE
            elif sample["direction"] == "Down":
                rotate_option = cv2.ROTATE_180
            else:
                raise Exception(f'No such direction (={sample["direction"]}) rotation')
            sample["image"] = cv2.rotate(sample["image"], rotate_option)
            if "mask" in sample:
                sample["mask"] = cv2.rotate(sample["mask"], rotate_option)
            if "depth" in sample:
                sample["depth"] = cv2.rotate(sample["depth"], rotate_option)
            if "lowres_depth" in sample:
                sample["lowres_depth"] = cv2.rotate(sample["lowres_depth"], rotate_option)
            if "confidence" in sample:
                sample["confidence"] = cv2.rotate(sample["confidence"], rotate_option)
        return sample


class UnPrepareForNet:
    def __init__(self):
        pass

    def __str__(self):
        return "UnPrepareForNet"

    def __repr__(self):
        return "UnPrepareForNet"

    def __call__(self, sample):
        image = np.transpose(sample["image"], (1, 2, 0))
        sample["image"] = image
        if "mask" in sample:
            sample["mask"] = sample["mask"][0]
        if "confidence" in sample:
            sample["confidence"] = sample["confidence"][0]
        if "depth" in sample:
            sample["depth"] = sample["depth"][0]
        if "lowres_depth" in sample:
            sample["lowres_depth"] = sample["lowres_depth"][0]
        if "disparity" in sample:
            sample["disparity"] = sample["disparity"][0]
        if "disparity_mask" in sample:
            sample["disparity_mask"] = sample["disparity_mask"][0]
        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"][0]
        if "semantic" in sample:
            sample["semantic"] = sample["semantic"][0]
        if "mesh_depth" in sample:
            sample["mesh_depth"] = sample["mesh_depth"][0]
        if "prompt_depth" in sample:
            sample["prompt_depth"] = sample["prompt_depth"][0]
        if "prompt_mask" in sample:
            sample["prompt_mask"] = sample["prompt_mask"][0]
        if "prompt_disparity" in sample:
            sample["prompt_disparity"] = sample["prompt_disparity"][0]
        return sample


class PrepareForNet:
    """Prepare sample for usage as network input."""

    def __init__(self):
        pass

    def __str__(self):
        return "PrepareForNet"

    def __repr__(self):
        return "PrepareForNet"

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)
        # gray_image = np.transpose(sample["gray_image"], (2, 0, 1))
        # sample["gray_image"] = np.ascontiguousarray(gray_image).astype(np.float32)

        if "right_image" in sample:
            right_image = np.transpose(sample["right_image"], (2, 0, 1))
            sample["right_image"] = np.ascontiguousarray(right_image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.uint8)
            sample["mask"] = np.ascontiguousarray(sample["mask"])[None]

        if "disparity_mask" in sample:
            disparity_mask = sample["disparity_mask"].astype(np.uint8)
            sample["disparity_mask"] = np.ascontiguousarray(disparity_mask)[None]

        if "confidence" in sample:
            sample["confidence"] = sample["confidence"].astype(np.uint8)
            sample["confidence"] = np.ascontiguousarray(sample["confidence"])[None]

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)[None]

        if "disparity" in sample:
            disparity = sample["disparity"].astype(np.float32)
            sample["disparity"] = np.ascontiguousarray(disparity)[None]

        if "mesh_depth" in sample:
            mesh_depth = sample["mesh_depth"].astype(np.float32)
            sample["mesh_depth"] = np.ascontiguousarray(mesh_depth)[None]

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"].astype(np.float32)
            sample["semseg_mask"] = np.ascontiguousarray(sample["semseg_mask"])[None]

        if "semantic" in sample:
            sample["semantic"] = sample["semantic"].astype(np.uint8)
            sample["semantic"] = np.ascontiguousarray(sample["semantic"])[None]

        if "lowres_depth" in sample:
            lowres_depth = sample["lowres_depth"].astype(np.float32)
            sample["lowres_depth"] = np.ascontiguousarray(lowres_depth)[None]

        if "prompt_depth" in sample:
            prompt_depth = sample["prompt_depth"].astype(np.float32)
            sample["prompt_depth"] = np.ascontiguousarray(prompt_depth)[None]

        if "prompt_mask" in sample:
            prompt_mask = sample["prompt_mask"].astype(np.uint8)
            sample["prompt_mask"] = np.ascontiguousarray(prompt_mask)[None]

        if "prompt_disparity" in sample:
            prompt_disparity = sample["prompt_disparity"].astype(np.float32)
            sample["prompt_disparity"] = np.ascontiguousarray(prompt_disparity)[None]


        if "lr_image" in sample:
            lr_image = np.transpose(sample["lr_image"], (2, 0, 1))
            sample["lr_image"] = np.ascontiguousarray(lr_image).astype(np.float32)


        if "cell" in sample:
            cell = sample["cell"].astype(np.float32)
            sample["cell"] = np.ascontiguousarray(cell)


        if "sampled_coord_depth" in sample:
            hr_coord_depth = sample["sampled_coord_depth"].astype(np.float32)
            sample["sampled_coord_depth"] = np.ascontiguousarray(hr_coord_depth)
        
        if "sampled_depth" in sample:
            hr_depth_flat = sample["sampled_depth"].astype(np.float32)
            sample["sampled_depth"] = np.ascontiguousarray(hr_depth_flat)

        if "sampled_coord_disparity" in sample:
            hr_coord_disparity = sample["sampled_coord_disparity"].astype(np.float32)
            sample["sampled_coord_disparity"] = np.ascontiguousarray(hr_coord_disparity)
        
        if "sampled_disparity" in sample:
            hr_disparity_flat = sample["sampled_disparity"].astype(np.float32)
            sample["sampled_disparity"] = np.ascontiguousarray(hr_disparity_flat)


        if "hr_sampled_coord_depth" in sample:
            hr_coord_depth = sample["hr_sampled_coord_depth"].astype(np.float32)
            sample["hr_sampled_coord_depth"] = np.ascontiguousarray(hr_coord_depth)
        
        if "hr_sampled_depth" in sample:
            hr_depth_flat = sample["hr_sampled_depth"].astype(np.float32)
            sample["hr_sampled_depth"] = np.ascontiguousarray(hr_depth_flat)

        if "hr_sampled_coord_disparity" in sample:
            hr_coord_disparity = sample["hr_sampled_coord_disparity"].astype(np.float32)
            sample["hr_sampled_coord_disparity"] = np.ascontiguousarray(hr_coord_disparity)
        
        if "hr_sampled_disparity" in sample:
            hr_disparity_flat = sample["hr_sampled_disparity"].astype(np.float32)
            sample["hr_sampled_disparity"] = np.ascontiguousarray(hr_disparity_flat)

        if "global_coord_depth" in sample:
            global_coord = sample["global_coord_depth"].astype(np.float32)
            sample["global_coord_depth"] = np.ascontiguousarray(global_coord)

        if "global_value_depth" in sample:
            global_value = sample["global_value_depth"].astype(np.float32)
            sample["global_value_depth"] = np.ascontiguousarray(global_value)

        if "global_mask_depth" in sample:
            global_mask = sample["global_mask_depth"].astype(np.uint8)
            sample["global_mask_depth"] = np.ascontiguousarray(global_mask)

        if "global_cell_depth" in sample:
            global_cell = sample["global_cell_depth"].astype(np.float32)
            sample["global_cell_depth"] = np.ascontiguousarray(global_cell)

        if "local_coord_depth" in sample:
            local_coord = sample["local_coord_depth"].astype(np.float32)
            sample["local_coord_depth"] = np.ascontiguousarray(local_coord)

        if "local_value_depth" in sample:
            local_value = sample["local_value_depth"].astype(np.float32)
            sample["local_value_depth"] = np.ascontiguousarray(local_value)

        if "local_mask_depth" in sample:
            local_mask = sample["local_mask_depth"].astype(np.uint8)
            sample["local_mask_depth"] = np.ascontiguousarray(local_mask)

        if "local_cell_depth" in sample:
            local_cell = sample["local_cell_depth"].astype(np.float32)
            sample["local_cell_depth"] = np.ascontiguousarray(local_cell)

        
        if "global_coord_disparity" in sample:
            global_coord = sample["global_coord_disparity"].astype(np.float32)
            sample["global_coord_disparity"] = np.ascontiguousarray(global_coord)

        if "global_value_disparity" in sample:
            global_value = sample["global_value_disparity"].astype(np.float32)
            sample["global_value_disparity"] = np.ascontiguousarray(global_value)

        if "global_mask_disparity" in sample:
            global_mask = sample["global_mask_disparity"].astype(np.uint8)
            sample["global_mask_disparity"] = np.ascontiguousarray(global_mask)

        if "global_cell_disparity" in sample:
            global_cell = sample["global_cell_disparity"].astype(np.float32)
            sample["global_cell_disparity"] = np.ascontiguousarray(global_cell)

        if "local_coord_disparity" in sample:
            local_coord = sample["local_coord_disparity"].astype(np.float32)
            sample["local_coord_disparity"] = np.ascontiguousarray(local_coord)

        if "local_value_disparity" in sample:
            local_value = sample["local_value_disparity"].astype(np.float32)
            sample["local_value_disparity"] = np.ascontiguousarray(local_value)

        if "local_mask_disparity" in sample:
            local_mask = sample["local_mask_disparity"].astype(np.uint8)
            sample["local_mask_disparity"] = np.ascontiguousarray(local_mask)

        if "local_cell_disparity" in sample:
            local_cell = sample["local_cell_disparity"].astype(np.float32)
            sample["local_cell_disparity"] = np.ascontiguousarray(local_cell)

        return sample


def cv2_resize(image, size, interpolation=cv2.INTER_LINEAR):
    return cv2.resize(image, size, interpolation=interpolation)[None]


def random_simu(depth, tar_size, use_bi=True, mean=-0.01, std=0.005):
    img_w, img_h = tar_size
    depth = cv2.resize(depth, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    if np.random.random() < 0.2:
        return depth.astype(np.float32)
    if img_w == img_h:
        dist_coef = 0.0
    else:
        dist_coef = 2e-5
    orig_depth = depth
    rand = np.random.randn(img_h * img_w).reshape(img_h, img_w)
    rand = rand * std + mean
    np.min(depth)
    depth = depth + rand * depth
    if np.random.random() < float(os.environ.get("spot_mask_prob", 0.5)):
        return depth.astype(np.float32)
    spot_mask = GenerateSpotMask(img_h, img_w, stride=3, dist_coef=dist_coef)
    sparse_depth = np.zeros_like(depth)
    spot_mask = spot_mask == 1.0
    sparse_depth[spot_mask] = depth[spot_mask]

    sparse_depth = interp_depth_rgb(sparse_depth, orig_depth, speed=5, k=4)
    if use_bi:
        bi_sparse_depth = bilateralFilter(orig_depth, 5, 0.01, 50.0, sparse_depth)
        return bi_sparse_depth.astype(np.float32)
    else:
        return sparse_depth.astype(np.float32)
    import matplotlib.pyplot as plt

    plt.subplot(231)
    plt.imshow(orig_depth)
    plt.axis("off")
    plt.subplot(232)
    plt.imshow(depth)
    plt.axis("off")
    plt.subplot(233)
    plt.imshow(sparse_depth)
    plt.axis("off")
    plt.subplot(234)
    plt.imshow(spot_mask)
    plt.axis("off")
    plt.subplot(235)
    plt.imshow(bi_sparse_depth)
    plt.axis("off")
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig("test.jpg", dpi=300)
    return bi_sparse_depth


def random_simu_wtof(depth, tof, tar_size, use_bi=True, mean=0.01, std=0.005):
    img_w, img_h = tar_size
    depth = cv2.resize(depth, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    tof = cv2.resize(tof, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    # if np.random.random() < 0.2:
    #     return depth.astype(np.float32)
    if img_w == img_h:
        dist_coef = 0.0
    else:
        dist_coef = 2e-5
    orig_depth = depth

    rand = np.random.randn(img_h * img_w).reshape(img_h, img_w)
    rand = rand * std + mean
    depth = depth + rand * depth
    # if np.random.random() < float(os.environ.get('spot_mask_prob', 0.5)):
    #     return depth.astype(np.float32)
    spot_mask = GenerateSpotMask(img_h, img_w, stride=3, dist_coef=dist_coef)
    sparse_depth = np.zeros_like(depth)
    spot_mask = spot_mask == 1.0
    tof_mask = spot_mask & (tof != 0)
    depth_mask = spot_mask & (depth != 0) & (not tof_mask)
    Log.info("num tof_mask: ", tof_mask.sum())
    if tof_mask.sum() != 0:
        sparse_depth[tof_mask] = tof[tof_mask]
    else:
        depth_mask = spot_mask & (depth != 0)
        sparse_depth[depth_mask] = depth[depth_mask]
    if depth_mask.sum() != 0:
        sparse_depth[depth_mask] = depth[depth_mask]
    sparse_depth = interp_depth_rgb(sparse_depth, orig_depth, speed=5, k=4)
    if use_bi:
        bi_sparse_depth = bilateralFilter(orig_depth, 5, 0.01, 50.0, sparse_depth)
        return bi_sparse_depth.astype(np.float32)
    else:
        return sparse_depth.astype(np.float32)
    import matplotlib.pyplot as plt

    plt.subplot(231)
    plt.imshow(orig_depth)
    plt.axis("off")
    plt.subplot(232)
    plt.imshow(depth)
    plt.axis("off")
    plt.subplot(233)
    plt.imshow(sparse_depth)
    plt.axis("off")
    plt.subplot(234)
    plt.imshow(spot_mask)
    plt.axis("off")
    plt.subplot(235)
    plt.imshow(bi_sparse_depth)
    plt.axis("off")
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig("test.jpg", dpi=300)
    return bi_sparse_depth


class CompLowRes:
    def __init__(self, height=None, width=None, ranges=None, range_prob=None, interpolation="cv2"):
        self._height = height
        self._width = width
        self._ranges = ranges
        self._range_prob = range_prob
        self._interpolation = interpolation
        self.idx = 0

    def __str__(self):
        return f"SimLowRes: height: {self._height}, width: {self._width}, ranges: {self._ranges}, range_prob: {self._range_prob}"

    def __call__(self, sample):
        try:
            rgb = np.transpose(sample["image"], (1, 2, 0))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            lowres_depth = sample["lowres_depth"][0]
            if (lowres_depth != 0).sum() <= 10:
                sample["lowres_depth"] = cv2_resize(
                    sample["depth"][0], (lowres_depth.shape[1], lowres_depth.shape[0]), interpolation=cv2.INTER_LINEAR
                )
                return sample

            if gray.shape[0] != lowres_depth.shape[0] or gray.shape[1] != lowres_depth.shape[1]:
                gray = cv2.resize(gray, (lowres_depth.shape[1], lowres_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
            sparse_depth = interp_depth_rgb(lowres_depth, gray, speed=5, k=4)
            if rgb.shape[0] < sparse_depth.shape[0] or rgb.shape[1] < sparse_depth.shape[1]:
                sparse_depth = cv2.resize(sparse_depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            # bi_sparse_depth = bilateralFilter(gray, 5, 0.01, 50., sparse_depth)
            sample["lowres_depth"] = sparse_depth[None]
            sample["sparse_depth"] = lowres_depth[None]
            # import ipdb; ipdb.set_trace()
            # import matplotlib.pyplot as plt
            # plt.subplot(131)
            # plt.imshow(sample['image'].transpose(1, 2, 0))
            # plt.axis('off')
            # plt.subplot(132)
            # plt.imshow(sparse_depth)
            # plt.axis('off')
            # plt.subplot(133)
            # plt.imshow(lowres_depth)
            # plt.axis('off')
            # plt.tight_layout()
            # plt.savefig(f'test_comp_{self.idx}.jpg', dpi=300)
            # self.idx += 1
        except Exception:
            pass
        return sample


class SimLowRes:
    def __init__(self, height=None, width=None, ranges=None, range_prob=None, interpolation="cv2", use_bi=True):
        self._height = height
        self._width = width
        self._ranges = ranges
        self._range_prob = range_prob
        self._interpolation = interpolation
        self._use_bi = use_bi

    def __str__(self):
        return f"SimLowRes: height: {self._height}, width: {self._width}, ranges: {self._ranges}, range_prob: {self._range_prob}"

    def __call__(self, sample):
        assert "lowres_depth" not in sample, "lowres depth in sample"
        if self._height is not None:
            tar_size = (self._width, self._height)
        else:
            assert sample["depth"].shape[1] == sample["depth"].shape[2]
            choice = np.random.choice(np.arange(len(self._range_prob)), p=self._range_prob)
            if choice == 0:
                tar_size = (self._ranges[0], self._ranges[0])
            elif choice == 2:
                tar_size = (self._ranges[1], self._ranges[1])
            elif choice == 1:
                tar_size = int(np.random.random() * (self._ranges[1] - self._ranges[0]) + self._ranges[0])
                tar_size = (tar_size, tar_size)
        if self._interpolation == "cv2":
            sample["lowres_depth"] = cv2_resize(sample["depth"][0], tar_size, interpolation=cv2.INTER_LINEAR)
        elif self._interpolation == "random_simu":
            sample["lowres_depth"] = random_simu(sample["depth"][0], tar_size, use_bi=self._use_bi)[None]
        elif self._interpolation == "copy":
            sample["lowres_depth"] = np.copy(sample["depth"][0][None])
        return sample


class SimLowResHammer:
    def __init__(self, height=None, width=None, ranges=None, range_prob=None, interpolation="cv2", use_bi=True):
        self._height = height
        self._width = width
        self._ranges = ranges
        self._range_prob = range_prob
        self._interpolation = interpolation
        self._use_bi = use_bi

    def __str__(self):
        return f"SimLowResHammer: height: {self._height}, width: {self._width}, ranges: {self._ranges}, range_prob: {self._range_prob}"

    def __call__(self, sample):
        assert "lowres_depth" in sample, "lowres depth not in sample"
        if self._height is not None:
            tar_size = (self._width, self._height)
        else:
            assert sample["depth"].shape[1] == sample["depth"].shape[2]
            choice = np.random.choice(np.arange(len(self._range_prob)), p=self._range_prob)
            if choice == 0:
                tar_size = (self._ranges[0], self._ranges[0])
            elif choice == 2:
                tar_size = (self._ranges[1], self._ranges[1])
            elif choice == 1:
                tar_size = int(np.random.random() * (self._ranges[1] - self._ranges[0]) + self._ranges[0])
                tar_size = (tar_size, tar_size)
        if self._interpolation == "cv2":
            sample["lowres_depth"] = cv2_resize(sample["lowres_depth"][0], tar_size, interpolation=cv2.INTER_NEAREST)
        elif self._interpolation == "random_simu":
            sample["lowres_depth"] = random_simu_wtof(
                sample["depth"][0], sample["lowres_depth"][0], tar_size, use_bi=self._use_bi
            )[None]
        return sample


class SimuCarLidar:
    def __init__(self, tar_lines=64, reserve_ratio=0.5, min_pitch=-0.07):
        self._tar_lines = tar_lines
        self._reserve_ratio = reserve_ratio
        self._min_pitch = min_pitch
        self.ixt = np.array([[640.0, 0.0, 315], [0.0, 640.0, 210.0], [0.0, 0.0, 1.0]])

    def __str__(self):
        return f"SimuCarLidar: tar_lines: {self._tar_lines}, reserve_ratio: {self._reserve_ratio}, min_pitch: {self._min_pitch}"

    def __call__(self, sample):
        assert "lowres_depth" not in sample, "lowres depth in sample"
        depth_map = sample["depth"][0]
        v, u = np.nonzero(depth_map)
        z = depth_map[v, u]

        points = np.linalg.inv(self.ixt) @ (np.vstack([u, v, np.ones_like(u)]) * z)
        points = points.transpose([1, 0])

        scan_y = points[:, 1]
        distance = np.linalg.norm(points, 2, axis=1)
        pitch = np.arcsin(scan_y / distance)
        num_points = np.shape(pitch)[0]
        pitch = np.reshape(pitch, (num_points, 1))
        max_pitch = np.max(pitch)
        min_pitch = np.min(pitch)

        input_lines = depth_map.shape[0]
        num_lines = int(input_lines * (self._min_pitch - max_pitch) / (min_pitch - max_pitch)) - 1
        keep_ratio = self._tar_lines / num_lines

        angle_interval = (max_pitch - self._min_pitch) / num_lines
        angle_label = np.round((pitch - min_pitch) / angle_interval)

        sampling_mask = angle_label % int(1.0 / keep_ratio) == 0
        sampling_mask = sampling_mask & (pitch >= self._min_pitch)

        final_mask = sampling_mask.reshape(depth_map.shape)

        random_mask = np.random.random(final_mask.shape) < self._reserve_ratio

        grid_mask = np.zeros_like(final_mask, dtype=bool)
        grid_mask[:, ::2] = True

        max_val = 80
        trunc_val = 50
        distance_drop = ((depth_map - trunc_val) / (max_val - trunc_val)) ** 3 + np.random.random(depth_map.shape)
        final_mask = final_mask & grid_mask & random_mask & (distance_drop < 1.0)

        output_depth = np.zeros_like(depth_map)
        output_depth[final_mask] = depth_map[final_mask]

        sample["lowres_depth"] = output_depth[None]
        return sample


class SimuCarLidarApollo:
    def __init__(self, tar_lines=64, reserve_ratio=0.5, min_pitch=-0.07):
        self._tar_lines = tar_lines
        self._reserve_ratio = reserve_ratio
        self._min_pitch = min_pitch
        self.ixt = np.array([[2015.0, 0.0, 810.0], [0.0, 2015.0, 540.0], [0.0, 0.0, 1.0]])
        self.ixt[:2] *= 0.62222222

    def __str__(self):
        return f"SimuCarLidar: tar_lines: {self._tar_lines}, reserve_ratio: {self._reserve_ratio}, min_pitch: {self._min_pitch}"

    def __call__(self, sample):
        assert "lowres_depth" not in sample, "lowres depth in sample"
        depth_map = sample["depth"][0]
        v, u = np.nonzero(depth_map)
        z = depth_map[v, u]

        points = np.linalg.inv(self.ixt) @ (np.vstack([u, v, np.ones_like(u)]) * z)
        points = points.transpose([1, 0])

        scan_y = points[:, 1]
        distance = np.linalg.norm(points, 2, axis=1)
        pitch = np.arcsin(scan_y / distance)
        num_points = np.shape(pitch)[0]
        pitch = np.reshape(pitch, (num_points, 1))
        max_pitch = np.max(pitch)
        min_pitch = np.min(pitch)

        input_lines = depth_map.shape[0]
        num_lines = int(input_lines * (self._min_pitch - max_pitch) / (min_pitch - max_pitch)) - 1
        keep_ratio = self._tar_lines / num_lines

        angle_interval = (max_pitch - self._min_pitch) / num_lines
        angle_label = np.round((pitch - min_pitch) / angle_interval)

        sampling_mask = angle_label % int(1.0 / keep_ratio) == 0
        sampling_mask = sampling_mask & (pitch >= self._min_pitch)

        final_mask = sampling_mask.reshape(depth_map.shape)

        random_mask = np.random.random(final_mask.shape) < self._reserve_ratio

        grid_mask = np.zeros_like(final_mask, dtype=bool)
        grid_mask[:, ::3] = True

        max_val = 80
        trunc_val = 50
        distance_drop = ((depth_map - trunc_val) / (max_val - trunc_val)) ** 3 + np.random.random(depth_map.shape)
        final_mask = final_mask & grid_mask & random_mask & (distance_drop < 1.0)

        output_depth = np.zeros_like(depth_map)
        output_depth[final_mask] = depth_map[final_mask]

        # import imageio
        # imageio.imwrite('test.png', (final_mask * 255).astype(np.uint8))
        # imageio.imwrite('test.jpg', (sample['image'].transpose(1, 2, 0) * 255.).astype(np.uint8))
        # import ipdb; ipdb.set_trace()
        sample["lowres_depth"] = output_depth[None]
        return sample


class Crop:
    """Crop sample for batch-wise training. Image is of shape CxHxW"""

    def __init__(self, size, down_scale=1, center=False, down_scales=None, down_scale_prob=None):
        self.size = size
        self._center = center
        # downs_scale = 1, 2, 4, 7.5, 8
        assert down_scale in [-1, 1, 2, 3.75, 4, 7.5, 8, 14, 15, 28], "Wrong down_scale"
        self._down_scale = down_scale
        local_dict = {-1: 1, 1: 1, 2: 2, 3.75: 15, 4: 4, 8: 8, 7.5: 15, 14: 1, 28: 1, 15: 15}
        self._local_dict = local_dict
        self._ensure_round = local_dict[down_scale]
        self._down_scales = down_scales
        self._down_scale_prob = down_scale_prob
        # Log.info(f"Crop size: {size}, down_scale: {down_scale}, center: {center}")

    def __str__(self):
        return f"RandomCrop: size: {self.size}, down_scale: {self._down_scale}, center: {self._center}"

    def get_bbox(self, sample, h, w, ensure_round, target_height=None, target_width=None):
        if isinstance(self.size, int):
            if self.size == -1:
                return 0, 0, h, w
            assert h >= self.size and w >= self.size, "Wrong size"
            if h - self.size == 0:
                h_start = 0
            else:
                h_start = np.random.randint(0, h - self.size)
            if w - self.size == 0:
                w_start = 0
            else:
                w_start = np.random.randint(0, w - self.size)
            if "lowres_depth" in sample:
                h_start = h_start // ensure_round * ensure_round
                w_start = w_start // ensure_round * ensure_round
            if "prompt_depth" in sample:
                h_start = h_start // ensure_round * ensure_round
                w_start = w_start // ensure_round * ensure_round
            h_end = h_start + self.size
            w_end = w_start + self.size
        elif isinstance(self.size, ListConfig):
            if self.size == []:
                return 0, 0, h, w
            if target_height is not None and target_width is not None:
                new_size_h, new_size_w = target_height, target_width
            else:
                new_size_h, new_size_w = self.size
            assert h >= new_size_h and w >= new_size_w, "Wrong size"
            h_start = (h - new_size_h) // 2
            w_start = (w - new_size_w) // 2
            if "lowres_depth" in sample:
                h_start = h_start // ensure_round * ensure_round
                w_start = w_start // ensure_round * ensure_round
            if "prompt_depth" in sample:
                h_start = h_start // ensure_round * ensure_round
                w_start = w_start // ensure_round * ensure_round
            h_end = h_start + new_size_h
            w_end = w_start + new_size_w
        return h_start, w_start, h_end, w_end

    def __call__(self, sample):
        if self._down_scales is not None:
            down_scale = np.random.choice(self._down_scales, p=self._down_scale_prob)
            ensure_round = self._local_dict[down_scale]
        else:
            down_scale = self._down_scale
            ensure_round = self._ensure_round
        if "height" in sample and "width" in sample:
            h_start, w_start, h_end, w_end = self.get_bbox(
                sample,
                sample["image"].shape[-2],
                sample["image"].shape[-1],
                ensure_round,
                target_height=sample["height"],
                target_width=sample["width"],
            )
        elif isinstance(self.size, int) or (isinstance(self.size, ListConfig) and (len(self.size) == 2)):
            h_start, w_start, h_end, w_end = self.get_bbox(
                sample, sample["image"].shape[-2], sample["image"].shape[-1], ensure_round
            )
        else:
            h_start, w_start, h_end, w_end = self.size

        sample["image"] = sample["image"][:, h_start:h_end, w_start:w_end]

        if "right_image" in sample:
            sample["right_image"] = sample["right_image"][:, h_start:h_end, w_start:w_end]

        if "depth" in sample:
            sample["depth"] = sample["depth"][:, h_start:h_end, w_start:w_end]

        if "mask" in sample:
            sample["mask"] = sample["mask"][:, h_start:h_end, w_start:w_end]

        if "disparity" in sample:
            sample["disparity"] = sample["disparity"][:, h_start:h_end, w_start:w_end]

        if "disparity_mask" in sample:
            sample["disparity_mask"] = sample["disparity_mask"][:, h_start:h_end, w_start:w_end]

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"][:, h_start:h_end, w_start:w_end]

        if "prompt_depth" in sample:
            ds = down_scale
            sample["prompt_depth"] = sample["prompt_depth"][
                :, int(h_start / ds) : int(h_end / ds), int(w_start / ds) : int(w_end / ds)
            ]

            sample["prompt_disparity"] = sample["prompt_disparity"][
                :, int(h_start / ds) : int(h_end / ds), int(w_start / ds) : int(w_end / ds)
            ]

            sample["prompt_mask"] = sample["prompt_mask"][
                :, int(h_start / ds) : int(h_end / ds), int(w_start / ds) : int(w_end / ds)
            ]

        if "confidence" in sample:
            ds = down_scale
            sample["confidence"] = sample["confidence"][
                :, int(h_start / ds) : int(h_end / ds), int(w_start / ds) : int(w_end / ds)
            ]
        return sample


class TopCrop:
    """TopCrop sample for batch-wise training. Image is of shape CxHxW"""

    def __init__(self, size):
        self.size = size

    def __str__(self):
        return f"TopCrop: size: {self.size}"

    def __call__(self, sample):
        sample["image"] = sample["image"][:, self.size :, :]

        if "depth" in sample:
            sample["depth"] = sample["depth"][self.size :, :]

        if "mask" in sample:
            sample["mask"] = sample["mask"][self.size :, :]

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"][:, self.size :, :]

        if "lowres_depth" in sample:
            sample["lowres_depth"] = sample["lowres_depth"][self.size :, :]

        if "confidence" in sample:
            sample["confidence"] = sample["confidence"][:, self.size :, :]

        return sample


class Resize:
    """Resize sample to given size (width, height)."""

    def __init__(
        self,
        width=None,
        height=None,
        resize_ratio=None,
        resize_range=None,  # [min_length, max_length]
        resize_target=False,
        resize_prompt=True, 
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
        resize_lowres=False,
        normalize_disp=False,
        use_dynamic_depth=False,
        dynamic_depth_min_size=256,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.width = width
        self.height = height
        self.__width = width
        self.__height = height
        self.__resize_ratio = resize_ratio
        self.__resize_range = resize_range
        self.__normalize_disp = normalize_disp
        self.__use_dynamic_depth = use_dynamic_depth
        self.__dynamic_depth_min_size = dynamic_depth_min_size

        # Validate that only one resizing method is specified
        resize_methods_specified = sum(
            [(width is not None and height is not None), resize_ratio is not None, resize_range is not None]
        )
        if resize_methods_specified != 1:
            raise ValueError(
                "Exactly one resize method must be specified: "
                "either (width and height), resize_ratio, or resize_range"
            )

        self.__resize_target = resize_target
        self.__resize_prompt = resize_prompt
        self.__resize_lowres = resize_lowres
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def __str__(self):
        return f"Resize(w={self.__width}, h={self.__height}, method={self.__resize_method})"

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)
        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        if self.__resize_ratio is not None:
            __height = int(height * self.__resize_ratio)
            __width = int(width * self.__resize_ratio)
        elif self.__resize_range is not None:
            if width < height:
                __width = np.random.randint(self.__resize_range[0], self.__resize_range[1])
                __height = int(__width * height / width)
            else:
                __height = np.random.randint(self.__resize_range[0], self.__resize_range[1])
                __width = int(__height * width / height)
        else:
            __height = self.__height if self.__height is not None else height
            __width = self.__width if self.__width is not None else width
        scale_height = __height / height
        scale_width = __width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def get_size_from_sample(self, width, height, target_width, target_height):
        scale_height = target_height / height
        scale_width = target_width / width
        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=target_height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=target_width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=target_height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=target_width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def get_depth_size(self, org_width, org_height):
        random_w = random.randint(self.__dynamic_depth_min_size, org_width)
        random_h = random.randint(self.__dynamic_depth_min_size, org_height)
        return random_w, random_h

    def __call__(self, sample):
        if "height" in sample and "width" in sample:
            fov_ratio = sample.get("fov_ratio", 1.0)
            width, height = sample["width"], sample["height"]
            width, height = int(width * fov_ratio), int(height * fov_ratio)
            width, height = self.get_size_from_sample(
                sample["image"].shape[1], sample["image"].shape[0], width, height
            )
        else:
            width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])
        if width == sample["image"].shape[1] and height == sample["image"].shape[0]:
            return sample
        Log.debug("Resize: {} -> {}".format(sample["image"].shape, (height, width)))
        # resize sample
        orig_height, orig_width = sample["image"].shape[:2]
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )
        # sample["gray_image"] = cv2.resize(
        #     sample["gray_image"],
        #     (width, height),
        #     interpolation=self.__image_interpolation_method,
        # )[:,:,None]

        if "right_image" in sample:
            sample["right_image"] = cv2.resize(
                sample["right_image"],
                (width, height),
                interpolation=self.__image_interpolation_method,
            )

        # if 'lowres_depth' in sample:
        #     width_low, height_low = int(width / 7.5), int(height / 7.5)
        #     sample['lowres_depth'] = cv2.resize(
        #         sample['lowres_depth'],
        #         (width_low, height_low),
        #         interpolation=cv2.INTER_LINEAR
        #     )
        if self.__resize_prompt:
            Log.debug("Resize Prompt")
            if "prompt_depth" in sample:
                sample["prompt_depth"] = cv2.resize(
                    sample["prompt_depth"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "prompt_mask" in sample:
                sample["prompt_mask"] = cv2.resize(
                    sample["prompt_mask"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "prompt_disparity" in sample:
                sample["prompt_disparity"] = cv2.resize(
                    sample["prompt_disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

        if self.__resize_target:
            Log.debug("Resize target")
            if self.__use_dynamic_depth:
                width, height = self.get_depth_size(orig_width, orig_height)
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                if self.__normalize_disp:
                    sample["disparity"] = sample["disparity"] * width / orig_width

            if "depth" in sample:
                sample["depth"] = cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST)

            if "normal" in sample:
                sample["normal"] = cv2.resize(sample["normal"], (width, height), interpolation=cv2.INTER_NEAREST)

            if "mesh_depth" in sample:
                sample["mesh_depth"] = cv2.resize(
                    sample["mesh_depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            if "semseg_mask" in sample:
                sample["semseg_mask"] = F.interpolate(
                    torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...], (height, width), mode="nearest"
                ).numpy()[0, 0]

            if "semantic" in sample:
                sample["semantic"] = cv2.resize(
                    sample["semantic"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                # sample["mask"] = sample["mask"].astype(bool)
            if "disparity_mask" in sample:
                sample["disparity_mask"] = cv2.resize(
                    sample["disparity_mask"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            # if "prompt_depth" in sample:
            #     sample["prompt_depth"] = cv2.resize(
            #         sample["prompt_depth"],
            #         (width, height),
            #         interpolation=cv2.INTER_NEAREST,
            #     )

            # if "prompt_mask" in sample:
            #     sample["prompt_mask"] = cv2.resize(
            #         sample["prompt_mask"],
            #         (width, height),
            #         interpolation=cv2.INTER_NEAREST,
            #     )

            # if "prompt_disparity" in sample:
            #     sample["prompt_disparity"] = cv2.resize(
            #         sample["prompt_disparity"],
            #         (width, height),
            #         interpolation=cv2.INTER_NEAREST,
            #     )

            if "focal" in sample:
                # assert width / height == orig_width / orig_height
                sample["focal"] = sample["focal"] * width / orig_width
        if self.__resize_lowres:
            sample["lowres_depth"] = cv2.resize(
                sample["lowres_depth"], (width, height), interpolation=cv2.INTER_NEAREST
            )
        # print(sample['image'].shape, sample['depth'].shape)
        return sample


class Crop_Resize:
    """Resize sample to given size (width, height), with optional center crop before resize."""

    def __init__(
        self,
        width=None,
        height=None,
        resize_ratio=None,
        resize_range=None,  # [min_length, max_length]
        resize_target=False,
        resize_prompt=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
        resize_lowres=False,
        normalize_disp=False,
        crop_before_resize=False,
        depth_target_width=None,
        depth_target_height=None,
    ):
        self.width = width
        self.height = height
        self.__width = width
        self.__height = height
        self.__resize_ratio = resize_ratio
        self.__resize_range = resize_range
        self.__normalize_disp = normalize_disp
        self.__crop_before_resize = crop_before_resize

        self.__depth_target_width = depth_target_width
        self.__depth_target_height = depth_target_height

        resize_methods_specified = sum(
            [(width is not None and height is not None), resize_ratio is not None, resize_range is not None]
        )
        if resize_methods_specified != 1:
            raise ValueError(
                "Exactly one resize method must be specified: "
                "either (width and height), resize_ratio, or resize_range"
            )

        self.__resize_target = resize_target
        self.__resize_prompt = resize_prompt
        self.__resize_lowres = resize_lowres
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def __str__(self):
        return f"Resize(w={self.__width}, h={self.__height}, method={self.__resize_method}, crop={self.__crop_before_resize})"

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)
        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)
        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)
        return y

    def get_size(self, width, height):
        __height = self.__height if self.__height is not None else height
        __width = self.__width if self.__width is not None else width

        scale_height = __height / height
        scale_width = __width / width

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)
    
    def get_size_from_sample(self, width, height, target_width, target_height):
        scale_height = target_height / height
        scale_width = target_width / width

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=target_height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=target_width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=target_height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=target_width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    @staticmethod
    def center_crop_to_aspect(img, target_w, target_h):
        h, w = img.shape[:2]
        orig_aspect = w / h
        target_aspect = target_w / target_h

        if abs(orig_aspect - target_aspect) < 1e-6:
            return img

        if orig_aspect > target_aspect:
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            return img[:, start_x:start_x + new_w]
        else:
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            return img[start_y:start_y + new_h, :]

    def __call__(self, sample):
        if "height" in sample and "width" in sample:
            fov_ratio = sample.get("fov_ratio", 1.0)
            width, height = sample["width"], sample["height"]
            width, height = int(width * fov_ratio), int(height * fov_ratio)
            width, height = self.get_size_from_sample(
                sample["image"].shape[1], sample["image"].shape[0], width, height
            )
        else:
            width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])

        if width == sample["image"].shape[1] and height == sample["image"].shape[0]:
            return sample

        Log.debug("Resize: {} -> {}".format(sample["image"].shape, (height, width)))
        orig_height, orig_width = sample["image"].shape[:2]

        # === Step1: Center Crop before resize ===
        sample["image"] = self.center_crop_to_aspect(sample["image"], width, height)
        if "right_image" in sample:
            sample["right_image"] = self.center_crop_to_aspect(sample["right_image"], width, height)

        for key in ["prompt_depth", "prompt_mask", "prompt_disparity", "disparity", "depth", "mesh_depth", "semantic", "mask", "disparity_mask", "semseg_mask", "highfreq_mask"]:
            if key in sample:
                sample[key] = self.center_crop_to_aspect(sample[key], width, height)

        # === Step2: Resize ===
        sample["image"] = cv2.resize(
            sample["image"], (width, height), interpolation=self.__image_interpolation_method
        )

        if "right_image" in sample:
            sample["right_image"] = cv2.resize(
                sample["right_image"], (width, height), interpolation=self.__image_interpolation_method
            )

        if self.__resize_prompt:
            Log.debug("Resize Prompt")
            if "prompt_depth" in sample:
                sample["prompt_depth"] = cv2.resize(
                    sample["prompt_depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )
            if "prompt_mask" in sample:
                sample["prompt_mask"] = cv2.resize(
                    sample["prompt_mask"], (width, height), interpolation=cv2.INTER_NEAREST
                )
            if "prompt_disparity" in sample:
                sample["prompt_disparity"] = cv2.resize(
                    sample["prompt_disparity"], (width, height), interpolation=cv2.INTER_NEAREST
                )

        if self.__resize_target:
            Log.debug("Resize target")
            if self.__depth_target_width is not None and self.__depth_target_height is not None:
                resize_target_width, resize_target_height = self.__depth_target_width, self.__depth_target_height
            else:
                resize_target_width, resize_target_height = width, height
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"], (resize_target_width, resize_target_height), interpolation=cv2.INTER_NEAREST
                )
                if self.__normalize_disp:
                    sample["disparity"] = sample["disparity"] * resize_target_width / orig_width

            if "depth" in sample:
                sample["depth"] = cv2.resize(sample["depth"], (resize_target_width, resize_target_height), interpolation=cv2.INTER_NEAREST)

            if "mesh_depth" in sample:
                sample["mesh_depth"] = cv2.resize(
                    sample["mesh_depth"], (resize_target_width, resize_target_height), interpolation=cv2.INTER_NEAREST
                )

            if "semseg_mask" in sample:
                sample["semseg_mask"] = F.interpolate(
                    torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...],
                    (resize_target_height, resize_target_width),
                    mode="nearest",
                ).numpy()[0, 0]

            if "semantic" in sample:
                sample["semantic"] = cv2.resize(
                    sample["semantic"], (resize_target_width, resize_target_height), interpolation=cv2.INTER_NEAREST
                )

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32), (resize_target_width, resize_target_height), interpolation=cv2.INTER_NEAREST
                )

            if "disparity_mask" in sample:
                sample["disparity_mask"] = cv2.resize(
                    sample["disparity_mask"], (resize_target_width, resize_target_height), interpolation=cv2.INTER_NEAREST
                )

            if "highfreq_mask" in sample:
                sample["highfreq_mask"] = cv2.resize(
                    sample["highfreq_mask"], (resize_target_width, resize_target_height), interpolation=cv2.INTER_NEAREST
                )

            if "focal" in sample:
                sample["focal"] = sample["focal"] * resize_target_width / orig_width

        if self.__resize_lowres and "lowres_depth" in sample:
            sample["lowres_depth"] = cv2.resize(
                sample["lowres_depth"], (width, height), interpolation=cv2.INTER_NEAREST
            )

        return sample

    

class Resize_Crop(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width=None,
        height=None,
        resize_ratio=None,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
        resize_lowres=False,
        crop_type='random'
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.width = width
        self.height = height
        self.__width = width
        self.__height = height
        self.__resize_ratio = resize_ratio
        self.crop_type = crop_type
        assert (
            (width is not None and height is not None) or resize_ratio is not None
        )
        assert (
            (width is None and height is None) or resize_ratio is None
        )

        self.__resize_target = resize_target
        self.__resize_lowres = resize_lowres
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method
        # Log.info(f"Resize width: {width}, height: {height}, resize_target: {resize_target}, keep_aspect_ratio: {keep_aspect_ratio}, ensure_multiple_of: {ensure_multiple_of}, resize_method: {resize_method}")

    def __str__(self):
        return "Resize: width: {}, height: {}, resize_target: {}, keep_aspect_ratio: {}, ensure_multiple_of: {}, resize_method: {}".format(self.__width, self.__height, self.__resize_target, self.__keep_aspect_ratio, self.__multiple_of, self.__resize_method)

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)
        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        if self.__resize_ratio is not None:
            __height = int(height * self.__resize_ratio)
            __width = int(width * self.__resize_ratio)
        else:
            __height = self.__height if self.__height is not None else height
            __width = self.__width if self.__width is not None else width
        scale_height = __height / height
        scale_width = __width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(
                f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )
        if width == sample['image'].shape[1] and height == sample['image'].shape[0]:
            return sample
        Log.debug(
            'Resize: {} -> {}'.format(sample["image"].shape, (height, width)))
        # resize sample
        orig_height, orig_width = sample['image'].shape[:2]
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )
        # sample["gray_image"] = cv2.resize(
        #     sample["gray_image"],
        #     (width, height),
        #     interpolation=self.__image_interpolation_method,
        # )[:,:,None]

        # crop sample
        crop_size = min(width, height)
        if self.crop_type == 'random':
            # random crop
            top = np.random.randint(0, height - crop_size + 1)
            left = np.random.randint(0, width - crop_size + 1)
        else:
            # center crop
            top = (height - crop_size) // 2
            left = (width - crop_size) // 2
        sample["image"] = sample["image"][top:top+crop_size, left:left+crop_size]
        # sample["gray_image"] = sample["gray_image"][top:top+crop_size, left:left+crop_size]

        # if 'lowres_depth' in sample:
        #     width_low, height_low = int(width / 7.5), int(height / 7.5)
        #     sample['lowres_depth'] = cv2.resize(
        #         sample['lowres_depth'],
        #         (width_low, height_low),
        #         interpolation=cv2.INTER_LINEAR
        #     )

        if self.__resize_target:
            Log.debug('Resize target')
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                # crop sample
                sample["disparity"] = sample["disparity"][top:top+crop_size, left:left+crop_size]
            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width,
                                      height), interpolation=cv2.INTER_NEAREST
                )
                # crop sample
                sample["depth"] = sample["depth"][top:top+crop_size, left:left+crop_size]

            if "mesh_depth" in sample:
                sample["mesh_depth"] = cv2.resize(
                    sample["mesh_depth"], (width,
                                           height), interpolation=cv2.INTER_NEAREST
                )

            if "semseg_mask" in sample:
                sample["semseg_mask"] = F.interpolate(torch.from_numpy(sample["semseg_mask"]).float()[
                                                      None, None, ...], (height, width), mode='nearest').numpy()[0, 0]

            if "semantic" in sample:
                sample["semantic"] = cv2.resize(
                    sample["semantic"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                sample["mask"] = sample["mask"][top:top+crop_size, left:left+crop_size]
                # sample["mask"] = sample["mask"].astype(bool)

            if "disparity_mask" in sample:
                sample["disparity_mask"] = cv2.resize(
                    sample["disparity_mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                sample["disparity_mask"] = sample["disparity_mask"][top:top+crop_size, left:left+crop_size]

            if "prompt_depth" in sample:
                sample["prompt_depth"] = cv2.resize(
                    sample["prompt_depth"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                sample["prompt_depth"] = sample["prompt_depth"][top:top+crop_size, left:left+crop_size]

            if "prompt_mask" in sample:
                sample["prompt_mask"] = cv2.resize(
                    sample["prompt_mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                sample["prompt_mask"] = sample["prompt_mask"][top:top+crop_size, left:left+crop_size]

            if "prompt_disparity" in sample:
                sample["prompt_disparity"] = cv2.resize(
                    sample["prompt_disparity"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                sample["prompt_disparity"] = sample["prompt_disparity"][top:top+crop_size, left:left+crop_size]

            if 'focal' in sample:
                # assert width / height == orig_width / orig_height
                sample['focal'] = sample['focal'] * width / orig_width
        if self.__resize_lowres:
            sample['lowres_depth'] = cv2.resize(
            sample['lowres_depth'],
            (width, height),
            interpolation=cv2.INTER_NEAREST
            )
        # print(sample['image'].shape, sample['depth'].shape)
        return sample


class Resize_512_crop(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width=None,
        height=None,
        image_interpolation_method=cv2.INTER_AREA,
        crop_type='center'
    ):
        self.target_width = width
        self.target_height = height
        self.crop_type = crop_type

    def __call__(self, sample):
        # resize sample
        orig_h, orig_w = sample['image'].shape[:2]

        if orig_h > orig_w:
            sample['image'] = cv2.rotate(sample['image'], cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        orig_h, orig_w = sample['image'].shape[:2]

        resize_h = self.target_height
        resize_w = int(resize_h/orig_h*orig_w)

        sample["image"] = cv2.resize(
            sample["image"],
            (resize_w, resize_h),
            interpolation=cv2.INTER_AREA,
        )

        if resize_w >= self.target_width:
            left = (resize_w - self.target_width) // 2
            sample["image"] = sample["image"][:, left:left+self.target_width]
        
        else:
            sample["image"] = cv2.resize(
            sample["image"],
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA,
            )
        
        if "depth" in sample:
            sample["depth"] = cv2.resize(
                    sample["depth"], (resize_w, resize_h), interpolation=cv2.INTER_NEAREST
                )
            # crop sample
            if resize_w >= self.target_width:
                left = (resize_w - self.target_width) // 2
                sample["depth"] = sample["depth"][:, left:left+self.target_width]
        
            else:
                sample["depth"] = cv2.resize(
                sample["depth"],
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_NEAREST,
                )
        
        if "mask" in sample:
            sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32), (resize_w, resize_h), interpolation=cv2.INTER_NEAREST
                )
            # crop sample
            if resize_w >= self.target_width:
                left = (resize_w - self.target_width) // 2
                sample["mask"] = sample["mask"][:, left:left+self.target_width]
        
            else:
                sample["mask"] = cv2.resize(
                sample["mask"],
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_NEAREST,
                )

        return sample


class Center_Crop(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width=None,
        height=None,
        image_interpolation_method=cv2.INTER_AREA,
        crop_type='center'
    ):
        self.target_width = width
        self.target_height = height
        self.crop_type = crop_type

    def __call__(self, sample):
        # resize sample
        orig_h, orig_w = sample['image'].shape[:2]

        left = (orig_w - self.target_width) // 2
        top = (orig_h - self.target_height) // 2

        sample["image"] = sample["image"][top:top+self.target_height, left:left+self.target_width]
        
        if "depth" in sample:
            sample["depth"] = sample["depth"][top:top+self.target_height, left:left+self.target_width]
        
        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)[top:top+self.target_height, left:left+self.target_width]

        return sample
    
    
class Normalize:
    """Normalize sample to given mean and std."""

    def __init__(
        self,
        mean,
        std,
    ):
        """Init.

        Args:
            mean (ListConfig): Normalize mean
            std (ListConfig): Normalize std
        """
        self.mean = np.array(mean).astype(np.float32) / 256.0
        self.std = np.array(std).astype(np.float32) / 256.0

    def __str__(self):
        return f"Normalize: mean: {self.mean}, std: {self.std}"

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.mean[:, None, None]) / self.std[:, None, None]
        return sample


class RandomDrop:
    def __init__(self, drop_min, drop_max, drop_down_size):
        self.drop_min = drop_min
        self.drop_max = drop_max
        self.drop_down_size = drop_down_size

        self.idx = 0

    def __call__(self, sample):
        # from 1 to self.drop_down_size
        drop_img_down = np.random.random() * (self.drop_down_size - 1) + 1
        down_ratio = 1 / drop_img_down
        lowres_depth = sample["lowres_depth"]
        h, w = lowres_depth.shape[-2:]
        h_down = int(h * down_ratio)
        w_down = int(w * down_ratio)
        lowres_depth_mask = cv2.resize(sample["lowres_depth"][0], (w_down, h_down), interpolation=cv2.INTER_AREA)

        # random drop depth according to depth distance drop_min->prob 0, drop_max->prob 1
        drop_prob = (lowres_depth_mask - self.drop_min) / (self.drop_max - self.drop_min)
        # drop_prob = np.clip(drop_prob, 0, 1)
        drop_mask = np.asarray(np.random.random(lowres_depth_mask.shape) < drop_prob, dtype=np.float32)
        drop_mask = cv2.resize(drop_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        lowres_depth[0][drop_mask > 0.5] = 0.0
        # import ipdb; ipdb.set_trace()
        # import matplotlib.pyplot as plt
        # plt.subplot(121)
        # plt.imshow(lowres_depth[0])
        # plt.axis('off')
        # plt.subplot(122)
        # plt.imshow(sample['image'].transpose(1, 2, 0))
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig(f'test_{self.idx}.jpg', dpi=300)
        # self.idx += 1
        return sample


class RandomKeep:
    def __init__(self, min_keep, max_keep, ratio=0.1):
        self.ratio = ratio
        self.min_keep = min_keep
        self.max_keep = max_keep

    def __call__(self, sample):
        lowres_depth = sample["lowres_depth"][0]
        ratio = self.ratio
        current_num = np.sum(lowres_depth != 0)
        if current_num == 0:
            return sample
        min_ratio = self.min_keep / current_num
        max_ratio = self.max_keep / current_num
        ratio = max(min_ratio, min(max_ratio, ratio))
        drop_mask = np.random.random(lowres_depth.shape) > ratio
        lowres_depth[drop_mask] = 0.0
        return sample


# class RandomSample(object):
#     def __init__(self, pts_num):
#         self.pts_num = pts_num

#     def __call__(self, sample):
#         lowres_depth = sample['lowres_depth'][0]
#         height, width = lowres_depth.shape
#         index = np.random.choice(height * width, self.pts_num, replace=False)
#         lowres_depth = lowres_depth.reshape(-1)
#         sample_mask = np.ones_like(lowres_depth)
#         sample_mask[index] = 0.
#         lowres_depth[sample_mask.astype(bool)] = 0.
#         lowres_depth = lowres_depth.reshape(height, width)
#         return sample


class RandomSample:
    def __init__(self, pts_num=None, pts_range=None, fps=False):  # [min_num, max_num]
        assert (pts_num is not None and pts_range is None) or (pts_num is None and pts_range is not None)
        self.pts_num = pts_num
        self.pts_range = pts_range
        self.fps = fps
        if self.fps:
            raise NotImplementedError("FPS is not implemented")

    def get_num_points(self, num_valid_points):

        if num_valid_points == 0:
            return 0

        if self.pts_range is not None:
            # Return valid points count if below minimum threshold
            if num_valid_points <= self.pts_range[0]:
                return num_valid_points

            # Random number within range if above maximum threshold
            if num_valid_points > self.pts_range[1]:
                return np.random.randint(self.pts_range[0], self.pts_range[1])

            # Random number between min threshold and available points
            return np.random.randint(self.pts_range[0], num_valid_points)
        else:
            # For fixed point count mode, return either all valid points
            # or the specified number, whichever is smaller
            return min(num_valid_points, self.pts_num)

    def __call__(self, sample):
        if self.pts_num == -1:
            return sample
        if self.pts_num is None and self.pts_range[0] == 0: 
            if np.random.random() < 0.1:
                sample["prompt_mask"] = np.zeros_like(sample["prompt_mask"])
                sample["prompt_depth"] = np.zeros_like(sample["prompt_depth"])
                sample["prompt_disparity"] = np.zeros_like(sample["prompt_disparity"])
                return sample
            
        num_valid_depth = sample["downsampled"]["lr_prompt_mask"].sum() if "downsampled" in sample else sample["prompt_mask"].sum()

        num_pts = self.get_num_points(num_valid_depth)

        if num_pts == num_valid_depth:
            return sample

        assert num_pts < num_valid_depth
        prompt_mask = sample["downsampled"]["lr_prompt_mask"] if "downsampled" in sample else sample["prompt_mask"]
        # Get indices of non-zero points in the prompt mask
        valid_indices = np.where(prompt_mask > 0)

        # Randomly select num_pts indices from the valid points
        random_indices = np.random.choice(valid_indices[0].size, num_pts, replace=False)

        # Create a new mask with only the selected points
        # new_mask = np.zeros_like(prompt_mask).reshape(-1)
        # new_mask[random_indices] = 1
        # new_mask = new_mask.reshape(prompt_mask.shape)
        new_mask = np.zeros_like(prompt_mask)
        new_mask[
            valid_indices[0][random_indices], valid_indices[1][random_indices], valid_indices[2][random_indices]
        ] = 1
        # Update the prompt mask with the new mask
        sample["prompt_mask"] = new_mask
        sample["prompt_disparity"][new_mask == 0] = 0.0
        sample["prompt_depth"][new_mask == 0] = 0.0

        return sample


class FixedSample:
    def __init__(self, pts_num=1000):
        assert pts_num > 0, "pts_num must be positive"
        self.pts_num = pts_num

    def get_num_points(self, num_valid_points):
        return min(num_valid_points, self.pts_num)

    def __call__(self, sample):
        if self.pts_num == -1:
            return sample

        num_valid_depth = (
            sample["downsampled"]["lr_prompt_mask"].sum()
            if "downsampled" in sample
            else sample["prompt_mask"].sum()
        )

        num_pts = self.get_num_points(num_valid_depth)

        if num_pts == num_valid_depth:
            return sample

        assert num_pts < num_valid_depth
        prompt_mask = (
            sample["downsampled"]["lr_prompt_mask"]
            if "downsampled" in sample
            else sample["prompt_mask"]
        )

        valid_indices = np.where(prompt_mask > 0)

        selected_indices = tuple(idx[:num_pts] for idx in valid_indices)

        new_mask = np.zeros_like(prompt_mask)
        new_mask[selected_indices] = 1

        sample["prompt_mask"] = new_mask
        sample["prompt_disparity"][new_mask == 0] = 0.0
        sample["prompt_depth"][new_mask == 0] = 0.0

        return sample


class RandomLineSample:
    def __init__(self, line_width_min=2, line_width_max=5, degree=45):
        self.line_width_min = line_width_min
        self.line_width_max = line_width_max
        self.degree = degree

    def get_line_mask(self, height, width):
        # Create an empty black image (all zeros)
        mask = np.zeros((height, width), dtype=np.uint8)
        # Define the center of the image
        center_x, center_y = width // 2, height // 2
        # Randomly select line width between the min and max values
        line_width = random.randint(self.line_width_min, self.line_width_max)
        # Randomly select a rotation angle between -15 and 15 degrees
        angle = random.uniform(-self.degree, self.degree)
        # Define the line's start and end points (horizontal line centered in the image)
        start_x = 0
        end_x = width
        start_y = center_y - line_width // 2
        end_y = center_y + line_width // 2
        # Rotate the line
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
        points = np.array([[start_x, start_y], [end_x, end_y]], dtype=np.float32)
        rotated_points = cv2.transform(np.array([points]), rotation_matrix)[0]
        # Draw the rotated line on the mask
        cv2.line(mask, tuple(rotated_points[0].astype(int)), tuple(rotated_points[1].astype(int)), 1, line_width)
        return mask

    def __call__(self, sample):
        prompt_mask = sample["prompt_mask"]
        _, h, w = prompt_mask.shape
        line_mask = self.get_line_mask(h, w)
        sample["prompt_mask"] = prompt_mask * line_mask[None]
        sample["prompt_depth"][sample["prompt_mask"] == 0] = 0.0
        sample["prompt_disparity"][sample["prompt_mask"] == 0] = 0.0
        return sample


class RandomDoubleLineSample:
    def __init__(self, line_width_min=2, line_width_max=5, degree=45):
        self.line_width_min = line_width_min
        self.line_width_max = line_width_max
        self.degree = degree

    def get_line_mask(self, height, width):
        # Create an empty black image (all zeros)
        mask = np.zeros((height, width), dtype=np.uint8)
        # Define the center of the image
        center_x, center_y = width // 2, height // 2
        # Randomly select line width between the min and max values
        line_width = random.randint(self.line_width_min, self.line_width_max)
        # Randomly select a rotation angle between -degree and degree
        angle1 = random.uniform(-self.degree, self.degree)

        # First line - extending across the entire image
        # Calculate the diagonal length to ensure line covers the entire image
        diagonal_length = int(np.sqrt(width**2 + height**2))
        # Define the line's start and end points (horizontal line centered in the image)
        start_x = center_x - diagonal_length
        end_x = center_x + diagonal_length
        start_y = center_y
        end_y = center_y
        # Rotate the line
        rotation_matrix1 = cv2.getRotationMatrix2D((center_x, center_y), angle1, 1)
        points1 = np.array([[start_x, start_y], [end_x, end_y]], dtype=np.float32)
        rotated_points1 = cv2.transform(np.array([points1]), rotation_matrix1)[0]
        # Draw the rotated line on the mask
        cv2.line(mask, tuple(rotated_points1[0].astype(int)), tuple(rotated_points1[1].astype(int)), 1, line_width)

        # Second line with angle between 60 to 90 degrees from the first line
        angle_diff = random.uniform(60, 90)
        # Randomly decide if the second line is clockwise or counter-clockwise from the first
        if random.random() > 0.5:
            angle2 = angle1 + angle_diff
        else:
            angle2 = angle1 - angle_diff

        # Create the second line - also extending across the entire image
        rotation_matrix2 = cv2.getRotationMatrix2D((center_x, center_y), angle2, 1)
        points2 = np.array([[start_x, start_y], [end_x, end_y]], dtype=np.float32)
        rotated_points2 = cv2.transform(np.array([points2]), rotation_matrix2)[0]
        # Draw the second rotated line on the mask
        cv2.line(mask, tuple(rotated_points2[0].astype(int)), tuple(rotated_points2[1].astype(int)), 1, line_width)

        return mask

    def __call__(self, sample):
        prompt_mask = sample["prompt_mask"]
        _, h, w = prompt_mask.shape
        line_mask = self.get_line_mask(h, w)
        sample["prompt_mask"] = prompt_mask * line_mask[None]
        sample["prompt_depth"][sample["prompt_mask"] == 0] = 0.0
        sample["prompt_disparity"][sample["prompt_mask"] == 0] = 0.0
        return sample


# class RandomDoubleLineSample:
#     def __init__(self, line_width_min=3, line_width_max=5, degree=15):
#         self.line_width_min = line_width_min
#         self.line_width_max = line_width_max
#         self.degree = degree

#     def __call__(self, sample):
#         pass


class RandomLine:

    def __call__(self, sample):
        lowres_depth = sample["lowres_depth"]
        if (lowres_depth > 0).sum() <= 100:
            lowres_depth = sample["depth"]
        if (lowres_depth > 0).sum() <= 100:
            return sample
        lowres_depth = lowres_depth[0].copy()

        H, W = lowres_depth.shape
        # randomly generate a image mask where it contains a random line pixel 1, and other pixels = 0
        # Create an empty mask
        mask = np.zeros((H, W), dtype=np.float32)

        # Get non-zero depth points
        nonzero_y, nonzero_x = np.where(lowres_depth > 0.0)
        if len(nonzero_y) == 0:
            # If no valid depth points, return original sample
            return sample

        # Randomly select two points to form a line
        idx1, idx2 = np.random.choice(len(nonzero_y), 2, replace=False)
        y1, x1 = nonzero_y[idx1], nonzero_x[idx1]
        y2, x2 = nonzero_y[idx2], nonzero_x[idx2]

        # Calculate line slope and intercept
        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Extend line to image boundaries
            x_start, x_end = 0, W - 1
            y_start = int(slope * x_start + intercept)
            y_end = int(slope * x_end + intercept)

            # Clip y-coordinates to image boundaries
            if y_start < 0 or y_start >= H:
                # If y is out of bounds, calculate where the line intersects the top or bottom
                if y_start < 0:
                    y_start = 0
                    x_start = int((y_start - intercept) / slope)
                else:
                    y_start = H - 1
                    x_start = int((y_start - intercept) / slope)

            if y_end < 0 or y_end >= H:
                if y_end < 0:
                    y_end = 0
                    x_end = int((y_end - intercept) / slope)
                else:
                    y_end = H - 1
                    x_end = int((y_end - intercept) / slope)
        else:
            # Vertical line
            x_start = x_end = x1
            y_start, y_end = 0, H - 1

        # Calculate points along the extended line
        num_points = max(abs(x_end - x_start), abs(y_end - y_start)) * 2  # Ensure enough points for a smooth line
        t = np.linspace(0, 1, num_points)
        line_x = (x_start * (1 - t) + x_end * t).astype(int)
        line_y = (y_start * (1 - t) + y_end * t).astype(int)

        # Clip coordinates to image boundaries (for safety)
        line_x = np.clip(line_x, 0, W - 1)
        line_y = np.clip(line_y, 0, H - 1)
        # Create mask using numpy indexing
        valid_depth_points = lowres_depth[line_y, line_x] > 0.1
        mask[line_y[valid_depth_points], line_x[valid_depth_points]] = 1.0
        # Apply the mask to the depth image
        filtered_depth = lowres_depth * mask
        if mask.sum() > 2000:
            pass
        sample["lowres_depth"] = filtered_depth[None]  # Add channel dimension

        return sample


# class RandomDropFar:
#     def __init__(self, start_range, act_prob):
#         self.start_range = start_range
#         self.act_prob = act_prob

#     def __call__(self, sample):
#         if np.random.random() > self.act_prob:
#             return sample
#         prompt_depth = sample["prompt_depth"]
#         prompt_mask = sample["prompt_mask"]
#         prompt_values = prompt_depth[prompt_mask == 1]
#         far_range = np.random.random() * (1 - self.start_range) + self.start_range
#         start_clip_depth = np.percentile(prompt_values, far_range * 100)
#         max_depth = np.max(prompt_values)
#         # Calculate drop probability based on depth value
#         # The deeper the point, the higher the probability to drop it
#         np.zeros_like(prompt_mask)

#         # Only consider points that are in the far range (between start_clip_depth and max_depth)
#         far_points = (prompt_depth > start_clip_depth) & (prompt_mask == 1)

#         if np.any(far_points):
#             # Normalize depths to [0, 1] range for probability calculation
#             normalized_depths = (prompt_depth[far_points] - start_clip_depth) / (max_depth - start_clip_depth + EPS)

#             # Generate random values for each point
#             random_values = np.random.random(normalized_depths.shape)

#             # Points with random value less than normalized depth will be dropped
#             # This ensures deeper points have higher probability of being dropped
#             points_to_drop = random_values < normalized_depths

#             # Create temporary mask for indexing
#             temp_mask = np.zeros_like(prompt_mask)
#             temp_mask[far_points] = points_to_drop

#             # Update the prompt mask by removing the dropped points
#             sample["prompt_mask"] = prompt_mask * (1 - temp_mask)
#         return sample


class RandomNoise:
    def __init__(self, std, act_prob):
        self.std = std
        self.act_prob = act_prob

    def __call__(self, sample):
        if np.random.random() > self.act_prob:
            return sample
        rand_noise = np.random.randn(*sample["prompt_depth"].shape) * self.std
        prompt_depth = sample["prompt_depth"] + rand_noise * sample["prompt_depth"]
        prompt_mask = np.logical_and(sample["prompt_mask"], prompt_depth > EPS).astype(np.uint8)
        if prompt_mask.sum() <= 1000:
            return sample
        sample["prompt_depth"] = prompt_depth
        sample["prompt_mask"] = prompt_mask
        if prompt_mask.sum() >= 100:
            sample["prompt_disparity"][prompt_mask == 1] = 1.0 / (
                np.clip(prompt_depth[prompt_mask == 1], a_min=1e-3, a_max=None)
            )
        return sample


class RandomDropPatch:
    def __init__(self, scales=[3, 4], probs=[0.5, 0.5], act_prob=0.5):
        self.scales = scales
        self.probs = probs
        self.act_prob = act_prob

    def __call__(self, sample):
        if np.random.random() > self.act_prob:
            return sample
        if sample["prompt_mask"].sum() <= 10:
            return sample
        # print("Drop patch starting", sample['prompt_mask'].sum())
        scale = np.random.choice(self.scales, p=self.probs)
        height, width = sample["prompt_depth"].shape[1:]
        patch_h, patch_w = height // scale, width // scale
        num_patch = np.random.randint(1, scale * scale // 2)
        drop_mask = np.ones_like(sample["prompt_mask"])
        for _ in range(num_patch):
            h_start = np.random.randint(0, height - 1)
            w_start = np.random.randint(0, width - 1)
            drop_mask[
                :,
                max(h_start - patch_h // 2, 0) : min(h_start + patch_h // 2, height - 1),
                max(w_start - patch_w // 2, 0) : min(w_start + patch_w // 2, width - 1),
            ] = 0
        sample["prompt_mask"] = np.logical_and(sample["prompt_mask"], drop_mask).astype(np.uint8)
        sample["prompt_depth"][sample["prompt_mask"] == 0] = 0.0
        sample["prompt_disparity"][sample["prompt_mask"] == 0] = 0.0
        # print("Drop patch ending", sample['prompt_mask'].sum())
        return sample


class RandomDropFar:
    def __init__(self, drop_far=[0.5, 0.9], act_prob=0.5):
        self.drop_far = drop_far
        self.act_prob = act_prob

    def __call__(self, sample):
        if np.random.random() > self.act_prob:
            return sample
        if sample["prompt_mask"].sum() <= 10:
            return sample
        # print("Drop far starting", sample['prompt_mask'].sum())
        start_drop = np.random.random() * (self.drop_far[1] - self.drop_far[0]) + self.drop_far[0]
        depth_drop = np.percentile(sample["depth"][sample["mask"] != 0], start_drop * 100)
        sample["prompt_mask"] = np.logical_and(sample["prompt_mask"], sample["prompt_depth"] < depth_drop).astype(
            np.uint8
        )
        sample["prompt_depth"][sample["prompt_mask"] == 0] = 0.0
        sample["prompt_disparity"][sample["prompt_mask"] == 0] = 0.0
        # print("Drop far ending", sample['prompt_mask'].sum())
        return sample


class DropAllPrompt:

    def __call__(self, sample):
        sample["prompt_mask"] = np.zeros_like(sample["prompt_mask"])
        sample["prompt_depth"] = np.zeros_like(sample["prompt_depth"])
        sample["prompt_disparity"] = np.zeros_like(sample["prompt_disparity"])
        return sample


class SampleQueryPairs:
    def __init__(self, sample_q=None, split="train", quantile=0.02, debug=False):
        self.sample_q = sample_q
        self.split = split
        self.quantile = quantile
        self.debug = debug
    
    def __call__(self, sample):
        def to_pixel_samples(depth, disparity, mask, disparity_mask, split):
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

            if split == "train":
                coord_depth = coord[mask > 0]
                depth = depth[mask > 0]
                coord_disparity = coord[disparity_mask > 0]
                disparity = disparity[disparity_mask > 0]
            else:
                coord_depth = coord.copy()
                depth = depth.copy()
                coord_disparity = coord.copy()
                disparity = disparity.copy()

            return coord_depth, depth, coord_disparity, disparity

        if "depth" not in sample or "image" not in sample:
            return sample

        depth = sample["depth"]  # [H, W]
        disparity = sample["disparity"]
        mask = sample["mask"]   
        disparity_mask = sample["disparity_mask"] 

        # to pixel samples
        coord_depth, depth_flat, coord_disparity, disparity_flat = to_pixel_samples(depth, disparity, mask, disparity_mask, self.split)
        if self.sample_q is not None:
            if self.debug:
                depth_sample_lst = np.arange(len(coord_depth))[:self.sample_q]
                disparity_sample_lst = np.arange(len(coord_disparity))[:self.sample_q]
            else:
                depth_sample_lst = np.random.choice(
                    len(coord_depth), self.sample_q, replace=False)
                disparity_sample_lst = np.random.choice(
                    len(coord_disparity), self.sample_q, replace=False)
            coord_depth = coord_depth[depth_sample_lst]        # [N, 2]
            depth_flat = depth_flat[depth_sample_lst]          # [N, 1]
            coord_disparity = coord_disparity[disparity_sample_lst]     
            disparity_flat = disparity_flat[disparity_sample_lst]          

        # compute cell size for original high resolution depth map
        cell = np.ones_like(coord_depth)
        cell[:, 0] *= 2 / depth.shape[-2]
        cell[:, 1] *= 2 / depth.shape[-1]

        sample["cell"] = cell
        sample["sampled_coord_depth"] = coord_depth
        sample["sampled_depth"] = depth_flat
        sample["sampled_coord_disparity"] = coord_disparity
        sample["sampled_disparity"] = disparity_flat

        sample["reference_meta"] = {}
        sample["reference_meta"]["depth_min"] = _safe_quantile(depth[mask > 0], self.quantile)
        sample["reference_meta"]["depth_max"] = _safe_quantile(depth[mask > 0], 1 - self.quantile)
        sample["reference_meta"]["depth_median"] = _safe_quantile(depth[mask > 0], 0.5)
        sample["reference_meta"]["disparity_min"] = _safe_quantile(disparity[disparity_mask > 0], self.quantile)
        sample["reference_meta"]["disparity_max"] = _safe_quantile(disparity[disparity_mask > 0], 1 - self.quantile)
        sample["reference_meta"]["disparity_median"] = _safe_quantile(disparity[disparity_mask > 0], 0.5)

        if self.split == "train":
            sample.pop("depth")
            sample.pop("disparity")
            sample.pop("mask")
            sample.pop("disparity_mask")

        return sample


class RapidSampleQueryPairs:
    def __init__(self, sample_q=10000, split="train", quantile=0.02, geometry_type="depth", debug=False):
        self.sample_q = sample_q
        self.split = split
        self.quantile = quantile
        self.geometry_type = geometry_type
        self.debug = debug
    
    def __call__(self, sample):
        if "depth" not in sample or "image" not in sample:
            return sample

        depth = sample["depth"]  # [H, W]
        disparity = sample["disparity"]
        mask = sample["mask"]   
        disparity_mask = sample["disparity_mask"] 
        normal = sample.get("normal", None)
        normal_mask = (depth <= 50).astype(bool, copy=False)

        if self.geometry_type == "depth":
            depth = np.log(depth + 1.0)

        # to pixel samples
        coord_depth, depth_flat, coord_disparity, disparity_flat, normal_for_depth, normal_for_disparity, sampled_normal_mask = self.to_pixel_samples(depth, disparity, mask, disparity_mask, normal, normal_mask, self.split, self.sample_q) 

        # compute cell size for original high resolution depth map
        cell = np.ones_like(coord_depth)
        cell[:, 0] *= 2 / depth.shape[-2]  
        cell[:, 1] *= 2 / depth.shape[-1]  

        sample["cell"] = cell
        sample["hr_size"] = (depth.shape[0], depth.shape[1])
        sample["sampled_coord_depth"] = coord_depth
        sample["sampled_depth"] = depth_flat
        sample["sampled_coord_disparity"] = coord_disparity
        sample["sampled_disparity"] = disparity_flat
        sample["sampled_normal_for_depth"] = normal_for_depth
        sample["sampled_normal_for_disparity"] = normal_for_disparity
        sample["sampled_normal_mask"] = sampled_normal_mask

        sample["reference_meta"] = {}

        valid_depth = depth[mask > 0]
        if valid_depth.size > 10:
            sample["reference_meta"]["depth_min"]     = _safe_quantile(valid_depth, self.quantile)
            sample["reference_meta"]["depth_max"]     = _safe_quantile(valid_depth, 1 - self.quantile)
            sample["reference_meta"]["depth_median"]  = _safe_quantile(valid_depth, 0.5)
        else:
            sample["reference_meta"]["depth_min"]     = None
            sample["reference_meta"]["depth_max"]     = None
            sample["reference_meta"]["depth_median"]  = None

        valid_disp = disparity[disparity_mask > 0]
        if valid_disp.size > 10:
            sample["reference_meta"]["disparity_min"]     = _safe_quantile(valid_disp, self.quantile)
            sample["reference_meta"]["disparity_max"]     = _safe_quantile(valid_disp, 1 - self.quantile)
            sample["reference_meta"]["disparity_median"]  = _safe_quantile(valid_disp, 0.5)
        else:
            sample["reference_meta"]["disparity_min"]     = None
            sample["reference_meta"]["disparity_max"]     = None
            sample["reference_meta"]["disparity_median"]  = None

        if self.split == "train":
            sample.pop("depth")
            sample.pop("disparity")
            sample.pop("mask")
            sample.pop("disparity_mask")
            sample.pop("normal", None)
        
        return sample

    def to_pixel_samples(self, depth, disparity, mask, disparity_mask, normal, normal_mask, split, sample_q=10000):
        H, W = depth.shape[:2]

        if split == "train":
            mask = mask.astype(bool, copy=False)
            idx_d = mask.ravel(order="K").nonzero()[0]
            # idx_d = np.flatnonzero(mask > 0)
            if idx_d.size == 0:
                coord_depth = np.empty((0, 2), dtype=np.float32)
                depth_vals = np.empty((0, 1), dtype=depth.dtype)
                normal_for_depth = np.empty((0, 3), dtype=np.float32) 
                sampled_normal_mask = np.empty((0,), dtype=bool)
            else:
                if sample_q is not None:
                    sel_d = np.random.choice(idx_d, size=sample_q, replace=(sample_q > idx_d.size))
                else:
                    sel_d = idx_d
                y_d, x_d = np.divmod(sel_d, W)
                coord_depth = norm_coord(y_d, x_d, H, W)
                depth_vals = depth[y_d, x_d].reshape(-1, 1)
                normal_for_depth = normal[y_d, x_d].reshape(-1, 3) if normal is not None else np.empty((0, 3), dtype=np.float32) 
                sampled_normal_mask = normal_mask[y_d, x_d].reshape(-1) if normal is not None else np.empty((0), dtype=np.float32) 
            
            disparity_mask = disparity_mask.astype(bool, copy=False)
            idx_p = disparity_mask.ravel(order="K").nonzero()[0]
            # idx_p = np.flatnonzero(disparity_mask > 0)
            if idx_p.size == 0:
                coord_disparity = np.empty((0, 2), dtype=np.float32)
                disparity_vals = np.empty((0, 1), dtype=disparity.dtype)
                normal_for_disparity = np.empty((0, 3), dtype=np.float32)
            else:
                if sample_q is not None:
                    sel_p = np.random.choice(idx_p, size=sample_q, replace=(sample_q > idx_p.size)) 
                else:
                    sel_p = idx_p
                y_p, x_p = np.divmod(sel_p, W)
                coord_disparity = norm_coord(y_p, x_p, H, W)
                disparity_vals = disparity[y_p, x_p].reshape(-1, 1)
                normal_for_disparity = normal[y_p, x_p].reshape(-1, 3) if normal is not None else np.empty((0, 3), dtype=np.float32)

            return coord_depth, depth_vals, coord_disparity, disparity_vals, normal_for_depth, normal_for_disparity, sampled_normal_mask
        
        else:
            ys = (2.0 * (np.arange(H, dtype=np.float32) + 0.5) / H) - 1.0
            xs = (2.0 * (np.arange(W, dtype=np.float32) + 0.5) / W) - 1.0
            yy, xx = np.meshgrid(ys, xs, indexing='ij')  # [H,W]
            coord = np.stack([yy, xx], axis=-1).reshape(-1, 2).astype(np.float32, copy=False)

            depth_vals     = depth.reshape(-1, 1)
            disparity_vals = disparity.reshape(-1, 1)
            normal_for_depth = normal.reshape(-1, 3) if normal is not None else np.empty((0, 3), dtype=np.float32)
            normal_for_disparity = normal.reshape(-1, 3) if normal is not None else np.empty((0, 3), dtype=np.float32)
            sampled_normal_mask = normal_mask.reshape(-1) if normal is not None else np.empty((0, 3), dtype=np.float32)

            return coord, depth_vals, coord.copy(), disparity_vals, normal_for_depth, normal_for_disparity, sampled_normal_mask


class UniformSampleQueryPairs:
    def __init__(self, sample_q=None, split="train", debug=False):
        self.sample_q = sample_q
        self.split = split
        self.debug = debug
    
    def __call__(self, sample):
        def _choose_ny_nx(H, W, N):
            if N <= 0:
                return 0, 0
            aspect = H / W
            ny0 = max(1, int(np.sqrt(N * aspect)))
            cands = []
            for ny in range(max(1, ny0 - 2), ny0 + 3):
                for nx in (max(1, int(np.ceil(N / max(ny,1)))),
                        max(1, int(np.floor(N / max(ny,1))))):
                    prod = ny * nx
                    ar_err = abs((ny / nx) - aspect)
                    cands.append(((0 if prod >= N else 1), abs(prod - N), ar_err, prod, ny, nx))
            cands.sort()
            _, _, _, _, ny, nx = cands[0]
            return ny, nx

        def _masked_uniform_indices(mask, H, W, N):
            mask = mask.astype(bool)
            true_flat = np.flatnonzero(mask)
            M_true = true_flat.size

            if N <= 0 or M_true == 0:
                return np.empty((0,), dtype=int)
            if N >= M_true:
                return true_flat

            ny, nx = _choose_ny_nx(H, W, N)
            ny = max(1, min(ny, H))
            nx = max(1, min(nx, W))

            y = true_flat // W
            x = true_flat %  W

            while True:
                ybin = (y * ny) // H
                xbin = (x * nx) // W
                ybin = np.minimum(ybin, ny - 1)
                xbin = np.minimum(xbin, nx - 1)

                bid = ybin * nx + xbin
                uniq_bid, first_idx = np.unique(bid, return_index=True)
                reps = true_flat[first_idx]

                if reps.size >= N or (ny == H and nx == W):
                    break

                factor = max(2, int(np.ceil(np.sqrt(N / max(1, reps.size)))))
                ny = min(H, ny * factor)
                nx = min(W, nx * factor)

            if reps.size >= N:
                keep = np.linspace(0, reps.size - 1, N, dtype=int)
                chosen_flat = reps[keep]
            else:
                remaining = np.setdiff1d(true_flat, reps, assume_unique=False)
                need = N - reps.size
                extra = remaining[np.linspace(0, remaining.size - 1, need, dtype=int)]
                chosen_flat = np.concatenate([reps, extra])

            return chosen_flat

        def sample_depth_disparity(H, W, N, depth_mask, disparity_mask):
            depth_idx = _masked_uniform_indices(depth_mask, H, W, N)
            disp_idx  = _masked_uniform_indices(disparity_mask, H, W, N)
            return depth_idx, disp_idx

        def to_pixel_samples(depth, disparity, mask, disparity_mask, split, sample_n):
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

            if split == "train":
                depth_idx, disparity_idx = sample_depth_disparity(h, w, sample_n, mask, disparity_mask)
                coord_depth = coord[depth_idx]
                coord_disparity = coord[disparity_idx]
                depth = depth[depth_idx]
                disparity = disparity[disparity_idx]
            else:
                coord_depth = coord.copy()
                coord_disparity = coord.copy()
                depth = depth.copy()
                disparity = disparity.copy()

            return coord_depth, depth, coord_disparity, disparity

        if "depth" not in sample or "image" not in sample:
            return sample

        depth = sample["depth"]  # [H, W]
        disparity = sample["disparity"]
        mask = sample["mask"]   
        disparity_mask = sample["disparity_mask"] 

        # to pixel samples
        coord_depth, depth_flat, coord_disparity, disparity_flat = to_pixel_samples(depth, disparity, mask, disparity_mask, self.split, self.sample_q)

        # compute cell size for original high resolution depth map
        cell = np.ones_like(coord_depth)
        cell[:, 0] *= 2 / depth.shape[-2]
        cell[:, 1] *= 2 / depth.shape[-1]

        sample["cell"] = cell
        sample["sampled_coord_depth"] = coord_depth
        sample["sampled_depth"] = depth_flat
        sample["sampled_coord_disparity"] = coord_disparity
        sample["sampled_disparity"] = disparity_flat
        
        return sample
    

class ImportanceSampling:
    def __init__(self, sample_q=None, split="train", quantile=0.02, debug=False):
        self.sample_q = sample_q
        self.split = split
        self.quantile = quantile
        self.debug = debug
    
    def __call__(self, sample):
        if "depth" not in sample or "image" not in sample:
            return sample

        depth = sample["depth"]  # [H, W]
        disparity = sample["disparity"]
        mask = sample["mask"]   
        disparity_mask = sample["disparity_mask"] 

        # to pixel samples
        if self.split == 'train':
            coord_depth, depth_flat, coord_disparity, disparity_flat = importance_sampling(depth, disparity, mask, disparity_mask, n=self.sample_q, debug=self.debug)      
        else:
            coord_depth, depth_flat, coord_disparity, disparity_flat = pixel_sampling(depth, disparity, mask, disparity_mask)        

        # compute cell size for original high resolution depth map
        cell = np.ones_like(coord_depth)
        cell[:, 0] *= 2 / depth.shape[-2]
        cell[:, 1] *= 2 / depth.shape[-1]

        sample["cell"] = cell
        sample["sampled_coord_depth"] = coord_depth
        sample["sampled_depth"] = depth_flat
        sample["sampled_coord_disparity"] = coord_disparity
        sample["sampled_disparity"] = disparity_flat
        
        sample["reference_meta"] = {}
        sample["reference_meta"]["depth_min"] = _safe_quantile(depth[mask > 0], self.quantile)
        sample["reference_meta"]["depth_max"] = _safe_quantile(depth[mask > 0], 1 - self.quantile)
        sample["reference_meta"]["depth_median"] = _safe_quantile(depth[mask > 0], 0.5)
        sample["reference_meta"]["disparity_min"] = _safe_quantile(disparity[disparity_mask > 0], self.quantile)
        sample["reference_meta"]["disparity_max"] = _safe_quantile(disparity[disparity_mask > 0], 1 - self.quantile)
        sample["reference_meta"]["disparity_median"] = _safe_quantile(disparity[disparity_mask > 0], 0.5)

        if self.split == "train":
            sample.pop("depth")
            sample.pop("disparity")
            sample.pop("mask")
            sample.pop("disparity_mask")

        return sample


class DownsampleImplicit:

    def __init__(self, scale=4.0, scale_min=1.0, scale_max=8.0, inp_size=None, sample_q=None, multiple=14, image_interpolation_method=cv2.INTER_AREA, split="train", debug=False):
        self.scale = scale
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.inp_size = inp_size
        self.sample_q = sample_q
        self.multiple = multiple 
        self.image_interpolation_method = image_interpolation_method
        self.split = split
        self.debug = debug

    def floor_to_multiple(self, x):
        return (x // self.multiple) * self.multiple
    
    def __call__(self, sample):
        def resize_fn(img, size):
            # img: numpy array [H, W, 3] or [H, W]
            if img.ndim == 3:
                out = cv2.resize(img, (size[1], size[0]), interpolation=self.image_interpolation_method)  
            elif img.ndim == 2:
                out = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")
            return out

        def to_pixel_samples(depth, disparity, mask, disparity_mask, split):
            h, w = depth.shape

            def make_coord_seq(n, v0=-1, v1=1):
                r = (v1 - v0) / (2 * n)
                return v0 + r + (2 * r) * np.arange(n, dtype=np.float32)

            ys = make_coord_seq(h)
            xs = make_coord_seq(w)
            yy, xx = np.meshgrid(ys, xs, indexing='ij')  # shape [H, W]
            coord = np.stack([yy, xx], axis=-1).reshape(-1, 2)

            depth = depth.reshape(-1, 1)
            disparity = disparity.reshape(-1, 1)
            mask = mask.reshape(-1)
            disparity_mask = disparity_mask.reshape(-1)

            if split == "train":
                coord_depth = coord[mask > 0]
                depth = depth[mask > 0]
                coord_disparity = coord[disparity_mask > 0]
                disparity = disparity[disparity_mask > 0]
            else:
                coord_depth = coord.copy()
                depth = depth.copy()
                coord_disparity = coord.copy()
                disparity = disparity.copy()

            return coord_depth, depth, coord_disparity, disparity

        if "depth" not in sample or "image" not in sample:
            return sample

        hr_rgb = sample["image"]    # [H, W, 3]
        hr_depth = sample["depth"]  # [H, W]
        hr_disparity = sample["disparity"]
        hr_mask = sample["mask"]   
        hr_disparity_mask = sample["disparity_mask"] 
        hr_prompt_depth = sample["prompt_depth"]  
        hr_prompt_disparity = sample["prompt_disparity"]
        hr_prompt_mask = sample["prompt_mask"]
        
        # s = random.uniform(self.scale_min, self.scale_max)  # not implemented yet
        s = self.scale

        if self.inp_size is None:
            h_lr = self.floor_to_multiple(math.floor(hr_depth.shape[0] / s + 1e-9))
            w_lr = self.floor_to_multiple(math.floor(hr_depth.shape[1] / s + 1e-9))
            h_hr = self.floor_to_multiple(round(h_lr * s))
            w_hr = self.floor_to_multiple(round(w_lr * s))

            # crop first 
            hr_rgb = hr_rgb[:h_hr, :w_hr, :]

            hr_depth = hr_depth[:h_hr, :w_hr]
            hr_disparity = hr_disparity[:h_hr, :w_hr]
            hr_mask = hr_mask[:h_hr, :w_hr]
            hr_disparity_mask = hr_disparity_mask[:h_hr, :w_hr]

            hr_prompt_depth = hr_prompt_depth[:h_hr, :w_hr]
            hr_prompt_disparity = hr_prompt_disparity[:h_hr, :w_hr]
            hr_prompt_mask = hr_prompt_mask[:h_hr, :w_hr]

            # downsample rgb, prompt dpt, disparity, mask
            lr_rgb = resize_fn(hr_rgb, (h_lr, w_lr))
            lr_prompt_depth = resize_fn(hr_prompt_depth, (h_lr, w_lr))
            lr_prompt_disparity = resize_fn(hr_prompt_disparity, (h_lr, w_lr))
            lr_prompt_mask = resize_fn(hr_prompt_mask, (h_lr, w_lr))
        else:
            w_lr = self.floor_to_multiple(self.inp_size)
            w_hr = self.floor_to_multiple(round(w_lr * s))
            x0 = random.randint(0, hr_depth.shape[0] - w_hr)
            y0 = random.randint(0, hr_depth.shape[1] - w_hr)

            # crop first 
            hr_rgb = hr_rgb[x0: x0 + w_hr, y0: y0 + w_hr, :]

            hr_depth = hr_depth[x0: x0 + w_hr, y0: y0 + w_hr]
            hr_disparity = hr_disparity[x0: x0 + w_hr, y0: y0 + w_hr]
            hr_mask = hr_mask[x0: x0 + w_hr, y0: y0 + w_hr]
            hr_disparity_mask = hr_disparity_mask[x0: x0 + w_hr, y0: y0 + w_hr]

            hr_prompt_depth = hr_prompt_depth[x0: x0 + w_hr, y0: y0 + w_hr] 
            hr_prompt_disparity = hr_prompt_disparity[x0: x0 + w_hr, y0: y0 + w_hr]  
            hr_prompt_mask = hr_prompt_mask[x0: x0 + w_hr, y0: y0 + w_hr] 

            # downsample rgb, prompt dpt, disparity, mask
            lr_rgb = resize_fn(hr_rgb, (w_lr, w_lr))
            lr_prompt_depth = resize_fn(hr_prompt_depth, (w_lr, w_lr))
            lr_prompt_disparity = resize_fn(hr_prompt_disparity, (w_lr, w_lr))
            lr_prompt_mask = resize_fn(hr_prompt_mask, (w_lr, w_lr))

        # to pixel samples
        hr_coord_depth, hr_depth_flat, hr_coord_disparity, hr_disparity_flat = to_pixel_samples(hr_depth, hr_disparity, hr_mask, hr_disparity_mask, self.split)
        if self.sample_q is not None:
            if self.debug:
                depth_sample_lst = np.arange(len(hr_coord_depth))[:self.sample_q]
                disparity_sample_lst = np.arange(len(hr_coord_disparity))[:self.sample_q]
            else:
                depth_sample_lst = np.random.choice(
                    len(hr_coord_depth), self.sample_q, replace=False)
                disparity_sample_lst = np.random.choice(
                    len(hr_coord_disparity), self.sample_q, replace=False)
            hr_coord_depth = hr_coord_depth[depth_sample_lst]        # [N, 2]
            hr_depth_flat = hr_depth_flat[depth_sample_lst]          # [N, 1]
            hr_coord_disparity = hr_coord_disparity[disparity_sample_lst]     
            hr_disparity_flat = hr_disparity_flat[disparity_sample_lst]          

        # compute cell size for original high resolution depth map
        cell = np.ones_like(hr_coord_depth)
        cell[:, 0] *= 2 / hr_depth.shape[-2]
        cell[:, 1] *= 2 / hr_depth.shape[-1]

        sample["image"] = hr_rgb  
        sample["depth"] = hr_depth
        sample["mask"] = hr_mask
        sample["disparity"] = hr_disparity
        sample["disparity_mask"] = hr_disparity_mask

        sample["lr_image"] = lr_rgb
        sample["prompt_depth"] = lr_prompt_depth
        sample["prompt_disparity"] = lr_prompt_disparity
        sample["prompt_mask"] = lr_prompt_mask

        sample["cell"] = cell
        sample["hr_sampled_coord_depth"] = hr_coord_depth
        sample["hr_sampled_depth"] = hr_depth_flat
        sample["hr_sampled_coord_disparity"] = hr_coord_disparity
        sample["hr_sampled_disparity"] = hr_disparity_flat
        
        return sample



class PatchSampleQueryPairs:
    
    def __init__(
        self, 
        num_patches: int = 2000, 
        patch_size: int = 5, 
        split: str = "train", 
        quantile: float = 0.02, 
        geometry_type: str = "depth",
        use_log_depth: bool = True,
        min_distance_ratio: float = 0.5,
        debug: bool = False
    ):
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.half_size = patch_size // 2
        self.split = split
        self.quantile = quantile
        self.geometry_type = geometry_type
        self.use_log_depth = use_log_depth
        self.min_distance_ratio = min_distance_ratio
        self.min_distance = int(patch_size * min_distance_ratio)
        self.debug = debug

        offsets = np.arange(-self.half_size, self.half_size + 1)
        dy, dx = np.meshgrid(offsets, offsets, indexing='ij')
        self.dy_offsets = dy.flatten()
        self.dx_offsets = dx.flatten()
        self.patch_pixel_count = patch_size * patch_size

    def __call__(self, sample):
        if "depth" not in sample or "image" not in sample:
            return sample

        depth = sample["depth"].copy()
        disparity = sample["disparity"].copy()
        mask = sample["mask"]
        disparity_mask = sample["disparity_mask"]
        normal = sample.get("normal", None)
        normal_mask = (depth <= 50).astype(bool, copy=False) if normal is not None else None
        
        H, W = depth.shape[:2]

        if self.use_log_depth and self.geometry_type == "depth":
            depth = np.log(depth + 1.0)

        if self.split == "train":
            results = self._sample_patches_train(
                depth, disparity, mask, disparity_mask, normal, normal_mask, H, W
            )
            (coord, vals, val_mask, normals, normal_masks, patch_info) = results
            
            cell = np.ones_like(coord)
            cell[:, 0] *= 2 / H
            cell[:, 1] *= 2 / W
            
            sample["cell"] = cell
            sample["hr_size"] = (H, W)
            
            sample[f"sampled_coord_{self.geometry_type}"] = coord
            sample[f"sampled_{self.geometry_type}"] = vals
            sample[f"sampled_{self.geometry_type}_mask"] = val_mask
            sample[f"sampled_normal_for_{self.geometry_type}"] = normals
            sample[f"sampled_normal_mask_for_{self.geometry_type}"] = normal_masks
            sample[f"patch_info_{self.geometry_type}"] = patch_info

        else:
            coord, vals = self._sample_full(depth, H, W)
            
            cell = np.ones_like(coord)
            cell[:, 0] *= 2 / H
            cell[:, 1] *= 2 / W
            
            sample["cell"] = cell
            sample["hr_size"] = (H, W)
            sample["sampled_coord_depth"] = coord
            sample["sampled_depth"] = vals
            sample["sampled_coord_disparity"] = coord.copy()
            sample["sampled_disparity"] = disparity.reshape(-1, 1)
            sample["sampled_normal_for_depth"] = normal.reshape(-1, 3) if normal is not None else np.empty((0, 3), dtype=np.float32)
            sample["sampled_normal_for_disparity"] = normal.reshape(-1, 3) if normal is not None else np.empty((0, 3), dtype=np.float32)
            sample["sampled_normal_mask"] = normal_mask.reshape(-1) if normal_mask is not None else np.empty((0,), dtype=bool)
        
        sample["reference_meta"] = self._compute_reference_meta(
            sample["depth"], sample["mask"],
            sample["disparity"], sample["disparity_mask"]
        )
        
        if self.split == "train":
            sample.pop("depth", None)
            sample.pop("disparity", None)
            sample.pop("mask", None)
            sample.pop("disparity_mask", None)
            sample.pop("normal", None)
        
        return sample

    def _sample_patches_train(self, depth, disparity, mask, disparity_mask, normal, normal_mask, H, W):
        if self.geometry_type == "depth":
            value_map = depth
            valid_mask = mask.astype(bool, copy=False)
        else:
            value_map = disparity
            valid_mask = disparity_mask.astype(bool, copy=False)
        
        centers = self._select_patch_centers(valid_mask, H, W, self.num_patches)
        
        y_coords = centers[:, 0:1] + self.dy_offsets[None, :]
        x_coords = centers[:, 1:2] + self.dx_offsets[None, :]
        
        y_flat = np.clip(y_coords.flatten(), 0, H - 1)
        x_flat = np.clip(x_coords.flatten(), 0, W - 1)
        
        coords = norm_coord(y_flat, x_flat, H, W)
        
        values = value_map[y_flat, x_flat].reshape(-1, 1)
        
        validity_mask = valid_mask[y_flat, x_flat]
        
        if normal is not None:
            normals = normal[y_flat, x_flat]
            normal_masks = normal_mask[y_flat, x_flat]
        else:
            total_points = self.num_patches * self.patch_pixel_count
            normals = np.zeros((total_points, 3), dtype=np.float32)
            normal_masks = np.zeros((total_points,), dtype=bool)
        
        patch_info = {
            'num_patches': self.num_patches,
            'patch_pixel_count': self.patch_pixel_count,
            'patch_size': self.patch_size,
        }
        
        return coords, values, validity_mask, normals, normal_masks, patch_info
    
    def _sample_full(self, depth, H, W):
        ys = (2.0 * (np.arange(H, dtype=np.float32) + 0.5) / H) - 1.0
        xs = (2.0 * (np.arange(W, dtype=np.float32) + 0.5) / W) - 1.0
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        coord = np.stack([yy, xx], axis=-1).reshape(-1, 2).astype(np.float32, copy=False)
        vals = depth.reshape(-1, 1)
        return coord, vals
    
    def _select_patch_centers(self, mask, H, W, num_patches):
        valid_y_min = self.half_size
        valid_y_max = H - self.half_size
        valid_x_min = self.half_size
        valid_x_max = W - self.half_size
        
        if valid_y_max <= valid_y_min or valid_x_max <= valid_x_min:
            y_centers = np.random.randint(max(0, valid_y_min), min(H, valid_y_max + 1), size=num_patches)
            x_centers = np.random.randint(max(0, valid_x_min), min(W, valid_x_max + 1), size=num_patches)
            return np.stack([y_centers, x_centers], axis=1)
        
        if self.min_distance <= 0:
            return self._simple_random_centers(mask, valid_y_min, valid_y_max, valid_x_min, valid_x_max, num_patches)
        
        return self._distance_constrained_centers(
            mask, H, W, valid_y_min, valid_y_max, valid_x_min, valid_x_max, num_patches
        )
    
    def _simple_random_centers(self, mask, valid_y_min, valid_y_max, valid_x_min, valid_x_max, num_patches):
        n_candidates = num_patches * 5
        y_candidates = np.random.randint(valid_y_min, valid_y_max + 1, size=n_candidates)
        x_candidates = np.random.randint(valid_x_min, valid_x_max + 1, size=n_candidates)
        
        valid_mask = mask[y_candidates, x_candidates]
        valid_y = y_candidates[valid_mask]
        valid_x = x_candidates[valid_mask]
        
        if len(valid_y) >= num_patches:
            indices = np.random.choice(len(valid_y), num_patches, replace=False)
            return np.stack([valid_y[indices], valid_x[indices]], axis=1)
        elif len(valid_y) > 0:
            n_needed = num_patches - len(valid_y)
            y_extra = np.random.randint(valid_y_min, valid_y_max + 1, size=n_needed)
            x_extra = np.random.randint(valid_x_min, valid_x_max + 1, size=n_needed)
            all_y = np.concatenate([valid_y, y_extra])
            all_x = np.concatenate([valid_x, x_extra])
            return np.stack([all_y, all_x], axis=1)
        else:
            y_centers = np.random.randint(valid_y_min, valid_y_max + 1, size=num_patches)
            x_centers = np.random.randint(valid_x_min, valid_x_max + 1, size=num_patches)
            return np.stack([y_centers, x_centers], axis=1)
    
    def _distance_constrained_centers(self, mask, H, W, valid_y_min, valid_y_max, valid_x_min, valid_x_max, num_patches):
        n_candidates = num_patches * 20
        y_candidates = np.random.randint(valid_y_min, valid_y_max + 1, size=n_candidates)
        x_candidates = np.random.randint(valid_x_min, valid_x_max + 1, size=n_candidates)
        
        valid_in_mask = mask[y_candidates, x_candidates]
        
        valid_indices = np.where(valid_in_mask)[0]
        invalid_indices = np.where(~valid_in_mask)[0]
        np.random.shuffle(valid_indices)
        np.random.shuffle(invalid_indices)
        sorted_indices = np.concatenate([valid_indices, invalid_indices])
        
        y_sorted = y_candidates[sorted_indices]
        x_sorted = x_candidates[sorted_indices]
        
        selected_centers = []
        current_min_dist = self.min_distance
        
        while len(selected_centers) < num_patches and current_min_dist >= 0:
            for i in range(len(y_sorted)):
                if len(selected_centers) >= num_patches:
                    break
                    
                y, x = y_sorted[i], x_sorted[i]
                
                if len(selected_centers) == 0:
                    selected_centers.append((y, x))
                else:
                    selected_arr = np.array(selected_centers)
                    distances = np.sqrt((selected_arr[:, 0] - y) ** 2 + (selected_arr[:, 1] - x) ** 2)
                    if np.all(distances >= current_min_dist):
                        selected_centers.append((y, x))
            
            if len(selected_centers) < num_patches:
                current_min_dist = current_min_dist * 0.7
                if current_min_dist < 1:
                    current_min_dist = 0
        
        if len(selected_centers) < num_patches:
            n_needed = num_patches - len(selected_centers)
            y_extra = np.random.randint(valid_y_min, valid_y_max + 1, size=n_needed)
            x_extra = np.random.randint(valid_x_min, valid_x_max + 1, size=n_needed)
            for y, x in zip(y_extra, x_extra):
                selected_centers.append((y, x))
        
        centers = np.array(selected_centers[:num_patches])
        return centers
    
    def _compute_reference_meta(self, depth, mask, disparity, disparity_mask):
        meta = {}
        
        valid_depth = depth[mask > 0]
        if valid_depth.size > 10:
            meta["depth_min"] = _safe_quantile(valid_depth, self.quantile)
            meta["depth_max"] = _safe_quantile(valid_depth, 1 - self.quantile)
            meta["depth_median"] = _safe_quantile(valid_depth, 0.5)
        else:
            meta["depth_min"] = None
            meta["depth_max"] = None
            meta["depth_median"] = None
        
        valid_disp = disparity[disparity_mask > 0]
        if valid_disp.size > 10:
            meta["disparity_min"] = _safe_quantile(valid_disp, self.quantile)
            meta["disparity_max"] = _safe_quantile(valid_disp, 1 - self.quantile)
            meta["disparity_median"] = _safe_quantile(valid_disp, 0.5)
        else:
            meta["disparity_min"] = None
            meta["disparity_max"] = None
            meta["disparity_median"] = None
        
        return meta
    

class HybridSampleQueryPairs:
    
    def __init__(
        self,
        num_random_samples: int = 10000,
        num_patches: int = 10,
        patch_size: int = 64,
        split: str = "train",
        quantile: float = 0.02,
        geometry_type: str = "depth",
        use_log_depth: bool = True,
        min_distance_ratio: float = 0.5,
        debug: bool = False,
    ):
        self.num_random_samples = num_random_samples
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.half_size = patch_size // 2
        self.split = split
        self.quantile = quantile
        self.geometry_type = geometry_type
        self.use_log_depth = use_log_depth
        self.min_distance_ratio = min_distance_ratio
        self.min_distance = int(patch_size * min_distance_ratio)
        self.debug = debug
        
        offsets = np.arange(patch_size) - self.half_size  # [-half_size, ..., half_size-1]
        dy, dx = np.meshgrid(offsets, offsets, indexing='ij')
        self.dy_offsets = dy.flatten()
        self.dx_offsets = dx.flatten()
        self.patch_pixel_count = patch_size * patch_size
    
    def __call__(self, sample):
        if "depth" not in sample or "image" not in sample:
            return sample
        
        depth = sample["depth"].copy()
        disparity = sample["disparity"].copy()
        mask = sample["mask"]
        disparity_mask = sample["disparity_mask"]
        
        H, W = depth.shape[:2]
        
        if self.geometry_type == "depth":
            if self.use_log_depth:
                value_for_loss = np.log(depth + 1.0)
            else:
                value_for_loss = depth
            valid_mask = mask
        else:  # disparity
            value_for_loss = disparity
            valid_mask = disparity_mask
        

        random_results = self._random_sample(value_for_loss, valid_mask, H, W)

        patch_results = self._patch_sample(value_for_loss, valid_mask, H, W)

        sample["hr_size"] = (H, W)
        
        sample[f"global_coord_{self.geometry_type}"] = random_results["coord"]
        sample[f"global_value_{self.geometry_type}"] = random_results["value"]
        sample[f"global_mask_{self.geometry_type}"] = random_results["mask"]
        sample[f"global_cell_{self.geometry_type}"] = random_results["cell"]

        sample[f"local_coord_{self.geometry_type}"] = patch_results["coord"]
        sample[f"local_value_{self.geometry_type}"] = patch_results["value"]
        sample[f"local_mask_{self.geometry_type}"] = patch_results["mask"]
        sample[f"local_cell_{self.geometry_type}"] = patch_results["cell"]
        sample[f"patch_info_{self.geometry_type}"] = patch_results["patch_info"]

        sample["reference_meta"] = self._compute_reference_meta(
            sample["depth"], sample["mask"],
            sample["disparity"], sample["disparity_mask"]
        )
        
        sample.pop("depth", None)
        sample.pop("disparity", None) 
        sample.pop("mask", None)
        sample.pop("disparity_mask", None)
        
        return sample
    
    def _random_sample(self, value, mask, H, W):
        mask_bool = mask.astype(bool, copy=False)
        idx = mask_bool.ravel().nonzero()[0]
        
        if idx.size == 0:
            coord = np.empty((0, 2), dtype=np.float32)
            vals = np.empty((0, 1), dtype=value.dtype)
            val_mask = np.empty((0,), dtype=bool)
        else:
            if self.num_random_samples is not None:
                sel = np.random.choice(idx, size=self.num_random_samples, replace=(self.num_random_samples > idx.size))
            else:
                sel = idx
            y, x = np.divmod(sel, W)
            coord = norm_coord(y, x, H, W)
            vals = value[y, x].reshape(-1, 1)
            val_mask = mask[y, x].astype(bool)
        
        cell = np.ones_like(coord)
        cell[:, 0] *= 2 / H
        cell[:, 1] *= 2 / W
        
        return {
            "coord": coord,
            "value": vals,
            "mask": val_mask,
            "cell": cell,
        }
    
    def _patch_sample(self, value, mask, H, W):
        mask_bool = mask.astype(bool, copy=False)
        centers = self._select_patch_centers(mask_bool, H, W, self.num_patches)
        
        y_coords = centers[:, 0:1] + self.dy_offsets[None, :]
        x_coords = centers[:, 1:2] + self.dx_offsets[None, :]
        
        y_flat = np.clip(y_coords.flatten(), 0, H - 1)
        x_flat = np.clip(x_coords.flatten(), 0, W - 1)
        
        coord = norm_coord(y_flat, x_flat, H, W)
        vals = value[y_flat, x_flat].reshape(-1, 1)
        val_mask = mask_bool[y_flat, x_flat]
        
        cell = np.ones_like(coord)
        cell[:, 0] *= 2 / H
        cell[:, 1] *= 2 / W
        
        patch_info = {
            'num_patches': self.num_patches,
            'patch_pixel_count': self.patch_pixel_count,
            'patch_size': self.patch_size,
        }
        
        return {
            "coord": coord,
            "value": vals,
            "mask": val_mask,
            "cell": cell,
            "patch_info": patch_info,
        }
    
    def _sample_full(self, value, H, W):
        ys = (2.0 * (np.arange(H, dtype=np.float32) + 0.5) / H) - 1.0
        xs = (2.0 * (np.arange(W, dtype=np.float32) + 0.5) / W) - 1.0
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        coord = np.stack([yy, xx], axis=-1).reshape(-1, 2).astype(np.float32, copy=False)
        vals = value.reshape(-1, 1)
        return coord, vals
    
    def _select_patch_centers(self, mask, H, W, num_patches):
        valid_y_min = self.half_size
        valid_y_max = H - self.half_size
        valid_x_min = self.half_size
        valid_x_max = W - self.half_size
        
        if valid_y_max <= valid_y_min or valid_x_max <= valid_x_min:
            y_centers = np.random.randint(max(0, valid_y_min), min(H, valid_y_max + 1), size=num_patches)
            x_centers = np.random.randint(max(0, valid_x_min), min(W, valid_x_max + 1), size=num_patches)
            return np.stack([y_centers, x_centers], axis=1)
        
        if self.min_distance <= 0:
            return self._simple_random_centers(mask, valid_y_min, valid_y_max, valid_x_min, valid_x_max, num_patches)
        
        return self._distance_constrained_centers(
            mask, H, W, valid_y_min, valid_y_max, valid_x_min, valid_x_max, num_patches
        )
    
    def _simple_random_centers(self, mask, valid_y_min, valid_y_max, valid_x_min, valid_x_max, num_patches):
        n_candidates = num_patches * 5
        y_candidates = np.random.randint(valid_y_min, valid_y_max + 1, size=n_candidates)
        x_candidates = np.random.randint(valid_x_min, valid_x_max + 1, size=n_candidates)
        
        valid_mask = mask[y_candidates, x_candidates]
        valid_y = y_candidates[valid_mask]
        valid_x = x_candidates[valid_mask]
        
        if len(valid_y) >= num_patches:
            indices = np.random.choice(len(valid_y), num_patches, replace=False)
            return np.stack([valid_y[indices], valid_x[indices]], axis=1)
        elif len(valid_y) > 0:
            n_needed = num_patches - len(valid_y)
            y_extra = np.random.randint(valid_y_min, valid_y_max + 1, size=n_needed)
            x_extra = np.random.randint(valid_x_min, valid_x_max + 1, size=n_needed)
            all_y = np.concatenate([valid_y, y_extra])
            all_x = np.concatenate([valid_x, x_extra])
            return np.stack([all_y, all_x], axis=1)
        else:
            y_centers = np.random.randint(valid_y_min, valid_y_max + 1, size=num_patches)
            x_centers = np.random.randint(valid_x_min, valid_x_max + 1, size=num_patches)
            return np.stack([y_centers, x_centers], axis=1)
    
    def _distance_constrained_centers(self, mask, H, W, valid_y_min, valid_y_max, valid_x_min, valid_x_max, num_patches):
        n_candidates = num_patches * 20
        y_candidates = np.random.randint(valid_y_min, valid_y_max + 1, size=n_candidates)
        x_candidates = np.random.randint(valid_x_min, valid_x_max + 1, size=n_candidates)
        
        valid_in_mask = mask[y_candidates, x_candidates]
        
        valid_indices = np.where(valid_in_mask)[0]
        invalid_indices = np.where(~valid_in_mask)[0]
        np.random.shuffle(valid_indices)
        np.random.shuffle(invalid_indices)
        sorted_indices = np.concatenate([valid_indices, invalid_indices])
        
        y_sorted = y_candidates[sorted_indices]
        x_sorted = x_candidates[sorted_indices]
        
        selected_centers = []
        current_min_dist = self.min_distance
        
        while len(selected_centers) < num_patches and current_min_dist >= 0:
            for i in range(len(y_sorted)):
                if len(selected_centers) >= num_patches:
                    break
                    
                y, x = y_sorted[i], x_sorted[i]
                
                if len(selected_centers) == 0:
                    selected_centers.append((y, x))
                else:
                    selected_arr = np.array(selected_centers)
                    distances = np.sqrt((selected_arr[:, 0] - y) ** 2 + (selected_arr[:, 1] - x) ** 2)
                    if np.all(distances >= current_min_dist):
                        selected_centers.append((y, x))
            
            if len(selected_centers) < num_patches:
                current_min_dist = current_min_dist * 0.7
                if current_min_dist < 1:
                    current_min_dist = 0
        
        if len(selected_centers) < num_patches:
            n_needed = num_patches - len(selected_centers)
            y_extra = np.random.randint(valid_y_min, valid_y_max + 1, size=n_needed)
            x_extra = np.random.randint(valid_x_min, valid_x_max + 1, size=n_needed)
            for y, x in zip(y_extra, x_extra):
                selected_centers.append((y, x))
        
        centers = np.array(selected_centers[:num_patches])
        return centers
    
    def _compute_reference_meta(self, depth, mask, disparity, disparity_mask):
        meta = {}
        
        valid_depth = depth[mask > 0]
        if valid_depth.size > 10:
            meta["depth_min"] = _safe_quantile(valid_depth, self.quantile)
            meta["depth_max"] = _safe_quantile(valid_depth, 1 - self.quantile)
            meta["depth_median"] = _safe_quantile(valid_depth, 0.5)
        else:
            meta["depth_min"] = None
            meta["depth_max"] = None
            meta["depth_median"] = None
        
        valid_disp = disparity[disparity_mask > 0]
        if valid_disp.size > 10:
            meta["disparity_min"] = _safe_quantile(valid_disp, self.quantile)
            meta["disparity_max"] = _safe_quantile(valid_disp, 1 - self.quantile)
            meta["disparity_median"] = _safe_quantile(valid_disp, 0.5)
        else:
            meta["disparity_min"] = None
            meta["disparity_max"] = None
            meta["disparity_median"] = None
        
        return meta
