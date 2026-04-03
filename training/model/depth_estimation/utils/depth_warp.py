import math
import torch
import torch.nn as nn
from einops import rearrange

EPS = 1e-3


class WarpIdentity:
    def warp(self, depth, **kwargs):
        return depth, depth > 0.0, None

    def unwarp(self, depth, **kwargs):
        return depth


class WarpFixMinMax:
    def __init__(self, near_depth=1.0, far_depth=80.0, **kwargs):
        self._near_depth = near_depth
        self._far_depth = far_depth

    def warp(self, depth, **kwargs):
        return (depth - self._near_depth) / (self._far_depth - self._near_depth)

    def unwarp(self, depth, **kwargs):
        return depth * (self._far_depth - self._near_depth) + self._near_depth


class WarpMinMaxGT:
    def warp(self, depth, reference, gt_depth, **kwargs):
        if "reference_meta" in kwargs:
            depth_min, depth_max = kwargs["reference_meta"]
        else:
            reference = gt_depth if gt_depth is not None else reference
        quantile = 0.02
        depth_min = torch.quantile(reference.reshape(depth.shape[0], -1), quantile, dim=-1, keepdim=True)
        depth_max = torch.quantile(reference.reshape(depth.shape[0], -1), 1 - quantile, dim=-1, keepdim=True)
        if ((depth_max - depth_min) < EPS).any():
            depth_max[(depth_max - depth_min) < EPS] = depth_min[(depth_max - depth_min) < EPS] + EPS
        return (
            (depth - depth_min[:, None, None]) / (depth_max - depth_min)[:, None, None],
            (depth > EPS),
            (depth_min, depth_max),
        )

    def unwarp(self, depth, reference, gt_depth, **kwargs):
        reference = gt_depth if gt_depth is not None else reference
        depth_min, depth_max = (
            reference.reshape(depth.shape[0], -1).min(1, keepdim=True)[0],
            reference.reshape(depth.shape[0], -1).max(1, keepdim=True)[0],
        )
        if ((depth_max - depth_min) < EPS).any():
            depth_max[(depth_max - depth_min) < EPS] = depth_min[(depth_max - depth_min) < EPS] + EPS
        return depth * (depth_max - depth_min)[:, None, None] + depth_min[:, None, None]


class WarpMinMax:
    def warp(self, depth, reference, **kwargs):
        depth_min, depth_max = (
            reference.reshape(depth.shape[0], -1).min(1, keepdim=True)[0],
            reference.reshape(depth.shape[0], -1).max(1, keepdim=True)[0],
        )
        if ((depth_max - depth_min) < EPS).any():
            depth_max[(depth_max - depth_min) < EPS] = depth_min[(depth_max - depth_min) < EPS] + EPS
        return (depth - depth_min[:, None, None]) / (depth_max - depth_min)[:, None, None]

    def unwarp(self, depth, reference, **kwargs):
        depth_min, depth_max = (
            reference.reshape(depth.shape[0], -1).min(1, keepdim=True)[0],
            reference.reshape(depth.shape[0], -1).max(1, keepdim=True)[0],
        )
        if ((depth_max - depth_min) < EPS).any():
            depth_max[(depth_max - depth_min) < EPS] = depth_min[(depth_max - depth_min) < EPS] + EPS
        return depth * (depth_max - depth_min)[:, None, None] + depth_min[:, None, None]


class WarpLogFix:
    def __init__(self, near_depth=1.0, far_depth=80.0, **kwargs):
        self._near_depth = near_depth
        self._far_depth = far_depth

    def warp(self, depth, **kwargs):
        depth = torch.clamp(depth, self._near_depth, self._far_depth)
        return torch.log(depth / self._near_depth) / math.log(self._far_depth / self._near_depth)

    def unwarp(self, depth, **kwargs):
        return torch.exp(depth * math.log(self._far_depth / self._near_depth)) * self._near_depth


class WarpLog:
    def warp(self, depth, reference, **kwargs):
        depth_min, depth_max = (
            reference.reshape(depth.shape[0], -1).min(1, keepdim=True)[0],
            reference.reshape(depth.shape[0], -1).max(1, keepdim=True)[0],
        )
        if ((depth_max - depth_min) < EPS).any():
            depth_max[(depth_max - depth_min) < EPS] = depth_min[(depth_max - depth_min) < EPS] + EPS
        depth = torch.clamp(depth, depth_min[:, None, None], depth_max[:, None, None])
        return torch.log((depth / depth_min[:, None, None]) / torch.log(depth_max / depth_min)[:, None, None])

    def unwarp(self, depth, reference, **kwargs):
        depth_min, depth_max = (
            reference.reshape(depth.shape[0], -1).min(1, keepdim=True)[0],
            reference.reshape(depth.shape[0], -1).max(1, keepdim=True)[0],
        )
        if ((depth_max - depth_min) < EPS).any():
            depth_max[(depth_max - depth_min) < EPS] = depth_min[(depth_max - depth_min) < EPS] + EPS
        return torch.exp(depth * torch.log(depth_max / depth_min)[:, None, None]) * depth_min[:, None, None]


class WarpPercentile:
    def __init__(self, quantile=0.02, **kwargs):
        self._quantile = quantile

    def warp(self, depth, **kwargs):

        if kwargs.get("reference_meta", None) is not None:
            depth_min, depth_max = kwargs["reference_meta"]
            return (depth - depth_min) / (depth_max - depth_min), (depth > EPS), (depth_min, depth_max)
        ground_truth = kwargs.get("ground_truth", depth)
        ground_truth_mask = kwargs.get("ground_truth_mask", None)
        if ground_truth_mask is None:
            ground_truth_mask = ground_truth > EPS
        depth_min, depth_max = [], []
        batch_size = depth.shape[0]
        for b in range(batch_size):
            depth_min_ = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], self._quantile)
            depth_max_ = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], 1 - self._quantile)
            if depth_min_ == depth_max_:
                depth_min_ = depth_min_ - EPS
                depth_max_ = depth_max_ + EPS
            depth_min.append(depth_min_)
            depth_max.append(depth_max_)
        depth_min = torch.stack(depth_min, dim=0)
        depth_max = torch.stack(depth_max, dim=0)
        depth_max = rearrange(depth_max, "b -> b 1 1 1")
        depth_min = rearrange(depth_min, "b -> b 1 1 1")
        return (depth - depth_min) / (depth_max - depth_min), (depth > EPS), (depth_min, depth_max)

    def unwarp(self, depth, **kwargs):
        reference_meta = kwargs.get("reference_meta", None)
        if reference_meta is None:
            raise ValueError("reference_meta is required")
        if reference_meta is not None:
            depth_min, depth_max = reference_meta
            return depth * (depth_max - depth_min) + depth_min
        ground_truth = kwargs.get("ground_truth", depth)
        ground_truth_mask = kwargs.get("ground_truth_mask", None)
        if ground_truth_mask is None:
            ground_truth_mask = ground_truth > EPS
        depth_min, depth_max = [], []
        batch_size = depth.shape[0]
        for b in range(batch_size):
            depth_min_ = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], self._quantile)
            depth_max_ = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], 1 - self._quantile)
            if depth_min_ == depth_max_:
                depth_min_ = depth_min_ - EPS
                depth_max_ = depth_max_ + EPS
            depth_min.append(depth_min_)
            depth_max.append(depth_max_)
        depth_min = torch.stack(depth_min, dim=0)
        depth_max = torch.stack(depth_max, dim=0)
        depth_max = rearrange(depth_max, "b -> b 1 1 1")
        depth_min = rearrange(depth_min, "b -> b 1 1 1")
        return depth * (depth_max - depth_min) + depth_min


class WarpLogPercentile:
    def __init__(self, quantile=0.02, **kwargs):
        self._quantile = quantile

    def warp(self, depth, **kwargs):
        depth = torch.log(depth+1.)
        if kwargs.get("reference_meta", None) is not None:
            depth_min, depth_max = kwargs["reference_meta"]
            return (depth - depth_min) / torch.clamp(depth_max - depth_min, min=1e-3), (depth > EPS), (depth_min, depth_max)
      
        ground_truth = kwargs["ground_truth"]
        ground_truth = torch.log(ground_truth+1.)  # log-depth here !
        ground_truth_mask = kwargs.get("ground_truth_mask", None)
        if ground_truth_mask is None:
            ground_truth_mask = ground_truth > EPS

        depth_min, depth_max = [], []
        batch_size = depth.shape[0]
        for b in range(batch_size):
            depth_min_ = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], self._quantile)
            depth_max_ = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], 1 - self._quantile)
            if depth_min_ == depth_max_:
                depth_min_ = depth_min_ - EPS
                depth_max_ = depth_max_ + EPS
            depth_min.append(depth_min_)
            depth_max.append(depth_max_)
        depth_min = torch.stack(depth_min, dim=0)
        depth_max = torch.stack(depth_max, dim=0)
        depth_min = rearrange(depth_min, "b -> b 1 1 1")
        depth_max = rearrange(depth_max, "b -> b 1 1 1")

        return (depth - depth_min) / (depth_max - depth_min), (depth > EPS), (depth_min, depth_max)

    def unwarp(self, depth, **kwargs):
        depth_min, depth_max = kwargs.get("reference_meta")
        denorm_depth = depth * torch.clamp(depth_max - depth_min, min=1e-3) + depth_min
        return torch.exp(denorm_depth) - 1.


class WarpMedian:
    def __init__(self, **kwargs):
        self.warp_type = "Median"

    def warp(self, depth, **kwargs):
        if kwargs.get("reference_meta", None) is not None:
            median_val = kwargs["reference_meta"]
            return depth / torch.clamp(median_val, min=1e-3), (depth > EPS), median_val
        prompt_depth = kwargs.get("prompt_depth")
        prompt_mask = kwargs.get("prompt_mask")
        median_val = []
        batch_size = depth.shape[0]
        for b in range(batch_size):
            if (prompt_mask[b] > 0).sum() <= 5:
                if "ground_truth" in kwargs:
                    ground_truth = kwargs["ground_truth"]
                    ground_truth_mask = kwargs["ground_truth_mask"]
                    median = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], 0.5)
                else:
                    raise ValueError("ground_truth is required")
            else:
                median = torch.quantile(prompt_depth[b][prompt_mask[b] > 0.0], 0.5)
                if (median <= 1e-2).any():
                    ground_truth = kwargs["ground_truth"]
                    ground_truth_mask = kwargs["ground_truth_mask"]
                    median = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], 0.5)
            median_val.append(median)
        median_val = torch.stack(median_val, dim=0)
        median_val = rearrange(median_val, "b -> b 1 1 1")
        return depth / torch.clamp(median_val, min=1e-2), (depth >= 0) & (prompt_mask > 0.0), median_val

    def unwarp(self, depth, **kwargs):
        median_val = kwargs.get("reference_meta")
        return depth * torch.clamp(median_val, min=1e-3)


class WarpLogMedian:
    def __init__(self, **kwargs):
        self.warp_type = "LogMedian"

    def warp(self, depth, **kwargs):
        depth = torch.log(depth+1.)
        if kwargs.get("reference_meta", None) is not None:
            median_val = kwargs["reference_meta"]
            return depth / torch.clamp(median_val, min=1e-3), (depth > EPS), median_val
        prompt_depth = kwargs.get("prompt_depth")
        prompt_mask = kwargs.get("prompt_mask")
        prompt_depth = torch.log(prompt_depth+1.)
        median_val = []
        batch_size = depth.shape[0]
        for b in range(batch_size):
            if (prompt_mask[b] > 0).sum() <= 5:
                if "ground_truth" in kwargs:
                    ground_truth = kwargs["ground_truth"]
                    ground_truth_mask = kwargs["ground_truth_mask"]
                    ground_truth = torch.log(ground_truth+1.)
                    median = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], 0.5)
                else:
                    raise ValueError("ground_truth is required")
            else:
                median = torch.quantile(prompt_depth[b][prompt_mask[b] > 0.0], 0.5)
                if (median <= 1e-2).any():
                    ground_truth = kwargs["ground_truth"]
                    ground_truth_mask = kwargs["ground_truth_mask"]
                    ground_truth = torch.log(ground_truth+1.)
                    median = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], 0.5)
            median_val.append(median)
        median_val = torch.stack(median_val, dim=0)
        median_val = rearrange(median_val, "b -> b 1 1 1")
        return depth / torch.clamp(median_val, min=1e-2), (depth >= 0) & (prompt_mask > 0.0), median_val

    def unwarp(self, depth, **kwargs):
        median_val = kwargs.get("reference_meta")
        return torch.exp(depth * torch.clamp(median_val, min=1e-3)) - 1.


class WarpLogMinMax:
    def __init__(self, **kwargs):
        self.warp_type = "LogMinMax"

    def warp(self, depth, lower_thresh=0.02, upper_thresh=0.98, **kwargs):
        depth = torch.log(depth+1.)
        if kwargs.get("reference_meta", None) is not None:
            depth_min, depth_max = kwargs["reference_meta"]
            return (depth - depth_min) / torch.clamp(depth_max - depth_min, min=1e-3) - 0.5, (depth > EPS), (depth_min, depth_max)
        prompt_depth = kwargs.get("prompt_depth")
        prompt_mask = kwargs.get("prompt_mask")
        prompt_depth = torch.log(prompt_depth+1.)
        
        batch_size = depth.shape[0]
        depth_min_list = []
        depth_max_list = []
        for b in range(batch_size):
            if (prompt_mask[b] > 0).sum() <= 5:
                if "ground_truth" in kwargs:
                    ground_truth = kwargs["ground_truth"]
                    ground_truth = torch.log(ground_truth+1.)
                    ground_truth_mask = kwargs["ground_truth_mask"]
                    depth_min = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], lower_thresh)
                    depth_max = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], upper_thresh)
                else:
                    raise ValueError("ground_truth is required")
            else:
                depth_min = torch.quantile(prompt_depth[b][prompt_mask[b] > 0.0], lower_thresh)
                depth_max = torch.quantile(prompt_depth[b][prompt_mask[b] > 0.0], upper_thresh)
                if (depth_max - depth_min <= 1e-2).any():
                    ground_truth = kwargs["ground_truth"]
                    ground_truth = torch.log(ground_truth+1.)
                    ground_truth_mask = kwargs["ground_truth_mask"]
                    depth_min = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], lower_thresh)
                    depth_max = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], upper_thresh)
            
            depth_min_list.append(depth_min)
            depth_max_list.append(depth_max)
            
        depth_min = rearrange(torch.stack(depth_min_list), "b -> b 1 1 1")
        depth_max = rearrange(torch.stack(depth_max_list), "b -> b 1 1 1")
        
        new_depth = (depth - depth_min) / torch.clamp(depth_max - depth_min, min=1e-3) - 0.5
        new_mask = (depth >= 0) & (prompt_mask > 0.0)
        new_depth[new_mask == 0] = 0.0

        return new_depth, new_mask, (depth_min, depth_max)

    def unwarp(self, depth, **kwargs):
        depth_min, depth_max = kwargs.get("reference_meta")
        denorm_depth = (depth + 0.5) * torch.clamp(depth_max - depth_min, min=1e-3) + depth_min
        return torch.exp(denorm_depth) - 1.
    

class WarpLogGTMinMax:
    def __init__(self, **kwargs):
        self.warp_type = "GTLogMinMax"

    def warp(self, depth, lower_thresh=0.02, upper_thresh=0.98, **kwargs):
        if kwargs.get("reference_meta", None) is not None:
            depth = torch.log(depth+1.)   # gt depth
            depth_min, depth_max = kwargs["reference_meta"]
            new_depth = (depth - depth_min) / torch.clamp(depth_max - depth_min, min=1e-3)
            new_depth = torch.clamp(new_depth, 0, 1) - 0.5
            return new_depth, (depth > EPS), (depth_min, depth_max)
        
        prompt_depth = kwargs.get("prompt_depth")
        prompt_mask = kwargs.get("prompt_mask")
        prompt_depth = torch.log(prompt_depth+1.)

        gt_depth = kwargs.get("ground_truth")
        gt_depth_mask = kwargs.get("ground_truth_mask")
        gt_depth = torch.log(gt_depth+1.)

        batch_size = depth.shape[0]
        gt_depth_min_list = []
        gt_depth_max_list = []
        for b in range(batch_size):
            depth_min = torch.quantile(gt_depth[b][gt_depth_mask[b] > 0.0], lower_thresh)
            depth_max = torch.quantile(gt_depth[b][gt_depth_mask[b] > 0.0], upper_thresh)

            gt_depth_min_list.append(depth_min)
            gt_depth_max_list.append(depth_max)
            
        gt_depth_min = rearrange(torch.stack(gt_depth_min_list), "b -> b 1 1 1")
        gt_depth_max = rearrange(torch.stack(gt_depth_max_list), "b -> b 1 1 1")
        
        new_prompt_depth = (prompt_depth - gt_depth_min) / torch.clamp(gt_depth_max - gt_depth_min, min=1e-3)
        new_prompt_depth = torch.clamp(new_prompt_depth, 0, 1) - 0.5
        new_prompt_mask = (prompt_depth >= 0) & (prompt_mask > 0.0)
        new_prompt_depth[new_prompt_mask == 0] = 0.0
        
        return new_prompt_depth, new_prompt_mask, (gt_depth_min, gt_depth_max)

    def unwarp(self, depth, **kwargs):
        depth_min, depth_max = kwargs.get("reference_meta")
        denorm_depth = (depth + 0.5) * torch.clamp(depth_max - depth_min, min=1e-3) + depth_min
        return torch.exp(denorm_depth) - 1
    

class WarpLogMinMaxPositive:
    def __init__(self, **kwargs):
        self.warp_type = "LogMinMax"

    def warp(self, depth, lower_thresh=0.02, upper_thresh=0.98, **kwargs):
        depth = torch.log(depth+1.)
        if kwargs.get("reference_meta", None) is not None:
            depth_min, depth_max = kwargs["reference_meta"]
            return (depth - depth_min) / torch.clamp(depth_max - depth_min, min=1e-3) - 0.5, (depth > EPS), (depth_min, depth_max)
        prompt_depth = kwargs.get("prompt_depth")
        prompt_mask = kwargs.get("prompt_mask")
        prompt_depth = torch.log(prompt_depth+1.)
        
        batch_size = depth.shape[0]
        depth_min_list = []
        depth_max_list = []
        for b in range(batch_size):
            if (prompt_mask[b] > 0).sum() <= 5:
                if "ground_truth" in kwargs:
                    ground_truth = kwargs["ground_truth"]
                    ground_truth = torch.log(ground_truth+1.)
                    ground_truth_mask = kwargs["ground_truth_mask"]
                    depth_min = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], lower_thresh)
                    depth_max = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], upper_thresh)
                else:
                    raise ValueError("ground_truth is required")
            else:
                depth_min = torch.quantile(prompt_depth[b][prompt_mask[b] > 0.0], lower_thresh)
                depth_max = torch.quantile(prompt_depth[b][prompt_mask[b] > 0.0], upper_thresh)
                if (depth_max - depth_min <= 1e-2).any():
                    ground_truth = kwargs["ground_truth"]
                    ground_truth = torch.log(ground_truth+1.)
                    ground_truth_mask = kwargs["ground_truth_mask"]
                    depth_min = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], lower_thresh)
                    depth_max = torch.quantile(ground_truth[b][ground_truth_mask[b] > 0.0], upper_thresh)
            
            depth_min_list.append(depth_min)
            depth_max_list.append(depth_max)
            
        depth_min = rearrange(torch.stack(depth_min_list), "b -> b 1 1 1")
        depth_max = rearrange(torch.stack(depth_max_list), "b -> b 1 1 1")
        
        new_depth = (depth - depth_min) / torch.clamp(depth_max - depth_min, min=1e-3)
        new_mask = (depth >= 0) & (prompt_mask > 0.0)
        new_depth[new_mask == 0] = 0.0

        return new_depth, new_mask, (depth_min, depth_max)

    def unwarp(self, depth, **kwargs):
        depth_min, depth_max = kwargs.get("reference_meta")
        denorm_depth = depth * torch.clamp(depth_max - depth_min, min=1e-3) + depth_min
        return torch.exp(denorm_depth) - 1.


def _compute_scale(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = b_0[valid] / (a_00[valid] + 1e-6)

    return x_0


def _compute_scale_and_shift_gpu(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1
