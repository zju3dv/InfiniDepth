import torch
from einops import rearrange
EPS = 1e-6
class WarpMedian:
    def __init__(self, **kwargs):
        pass
    def warp(self, depth, **kwargs):
        if kwargs.get("reference_meta", None) is not None:
            median_val = kwargs["reference_meta"]
            return depth / torch.clamp(median_val, min=1e-3), (depth > EPS), median_val
        prompt_depth = kwargs.get("prompt_depth")
        prompt_mask = kwargs.get("prompt_mask")
        median_val = []
        batch_size = depth.shape[0]
        for b in range(batch_size):
            median = torch.quantile(prompt_depth[b][prompt_mask[b] > 0.0], 0.5)
            median_val.append(median)
        median_val = torch.stack(median_val, dim=0)
        median_val = rearrange(median_val, "b -> b 1 1 1")
        return depth / torch.clamp(median_val, min=1e-2), (depth > EPS) & (prompt_mask > 0.0), median_val
    def unwarp(self, depth, **kwargs):
        median_val = kwargs.get("reference_meta")
        return depth * torch.clamp(median_val, min=1e-3)