"""
Utility functions.
"""

import torch


def expand_dims(tensor: torch.Tensor, ndim: int):
    """
    Expand tensor to target ndim. New dims are added to the right.
    For example, if the tensor shape was (8,), target ndim is 4, return (8, 1, 1, 1).
    """
    shape = tensor.shape + (1,) * (ndim - tensor.ndim)
    return tensor.reshape(shape)


def assert_schedule_timesteps_compatible(schedule, timesteps):
    """
    Check if schedule and timesteps are compatible.
    """
    if schedule.T != timesteps.T:
        raise ValueError("Schedule and timesteps must have the same T.")
    if schedule.is_continuous() != timesteps.is_continuous():
        raise ValueError("Schedule and timesteps must have the same continuity.")


def classifier_free_guidance(
    pos: torch.Tensor,
    neg: torch.Tensor,
    scale: float,
    rescale: float = 0.0,
):
    """
    Apply classifier-free guidance.
    """
    # Classifier-free guidance (https://arxiv.org/abs/2207.12598)
    cfg = neg + scale * (pos - neg)

    # Classifier-free guidance rescale (https://arxiv.org/pdf/2305.08891.pdf)
    if rescale != 0.0:
        pos_std = pos.std(dim=list(range(1, pos.ndim)), keepdim=True)
        cfg_std = cfg.std(dim=list(range(1, cfg.ndim)), keepdim=True)
        factor = pos_std / cfg_std
        factor = rescale * factor + (1 - rescale)
        cfg *= factor

    return cfg
