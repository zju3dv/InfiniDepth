# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch

def cast_to(o, dtype: torch.dtype):
    if isinstance(o, torch.Tensor):
        return o.to(dtype)
    if isinstance(o, tuple):
        return tuple(cast_to(u, dtype) for u in o)
    if isinstance(o, list):
        return list(cast_to(u, dtype) for u in o)
    if isinstance(o, dict):
        return {k: cast_to(u, dtype) for k, u in o.items()}
    if isinstance(o, (bool, int, float, str, type(None))):
        return o
    raise NotImplementedError(f"Unsupported type: {type(o)}")
