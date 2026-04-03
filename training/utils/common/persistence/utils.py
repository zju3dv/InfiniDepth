import os
import uuid
from typing import Any, Optional
import torch

from common.fs import is_hdfs_path, mkdir

_local_dir = None


def get_local_dir():
    """
    Get a local directory for temporary storage for this process.
    """
    global _local_dir
    if _local_dir is None:
        _local_dir = os.path.join("persistence", str(uuid.uuid4()))
        mkdir(_local_dir)
    return _local_dir


def get_local_path(path: str) -> str:
    """
    Get a local path for storing the file.
    If the path is already a local path, directly return.
    """
    if is_hdfs_path(path):
        path = os.path.join(get_local_dir(), os.path.basename(path))
    else:
        mkdir(os.path.dirname(path))
    return path


def convert_dtype(states: Any, dtype: Optional[torch.dtype] = None):
    """
    Recursively convert the state_dict to device and dtype.
    """
    if dtype is None:
        return states
    if torch.is_tensor(states):
        return states.to("cpu", dtype)
    if isinstance(states, dict):
        return {k: convert_dtype(v, dtype) for k, v in states.items()}
    if isinstance(states, list):
        return [convert_dtype(v, dtype) for v in states]
    return states
