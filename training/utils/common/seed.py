import contextlib
import os
import random
from typing import Optional
import numpy as np
import torch

from common.distributed import get_global_rank


def set_seed(seed: Optional[int], same_across_ranks: bool = False):
    r"""Function that sets the seed for pseudo-random number generators."""
    if seed is not None:
        seed += get_global_rank() if not same_across_ranks else 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def enable_full_determinism():
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    """
    # Enable PyTorch deterministic mode. This potentially requires either the environment
    # variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
    # depending on the CUDA version, so we set them both here
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

@contextlib.contextmanager
def local_seed(seed: Optional[int]):
    """
    Create a local context with seed is set, but exit back to the original random state.
    If seed is None, do nothing.
    """
    if seed is not None:
        random_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(random_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
    else:
        yield