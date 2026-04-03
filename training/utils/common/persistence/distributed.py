"""
Utility functions for saving distributed states.
"""

from torch._dynamo.eval_frame import OptimizedModule
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel,
    StateDictType,
)
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from common.distributed import get_device, get_global_rank


def get_model_states(model: Module):
    """
    Get model state dict.
    Call by all ranks. Only use the result on rank 0.
    """
    if isinstance(model, OptimizedModule):
        model = model._orig_mod
    if isinstance(model, DistributedDataParallel):
        model = model.module
    if isinstance(model, FullyShardedDataParallel):
        configure_fsdp_states(model)
    return model.state_dict()


def get_optimizer_states(optimizer: Optimizer):
    """
    Get optimizer state dict.
    Call by all ranks. Only use the result on rank 0.
    """
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        optimizer.consolidate_state_dict(to=0)
    return optimizer.state_dict() if get_global_rank() == 0 else None


def get_fsdp_optimizer_states(optimizer: Optimizer, model: FullyShardedDataParallel):
    """
    Get fsdp optimizer state dict.
    Call by all ranks. Only use the result on rank 0.
    """
    configure_fsdp_states(model)
    states = optimizer.state_dict()
    states = FullyShardedDataParallel.optim_state_dict(
        model=model,
        optim=optimizer,
        optim_state_dict=states,
    )
    return states


def configure_fsdp_states(model: FullyShardedDataParallel):
    """
    Configure fsdp state dict type.
    """
    FullyShardedDataParallel.set_state_dict_type(
        module=model,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    )


def fix_zero_optimizer_adamw_fused_states(optimizer: ZeroRedundancyOptimizer):
    """
    Fix loading issue for ZeroRedundancyOptimizer when used with AdamW.
    Issue: https://github.com/pytorch/pytorch/issues/124133
    For regular AdamW, ZeroRedundancyOptimizer keeps "step" on CPU, but this shouldn't
    be done for fused AdamW. So here we fix it.
    """
    for state in optimizer.optim.state.values():
        for name, value in state.items():
            state[name] = value.to(get_device())
