"""
Distributed ops for supporting sequence parallel.
"""

from typing import Any, Tuple
import torch
import torch.distributed as dist
from torch import Tensor

from common.distributed.advanced import (
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
)


def single_all_to_all(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
    async_op: bool = False,
):
    """
    A function to do all-to-all on a tensor
    """
    seq_world_size = dist.get_world_size(group)
    prev_scatter_dim = scatter_dim
    if scatter_dim != 0:
        local_input = local_input.transpose(0, scatter_dim)
        if gather_dim == 0:
            gather_dim = scatter_dim
        scatter_dim = 0

    inp_shape = list(local_input.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // seq_world_size
    input_t = local_input.reshape(
        [seq_world_size, inp_shape[scatter_dim]] + inp_shape[scatter_dim + 1 :]
    ).contiguous()
    output = torch.empty_like(input_t)
    comm = dist.all_to_all_single(output, input_t, group=group, async_op=async_op)
    if async_op:
        # let user's code transpose & reshape
        return output, comm, prev_scatter_dim

    # first dim is seq_world_size, so we can split it directly
    output = torch.cat(output.split(1), dim=gather_dim + 1).squeeze(0)
    if prev_scatter_dim:
        output = output.transpose(0, prev_scatter_dim).contiguous()
    return output


class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        if async_op:
            output, comm, prev_scatter_dim = single_all_to_all(
                local_input, scatter_dim, gather_dim, group, async_op=async_op
            )
            ctx.prev_scatter_dim = prev_scatter_dim
            return output, comm

        return single_all_to_all(local_input, scatter_dim, gather_dim, group, async_op=async_op)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        if ctx.async_op:
            input_t = torch.cat(grad_output[0].split(1), dim=ctx.gather_dim + 1).squeeze(0)
            if ctx.prev_scatter_dim:
                input_t = input_t.transpose(0, ctx.prev_scatter_dim)
        else:
            input_t = grad_output[0]
        return (
            None,
            single_all_to_all(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, async_op=False),
            None,
            None,
            None,
        )


class Slice(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, local_input: Tensor, dim: int) -> Tensor:
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        seq_world_size = dist.get_world_size(group)
        ctx.seq_world_size = seq_world_size
        ctx.dim = dim
        dim_size = local_input.shape[dim]
        return local_input.split(dim_size // seq_world_size, dim=dim)[ctx.rank].contiguous()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor, None]:
        dim_size = list(grad_output.size())
        split_size = dim_size[0]
        dim_size[0] = dim_size[0] * ctx.seq_world_size
        output = torch.empty(dim_size, dtype=grad_output.dtype, device=torch.cuda.current_device())
        dist._all_gather_base(output, grad_output, group=ctx.group)
        return (None, torch.cat(output.split(split_size), dim=ctx.dim), None)


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, local_input: Tensor, dim: int) -> Tensor:
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.dim = dim
        seq_world_size = dist.get_world_size(group)
        ctx.seq_world_size = seq_world_size
        dim_size = list(local_input.size())
        split_size = dim_size[0]
        ctx.part_size = dim_size[dim]
        dim_size[0] = dim_size[0] * seq_world_size
        output = torch.empty(dim_size, dtype=local_input.dtype, device=torch.cuda.current_device())
        dist._all_gather_base(output, local_input.contiguous(), group=ctx.group)
        return torch.cat(output.split(split_size), dim=dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor]:
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.dim)[ctx.rank].contiguous(),
            None,
        )


def gather_seq_scatter_heads_qkv(qkv_tensor: Tensor, seq_dim: int, restore_shape: bool = True):
    """
    A func to sync splited qkv tensor
    qkv_tensor: the tensor we want to do alltoall with. The last dim must
        be the projection_idx, which we will split into 3 part. After
        spliting, the gather idx will be projecttion_idx + 1
    seq_dim: gather_dim for all2all comm
    restore_shape: if True, output will has the same shape length as input
    """
    group = get_sequence_parallel_group()
    if not group:
        return qkv_tensor
    world = get_sequence_parallel_world_size()
    orig_shape = qkv_tensor.shape
    scatter_dim = qkv_tensor.dim()
    bef_all2all_shape = list(orig_shape)
    qkv_proj_dim = bef_all2all_shape[-1]
    bef_all2all_shape = bef_all2all_shape[:-1] + [3, qkv_proj_dim // 3]
    qkv_tensor = qkv_tensor.view(bef_all2all_shape)
    qkv_tensor = SeqAllToAll.apply(group, qkv_tensor, scatter_dim, seq_dim, False)
    if restore_shape:
        out_shape = list(orig_shape)
        out_shape[seq_dim] *= world
        out_shape[-1] = qkv_proj_dim // world
        qkv_tensor = qkv_tensor.view(out_shape)
    return qkv_tensor


def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int) -> Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel
    """
    group = get_sequence_parallel_group()
    if not group:
        return x
    return SeqAllToAll.apply(group, x, seq_dim, head_dim, False)


def gather_seq_scatter_heads(x: Tensor, seq_dim: int, head_dim: int) -> Tensor:
    """
    A func to sync embedding input with alltoall in sequence parallel
    """
    group = get_sequence_parallel_group()
    if not group:
        return x
    return SeqAllToAll.apply(group, x, head_dim, seq_dim, False)


def scatter_heads(x: Tensor, dim: int) -> Tensor:
    """
    A func to split heads before attention in sequence parallel
    """
    group = get_sequence_parallel_group()
    if not group:
        return x
    return Slice.apply(group, x, dim)


def gather_heads(x: Tensor, dim: int) -> Tensor:
    """
    A func to gather heads for the attention result in sequence parallel
    """
    group = get_sequence_parallel_group()
    if not group:
        return x
    return Gather.apply(group, x, dim)


def slice_input_tensor(x: Tensor, dim: int, rank: int, world: int):
    chunk_size = x.size(dim) // world
    return x.split(chunk_size, dim=dim)[rank]
