# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
from typing import Any, Callable, Dict, List, Tuple
import hydra
import torch
from torch import Tensor, nn

from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp

logger = logging.getLogger("dinov2")


try:
    from xformers.ops import fmha, index_select_cat, scaled_index_add

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        **kwargs,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list, *args, **kwargs):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            assert XFORMERS_AVAILABLE, "Please install xFormers for nested tensors usage"
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError


class PromptNestedTensorBlock(NestedTensorBlock):
    def __init__(self, prompt_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_config = prompt_config
        self.prompt_model = hydra.utils.instantiate(prompt_config)

    def forward(self, x_or_x_list, prompt_depth, prompt_mask, patch_h, patch_w):
        if isinstance(x_or_x_list, Tensor):
            x_or_x_list[:, 1:] = self.prompt_model(x_or_x_list[:, 1:], prompt_depth, prompt_mask, patch_h, patch_w)
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            raise NotImplementedError
            assert XFORMERS_AVAILABLE, "Please install xFormers for nested tensors usage"
            x_list = [self.prompt_model(x, prompt_depth, prompt_mask, patch_h, patch_w) for x in x_or_x_list]
            return self.forward_nested(x_list)
        else:
            raise AssertionError


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class ZeroBlock(nn.Module):
    def __init__(self, dim, use_linear=False):
        super().__init__()
        self.dim = dim
        self.use_linear = use_linear
        if use_linear:
            self.proj = zero_module(nn.Linear(dim, dim))
        else:
            self.proj = zero_module(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0))

    def forward(self, x, patch_h, patch_w):
        if self.use_linear:
            return self.proj(x)
        else:
            cls, x = x[:, :1], x[:, 1:]  # [B, 1, dim] and [B, patch_h * patch_w, dim]
            cls = cls[:, 0, :, None, None]  # [B, dim, 1, 1]
            cls = self.proj(cls)  # [B, dim, 1, 1]
            cls = cls.flatten(2).transpose(1, 2)  # [B, 1, dim]

            x = x.permute(0, 2, 1).reshape(-1, self.dim, patch_h, patch_w)  # [B, dim, patch_h, patch_w]
            x = self.proj(x)  # [B, dim, patch_h, patch_w]
            x = x.flatten(2).transpose(1, 2)  # [B, patch_h * patch_w, dim]
            return torch.cat((cls, x), 1)  # [B, patch_h * patch_w + 1, dim]


class ControlNetNestedTensorBlock(NestedTensorBlock):
    def __init__(self, dim, prompt_config, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)
        # freeze the parameters of the current block
        for p in self.parameters():
            p.requires_grad = False
        self.copy_block = NestedTensorBlock(dim, *args, **kwargs)
        self.config = prompt_config
        self.zero_in = ZeroBlock(dim, use_linear=self.config.get("use_linear", False))
        self.zero_out = ZeroBlock(dim, use_linear=self.config.get("use_linear", False))
        self.prompt_model = hydra.utils.instantiate(prompt_config)

    def forward(self, x_or_x_list, prompt_depth, prompt_mask, patch_h, patch_w):
        if isinstance(x_or_x_list, Tensor):
            if self.config.get("detach_input", False):
                prompt_x = x_or_x_list.detach()
            else:
                prompt_x = x_or_x_list
            orig_prompt_x = prompt_x.clone()
            prompt_x[:, 1:] = self.prompt_model(
                prompt_x[:, 1:], prompt_depth, prompt_mask, patch_h, patch_w
            )  # b, (patch_h * patch_w + 1), dim
            prompt_x = self.zero_in(prompt_x, patch_h, patch_w)
            prompt_x = self.copy_block(prompt_x + orig_prompt_x)
            prompt_x = self.zero_out(prompt_x, patch_h, patch_w)

            x = super().forward(x_or_x_list)
            return x + prompt_x
        elif isinstance(x_or_x_list, list):
            raise NotImplementedError
            assert XFORMERS_AVAILABLE, "Please install xFormers for nested tensors usage"
            prompt_x_list = [self.prompt_model(x, prompt_depth, prompt_mask, patch_h, patch_w) for x in x_or_x_list]
            prompt_x_list = [self.zero_in(x) for x in prompt_x_list]
            x_copy_list = [self.copy_block(x) for x in prompt_x_list]
            x_copy_list = [self.zero_out(x) for x in x_copy_list]
            x_list = [x + x_copy for x, x_copy in zip(super().forward(x_copy_list), x_copy_list)]
            return x_list
        else:
            raise AssertionError
