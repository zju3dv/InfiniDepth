from typing import Callable, Optional, Union
import torch
from einops import rearrange
from ...utils.logger import Log
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from xformers.ops import memory_efficient_attention, unbind

    XFORMERS_AVAILABLE = True
except ImportError:
    Log.warn("xFormers not available")
    XFORMERS_AVAILABLE = False


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pe: str = "normal",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv_context = nn.Linear(context_dim, context_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pe = pe
        if self.pe == "qk":
            self.norm1 = nn.LayerNorm(dim // num_heads)
            self.norm2 = nn.LayerNorm(dim // num_heads)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        # x is the query tensor, context is the key/value tensor
        B, N, C = x.shape
        _, M, _ = context.shape

        qkv_x = self.qkv(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_x = qkv_x[0] * self.scale

        qkv_context = (
            self.qkv_context(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        k_context, v_context = qkv_context[0], qkv_context[0]

        # Cross-attention: query from x and key/value from context
        attn = q_x @ k_context.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v_context).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffCrossAttention(CrossAttention):
    def forward(
        self, x: Tensor, context: Tensor, x_pe: Tensor = None, context_pe: Tensor = None, attn_bias=None
    ) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x, context)

        B, N, C = x.shape
        _, M, C_context = context.shape

        qkv_x = self.qkv(x).reshape(B, N, 1, self.num_heads, C // self.num_heads)
        (q_x,) = unbind(qkv_x, 2)

        qkv_context = self.qkv_context(context).reshape(B, M, 2, self.num_heads, C_context // self.num_heads)
        k_context, v_context = unbind(qkv_context, 2)

        if self.pe == "qk":
            q_x = self.norm1(q_x + rearrange(x_pe, "b n (m c) -> b n m c", m=self.num_heads))
            k_context = self.norm2(k_context + rearrange(context_pe, "b n (m c) -> b n m c", m=self.num_heads))
        elif self.pe == "apply":
            pass
        # Memory-efficient cross-attention
        x = memory_efficient_attention(
            q_x.to(dtype=v_context.dtype), k_context.to(dtype=v_context.dtype), v_context, attn_bias=attn_bias
        )
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pe: str = "qk",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pe = pe
        if self.pe == "qk":
            self.norm1 = nn.LayerNorm(dim // num_heads)
            self.norm2 = nn.LayerNorm(dim // num_heads)

    def forward(self, x: Tensor, x_pe: Tensor = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        if self.pe == "qk":
            if x_pe is None:
                raise ValueError("x_pe must be provided when pe='qk'")
            x_pe = rearrange(x_pe, "b n (m c) -> b n m c", m=self.num_heads)
            q = self.norm1(q + x_pe)
            k = self.norm2(k + x_pe)
        elif self.pe == "apply":
            pass

        # Keep behavior aligned with MemEffAttention when norm changes q/k dtype.
        q = q.to(dtype=v.dtype)
        k = k.to(dtype=v.dtype)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use SDPA and run attention kernel in float32 under mixed precision for better numerical stability.
        if q.dtype in (torch.float16, torch.bfloat16):
            q_attn = q.float()
            k_attn = k.float()
            v_attn = v.float()
        else:
            q_attn = q
            k_attn = k
            v_attn = v
        x = F.scaled_dot_product_attention(
            q_attn,
            k_attn,
            v_attn,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.to(dtype=v.dtype).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(
        self,
        x: Tensor,
        x_pe: Tensor = None,
        attn_bias=None,
    ) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x, x_pe)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)
        if self.pe == "qk":
            q = self.norm1(q + rearrange(x_pe, "b n (m c) -> b n m c", m=self.num_heads))
            k = self.norm2(k + rearrange(x_pe, "b n (m c) -> b n m c", m=self.num_heads))
        elif self.pe == "apply":
            pass
        # this is important
        # as q, k after norm1/norm2 have different dtype
        # which will cause error in memory_efficient_attention
        x = memory_efficient_attention(q.to(dtype=v.dtype), k.to(dtype=v.dtype), v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
