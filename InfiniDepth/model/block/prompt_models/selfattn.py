from typing import Tuple
import torch
import torch.nn as nn
from ..perceive_io import LayerScale, MemEffAttention, Mlp
from .rope import RotaryPositionEmbedding2D
from .utils.pe_utils import PositionEmbeddingRandom
from torch import Tensor

acc_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else (torch.float16 if torch.cuda.is_available() else torch.float32)
)


class SelfAttnPromptModel(nn.Module):
    def __init__(
        self,
        transformer_dim: int = 1024,
        num_blocks: int = 1,
        num_heads: int = 4,
        pe: str = "normal",
        image_pe_method: str = "patch",  # image
        **kwargs,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks 
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.pe = pe
        pe_dim = transformer_dim // 2
        if self.pe == "apply":
            pe_dim = pe_dim // num_heads
        self.pe_layer = PositionEmbeddingRandom(pe_dim, image_pe_method=image_pe_method)
        self.prompt_blocks = nn.ModuleList(
            [
                SelfAttenPromptBlock(dim=transformer_dim, num_heads=num_heads, first_block=(i == 0), pe=pe)
                for i in range(num_blocks)
            ]
        )
        self.depth2feature = nn.Sequential(
            nn.Linear(1, transformer_dim // 2),
            nn.GELU(),
            nn.Linear(transformer_dim // 2, transformer_dim),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        prompt_depth: torch.Tensor,
        prompt_mask: torch.Tensor,
        patch_h: int,
        patch_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.
        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.
        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        B, _, H, W = prompt_depth.shape
        image_pe = self.pe_layer((patch_h, patch_w)).permute(1, 2, 0)  # CxHxW -> HxWxC
        prompt_embeddings_list = []
        image_embeddings_list = []
        for b in range(B):
            valid_pts_num = (prompt_mask[b, 0] > 0.0).sum()
            if valid_pts_num == 0:
                image_embeddings_item = image_embeddings[b : (b + 1)]
                image_embeddings_list.append(image_embeddings_item)
                continue
            sparse_depth_pos = (prompt_mask[b, 0] > 0.0).nonzero().float()
            sparse_depth_pos[:, 0] = (sparse_depth_pos[:, 0] + 0.5) / H
            sparse_depth_pos[:, 1] = (sparse_depth_pos[:, 1] + 0.5) / W
            sparse_depth = prompt_depth[b, 0][prompt_mask[b, 0] > 0.0]
            prompt_embeddings = self.depth2feature(sparse_depth[:, None])[None, ...]   # 1, N, C
            prompt_pe = self.pe_layer._pe_encoding(sparse_depth_pos[None, :, [1, 0]])  # 1, N, C
            query_pe = image_pe.reshape(1, -1, image_pe.shape[-1])
            prompt = prompt_embeddings  # + prompt_pe
            query = image_embeddings[b : (b + 1)]  # + query_pe
            with torch.autocast("cuda", enabled=True, dtype=acc_dtype):
                for block in self.prompt_blocks:
                    query, prompt = block(query, query_pe, prompt, prompt_pe)
            image_embeddings_list.append(query[..., : image_embeddings.shape[-1]])
            prompt_embeddings_list.append(prompt)
        image_embeddings = torch.cat(image_embeddings_list, dim=0)
        return image_embeddings


class SelfAttnRopePromptModel(nn.Module):
    def __init__(
        self,
        transformer_dim: int = 1024,
        num_blocks: int = 1,
        num_heads: int = 4,
        pe: str = "normal",
        image_pe_method: str = "patch",  # image
        **kwargs,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.pe = pe
        pe_dim = transformer_dim // 2
        if self.pe == "apply":
            pe_dim = pe_dim // num_heads
        if self.pe.startswith("rope"):
            self.pe_layer = RotaryPositionEmbedding2D(
                frequency=float(self.pe.split("rope")[1]), feat_dim=pe_dim // num_heads
            )
        else:
            self.pe_layer = PositionEmbeddingRandom(pe_dim, image_pe_method=image_pe_method)
        self.prompt_blocks = nn.ModuleList(
            [
                SelfAttenPromptBlock(
                    dim=transformer_dim, num_heads=num_heads, first_block=(i == 0), pe=pe, use_sep=False
                )
                for i in range(num_blocks)
            ]
        )
        self.depth2feature = nn.Sequential(
            nn.Linear(1, transformer_dim // 2),
            nn.GELU(),
            nn.Linear(transformer_dim // 2, transformer_dim),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        prompt_depth: torch.Tensor,
        prompt_mask: torch.Tensor,
        patch_h: int,
        patch_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.
        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.
        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        B, _, H, W = prompt_depth.shape
        image_pe = self.pe_layer((patch_h, patch_w), device=prompt_depth.device).permute(1, 2, 0)  # CxHxW -> HxWxC
        prompt_embeddings_list = []
        image_embeddings_list = []
        for b in range(B):
            valid_pts_num = (prompt_mask[b, 0] > 0.0).sum()
            if valid_pts_num == 0:
                image_embeddings_item = image_embeddings[b : (b + 1)]
                image_embeddings_list.append(image_embeddings_item)
                continue
            sparse_depth_pos = (prompt_mask[b, 0] > 0.0).nonzero().int()
            sparse_depth_pos[:, 0] = sparse_depth_pos[:, 0] * 2
            sparse_depth_pos[:, 1] = sparse_depth_pos[:, 1] * 2
            sparse_depth = prompt_depth[b, 0][prompt_mask[b, 0] > 0.0]
            prompt_embeddings = self.depth2feature(sparse_depth[:, None])[None, ...]  # 1, N, C
            prompt_pe = self.pe_layer._pe_encoding(sparse_depth_pos[None])  # 1, N, C
            query_pe = image_pe.reshape(1, -1, image_pe.shape[-1])
            prompt = prompt_embeddings  # + prompt_pe
            query = image_embeddings[b : (b + 1)]  # + query_pe
            with torch.autocast("cuda", enabled=True, dtype=acc_dtype):
                for block in self.prompt_blocks:
                    query, prompt = block(query, query_pe, prompt, prompt_pe)
            image_embeddings_list.append(query[..., : image_embeddings.shape[-1]])
            prompt_embeddings_list.append(prompt)
        image_embeddings = torch.cat(image_embeddings_list, dim=0)
        return image_embeddings
    

class SelfAttenPromptBlock(nn.Module):
    """
    Self-attention block for prompt-based processing that handles both query and context features.
    Supports different positional encoding strategies.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        init_values: float = 0.0,
        first_block: bool = False,
        pe: str = "normal",
        use_sep: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.first_block = first_block
        self.pe = pe

        # Attention components
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MemEffAttention(dim, num_heads=num_heads, pe=pe)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        # MLP components
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=dim * 4)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        # Separator token for concatenating query and context
        pe_dim = dim
        self.use_sep = use_sep
        if use_sep:
            self.sep = nn.Parameter(torch.randn(1, 1, pe_dim))
        else:
            self.sep = None

        # Special separator for positional encoding if needed
        if self.use_sep:
            if self.pe != "normal":
                self.sep_pe = nn.Parameter(torch.randn(1, 1, pe_dim))
            else:
                self.sep_pe = None

    def forward(self, x: Tensor, x_pe: Tensor, context: Tensor, context_pe: Tensor) -> Tuple[Tensor, Tensor]:
        # Apply positional encoding if this is the first block and using normal PE
        if self.pe == "normal" and self.first_block:
            x = x + x_pe
            context = context + context_pe

        # Record original sequence lengths
        x_len, context_len = x.shape[1], context.shape[1]

        # Concatenate query, separator token, and context
        if self.use_sep:
            x = torch.cat([x, self.sep, context], dim=1)
        else:
            x = torch.cat([x, context], dim=1)

        # Handle positional encoding concatenation if needed
        if self.pe != "normal":
            if self.use_sep:
                x_pe = torch.cat([x_pe, self.sep_pe, context_pe], dim=1)
            else:
                x_pe = torch.cat([x_pe, context_pe], dim=1)
            x = x + self.ls1(self.attn(self.norm1(x), x_pe))
        else:
            # Apply standard attention
            x = x + self.ls1(self.attn(self.norm1(x)))

        # Apply MLP
        x = x + self.ls2(self.mlp(self.norm2(x)))

        # Split back into query and context
        query = x[:, :x_len, :]
        if self.use_sep:
            context = x[:, x_len + 1 : x_len + 1 + context_len, :]
        else:
            context = x[:, x_len : x_len + context_len, :]

        return query, context
