import torch
import torch.nn as nn
from typing import Tuple
from ..perceive_io import LayerScale, MemEffCrossAttention, Mlp
from .utils.pe_utils import PositionEmbeddingRandom
from torch import Tensor


class CrossAttnPromptModel(nn.Module):
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
                CrossAttenPromptBlock(dim=transformer_dim, num_heads=num_heads, first_block=(i == 0), pe=pe)
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
            prompt_embeddings = self.depth2feature(sparse_depth[:, None])[None, ...]  # 1, N, C
            prompt_pe = self.pe_layer._pe_encoding(sparse_depth_pos[None, :, [1, 0]])  # 1, N, C
            query_pe = image_pe.reshape(1, -1, image_pe.shape[-1])
            prompt = prompt_embeddings  # + prompt_pe
            query = image_embeddings[b : (b + 1)]  # + query_pe
            for block in self.prompt_blocks:
                query, prompt = block(query, query_pe, prompt, prompt_pe)
            image_embeddings_list.append(query[..., : image_embeddings.shape[-1]])
            prompt_embeddings_list.append(prompt)
        image_embeddings = torch.cat(image_embeddings_list, dim=0)
        return image_embeddings


class CrossAttenPromptBlock(nn.Module):
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
        **kwargs,
    ) -> None:
        super().__init__()
        self.first_block = first_block
        self.pe = pe

        # Attention components
        self.norm1_x = nn.LayerNorm(dim)
        self.norm1_x_after = nn.LayerNorm(dim)
        self.attn_x = MemEffCrossAttention(dim, context_dim=dim, num_heads=num_heads, pe=pe)
        self.ls1_x = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm1_context = nn.LayerNorm(dim)
        self.attn_context = MemEffCrossAttention(dim, context_dim=dim, num_heads=num_heads, pe=pe)
        self.ls1_context = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        # MLP components
        self.norm2_x = nn.LayerNorm(dim)
        self.mlp_x = Mlp(dim, hidden_features=dim * 4)
        self.ls2_x = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2_context = nn.LayerNorm(dim)
        self.mlp_context = Mlp(dim, hidden_features=dim * 4)
        self.ls2_context = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor, x_pe: Tensor, context: Tensor, context_pe: Tensor) -> Tuple[Tensor, Tensor]:
        # Apply positional encoding if this is the first block and using normal PE
        if self.pe == "normal" and self.first_block:
            x = x + x_pe
            context = context + context_pe

        # Handle positional encoding concatenation if needed
        if self.pe != "normal":
            x = x + self.ls1_x(
                self.attn_x(self.norm1_x(x), context=self.norm1_context(context), x_pe=x_pe, context_pe=context_pe)
            )
            context = context + self.ls1_context(
                self.attn_context(
                    self.norm1_context(context), context=self.norm1_x_after(x), x_pe=context_pe, context_pe=x_pe
                )
            )
        else:
            # Apply standard attention
            x = x + self.ls1_x(self.attn_x(self.norm1_x(x), context=self.norm1_context(context)))
            context = context + self.ls1_context(
                self.attn_context(self.norm1_context(context), context=self.norm1_x_after(x))
            )

        # Apply MLP
        x = x + self.ls2_x(self.mlp_x(self.norm2_x(x)))
        context = context + self.ls2_context(self.mlp_context(self.norm2_context(context)))

        return x, context
