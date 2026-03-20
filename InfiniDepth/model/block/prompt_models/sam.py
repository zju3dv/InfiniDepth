# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Type
import torch
from .utils.pe_utils import PositionEmbeddingRandom
from .utils.transformer import TwoWayTransformer
from torch import nn

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class SAMPromptModel(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        mlp_dim: int = 2048,
        num_heads: int = 8,
        activation: Type[nn.Module] = nn.GELU,
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
        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer(
            depth=2, embedding_dim=transformer_dim, num_heads=num_heads, mlp_dim=mlp_dim
        )
        self.pe_layer = PositionEmbeddingRandom(transformer_dim // 2)
        self.depth2feature = nn.Sequential(
            nn.Linear(1, transformer_dim // 2), nn.ReLU(True), nn.Linear(transformer_dim // 2, transformer_dim)
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

        # prompt_embeddings_list = []
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
            prompt_embeddings_item, image_embeddings_item = self.transformer(
                image_embeddings[b : (b + 1)],
                image_pe.reshape(1, -1, image_pe.shape[-1]),
                prompt_embeddings,
                prompt_pe,
            )
            image_embeddings_list.append(image_embeddings_item)
        image_embeddings = torch.cat(image_embeddings_list, dim=0)
        return image_embeddings


# # Lightly adapted from
# # https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
# class MLP(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         output_dim: int,
#         num_layers: int,
#         sigmoid_output: bool = False,
#     ) -> None:
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
#         self.sigmoid_output = sigmoid_output

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         if self.sigmoid_output:
#             x = F.sigmoid(x)
#         return x
