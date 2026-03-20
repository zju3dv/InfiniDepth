# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from functools import partial

import torch

from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter
from dinov3.eval.segmentation.models.heads.mask2former_head import Mask2FormerHead


BACKBONE_INTERMEDIATE_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vit7b16": [9, 19, 29, 39],
}


class FeatureDecoder(torch.nn.Module):
    def __init__(self, segmentation_model: torch.nn.ModuleList, autocast_ctx):
        super().__init__()
        self.segmentation_model = segmentation_model
        self.autocast_ctx = autocast_ctx

    def forward(self, inputs):
        with self.autocast_ctx():
            for module in self.segmentation_model:
                inputs = module.forward(inputs)
        return inputs

    def predict(self, inputs, rescale_to=(512, 512)):
        with torch.inference_mode():
            with self.autocast_ctx():
                out = self.segmentation_model[0](inputs)  # backbone forward
                out = self.segmentation_model[1].predict(out, rescale_to=rescale_to)  # decoder head prediction
        return out


def build_segmentation_decoder(
    backbone_model,
    backbone_name,
    decoder_type,
    hidden_dim=2048,
    num_classes=150,
    autocast_dtype=torch.bfloat16,
):
    autocast_ctx = partial(torch.autocast, device_type="cuda", enabled=True, dtype=autocast_dtype)
    if decoder_type == "m2f":
        backbone_model = DINOv3_Adapter(
            backbone_model,
            interaction_indexes=BACKBONE_INTERMEDIATE_LAYERS[backbone_name],
        )
        backbone_model.eval()
        embed_dim = backbone_model.backbone.embed_dim
        patch_size = backbone_model.patch_size
        decoder = Mask2FormerHead(
            input_shape={
                "1": [embed_dim, patch_size * 4, patch_size * 4, 4],
                "2": [embed_dim, patch_size * 2, patch_size * 2, 4],
                "3": [embed_dim, patch_size, patch_size, 4],
                "4": [embed_dim, int(patch_size / 2), int(patch_size / 2), 4],
            },
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            ignore_value=255,
        )
    else:
        raise ValueError(f'Unsupported decoder "{decoder_type}"')

    segmentation_model = FeatureDecoder(
        torch.nn.ModuleList(
            [
                backbone_model,
                decoder,
            ]
        ),
        autocast_ctx=autocast_ctx,
    )
    return segmentation_model
