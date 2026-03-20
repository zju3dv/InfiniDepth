# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
from dinov3.eval.dense.depth.utils import cast_to

from .dpt_head import DPTHead
from .encoder import BackboneLayersSet, DinoVisionTransformerWrapper, PatchSizeAdaptationStrategy


class FeaturesToDepth(torch.nn.Module):
    def __init__(
        self,
        min_depth=0.001,
        max_depth=80,
        bins_strategy="linear",
        norm_strategy="linear",
    ):
        """
        Module which converts a feature maps into a depth map

        Args:
        min_depth (float): minimum depth, used to calibrate the depth range
        max_depth (float): maximum depth, used to calibrate the depth range
        bins_strategy (str): Choices are 'linear' or 'log', for Uniform or Scale Invariant distributions for depth bins.
                             See AdaBins [1] for more details.
        norm_strategy (str): Choices are 'linear', 'softmax' or 'sigmoid', for the conversion of features to depth logits
        scale_up (bool): If true, and only if regression by classification is not used, the result is multiplied by max_depth


        Example:
        x = depth_model(input_image)  # N C H W
        - If pure regression (C == 1), depth is obtained by scaling and/or shifting x
        - If C > 1, bins are used:
            Depth is obtained as a weighted sum of depth bins, where weights are predicted logits. (see AdaBins [1] for more details)

        [1] AdaBins: https://github.com/shariqfarooq123/AdaBins
        """
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        assert bins_strategy in ["linear", "log"], "Support bins_strategy: linear, log"
        assert norm_strategy in ["linear", "softmax", "sigmoid"], "Support norm_strategy: linear, softmax, sigmoid"

        self.bins_strategy = bins_strategy
        self.norm_strategy = norm_strategy

    def forward(self, x):
        n_bins = x.shape[1]  # N n_bins H W
        if n_bins > 1:
            if self.bins_strategy == "linear":
                bins = torch.linspace(self.min_depth, self.max_depth, n_bins, device=x.device)
            elif self.bins_strategy == "log":
                bins = torch.linspace(
                    torch.log(torch.tensor(self.min_depth)),
                    torch.log(torch.tensor(self.max_depth)),
                    n_bins,
                    device=x.device,
                )
                bins = torch.exp(bins)

            # following Adabins, default linear
            if self.norm_strategy == "linear":
                logit = torch.relu(x)
                eps = 0.1
                logit = logit + eps
                logit = logit / logit.sum(dim=1, keepdim=True)
            elif self.norm_strategy == "softmax":
                logit = torch.softmax(x, dim=1)
            elif self.norm_strategy == "sigmoid":
                logit = torch.sigmoid(x)
                logit = logit / logit.sum(dim=1, keepdim=True)

            output = torch.einsum("ikmn,k->imn", [logit, bins]).unsqueeze(dim=1)
        else:
            # standard regression
            output = torch.relu(x) + self.min_depth
        return output


def make_head(
    embed_dims: int | list[int],
    n_output_channels: int,
    use_batchnorm: bool = False,
    use_cls_token: bool = False,
    # upsample: int = 4,
    head_type: str = "linear",
    **kwargs,
) -> torch.nn.Module:

    if isinstance(embed_dims, int):
        embed_dims = [embed_dims]

    if head_type == "linear":
        raise NotImplementedError
    elif head_type == "dpt":
        decoder = DPTHead(
            in_channels=embed_dims,
            n_output_channels=n_output_channels,
            readout_type="project" if use_cls_token else "ignore",
            use_batchnorm=use_batchnorm,
            **kwargs,  # TODO add here post_process_channels, n_hidden_channels
        )
    else:
        raise NotImplementedError("only linear and DPT head supported")
    return decoder


class EncoderDecoder(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        encoder_dtype=torch.float,
        decoder_dtype=torch.float,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_dtype = encoder_dtype
        self.decoder_dtype = decoder_dtype
        self.is_cuda = torch.cuda.is_available()

    def forward(self, x):
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", enabled=False):
            x = x.to(self.encoder_dtype)
            x = self.encoder(x)
            x = cast_to(x, self.decoder_dtype)
        return self.decoder(x)


def build_depther(
    backbone: torch.nn.Module,
    backbone_out_layers: list[int] | BackboneLayersSet,
    n_output_channels: int,
    use_backbone_norm: bool = False,
    use_batchnorm: bool = False,
    use_cls_token: bool = False,
    adapt_to_patch_size: PatchSizeAdaptationStrategy = PatchSizeAdaptationStrategy.CENTER_PADDING,
    head_type: str = "dpt",
    encoder_dtype: torch.dtype = torch.float,
    decoder_dtype: torch.dtype = torch.float,
    # depth args
    min_depth: float = 0.001,
    max_depth: float = 10.0,
    bins_strategy: str = "linear",
    norm_strategy: str = "linear",
    **kwargs,
):
    encoder = DinoVisionTransformerWrapper(
        backbone_model=backbone,
        backbone_out_layers=backbone_out_layers,
        use_backbone_norm=use_backbone_norm,
        adapt_to_patch_size=adapt_to_patch_size,
    )
    encoder = encoder.to(encoder_dtype)
    encoder.eval()

    decoder = make_head(
        encoder.embed_dims,
        n_output_channels=n_output_channels,
        use_batchnorm=use_batchnorm,
        use_cls_token=use_cls_token,
        head_type=head_type,
        **kwargs,
    )
    decoder.eval()

    features_to_depth = FeaturesToDepth(
        min_depth=min_depth,
        max_depth=max_depth,
        bins_strategy=bins_strategy,
        norm_strategy=norm_strategy,
    )

    return torch.nn.Sequential(
        EncoderDecoder(
            encoder,
            decoder,
            encoder_dtype=encoder_dtype,
            decoder_dtype=decoder_dtype,
        ),
        features_to_depth,
    )
