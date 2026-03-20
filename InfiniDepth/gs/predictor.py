from dataclasses import dataclass

import torch
from torch import nn

from .adapter import GaussianAdapter, GaussianAdapterCfg
from .projection import sample_image_grid
from .types import Gaussians


@dataclass
class GSPredictorCfg:
    rgb_feature_dim: int = 64
    depth_feature_dim: int = 32
    dino_reduced_dim: int = 128
    gaussian_regressor_channels: int = 64
    num_surfaces: int = 1
    gaussian_scale_min: float = 1e-10
    gaussian_scale_max: float = 5.0
    sh_degree: int = 2


class GSPixelAlignPredictor(nn.Module):
    def __init__(self, dino_feature_dim: int = 1024, cfg: GSPredictorCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GSPredictorCfg()
        cfg = self.cfg

        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, cfg.rgb_feature_dim, 3, 1, 1),
            nn.GELU(),
        )
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, cfg.depth_feature_dim, 3, 1, 1),
            nn.GELU(),
        )
        self.dino_projector = nn.Sequential(
            nn.Conv2d(dino_feature_dim, 256, 1),
            nn.GELU(),
            nn.Conv2d(256, cfg.dino_reduced_dim, 1),
        )

        reg_in = cfg.rgb_feature_dim + cfg.depth_feature_dim + cfg.dino_reduced_dim
        self.gaussian_regressor = nn.Sequential(
            nn.Conv2d(reg_in, cfg.gaussian_regressor_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(cfg.gaussian_regressor_channels, cfg.gaussian_regressor_channels, 3, 1, 1),
        )

        self.gaussian_adapter = GaussianAdapter(
            GaussianAdapterCfg(
                gaussian_scale_min=cfg.gaussian_scale_min,
                gaussian_scale_max=cfg.gaussian_scale_max,
                sh_degree=cfg.sh_degree,
            )
        )

        num_gaussian_parameters = self.gaussian_adapter.d_in + 2 + 1
        head_in = cfg.gaussian_regressor_channels + cfg.rgb_feature_dim + cfg.dino_reduced_dim
        self.gaussian_head = nn.Sequential(
            nn.Conv2d(head_in, num_gaussian_parameters, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(num_gaussian_parameters, num_gaussian_parameters, 3, 1, 1, padding_mode="replicate"),
        )

    @torch.no_grad()
    def load_from_infinidepth_gs_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        own_sd = self.state_dict()
        load_sd = {}
        for k, _ in own_sd.items():
            prefixed = f"encoder.{k}"
            if prefixed in state_dict and state_dict[prefixed].shape == own_sd[k].shape:
                load_sd[k] = state_dict[prefixed]
        self.load_state_dict(load_sd, strict=False)

    def forward(
        self,
        image: torch.Tensor,
        depthmap: torch.Tensor,
        dino_tokens: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> Gaussians:
        b, _, h, w = image.shape

        _, n_all, c = dino_tokens.shape
        patch_h = h // 16
        patch_w = w // 16
        n_patch = patch_h * patch_w
        if n_all < n_patch:
            raise ValueError(f"Invalid token count: got {n_all}, expected at least {n_patch}")
        n_reg = n_all - n_patch
        dino_patch_tokens = dino_tokens[:, n_reg:, :]  # [B, patch_h*patch_w, C]
        dino_patch_tokens = dino_patch_tokens.reshape(b, patch_h, patch_w, c).permute(0, 3, 1, 2)

        rgb_feat = self.rgb_encoder(image)
        depth_feat = self.depth_encoder(depthmap)
        dino_feat = self.dino_projector(dino_patch_tokens)

        dino_feat_map = torch.nn.functional.interpolate(
            dino_feat, size=(h, w), mode="bilinear", align_corners=False
        )

        reg_input = torch.cat([rgb_feat, depth_feat, dino_feat_map], dim=1)
        reg_feat = self.gaussian_regressor(reg_input)
        head_input = torch.cat([reg_feat, rgb_feat, dino_feat_map], dim=1)
        raw = self.gaussian_head(head_input)  # [B, Cg, H, W]

        raw = raw.permute(0, 2, 3, 1).reshape(b, h * w, -1)  # [B, HW, Cg]
        opacities = torch.sigmoid(raw[..., :1]).squeeze(-1)   # [B, HW]
        gaussian_core = raw[..., 1:]                          # [B, HW, Cg-1]

        # One surface per pixel in this lightweight integration.
        offset_xy = torch.sigmoid(gaussian_core[..., :2])     # [B, HW, 2], in [0,1]
        raw_gaussians = gaussian_core[..., 2:]                # [B, HW, 7+3*d_sh]

        base = sample_image_grid(h, w, image.device).unsqueeze(0).expand(b, -1, -1)
        coords = base + (offset_xy - 0.5)

        depths = depthmap[:, 0].reshape(b, -1)
        return self.gaussian_adapter(
            image=image,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            coordinates_xy=coords,
            depths=depths,
            opacities=opacities,
            raw_gaussians=raw_gaussians,
        )
