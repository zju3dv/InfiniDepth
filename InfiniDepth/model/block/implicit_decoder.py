import enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, output_act='elu'):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers += [nn.Linear(lastv, hidden), nn.ReLU()]
            lastv = hidden

        if out_dim is not None:
            layers.append(nn.Linear(lastv, out_dim))
            act = {
                "sigmoid": nn.Sigmoid(),
                "relu": nn.ReLU(),
                "elu": nn.ELU(),
            }.get(output_act, nn.Identity())
            layers.append(act)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class ImplicitHead(nn.Module):
    """
    Implicit head that fuses DINOv3 semantic features and BasicEncoder low-level features.

    Args:
        hidden_dim: DINOv3 feature dimension (e.g., 1024)
        basic_dim: BasicEncoder feature dimension (e.g., 128)
        fusion_type: Feature fusion strategy
            - "concat": Simple concatenation
            - "cross_attn": Cross-attention between features
            - "gated": Gated fusion with learnable weights
        out_dim: Output dimension (1 for depth)
        hidden_list: MLP hidden layer dimensions
    """
    def __init__(
            self,
            hidden_dim,  # 1024 for DINOv3
            basic_dim=128,  # BasicEncoder output dim
            fusion_type="gated",  # concat, gated
            out_dim=1,
            hidden_list=[1024, 256, 32],
            ):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.basic_dim = basic_dim
        self.fusion_type = fusion_type

        # Determine input dimension based on fusion type
        if fusion_type == "concat":
            # Simple concatenation
            in_channels = hidden_dim + basic_dim
        elif fusion_type == "gated":
            # Gated fusion with learnable weights
            self.gate_proj = nn.Linear(basic_dim, hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            in_channels = hidden_dim
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.out_layer = MLP(
            in_dim=in_channels,
            out_dim=out_dim,
            hidden_list=hidden_list,
            output_act='elu'
        )

    def _encode_feat(self, features, patch_h, patch_w):
        """Extract DINOv3 feature map."""
        x = features[-1][0]
        out_feat = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
        return out_feat

    def _decode_dpt(self, feat, basic_feat, coord):
        """
        Query features at given coordinates and fuse them.

        Args:
            feat: DINOv3 feature map [B, hidden_dim, H_dino, W_dino]
            basic_feat: BasicEncoder feature map [B, basic_dim, H_basic, W_basic]
            coord: Query coordinates [B, N, 2] in range [-1, 1]

        Returns:
            pred: Predicted depth [B, N, 1]
        """
        coord_ = coord.clone()
        coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

        # Sample DINOv3 features at query coordinates
        q_feat_dino = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)  # [B, N, hidden_dim]

        # Sample BasicEncoder features at query coordinates (if available)
        if basic_feat is not None:
            q_feat_basic = F.grid_sample(
                basic_feat, coord_.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False
            )[:, :, 0, :].permute(0, 2, 1)  # [B, N, basic_dim]

            # Fuse features based on fusion type
            q_feat_fused = self._fuse_features(q_feat_dino, q_feat_basic)
        else:
            # If no basic features, use only DINOv3
            q_feat_fused = q_feat_dino

        # Predict depth
        pred = self.out_layer(q_feat_fused)
        return pred

    def _fuse_features(self, feat_dino, feat_basic):
        """
        Fuse DINOv3 and BasicEncoder features.

        Args:
            feat_dino: [B, N, hidden_dim]
            feat_basic: [B, N, basic_dim]

        Returns:
            fused_feat: [B, N, fused_dim]
        """
        if self.fusion_type == "concat":
            # Simple concatenation
            return torch.cat([feat_dino, feat_basic], dim=-1)

        elif self.fusion_type == "gated":
            # Gated fusion with learnable weights
            feat_basic_proj = self.gate_proj(feat_basic)  # [B, N, hidden_dim]
            gate_input = torch.cat([feat_dino, feat_basic_proj], dim=-1)
            gate_weights = self.gate(gate_input)  # [B, N, hidden_dim]
            return gate_weights * feat_dino + (1 - gate_weights) * feat_basic_proj

    def forward(self, features, basic_feat, patch_h, patch_w, coords):
        """
        Forward pass.

        Args:
            features: DINOv3 features from backbone
            basic_feat: BasicEncoder features [B, basic_dim, H/4, W/4]
            patch_h, patch_w: DINOv3 feature map spatial size
            coords: Query coordinates [B, N, 2]

        Returns:
            dpt_pred: Predicted depth [B, N, 1]
        """
        # Extract DINOv3 feature map
        feat = self._encode_feat(features, patch_h, patch_w)  # [B, hidden_dim, H/14, W/14]

        # Query and fuse features at coordinates
        dpt_pred = self._decode_dpt(feat, basic_feat, coords)

        return dpt_pred
