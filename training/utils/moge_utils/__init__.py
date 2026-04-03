from .alignment import (
    align,
    align_depth_scale,
    align_depth_affine,
    align_points_scale,
    align_points_scale_z_shift,
    align_points_scale_xyz_shift,
    align_points_z_shift,
    align_sampled_points_scale_z_shift,
    align_sampled_points_scale_xyz_shift,
)

from .geometry_torch import (
    weighted_mean,
    harmonic_mean,
    geometric_mean,
    normalized_view_plane_uv,
    angle_diff_vec3,
    focal_to_fov,
    fov_to_focal,
    compute_points_from_depth_and_coord,
)

from .moge_sampled_losses import (
    MoGeSampledGlobalLoss,
    MoGeSampledLocalLoss,
    MoGeSampledLocalLossVectorized,
    MoGeSampledEdgeLoss,
    MoGeCombinedSampledLoss,
)

__all__ = [
    # Alignment functions
    'align',
    'align_depth_scale',
    'align_depth_affine',
    'align_points_scale',
    'align_points_scale_z_shift',
    'align_points_scale_xyz_shift',
    'align_points_z_shift',
    'align_sampled_points_scale_z_shift',
    'align_sampled_points_scale_xyz_shift',
    # Geometry functions
    'weighted_mean',
    'harmonic_mean',
    'geometric_mean',
    'normalized_view_plane_uv',
    'angle_diff_vec3',
    'focal_to_fov',
    'fov_to_focal',
    'compute_points_from_depth_and_coord',
    # Loss functions
    'MoGeSampledGlobalLoss',
    'MoGeSampledLocalLoss',
    'MoGeSampledLocalLossVectorized',
    'MoGeSampledEdgeLoss',
    'MoGeCombinedSampledLoss',
]
