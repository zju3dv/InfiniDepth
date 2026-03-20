from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from torch import Tensor

def _construct_attributes(d_sh: int) -> list[str]:
    attrs = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    n_rest = 3 * max(d_sh - 1, 0)
    attrs.extend([f"f_rest_{i}" for i in range(n_rest)])
    attrs.extend(["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    return attrs

def export_ply(
    means: Float[Tensor, "gaussian 3"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: str | Path,
    scales: Float[Tensor, "gaussian 3"] | None = None,
    rotations: Float[Tensor, "gaussian 4"] | None = None,
    covariances: Float[Tensor, "gaussian 3 3"] | None = None,  # Use covariances directly
    shift_to_center: bool = True,
    save_sh_dc_only: bool = True,  # Changed default to False to preserve quality
    center_method: str = "mean",  # "mean", "median", or "bbox_center"
    apply_coordinate_transform: bool = False,  # Apply x90° rotation for viewer compatibility
    focal_length_px: float | tuple[float, float] | None = None,
    principal_point_px: tuple[float, float] | None = None,
    image_shape: tuple[int, int] | None = None,  # (height, width)
    extrinsic_matrix: np.ndarray | torch.Tensor | None = None,
    color_space_index: int | None = None,
):
    path = Path(path)

    # Check input consistency
    if covariances is None and (scales is None or rotations is None):
        raise ValueError("Either provide covariances or both scales and rotations")
    
    # Fast covariance to scale/rotation conversion using batch operations
    if covariances is not None:
        # Batch eigenvalue decomposition - much faster than individual decompositions
        eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
        scales = torch.sqrt(torch.clamp(eigenvalues, min=1e-8))
        
        # Fast batch conversion from rotation matrices to quaternions
        # Using direct mathematical conversion instead of scipy loops
        def rotation_matrix_to_quaternion_batch(R):
            """Fast batch conversion from rotation matrices to quaternions"""
            trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
            
            # Pre-allocate quaternion tensor
            quat = torch.zeros(R.shape[0], 4, dtype=R.dtype, device=R.device)
            
            # Case 1: trace > 0
            mask1 = trace > 0
            if mask1.any():
                s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
                quat[mask1, 0] = 0.25 * s  # qw
                quat[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s  # qx
                quat[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s  # qy
                quat[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s  # qz
            
            # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
            mask2 = ~mask1 & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
            if mask2.any():
                s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
                quat[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s  # qw
                quat[mask2, 1] = 0.25 * s  # qx
                quat[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s  # qy
                quat[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s  # qz
            
            # Case 3: R[1,1] > R[2,2]
            mask3 = ~mask1 & ~mask2 & (R[..., 1, 1] > R[..., 2, 2])
            if mask3.any():
                s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
                quat[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s  # qw
                quat[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s  # qx
                quat[mask3, 2] = 0.25 * s  # qy
                quat[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s  # qz
            
            # Case 4: else
            mask4 = ~mask1 & ~mask2 & ~mask3
            if mask4.any():
                s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
                quat[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s  # qw
                quat[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s  # qx
                quat[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s  # qy
                quat[mask4, 3] = 0.25 * s  # qz
            
            return quat
        
        # Ensure proper rotation matrices
        det = torch.det(eigenvectors)
        eigenvectors = torch.where(det.unsqueeze(-1).unsqueeze(-1) < 0, 
                                 -eigenvectors, eigenvectors)
        
        # Fast batch conversion
        rotations = rotation_matrix_to_quaternion_batch(eigenvectors)
    
    # Apply centering - vectorized operations
    if shift_to_center:
        if center_method == "mean":
            center = means.mean(dim=0)
        elif center_method == "median":
            center = means.median(dim=0).values
        elif center_method == "bbox_center":
            center = (means.min(dim=0).values + means.max(dim=0).values) / 2
        else:
            raise ValueError(f"Unknown center_method: {center_method}")
        means = means - center

    # Fast coordinate transformation using batch operations
    if apply_coordinate_transform:
        # X-axis 90° rotation matrix
        rot_x = torch.tensor([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=means.dtype, device=means.device)
        
        # Apply to positions - batch matrix multiplication
        means = means @ rot_x.T
        
        # Apply to rotations - batch quaternion operations
        transform_quat = torch.tensor([0.7071068, 0.7071068, 0.0, 0.0], 
                                    dtype=rotations.dtype, device=rotations.device)  # 90° around X
        
        # Batch quaternion multiplication
        w1, x1, y1, z1 = transform_quat[0], transform_quat[1], transform_quat[2], transform_quat[3]
        w2, x2, y2, z2 = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
        
        rotations = torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
            w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
        ], dim=1)

    # Convert to numpy for PLY writing - single conversion
    means_np = means.detach().cpu().numpy()
    scales_np = scales.detach().cpu().numpy()
    rotations_np = rotations.detach().cpu().numpy()
    opacities_np = opacities.detach().cpu().numpy()
    harmonics_np = harmonics.detach().cpu().numpy()

    # Process harmonics
    f_dc = harmonics_np[..., 0]
    f_rest = harmonics_np[..., 1:].reshape(harmonics_np.shape[0], -1)
    
    d_sh = harmonics_np.shape[-1]
    dtype_full = [
        (attribute, "f4")
        for attribute in _construct_attributes(1 if save_sh_dc_only else d_sh)
    ]
    elements = np.empty(means_np.shape[0], dtype=dtype_full)
    
    # Build attributes list
    attributes = [
        means_np,
        np.zeros_like(means_np),  # normals
        f_dc,
    ]
    
    if not save_sh_dc_only:
        attributes.append(f_rest)
    
    # Apply inverse sigmoid to opacity for storage (viewer will apply sigmoid when loading)
    # logit(opacity) = log(opacity / (1 - opacity))
    opacities_clamped = np.clip(opacities_np, 1e-6, 1 - 1e-6)  # Clamp to avoid log(0) or log(inf)
    opacities_logit = np.log(opacities_clamped / (1 - opacities_clamped))
    
    attributes.extend([
        opacities_logit.reshape(-1, 1),
        np.log(scales_np),
        rotations_np
    ])

    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    ply_elements = [PlyElement.describe(elements, "vertex")]

    if focal_length_px is not None and image_shape is not None:
        image_height, image_width = image_shape
        if isinstance(focal_length_px, tuple):
            fx, fy = float(focal_length_px[0]), float(focal_length_px[1])
        else:
            fx = fy = float(focal_length_px)
        if principal_point_px is None:
            cx = image_width * 0.5
            cy = image_height * 0.5
        else:
            cx = float(principal_point_px[0])
            cy = float(principal_point_px[1])

        dtype_image_size = [("image_size", "u4")]
        image_size_array = np.empty(2, dtype=dtype_image_size)
        image_size_array[:] = np.array([image_width, image_height], dtype=np.uint32)
        ply_elements.append(PlyElement.describe(image_size_array, "image_size"))

        dtype_intrinsic = [("intrinsic", "f4")]
        intrinsic_array = np.empty(9, dtype=dtype_intrinsic)
        intrinsic = np.array(
            [
                fx,
                0.0,
                cx,
                0.0,
                fy,
                cy,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )
        intrinsic_array[:] = intrinsic.flatten()
        ply_elements.append(PlyElement.describe(intrinsic_array, "intrinsic"))

        dtype_extrinsic = [("extrinsic", "f4")]
        extrinsic_array = np.empty(16, dtype=dtype_extrinsic)
        if extrinsic_matrix is None:
            extrinsic_np = np.eye(4, dtype=np.float32)
        elif torch.is_tensor(extrinsic_matrix):
            extrinsic_np = extrinsic_matrix.detach().cpu().numpy().astype(np.float32)
        else:
            extrinsic_np = np.asarray(extrinsic_matrix, dtype=np.float32)
        if extrinsic_np.shape != (4, 4):
            raise ValueError(f"extrinsic_matrix must have shape (4,4), got {extrinsic_np.shape}")
        extrinsic_array[:] = extrinsic_np.flatten()
        ply_elements.append(PlyElement.describe(extrinsic_array, "extrinsic"))

        dtype_color_space = [("color_space", "u1")]
        color_space_array = np.empty(1, dtype=dtype_color_space)
        color_space_array[:] = np.array([1 if color_space_index is None else color_space_index], dtype=np.uint8)
        ply_elements.append(PlyElement.describe(color_space_array, "color_space"))

    PlyData(ply_elements).write(path)
