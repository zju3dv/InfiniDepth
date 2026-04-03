import torch
import training.utils.matrix as matrix


def get_camera_mat(mat, distance, angle):
    """_summary_
    We assume camera always rotate with distance and angle, points at the mat.
    We also assume z-axis is upward, x-axis is facing.

    Args:
        mat (Tensor): [*, 4, 4]
        pos (Tensor): [*]
        angle (Tensor):[*]
    """
    #  FIXME: not opencv coordinate
    # put z-axis on the ground (-x axis)
    y_axis = torch.zeros(angle.shape[:-1] + (3,), device=angle.device)
    y_axis[..., 1] = 1.0
    cam_rot = matrix.quat_from_angle_axis(-torch.pi / 2 + torch.zeros_like(angle), y_axis)  # [*, 4]
    cam_rotmat = matrix.rot_matrix_from_quaternion(cam_rot)  # [*, 3, 3]

    # now for camera, x-axis is upward
    x_axis = torch.zeros(angle.shape[:-1] + (3,), device=angle.device)
    x_axis[..., 0] = 1.0
    cam_rot_ = matrix.quat_from_angle_axis(angle, x_axis)  # [*, 4]
    cam_rotmat_ = matrix.rot_matrix_from_quaternion(cam_rot_)  # [*, 3, 3]
    cam_rotmat = matrix.get_mat_BfromA(cam_rotmat, cam_rotmat_)  # [*, 3, 3]
    pos = torch.stack((torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)), dim=-1)  # [*, 3]
    pos = pos * distance[..., None]
    cam_mat = matrix.get_TRS(cam_rotmat, pos)
    cam_mat = matrix.get_mat_BfromA(mat, cam_mat)
    return cam_mat


def get_camera_mat_zface(mat, distance, hor_angle, elevation_angle=None, is_opencv=True):
    """_summary_
    We assume camera always rotate with distance, hor_angle, and elevation_angle, points at the mat.
    We also assume y-axis is upward, z-axis is facing.

    Args:
        mat (Tensor): [*, 4, 4]
        pos (Tensor): [*]
        hor_angle (Tensor):[*]
        elevation_angle (Tensor):[*]
    """
    # rotate to concentrate on human
    y_axis = torch.zeros(hor_angle.shape[:-1] + (3,), device=hor_angle.device)
    y_axis[..., 1] = 1.0
    cam_rot = matrix.quat_from_angle_axis(hor_angle, y_axis)  # [*, 4]
    cam_rotmat = matrix.rot_matrix_from_quaternion(cam_rot)  # [*, 3, 3]

    if elevation_angle is not None:
        x_axis = torch.zeros(elevation_angle.shape[:-1] + (3,), device=elevation_angle.device)
        x_axis[..., 0] = 1.0
        cam_rot_ = matrix.quat_from_angle_axis(elevation_angle, x_axis)  # [*, 4]
        cam_rotmat_ = matrix.rot_matrix_from_quaternion(cam_rot_)  # [*, 3, 3]
        cam_rotmat = matrix.get_mat_BfromA(cam_rotmat, cam_rotmat_)  # [*, 3, 3]
    else:
        elevation_angle = torch.zeros_like(hor_angle)

    if is_opencv:
        # rotate to opencv
        z_axis = torch.zeros(hor_angle.shape[:-1] + (3,), device=hor_angle.device)
        z_axis[..., 2] = 1.0
        cam_rot_ = matrix.quat_from_angle_axis(torch.pi + torch.zeros_like(hor_angle), z_axis)  # [*, 4]
        cam_rotmat_ = matrix.rot_matrix_from_quaternion(cam_rot_)  # [*, 3, 3]
        cam_rotmat = matrix.get_mat_BfromA(cam_rotmat, cam_rotmat_)  # [*, 3, 3]
    xz_dist = torch.cos(elevation_angle).abs() * distance
    pos = torch.stack(
        (-torch.sin(hor_angle) * xz_dist, torch.sin(elevation_angle) * distance, -torch.cos(hor_angle) * xz_dist),
        dim=-1,
    )  # [*, 3]
    cam_mat = matrix.get_TRS(cam_rotmat, pos)
    cam_mat = matrix.get_mat_BfromA(mat, cam_mat)
    return cam_mat


def cartesian_to_spherical(xyz):
    """_summary_
    From zero1to3: https://github.com/cvlab-columbia/zero123/blob/main/zero123/ldm/data/simple.py#L248
    Given a position in cartesian coordinates, return the spherical coordinates
    Because we assume camera is pointing at the center, so do not need camera rotation.
    Original code assumes z is upward, we modify it to y is upward.

    ###################### Original Code ################################
    xy = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    z = torch.sqrt(xy + xyz[..., 2] ** 2)
    theta = torch.arctan2(torch.sqrt(xy), xyz[..., 2])  # for elevation angle defined from Z-axis down
    azimuth = torch.arctan2(xyz[..., 1], xyz[..., 0])
    #####################################################################

    Args:
        xyz (_tensor_): [..., 3]

    Returns:
        _(..., 3) spherical coordinate
    """
    xy = xyz[..., :1] ** 2 + xyz[..., 2:3] ** 2
    z = torch.sqrt(xy + xyz[..., 1:2] ** 2)
    theta = torch.arctan2(torch.sqrt(xy), xyz[..., 1:2])  # for elevation angle defined from Z-axis down
    azimuth = torch.arctan2(xyz[..., 2:3], xyz[..., :1])
    return torch.cat([theta, azimuth, z], dim=-1)
