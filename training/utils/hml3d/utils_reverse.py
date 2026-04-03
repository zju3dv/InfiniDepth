import torch
import torch.nn.functional as F
from training.utils.logger import Log
from training.utils.hml3d import REPEAT_LAST_FRAME, ZERO_FRAME_AHEAD

# =========== quaternion-conversion  =========== #


def qinv(q):
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


# =========== functional  =========== #


def recover_root_rot_pos(data, abs_3d=False, from_velocity_padding_strategy=None):
    """
    data: Shape (..., D). The first 4 elements of D are (rot-y, x, z, y)
    """
    if abs_3d:
        # Y-axis rotaion is absolute (already summed)
        r_rot_ang = data[..., 0]
    else:
        rot_vel = data[..., 0]
        if from_velocity_padding_strategy == REPEAT_LAST_FRAME:
            r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
            r_rot_ang[..., 1:] = rot_vel[..., :-1]
        elif from_velocity_padding_strategy == ZERO_FRAME_AHEAD:
            r_rot_ang = rot_vel.clone()  # the velocity of first frame is always zero
        else:
            raise ValueError(f"Unknown velocity padding strategy: {from_velocity_padding_strategy}")

        # Get Y-axis rotation from rotation velocity
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)

    if abs_3d:
        # r_pos is absolute and not depends on Y-axis rotation. And already summed
        # (x,z) [0,2] <= (x,z) [1,2]
        r_pos[..., :, [0, 2]] = data[..., :, 1:3]
    else:
        # Add Y-axis rotation to root position
        # (x,z) [0,2] <= (x,z) [1,2]
        # adding zero at 0 index
        # data   [+1, -2, -3, +5, xx]
        # r_pose [0, +1, -2, -3, +5]
        # r_pos[..., 1be:, [0, 2]] = data[..., :-1, 1:3]
        if from_velocity_padding_strategy == REPEAT_LAST_FRAME:
            r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3].float()
        elif from_velocity_padding_strategy == ZERO_FRAME_AHEAD:
            r_pos[..., 1:, [0, 2]] = data[..., 1:, 1:3].float()  # the velocity of first frame is always zero
        else:
            raise ValueError(f"Unknown velocity padding strategy: {from_velocity_padding_strategy}")
        r_pos = qrot(qinv(r_rot_quat), r_pos)
        r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num=22, abs_3d=False, from_velocity_padding_strategy=None):
    """
    Args:
        data: Shape (..., D). The first 4 elements of D are (rot-y, x, z, y)
    Returns:
        positions: (B, F, 22, 3)
    """
    assert from_velocity_padding_strategy != None, "the recover strategy must be given!"
    assert data.shape[-1] == 263
    r_rot_quat, r_pos = recover_root_rot_pos(
        data, abs_3d=abs_3d, from_velocity_padding_strategy=from_velocity_padding_strategy
    )  # (..., 0), (..., 1:3)

    # joint positions
    positions = data[..., 4 : 4 + (joints_num - 1) * 3]  # (..., 63)
    positions = positions.view(positions.shape[:-1] + (joints_num - 1, 3))  # (..., 21, 3)

    # apply y-axis rotation to joints
    r_rot_quat = r_rot_quat.unsqueeze(-2).expand(positions.shape[:-1] + (4,))
    positions = qrot(qinv(r_rot_quat), positions)

    # apply root xz to joints
    positions[..., 0] += r_pos[..., [0]]
    positions[..., 2] += r_pos[..., [2]]

    # append root to joints
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def convert_bf3_to_b4f(x, pad_f_to=False):
    """
    Very useful in converting cond_traj(B,F,3) to HMLVec4(B,4,F)
    Args:
        x (tensor): Shape (B, F, 3)
        pad_f_to (int): pad the last dim to this length
    """
    assert x.shape[-1] == 3 and len(x.shape) == 3
    x_out = F.pad(x[..., [0, 2, 1]].permute(0, 2, 1), (0, 0, 1, 0), value=0.0)
    if pad_f_to:
        assert pad_f_to >= x_out.size(-1)
        x_out = F.pad(x_out, (0, pad_f_to - x_out.size(-1)), value=0.0)
    return x_out


def convert_bfj3_to_b263f(x, pad_f_to=False):
    """
    Very useful in converting cond_motion(B,F,22,3) to HMLVec263(B,263,F)
    This function works similar as convert_bf3_to_b4f
    """
    assert x.shape[-2:] == (22, 3) and len(x.shape) == 4
    b4f = convert_bf3_to_b4f(x[:, :, 0], pad_f_to)
    # TODO: currently only works correctly when only root joint is given
    assert (x[:, :, 1:] == 0).all(), "currently only works correctly when only root joint is given"
    out = F.pad(b4f, (0, 0, 0, 263 - 4), value=0.0)
    return out


# =========== functional (motion to hmlvec263)  =========== #


def make_hmlvec263_abs(hmlvec263):
    """
    hmlvec263: Shape (..., D). The 3/4 of first elements(rot-y, x, z, y) are relative
    """
    device = hmlvec263.device
    shape_wo263 = hmlvec263.shape[:-1]

    # Get Y-axis rotation from rotation velocity, the last element is not used
    rot_vel = hmlvec263[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(device)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    # Add Y-axis rotation to root position
    r_rot_quat = torch.zeros(shape_wo263 + (4,)).to(device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)
    r_pos = torch.zeros(shape_wo263 + (3,)).to(device)
    r_pos[..., 1:, [0, 2]] = hmlvec263[..., :-1, [1, 2]].float()
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = hmlvec263[..., 3]  # y is absolute

    # Overwrite
    hmlvec263[..., 0] = r_rot_ang  # rot-y
    hmlvec263[..., 1:3] = r_pos[..., [0, 2]]  # x, z
    return hmlvec263


def convert_hmlvec263_to_motion(
    hmlvec263,
    seq_len=None,
    abs_3d=False,
    from_velocity_padding_strategy=REPEAT_LAST_FRAME,
):
    """
    Args:
        hmlvec263 (torch.Tensor): (B, 263, F), absolute
        seq_len (torch.Tensor): (B,)
        abs_3d (bool): if True, the output(first 4) is absolute 3d. By default, HML3D uses False
        from_velocity_padding_strategy (str): the hml263 velocity padding strategy used in convert_motion_to_hmlvec263
    Returns:
        motion (torch.Tensor): (B, F, 22, 3)
    """
    device = hmlvec263.device
    B, _, F = hmlvec263.shape

    if seq_len is None:
        seq_len = torch.tensor([F] * B, device=device)
        # Log.warn("seq_len is None, use F as seq_len")

    mask_invalid_frames = torch.arange(F, device=device)[None].repeat(B, 1) >= seq_len[:, None]
    # Recover with root data and ric.
    motion = recover_from_ric(
        hmlvec263.permute(0, 2, 1),
        abs_3d=abs_3d,
        from_velocity_padding_strategy=from_velocity_padding_strategy,
    )
    # Erase the invalid frames.
    motion[mask_invalid_frames] = 0
    return motion
