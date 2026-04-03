import numpy as np
import torch
from pathlib import Path
import os

from .skeleton import Skeleton
from .quaternion import *
from .paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from training.utils.logger import Log
from training.utils.debug_utils import detectNaN

# Lower legs
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1
joints_num = 22

n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain

# Get offsets of target skeleton
example_data = np.load("lib/utils/hml3d/000021.npy")
example_data = example_data.reshape(len(example_data), -1, 3)
example_data = torch.from_numpy(example_data)
tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
# (joints_num, 3)
tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

# velocity padding strategy
REPEAT_LAST_FRAME = "repeat_last_frame"
ZERO_FRAME_AHEAD = "zero_frame_ahead"


def uniform_skeleton_batch(positions):
    B, F, J, _ = positions.shape
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, positions.device)
    src_offset = src_skel.get_offsets_joints_batch(positions[:, 0])
    tgt_offset = tgt_offsets.clone().to(positions.device)  # J, 3
    # print(src_offset)
    # print(tgt_offset)
    """Calculate Scale Ratio as the ratio of legs"""
    src_leg_len = src_offset[:, l_idx1].abs().max(dim=-1)[0] + src_offset[:, l_idx2].abs().max(dim=-1)[0]
    tgt_leg_len = tgt_offset[l_idx1].abs().max(dim=-1)[0] + tgt_offset[l_idx2].abs().max(dim=-1)[0]

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, :, 0]
    tgt_root_pos = src_root_pos * scale_rt[..., None, None]

    """Inverse Kinematics"""
    quat_params = src_skel.inverse_kinematics(positions, face_joint_indx, use_original_edition=True)  # B*F, J, 4
    # print(quat_params.shape)

    """Forward Kinematics"""
    src_skel.set_offset(tgt_offsets)
    tgt_root_pos = tgt_root_pos.reshape(-1, 3)
    new_joints = src_skel.forward_kinematics(quat_params, tgt_root_pos)
    new_joints = new_joints.reshape(B, F, J, -1)
    return new_joints


def standardize_motion(motion, is_put_on_the_floor=False, use_original_edition=False):
    """Make an arbitrary motion to a standard motion.

    1. Put the motion on the xz plane.
    2. Make the motion starts at (0, y, 0).
    3. Make the motion face z+ at the first frame.
    (TODO: we ignore the scale things here.)

    ### Args:
    - `motion` (torch.Tensor): (F, J=22, 3), arbitrary joints position of each frame
    ### Returns:
    - `motion` (torch.Tensor): (F, J=22, 3), standardized joints position of each frame
    """
    F = motion.shape[0]  # the number of frames
    device = motion.device

    if use_original_edition:
        motion = uniform_skeleton_batch(motion)

    # 1. Put the motion on the xz plane.
    # this shouldn't be use in our method, cause we have floor in the scene.
    if is_put_on_the_floor:
        floor_y = motion[..., 1].min()
        motion[..., 1] -= floor_y

    # 2. Make the motion starts at (0, y, 0).
    xz_mask = torch.tensor([1, 0, 1], dtype=torch.float32).to(device)
    xz_offset = motion[..., :1, :1, :] * xz_mask  # filter the xz of the first pelvis position
    motion -= xz_offset

    # 3. Make the motion face z+ at the first frame.
    # 3.1. Get helper vector.
    # pos_z_vec = torch.tensor([0, 0, 1], dtype=torch.float32).to(device)
    # pos_y_vec = torch.tensor([0, 1, 0], dtype=torch.float32).to(device)
    # 3.2. Use cross product to calculate the current forward direction of the initial pose.
    r_hip, l_hip, r_sdr, l_sdr = face_joint_indx  # use hip and shoulder to get the cross vector
    cross_hip = motion[..., :1, r_hip : r_hip + 1, :] - motion[..., :1, l_hip : l_hip + 1, :]
    cross_sdr = motion[..., :1, r_sdr : r_sdr + 1, :] - motion[..., :1, l_sdr : l_sdr + 1, :]
    cross_vec = cross_hip + cross_sdr  # (3, )

    pos_y_vec = torch.zeros_like(cross_vec)
    pos_y_vec[..., 1] = 1.0
    forward_vec = torch.cross(pos_y_vec, cross_vec, dim=-1)
    forward_vec = forward_vec / torch.clamp(torch.norm(forward_vec, dim=-1, keepdim=True), min=1e-8)

    # 3.3. Get the transformation quaternion from the current forward direction to z+ axis.
    pos_z_vec = torch.zeros_like(forward_vec)
    pos_z_vec[..., 2] = 1.0
    quat_trans = qbetween(forward_vec, pos_z_vec)  # (4,)
    quat_trans = quat_trans.expand(motion.shape[:-1] + (4,))  # (F, 22, 4)
    # 3.4. Perform the in-place transformation.
    motion = qrot(quat_trans, motion)

    return motion


def motion2poses(motion, rot2z):
    """Convert the motion to spacewalk pose, i.e. remove all movement and rotation.

    ### Args:
    - `motion`(torch.Tensor): (B*F, J=22, 3), batch of joints position of each frame
    - `rot2z`(torch.Tensor): (B*F, 4), batch of rotation from current face direction to z+
    ### Returns:
    - `poses` (torch.Tensor): (B*F, J=22, 3), batch of joints position of each pose
    """
    BF, J, _ = motion.shape
    device = motion.device

    # 1. Remove translation.
    pelvises = motion[:, 0]  # (BF, 3)
    xz_mask = torch.tensor([1, 0, 1], dtype=torch.float32).to(device)  # (3,)
    xz_offset = pelvises * xz_mask  # (BF, 3), filter the xz of the first pelvis position
    pose = motion - xz_offset[:, None]  # (BF, J=22, 3)

    # 2. Remove rotation about y-axis, i.e. never turn around.
    joints_rot2z = rot2z[:, None].expand(BF, J, 4)  # (BF, J, 4)
    pose = qrot(joints_rot2z, pose)

    return pose


def detect_foot_contact(motion, thre, padding_strategy=REPEAT_LAST_FRAME):
    """Label if the foot contact the floor.

    If the movement is large enough, it will be 1.0, otherwise 0.0.
    # TODO: Is this really ok? What if the movement is obvious? It will always be 0?

    ### Args:
    - `motion`(torch.Tensor): ((B), J=22, 3), joints position of each frame
    - `thre`(float): threshold factor to detect the foot contact the floor
    ### Returns:
    - `l_fc_labels`(torch.Tensor): ((B), 2), double foot contact labels of left foot
    - `r_fc_labels`(torch.Tensor): ((B), 2), double foot contact labels of right foot
    """
    device = motion.device
    motion_shape = list(motion.shape)
    vel_factor = torch.tensor([thre, thre]).expand(motion_shape[:-2] + [2]).to(device)
    # support no batch input
    if len(motion_shape) == 3:
        motion = motion[None]

    feet_l_xyz = motion[:, 1:, fid_l, :] - motion[:, :-1, fid_l, :]  # (F-1, 2, 3)
    feet_l_l2dis = torch.norm(feet_l_xyz, dim=-1)  # (F-1, 2)
    if padding_strategy == REPEAT_LAST_FRAME:
        feet_l_l2dis = torch.cat(
            [feet_l_l2dis, feet_l_l2dis[:, [-1], :].clone()], dim=-2
        )  # ((B), F, 1), padding by append the last frame again
    elif padding_strategy == ZERO_FRAME_AHEAD:
        feet_l_l2dis = torch.cat(
            [feet_l_l2dis[:, [-1], :].clone().fill_(0), feet_l_l2dis], dim=-2
        )  # ((B), F, 1), padding zero ahead
    else:
        raise ValueError(f"Unknown velocity padding strategy: {padding_strategy}")
    feet_l = (feet_l_l2dis**2 < vel_factor).float()  # (F, 2)

    feet_r_xyz = motion[:, 1:, fid_r, :] - motion[:, :-1, fid_r, :]  # (F-1, 3)
    feet_r_l2dis = torch.norm(feet_r_xyz, dim=-1)  # (F-1, 1)
    if padding_strategy == REPEAT_LAST_FRAME:
        feet_r_l2dis = torch.cat(
            [feet_r_l2dis, feet_r_l2dis[:, [-1], :].clone()], dim=-2
        )  # ((B), F, 1), padding by append the last frame again
    elif padding_strategy == ZERO_FRAME_AHEAD:
        feet_r_l2dis = torch.cat(
            [feet_r_l2dis[:, [-1], :].clone().fill_(0), feet_r_l2dis], dim=-2
        )  # ((B), F, 1), padding zero ahead
    else:
        raise ValueError(f"Unknown velocity padding strategy: {padding_strategy}")
    feet_r = (feet_r_l2dis**2 < vel_factor).float()  # (F, 2)

    return feet_l, feet_r


def convert_motion_to_hmlvec263(
    motion,
    seq_len=None,
    feet_thre=0.002,
    return_abs=False,
    use_original_edition=False,
    smooth_motion=True,
    velocity_padding_strategy=REPEAT_LAST_FRAME,  # val(REPEAT_LAST_FRAME) | val(ZERO_FRAME_AHEAD)
):
    """
    ### Args:
    - `motion` (torch.Tensor): (B, F, J=22, 3), batch of joints position of each frame
    - `feet_threshold` (float): threshold to detect whether the foot contact the floor
    - `return_abs` (bool): whether to return the absolute value of rotation and linear velocity, `False` by default
    - `use_original_edition` (bool): whether to use the original edition
    - `seq_len` (tensor): if the motion is uneven, i.e. each sequence may have different length, the limits should be given
    - `smooth_motion` (tensor): whether to perform smooth_forward while inverse kinematics
    - `velocity_padding_strategy` (string): how to deal with the dimension loss of "velocity"
    ### Returns:
    - `hmlvec263`: (B, 263, F-1)
    """
    assert isinstance(motion, torch.Tensor), "The input `motion` should be a torch Tensor!"
    assert isinstance(seq_len, torch.Tensor) or seq_len is None, "The input `seq_len` should be a torch Tensor or None!"
    device = motion.device
    motion = motion.detach().clone()
    B, F, J, _ = motion.shape

    if seq_len is None:
        seq_len = torch.tensor([F] * B, device=device)
        # Log.warn("seq_len is None, use F as seq_len")
    index_batch = torch.arange(B, device=device)
    index_last_frame = seq_len - 1
    mask_invalid_frames = torch.arange(F, device=device)[None].repeat(B, 1) >= seq_len[:, None]
    # Add this after finalize the code

    # 1. Standardize the motion.
    # no need to do this cause we already do this before conversion
    if use_original_edition:
        motion = standardize_motion(motion, use_original_edition=use_original_edition)

    # 2. Convert absolute xyz expression to joint-relative quaternion.
    skel = Skeleton(n_raw_offsets, kinematic_chain, device)
    quat_params = skel.inverse_kinematics(
        motion, face_joint_indx, smooth_forward=smooth_motion, use_original_edition=use_original_edition
    )  # (B*F, J=22, 4)
    quat_params = quat_params.reshape(B, F, J, 4)

    # ================================================================================ #
    # Tips: things below are not a linear logic, check the program flow to get clearly #
    #       understanding.                                                             #
    # ================================================================================ #

    # 3. Calculate hml263 rotation velocity.
    # we use pelvis's quaternion to express the body's direction(the quat describe the rotation from now to z+)
    pelvis_rot2z = quat_params[:, :, 0]  # (B, F, 4)
    body_abs_r_ang = torch.asin(pelvis_rot2z[:, :, 2:3])

    if return_abs:
        hml263_abs_r = body_abs_r_ang
    else:
        # pelvis_rot2prev = qmul(pelvis_rot2z[:, 1:], qinv(pelvis_rot2z[:, :-1]))  # (B, F-1, 4), rotation from i+1 to i
        # in `pelvis_rot2prev`, w^2 + y^2 = 1, x = z = 0, so `asin(y)` is enough to recover the quaternion
        hml263_vel_r = body_abs_r_ang[:, 1:] - body_abs_r_ang[:, :-1]  # (B, F-1, 1)
        zero_row = hml263_vel_r[:, [0]].clone().fill_(0)  # (B, 1, 1)
        if velocity_padding_strategy == REPEAT_LAST_FRAME:
            # append zero to adjust the length, and then copy the last frame
            hml263_vel_r = torch.cat([hml263_vel_r, zero_row], dim=1)  # (B, F, 1)
            hml263_vel_r[index_batch, index_last_frame] = hml263_vel_r[index_batch, index_last_frame - 1].clone()
        elif velocity_padding_strategy == ZERO_FRAME_AHEAD:
            # append the first frame by zero, i.e. the first frame has no velocity
            hml263_vel_r = torch.cat([zero_row, hml263_vel_r], dim=1)  # (B, F, 1)
        else:
            raise ValueError(f"Unknown velocity padding strategy: {velocity_padding_strategy}")

    # 4. Calculate hml263 linear velocity at xz-plane. It's velocity without rotation, i.e. the
    #    forward velocity. This part is quite alike to 9., which need all the velocity instead of
    #    only xz-velocity of pelvis we need here.
    if return_abs:
        hml264_abs_l = motion[:, :, 0, [0, 2]]
    else:
        joints_vel = motion[:, 1:] - motion[:, :-1]  # (B, F-1, 22, 3)
        body_rot2z = pelvis_rot2z[:, :, None].expand(B, F, J, 4)  # (B, F, 4) -> (B, F, 1, 4) -> (B, F, 22, 4)
        joints_vel_relative = qrot(body_rot2z[:, 1:], joints_vel)  # (B, F, 22, 3)
        zero_row = joints_vel_relative[:, [0]].clone().fill_(0)
        if velocity_padding_strategy == REPEAT_LAST_FRAME:
            # append zero to adjust the length, and then copy the last frame
            joints_vel_relative = torch.cat([joints_vel_relative, zero_row], dim=1)  # (B, F, 22, 3),
            joints_vel_relative[index_batch, index_last_frame] = joints_vel_relative[
                index_batch, index_last_frame - 1
            ].clone()
        elif velocity_padding_strategy == ZERO_FRAME_AHEAD:
            # append the first frame by zero, i.e. the first frame has no velocity
            joints_vel_relative = torch.cat([zero_row, joints_vel_relative], dim=1)  # (B, F, 22, 3)
        else:
            raise ValueError(f"Unknown velocity padding strategy: {velocity_padding_strategy}")

        hml263_vel_l = joints_vel_relative[:, :, 0, [0, 2]]  # (B, F, 2), only xz-velocity of pelvis is needed here

    # 5. After getting the rotation information, we can get the pure poses which has nothing to do
    #    with rotation and translation, i.e. the spacewalk pose.
    poses = motion2poses(motion.reshape(B * F, J, 3), pelvis_rot2z.reshape(B * F, 4))  # (B*F, 22, 3)
    poses = poses.reshape(B, F, J, 3)  # (B, F, 22, 3)

    # 6. Get pelvis's height.
    pelvis_y = poses[:, :, 0, 1]  # (B, F,)
    hml263_pelvis_y = pelvis_y[:, :, None]  # (B, F, 1)

    # 7. Get rotation invariant coordinate, i.e. ric. That's what contained by `poses`.
    # remove last frame & remove pelvis & flatten
    ric = poses[:, :, 1:]  # (B, F, 21, 3)
    hml263_ric = ric.flatten(start_dim=2)  # (B, F, 21*3)

    # 8. Get continuous 6D representation.
    cont6d = quaternion_to_cont6d(quat_params)  # (B, F, 22, 6)
    # remove last frame & remove pelvis & flatten
    rot = cont6d[:, :, 1:]  # (B, F, 21, 6)
    hml263_rot = rot.flatten(start_dim=2)  # (B, F, 21*6)

    # 9. Calculate hml263 linear velocity of all joints. Perform the same things at 4. But for all joints.
    if return_abs:
        hml263_abs_l_full = motion.flatten(start_dim=2)  # (B, F, 22*3)
    else:
        hml263_vel_l_full = joints_vel_relative.flatten(start_dim=2)  # (B, F, 22*3)

    # 10. Get foot contacts.
    feet_contact_labels = detect_foot_contact(
        motion, feet_thre, padding_strategy=velocity_padding_strategy
    )  # 2 * (B, F, 2)
    hml263_l_fc_labels, hml263_r_fc_labels = feet_contact_labels

    if return_abs:
        hml263_items = [
            hml263_abs_r,  # (B, F, 1), rotation about y-axis, i.e. turn around
            hml264_abs_l,  # (B, F, 2), linear forward translation
            hml263_pelvis_y,  # (B, F, 1)
            hml263_ric,  # (B, F, 21*3)
            hml263_rot,  # (B, F, 21*6), continuous 6D representation
            hml263_abs_l_full,  # (B, F, 22*3), linear velocity of forward, but all joints and also y-velocity
            hml263_l_fc_labels,  # (B, F, 2)
            hml263_r_fc_labels,  # (B, F, 2)
        ]
    else:
        hml263_items = [
            hml263_vel_r,  # (B, F, 1), rotation velocity about y-axis, i.e. turn around
            hml263_vel_l,  # (B, F, 2), linear velocity of forward direction
            hml263_pelvis_y,  # (B, F, 1)
            hml263_ric,  # (B, F, 21*3)
            hml263_rot,  # (B, F, 21*6), continuous 6D representation
            hml263_vel_l_full,  # (B, F, 22*3), linear velocity of forward, but all joints and also y-velocity
            hml263_l_fc_labels,  # (B, F, 2)
            hml263_r_fc_labels,  # (B, F, 2)
        ]
        # Log.debug(f"hml263_vel_r. {detectNaN(hml263_vel_r)}")
        # Log.debug(f"hml263_vel_l. {detectNaN(hml263_vel_l)}")
        # Log.debug(f"hml263_pelvis_y. {detectNaN(hml263_pelvis_y)}")
        # Log.debug(f"hml263_ric. {detectNaN(hml263_ric)}")
        # Log.debug(f"hml263_rot. {detectNaN(hml263_rot)}")
        # Log.debug(f"hml263_vel_l_full. {detectNaN(hml263_vel_l_full)}")
        # Log.debug(f"hml263_l_fc_labels. {detectNaN(hml263_l_fc_labels)}")
        # Log.debug(f"hml263_r_fc_labels. {detectNaN(hml263_r_fc_labels)}")
    hml263 = torch.concat(hml263_items, axis=2)  # (B, F, 1 + 2 + 1 + 21*3 + 21*6 + 22*3 + 2 + 2 = 263)
    hml263[mask_invalid_frames] = 0

    return hml263.transpose(1, 2)  # (B, 263, F)


def uniform_skeleton(positions):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = tgt_offsets.numpy()
    # print(src_offset)
    # print(tgt_offset)
    """Calculate Scale Ratio as the ratio of legs"""
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    """Inverse Kinematics"""
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    """Forward Kinematics"""
    src_skel.set_offset(tgt_offsets)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def convert_motion_to_hmlvec263_original(positions, feet_thre=0.002):
    positions = uniform_skeleton(positions)

    """Put on Floor"""
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     print(floor_height)

    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    """XZ at origin"""
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    """All initially face Z+"""
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init**2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    """New ground truth positions"""
    global_positions = positions.copy()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r

    #
    feet_l, feet_r = foot_detect(positions, feet_thre)

    """Quaternion and Cartesian representation"""
    r_rot = None

    def get_rifke(positions):
        """Local pose"""
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        """All pose face Z+"""
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        """Quaternion to continuous 6D"""
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        """Root Linear Velocity"""
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        """Root Angular Velocity"""
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    """Root height"""
    root_y = positions[:, 0, 1:2]

    """Root rotation and linear velocity"""
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    """Get Joint Rotation Representation"""
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    """Get Joint Rotation Invariant Position Represention"""
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    """Get Joint Velocity Representation"""
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1), global_positions[1:] - global_positions[:-1]
    )
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity
