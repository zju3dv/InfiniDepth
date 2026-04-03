import numpy as np
import torch
from pathlib import Path

from .skeleton import Skeleton
from .quaternion import *
from .paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from training.utils.logger import Log

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


def standardize_motion(motion):
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

    # 1. Put the motion on the xz plane.
    # this shouldn't be use in our method, cause we have floor in the scene.
    # floor_y = motion[:, :, 1].min()
    # motion[:, :, 1] -= floor_y

    # 2. Make the motion starts at (0, y, 0).
    xz_mask = torch.tensor([1, 0, 1], dtype=torch.float32).to(device)
    xz_offset = motion[0, 0] * xz_mask  # filter the xz of the first pelvis position
    motion -= xz_offset

    # 3. Make the motion face z+ at the first frame.
    # 3.1. Get helper vector.
    pos_z_vec = torch.tensor([0, 0, 1], dtype=torch.float32).to(device)
    pos_y_vec = torch.tensor([0, 1, 0], dtype=torch.float32).to(device)
    # 3.2. Use cross product to calculate the current forward direction of the initial pose.
    r_hip, l_hip, r_sdr, l_sdr = face_joint_indx  # use hip and shoulder to get the cross vector
    cross_hip = motion[0, r_hip] - motion[0, l_hip]
    cross_sdr = motion[0, r_sdr] - motion[0, l_sdr]
    cross_vec = cross_hip + cross_sdr  # (3, )
    forward_vec = torch.cross(pos_y_vec, cross_vec, dim=-1)
    forward_vec = forward_vec / torch.norm(forward_vec, dim=-1, keepdim=True)
    # 3.3. Get the transformation quaternion from the current forward direction to z+ axis.
    quat_trans = qbetween(forward_vec, pos_z_vec).squeeze()  # (4,)
    quat_trans = quat_trans[None, None].expand(F, 22, 4)  # (F, 22, 4)
    # 3.4. Perform the in-place transformation.
    motion = qrot(quat_trans, motion)
    # Log.debug(quat_trans)

    return motion


def motion2poses(motion, rot2z):
    """Convert the motion to spacewalk pose, i.e. remove all movement and rotation.

    ### Args:
    - `motion`(torch.Tensor): (F, J=22, 3), joints position of each frame
    - `rot2z`(torch.Tensor): (F, 4), rotation from current face direction to z+
    ### Returns:
    - `poses` (torch.Tensor): (F, J=22, 3), joints position of each pose
    """
    F, J, _ = motion.shape
    device = motion.device

    # 1. Remove translation.
    pelvises = motion[:, 0]  # (F, 3)
    xz_mask = torch.tensor([1, 0, 1], dtype=torch.float32).to(device)  # (3,)
    xz_offset = pelvises * xz_mask  # (F, 3), filter the xz of the first pelvis position
    pose = motion - xz_offset[:, None]  # (F, J=22, 3)

    # 2. Remove rotation about y-axis, i.e. never turn around.
    joints_rot2z = rot2z[:, None].expand(F, J, 4)  # (F, J, 4)
    pose = qrot(joints_rot2z, pose)

    return pose


def detect_foot_contact(motion, thre):
    """Label if the foot contact the floor.

    If the movement is large enough, it will be 1.0, otherwise 0.0.
    # TODO: Is this really ok? What if the movement is obvious? It will always be 0?

    ### Args:
    - `motion`(torch.Tensor): (F, J=22, 3), joints position of each frame
    - `thre`(float): threshold factor to detect the foot contact the floor
    ### Returns:
    - `l_fc_labels`(torch.Tensor): (F-1, 2), double foot contact labels of left foot
    - `r_fc_labels`(torch.Tensor): (F-1, 2), double foot contact labels of right foot
    """
    F, J, _ = motion.shape
    vel_factor = torch.tensor([thre, thre]).expand(F - 1, 2)

    feet_l_xyz = motion[1:, fid_l] - motion[:-1, fid_l]  # (F-1, 2, 3)
    feet_l_l2dis = torch.norm(feet_l_xyz, dim=-1)  # (F-1, 2)
    feet_l = (feet_l_l2dis**2 < vel_factor).float()  # (F-1, 2)
    feet_r_xyz = motion[1:, fid_r] - motion[:-1, fid_r]  # (F-1, 3)
    feet_r_l2dis = torch.norm(feet_r_xyz, dim=-1)  # (F-1, 1)
    feet_r = (feet_r_l2dis**2 < vel_factor).float()

    return feet_l, feet_r


def convert_single_motion_to_hmlvec263(motion, feet_thre=0.002, use_original_edition=False):
    """
    ### Args:
    - `motion` (torch.Tensor): (F, J=22, 3), joints position of each frame
    - `feet_threshold` (float): threshold to detect whether the foot contact the floor
    - `original_edition` (bool): whether to use the original edition
    ### Returns:
    - `hmlvec263`: (263, F-1)
    """
    assert isinstance(motion, torch.Tensor), "The input `position` should be a torch Tensor!"
    device = motion.device
    motion = motion.detach().clone()
    F, J, _ = motion.shape

    # 1. Standardize the motion.
    # no need to do this cause we already do this before conversion
    # motion = standardize_motion(motion)

    # 2. Convert absolute xyz expression to joint-relative quaternion.
    skel = Skeleton(n_raw_offsets, kinematic_chain, device)
    quat_params = skel.inverse_kinematics(
        motion, face_joint_indx, smooth_forward=True, use_original_edition=use_original_edition
    )  # (F, 22, 4)

    # ================================================================================ #
    # Tips: things below are not a linear logic, check the program flow to get clearly #
    #       understanding.                                                             #
    # ================================================================================ #

    # 3. Calculate hml263 rotation velocity.
    # we use pelvis's quaternion to express the body's direction(the quat describe the rotation from now to z+)
    pelvis_rot2z = quat_params[:, 0]  # (F, 4)
    pelvis_rot2prev = qmul(pelvis_rot2z[1:], qinv(pelvis_rot2z[:-1]))  # (F-1, 4), rotation from i+1 to i
    # in `pelvis_rot2prev`, w^2 + y^2 = 1, x = z = 0, so `asin(y)` is enough to recover the quaternion
    hml263_vel_r = torch.asin(pelvis_rot2prev[:, 2:3])  # (F-1, 1)

    # 4. Calculate hml263 linear velocity at xz-plane. It's velocity without rotation, i.e. the
    #    forward velocity. This part is quite alike to 9., which need all the velocity instead of
    #    only xz-velocity of pelvis we need here.
    joints_vel = motion[1:] - motion[:-1]  # (F-1, 22, 3)
    body_rot2z = pelvis_rot2z[:, None].expand(F, J, 4)  # (F, 22, 4)
    if use_original_edition:
        forward_vel = qrot(body_rot2z[1:], joints_vel)  # (F-1, 22, 3)
        hml263_vel_l = forward_vel[:, 0, [0, 2]]  # (F-1, 2), only xz-velocity of pelvis is needed here
        forward_vel = qrot(body_rot2z[:-1], joints_vel)  # (F-1, 22, 3), use different rot matrix
    else:
        forward_vel = qrot(body_rot2z[:-1], joints_vel)  # (F-1, 22, 3)
        hml263_vel_l = forward_vel[:, 0, [0, 2]]  # (F-1, 2), only xz-velocity of pelvis is needed here

    # 5. After getting the rotation information, we can get the pure poses which has nothing to do
    #    with rotation and translation, i.e. the spacewalk pose.
    poses = motion2poses(motion, pelvis_rot2z)  # (F, 22, 3)

    # 6. Get pelvis's height.
    pelvis_y = poses[:-1, 0, 1]  # (F-1,)
    hml263_pelvis_y = pelvis_y[:, None]  # (F-1, 1)

    # 7. Get rotation invariant coordinate, i.e. ric. That's what contained by `poses`.
    # remove last frame & remove pelvis & flatten
    ric = poses[:-1, 1:]  # (F-1, 21, 3)
    hml263_ric = ric.flatten(start_dim=1)  # (F-1, 21*3)

    # 8. Get continuous 6D representation.
    cont6d = quaternion_to_cont6d(quat_params)  # (F, 22, 6)
    # remove last frame & remove pelvis & flatten
    rot = cont6d[:-1, 1:]
    hml263_rot = rot.flatten(start_dim=1)  # (F-1, 21*6)

    # 9. Calculate hml263 linear velocity of all joints. Perform the same things at 4. But for all joints.
    hml263_vel_l_full = forward_vel.flatten(start_dim=1)  # (F-1, 22*3)

    # 10. Get foot contacts.
    feet_contact_labels = detect_foot_contact(motion, feet_thre)  # 2 * (F-1, 2)
    hml263_l_fc_labels, hml263_r_fc_labels = feet_contact_labels

    hml263_items = [
        hml263_vel_r,  # (F-1, 1), rotation velocity about y-axis, i.e. turn around
        hml263_vel_l,  # (F-1, 2), linear velocity of forward direction
        hml263_pelvis_y,  # (F-1, 1)
        hml263_ric,  # (F-1, 21*3)
        hml263_rot,  # (F-1, 21*6), continuous 6D representation
        hml263_vel_l_full,  # (F-1, 22*3), linear velocity of forward, but all joints and also y-velocity
        hml263_l_fc_labels,  # (F-1, 2)
        hml263_r_fc_labels,  # (F-1, 2)
    ]
    hml263 = torch.concat(hml263_items, axis=1)  # (F-1, 1 + 2 + 1 + 21*3 + 21*6 + 22*3 + 2 + 2 = 263)
    hml263 = torch.cat([hml263, hml263[[-1]].clone()], dim=0)  # (F, 263), padding by append the last frame again
    return hml263.transpose(0, 1)  # (263, F-1)


# This functions should be deprecated after the fully torch version is implemented.
def convert_single_motion_to_hmlvec263_np(positions, feet_thre=0.002):
    """
    Args:
        positions, torch.Tensor: (F, J=22, 3)
    Returns:
        hmlvec263: (263, F)
        # positions: (F, J, 3), local pose (-xz, face z+)  # l_velocity: (F-1, 2)
    """
    assert isinstance(positions, torch.Tensor)
    device = positions.device
    positions = positions.detach().cpu().numpy().copy()  # prevent in-place modification

    # """Uniform Skeleton"""
    # def uniform_keleton():
    #     #? comments: it seems this function is not necessary
    #     #! It's heavy!
    #     positions = uniform_skeleton(positions, tgt_offsets)
    # uniform_keleton()

    # """✅ Put on Floor"""
    # floor_height = positions.min(axis=0).min(axis=0)[1]
    # positions[:, :, 1] -= floor_height
    # #     print(floor_height)
    # #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)
    # """✅ XZ at origin"""
    # root_pos_init = positions[0]
    # root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    # positions = positions - root_pose_init_xz
    # # '''Move the first pose to origin '''
    # # root_pos_init = positions[0]
    # # positions = positions - root_pos_init[0]

    # """✅ All initially face Z+"""
    # r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    # across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    # across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    # across = across1 + across2
    # across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]
    # # get the current face direction
    # # forward (3,), rotate around y-axis
    # forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # # forward (3, ,)
    # forward_init = forward_init / np.sqrt((forward_init**2).sum(axis=-1))[..., np.newaxis]
    # # Z+ unit vector
    # target = np.array([[0, 0, 1]])

    # # we want forward_init be like target through rotation
    # # xyz(3,) to quaternion(4,)
    # root_quat_init = qbetween_np(forward_init, target)
    # root_quat_init = root_quat_init / np.sqrt((root_quat_init**2).sum(axis=-1))[..., np.newaxis]
    # root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    # positions_b = positions.copy()
    # positions = qrot_np(root_quat_init, positions)

    """✅ New ground truth positions"""
    global_positions = positions.copy()

    """✅ Get Foot Contacts """

    def foot_detect(positions, thres):
        # get movements delta each frame
        # if the movements is large enough, it will be 1.0 otherwise 0.0

        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        # feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    """Quaternion and Cartesian representation"""

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        """✅ Quaternion to continuous 6D"""
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        """✅ Root Linear Velocity"""
        # (seq_len - 1, 3) #TODO: should positions here be from rifke?
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        """Root Angular Velocity"""
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)

    def get_rifke(positions):
        """✅ Local pose"""
        # remove "movement"
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        """✅ All pose face Z+"""
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    positions = get_rifke(positions)

    # r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    """Root height"""
    root_y = positions[:, 0, 1:2]

    """Root rotation and linear velocity"""
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)  # 1 + 2 + 1

    """Get Joint Rotation Invariant Position Represention"""
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    """Get Joint Rotation Representation"""
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    """Get Joint Velocity Representation"""
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1), global_positions[1:] - global_positions[:-1]
    )
    local_vel = local_vel.reshape(len(local_vel), -1)
    hmlvec263 = np.concatenate(
        [root_data, ric_data[:-1], rot_data[:-1], local_vel, feet_l, feet_r],
        axis=-1,
    )

    # TODO: make this function fully torch
    hmlvec263 = torch.from_numpy(hmlvec263).to(device).permute(1, 0)  # (263, F-1)
    hmlvec263 = torch.cat([hmlvec263, hmlvec263[:, [-1]].clone()], dim=-1)  # (263, F)

    return hmlvec263  # , positions, l_velocity
