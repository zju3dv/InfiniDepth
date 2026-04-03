import numpy as np
import torch
from pathlib import Path

from ..hml3d.skeleton import Skeleton
from ..hml3d.quaternion import *
from ..hml3d.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from training.utils.logger import Log
from training.utils.matrix import quat_wxyz2xyzw, quat_to_exp_map

# Lower legs
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1
joints_num = 22

t2m_raw_offsets = np.array(
    [
        [0, 0, 0],  # Pelvis
        [1, 0, 0],  # L_Hip
        [-1, 0, 0],  # R_Hip
        [0, 1, 0],  # Spine1
        [0, -1, 0],  # L_Knee
        [0, -1, 0],  # R_Knee
        [0, 1, 0],  # Spine2
        [0, -1, 0],  # L_Ankle
        [0, -1, 0],  # R_Ankle
        [0, 1, 0],  # Spine3
        [0, 0, 1],  # L_Toe
        [0, 0, 1],  # R_Toe
        [0, 1, 0],  # Neck
        [1, 0, 0],  # L_Collar
        [-1, 0, 0],  # R_Collar
        [0, 0, 1],  # Head
        [0, -1, 0],  # L_Shoulder
        [0, -1, 0],  # R_Shoulder
        [0, -1, 0],  # L_Elbow
        [0, -1, 0],  # R_Elbow
        [0, -1, 0],  # L_Wrist
        [0, -1, 0],  # R_Wrist
    ]
)

n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain


def convert_pos_to_root_aarot(motion):
    """
    # BUG: Does not work
    ### Args:
    - `motion` (torch.Tensor): (F, J=22, 3), batch of joints position of each frame
    ### Returns:
    - `pose`: (F, 3 + J*3)
    """
    assert isinstance(motion, torch.Tensor), "The input `position` should be a torch Tensor!"
    motion = motion.detach().cpu().clone()
    device = motion.device
    F, J, _ = motion.shape

    # 1. Standardize the motion.
    # no need to do this cause we already do this before conversion
    # motion = standardize_motion(motion)

    # 2. Convert absolute xyz expression to joint-relative quaternion.
    skel = Skeleton(n_raw_offsets, kinematic_chain, device)
    # BUG: Does not work
    quat_params = skel.inverse_kinematics(motion, face_joint_indx, smooth_forward=True)  # (B*F, J=22, 4)
    quat_params = quat_params.reshape(F, J, 4)
    root = motion[:, 0, :]
    quat_params = quat_wxyz2xyzw(quat_params)
    pose_aa = quat_to_exp_map(quat_params)
    pose_aa = pose_aa.reshape(F, -1)
    pose = torch.cat([root, pose_aa], dim=-1)

    return pose
