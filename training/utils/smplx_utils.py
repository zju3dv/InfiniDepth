import torch
import torch.nn.functional as F
import numpy as np
import smplx
import pickle
from smplx import SMPL, SMPLX, SMPLXLayer
from training.utils.body_model import BodyModelSMPLH, BodyModelSMPLX
from pytorch3d.transforms import axis_angle_to_matrix

# fmt: off
SMPLH_PARENTS = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
                              16, 17, 18, 19, 20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
                              35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50])
# fmt: on


def make_smplx(type="neu_fullpose", **kwargs):
    if type == "neu_fullpose":
        model = smplx.create(
            model_path="inputs/models/smplx/SMPLX_NEUTRAL.npz", use_pca=False, flat_hand_mean=True, **kwargs
        )
    elif type == "rich-smplx":
        # https://github.com/paulchhuang/rich_toolkit/blob/main/smplx2images.py
        bm_kwargs = {
            "model_type": "smplx",
            "gender": kwargs.get("gender", "male"),
            "num_pca_comps": 12,
            "flat_hand_mean": False,
            # create_expression=True, create_jaw_pose=Ture
        }
        # A /smplx folder should exist under the model_path
        model = BodyModelSMPLX(model_path="inputs/checkpoints/body_models", **bm_kwargs)
    elif type == "rich-smplh":
        bm_kwargs = {
            "model_type": "smplh",
            "gender": kwargs.get("gender", "male"),
            "use_pca": False,
            "flat_hand_mean": True,
        }
        model = BodyModelSMPLH(model_path="inputs/models", **bm_kwargs)

    elif type in ["smplx-circle", "smplx-groundlink"]:
        # don't use hand
        bm_kwargs = {
            "model_path": "inputs/checkpoints/body_models",
            "model_type": "smplx",
            "gender": kwargs.get("gender"),
            "num_betas": 16,
            "num_expression": 0,
        }
        model = BodyModelSMPLX(**bm_kwargs)

    elif type == "smplx-motionx":
        layer_args = {
            "create_global_orient": False,
            "create_body_pose": False,
            "create_left_hand_pose": False,
            "create_right_hand_pose": False,
            "create_jaw_pose": False,
            "create_leye_pose": False,
            "create_reye_pose": False,
            "create_betas": False,
            "create_expression": False,
            "create_transl": False,
        }

        bm_kwargs = {
            "model_type": "smplx",
            "model_path": "inputs/checkpoints/body_models",
            "gender": "neutral",
            "use_pca": False,
            "use_face_contour": True,
            **layer_args,
        }
        model = smplx.create(**bm_kwargs)

    elif type == "smplx-samp":
        # don't use hand
        bm_kwargs = {
            "model_path": "inputs/checkpoints/body_models",
            "model_type": "smplx",
            "gender": kwargs.get("gender"),
            "num_betas": 10,
            "num_expression": 0,
        }
        model = BodyModelSMPLX(**bm_kwargs)

    elif type == "smplx-bedlam":
        # don't use hand
        bm_kwargs = {
            "model_path": "inputs/checkpoints/body_models",
            "model_type": "smplx",
            "gender": kwargs.get("gender"),
            "num_betas": 11,
            "num_expression": 0,
        }
        model = BodyModelSMPLX(**bm_kwargs)

    elif type in ["smplx-layer", "smplx-fit3d"]:
        # Use layer
        if type == "smplx-fit3d":
            assert (
                kwargs.get("gender") == "neutral"
            ), "smplx-fit3d use neutral model: https://github.com/sminchisescu-research/imar_vision_datasets_tools/blob/e8c8f83ffac23cc36adf8ec8d0fd1c55679484ef/util/smplx_util.py#L15C34-L15C34"

        bm_kwargs = {
            "model_path": "inputs/checkpoints/body_models/smplx",
            "gender": kwargs.get("gender"),
            "num_betas": 10,
            "num_expression": 10,
        }
        model = SMPLXLayer(**bm_kwargs)

    elif type == "smpl":
        bm_kwargs = {
            "model_path": "inputs/checkpoints/body_models/smpl",
            "gender": "neutral",
            "num_betas": 10,
            "create_body_pose": False,
            "create_betas": False,
            "create_global_orient": False,
            "create_transl": False,
        }
        bm_kwargs.update(kwargs)
        model = SMPL(**bm_kwargs)

    else:
        raise NotImplementedError

    return model


def load_parents(npz_path="models/smplx/SMPLX_NEUTRAL.npz"):
    smplx_struct = np.load("models/smplx/SMPLX_NEUTRAL.npz", allow_pickle=True)
    parents = smplx_struct["kintree_table"][0].astype(np.long)
    parents[0] = -1
    return parents


def load_smpl_faces(npz_path="models/smplh/SMPLH_FEMALE.pkl"):
    with open(npz_path, "rb") as f:
        smpl_model = pickle.load(f, encoding="latin1")
    faces = np.array(smpl_model["f"].astype(np.int64))
    return faces


def decompose_fullpose(fullpose, model_type="smplx"):
    assert model_type == "smplx"

    fullpose_dict = {
        "global_orient": fullpose[..., :3],
        "body_pose": fullpose[..., 3:66],
        "jaw_pose": fullpose[..., 66:69],
        "leye_pose": fullpose[..., 69:72],
        "reye_pose": fullpose[..., 72:75],
        "left_hand_pose": fullpose[..., 75:120],
        "right_hand_pose": fullpose[..., 120:165],
    }

    return fullpose_dict


def compose_fullpose(fullpose_dict, model_type="smplx"):
    assert model_type == "smplx"
    fullpose = torch.cat(
        [
            fullpose_dict[k]
            for k in [
                "global_orient",
                "body_pose",
                "jaw_pose",
                "leye_pose",
                "reye_pose",
                "left_hand_pose",
                "right_hand_pose",
            ]
        ],
        dim=-1,
    )
    return fullpose


def compute_R_from_kinetree(rot_mats, parents):
    """operation of lbs/batch_rigid_transform, focus on 3x3 R only
    Parameters
    ----------
    rot_mats: torch.tensor BxNx3x3
        Tensor of rotation matrices
    parents : torch.tensor BxN
        The kinematic tree of each object

    Returns
    -------
    R : torch.tensor BxNx3x3
        Tensor of rotation matrices
    """
    rot_mat_chain = [rot_mats[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(rot_mat_chain[parents[i]], rot_mats[:, i])
        rot_mat_chain.append(curr_res)

    R = torch.stack(rot_mat_chain, dim=1)
    return R


def compute_relR_from_kinetree(R, parents):
    """Inverse operation of lbs/batch_rigid_transform, focus on 3x3 R only
    Parameters
    ----------
    R : torch.tensor BxNx4x4 or BxNx3x3
        Tensor of rotation matrices
    parents : torch.tensor BxN
        The kinematic tree of each object

    Returns
    -------
    rot_mats: torch.tensor BxNx3x3
        Tensor of rotation matrices
    """
    R = R[:, :, :3, :3]

    Rp = R[:, parents]  # Rp[:, 0] is invalid
    rot_mats = Rp.transpose(2, 3) @ R
    rot_mats[:, 0] = R[:, 0]

    return rot_mats


def quat_mul(x, y):
    """
    Performs quaternion multiplication on arrays of quaternions

    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    # res = np.concatenate(
    #     [
    #         y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
    #         y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
    #         y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
    #         y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0,
    #     ],
    #     axis=-1,
    # )
    res = torch.cat(
        [
            y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
            y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
            y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
            y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0,
        ],
        axis=-1,
    )

    return res


def quat_inv(q):
    """
    Inverts a tensor of quaternions

    :param q: quaternion tensor
    :return: tensor of inverted quaternions
    """
    # res = np.asarray([1, -1, -1, -1], dtype=np.float32) * q
    res = torch.tensor([1, -1, -1, -1], device=q.device).float() * q
    return res


def quat_mul_vec(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).

    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    # t = 2.0 * np.cross(q[..., 1:], x)
    t = 2.0 * torch.cross(q[..., 1:], x)
    # res = x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)
    res = x + q[..., 0][..., None] * t + torch.cross(q[..., 1:], t)

    return res


def inverse_kinematics_motion(
    global_pos,
    global_rot,
    parents=SMPLH_PARENTS,
):
    """
    Args:
        global_pos : (B, T, J-1, 3)
        global_rot (q) : (B, T, J-1, 4)
        parents : SMPLH_PARENTS
    Returns:
        local_pos : (B, T, J-1, 3)
        local_rot (q) : (B, T, J-1, 4)
    """
    J = 22
    local_pos = quat_mul_vec(
        quat_inv(global_rot[..., parents[1:J], :]),
        global_pos - global_pos[..., parents[1:J], :],
    )
    local_rot = (quat_mul(quat_inv(global_rot[..., parents[1:J], :]), global_rot),)
    return local_pos, local_rot


def forward_kinematics_motion(
    root_orient,
    pose_body,
    trans,
    joints_zero,
    smplh_parents=SMPLH_PARENTS,
    rot_type="pose_body",
):
    """
    Args:
        root_orient : (B, T, 3) for `pose_body`, (B, T, 3, 3) for `R`
        pose_body : (B, T, (J-1)*3) for `pose_body`, (B, T, J-1, 3, 3) for `R`
        trans : (B, T, 3)
        joints_zero : (B, J, 3)
        rot_type: pose_body, R
    Returns:
        posed_joints: (B, T, J, 3)
        R_global: (B, T, J, 3, 3)
        A: (B, T, J, 4, 4)
    """
    J = joints_zero.shape[1]  # 22 for smplh
    B, T = root_orient.shape[:2]
    if rot_type == "pose_body":
        rot_aa = torch.cat([root_orient, pose_body], dim=-1).reshape(B, T, -1, 3)
        rot_mats = axis_angle_to_matrix(rot_aa)  # (B, T, J, 3, 3)
    elif rot_type == "R":
        rot_mats = torch.cat([root_orient[:, :, None], pose_body], dim=2)

    joints_zero = torch.unsqueeze(joints_zero, dim=-1)  # (B, J, 3, 1)
    rel_joints = joints_zero.clone()
    rel_joints[:, 1:] -= joints_zero[:, smplh_parents[1:J]]
    rel_joints = rel_joints[:, None].expand(-1, T, -1, -1, -1)  # (B, T, J, 3, 1)

    transforms_mat = transform_mat(rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).reshape(B * T, J, 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, J):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[smplh_parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)

    # --- Returns
    # The last column of the transformations contains the posed joints, in the cam
    # NOTE: FK adds trans
    posed_joints = transforms[:, :, :3, 3].reshape(B, T, -1, 3) + trans.unsqueeze(2)

    # The rot of each joint in the cam
    R_global = transforms[:, :, :3, :3].reshape(B, T, J, 3, 3)

    # Relative transform, for LBS
    joints_homogen = F.pad(joints_zero, [0, 0, 0, 1])  # (B, J, 3->4, 1)
    joints_homogen = joints_homogen[:, None].expand(-1, T, -1, -1, -1)  # (B, T, J, 4, 1)
    transforms = transforms.reshape(B, T, J, 4, 4)
    A = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0])

    return posed_joints, R_global, A


def transform_mat(R, t):
    """Creates a batch of transformation matrices
    Args:
        - R: Bx3x3 array of a batch of rotation matrices
        - t: Bx3x1 array of a batch of translation vectors
    Returns:
        - T: Bx4x4 Transformation matrix
    """
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def normalize_joints(joints):
    """
    Args:
        joints: (B, *, J, 3)
    """
    LR_hips_xy = joints[..., 2, [0, 1]] - joints[..., 1, [0, 1]]
    LR_shoulders_xy = joints[..., 17, [0, 1]] - joints[..., 16, [0, 1]]
    LR_xy = (LR_hips_xy + LR_shoulders_xy) / 2  # (B, *, J, 2)

    x_dir = F.pad(F.normalize(LR_xy, 2, -1), (0, 1), "constant", 0)  # (B, *, 3)
    z_dir = torch.zeros_like(x_dir)  # (B, *, 3)
    z_dir[..., 2] = 1
    y_dir = torch.cross(z_dir, x_dir, dim=-1)

    joints_normalized = (joints - joints[..., [0], :]) @ torch.stack([x_dir, y_dir, z_dir], dim=-1)
    return joints_normalized


@torch.no_grad()
def compute_Rt_af2az(joints, inverse=False):
    """Assume z coord is upward
    Args:
        joints: (B, J, 3), in the start-frame
    Returns:
        R_af2az: (B, 3, 3)
        t_af2az: (B, 3)
    """
    t_af2az = joints[:, 0, :].detach().clone()
    t_af2az[:, 2] = 0  # do not modify z

    LR_xy = joints[:, 2, [0, 1]] - joints[:, 1, [0, 1]]  # (B, 2)
    I_mask = LR_xy.pow(2).sum(-1) < 1e-4  # do not rotate, when can't decided the face direction
    x_dir = F.pad(F.normalize(LR_xy, 2, -1), (0, 1), "constant", 0)  # (B, 3)
    z_dir = torch.zeros_like(x_dir)
    z_dir[..., 2] = 1
    y_dir = torch.cross(z_dir, x_dir, dim=-1)
    R_af2az = torch.stack([x_dir, y_dir, z_dir], dim=-1)  # (B, 3, 3)
    R_af2az[I_mask] = torch.eye(3).to(R_af2az)

    if inverse:
        R_az2af = R_af2az.transpose(1, 2)
        t_az2af = -(R_az2af @ t_af2az.unsqueeze(2)).squeeze(2)
        return R_az2af, t_az2af
    else:
        return R_af2az, t_af2az


def finite_difference_forward(x, dim_t=1, dup_last=True):
    if dim_t == 1:
        v = x[:, 1:] - x[:, :-1]
        if dup_last:
            v = torch.cat([v, v[:, [-1]]], dim=1)
    else:
        raise NotImplementedError

    return v


def compute_joints_zero(betas, gender):
    """
    Args:
        betas: (16)
        gender: 'male' or 'female'
    Returns:
        joints_zero: (22, 3)
    """
    body_model = {
        "male": make_smplx(type="humor", gender="male"),
        "female": make_smplx(type="humor", gender="female"),
    }

    smpl_params = {
        "root_orient": torch.zeros((1, 3)),
        "pose_body": torch.zeros((1, 63)),
        "betas": betas[None],
        "trans": torch.zeros(1, 3),
    }
    joints_zero = body_model[gender](**smpl_params).Jtr[0, :22]
    return joints_zero
