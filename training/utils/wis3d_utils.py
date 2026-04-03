from wis3d import Wis3D
from pathlib import Path
from datetime import datetime
import torch
import numpy as np


def make_wis3d(output_dir="outputs/wis3d", name="debug", time_postfix=False):
    """
    Make a Wis3D instance. e.g.:
        from training.utils.wis3d_utils import make_wis3d
        wis3d = make_wis3d(time_postfix=True)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if time_postfix:
        time_str = datetime.now().strftime("%m%d-%H%M-%S")
        name = f"{name}_{time_str}"
        print(f"Creating Wis3D {name}")
    wis3d = Wis3D(output_dir.absolute(), name)
    return wis3d


color_schemes = {
    "red": ([255, 168, 154], [153, 17, 1]),
    "green": ([183, 255, 191], [0, 171, 8]),
    "blue": ([183, 255, 255], [0, 0, 255]),
    "cyan": ([183, 255, 255], [0, 255, 255]),
    "magenta": ([255, 183, 255], [255, 0, 255]),
    "black": ([0, 0, 0], [0, 0, 0]),
    "orange": ([255, 183, 0], [255, 128, 0]),
}


def get_gradient_colors(scheme="red", num_points=120, alpha=1.0):
    """
    Return a list of colors that are gradient from start to end.
    """
    start_rgba = torch.tensor(color_schemes[scheme][0] + [255 * alpha]) / 255
    end_rgba = torch.tensor(color_schemes[scheme][1] + [255 * alpha]) / 255
    colors = torch.stack([torch.linspace(s, e, steps=num_points) for s, e in zip(start_rgba, end_rgba)], dim=-1)
    return colors


def get_const_colors(name="red", partial_shape=(120, 5), alpha=1.0):
    """
    Return colors (partial_shape, 4)
    """
    rgba = torch.tensor(color_schemes[name][1] + [255 * alpha]) / 255
    partial_shape = tuple(partial_shape)
    colors = rgba[None].repeat(*partial_shape, 1)
    return colors


# ================== Colored Motion Sequence ================== #


KINEMATIC_CHAINS = {
    "smpl22": [
        [0, 2, 5, 8, 11],  # right-leg
        [0, 1, 4, 7, 10],  # left-leg
        [0, 3, 6, 9, 12, 15],  # spine
        [9, 14, 17, 19, 21],  # right-arm
        [9, 13, 16, 18, 20],  # left-arm
    ],
    "h36m17": [
        [0, 1, 2, 3],  # right-leg
        [0, 4, 5, 6],  # left-leg
        [0, 7, 8, 9, 10],  # spine
        [8, 14, 15, 16],  # right-arm
        [8, 11, 12, 13],  # left-arm
    ],
    "coco17": [
        [12, 14, 16],  # right-leg
        [11, 13, 15],  # left-leg
        [4, 2, 0, 1, 3],  # replace spine with head
        [6, 8, 10],  # right-arm
        [5, 7, 9],  # left-arm
    ],
}


def add_motion_as_lines(motion, wis3d, name="joints22", skeleton_type="smpl22"):
    """
    Args:
        motion (tensor): (L, J, 3)
    """
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)
    kinematic_chain = KINEMATIC_CHAINS[skeleton_type]
    color_names = ["red", "green", "blue", "cyan", "magenta"]
    s_points = []
    e_points = []
    m_colors = []
    length = motion.shape[0]
    device = motion.device
    for chain, color_name in zip(kinematic_chain, color_names):
        num_line = len(chain) - 1
        s_points.append(motion[:, chain[:-1]])
        e_points.append(motion[:, chain[1:]])
        color_ = get_const_colors(color_name, partial_shape=(length, num_line), alpha=1.0).to(device)  # (120, 4, 4)
        m_colors.append(color_[..., :3] * 255)  # (120, 4, 3)
    s_points = torch.cat(s_points, dim=1)  # (L, ?, 3)
    e_points = torch.cat(e_points, dim=1)
    m_colors = torch.cat(m_colors, dim=1)

    for f in range(length):
        wis3d.set_scene_id(f)
        wis3d.add_lines(s_points[f], e_points[f], m_colors[f], name=name)


def add_prog_motion_as_lines(motion, wis3d, name="joints22", skeleton_type="smpl22"):
    """
    Args:
        motion (tensor): (P, L, J, 3)
    """
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)
    P, L, J, _ = motion.shape
    device = motion.device

    kinematic_chain = KINEMATIC_CHAINS[skeleton_type]
    color_names = ["red", "green", "blue", "cyan", "magenta"]
    s_points = []
    e_points = []
    m_colors = []
    for chain, color_name in zip(kinematic_chain, color_names):
        num_line = len(chain) - 1
        s_points.append(motion[:, :, chain[:-1]])
        e_points.append(motion[:, :, chain[1:]])
        color_ = get_gradient_colors(color_name, L, alpha=1.0).to(device)  # (L, 4)
        color_ = color_[None, :, None, :].repeat(P, 1, num_line, 1)  # (P, L, num_line, 4)
        m_colors.append(color_[..., :3] * 255)  # (P, L, num_line, 3)
    s_points = torch.cat(s_points, dim=-2)  # (L, ?, 3)
    e_points = torch.cat(e_points, dim=-2)
    m_colors = torch.cat(m_colors, dim=-2)

    s_points = s_points.reshape(P, -1, 3)
    e_points = e_points.reshape(P, -1, 3)
    m_colors = m_colors.reshape(P, -1, 3)

    for p in range(P):
        wis3d.set_scene_id(p)
        wis3d.add_lines(s_points[p], e_points[p], m_colors[p], name=name)


def add_joints_motion_as_spheres(joints, wis3d, radius=0.05, name="joints", label_each_joint=False):
    """Visualize skeleton as spheres to explore the skeleton.
    Args:
        joints: (NF, NJ, 3)
        wis3d
        radius: radius of the spheres
        name
        label_each_joint: if True, each joints will have a label in wis3d (then you can interact with it, but it's slower)
    """
    colors = torch.zeros_like(joints).float()
    n_frames = joints.shape[0]
    n_joints = joints.shape[1]
    for i in range(n_joints):
        colors[:, i, 1] = 255 / n_joints * i
        colors[:, i, 2] = 255 / n_joints * (n_joints - i)
    for f in range(n_frames):
        wis3d.set_scene_id(f)
        if label_each_joint:
            for i in range(n_joints):
                wis3d.add_spheres(
                    joints[f, i].float(),
                    radius=radius,
                    colors=colors[f, i],
                    name=f"{name}-j{i}",
                )
        else:
            wis3d.add_spheres(
                joints[f].float(),
                radius=radius,
                colors=colors[f],
                name=f"{name}",
            )
