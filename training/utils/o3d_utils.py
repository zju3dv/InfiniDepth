import numpy as np
import torch

try:
    import open3d as o3d
except:
    print("open3d not installed")
from .wis3d_utils import get_const_colors, color_schemes
from .skeleton_utils import SMPL_SKELETON, NBA_SKELETON, GYM_SKELETON, COLOR_NAMES
import training.utils.matrix as matrix
import smplx


STATE = {
    "play": True,
    "reset": False,
    "next": False,
    "back": False,
    "after": False,
    "prev": False,
    "iter_play": False,
}


COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

cot2_viz_pastel_color_schemes = {
    0: ([255, 168, 154], [216, 140, 172]),
    1: ([183, 255, 191], [239, 203, 157]),
    2: ([183, 255, 255], [114, 183, 156]),
    3: ([183, 255, 255], [148, 173, 210]),
    4: ([255, 183, 255], [189, 152, 216]),
}


def get_worldcoordinate_line_set():
    points = [
        [0, 0, 0],
        [3, 0, 0],
        [3, 3, 0],
        [0, 3, 0],
        [0, 0, 3],
    ]
    lines = [
        [0, 1],
        [3, 0],
        [0, 4],
    ]
    colors = ["red", "green", "blue"]
    colors = np.array([color_schemes[c][1] for c in colors]) / 255.0
    w_coord_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    w_coord_line_set.colors = o3d.utility.Vector3dVector(colors)
    return w_coord_line_set


def get_coordinate_mesh(mat=None, size=1.0):
    # mat: (4, 4)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if mat is not None:
        mesh.transform(mat)
    return mesh


def get_ground_mesh(pos, upaxis="y"):
    """_summary_

    Args:
        pos (tensor): (progress, T, J, 3)
        upaxis (str, optional): upward axis name. Defaults to "y".
    """
    # 0.02 for foot height
    foot_thresh = 0.02
    if upaxis == "y":
        lowest_value = pos[-1][..., 1].min()
        translation = np.array([-5.0, lowest_value - 1.0 - foot_thresh, -5.0])
        mesh = o3d.geometry.TriangleMesh.create_box(width=10, height=1, depth=10)
    elif upaxis == "z":
        lowest_value = pos[-1][..., 2].min()
        translation = np.array([-5.0, -5.0, lowest_value - 1.0 - foot_thresh])
        mesh = o3d.geometry.TriangleMesh.create_box(width=10, height=10, depth=1)

    else:
        lowest_value = pos[-1][..., 0].min()
        translation = np.array([lowest_value - 1.0 - foot_thresh, -5.0, -5.0])
        mesh = o3d.geometry.TriangleMesh.create_box(width=1, height=10, depth=10)
    # relative = False, translate the center of the mesh to the target position
    # mesh.translate(translation, relative=False)
    mesh.translate(translation)
    return mesh


def add_camera_line_set(w2c):
    # w2c: (4, 4)
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
    ]
    colors = ["red", "green", "blue"]
    colors = np.array([color_schemes[c][1] for c in colors]) / 255.0
    cam_line_pts = matrix.get_position_from(np.array(points), np.linalg.inv(w2c))

    camera_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(cam_line_pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    camera_line_set.colors = o3d.utility.Vector3dVector(colors)
    return camera_line_set


def play_callback(vis, key, action):
    print("play", "key", key, "action", action)
    STATE["play"] = True


def stop_callback(vis, key, action):
    print("stop", "key", key, "action", action)
    STATE["play"] = False


def reset_callback(vis, key, action):
    print("reset", "key", key, "action", action)
    STATE["reset"] = True


def next_callback(vis, key, action):
    print("next", "key", key, "action", action)
    if key == 1 or key == 2:
        STATE["next"] = True


def back_callback(vis, key, action):
    print("back", "key", key, "action", action)
    if key == 1 or key == 2:
        STATE["back"] = True


def after_callback(vis, key, action):
    print("after", "key", key, "action", action)
    if key == 1 or key == 2:
        STATE["after"] = True


def prev_callback(vis, key, action):
    print("prev", "key", key, "action", action)
    if key == 1 or key == 2:
        STATE["prev"] = True


def add_skeleton(pos, r=0.05, resolution=10, skeleton_type="smpl"):
    if skeleton_type == "smpl":
        skeleton = SMPL_SKELETON
    elif skeleton_type == "nba":
        skeleton = NBA_SKELETON
    elif skeleton_type == "gym":
        skeleton = GYM_SKELETON
    else:
        raise NotImplementedError

    kinematic_chain = [
        [skeleton["joints"].index(skeleton_name) for skeleton_name in sub_skeleton_names]
        for sub_skeleton_names in skeleton["kinematic_chain"]
    ]
    J = pos.shape[0]

    color_names = COLOR_NAMES[: len(kinematic_chain)]
    m_colors = []
    bones = []
    for chain, color_name in zip(kinematic_chain, color_names):
        num_line = len(chain) - 1
        color_ = get_const_colors(color_name, partial_shape=(num_line,), alpha=1.0)
        m_colors.append(color_[..., :3].cpu().detach().numpy())
        bones.append(np.stack((np.array(chain)[:-1], np.array(chain)[1:]), axis=-1))
    m_colors = np.concatenate(m_colors, axis=0)
    bones = np.concatenate(bones, axis=0)

    color_names = COLOR_NAMES
    joints_category = [
        [skeleton["joints"].index(skeleton_name) for skeleton_name in sub_skeleton_names]
        for sub_skeleton_names in skeleton["joints_category"]
    ]
    joints_mesh = []
    joints_color = []
    for i in range(J):
        for j, joints_ in enumerate(joints_category):
            if i in joints_:
                joints_color.append(color_schemes[color_names[j]][1])
                break
    joints_color = np.array(joints_color) / 255.0
    for i in range(J):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=resolution)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(joints_color[i])
        joints_mesh.append(mesh)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pos)
    line_set.lines = o3d.utility.Vector2iVector(bones)
    line_set.colors = o3d.utility.Vector3dVector(m_colors)
    return joints_mesh, line_set


def add_line(pos_s, pos_e):
    color_names = [i % 5 for i in range(len(pos_s))]
    m_colors = np.array([cot2_viz_pastel_color_schemes[c][1] for c in color_names]) / 255.0
    bones = [[i, i + len(pos_s)] for i in range(len(pos_s))]

    pos = np.concatenate((pos_s, pos_e), axis=0)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pos)
    line_set.lines = o3d.utility.Vector2iVector(bones)
    line_set.colors = o3d.utility.Vector3dVector(m_colors)
    return line_set


def pos_2dto3d(pos_2d, w2c, z=1, is_pinhole=True):
    # we assume pinhole camera
    pos_2d_shape = pos_2d.shape
    pos_2d = pos_2d.reshape(-1, 2)
    if is_pinhole:
        pos_3d = np.concatenate((pos_2d * z, z * np.ones((pos_2d.shape[0], 1))), axis=-1)
    else:
        pos_3d = np.concatenate((pos_2d, z * np.ones((pos_2d.shape[0], 1))), axis=-1)
    pos_3d = matrix.get_position_from(pos_3d, np.linalg.inv(w2c))
    pos_3d = pos_3d.reshape(pos_2d_shape[:-1] + (3,))
    return pos_3d


class o3d_skeleton_animation:
    def __init__(
        self, pos, pos_2d=None, w2c=None, pred_pos=None, is_pinhole=False, name="", upaxis="y", skeleton_type="smpl"
    ):
        """_summary_

        Args:
            NOTE: sometimes J may be >22 as we use virtual next frame root for global motions
            pos (tensor): (progress, T, J, 3) joints positions in world coordinate
            pos_2d (tensor): (progress, V, T*J, 2) 2d joints positions relative to each camera view
            pred_pos (tensor): (progress, V, T*J, 3) 3d joints positions directly predicted by LRM.
            w2c (tensor): (V, 4, 4) world2camera matrix (inverse of camera transformation)
            is_pinhole (bool): if true, we use perspective, otherwise orthogonal projection
            name (str, optional): text description of this motion sequence. Defaults to "".
            upaxis (str, optional): upward axis name. Defaults to "y".
            skeleton_type (str, optional): "smpl" or "nba". Defaults to "smpl".
        """
        self.pos = pos
        self.pos_2d = pos_2d
        self.pred_pos = pred_pos
        self.w2c = w2c
        self.is_pinhole = is_pinhole

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_action_callback(ord("P"), play_callback)  # Play animation
        vis.register_key_action_callback(ord("S"), stop_callback)  # Stop playing
        vis.register_key_action_callback(ord("R"), reset_callback)  # Reset
        vis.register_key_action_callback(ord("N"), next_callback)  # Next
        vis.register_key_action_callback(ord("B"), back_callback)  # Previous
        vis.register_key_action_callback(ord("U"), after_callback)  # Next ddpm iter
        vis.register_key_action_callback(ord("Y"), prev_callback)  # Previous ddpm iter

        vis.create_window(name)
        self.vis = vis
        self.iter_i = pos.shape[0] - 1
        self.time_i = 0
        vis.add_geometry(get_worldcoordinate_line_set())
        vis.add_geometry(get_coordinate_mesh(size=0.2))
        self._setup_3d_skeleton(pos, upaxis, skeleton_type)
        if pos_2d is not None:
            self._setup_2d_skeleton(pos_2d, w2c, is_pinhole, skeleton_type)
            self._setup_camera(w2c)
        if pred_pos is not None:
            self._setup_pred_3d_skeleton(pred_pos, skeleton_type)

        self.vis.register_animation_callback(self.animation_callback)
        self.vis.run()
        self.vis.destroy_window()

    def _setup_3d_skeleton(self, pos, upaxis, skeleton_type):
        if isinstance(pos, torch.Tensor):
            pos = torch.clone(pos).cpu().detach().numpy()
        J = pos.shape[-2]
        self.J = J
        pos = pos.reshape(pos.shape[0], -1, J, 3)  # N, T, J, 3
        self.skeleton = add_skeleton(pos[0, 0], skeleton_type=skeleton_type)  # Add 3d skeleton mesh

        self.vis.add_geometry(get_ground_mesh(pos, upaxis=upaxis))

        for j in range(self.J):
            self.vis.add_geometry(self.skeleton[0][j])
        self.vis.add_geometry(self.skeleton[1])
        self.pos = pos

    def _setup_2d_skeleton(self, pos_2d, w2c, is_pinhole, skeleton_type):
        if isinstance(pos_2d, torch.Tensor):
            pos_2d = torch.clone(pos_2d).cpu().detach().numpy()
        if isinstance(w2c, torch.Tensor):
            w2c = torch.clone(w2c).cpu().detach().numpy()
        pos_2d = pos_2d.reshape(pos_2d.shape[:2] + (-1, self.J, 2))  # N, V, T, J, 2
        self.cam_skeleton = []  # skeleton on camera plane, z = 0.25
        self.cam_skeleton_line = []  # line start from skeleton to z = 1.5 on camera plane
        self.cam_skeleton_pos = []  # start of line
        self.cam_skeleton_pos_end = []  # end of line
        # Add 2d skeleton mesh of each view
        for i in range(pos_2d.shape[1]):
            pos_3d_ = pos_2dto3d(pos_2d[:, i], w2c[i], z=0.25, is_pinhole=is_pinhole)  # N, T, J, 3
            self.cam_skeleton.append(add_skeleton(pos_3d_[0, 0], r=0.025, skeleton_type=skeleton_type))
            self.cam_skeleton_pos.append(pos_3d_[:, None])
            for j in range(self.J):
                self.vis.add_geometry(self.cam_skeleton[i][0][j])
            self.vis.add_geometry(self.cam_skeleton[i][1])

            pos_3d_e = pos_2dto3d(pos_2d[:, i], w2c[i], z=1.5, is_pinhole=is_pinhole)
            self.cam_skeleton_line.append(add_line(pos_3d_[0, 0], pos_3d_e[0, 0]))
            self.vis.add_geometry(self.cam_skeleton_line[i])
            self.cam_skeleton_pos_end.append(pos_3d_e[:, None])
        self.cam_skeleton_pos = np.concatenate(self.cam_skeleton_pos, axis=1)
        self.cam_skeleton_pos_end = np.concatenate(self.cam_skeleton_pos_end, axis=1)

    def _setup_camera(self, w2c):
        if isinstance(w2c, torch.Tensor):
            w2c = torch.clone(w2c).cpu().detach().numpy()
        for i, w2c_ in enumerate(w2c):
            self.vis.add_geometry(add_camera_line_set(w2c_))
            size = 0.5 if i == 0 else 0.2
            self.vis.add_geometry(get_coordinate_mesh(mat=np.linalg.inv(w2c_), size=size))

    def _setup_pred_3d_skeleton(self, pos, skeleton_type):
        if isinstance(pos, torch.Tensor):
            pos = torch.clone(pos).cpu().detach().numpy()
        pos = pos.reshape(pos.shape[:2] + (-1, self.J, 3))  # N, V, T, J, 3
        self.pred_skeleton = []
        for i in range(pos.shape[1]):
            self.pred_skeleton.append(add_skeleton(pos[0, i, 0], r=0.025, skeleton_type=skeleton_type))
            for j in range(self.J):
                self.vis.add_geometry(self.pred_skeleton[i][0][j])
            self.vis.add_geometry(self.pred_skeleton[i][1])
        self.pred_pos = pos

    def animation_callback(self, vis):
        if STATE["play"]:
            self.time_i += 1
            if self.time_i >= self.pos.shape[1]:
                self.time_i = 0
        if STATE["next"]:
            self.time_i += 1
            STATE["next"] = False
        if STATE["back"]:
            self.time_i -= 1
            STATE["back"] = False
        if STATE["reset"]:
            self.time_i = 0
            STATE["reset"] = False
        if STATE["after"]:
            self.iter_i += 1
            STATE["after"] = False
        if STATE["prev"]:
            self.iter_i -= 1
            STATE["prev"] = False
        print(f"STEP: {self.iter_i}, TIME: {self.time_i}")
        self.time_i = min(self.pos.shape[1] - 1, self.time_i)
        self.time_i = max(0, self.time_i)
        self.iter_i = min(self.pos.shape[0] - 1, self.iter_i)
        self.iter_i = max(0, self.iter_i)

        self.update_skeleton(self.pos[self.iter_i, self.time_i], self.skeleton)
        if self.pos_2d is not None:
            for i in range(self.cam_skeleton_pos.shape[1]):
                self.update_skeleton(self.cam_skeleton_pos[self.iter_i, i, self.time_i], self.cam_skeleton[i])
                self.update_line(
                    self.cam_skeleton_pos[self.iter_i, i, self.time_i],
                    self.cam_skeleton_pos_end[self.iter_i, i, self.time_i],
                    self.cam_skeleton_line[i],
                )
        if self.pred_pos is not None:
            for i in range(self.pred_pos.shape[1]):
                self.update_skeleton(self.pred_pos[self.iter_i, i, self.time_i], self.pred_skeleton[i])

        self.vis.poll_events()
        self.vis.update_renderer()

    def update_skeleton(self, pos, skeleton):
        for i in range(self.J):
            skeleton[0][i].translate(pos[i], relative=False)
            self.vis.update_geometry(skeleton[0][i])
        skeleton[1].points = o3d.utility.Vector3dVector(pos)
        self.vis.update_geometry(skeleton[1])

    def update_line(self, pos_s, pos_e, line):
        pos = np.concatenate((pos_s, pos_e), axis=0)
        line.points = o3d.utility.Vector3dVector(pos)
        self.vis.update_geometry(line)


def vis_smpl_forward_animation(transl, pose):
    points = [
        [0, 0, 0],
        [3, 0, 0],
        [3, 3, 0],
        [0, 3, 0],
        [0, 0, 3],
    ]
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 4],
    ]
    colors = [COLORS[0], COLORS[2], COLORS[2], COLORS[1], COLORS[2]]
    ground_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ground_line_set.colors = o3d.utility.Vector3dVector(colors)
    smpl_model = smplx.create(
        "./inputs/models",
        model_type="smplh",
        gender="male",
        num_betas=16,
        batch_size=1,
    )

    output = smpl_model(
        body_pose=pose[:1, 3:],
        global_orient=pose[:1, :3],
        transl=transl[:1],
        return_verts=True,
    )
    verts = output.vertices.detach().cpu().numpy()

    def play_callback(vis, key, action):
        print("play", "key", key, "action", action)
        STATE["play"] = True

    def stop_callback(vis, key, action):
        print("stop", "key", key, "action", action)
        STATE["play"] = False

    def reset_callback(vis, key, action):
        print("reset", "key", key, "action", action)
        STATE["reset"] = True
        STATE["play"] = False

    def next_callback(vis, key, action):
        print("next", "key", key, "action", action)
        STATE["next"] = True

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.register_key_action_callback(ord("P"), play_callback)
    vis.register_key_action_callback(ord("S"), stop_callback)
    vis.register_key_action_callback(ord("R"), reset_callback)
    vis.register_key_action_callback(ord("N"), next_callback)

    smpl_mesh = o3d.geometry.TriangleMesh()
    smpl_mesh.vertices = o3d.utility.Vector3dVector(verts[0])
    smpl_mesh.triangles = o3d.utility.Vector3iVector(smpl_model.faces)
    smpl_mesh.compute_vertex_normals()
    smpl_mesh.paint_uniform_color([0.3, 0.3, 0.3])

    vis.add_geometry(smpl_mesh)
    vis.add_geometry(ground_line_set)

    MAX_N = pose.shape[0]
    i = 0

    def animation_callback(vis):
        nonlocal i
        if STATE["play"]:
            i += 1
        if STATE["next"]:
            i += 1
            STATE["next"] = False
        if i >= MAX_N:
            i = MAX_N - 1
        if STATE["reset"]:
            i = 0
            STATE["reset"] = False
        output = smpl_model(
            body_pose=pose[i : i + 1, 3:],
            global_orient=pose[i : i + 1, :3],
            transl=transl[i : i + 1],
            return_verts=True,
        )
        verts = output.vertices.detach().cpu().numpy()
        smpl_mesh.vertices = o3d.utility.Vector3dVector(verts[0])
        smpl_mesh.compute_vertex_normals()
        smpl_mesh.paint_uniform_color([0.3, 0.3, 0.3])
        vis.update_geometry(smpl_mesh)
        vis.poll_events()
        vis.update_renderer()

    vis.register_animation_callback(animation_callback)
    vis.run()
    vis.destroy_window()
