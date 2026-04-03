import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .wis3d_utils import get_const_colors, color_schemes
from .skeleton_utils import SMPL_SKELETON, NBA_SKELETON, GYM_SKELETON, COLOR_NAMES
import training.utils.matrix as matrix


def get_kinematic_chain(skeleton_type="nba"):
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
    return kinematic_chain


def get_skeleton_lines(skeleton_connections, ax):
    connections = []
    color_names = COLOR_NAMES
    for i, skel_con in enumerate(skeleton_connections):
        for _ in range(len(skel_con) - 1):
            c = np.array(color_schemes[color_names[i]][1]) / 255.0
            (line,) = ax.plot([], [], "k-", color=c)
            connections.append(line)
    return connections


class plt_skeleton_animation:
    def __init__(self, pos, skeleton_type="nba"):
        """_summary_

        Args:
            NOTE: sometimes J may be >22 as we use virtual next frame root for global motions
            pos (tensor): (progress, T, J, 3) joints positions (x, y) and confidence
        """
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()
        self.pos = pos
        self.skeleton_type = skeleton_type
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        points = [plt.plot([], [], "o")[0] for _ in range(pos.shape[1])]
        self.skeleton_connections = get_kinematic_chain(skeleton_type)
        self.connections = get_skeleton_lines(self.skeleton_connections, ax)
        self.points = points

        self.paused = False
        self.current_frame = 0

        self.animation = FuncAnimation(fig, self.update, frames=pos.shape[0], init_func=self.plt_init, blit=True)
        fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.show()

    def plt_init(self):
        x_min, y_min, _ = self.pos.min(axis=(0, 1))
        x_max, y_max, _ = self.pos.max(axis=(0, 1))
        self.ax.set_xlim(x_min - 1, x_max + 1)
        self.ax.set_ylim(y_min - 1, y_max + 1)
        return *self.points, *self.connections

    def update(self, frame):
        x, y = self.pos[frame, :, 0], self.pos[frame, :, 1]
        confidence = self.pos[frame, :, 2]
        for i in range(self.pos.shape[1]):
            self.points[i].set_data(x[i], y[i])
            confidence[i] = min(confidence[i], 1.0)
            confidence[i] = max(confidence[i], 0.0)
            self.points[i].set_alpha(confidence[i])
            self.points[i].set_markersize(20 * confidence[i])
        i = 0
        for skel_con in self.skeleton_connections:
            for j in range(len(skel_con) - 1):
                a = skel_con[j]
                b = skel_con[j + 1]
                self.connections[i].set_data([x[a], x[b]], [y[a], y[b]])
                i += 1
        return *self.points, *self.connections

    def on_key(self, event):
        if event.key == "p":
            if self.paused:
                self.animation.event_source.start()
            else:
                self.animation.event_source.stop()
            self.paused = ~self.paused
