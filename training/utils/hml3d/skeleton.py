from .quaternion import *
import scipy.ndimage.filters as filters
from einops import rearrange
import torch.nn.functional as F
from scipy.ndimage._filters import _gaussian_kernel1d

from training.utils.logger import Log
from training.utils.debug_utils import detectNaN


class Skeleton(object):
    # To use torch, we need to record the gaussian kernel in advance.
    # kernel_smooth = _gaussian_kernel1d(sigma=20, order=0, radius=int(4 * 20 + 0.5))  # (161,)
    kernel_smooth = _gaussian_kernel1d(sigma=10, order=0, radius=int(4 * 10 + 0.5))  # (161,)
    kernel_smooth = rearrange(torch.from_numpy(kernel_smooth).float(), "k -> 1 1 k")  # (1, 1, K)

    def __init__(self, offset, kinematic_tree, device):
        self.device = device
        self._raw_offset_np = offset.numpy()
        self._raw_offset = offset.clone().detach().to(device).float()
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [0] * len(self._raw_offset)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j - 1]

    def njoints(self):
        return len(self._raw_offset)

    def offset(self):
        return self._offset

    def set_offset(self, offsets):
        self._offset = offsets.clone().detach().to(self.device).float()

    def kinematic_tree(self):
        return self._kinematic_tree

    def parents(self):
        return self._parents

    # joints (batch_size, joints_num, 3)
    def get_offsets_joints_batch(self, joints):
        assert len(joints.shape) == 3
        _offsets = self._raw_offset.expand(joints.shape[0], -1, -1).clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[:, i] = (
                torch.norm(joints[:, i] - joints[:, self._parents[i]], p=2, dim=1)[:, None] * _offsets[:, i]
            )

        self._offset = _offsets.detach()
        return _offsets

    # joints (joints_num, 3)
    def get_offsets_joints(self, joints):
        assert len(joints.shape) == 2
        _offsets = self._raw_offset.clone()
        for i in range(1, self._raw_offset.shape[0]):
            # print(joints.shape)
            _offsets[i] = torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0) * _offsets[i]

        self._offset = _offsets.detach()
        return _offsets

    # face_joint_idx should follow the order of right hip, left hip, right shoulder, left shoulder
    # joints (B, J, 3)
    def inverse_kinematics(self, joints, face_joint_idx, smooth_forward=True, use_original_edition=False):
        """Get the joint-relative quaternions of all joints.

        FYA: the returned quaternions are actually have two parts:
        1. quat_params[pelvis] means the rotation(to z+) of whole body about y-axis;
        2. all other quat_params means the rotation(to parent) of the bone relative to its parent bone;

        ### Args:
        - `joints`(torch.Tensor): ((B), F, J, 3), frames of the 3D coordinates of joints
        - `face_joint_idx`(list): (4,), the indices of joints that are used to determine the forward direction of the body
        - `smooth_forward`(bool): whether to smooth the forward direction, `True` by default
        - `use_original_edition` (bool): whether to use the original implementation which may have bug
        ### Returns:
        - `quat_params`(torch.Tensor): ((B), F, J, 4), frames of the joint-relative quaternions of all joints

        """
        assert len(face_joint_idx) == 4
        device = joints.device
        # support not batch input
        if len(joints.shape) == 3:
            joints = joints.unsqueeze(0)  # (1, F, J, 3)
        B, FR, J, _ = joints.shape
        joints = joints.reshape(B * FR, J, 3)

        pos_y_vec = torch.tensor([0, 1, 0]).to(device).float()  # (3,)
        pos_z_vec = torch.tensor([0, 0, 1]).to(device).float()  # (3,)

        # 1. Get forward vec.
        if use_original_edition:
            l_hip, r_hip, r_sdr, l_sdr = face_joint_idx
        else:
            r_hip, l_hip, r_sdr, l_sdr = face_joint_idx

        across_hip = joints[:, r_hip] - joints[:, l_hip]
        across_sdr = joints[:, r_sdr] - joints[:, l_sdr]
        cross_vec = across_hip + across_sdr  # (B*F, 3)

        # 1.2. Use cross product to get forward vec.
        forward_vec = torch.cross(pos_y_vec[None], cross_vec, axis=-1)  # (B*F, 3)
        forward_vec = forward_vec / (torch.norm(forward_vec, dim=-1, keepdim=True) + 1e-9)  # (B*F, 3)
        # ! TODO: this branch will change the initial forward direction away from the z+ axis!
        if smooth_forward:
            # correlate1d(input, weights, axis, output, mode, cval, 0)
            self.kernel_smooth = self.kernel_smooth.to(device)  # (1, 1, K)
            forward_vec = F.conv1d(
                rearrange(forward_vec, "(b f) d -> (b d) 1 f", b=B),
                self.kernel_smooth,
                padding="same",
            )
            forward_vec = rearrange(forward_vec, "(b d) 1 f -> (b f) d", b=B)

        # 2. Get quaternions.
        #    The quaternions of pelvis is relative to z+ axis while the quaternions
        #    of other joints are relative to previous on the kinematic chain.
        quat_params = torch.zeros(joints.shape[:-1] + (4,)).to(device)  # (B, J, 4)

        # 2.1. Get quaternions about z+ of root for each frame.
        forward_vec[..., 2] += 1e-9
        root_quat = qbetween(forward_vec, pos_z_vec[None])  # (B, 4)
        # TODO: in ideal situation, the root_quat is already towards z+, no need to do this.
        # But smooth_forward will destroy this property. But still force each frame's initial
        # root_quat to be towards z+.
        root_quat_batch = root_quat.reshape(B, FR, 4)
        root_quat_batch[:, 0] = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).to(device)
        root_quat = root_quat_batch.reshape(B * FR, 4)

        quat_params[:, 0] = root_quat

        # 2.2. Calculate the quaternions of other joints along the kinematic chain.
        for chain in self._kinematic_tree:
            # 2.2.1. Initialize the base quat of the chain.
            if use_original_edition:
                prev_absolute_quat = root_quat
            else:
                prev_absolute_quat = quat_params[:, chain[0]]

            for j in range(len(chain) - 1):
                # alias the joints index
                cur_j = chain[j]  # index of current joint
                nxt_j = chain[j + 1]  # index of next joint

                # 2.2.2. Get direction vector of bone in standard pose.
                if len(self._raw_offset.shape) == 2:
                    u = self._raw_offset[nxt_j]  # (3)
                    u = u[None]  # (1, 3), for broadcasting
                else:
                    u = self._raw_offset[:, nxt_j]  # (B, 3)
                    u = u[:, None].expand(-1, FR, -1).reshape(B * FR, 3)  # (B*F, 3)
                # 2.2.3. Get direction vector of bone in real pose.
                v = joints[:, nxt_j] - joints[:, cur_j]  # (B, 3)
                v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-9)  # normalization

                # 2.2.4. Get q-transformation from standard pose to real pose, i.e. absolute rotation.
                # Make undefined rotation stable
                ### FIXME: is this reasonable?
                ### v[torch.cross(u, v).abs().sum(-1) < 1e-9] += 1e-9
                v[torch.norm(v, dim=-1) < 1e-9] = u  # if too small(which might only possible when [0,0,0]), set it to u
                absolute_quat = qbetween(u, v)  # (B, 4)
                # 2.2.5. Get current quat relative to previous one, i.e. local rotation.
                relative_quat = qmul(qinv(prev_absolute_quat), absolute_quat)  # (B, 4)

                # 2.2.6. Update quat_params and propagate the absolute rotation.
                quat_params[:, nxt_j, :] = relative_quat
                prev_absolute_quat = absolute_quat

        return quat_params

    # face_joint_idx should follow the order of right hip, left hip, right shoulder, left shoulder
    # joints (B, J, 3)
    def inverse_kinematics_np(self, joints, face_joint_idx, smooth_forward=False):
        # Log.warn("This function is deserted! Use inverse_kinematics instead!")
        assert len(face_joint_idx) == 4
        """Get Forward Direction"""
        r_hip, l_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]
        # print(across1.shape, across2.shape)

        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = filters.gaussian_filter1d(forward, 20, axis=0, mode="nearest")
            # forward (batch_size, 3)

        """Get Root Rotation"""
        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

        """Inverse Kinematics"""
        quat_params = np.zeros(joints.shape[:-1] + (4,))  # (120, 22, 4)
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                # (B, 3), ref vec
                u = self._raw_offset_np[chain[j + 1]][np.newaxis, ...].repeat(joints.shape[0], axis=0)
                # (B, 3), real vec
                v = joints[:, chain[j + 1]] - joints[:, chain[j]]
                v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]
                rot_u_v = qbetween_np(u, v)

                R_loc = qmul_np(qinv_np(R), rot_u_v)

                quat_params[:, chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)  # this is equal to rot_u_v

        return quat_params

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        joints = torch.zeros(quat_params.shape[:-1] + (3,)).to(self.device)
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                # TODO: this is the difference between ours and the deserted np version
                R = quat_params[:, chain[0]]
            else:
                R = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(len(quat_params), -1).detach().to(self.device)
            for i in range(1, len(chain)):
                R = qmul(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot(R, offset_vec) + joints[:, chain[i - 1]]
        return joints

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics_np(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        # Log.warn("This function is deserted! Use inverse_kinematics instead!")
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(len(quat_params), axis=0)
            for i in range(1, len(chain)):
                R = qmul_np(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot_np(R, offset_vec) + joints[:, chain[i - 1]]
        return joints

    def forward_kinematics_cont6d_np(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(cont6d_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix_np(cont6d_params[:, 0])
            else:
                matR = np.eye(3)[np.newaxis, :].repeat(len(cont6d_params), axis=0)
            for i in range(1, len(chain)):
                matR = np.matmul(matR, cont6d_to_matrix_np(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]][..., np.newaxis]
                # print(matR.shape, offset_vec.shape)
                joints[:, chain[i]] = np.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i - 1]]
        return joints

    def forward_kinematics_cont6d(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            # skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        joints = torch.zeros(cont6d_params.shape[:-1] + (3,)).to(cont6d_params.device)
        joints[..., 0, :] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix(cont6d_params[:, 0])
            else:
                matR = torch.eye(3).expand((len(cont6d_params), -1, -1)).detach().to(cont6d_params.device)
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, cont6d_to_matrix(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]].unsqueeze(-1)
                # print(matR.shape, offset_vec.shape)
                joints[:, chain[i]] = torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i - 1]]
        return joints
