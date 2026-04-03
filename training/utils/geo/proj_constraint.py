import torch
import torch.nn.functional as F
from einops import rearrange, repeat, einsum


def constraint(w_p3d, Ts_w2c, c_p2d, mode="persp"):
    """
    Args:
        w_p3d torch.Tensor: (B, N, 3)
        Ts_w2c torch.Tensor: (B, 4, 4)
        c_p2d torch.Tensor:  (B, N, 2)
        mode str: 'persp' or 'ortho'
    Returns:
        w_p3d torch.Tensor: (B, N, 3)
    """
    # modeling the ray from p2d to the front view of camera
    # line_point + a * line_dir = point
    c_lp = F.pad(c_p2d, (0, 1), value=1)  # (B, N, 3), z=1 camera plane
    if mode == "ortho":
        c_ld = torch.zeros_like(c_lp)  # (B, N, 3)
        c_ld[..., 2] = 1
    else:
        assert mode == "persp"
        c_ld = c_lp.clone()

    # get w_lp and w_ld
    Ts_c2w_ = torch.inverse(Ts_w2c)  # (B, 4, 4)
    R_c2w_ = Ts_c2w_[:, :3, :3]  # (B, 3, 3)
    t_c2w_ = Ts_c2w_[:, :3, 3].unsqueeze(1)  # (B, 1, 3)
    w_lp = einsum(R_c2w_, c_lp, "b c d, b n d -> b n c") + t_c2w_  # (B, N, 3)
    w_ld = einsum(R_c2w_, c_ld, "b c d, b n d -> b n c")

    # Since, (lp + a * ld - p3d) * ld = 0
    # We have, (lp - p3d) * ld + a * ld * ld = 0
    # We have, a = (p3d - lp) * ld / (ld * ld)
    a = einsum(w_p3d - w_lp, w_ld, "b n c, b n c -> b n") / einsum(w_ld, w_ld, "b n c, b n c -> b n")

    # w_p3d = w_lp + a * w_ld
    w_p3d_ = w_lp + a.unsqueeze(-1) * w_ld  # (B, N, 3)

    ### DEBUG: start
    # from training.utils.wis3d_utils import make_wis3d, add_joints22_motion_as_lines
    # wis3d = make_wis3d(name='constraint')
    # w_p3d_init = w_p3d.reshape(-1, 120, 22, 3)[0]
    # w_p3d_after = w_p3d_.reshape(-1, 120, 22, 3)[0]
    # add_joints22_motion_as_lines(w_p3d_init, wis3d, name='w_p3d_init')
    # add_joints22_motion_as_lines(w_p3d_after, wis3d, name='w_p3d_after')
    # w_cam_obs = w_lp.reshape(-1, 120, 22, 3)[0]
    # add_joints22_motion_as_lines(w_cam_obs, wis3d, name='w_cam_obs')
    ### DEBUG: end

    return w_p3d_
