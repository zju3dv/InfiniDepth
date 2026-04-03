import torch
from .pos2rot import convert_pos_to_root_aarot
from .pos2smpl import convert_pos_to_smpl


def postrefine(w_p3d, refine_way, client=None):
    """_summary_

    Args:
        w_p3d (Tensor): [B, L, 22, 3]
    """
    B, L, J, _ = w_p3d.shape
    new_w_p3d = []
    for i in range(B):
        refine_pose = convert_pos_to_smpl(w_p3d[i])

        # ### smpl vis ###
        # from training.utils.o3d_utils import vis_smpl_forward_animation

        # pose = refine_pose.detach().cpu()
        # vis_smpl_forward_animation(pose[:, :3], pose[:, 3:])
        # ################

        refine_pos = client.send_data(refine_pose)
        refine_pos = refine_pos.reshape(1, L, J, 3)
        new_w_p3d.append(torch.tensor(refine_pos))
    new_w_p3d = torch.cat(new_w_p3d, dim=0)
    new_w_p3d = new_w_p3d.to(w_p3d.device)
    return new_w_p3d
