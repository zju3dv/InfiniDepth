import time
import torch
import imageio
import numpy as np
from einops import rearrange
from skimage.draw import line
from torch.nn import functional as F
from easydict import EasyDict as edict
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer import NDCMultinomialRaysampler, ray_bundle_to_ray_points


def compute_epipolar_mask(src_frame, tgt_frame, imh, imw, dialate_mask=True, debug_depth=False, visualize_mask=False):
    """
    src_frame: source frame containing camera
    tgt_frame: target frame containing camera
    debug_depth: if True, uses depth map to compute epipolar lines on target image (debugging)
    visualize_mask: if True, saves a batched attention masks (debugging)
    """

    # generates raybundle using camera intrinsics and extrinsics
    src_ray_bundle = NDCMultinomialRaysampler(
        image_width=imw,
        image_height=imh,
        n_pts_per_ray=1,
        min_depth=1.0,
        max_depth=1.0,
    )(src_frame.camera)
    
    src_depth = getattr(src_frame, "depth_map", None)
    if debug_depth and src_depth is not None:
        src_depth = src_depth[:, 0, ..., None]
        src_depth[src_depth >= 100] = 100 # clip depth
    else:
        # get points in world space (at fixed depth)
        src_depth = 500. * torch.ones((1, imh, imw, 1), dtype=torch.float32, device=src_frame.camera.device)

    pts_world = ray_bundle_to_ray_points(src_ray_bundle._replace(lengths=src_depth)).squeeze(-2)

    # move source points to target screen space
    tgt_pts_screen = tgt_frame.camera.transform_points_screen(pts_world.squeeze(), image_size=(imh, imw))

    # move source camera center to target screen space
    src_center_tgt_screen = tgt_frame.camera.transform_points_screen(src_frame.camera.get_camera_center(), image_size=(imh, imw)).squeeze()

    # build epipolar mask (draw lines from source camera center to source points in target screen space)
    # start: source camera center, end: source points in target screen space

    # get flow of points 
    center_to_pts_flow = tgt_pts_screen[...,:2] - src_center_tgt_screen[...,:2]

    # normalize flow
    center_to_pts_flow = center_to_pts_flow / center_to_pts_flow.norm(dim=-1, keepdim=True)

    # get slope and intercept of lines
    slope = center_to_pts_flow[:,:,0:1] / center_to_pts_flow[:,:,1:2]
    intercept = tgt_pts_screen[:,:, 0:1] - slope * tgt_pts_screen[:,:, 1:2]

    # find intersection of lines with tgt screen (x = 0, x = imw, y = 0, y = imh)
    left = slope * 0 + intercept
    left_sane = (left <= imh) & (0 <= left)
    left = torch.cat([left, torch.zeros_like(left)], dim=-1)

    right = slope * imw + intercept
    right_sane = (right <= imh) & (0 <= right)
    right = torch.cat([right, torch.ones_like(right) * imw], dim=-1)

    top = (0 - intercept) / slope
    top_sane = (top <= imw) & (0 <= top)
    top = torch.cat([torch.zeros_like(top), top], dim=-1)

    bottom = (imh - intercept) / slope
    bottom_sane = (bottom <= imw) & (0 <= bottom)
    bottom = torch.cat([torch.ones_like(bottom) * imh, bottom], dim=-1)

    # find intersection of lines
    points_one = torch.zeros_like(left)
    points_two = torch.zeros_like(left)

    # collect points from [left, right, bottom, top] in sequence
    points_one = torch.where(left_sane.repeat(1,1,2), left, points_one)

    points_one_zero = (points_one.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_one = torch.where(right_sane.repeat(1,1,2) & points_one_zero, right, points_one)

    points_one_zero = (points_one.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_one = torch.where(bottom_sane.repeat(1,1,2) & points_one_zero, bottom, points_one)

    points_one_zero = (points_one.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_one = torch.where(top_sane.repeat(1,1,2) & points_one_zero, top, points_one)

    # collect points from [top, bottom, right, left] in sequence (opposite)
    points_two = torch.where(top_sane.repeat(1,1,2), top, points_two)

    points_two_zero = (points_two.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_two = torch.where(bottom_sane.repeat(1,1,2) & points_two_zero, bottom, points_two)

    points_two_zero = (points_two.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_two = torch.where(right_sane.repeat(1,1,2) & points_two_zero, right, points_two)

    points_two_zero = (points_two.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_two = torch.where(left_sane.repeat(1,1,2) & points_two_zero, left, points_two)

    # if source point lies inside target screen (find only one intersection)
    if (imh >= src_center_tgt_screen[0] >= 0) and (imw >= src_center_tgt_screen[1] >= 0):
        points_one_flow = points_one - src_center_tgt_screen[:2]
        points_one_flow_direction = (points_one_flow > 0)

        points_two_flow = points_two - src_center_tgt_screen[:2]
        points_two_flow_direction = (points_two_flow > 0)

        orig_flow_direction = (center_to_pts_flow > 0)

        # if flow direction is same as orig flow direction, pick points_one, else points_two
        points_one_alinged = (points_one_flow_direction == orig_flow_direction).all(dim=-1).unsqueeze(-1).repeat(1,1,2)
        points_one = torch.where(points_one_alinged, points_one, points_two)

        # points two is source camera center
        points_two = points_two * 0 + src_center_tgt_screen[:2]
    
    # if debug terminate with depth 
    if debug_depth:
        # remove points that are out of bounds (in target screen space)
        tgt_pts_screen_mask = (tgt_pts_screen[...,:2] < 0) | (tgt_pts_screen[...,:2] > imh)
        tgt_pts_screen_mask = ~tgt_pts_screen_mask.any(dim=-1, keepdim=True)

        depth_dist = torch.norm(src_center_tgt_screen[:2] - tgt_pts_screen[...,:2], dim=-1, keepdim=True)
        points_one_dist = torch.norm(src_center_tgt_screen[:2] - points_one, dim=-1, keepdim=True)
        points_two_dist = torch.norm(src_center_tgt_screen[:2] - points_two, dim=-1, keepdim=True)

        # replace where reprojected point is closer to source camera on target screen
        points_one = torch.where((depth_dist < points_one_dist) & tgt_pts_screen_mask, tgt_pts_screen[...,:2], points_one)
        points_two = torch.where((depth_dist < points_two_dist) & tgt_pts_screen_mask, tgt_pts_screen[...,:2], points_two)

    # build epipolar mask
    attention_mask = torch.zeros((imh * imw, imh, imw), dtype=torch.bool, device=src_frame.camera.device)

    # quantize points to pixel indices
    points_one = (points_one - 0.5).reshape(-1,2).long().numpy()
    points_two = (points_two - 0.5).reshape(-1,2).long().numpy()

    # iterate over points_one and points_two together and draw lines
    for idx, (p1, p2) in enumerate(zip(points_one, points_two)):
        # skip out of bounds points
        if p1.sum() == 0 and p2.sum() == 0:
            continue
        
        if not dialate_mask:
            # draw line from p1 to p2
            rr, cc = line(int(p1[1]), int(p1[0]), int(p2[1]), int(p2[0]))
            rr, cc = rr.astype(np.int32), cc.astype(np.int32)
            attention_mask[idx, rr, cc] = True
        else:
            # draw lines with mask dilation (from all neighbors of p1 to neighbors of p2)
            rrs, ccs = [], []
            for dx, dy in [(0,0), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]:  # 8 neighbors
                _p1 = [min(max(p1[0] + dy, 0), imh - 1), min(max(p1[1] + dx, 0), imw - 1)]
                _p2 = [min(max(p2[0] + dy, 0), imh - 1), min(max(p2[1] + dx, 0), imw - 1)]
                rr, cc = line(int(_p1[1]), int(_p1[0]), int(_p2[1]), int(_p2[0]))
                # Clip the coordinate to range [[0, 0], [H-1, W-1]]
                rr, cc = np.clip(rr, 0, imh - 1), np.clip(cc, 0, imw - 1)
                rrs.append(rr); ccs.append(cc)
            rrs, ccs = np.concatenate(rrs), np.concatenate(ccs)
            attention_mask[idx, rrs.astype(np.int32), ccs.astype(np.int32)] = True

    # reshape to (imh, imw, imh, imw)
    attention_mask = attention_mask.reshape(imh * imw, imh * imw)

    # stores flattened 2D attention mask 
    if visualize_mask:
        attention_mask = attention_mask.reshape(imh * imw, imh * imw)
        am_img = (attention_mask.squeeze().unsqueeze(-1).repeat(1,1,3).float().numpy() * 255).astype(np.uint8)
        imageio.imsave("batched_mask.png", am_img)

    return attention_mask


def compute_plucker_embed(frame, imw, imh):
    """ Computes Plucker coordinates for a Pytorch3D camera. """

    # get camera center
    cam_pos = frame.camera.get_camera_center()

    # get ray bundle
    src_ray_bundle = NDCMultinomialRaysampler(
        image_width=imw,
        image_height=imh,
        n_pts_per_ray=1,
        min_depth=1.0,
        max_depth=1.0,
    )(frame.camera)
    
    # get ray directions
    ray_dirs = F.normalize(src_ray_bundle.directions, dim=-1)

    # get plucker coordinates
    cross = torch.cross(cam_pos[:,None,None,:], ray_dirs, dim=-1)
    plucker = torch.cat((ray_dirs, cross), dim=-1)
    plucker = plucker.permute(0, 3, 1, 2)

    return plucker  # (B, 6, H, W, )


def compue_mask_plucker(tar_ext: torch.Tensor, src_ext: torch.Tensor, tar_ixt: torch.Tensor, src_ixt: torch.Tensor, Hz: int, Wz: int,
                        dialate_mask=True, debug_depth=False, visualize_mask=False):

    # Get pytorch3d frames (blender to opencv, then opencv to pytorch3d)
    tar_R, tar_T, tar_K = tar_ext[:, :3, :3], tar_ext[:, :3, 3], tar_ixt
    tar_c = cameras_from_opencv_projection(tar_R, tar_T, tar_K, torch.tensor([Hz, Wz]).float().unsqueeze(0))
    tar_cam = edict(camera=tar_c)

    src_R, src_T, src_K = src_ext[:, :3, :3], src_ext[:, :3, 3], src_ixt
    src_c = cameras_from_opencv_projection(src_R, src_T, src_K, torch.tensor([Hz, Wz]).float().unsqueeze(0))
    src_cam = edict(camera=src_c)

    # Compute epipolar masks
    tar_msk = compute_epipolar_mask(tar_cam, src_cam, Hz, Wz, dialate_mask, debug_depth, visualize_mask)
    src_msk = compute_epipolar_mask(src_cam, tar_cam, Hz, Wz, dialate_mask, debug_depth, visualize_mask)

    # Compute plucker coordinates
    tar_emb = compute_plucker_embed(tar_cam, Hz, Wz).squeeze()
    src_emb = compute_plucker_embed(src_cam, Hz, Wz).squeeze()

    return tar_msk, src_msk, tar_emb, src_emb


def compute_masks_pluckers(w2cs: torch.Tensor, ixts: torch.Tensor, H: int, W: int, factor: int = 8,
                           dialate_mask=True, debug_depth=False, visualize_mask=False):
    # Total number of views and resolutions
    n_views = len(w2cs)
    Hz, Wz = H // factor, W // factor

    # Adjust the intrinsics to the lower resolution
    ixts[:, :2] = ixts[:, :2] / factor

    # Intialize all epipolar masks to ones (i.e. all pixels are considered) and plucker embeddings to None
    msks = torch.ones(n_views, n_views, Hz * Wz, Hz * Wz, dtype=torch.bool)
    embs = [None for _ in range(n_views)]

    # Compute pairwise mask and plucker
    for i in range(n_views):
        for j in range(n_views):
            # Skip if itself
            if i == j: continue

            # Compute the epipolar mask between view i, j and their corresponding plucker embeddings
            msk0, msk1, emb0, emb1 = compue_mask_plucker(w2cs[i:i+1], w2cs[j:j+1], ixts[i:i+1], ixts[j:j+1], Hz, Wz,
                                                         dialate_mask=dialate_mask, debug_depth=debug_depth, visualize_mask=visualize_mask)
            msks[i, j], embs[i] = msk0, emb0
            msks[j, i], embs[j] = msk1, emb1

    # Reshape and stack things
    msks = rearrange(msks, 'v1 v2 p1 p2 -> (v1 p1) (v2 p2)')  # (V * Hz * Wz, V * Hz * Wz)
    embs = torch.stack(embs)  # (V, 6, Hz, Wz)

    return msks, embs
