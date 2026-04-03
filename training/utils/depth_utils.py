from training.utils.logger import Log
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RANSACRegressor
import cv2
from scipy.spatial import KDTree
import torch
import numpy as np
from enum import Enum, auto


class Normalization(Enum):
    # Type of depth normalization
    # will truncate the depth to fixed range and then normalize to [0, 1]
    MINMAX = auto()
    # will truncate the depth according to the 2% and 98% percentile and then normalize to [0, 1]
    SIMP = auto()
    LOG = auto()  # will perform log normalization
    DISP = auto()  # will perform 1/d normalization
    TRUNC = auto()  # will truncate the depth to fixed range
    NONORM = auto()  # will not perform normalization
    # will truncate the depth to fixed range and then normalize to [0, 1]
    minmax = auto()
    # will truncate the depth according to the 2% and 98% percentile and then normalize to [0, 1]
    simp = auto()
    log = auto()  # will perform log normalization
    disp = auto()  # will perform 1/d normalization
    trunc = auto()  # will truncate the depth to fixed range
    nonorm = auto()  # will not perform normalization


def norm_coord(y, x, H, W):
    y = y.astype(np.float32, copy=False)
    x = x.astype(np.float32, copy=False)
    cy = (2.0 * (y + 0.5) / H) - 1.0
    cx = (2.0 * (x + 0.5) / W) - 1.0
    return np.stack([cy, cx], axis=-1).astype(np.float32, copy=False)


def normalize_depth(norm_type, dpt: torch.Tensor, min=0.01, max=50.0):
    """ Perform depth normalization according to the specified type.
        Return a normalized depth map and the mask of valid pixels, all valid at the beginning.

    Args:
        norm_type: Enum, type of depth normalization
        dpt: torch.Tensor (1, H, W) or (S, 1, H, W), depth map to be normalized
        min: float, minimum depth value
        max: float, maximum depth value

    Returns:
        dpt: torch.Tensor (1, H, W) or (S, 1, H, W), normalized depth map
        msk: torch.Tensor (1, H, W) or (S, 1, H, W), mask of valid pixels
        min: float, actual minimum value that is used in the normalization
        max: float, actual maximum value that is used in the normalization
    """

    msk = torch.ones_like(dpt).byte()
    # Depth map may include nan values, replace them with the maximum value and update the mask
    msk[dpt != dpt] = 0

    # Check if there are valid pixels
    if msk.sum() > 0:
        # Replace nan values with the median value to avoid influencing torch.kthvalue() calculation
        dpt[dpt != dpt] = dpt[dpt == dpt].median()

        # Perform real depth normalization
        if norm_type == Normalization.MINMAX or norm_type == Normalization.minmax:
            msk[dpt < min], msk[dpt > max] = 0, 0
            dpt = torch.clamp(dpt, min, max)
            dpt = (dpt - min) / (max - min)
        elif norm_type == Normalization.SIMP or norm_type == Normalization.simp:
            min = torch.kthvalue(dpt.flatten(), int(dpt.numel() * 0.02)).values
            max = torch.kthvalue(dpt.flatten(), int(dpt.numel() * 0.98)).values
            msk[dpt < min], msk[dpt > max] = 0, 0
            dpt = torch.clamp(dpt, min, max)
            dpt = (dpt - min) / (max - min)
        elif norm_type == Normalization.LOG or norm_type == Normalization.log:
            msk[dpt < min], msk[dpt > max] = 0, 0
            dpt = torch.clamp(dpt, min, max)
            dpt = (torch.log(dpt) - torch.log(min)) / \
                (torch.log(max) - torch.log(min))
        elif norm_type == Normalization.DISP or norm_type == Normalization.disp:
            dpt = torch.clamp(dpt, min, max)
            dpt = 1.0 / dpt
        elif norm_type == Normalization.TRUNC or norm_type == Normalization.trunc:
            msk[dpt < min], msk[dpt > max] = 0, 0
            dpt = torch.clamp(dpt, min, max)
        elif norm_type == Normalization.NONORM or norm_type == Normalization.nonorm:
            pass
        else:
            raise NotImplementedError(
                f"Normalization type {norm_type} not implemented")

    # If all values are nan, set the depth map to all ones and the mask to all zeros
    else:
        dpt = torch.ones_like(dpt)

    return dpt, msk, min, max


def denormalize_depth(norm_type, dpt: torch.Tensor, min=0.01, max=50.0):
    """ Perform depth denormalization according to the specified type.
        This is the inverse operation of normalize_depth, invoked after the depth prediction for evaluation.

    Args:
        norm_type: Enum, type of depth normalization
        dpt: torch.Tensor (1, H, W) or (B, 1, H, W), normalized depth map to be denormalized
        min: float or torch.Tensor (B), minimum depth value
        max: float or torch.Tensor (B), maximum depth value

    Returns:
        dpt: torch.Tensor (1, H, W), denormalized depth map
    """
    # Deal with the nasty shape and type
    if isinstance(min, torch.Tensor):
        for _ in range(dpt.ndim - min.ndim):
            min = min[..., None]
    if isinstance(max, torch.Tensor):
        for _ in range(dpt.ndim - max.ndim):
            max = max[..., None]

    # Perform real depth denormalization
    if norm_type == Normalization.MINMAX or norm_type == Normalization.minmax:
        dpt = dpt * (max - min) + min
    elif norm_type == Normalization.SIMP or norm_type == Normalization.simp:
        dpt = dpt * (max - min) + min
    elif norm_type == Normalization.LOG or norm_type == Normalization.log:
        dpt = torch.exp(
            dpt * (torch.log(max) - torch.log(min)) + torch.log(min))
    elif norm_type == Normalization.DISP or norm_type == Normalization.disp:
        dpt = 1.0 / dpt
    elif norm_type == Normalization.TRUNC or norm_type == Normalization.trunc:
        pass
    elif norm_type == Normalization.NONORM or norm_type == Normalization.nonorm:
        pass
    else:
        raise NotImplementedError(
            f"Denormalization type {norm_type} not implemented")

    return dpt


def dspbdpt(dsp: torch.Tensor, tar_ixt: torch.Tensor, tar_ext: torch.Tensor, src_ext: torch.Tensor):
    """ Convert depth map to disparity map using the camera intrinsics.
        depth = focal * baseline / disparity.

    Args:
        dsp: torch.Tensor (1, H, W) or (B, 1, H, W), input disparity map
        tar_ixt: torch.Tensor (3, 3) or (B, 3, 3), target camera intrinsics
        tar_ext: torch.Tensor (4, 4) or (B, 4, 4), target camera extrinsics
        src_ext: torch.Tensor (4, 4) or (B, 4, 4), source camera extrinsics

    Returns:
        dpt: torch.Tensor (1, H, W) or (B, 1, H, W), converted depth map
    """
    # Get the focal length and baseline
    focal = tar_ixt[..., 0, 0]  # (B,)
    baseline = torch.norm(tar_ext[..., -1] - src_ext[..., -1], dim=-1)  # (B,)
    # Deal with nasty shapes
    focal = focal[None, None, None]
    baseline = baseline[None, None, None]

    # Convert disparity to depth
    dpt = focal * baseline / dsp

    return dpt


def bilateralFilter(src, d, sigmaColor, sigmaSpace, depth):
    '''
    INPUT:
    src: input image
    d: 	Diameter of each pixel neighborhood that is used during filtering. 
    sigmaColor: Filter sigma in the color space
    sigmaSpace: Filter sigma in the coordinate space.

    OUTPUT:
    dst: return image 
    '''
    # print("Running Bilateral Blur")
    # assert src.dtype == np.uint8
    # src = src.astype(np.int64) # avoid overflow
    src_min, src_max = src.min(), src.max()
    src = (src - src_min) / ((src_max - src_min) + 0.0001)
    ksize = (d, d)
    ret = np.zeros_like(depth)
    H, W = src.shape[:2]
    assert (ksize[0] % 2 == 1 and ksize[0] > 0)
    assert (ksize[1] % 2 == 1 and ksize[1] > 0)
    offsetX, offsetY = np.meshgrid(np.arange(ksize[0]), np.arange(ksize[1]))
    offsetX -= ksize[0] // 2
    offsetY -= ksize[1] // 2
    w1 = np.exp(-(offsetX ** 2 + offsetY ** 2) / (2 * sigmaSpace ** 2))
    # from tqdm import tqdm
    for i in range(0, H):
        for j in range(0, W):
            indY = offsetY + i
            indX = offsetX + j
            indY[indY < 0] = 0
            indY[indY >= H] = H - 1
            indX[indX < 0] = 0
            indX[indX >= W] = W - 1
            cropped_img = src[indY, indX]
            cropped_depth = depth[indY, indX]
            diff = -((cropped_img - src[i, j]) ** 2 / (2 * sigmaColor ** 2))
#             w2 = np.exp(diff / (2 * sigmaColor ** 2))
            w2 = np.exp(diff)
            ret[i, j] = (
                w1 * w2 * cropped_depth).reshape(-1).sum() / (w1 * w2).sum()
#             break
#         break
    return ret


def GenerateSpotMask(img_h, img_w, stride=11, dist_coef=2e-5, noise=0, plt_flag=False):
    '''
    Simulate pincushion distortion:
    --stride: 
    It controls the distance between neighbor spots7
    Suggest stride value:       5/6

    --dist_coef:
    It controls the curvature of the spot pattern
    Larger dist_coef distorts the pattern more.
    Suggest dist_coef value:    0 ~ 5e-5

    --noise:
    standard deviation of the spot shift
    Suggest noise value:        0 ~ 0.5
    '''

    # Generate Grid points
    hline_num = img_h//stride
    x_odd, y_odd = np.meshgrid(
        np.arange(stride//2, img_h, stride*2), np.arange(stride//2, img_w, stride))
    x_even, y_even = np.meshgrid(
        np.arange(stride//2+stride, img_h, stride*2), np.arange(stride, img_w, stride))
    x_u = np.concatenate((x_odd.ravel(), x_even.ravel()))
    y_u = np.concatenate((y_odd.ravel(), y_even.ravel()))
    x_u -= img_h//2
    y_u -= img_w//2

    # Distortion
    r_u = np.sqrt(x_u**2+y_u**2)
    r_d = r_u + dist_coef * r_u**3
    num_d = r_d.size
    if (((r_u-1e-5) < 0) & ((r_u+1e-5) > 0)).any():
        msk = (((r_u-1e-5) < 0) & ((r_u+1e-5) > 0))
        r_u[msk] += 1e-5
    sin_theta = x_u/r_u
    cos_theta = y_u/r_u
    x_d = np.round(r_d * sin_theta + img_h//2 +
                   np.random.normal(0, noise, num_d))
    y_d = np.round(r_d * cos_theta + img_w//2 +
                   np.random.normal(0, noise, num_d))
    idx_mask = (x_d < img_h) & (x_d > 0) & (y_d < img_w) & (y_d > 0)
    x_d = x_d[idx_mask].astype('int')
    y_d = y_d[idx_mask].astype('int')

    spot_mask = np.zeros((img_h, img_w))
    spot_mask[x_d, y_d] = 1
    return spot_mask  # , x_d, y_d


def interp_depth_rgb(sdpt: np.ndarray,
                     ref_img: np.ndarray,
                     speed=1,
                     k=4,
                     **kwargs):
    h, w = sdpt.shape
    lb = 1e-3
    if (sdpt <= lb).all():
        return np.zeros((h, w))

    val_x, val_y = np.where(sdpt > lb)
    inval_x, inval_y = np.where(sdpt <= lb)
    val_pos = np.stack([val_x, val_y], axis=1)
    inval_pos = np.stack([inval_x, inval_y], axis=1)
    ref_img_min = ref_img.min()
    ref_img_max = ref_img.max()
    ref_img = (ref_img - ref_img_min) / ((ref_img_max - ref_img_min) + 0.0001)
    val_color = ref_img[val_x, val_y]
    inval_color = ref_img[inval_x, inval_y]

    if (sdpt != 0).sum() < k:
        k = (sdpt != 0).sum()

    tree = KDTree(val_pos)
    dists, inds = tree.query(inval_pos, k=k)

    try:
        dpt = np.copy(sdpt).reshape(-1)
    except:
        # print(kwargs["frame"])
        import ipdb
        ipdb.set_trace()
    ######################
    if dpt.size == 0:
        import ipdb
        ipdb.set_trace()
    ######################
    if k == 1:
        dpt[inval_x * w +
            inval_y] = sdpt.reshape(-1,)[val_pos[inds][..., 0] * w + val_pos[inds][..., 1]]
    else:
        ######################
        dists = np.where(dists == 0, 1e-10, dists)
        weights = 1 / dists
        weights /= np.sum(weights, axis=1, keepdims=True)
        
        rgb_diff = (inval_color[:, None] - val_color[inds])
        rgb_sim = -rgb_diff * 0.5 + 0.5
        rgb_sim = np.exp(speed*rgb_sim) * 0.01
        rgb_weights = rgb_sim / np.sum(rgb_sim, axis=1, keepdims=True)

        weights = weights * rgb_weights
        # weights = weights
        weights = weights / \
            np.clip(np.sum(weights, axis=1, keepdims=True), 0.01, None)
        try:
            dpt = np.copy(sdpt).reshape(-1)
        except:
            import ipdb
            ipdb.set_trace()

        try:
            nearest_vals = sdpt[val_x[inds], val_y[inds]]
        except:
            print(sdpt.shape, val_x.shape, val_y.shape, inds.shape,
                  weights.shape, dists.shape, (sdpt != 0).sum())

        weighted_avg = np.sum(nearest_vals * weights, axis=1)
        dpt[inval_x * w + inval_y] = weighted_avg

    return dpt.reshape(h, w)


degree = 1
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
ransac = RANSACRegressor(max_trials=1000)
linear_model = make_pipeline(poly_features, ransac)


def scale_shift_align_depth(src, tar, mask=None, disp=False, fit_type="poly"):
    if mask is None:
        mask = np.ones_like(tar, dtype=bool)
    orig_src = src.copy()
    if tar.shape[0] != src.shape[0] or tar.shape[1] != src.shape[1]:
        src = cv2.resize(src, tar.shape[::-1], interpolation=cv2.INTER_NEAREST)
    if disp:
        mask = np.logical_and(np.logical_and(mask, src > 5e-4), src < 5e4)
    if mask.sum() == 0:
        Log.error(f"No valid pixels for {fit_type}")
        raise ValueError(f"No valid pixels for {fit_type}")
    tar_val = tar[mask].astype(np.float32)
    src_val = src[mask].astype(np.float32)
    if disp:
        tar_val = np.clip(tar_val, 1e-3, None)
        tar_val = 1 / tar_val

    if fit_type == "poly":
        try:
            a, b = np.polyfit(src_val, tar_val, deg=1)
        except:
            import ipdb
            ipdb.set_trace()
    elif fit_type == "ransac":
        linear_model.fit(src_val[:, None], tar_val[:, None])
        a = linear_model.named_steps["ransacregressor"].estimator_.coef_
        b = linear_model.named_steps["ransacregressor"].estimator_.intercept_
        a, b = a.item(), b.item()
    else:
        Log.debug("Unknown fit type")
        a, b = 1, 0
    Log.debug(f"Fit {fit_type}: scale: {a}, shift: {b}")
    if a < 0:
        Log.warn(
            f"Negative scale detected, {fit_type}, scale: {a}, shift: {b}")
        # a, b = 1, 0
        ret_tag = False
    else:
        ret_tag = True
    src_ = a * orig_src + b
    if disp:
        src_ = 1 / np.clip(src_, 1e-2, None)  # max 100 meters
    return ret_tag, src_
