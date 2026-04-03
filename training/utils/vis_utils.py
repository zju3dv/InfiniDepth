import os
from os.path import join
import cv2
import imageio
import matplotlib
import numpy as np
import open3d as o3d
import torch
from training.utils import geo_utils
from training.utils.logger import Log
from training.utils.parallel_utils import parallel_execution

# from dotmap import DotMap

# from training.utils.base_utils import dotdict


@torch.jit.script
def getWorld2View(R: torch.Tensor, t: torch.Tensor):
    """
    R: ..., 3, 3
    T: ..., 3, 1
    """
    sh = R.shape[:-2]
    T = torch.eye(4, dtype=R.dtype, device=R.device)  # 4, 4
    for i in range(len(sh)):
        T = T.unsqueeze(0)
    T = T.expand(sh + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3:] = t
    return T


@torch.jit.script
def getProjectionMatrix(K: torch.Tensor, H: torch.Tensor, W: torch.Tensor, znear: torch.Tensor, zfar: torch.Tensor):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]
    one = K[2, 2]

    P = torch.zeros(4, 4, dtype=K.dtype, device=K.device)

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = -(zfar + znear) / (znear - zfar)
    P[2, 3] = 2 * zfar * znear / (znear - zfar)

    P[3, 2] = one

    return P


def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))


class DotDict(dict):
    """Dictionary with dot notation access to its attributes"""

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        try:
            del self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __repr__(self):
        return f"DotDict({super().__repr__()})"


def convert_to_gaussian_camera(
    K: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    H: torch.Tensor,
    W: torch.Tensor,
    n: torch.Tensor,
    f: torch.Tensor,
    cpu_K: torch.Tensor,
    cpu_R: torch.Tensor,
    cpu_T: torch.Tensor,
    cpu_H: int,
    cpu_W: int,
    cpu_n: float = 0.01,
    cpu_f: float = 100.0,
):
    output = DotDict()

    output.image_height = cpu_H
    output.image_width = cpu_W

    output.K = K
    output.R = R
    output.T = T

    output.znear = cpu_n
    output.zfar = cpu_f

    # MARK: MIGHT SYNC IN DIST TRAINING, WHY?
    output.FoVx = focal2fov(cpu_K[0, 0].cpu(), cpu_W.cpu())
    # MARK: MIGHT SYNC IN DIST TRAINING, WHY?
    output.FoVy = focal2fov(cpu_K[1, 1].cpu(), cpu_H.cpu())

    # Use .float() to avoid AMP issues
    output.world_view_transform = getWorld2View(R, T).transpose(0, 1).float()  # this is now to be right multiplied
    output.projection_matrix = (
        getProjectionMatrix(K, H, W, n, f).transpose(0, 1).float()
    )  # this is now to be right multiplied
    output.full_proj_transform = torch.matmul(output.world_view_transform, output.projection_matrix).float()  # 4, 4
    output.camera_center = (-R.mT @ T)[..., 0].float()  # B, 3, 1 -> 3,

    # Set up rasterization configuration
    output.tanfovx = np.tan(output.FoVx * 0.5)
    output.tanfovy = np.tan(output.FoVy * 0.5)

    return output


@torch.jit.script
def affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth + 1e-6)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


row_col_ = {2: (2, 1), 7: (2, 4), 8: (2, 4), 9: (3, 3), 26: (4, 7)}

row_col_square = {2: (2, 1), 7: (3, 3), 8: (3, 3), 9: (3, 3), 26: (5, 5)}


def get_row_col(l, square):
    if square and l in row_col_square.keys():
        return row_col_square[l]
    if l in row_col_.keys():
        return row_col_[l]
    else:
        from math import sqrt

        row = int(sqrt(l) + 0.5)
        col = int(l / row + 0.5)
        if row * col < l:
            col = col + 1
        if row > col:
            row, col = col, row
        return row, col


def merge(images, row=-1, col=-1, resize=False, ret_range=False, square=False, resize_height=1000, **kwargs):
    if row == -1 and col == -1:
        row, col = get_row_col(len(images), square)
    height = images[0].shape[0]
    width = images[0].shape[1]
    # special case
    if height > width:
        if len(images) == 3:
            row, col = 1, 3
    if len(images[0].shape) > 2:
        ret_img = np.zeros((height * row, width * col, images[0].shape[2]), dtype=np.uint8) + 255
    else:
        ret_img = np.zeros((height * row, width * col), dtype=np.uint8) + 255
    ranges = []
    for i in range(row):
        for j in range(col):
            if i * col + j >= len(images):
                break
            img = images[i * col + j]
            # resize the image size
            img = cv2.resize(img, (width, height))
            ret_img[height * i : height * (i + 1), width * j : width * (j + 1)] = img
            ranges.append((width * j, height * i, width * (j + 1), height * (i + 1)))
    if resize:
        min_height = resize_height
        if ret_img.shape[0] > min_height:
            scale = min_height / ret_img.shape[0]
            ret_img = cv2.resize(ret_img, None, fx=scale, fy=scale)
            ranges = [(int(r[0] * scale), int(r[1] * scale), int(r[2] * scale), int(r[3] * scale)) for r in ranges]
    if ret_range:
        return ret_img, ranges
    return ret_img


def warp_rgbd_video(
    img_dir,
    depth_dir,
    tar_dir=None,
    conf_dir=None,
    read_depth_func=None,
    frame_sample=[0, 600, 1],
    focal=1440,
    tar_h=756,
    tar_w=1008,
    cx=None,
    cy=None,
    depth_format=".png",
    depth_prefix=None,
    depth_min=0.01,
    depth_max=4.0,
    pre_upsample=False,
    fps=60,
):
    # focal of iphone 13 pro: 1440-1443
    # focal of iphone 15 pro / Ipad: 1350-1400
    img_files = sorted([join(img_dir, f) for f in os.listdir(img_dir) if f.endswith((".png", ".jpg", ".jpeg"))])

    # sorted img_files by int(name) instead of str(name)
    img_files = sorted(img_files, key=lambda x: int(os.path.basename(x).split(".")[0]))
    depth_files = sorted([join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(depth_format)])
    depth_files = sorted(depth_files, key=lambda x: int(os.path.basename(x).split(".")[0]))
    if (len(img_files) - len(depth_files)) >= 2 or (len(depth_files) - len(img_files) >= 2):
        depth_files = []
        for i in range(len(img_files)):
            depth_files.append(join(depth_dir, os.path.basename(img_files[i]).replace(".jpg", ".png")))
    if depth_prefix is not None:
        depth_files = [f for f in depth_files if os.path.basename(f).startswith(depth_prefix)]

    if conf_dir is not None:
        conf_files = sorted([join(conf_dir, f) for f in os.listdir(conf_dir) if f.endswith(".png")])
    else:
        conf_files = [None for _ in depth_files]

    if (len(img_files) - 1) == len(depth_files):
        img_files = img_files[:-1]
    elif len(img_files) == (len(depth_files) - 1):
        depth_files = depth_files[:-1]
        conf_files = conf_files[:-1]

    # assert len(img_files) == len(depth_files), 'The number of images and depth maps should be the same.'
    # assert len(img_files) == len(conf_files)
    frame_len = len(img_files)
    if frame_sample[1] == -1:
        frame_sample[1] = frame_len

    img_files = img_files[frame_sample[0] : frame_sample[1] : frame_sample[2]]
    depth_files = depth_files[frame_sample[0] : frame_sample[1] : frame_sample[2]]
    conf_files = conf_files[frame_sample[0] : frame_sample[1] : frame_sample[2]]
    if tar_dir is not None:
        tar_files = [join(tar_dir, f"rgb/{i:06d}.jpg") for i in range(len(img_files))]
    else:
        tar_files = [None for img_file in img_files]
    fps = int(fps / frame_sample[2])

    # debug
    # img_files = img_files[:1]
    # import ipdb; ipdb.set_trace()

    parallel_execution(
        img_files,
        depth_files,
        conf_files,
        tar_files,
        focal,
        tar_h,
        tar_w,
        cx,
        cy,
        depth_min,
        depth_max,
        read_depth_func,
        pre_upsample,
        action=warp_rgbd_rast,
        # sequential=True,
        print_progress=True,
        desc="Warping RGBD",
        num_processes=32,
    )
    if tar_dir is not None:
        if os.path.exists(join(tar_dir, "rgb.mp4")):
            os.system("rm -rf {}".format(join(tar_dir, "rgb.mp4")))
        cmd = "ffmpeg -r {} -i {} -vcodec libx264 -crf 28 -pix_fmt yuv420p {}".format(
            fps * frame_sample[2], join(tar_dir, "rgb/%06d.jpg"), join(tar_dir, "rgb.mp4")
        )
        os.system(cmd)
        Log.info("Video saved at {}".format(join(tar_dir, "rgb.mp4")))


def warp_rgbd(
    img,
    depth,
    conf,
    tar_path,
    focal,
    tar_h=None,
    tar_w=None,
    cx=None,
    cy=None,
    depth_min=None,
    depth_max=None,
    read_depth_func=None,
    pre_upsample=False,
    upsample_method="linear",
):
    if isinstance(img, str):
        img = imageio.imread(img)
    if img.shape[0] == tar_h and img.shape[1] != tar_w:
        img = img[:, :tar_w]
    if img.shape[0] != tar_h and img.shape[1] == tar_w:
        img = img[:tar_h, :]
    if isinstance(depth, str):
        if read_depth_func is not None:
            depth = read_depth_func(depth)
        elif depth.endswith(".png"):
            depth = (np.asarray(imageio.imread(depth)) / 1000.0).astype(np.float64)
        elif depth.endswith(".npz"):
            depth = np.load(depth)["data"]
        else:
            raise ValueError("Unsupported depth format.")
    if isinstance(conf, str):
        conf = np.asarray(imageio.imread(conf))

    if pre_upsample:
        interpolation = cv2.INTER_LINEAR if upsample_method == "linear" else cv2.INTER_NEAREST
        depth = cv2.resize(depth, (tar_w, tar_h), interpolation=interpolation)
    if img.shape[0] != tar_h or img.shape[1] != tar_w:
        img = cv2.resize(img, (tar_w, tar_h), interpolation=cv2.INTER_LINEAR)
    img_height, img_width = img.shape[:2]
    fx, fy, cx, cy = focal, focal, img_width / 2, img_height / 2

    if depth.shape[0] != img_height:
        fx = fx * depth.shape[0] / img_height
        cx = cx * depth.shape[0] / img_height

    if depth.shape[1] != img_width:
        fy = fy * depth.shape[1] / img_width
        cy = cy * depth.shape[1] / img_width

    ixt = np.eye(3)
    ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = fx, fy, cx, cy

    color = cv2.resize(img, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
    points, colors = geo_utils.depth2pcd(depth, ixt, depth_min, depth_max, color, conf=conf)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if len(pcd.points) <= 1:
        raise ValueError("No points in the point cloud.")

    def create_transformation_matrix():
        translation = [-0.5, 0.5, 0.5] * 1.5
        rotation = o3d.geometry.get_rotation_matrix_from_xyz([-np.deg2rad(-8), -np.deg2rad(-8), 0])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform

    transform = create_transformation_matrix()
    pcd.transform(transform)
    points = np.asarray(pcd.points)
    height, width = depth.shape[:2]
    render_image = np.ones((depth.shape[0], depth.shape[1], 3), dtype=np.uint8) * 255

    cam_points = points @ ixt.T
    depth = cam_points[:, 2]
    image_points = cam_points[:, :2] / depth[:, None]
    x_coords = image_points[:, 0].astype(int)
    y_coords = image_points[:, 1].astype(int)
    valid_indices = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
    valid_x_coords = x_coords[valid_indices]
    valid_y_coords = y_coords[valid_indices]
    valid_colors = colors[valid_indices]
    valid_depths = depth[valid_indices]
    sorted_indices = np.argsort(valid_depths)[::-1]
    sorted_x_coords = valid_x_coords[sorted_indices]
    sorted_y_coords = valid_y_coords[sorted_indices]
    if valid_colors.dtype == np.float32 or valid_colors.dtype == np.float64:
        valid_colors = (valid_colors * 255).astype(np.uint8)
    sorted_colors = valid_colors[sorted_indices]
    render_image[sorted_y_coords, sorted_x_coords] = sorted_colors

    if render_image.shape[0] != tar_h or render_image.shape[1] != tar_w:
        render_image = cv2.resize(render_image, (tar_w, tar_h), interpolation=cv2.INTER_LINEAR)
    if tar_path is not None:
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        imageio.imwrite(tar_path, render_image)
    else:
        return render_image


def warp_rgbd_rast(
    img,
    depth,
    conf,
    tar_path,
    focal,
    tar_h=None,
    tar_w=None,
    cx=None,
    cy=None,
    depth_min=None,
    depth_max=None,
    read_depth_func=None,
    pre_upsample=False,
    upsample_method="linear",
):
    if isinstance(img, str):
        img = imageio.imread(img)
    if img.shape[0] == tar_h and img.shape[1] != tar_w:
        img = img[:, :tar_w]
    if img.shape[0] != tar_h and img.shape[1] == tar_w:
        img = img[:tar_h, :]
    if isinstance(depth, str):
        if read_depth_func is not None:
            depth = read_depth_func(depth)
        elif depth.endswith(".png"):
            depth = (np.asarray(imageio.imread(depth)) / 1000.0).astype(np.float64)
        elif depth.endswith(".npz"):
            depth = np.load(depth)["data"]
        else:
            raise ValueError("Unsupported depth format.")
    if isinstance(conf, str):
        conf = np.asarray(imageio.imread(conf))

    if pre_upsample:
        interpolation = cv2.INTER_LINEAR if upsample_method == "linear" else cv2.INTER_NEAREST
        depth = cv2.resize(depth, (tar_w, tar_h), interpolation=interpolation)
    if img.shape[0] != tar_h or img.shape[1] != tar_w:
        img = cv2.resize(img, (tar_w, tar_h), interpolation=cv2.INTER_LINEAR)
    img_height, img_width = img.shape[:2]
    fx, fy, cx, cy = focal, focal, img_width / 2, img_height / 2

    if depth.shape[0] != img_height:
        fx = fx * depth.shape[0] / img_height
        cx = cx * depth.shape[0] / img_height

    if depth.shape[1] != img_width:
        fy = fy * depth.shape[1] / img_width
        cy = cy * depth.shape[1] / img_width

    ixt = np.eye(3)
    ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = fx, fy, cx, cy

    color = cv2.resize(img, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
    points, colors = geo_utils.depth2pcd(depth, ixt, depth_min, depth_max, color, conf=conf)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if len(pcd.points) <= 1:
        return

    def create_transformation_matrix():
        translation = [-0.5, 0.5, 0.5]
        rotation = o3d.geometry.get_rotation_matrix_from_xyz([-np.deg2rad(-8), -np.deg2rad(-8), 0])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform

    transform = create_transformation_matrix()
    pcd.transform(transform)
    points = np.asarray(pcd.points)
    height, width = depth.shape[:2]

    image = render_pointcloud_diff_point_rasterization(
        c2w=np.eye(4),
        ixt=ixt,
        points=np.asarray(pcd.points),
        features=colors,
        H=height,
        W=width,
        scale=0.005,
        use_knn_scale=True,
        use_ndc_scale=False,
    )
    render_image = image[0][..., :3] + 1 * (1 - image[0][..., 3:])
    render_image = (render_image.detach().cpu().numpy() * 255).astype(np.uint8)

    # render_image = np.ones((depth.shape[0], depth.shape[1], 3), dtype=np.uint8) * 255

    # cam_points = points @ ixt.T
    # depth = cam_points[:, 2]
    # image_points = cam_points[:, :2] / depth[:, None]
    # x_coords = image_points[:, 0].astype(int)
    # y_coords = image_points[:, 1].astype(int)
    # valid_indices = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
    # valid_x_coords = x_coords[valid_indices]
    # valid_y_coords = y_coords[valid_indices]
    # valid_colors = colors[valid_indices]
    # valid_depths = depth[valid_indices]
    # sorted_indices = np.argsort(valid_depths)[::-1]
    # sorted_x_coords = valid_x_coords[sorted_indices]
    # sorted_y_coords = valid_y_coords[sorted_indices]
    # if valid_colors.dtype == np.float32 or valid_colors.dtype == np.float64:
    #     valid_colors = (valid_colors * 255).astype(np.uint8)
    # sorted_colors = valid_colors[sorted_indices]
    # render_image[sorted_y_coords, sorted_x_coords] = sorted_colors

    # if render_image.shape[0] != tar_h or render_image.shape[1] != tar_w:
    #     render_image = cv2.resize(render_image, (tar_w, tar_h), interpolation=cv2.INTER_LINEAR)
    if tar_path is not None:
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        imageio.imwrite(tar_path, render_image)
    else:
        return render_image


def render_pointcloud_diff_point_rasterization(
    c2w,
    ixt,
    points,
    features,
    H,
    W,
    gpu_id=0,
    occ=None,
    scale=0.035,  # if not using ndc scale, this is the scale in world space, should use 0.025
    use_ndc_scale=False,
    use_knn_scale=False,
    knn_scale_down=1.0,
    max_scale=0.05,
):
    """
    Render pointcloud using gsplat
    Requires points and features to have a batch dim to be discarded by us here
    """
    if len(c2w) == 1:
        c2w[0][None]
    else:
        c2w = c2w[None]
    if len(ixt) == 1:
        ixt[0][None]
    else:
        ixt = ixt[None]
    if len(points) == 1:
        points[0][None]
    else:
        points = points[None]
    if len(features) == 1:
        features[0][None]
    else:
        features = features[None]

    # device = get_device(gpu_id)
    device = torch.device("cuda:0")

    c2w = torch.as_tensor(c2w).to(device, non_blocking=True).float()[0]
    w2c = affine_inverse(c2w)
    cpu_ixt = torch.as_tensor(ixt[0])
    ixt = torch.as_tensor(ixt).to(device, non_blocking=True).float()[0]
    xyz3 = torch.as_tensor(points).to(device, non_blocking=True).float()[0]
    rgb3 = torch.as_tensor(features).to(device, non_blocking=True).float()[0, ..., :3]  # only rgb
    if occ is None:
        occ1 = torch.full_like(xyz3[..., :1], 1.0)
    else:
        occ1 = torch.as_tensor(occ).to(device, non_blocking=True).float()[..., None]
    scales = torch.full_like(xyz3[..., :1], scale)  # isometrical scale
    quats = xyz3.new_zeros(xyz3.shape[:-1] + (4,))  # identity quaternion
    quats[..., 3] = 1

    if use_ndc_scale:
        # Revert the ndc scale back to world space
        # gl_PointSize = abs(H * K[1][1] * radius / gl_Position.w) * radii_mult;  // need to determine size in pixels
        views = xyz3 @ w2c[:3, :3].mT + w2c[:3, 3]
        x, y, z = views.chunk(3, dim=-1)
        # scales = scales * z / ixt[0, 0] * 300
        scales = scales * z / ixt[0, 0] * H
    elif use_knn_scale:
        # Use KNN to determine point scale based on local point density
        from simple_knn._C import distCUDA2

        def knn(x: torch.Tensor, K: int = 4) -> torch.Tensor:
            from sklearn.neighbors import NearestNeighbors

            x_np = x.cpu().numpy()
            model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
            distances, _ = model.kneighbors(x_np)
            return torch.from_numpy(distances).to(x)

        dist2 = torch.clamp_min(distCUDA2(xyz3), 0.0000001)
        # dist2 = (knn(xyz3.float(), 4)[:, 1:] ** 2).mean(dim=-1)
        scales = torch.sqrt(dist2)[..., None] * knn_scale_down
        scales = torch.clamp(scales, max=max_scale)
        scales = torch.minimum(scales, torch.full_like(scales, scale))

    from diff_point_rasterization import PointRasterizationSettings, PointRasterizer

    # from easyvolcap.utils.gaussian_utils import convert_to_gaussian_camera
    gaussian_camera = convert_to_gaussian_camera(
        ixt,
        w2c[:3, :3],
        w2c[:3, 3:],
        torch.as_tensor(H).to(xyz3.device, non_blocking=True),
        torch.as_tensor(W).to(xyz3.device, non_blocking=True),
        torch.as_tensor(0.01).to(xyz3.device, non_blocking=True),
        torch.as_tensor(100.0).to(xyz3.device, non_blocking=True),
        cpu_ixt,
        None,
        None,
        torch.as_tensor(H),
        torch.as_tensor(W),
        torch.as_tensor(0.01),
        torch.as_tensor(100.0),
    )

    # Prepare rasterization settings for gaussian
    raster_settings = PointRasterizationSettings(
        image_height=gaussian_camera.image_height,
        image_width=gaussian_camera.image_width,
        tanfovx=gaussian_camera.tanfovx,
        tanfovy=gaussian_camera.tanfovy,
        bg=torch.full([3], 0.0, device=xyz3.device),  # GPU
        scale_modifier=1.0,
        viewmatrix=gaussian_camera.world_view_transform,
        projmatrix=gaussian_camera.full_proj_transform,
        sh_degree=0,
        campos=gaussian_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    scr = torch.zeros_like(xyz3, requires_grad=False) + 0  # gradient magic

    rasterizer = PointRasterizer(raster_settings=raster_settings)
    rendered_image, rendered_depth, rendered_alpha, radii = rasterizer(
        means3D=xyz3,
        means2D=scr,
        colors_precomp=rgb3,
        opacities=occ1,
        radius=scales,
    )

    rgb = rendered_image[None].permute(0, 2, 3, 1)
    acc = rendered_alpha[None].permute(0, 2, 3, 1)
    rendered_depth[None].permute(0, 2, 3, 1)

    return torch.cat([rgb, acc], dim=-1)  # 1, H, W, 4


def visualize_depth(
    depth: np.ndarray,
    depth_min=None,
    depth_max=None,
    percentile=2,
    ret_minmax=False,
    ret_type=np.uint8,
    cmap="Spectral",
):
    """
    Visualize a depth map using a colormap.

    Args:
        depth: Input depth map array
        depth_min: Minimum depth value for normalization. If None, uses percentile
        depth_max: Maximum depth value for normalization. If None, uses percentile
        percentile: Percentile for min/max computation if not provided
        ret_minmax: Whether to return min/max depth values
        ret_type: Return array type (uint8 or float)
        cmap: Matplotlib colormap name to use

    Returns:
        Colored depth visualization as numpy array
        If ret_minmax=True, also returns depth_min and depth_max
    """
    valid_mask = np.logical_and(depth > 1e-3, depth < 1e3)
    if depth_min is None:
        if valid_mask.sum() > 0:
            depth_min = np.percentile(depth[valid_mask], percentile)
        else:
            depth_min = 0
    if depth_max is None:
        if valid_mask.sum() > 0:
            depth_max = np.percentile(depth[valid_mask], 100 - percentile)
        else:
            depth_max = 1
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    if ret_type == np.uint8:
        img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)
    elif ret_type == np.float32 or ret_type == np.float64:
        img_colored_np = img_colored_np[0]
    else:
        raise ValueError(f"Invalid return type: {ret_type}")
    # img_colored_np[~valid_mask] = 0
    if ret_minmax:
        return img_colored_np, depth_min, depth_max
    else:
        return img_colored_np
