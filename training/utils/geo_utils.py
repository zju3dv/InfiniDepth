import numpy as np
import open3d as o3d


def resize_depth(depth, shape):
    # opencv like
    orig_h, orig_w = depth.shape
    new_w, new_h = shape
    new_depth = np.zeros((new_h, new_w), dtype=depth.dtype)
    nonzero = depth.nonzero()  # h, w
    new_y = nonzero[0] * (new_h / orig_h)
    new_x = nonzero[1] * (new_w / orig_w)
    new_x = np.clip(new_x.astype(np.int32), 0, new_w - 1)
    new_y = np.clip(new_y.astype(np.int32), 0, new_h - 1)
    new_depth[new_y, new_x] = depth[nonzero]
    return new_depth


def project_points(points, ext):
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = (ext.T @ points.T).T
    return points[:, :3]


def depth2pcd(
    depth,
    ixt,
    depth_min=None,
    depth_max=None,
    color=None,
    ext=None,
    conf=None,
    ret_pcd=False,
    clip_box=None,
    ret_mask=False,
):
    height, width = depth.shape
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    zz = depth.reshape(-1)
    mask = np.ones_like(xx, dtype=np.bool_)
    if depth_min is not None:
        mask &= zz >= depth_min
    if depth_max is not None:
        mask &= zz <= depth_max
    if conf is not None:
        mask &= conf.reshape(-1) == 2
    # xx = xx[mask]
    # yy = yy[mask]
    # zz = zz[mask]
    pcd = np.stack([xx, yy, np.ones_like(xx)], axis=1)
    pcd = pcd * zz[:, None]
    pcd = np.dot(pcd, np.linalg.inv(ixt).T)
    if ext is not None:
        pcd = np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1)
        pcd = np.dot(pcd, np.linalg.inv(ext).T)
    new_mask = np.logical_and(np.ones_like(pcd[:, 0]).astype(np.bool_), mask.reshape(-1))
    if clip_box is not None:
        assert len(clip_box) == 6
        for i, val in enumerate(clip_box):
            if val is None:
                continue
            if i == 0:
                new_mask &= pcd[:, 0] <= val
            elif i == 1:
                new_mask &= pcd[:, 1] <= val
            elif i == 2:
                new_mask &= pcd[:, 2] <= val
            elif i == 3:
                new_mask &= pcd[:, 0] >= val
            elif i == 4:
                new_mask &= pcd[:, 1] >= val
            elif i == 5:
                new_mask &= pcd[:, 2] >= val
    if color is not None:
        if color.dtype == np.uint8:
            color = color.astype(np.float32) / 255.0
        if ret_pcd:
            import open3d as o3d

            points = pcd
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3][new_mask])
            pcd.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3)[mask][new_mask])
        else:
            return pcd[:, :3][new_mask], color.reshape(-1, 3)[mask][new_mask]
    else:
        if ret_pcd:
            import open3d as o3d

            points = pcd
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3][new_mask])
            if ret_mask:
                return pcd, new_mask
        else:
            return pcd[:, :3][new_mask]
    return pcd


def export_pcd(path, points, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(path, pcd)
