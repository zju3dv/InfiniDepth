import numpy as np


def unproject_depthmap_ixt(dpt, ixt):
    h, w = dpt.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    uvd = np.stack([u, v, dpt], axis=-1)
    uvd = uvd.reshape(-1, 3)
    uvd[:, :2] = uvd[:, :2] * uvd[:, 2:]
    points = uvd @ np.linalg.inv(ixt).T
    return points

def unproject_depthmap_focal(dpt, focal):
    ixt = np.eye(3)
    h, w = dpt.shape
    ixt[0, 0], ixt[1, 1] = focal, focal
    ixt[0, 2], ixt[1, 2] = w/2., h/2.
    return unproject_depthmap_ixt(dpt, ixt)

