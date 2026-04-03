import numpy as np
import cv2

def parse_ixt(camera):
    ixt = np.eye(3)
    if camera.model == 'OPENCV':
        ixt[0, 0] = camera.params[0]
        ixt[1, 1] = camera.params[1]
        ixt[0, 2] = camera.params[2]
        ixt[1, 2] = camera.params[3]
    else:
        raise NotImplementedError
    return ixt

def parse_dist(camera):
    dist = np.zeros(5)
    if camera.model == 'OPENCV':
        dist[:4] = camera.params[4:]
    else:
        raise NotImplementedError
    return dist


def cv_undistort_img(rgb, ixt, dist, hw, resize=True):
    ixt = np.copy(ixt)
    ixt[0, 2] = ixt[0, 2] - 0.5
    ixt[1, 2] = ixt[1, 2] - 0.5
    new_ixt, roi = cv2.getOptimalNewCameraMatrix(ixt, dist, (rgb.shape[1], rgb.shape[0]), 0)
    h_orig, w_orig = rgb.shape[:2]
    rgb = cv2.undistort(rgb, ixt, dist, newCameraMatrix=new_ixt)
    x, y, w, h = roi
    rgb = rgb[y : y + h, x : x + w]
    rgb = cv2.resize(rgb, (w_orig, h_orig), interpolation=cv2.INTER_AREA)
    return rgb
    