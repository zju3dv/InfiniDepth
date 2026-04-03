import torch
from typing import List
from torch.nn import functional as F


def crop_using_xywh(x, y, w, h, K, *list_of_imgs):
    K = K.clone()
    K[..., :2, -1] -= torch.as_tensor([x, y], device=K.device)  # crop K

    if list_of_imgs[0].shape[-3] == 3 or list_of_imgs[0].shape[-3] == 4 or list_of_imgs[0].shape[-3] == 1:
        # (..., C, H, W)
        list_of_imgs = [img[..., y:y + h, x:x + w]
                        if isinstance(img, torch.Tensor) else
                        [im[..., y:y + h, x:x + w] for im in img]  # HACK: evil list comprehension
                        for img in list_of_imgs]

    else:
        # (..., H, W, C)
        list_of_imgs = [img[..., y:y + h, x:x + w, :]
                        if isinstance(img, torch.Tensor) else
                        [im[..., y:y + h, x:x + w, :] for im in img]  # HACK: evil list comprehension
                        for img in list_of_imgs]

    return K, *list_of_imgs


def get_xywh_from_hwc(H: int, W: int, crop: int):
    x = (W % crop) // 2
    w = W - (W % crop)
    y = (H % crop) // 2
    h = H - (H % crop)

    return x, y, w, h
