# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import math
from typing import Sequence

import PIL
import torch
from torchvision import transforms

logger = logging.getLogger("dinov3")


def make_interpolation_mode(mode_str: str) -> transforms.InterpolationMode:
    return {mode.value: mode for mode in transforms.InterpolationMode}[mode_str]


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

CROP_DEFAULT_SIZE = 224
RESIZE_DEFAULT_SIZE = int(256 * CROP_DEFAULT_SIZE / 224)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


def make_base_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Compose(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.append(make_base_transform(mean, std))
    transform = transforms.Compose(transforms_list)
    logger.info(f"Built classification train transform\n{transform}")
    return transform


class _MaxSizeResize(object):
    def __init__(
        self,
        max_size: int,
        interpolation: transforms.InterpolationMode,
    ):
        self._size = self._make_size(max_size)
        self._resampling = self._make_resampling(interpolation)

    def _make_size(self, max_size: int):
        return (max_size, max_size)

    def _make_resampling(self, interpolation: transforms.InterpolationMode):
        if interpolation == transforms.InterpolationMode.BICUBIC:
            return PIL.Image.Resampling.BICUBIC
        if interpolation == transforms.InterpolationMode.BILINEAR:
            return PIL.Image.Resampling.BILINEAR
        assert interpolation == transforms.InterpolationMode.NEAREST
        return PIL.Image.Resampling.NEAREST

    def __call__(self, image):
        image.thumbnail(size=self._size, resample=self._resampling)
        return image


def make_resize_transform(
    *,
    resize_size: int,
    resize_square: bool = False,
    resize_large_side: bool = False,  # Set the larger side to resize_size instead of the smaller
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
):
    assert not (resize_square and resize_large_side), "These two options can not be set together"
    if resize_square:
        logger.info("resizing image as a square")
        size = (resize_size, resize_size)
        transform = transforms.Resize(size=size, interpolation=interpolation)
        return transform
    elif resize_large_side:
        logger.info("resizing based on large side")
        transform = _MaxSizeResize(max_size=resize_size, interpolation=interpolation)
        return transform
    else:
        transform = transforms.Resize(resize_size, interpolation=interpolation)
        return transform


# Derived from make_classification_eval_transform() with more control over resize and crop
def make_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    resize_square: bool = False,
    resize_large_side: bool = False,  # Set the larger side to resize_size instead of the smaller
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = []
    resize_transform = make_resize_transform(
        resize_size=resize_size,
        resize_square=resize_square,
        resize_large_side=resize_large_side,
        interpolation=interpolation,
    )
    transforms_list.append(resize_transform)
    if crop_size:
        transforms_list.append(transforms.CenterCrop(crop_size))
    transforms_list.append(make_base_transform(mean, std))
    transform = transforms.Compose(transforms_list)
    logger.info(f"Built eval transform\n{transform}")
    return transform


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=transforms.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    return make_eval_transform(
        resize_size=resize_size,
        crop_size=crop_size,
        interpolation=interpolation,
        mean=mean,
        std=std,
        resize_square=False,
        resize_large_side=False,
    )


class MultipleResize(object):
    # A resize transform that makes the large side a multiple of a given number. That might change the aspect ratio.
    def __init__(self, interpolation=transforms.InterpolationMode.BILINEAR, multiple=1):
        self.multiple = multiple
        self.interpolation = interpolation

    def __call__(self, img):
        if self.multiple == 1:
            return img
        if hasattr(img, "shape"):
            h, w = img.shape[-2:]
        else:
            assert isinstance(
                img, PIL.Image.Image
            ), f"img should have a `shape` attribute or be a PIL Image, got {type(img)}"
            w, h = img.size
        new_h, new_w = [math.ceil(s / self.multiple) * self.multiple for s in (h, w)]
        resized_image = transforms.functional.resize(img, (new_h, new_w))
        return resized_image


def voc2007_classification_target_transform(label, n_categories=20):
    one_hot = torch.zeros(n_categories, dtype=int)
    for instance in label.instances:
        one_hot[instance.category_id] = True
    return one_hot


def imaterialist_classification_target_transform(label, n_categories=294):
    one_hot = torch.zeros(n_categories, dtype=int)
    one_hot[label.attributes] = True
    return one_hot


def get_target_transform(dataset_str):
    if "VOC2007" in dataset_str:
        return voc2007_classification_target_transform
    elif "IMaterialist" in dataset_str:
        return imaterialist_classification_target_transform
    return None
