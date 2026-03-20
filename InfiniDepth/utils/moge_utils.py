from __future__ import annotations

from typing import Optional

import torch


_MOGE2_MODEL_CACHE: dict[tuple[str, str], torch.nn.Module] = {}


def _get_moge2_model(pretrained_model_name_or_path: str, device: torch.device) -> torch.nn.Module:
    cache_key = (pretrained_model_name_or_path, str(device))
    if cache_key in _MOGE2_MODEL_CACHE:
        return _MOGE2_MODEL_CACHE[cache_key]

    try:
        from moge.model.v2 import MoGeModel
    except ImportError as exc:
        raise ImportError(
            "MoGe is not installed. Please install it first: "
            "`pip install git+https://github.com/microsoft/MoGe.git`"
        ) from exc

    model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device)
    model.eval()
    _MOGE2_MODEL_CACHE[cache_key] = model
    return model


def _squeeze_hw(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor[0]
    raise ValueError(f"Unexpected {name} shape from MoGe-2: {tuple(tensor.shape)}")


def _squeeze_33(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 2 and tensor.shape == (3, 3):
        return tensor
    if tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[1:] == (3, 3):
        return tensor[0]
    raise ValueError(f"Unexpected {name} shape from MoGe-2: {tuple(tensor.shape)}")


def _normalized_intrinsics_to_pixel_intrinsics(
    intrinsics: torch.Tensor,
    height: int,
    width: int,
) -> tuple[float, float, float, float]:
    intrinsics = _squeeze_33(intrinsics, "intrinsics")
    fx = float(intrinsics[0, 0].item() * width)
    fy = float(intrinsics[1, 1].item() * height)
    cx = float(intrinsics[0, 2].item() * width - 0.5)
    cy = float(intrinsics[1, 2].item() * height - 0.5)
    return fx, fy, cx, cy


@torch.no_grad()
def estimate_metric_depth_and_intrinsics_with_moge2(
    image: torch.Tensor,
    pretrained_model_name_or_path: str = "Ruicheng/moge-2-vitl-normal",
) -> tuple[torch.Tensor, torch.Tensor, Optional[tuple[float, float, float, float]]]:
    """Run MoGe-2 and return dense pred depth, valid mask, and optional pixel intrinsics."""
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(f"Expected image shape [1,3,H,W], got {tuple(image.shape)}")

    device = image.device
    model = _get_moge2_model(pretrained_model_name_or_path, device)

    output = model.infer(image[0], apply_mask=True)
    if "depth" not in output:
        raise KeyError("MoGe-2 output missing key 'depth'.")

    depth_hw = _squeeze_hw(output["depth"].to(device=device, dtype=torch.float32), "depth")
    mask_hw = output.get("mask")
    if mask_hw is None:
        mask_hw = torch.ones_like(depth_hw, dtype=torch.bool, device=device)
    else:
        mask_hw = _squeeze_hw(mask_hw.to(device=device), "mask")
        if mask_hw.dtype != torch.bool:
            mask_hw = mask_hw > 0.5

    valid_mask = mask_hw & torch.isfinite(depth_hw) & (depth_hw > 0)
    pred_depth = depth_hw * valid_mask.to(depth_hw.dtype)

    moge2_intrinsics = None
    output_intrinsics = output.get("intrinsics")
    if output_intrinsics is not None:
        height, width = image.shape[-2:]
        moge2_intrinsics = _normalized_intrinsics_to_pixel_intrinsics(
            output_intrinsics.to(device=device, dtype=torch.float32),
            height=height,
            width=width,
        )

    return (
        pred_depth.unsqueeze(0).unsqueeze(0),
        valid_mask.to(torch.float32).unsqueeze(0).unsqueeze(0),
        moge2_intrinsics,
    )


@torch.no_grad()
def estimate_metric_depth_with_moge2(
    image: torch.Tensor,
    pretrained_model_name_or_path: str = "Ruicheng/moge-2-vitl-normal",
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_depth, valid_mask, _ = estimate_metric_depth_and_intrinsics_with_moge2(
        image=image,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
    )
    return pred_depth, valid_mask


@torch.no_grad()
def estimate_camera_intrinsics_with_moge2(
    image: torch.Tensor,
    pretrained_model_name_or_path: str = "Ruicheng/moge-2-vitl-normal",
) -> Optional[tuple[float, float, float, float]]:
    _, _, intrinsics = estimate_metric_depth_and_intrinsics_with_moge2(
        image=image,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
    )
    return intrinsics
