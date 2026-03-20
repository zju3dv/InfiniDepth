import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d

from InfiniDepth.gs import Gaussians

from .io_utils import load_depth
from .moge_utils import (
    estimate_camera_intrinsics_with_moge2,
    estimate_metric_depth_and_intrinsics_with_moge2,
)
from .vis_utils import build_sky_model, run_skyseg


DEPTH_SOURCES = ("auto", "gt_depth", "moge2")
OUTPUT_RESOLUTION_MODES = ("upsample", "original", "specific")


@dataclass
class DepthOutputPaths:
    depth_output_dir: str
    pcd_output_dir: str
    depth_path: str
    pcd_path: str


def scale_intrinsics(fx, fy, cx, cy, org_h, org_w, h, w):
    sx, sy = w / float(org_w), h / float(org_h)
    return fx * sx, fy * sy, cx * sx, cy * sy


def resolve_camera_intrinsics(
    fx_org: Optional[float],
    fy_org: Optional[float],
    cx_org: Optional[float],
    cy_org: Optional[float],
    org_h: int,
    org_w: int,
    fallback_intrinsics: Optional[tuple[float, float, float, float]] = None,
) -> tuple[float, float, float, float]:
    default_focal = float(max(org_h, org_w))
    default_cx = float(org_w) / 2.0
    default_cy = float(org_h) / 2.0

    fallback_fx, fallback_fy, fallback_cx, fallback_cy = fallback_intrinsics or (
        default_focal,
        default_focal,
        default_cx,
        default_cy,
    )

    fx = float(fx_org) if fx_org is not None else float(fallback_fx)
    fy = float(fy_org) if fy_org is not None else float(fallback_fy)
    cx = float(cx_org) if cx_org is not None else float(fallback_cx)
    cy = float(cy_org) if cy_org is not None else float(fallback_cy)
    return fx, fy, cx, cy


def resolve_output_size_from_mode(
    output_resolution_mode: str,
    org_h: int,
    org_w: int,
    h: int,
    w: int,
    output_size: tuple[int, int],
    upsample_ratio: int,
) -> tuple[int, int]:
    if output_resolution_mode not in OUTPUT_RESOLUTION_MODES:
        raise ValueError(
            f"Unsupported output_resolution_mode: {output_resolution_mode}. "
            f"Choose from {OUTPUT_RESOLUTION_MODES}."
        )

    if output_resolution_mode == "specific":
        h_out, w_out = int(output_size[0]), int(output_size[1])
    elif output_resolution_mode == "original":
        h_out, w_out = int(org_h), int(org_w)
    else:
        if upsample_ratio < 1:
            raise ValueError("`upsample_ratio` must be >= 1 when output_resolution_mode=upsample.")
        h_out, w_out = int(h * upsample_ratio), int(w * upsample_ratio)

    if h_out <= 0 or w_out <= 0:
        raise ValueError(f"Invalid output size ({h_out}, {w_out}). Height and width must be positive.")
    return h_out, w_out


def default_dir_by_input_file(input_path: str, output_name: str) -> str:
    base_dir = Path(input_path).resolve().parent.parent
    return os.path.join(base_dir, output_name)


def has_missing_intrinsics(
    fx_org: Optional[float],
    fy_org: Optional[float],
    cx_org: Optional[float],
    cy_org: Optional[float],
) -> bool:
    return any(v is None for v in (fx_org, fy_org, cx_org, cy_org))


def prepare_metric_depth_inputs(
    input_depth_path: Optional[str],
    input_size: tuple[int, int],
    image: torch.Tensor,
    device: torch.device,
    moge2_pretrained: str,
    depth_load_kwargs: Optional[dict] = None,
    moge2_kwargs: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, Optional[tuple[float, float, float, float]]]:
    depth_load_kwargs = depth_load_kwargs or {}
    moge2_kwargs = moge2_kwargs or {}

    if input_depth_path is not None and os.path.exists(input_depth_path):
        gt_depth, prompt_depth, gt_depth_mask = load_depth(
            input_depth_path,
            input_size,
            **depth_load_kwargs,
        )
        gt_depth = gt_depth.to(device)
        prompt_depth = prompt_depth.to(device)
        gt_depth_mask = gt_depth_mask.to(device)
        return gt_depth, prompt_depth, gt_depth_mask, True, None

    pred_depth, gt_depth_mask, moge2_intrinsics = estimate_metric_depth_and_intrinsics_with_moge2(
        image=image,
        pretrained_model_name_or_path=moge2_pretrained,
        **moge2_kwargs,
    )
    gt_depth = pred_depth.clone().to(device)
    prompt_depth = pred_depth.clone().to(device)
    gt_depth_mask = gt_depth_mask.to(device)
    return gt_depth, prompt_depth, gt_depth_mask, False, moge2_intrinsics


def resolve_camera_intrinsics_for_inference(
    fx_org: Optional[float],
    fy_org: Optional[float],
    cx_org: Optional[float],
    cy_org: Optional[float],
    org_h: int,
    org_w: int,
    image: torch.Tensor,
    moge2_pretrained: str,
    moge2_intrinsics: Optional[tuple[float, float, float, float]] = None,
) -> tuple[float, float, float, float, str]:
    intrinsics_source = "user"
    fallback_intrinsics = None

    if has_missing_intrinsics(fx_org, fy_org, cx_org, cy_org):
        if moge2_intrinsics is None:
            try:
                moge2_intrinsics = estimate_camera_intrinsics_with_moge2(
                    image=image,
                    pretrained_model_name_or_path=moge2_pretrained,
                )
            except Exception as exc:
                print(f"[Warning] Failed to estimate intrinsics with MoGe-2: {exc}")

        if moge2_intrinsics is not None:
            _, _, h, w = image.shape
            fallback_intrinsics = scale_intrinsics(
                fx=moge2_intrinsics[0],
                fy=moge2_intrinsics[1],
                cx=moge2_intrinsics[2],
                cy=moge2_intrinsics[3],
                org_h=h,
                org_w=w,
                h=org_h,
                w=org_w,
            )
            intrinsics_source = "moge2"
        else:
            intrinsics_source = "default"

    fx, fy, cx, cy = resolve_camera_intrinsics(
        fx_org=fx_org,
        fy_org=fy_org,
        cx_org=cx_org,
        cy_org=cy_org,
        org_h=org_h,
        org_w=org_w,
        fallback_intrinsics=fallback_intrinsics,
    )
    return fx, fy, cx, cy, intrinsics_source


def build_scaled_intrinsics_matrix(
    fx_org: float,
    fy_org: float,
    cx_org: float,
    cy_org: float,
    org_h: int,
    org_w: int,
    h: int,
    w: int,
    device: torch.device,
) -> tuple[float, float, float, float, torch.Tensor]:
    fx, fy, cx, cy = scale_intrinsics(fx_org, fy_org, cx_org, cy_org, org_h, org_w, h, w)
    k = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    return fx, fy, cx, cy, k


def run_optional_sky_mask(
    image: torch.Tensor,
    enable_skyseg_model: bool,
    sky_model_ckpt_path: str,
) -> Optional[torch.Tensor]:
    if not enable_skyseg_model:
        return None

    if not os.path.exists(sky_model_ckpt_path):
        raise FileNotFoundError(
            f"Sky segmentation checkpoint not found: {sky_model_ckpt_path}. "
            "Disable `enable_skyseg_model` or provide a valid path."
        )

    _, _, h, w = image.shape
    sky_model = build_sky_model(model_path=sky_model_ckpt_path)
    sky_mask_np = run_skyseg(sky_model, input_size=(320, 320), image=image)
    sky_mask_np = cv2.resize(sky_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(sky_mask_np).to(image.device)


def run_optional_sampling_sky_mask(
    image: torch.Tensor,
    enable_skyseg_model: bool,
    sky_model_ckpt_path: str,
    dilate_px: int = 5,
) -> Optional[torch.Tensor]:
    if not enable_skyseg_model:
        return None

    if not os.path.exists(sky_model_ckpt_path):
        print(f"[Warning] Sky segmentation checkpoint not found: {sky_model_ckpt_path}. Skip GS sky filtering.")
        return None

    try:
        sky_model = build_sky_model(model_path=sky_model_ckpt_path)
    except Exception as exc:
        print(f"[Warning] Failed to initialize sky segmentation model: {exc}. Skip GS sky filtering.")
        return None

    batch_masks = []
    _, _, h, w = image.shape
    dilate_px = max(int(dilate_px), 0)
    kernel = None
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilate_px + 1, 2 * dilate_px + 1),
        )

    for bi in range(image.shape[0]):
        try:
            sky_mask_np = run_skyseg(sky_model, input_size=(320, 320), image=image[bi : bi + 1])
            sky_mask_np = cv2.resize(sky_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            sky_mask_np = (sky_mask_np > 127).astype("uint8")
            if kernel is not None:
                sky_mask_np = cv2.dilate(sky_mask_np, kernel, iterations=1)
        except Exception as exc:
            print(f"[Warning] Failed to run sky segmentation: {exc}. Skip GS sky filtering.")
            return None

        batch_masks.append(torch.from_numpy(sky_mask_np.astype(bool)))

    return torch.stack(batch_masks, dim=0).to(image.device)


def apply_sky_mask_to_depth(
    pred_depthmap: torch.Tensor,
    pred_2d_uniform_depth: torch.Tensor,
    sky_mask: Optional[torch.Tensor],
    h_sample: int,
    w_sample: int,
    sky_depth_value: float = 200.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sky_mask is None:
        return pred_depthmap, pred_2d_uniform_depth

    sky_mask_resized = (
        F.interpolate(
            sky_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(h_sample, w_sample),
            mode="nearest",
        )
        .bool()
        .squeeze()
    )
    pred_depthmap[:, :, sky_mask_resized] = sky_depth_value
    sky_mask_flat = sky_mask_resized.view(-1)
    pred_2d_uniform_depth[:, sky_mask_flat, :] = sky_depth_value
    return pred_depthmap, pred_2d_uniform_depth


def resolve_depth_output_paths(
    input_image_path: str,
    model_type: str,
    output_resolution_mode: str,
    upsample_ratio: int,
    h_sample: int,
    w_sample: int,
    depth_output_dir: Optional[str] = None,
    pcd_output_dir: Optional[str] = None,
) -> DepthOutputPaths:
    depth_dir = depth_output_dir or default_dir_by_input_file(input_image_path, "pred_depth")
    pcd_dir = pcd_output_dir or default_dir_by_input_file(input_image_path, "pred_pcd")

    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)

    stem = os.path.basename(input_image_path).split(".")[0]
    if output_resolution_mode == "specific":
        depth_path = os.path.join(
            depth_dir,
            f"{model_type}_{stem}_{h_sample}x{w_sample}.png",
        )
        pcd_path = os.path.join(
            pcd_dir,
            f"{model_type}_{stem}_{h_sample}x{w_sample}.ply",
        )
    elif output_resolution_mode == "original":
        depth_path = os.path.join(depth_dir, f"{model_type}_{stem}_org_res.png")
        pcd_path = os.path.join(pcd_dir, f"{model_type}_{stem}_org_res.ply")
    else:
        depth_path = os.path.join(depth_dir, f"{model_type}_{stem}_up_{upsample_ratio}.png")
        pcd_path = os.path.join(pcd_dir, f"{model_type}_{stem}_up_{upsample_ratio}.ply")

    return DepthOutputPaths(
        depth_output_dir=depth_dir,
        pcd_output_dir=pcd_dir,
        depth_path=depth_path,
        pcd_path=pcd_path,
    )


def build_camera_matrices(
    fx_org: float,
    fy_org: float,
    cx_org: float,
    cy_org: float,
    org_h: int,
    org_w: int,
    h: int,
    w: int,
    batch: int,
    device: torch.device,
) -> tuple[float, float, float, float, torch.Tensor, torch.Tensor]:
    fx, fy, cx, cy = scale_intrinsics(fx_org, fy_org, cx_org, cy_org, org_h, org_w, h, w)
    intrinsics = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).expand(batch, -1, -1)
    extrinsics = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).expand(batch, -1, -1)
    return fx, fy, cx, cy, intrinsics, extrinsics


def filter_gaussians_by_depth_ratio(
    pixel_gaussians: Gaussians,
    extrinsics: torch.Tensor,
    keep_far_ratio: float,
) -> tuple[Gaussians, int, int, float, float]:
    camera_position = extrinsics[0, :3, 3]
    gaussian_means = pixel_gaussians.means[0]
    distances = torch.norm(gaussian_means - camera_position.unsqueeze(0), dim=-1)
    max_depth = distances.max()
    depth_threshold = max_depth * keep_far_ratio
    near_mask = distances <= depth_threshold
    num_filtered = int((~near_mask).sum().item())
    num_kept = int(near_mask.sum().item())
    filtered_gaussians = Gaussians(
        means=pixel_gaussians.means[:, near_mask, :],
        covariances=None,
        harmonics=pixel_gaussians.harmonics[:, near_mask, :, :],
        opacities=pixel_gaussians.opacities[:, near_mask],
        scales=pixel_gaussians.scales[:, near_mask, :],
        rotations=pixel_gaussians.rotations[:, near_mask, :],
    )
    return filtered_gaussians, num_filtered, num_kept, float(depth_threshold.item()), float(max_depth.item())


def filter_gaussians_by_min_opacity(pixel_gaussians: Gaussians, min_opacity: float) -> Gaussians:
    if min_opacity <= 0.0:
        return pixel_gaussians
    keep = pixel_gaussians.opacities[0] >= min_opacity
    return Gaussians(
        means=pixel_gaussians.means[:, keep, :],
        covariances=None,
        harmonics=pixel_gaussians.harmonics[:, keep, :, :],
        opacities=pixel_gaussians.opacities[:, keep],
        scales=pixel_gaussians.scales[:, keep, :],
        rotations=pixel_gaussians.rotations[:, keep, :],
    )


def filter_gaussians_by_statistical_outlier(
    pixel_gaussians: Gaussians,
    enabled: bool = True,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
) -> tuple[Gaussians, int, int]:
    if not enabled:
        n = int(pixel_gaussians.means.shape[1])
        return pixel_gaussians, 0, n

    if pixel_gaussians.means.shape[0] != 1:
        raise ValueError("Statistical outlier filtering currently expects batch size == 1.")

    num_points = int(pixel_gaussians.means.shape[1])
    if num_points == 0:
        return pixel_gaussians, 0, 0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pixel_gaussians.means[0].detach().float().cpu().numpy())

    _, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=max(int(nb_neighbors), 1),
        std_ratio=float(std_ratio),
    )

    if len(inlier_indices) == 0:
        print("[Warning] Statistical outlier filtering removed all gaussians. Keep original gaussians.")
        return pixel_gaussians, 0, num_points

    keep = torch.zeros((num_points,), dtype=torch.bool, device=pixel_gaussians.means.device)
    keep[torch.from_numpy(np.asarray(inlier_indices, dtype=np.int64)).to(keep.device)] = True

    filtered_gaussians = Gaussians(
        means=pixel_gaussians.means[:, keep, :],
        covariances=None
        if pixel_gaussians.covariances is None
        else pixel_gaussians.covariances[:, keep, ...],
        harmonics=pixel_gaussians.harmonics[:, keep, :, :],
        opacities=pixel_gaussians.opacities[:, keep],
        scales=pixel_gaussians.scales[:, keep, :],
        rotations=pixel_gaussians.rotations[:, keep, :],
    )
    return filtered_gaussians


def unpack_gaussians_for_export(
    pixel_gaussians: Gaussians,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        pixel_gaussians.means[0],
        pixel_gaussians.harmonics[0],
        pixel_gaussians.opacities[0],
        pixel_gaussians.scales[0],
        pixel_gaussians.rotations[0],
    )


def resolve_ply_output_path(
    input_image_path: str,
    model_type: str,
    output_ply_dir: Optional[str] = None,
    output_ply_name: Optional[str] = None,
) -> tuple[str, str]:
    ply_dir = output_ply_dir or default_dir_by_input_file(input_image_path, "pred_gs")
    os.makedirs(ply_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_image_path))[0]
    ply_name = output_ply_name or f"{model_type}_{stem}_gaussians.ply"
    return ply_dir, os.path.join(ply_dir, ply_name)
