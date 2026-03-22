from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Optional


def _ensure_localhost_no_proxy() -> None:
    hosts = ["127.0.0.1", "localhost", "::1"]
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        values = [value.strip() for value in current.split(",") if value.strip()]
        changed = False
        for host in hosts:
            if host not in values:
                values.append(host)
                changed = True
        if changed or not current:
            os.environ[key] = ",".join(values)


_ensure_localhost_no_proxy()


def _ensure_hf_cache_dirs() -> None:
    hf_home = os.environ.get("HF_HOME", "/tmp/huggingface")
    hub_cache = os.environ.get("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    assets_cache = os.environ.get("HF_ASSETS_CACHE", os.path.join(hf_home, "assets"))

    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = hub_cache
    os.environ["HF_ASSETS_CACHE"] = assets_cache
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub_cache)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(hub_cache, exist_ok=True)
    os.makedirs(assets_cache, exist_ok=True)


_ensure_hf_cache_dirs()

import cv2
import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from InfiniDepth.gs import GSPixelAlignPredictor, export_ply
from InfiniDepth.utils.gs_utils import (
    _build_sparse_uniform_gaussians,
)
from InfiniDepth.utils.hf_demo_utils import (
    DemoArtifacts,
    ensure_session_output_dir,
    export_point_cloud_assets,
    preview_depth_file,
    save_demo_artifacts,
    scan_example_cases,
)
from InfiniDepth.utils.hf_gs_viewer import (
    APP_TEMP_ROOT as GS_VIEWER_ROOT,
    build_embedded_viewer_html,
    build_viewer_error_html,
)
from InfiniDepth.utils.inference_utils import (
    apply_sky_mask_to_depth,
    build_camera_matrices,
    build_scaled_intrinsics_matrix,
    filter_gaussians_by_statistical_outlier,
    prepare_metric_depth_inputs,
    resolve_camera_intrinsics_for_inference,
    resolve_output_size_from_mode,
    run_optional_sampling_sky_mask,
    run_optional_sky_mask,
    unpack_gaussians_for_export,
)
from InfiniDepth.utils.io_utils import depth_to_disparity
from InfiniDepth.utils.model_utils import build_model
from InfiniDepth.utils.sampling_utils import SAMPLING_METHODS


APP_ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = APP_ROOT / "example_huggingface"
INPUT_SIZE = (768, 1024)
APP_NAME = "infinidepth-hf-demo"
TASK_CHOICES = ["Depth", "3DGS"]
MODEL_CHOICES = ["InfiniDepth", "InfiniDepth_DepthSensor"]
OUTPUT_MODE_CHOICES = ["upsample", "original", "specific"]
GS_SAMPLE_POINT_NUM = 2000000
GS_COORD_DETERMINISTIC_SAMPLING = True

LOCAL_DEPTH_MODEL_PATHS = {
    "InfiniDepth": APP_ROOT / "checkpoints/depth/infinidepth.ckpt",
    "InfiniDepth_DepthSensor": APP_ROOT / "checkpoints/depth/infinidepth_depthsensor.ckpt",
}
LOCAL_GS_MODEL_PATHS = {
    "InfiniDepth": APP_ROOT / "checkpoints/gs/infinidepth_gs.ckpt",
    "InfiniDepth_DepthSensor": APP_ROOT / "checkpoints/gs/infinidepth_depthsensor_gs.ckpt",
}
HF_REPO_ID = "ritianyu/InfiniDepth"
HF_DEPTH_FILENAMES = {
    "InfiniDepth": "infinidepth.ckpt",
    "InfiniDepth_DepthSensor": "infinidepth_depthsensor.ckpt",
}
HF_GS_FILENAMES = {
    "InfiniDepth": "infinidepth_gs.ckpt",
    "InfiniDepth_DepthSensor": "infinidepth_depthsensor_gs.ckpt",
}
LOCAL_MOGE2_PATH = APP_ROOT / "checkpoints/moge-2-vitl-normal/model.pt"
HF_MOGE2_FILENAME = "moge2.pt"
LOCAL_SKYSEG_PATH = APP_ROOT / "checkpoints/sky/skyseg.onnx"
HF_SKYSEG_FILENAME = "skyseg.onnx"

EXAMPLE_CASES = scan_example_cases(EXAMPLES_DIR)
EXAMPLE_LOOKUP = {case.name: case for case in EXAMPLE_CASES}
DEFAULT_EXAMPLE_NAME = EXAMPLE_CASES[0].name if EXAMPLE_CASES else None
DEFAULT_EXAMPLE_INDEX = 0 if EXAMPLE_CASES else None
EXAMPLE_GALLERY_ITEMS = [(case.image_path, case.gallery_caption) for case in EXAMPLE_CASES]
DEPTH_VIEW_TAB_ID = "pcd-viewer-tab"
GS_VIEW_TAB_ID = "gs-viewer-tab"
gr.set_static_paths(paths=[str(GS_VIEWER_ROOT)])

CSS = """
#top-workspace {
    align-items: stretch;
}

#controls-column,
#inputs-column,
#outputs-column {
    min-width: 0;
}

#example-gallery {
    min-height: 280px;
}

#input-image {
    min-height: 420px;
}

#input-depth-preview {
    min-height: 240px;
}

#depth-model3d-viewer {
    height: 700px;
}

#depth-model3d-viewer canvas,
#depth-model3d-viewer model-viewer,
#depth-model3d-viewer .wrap,
#depth-model3d-viewer .container {
    height: 100% !important;
    max-height: 100% !important;
}

#gs-viewer-html {
    min-height: 748px;
    padding-bottom: 0.75rem;
}

#gs-viewer-html iframe {
    display: block;
    width: 100%;
    height: 700px !important;
    min-height: 700px !important;
}

#depth-preview,
#depth-comparison,
#depth-color {
    min-height: 260px;
}
"""


def _ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise gr.Error("CUDA GPU is required for InfiniDepth inference in this demo.")


def _resolve_repo_asset(local_path: Path, filename: str) -> str:
    if local_path.exists():
        return str(local_path)

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
    )


@lru_cache(maxsize=2)
def _resolve_depth_checkpoint(model_type: str) -> str:
    return _resolve_repo_asset(LOCAL_DEPTH_MODEL_PATHS[model_type], HF_DEPTH_FILENAMES[model_type])


@lru_cache(maxsize=2)
def _resolve_gs_checkpoint(model_type: str) -> str:
    return _resolve_repo_asset(LOCAL_GS_MODEL_PATHS[model_type], HF_GS_FILENAMES[model_type])


@lru_cache(maxsize=1)
def _resolve_skyseg_path() -> str:
    return _resolve_repo_asset(LOCAL_SKYSEG_PATH, HF_SKYSEG_FILENAME)


@lru_cache(maxsize=1)
def _resolve_moge2_source() -> str:
    return _resolve_repo_asset(LOCAL_MOGE2_PATH, HF_MOGE2_FILENAME)


@lru_cache(maxsize=1)
def _preload_repo_assets() -> tuple[str, ...]:
    depth_paths = tuple(_resolve_depth_checkpoint(model_type) for model_type in MODEL_CHOICES)
    gs_paths = tuple(_resolve_gs_checkpoint(model_type) for model_type in MODEL_CHOICES)
    return depth_paths + gs_paths + (_resolve_moge2_source(), _resolve_skyseg_path())


@lru_cache(maxsize=2)
def _load_model(model_type: str):
    _ensure_cuda()
    model_path = _resolve_depth_checkpoint(model_type)
    return build_model(model_type=model_type, model_path=model_path)


@lru_cache(maxsize=4)
def _load_gs_predictor(model_type: str, dino_feature_dim: int):
    _ensure_cuda()
    predictor = GSPixelAlignPredictor(dino_feature_dim=dino_feature_dim).to(torch.device("cuda"))
    predictor.load_from_infinidepth_gs_checkpoint(_resolve_gs_checkpoint(model_type))
    predictor.eval()
    return predictor


def _to_optional_float(value: Optional[float]) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise gr.Error("Input image must be an RGB image.")

    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0 if image.max() <= 1.0 else 255.0)
        if image.max() <= 1.0:
            image = image * 255.0
        return image.astype(np.uint8)

    return np.clip(image, 0, 255).astype(np.uint8)


def _prepare_image_tensors(image_rgb: np.ndarray) -> tuple[np.ndarray, torch.Tensor, tuple[int, int]]:
    image_rgb = _to_rgb_uint8(image_rgb)
    org_h, org_w = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, INPUT_SIZE[::-1], interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return image_rgb, image, (org_h, org_w)


def _format_depth_status(
    model_type: str,
    metric_depth_source: str,
    intrinsics_source: str,
    output_hw: tuple[int, int],
    depth_file: Optional[str],
) -> str:
    depth_label = Path(depth_file).name if depth_file else "None"
    return (
        f"Task: `Depth`\n\n"
        f"Model: `{model_type}`\n\n"
        f"Input depth: `{depth_label}`\n\n"
        f"Metric alignment source: `{metric_depth_source}`\n\n"
        f"Camera intrinsics source: `{intrinsics_source}`\n\n"
        f"Output size: `{output_hw[0]} x {output_hw[1]}`"
    )


def _format_gs_status(
    model_type: str,
    metric_depth_source: str,
    intrinsics_source: str,
    depth_file: Optional[str],
    gaussian_count: int,
) -> str:
    depth_label = Path(depth_file).name if depth_file else "None"
    return (
        f"Task: `GS`\n\n"
        f"Model: `{model_type}`\n\n"
        f"Input depth: `{depth_label}`\n\n"
        f"Metric alignment source: `{metric_depth_source}`\n\n"
        f"Camera intrinsics source: `{intrinsics_source}`\n\n"
        f"Exported gaussians: `{gaussian_count}`"
    )


def _load_example_image(example_name: str) -> tuple[np.ndarray, Optional[str], Optional[np.ndarray], str, str]:
    if not example_name:
        raise gr.Error("Select an example case first.")
    case = EXAMPLE_LOOKUP[example_name]

    image_bgr = cv2.imread(case.image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise gr.Error(f"Failed to load example image: {case.image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    depth_path = case.depth_path
    preview = None
    detail = f"Loaded example `{case.name}`."
    model_type = "InfiniDepth"

    if depth_path is not None:
        preview, depth_msg = preview_depth_file(depth_path)
        detail = f"Loaded example `{case.name}` with paired depth. {depth_msg}"
        model_type = "InfiniDepth_DepthSensor"

    return image_rgb, depth_path, preview, model_type, detail


def _selected_example_message(example_name: Optional[str]) -> str:
    if not example_name or example_name not in EXAMPLE_LOOKUP:
        return "Select an example thumbnail, then click `Load Example`."

    case = EXAMPLE_LOOKUP[example_name]
    mode_label = "RGB + depth" if case.has_depth else "RGB only"
    return f"Selected example: `{case.name}` ({mode_label})"


def _select_example(evt: gr.SelectData):
    if not EXAMPLE_CASES or evt.index is None:
        return None, _selected_example_message(None)

    index = evt.index
    if isinstance(index, (tuple, list)):
        index = index[0]
    case = EXAMPLE_CASES[int(index)]
    return case.name, _selected_example_message(case.name)


def _primary_view_for_task(task_type: str):
    selected_tab = GS_VIEW_TAB_ID if task_type == "3DGS" else DEPTH_VIEW_TAB_ID
    return gr.update(selected=selected_tab)


def _update_depth_preview(depth_path: Optional[str]) -> tuple[Optional[np.ndarray], str]:
    try:
        return preview_depth_file(depth_path)
    except Exception as exc:
        raise gr.Error(f"Failed to preview depth file: {exc}") from exc


def _settings_visibility(task_type: str, output_resolution_mode: str):
    is_depth = task_type == "Depth"
    return (
        gr.update(visible=is_depth),
        gr.update(visible=is_depth and output_resolution_mode == "upsample"),
        gr.update(visible=is_depth and output_resolution_mode == "specific"),
        gr.update(visible=is_depth and output_resolution_mode == "specific"),
        gr.update(visible=is_depth),
    )


def _normalize_filtered_gaussians(filtered_result):
    if isinstance(filtered_result, tuple):
        return filtered_result[0]
    return filtered_result


@torch.no_grad()
def _run_depth_inference(
    image: np.ndarray,
    depth_file: Optional[str],
    model_type: str,
    output_resolution_mode: str,
    upsample_ratio: int,
    specific_height: int,
    specific_width: int,
    enable_skyseg_model: bool,
    filter_point_cloud: bool,
    fx_org: Optional[float],
    fy_org: Optional[float],
    cx_org: Optional[float],
    cy_org: Optional[float],
    request: gr.Request,
):
    _ensure_cuda()
    if image is None:
        raise gr.Error("Upload an image or load an example before running inference.")
    if model_type == "InfiniDepth_DepthSensor" and not depth_file:
        raise gr.Error("InfiniDepth_DepthSensor requires an input depth file.")

    skyseg_path = _resolve_skyseg_path() if enable_skyseg_model else None

    image_rgb, image_tensor, (org_h, org_w) = _prepare_image_tensors(image)
    device = torch.device("cuda")
    image_tensor = image_tensor.to(device)
    model = _load_model(model_type)

    gt_depth, prompt_depth, gt_depth_mask, use_gt_depth, moge2_intrinsics = prepare_metric_depth_inputs(
        input_depth_path=depth_file,
        input_size=INPUT_SIZE,
        image=image_tensor,
        device=device,
        moge2_pretrained=_resolve_moge2_source(),
    )

    gt_disp = depth_to_disparity(gt_depth)
    prompt_disp = depth_to_disparity(prompt_depth)

    fx_org, fy_org, cx_org, cy_org, intrinsics_source = resolve_camera_intrinsics_for_inference(
        fx_org=_to_optional_float(fx_org),
        fy_org=_to_optional_float(fy_org),
        cx_org=_to_optional_float(cx_org),
        cy_org=_to_optional_float(cy_org),
        org_h=org_h,
        org_w=org_w,
        image=image_tensor,
        moge2_pretrained=_resolve_moge2_source(),
        moge2_intrinsics=moge2_intrinsics,
    )

    _, _, h, w = image_tensor.shape
    fx, fy, cx, cy, _ = build_scaled_intrinsics_matrix(
        fx_org=fx_org,
        fy_org=fy_org,
        cx_org=cx_org,
        cy_org=cy_org,
        org_h=org_h,
        org_w=org_w,
        h=h,
        w=w,
        device=image_tensor.device,
    )

    sky_mask = run_optional_sky_mask(
        image=image_tensor,
        enable_skyseg_model=enable_skyseg_model,
        sky_model_ckpt_path=skyseg_path or str(LOCAL_SKYSEG_PATH),
    )

    h_out, w_out = resolve_output_size_from_mode(
        output_resolution_mode=output_resolution_mode,
        org_h=org_h,
        org_w=org_w,
        h=h,
        w=w,
        output_size=(int(specific_height), int(specific_width)),
        upsample_ratio=int(upsample_ratio),
    )

    query_2d_uniform_coord = SAMPLING_METHODS["2d_uniform"]((h_out, w_out)).unsqueeze(0).to(device)
    pred_2d_uniform_depth, _ = model.inference(
        image=image_tensor,
        query_coord=query_2d_uniform_coord,
        gt_depth=gt_disp,
        gt_depth_mask=gt_depth_mask,
        prompt_depth=prompt_disp,
        prompt_mask=prompt_disp > 0,
    )
    pred_depthmap = pred_2d_uniform_depth.permute(0, 2, 1).reshape(1, 1, h_out, w_out)

    pred_depthmap, pred_2d_uniform_depth = apply_sky_mask_to_depth(
        pred_depthmap=pred_depthmap,
        pred_2d_uniform_depth=pred_2d_uniform_depth,
        sky_mask=sky_mask,
        h_sample=h_out,
        w_sample=w_out,
        sky_depth_value=200.0,
    )

    session_hash = getattr(request, "session_hash", None)
    output_dir = ensure_session_output_dir(APP_NAME, session_hash)

    pred_depth_np = pred_depthmap.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    artifacts = save_demo_artifacts(image_rgb=image_rgb, pred_depth=pred_depth_np, output_dir=output_dir)
    ply_path, glb_path = export_point_cloud_assets(
        sampled_coord=query_2d_uniform_coord.squeeze(0).cpu(),
        sampled_depth=pred_2d_uniform_depth.squeeze(0).squeeze(-1).cpu(),
        rgb_image=image_tensor.squeeze(0).cpu(),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        output_dir=output_dir,
        filter_flying_points=filter_point_cloud,
    )
    artifacts = DemoArtifacts(
        comparison_path=artifacts.comparison_path,
        color_depth_path=artifacts.color_depth_path,
        gray_depth_path=artifacts.gray_depth_path,
        raw_depth_path=artifacts.raw_depth_path,
        ply_path=ply_path,
        glb_path=glb_path,
    )

    metric_depth_source = "user depth" if use_gt_depth and depth_file else "MoGe-2"
    status = _format_depth_status(
        model_type=model_type,
        metric_depth_source=metric_depth_source,
        intrinsics_source=intrinsics_source,
        output_hw=(h_out, w_out),
        depth_file=depth_file,
    )
    return (
        status,
        artifacts.comparison_path,
        artifacts.color_depth_path,
        artifacts.gray_depth_path,
        glb_path,
        artifacts.download_files(),
        None,
        None,
    )


@torch.no_grad()
def _run_gs_inference(
    image: np.ndarray,
    depth_file: Optional[str],
    model_type: str,
    enable_skyseg_model: bool,
    fx_org: Optional[float],
    fy_org: Optional[float],
    cx_org: Optional[float],
    cy_org: Optional[float],
    request: gr.Request,
):
    _ensure_cuda()
    if image is None:
        raise gr.Error("Upload an image or load an example before running inference.")
    if model_type == "InfiniDepth_DepthSensor" and not depth_file:
        raise gr.Error("InfiniDepth_DepthSensor requires an input depth file for GS inference.")

    image_rgb, image_tensor, (org_h, org_w) = _prepare_image_tensors(image)
    del image_rgb
    device = torch.device("cuda")
    image_tensor = image_tensor.to(device)
    model = _load_model(model_type)

    gt_depth, prompt_depth, gt_depth_mask, use_gt_depth, moge2_intrinsics = prepare_metric_depth_inputs(
        input_depth_path=depth_file,
        input_size=INPUT_SIZE,
        image=image_tensor,
        device=device,
        moge2_pretrained=_resolve_moge2_source(),
    )

    gt_disp = depth_to_disparity(gt_depth)
    prompt_disp = depth_to_disparity(prompt_depth)

    fx_org, fy_org, cx_org, cy_org, intrinsics_source = resolve_camera_intrinsics_for_inference(
        fx_org=_to_optional_float(fx_org),
        fy_org=_to_optional_float(fy_org),
        cx_org=_to_optional_float(cx_org),
        cy_org=_to_optional_float(cy_org),
        org_h=org_h,
        org_w=org_w,
        image=image_tensor,
        moge2_pretrained=_resolve_moge2_source(),
        moge2_intrinsics=moge2_intrinsics,
    )

    b, _, h, w = image_tensor.shape
    _, _, _, _, intrinsics, extrinsics = build_camera_matrices(
        fx_org=fx_org,
        fy_org=fy_org,
        cx_org=cx_org,
        cy_org=cy_org,
        org_h=org_h,
        org_w=org_w,
        h=h,
        w=w,
        batch=b,
        device=device,
    )

    skyseg_path = _resolve_skyseg_path() if enable_skyseg_model else str(LOCAL_SKYSEG_PATH)
    sky_mask = run_optional_sampling_sky_mask(
        image=image_tensor,
        enable_skyseg_model=enable_skyseg_model,
        sky_model_ckpt_path=skyseg_path,
        dilate_px=0,
    )

    depthmap, dino_tokens, query_3d_uniform_coord, pred_depth_3d = model.inference_for_gs(
        image=image_tensor,
        intrinsics=intrinsics,
        gt_depth=gt_disp,
        gt_depth_mask=gt_depth_mask,
        prompt_depth=prompt_disp,
        prompt_mask=prompt_disp > 0,
        sky_mask=sky_mask,
        sample_point_num=GS_SAMPLE_POINT_NUM,
        coord_deterministic_sampling=GS_COORD_DETERMINISTIC_SAMPLING,
    )
    if query_3d_uniform_coord is None or pred_depth_3d is None:
        raise gr.Error("GS inference did not return 3D-uniform query outputs.")

    gs_predictor = _load_gs_predictor(model_type, int(dino_tokens.shape[-1]))
    dense_gaussians = gs_predictor(
        image=image_tensor,
        depthmap=depthmap,
        dino_tokens=dino_tokens,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )

    pixel_gaussians = _build_sparse_uniform_gaussians(
        dense_gaussians=dense_gaussians,
        query_3d_uniform_coord=query_3d_uniform_coord,
        pred_depth_3d=pred_depth_3d,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        h=h,
        w=w,
    )
    pixel_gaussians = _normalize_filtered_gaussians(filter_gaussians_by_statistical_outlier(pixel_gaussians))
    gaussian_count = int(pixel_gaussians.means.shape[1])
    if gaussian_count == 0:
        raise gr.Error("No valid gaussians remained after filtering.")

    means, harmonics, opacities, scales, rotations = unpack_gaussians_for_export(pixel_gaussians)

    session_hash = getattr(request, "session_hash", None)
    output_dir = ensure_session_output_dir(APP_NAME, session_hash)
    ply_path = output_dir / "gaussians.ply"

    export_ply(
        means=means,
        harmonics=harmonics,
        opacities=opacities,
        path=ply_path,
        scales=scales,
        rotations=rotations,
        focal_length_px=(fx_org, fy_org),
        principal_point_px=(cx_org, cy_org),
        image_shape=(org_h, org_w),
        extrinsic_matrix=extrinsics[0],
    )

    try:
        gs_viewer_html = build_embedded_viewer_html(ply_path)
    except Exception as exc:
        print(f"[Warning] Failed to build embedded GS viewer: {exc}")
        gs_viewer_html = build_viewer_error_html(str(exc), ply_path)

    metric_depth_source = "user depth" if use_gt_depth and depth_file else "MoGe-2"
    status = _format_gs_status(
        model_type=model_type,
        metric_depth_source=metric_depth_source,
        intrinsics_source=intrinsics_source,
        depth_file=depth_file,
        gaussian_count=gaussian_count,
    )
    download_files = [str(ply_path)]

    return (
        status,
        None,
        None,
        None,
        None,
        None,
        gs_viewer_html,
        download_files,
    )


def _run_inference(
    task_type: str,
    image: np.ndarray,
    depth_file: Optional[str],
    model_type: str,
    output_resolution_mode: str,
    upsample_ratio: int,
    specific_height: int,
    specific_width: int,
    enable_skyseg_model: bool,
    filter_point_cloud: bool,
    fx_org: Optional[float],
    fy_org: Optional[float],
    cx_org: Optional[float],
    cy_org: Optional[float],
    request: gr.Request,
):
    if task_type == "3DGS":
        return _run_gs_inference(
            image=image,
            depth_file=depth_file,
            model_type=model_type,
            enable_skyseg_model=enable_skyseg_model,
            fx_org=fx_org,
            fy_org=fy_org,
            cx_org=cx_org,
            cy_org=cy_org,
            request=request,
        )

    return _run_depth_inference(
        image=image,
        depth_file=depth_file,
        model_type=model_type,
        output_resolution_mode=output_resolution_mode,
        upsample_ratio=upsample_ratio,
        specific_height=specific_height,
        specific_width=specific_width,
        enable_skyseg_model=enable_skyseg_model,
        filter_point_cloud=filter_point_cloud,
        fx_org=fx_org,
        fy_org=fy_org,
        cx_org=cx_org,
        cy_org=cy_org,
        request=request,
    )


def _clear_outputs():
    return "", None, None, None, None, None, "", None


with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# InfiniDepth Demo")
    gr.Markdown(
        "Switch between depth inference and GS inference, choose `InfiniDepth` or `InfiniDepth_DepthSensor`, preview the generated result directly in the demo, and download exported assets."
    )

    selected_example_name = gr.State(DEFAULT_EXAMPLE_NAME)

    with gr.Row(elem_id="top-workspace"):
        with gr.Column(scale=4, min_width=320, elem_id="controls-column"):
            task_type = gr.Radio(label="Inference Task", choices=TASK_CHOICES, value="Depth")
            model_type = gr.Radio(label="Model Type", choices=MODEL_CHOICES, value="InfiniDepth")

            gr.Markdown("### Example Data")
            example_gallery = gr.Gallery(
                value=EXAMPLE_GALLERY_ITEMS,
                label="Example Data",
                show_label=False,
                columns=2,
                height=280,
                object_fit="cover",
                allow_preview=False,
                selected_index=DEFAULT_EXAMPLE_INDEX,
                elem_id="example-gallery",
            )
            example_selection = gr.Markdown(_selected_example_message(DEFAULT_EXAMPLE_NAME))
            load_example_btn = gr.Button("Load Example")

            with gr.Accordion("Depth Settings", open=True):
                output_resolution_mode = gr.Dropdown(
                    label="Output Resolution Mode",
                    choices=OUTPUT_MODE_CHOICES,
                    value="upsample",
                )
                upsample_ratio = gr.Slider(label="Upsample Ratio", minimum=1, maximum=4, step=1, value=1)
                specific_height = gr.Number(label="Specific Height", value=INPUT_SIZE[0], precision=0, visible=False)
                specific_width = gr.Number(label="Specific Width", value=INPUT_SIZE[1], precision=0, visible=False)
                enable_skyseg_model = gr.Checkbox(label="Apply Sky Mask", value=False)
                filter_point_cloud = gr.Checkbox(label="Filter Flying Points", value=True)

            with gr.Accordion("Optional Camera Intrinsics", open=False):
                fx_org = gr.Textbox(label="fx", value="", placeholder="auto")
                fy_org = gr.Textbox(label="fy", value="", placeholder="auto")
                cx_org = gr.Textbox(label="cx", value="", placeholder="auto")
                cy_org = gr.Textbox(label="cy", value="", placeholder="auto")

        with gr.Column(scale=5, min_width=360, elem_id="inputs-column"):
            input_image = gr.Image(
                label="Input Image",
                image_mode="RGB",
                type="numpy",
                sources=["upload", "clipboard", "webcam"],
                height=420,
                elem_id="input-image",
            )
            input_depth_file = gr.File(
                label="Optional Depth File",
                type="filepath",
                file_types=[".png", ".npy", ".npz", ".h5", ".hdf5", ".exr"],
            )
            input_depth_preview = gr.Image(
                label="Input Depth Preview",
                type="numpy",
                height=240,
                elem_id="input-depth-preview",
            )
            depth_info = gr.Markdown("No input depth loaded.")
            submit_btn = gr.Button("Run Inference", variant="primary")

        with gr.Column(scale=8, min_width=640, elem_id="outputs-column"):
            status_output = gr.Markdown()

            with gr.Tabs(selected=DEPTH_VIEW_TAB_ID, elem_id="primary-view-tabs") as primary_view_tabs:
                with gr.Tab("PCD Viewer", id=DEPTH_VIEW_TAB_ID, render_children=True):
                    depth_model_3d = gr.Model3D(
                        label="Point Cloud Viewer",
                        display_mode="solid",
                        clear_color=[1.0, 1.0, 1.0, 1.0],
                        height=700,
                        elem_id="depth-model3d-viewer",
                    )
                with gr.Tab("GS Viewer", id=GS_VIEW_TAB_ID, render_children=True):
                    gs_viewer_html = gr.HTML(elem_id="gs-viewer-html")

            with gr.Tabs(elem_id="secondary-output-tabs"):
                with gr.Tab("Depth Analysis", render_children=True):
                    depth_comparison = gr.Image(
                        label="RGB vs Depth",
                        type="filepath",
                        height=280,
                        elem_id="depth-comparison",
                    )
                    with gr.Row():
                        color_depth = gr.Image(
                            label="Colorized Depth",
                            type="filepath",
                            height=260,
                            elem_id="depth-color",
                        )
                        gray_depth = gr.Image(
                            label="Grayscale Depth",
                            type="filepath",
                            height=260,
                            elem_id="depth-preview",
                        )
                with gr.Tab("Downloads", render_children=True):
                    with gr.Row():
                        depth_download_files = gr.File(label="Depth Files", type="filepath")
                        gs_download_files = gr.File(label="GS Files", type="filepath")

    task_type.change(
        fn=_settings_visibility,
        inputs=[task_type, output_resolution_mode],
        outputs=[output_resolution_mode, upsample_ratio, specific_height, specific_width, filter_point_cloud],
    )
    task_type.change(
        fn=_primary_view_for_task,
        inputs=[task_type],
        outputs=[primary_view_tabs],
    )

    output_resolution_mode.change(
        fn=_settings_visibility,
        inputs=[task_type, output_resolution_mode],
        outputs=[output_resolution_mode, upsample_ratio, specific_height, specific_width, filter_point_cloud],
    )

    example_gallery.select(
        fn=_select_example,
        outputs=[selected_example_name, example_selection],
    )

    input_depth_file.change(
        fn=_update_depth_preview,
        inputs=[input_depth_file],
        outputs=[input_depth_preview, depth_info],
    )

    load_example_btn.click(
        fn=_load_example_image,
        inputs=[selected_example_name],
        outputs=[input_image, input_depth_file, input_depth_preview, model_type, depth_info],
    )

    submit_btn.click(
        fn=_primary_view_for_task,
        inputs=[task_type],
        outputs=[primary_view_tabs],
    ).then(
        fn=_clear_outputs,
        outputs=[
            status_output,
            depth_comparison,
            color_depth,
            gray_depth,
            depth_model_3d,
            depth_download_files,
            gs_viewer_html,
            gs_download_files,
        ],
    ).then(
        fn=_run_inference,
        inputs=[
            task_type,
            input_image,
            input_depth_file,
            model_type,
            output_resolution_mode,
            upsample_ratio,
            specific_height,
            specific_width,
            enable_skyseg_model,
            filter_point_cloud,
            fx_org,
            fy_org,
            cx_org,
            cy_org,
        ],
        outputs=[
            status_output,
            depth_comparison,
            color_depth,
            gray_depth,
            depth_model_3d,
            depth_download_files,
            gs_viewer_html,
            gs_download_files,
        ],
    )


if __name__ == "__main__":
    _preload_repo_assets()
    demo.queue().launch()
