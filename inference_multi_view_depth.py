from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import json

import cv2
import numpy as np
import torch
import tyro
from PIL import Image, ImageOps

from inference_depth import (
    DepthInferenceArgs,
    build_point_cloud_from_depth_result,
    load_depth_model,
    run_depth_inference,
    save_depth_inference_result,
    scale_align_depth_result,
)
from InfiniDepth.utils.inference_utils import (
    ensure_homogeneous_extrinsics,
    resolve_sequence_output_paths,
    scale_intrinsics_matrix_np,
)
from InfiniDepth.utils.io_utils import merge_point_clouds, save_point_cloud


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_DEPTH_EXTENSIONS = {".png", ".npy", ".npz", ".h5", ".hdf5", ".exr"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_CAMERA_PARAM_EXTENSIONS = {".txt"}


@dataclass
class SequenceDepthInferenceArgs:
    input_path: str
    input_depth_path: Optional[str] = None
    camera_intrinsics_dir: Optional[str] = None
    camera_extrinsics_dir: Optional[str] = None
    input_mode: Literal["auto", "images", "video"] = "auto"
    output_root: Optional[str] = None
    save_frame_pcd: bool = True
    save_merged_pcd: bool = True

    model_type: str = "InfiniDepth"
    depth_model_path: str = "checkpoints/depth/infinidepth.ckpt"
    moge2_pretrained: str = "checkpoints/moge-2-vitl-normal/model.pt"

    input_size: tuple[int, int] = (768, 1024)
    output_size: tuple[int, int] = (768, 1024)
    output_resolution_mode: Literal["upsample", "original", "specific"] = "upsample"
    upsample_ratio: int = 1

    enable_skyseg_model: bool = False
    sky_model_ckpt_path: str = "checkpoints/sky/skyseg.onnx"

    da3_model: str = "checkpoints/da3"
    da3_process_res: int = 504
    da3_process_res_method: str = "upper_bound_resize"
    da3_ref_view_strategy: Optional[str] = None
    align_to_da3_depth: bool = True
    da3_scale_align_conf_threshold: float = 0.0
    da3_scale_align_min_valid_pixels: int = 128


def _detect_input_mode(input_path: Path, input_mode: str) -> str:
    if input_mode != "auto":
        return input_mode
    if input_path.is_dir():
        return "images"
    suffix = input_path.suffix.lower()
    if suffix in _VIDEO_EXTENSIONS:
        return "video"
    if suffix in _IMAGE_EXTENSIONS:
        return "images"
    raise ValueError(f"Unable to infer input mode from path: {input_path}")


def _collect_paths_from_input(input_path: Path, allowed_extensions: set[str], label: str) -> list[str]:
    if input_path.is_file():
        if input_path.suffix.lower() not in allowed_extensions:
            raise ValueError(f"Unsupported {label} file: {input_path}")
        return [str(input_path.resolve())]

    frame_paths = sorted(
        str(path.resolve())
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_extensions
    )
    if not frame_paths:
        raise ValueError(f"No {label} files found under: {input_path}")
    return frame_paths


def _collect_camera_parameter_paths(camera_dir: str, expected_count: int, label: str) -> list[str]:
    directory = Path(camera_dir).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"{label} directory not found: {directory}")
    if not directory.is_dir():
        raise ValueError(f"{label} path must be a directory: {directory}")

    file_paths = sorted(
        str(path.resolve())
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _CAMERA_PARAM_EXTENSIONS
    )
    if not file_paths:
        raise ValueError(f"No {label} files found under: {directory}")
    if len(file_paths) != expected_count:
        raise RuntimeError(
            f"{label} count {len(file_paths)} does not match RGB frame count {expected_count}."
        )
    return file_paths


def _extract_video_frames(
    video_path: Path,
    frame_output_dir: str,
    *,
    frame_prefix: str,
) -> tuple[list[str], Optional[float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_paths: list[str] = []
    frame_index = 0
    output_dir = Path(frame_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_path = output_dir / f"{frame_prefix}_{frame_index:06d}.png"
        if not cv2.imwrite(str(frame_path), frame):
            cap.release()
            raise RuntimeError(f"Failed to write extracted frame: {frame_path}")
        frame_paths.append(str(frame_path))
        frame_index += 1

    cap.release()
    if not frame_paths:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    fps_value = float(fps) if fps and fps > 0 else None
    return frame_paths, fps_value


def _prepare_rgb_inputs(input_path: Path, input_mode: str, output_paths) -> tuple[list[str], Optional[float], str]:
    if input_mode == "video":
        frame_paths, source_fps = _extract_video_frames(
            input_path,
            output_paths.frame_source_dir,
            frame_prefix="frame",
        )
        return frame_paths, source_fps, "video"

    frame_paths = _collect_paths_from_input(input_path, _IMAGE_EXTENSIONS, "image")
    source_type = "image_file" if input_path.is_file() else "image_dir"
    return frame_paths, None, source_type


def _prepare_depth_inputs(
    input_depth_path: Optional[str],
    rgb_input_mode: str,
    output_paths,
    expected_count: int,
) -> tuple[Optional[list[str]], Optional[float], Optional[str]]:
    if input_depth_path is None:
        return None, None, None

    depth_path = Path(input_depth_path).expanduser().resolve()
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth input path not found: {depth_path}")

    if rgb_input_mode == "video" and depth_path.is_file() and depth_path.suffix.lower() in _VIDEO_EXTENSIONS:
        depth_source_dir = Path(output_paths.frame_source_dir).parent / "depth_source"
        depth_paths, depth_fps = _extract_video_frames(
            depth_path,
            str(depth_source_dir),
            frame_prefix="depth",
        )
        depth_source_type = "video"
    else:
        depth_paths = _collect_paths_from_input(depth_path, _DEPTH_EXTENSIONS, "depth")
        depth_fps = None
        depth_source_type = "file" if depth_path.is_file() else "dir"

    if len(depth_paths) != expected_count:
        raise RuntimeError(
            f"Depth input count {len(depth_paths)} does not match RGB frame count {expected_count}."
        )
    return depth_paths, depth_fps, depth_source_type


def _read_image_size(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image)
        return pil_image.height, pil_image.width


def _import_depth_anything3():
    try:
        from depth_anything_3 import DepthAnything3
        return DepthAnything3
    except ImportError:
        try:
            from depth_anything_3.api import DepthAnything3
            return DepthAnything3
        except ImportError as exc:
            raise ImportError(
                "Depth Anything 3 is not installed. Install it first, for example: "
                "`pip install git+https://github.com/ByteDance-Seed/depth-anything-3.git`."
            ) from exc


def _to_numpy(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _prediction_value(prediction, name: str):
    value = getattr(prediction, name, None)
    if value is None:
        aux = getattr(prediction, "aux", None)
        if isinstance(aux, dict):
            value = aux.get(name)
    return value


def _resolve_processed_hw(processed_images: Optional[np.ndarray], depth: np.ndarray) -> tuple[int, int]:
    if processed_images is not None:
        arr = np.asarray(processed_images)
        if arr.ndim == 4:
            if arr.shape[1] in (1, 3):
                return int(arr.shape[2]), int(arr.shape[3])
            return int(arr.shape[1]), int(arr.shape[2])
    if depth.ndim != 3:
        raise ValueError(f"Expected DA3 depth shape [N,H,W], got {depth.shape}")
    return int(depth.shape[1]), int(depth.shape[2])


def _run_da3_inference(model, frame_paths: list[str], args: SequenceDepthInferenceArgs, source_type: str) -> dict[str, np.ndarray | Optional[np.ndarray] | Optional[str] | int]:
    ref_view_strategy = args.da3_ref_view_strategy or ("middle" if source_type == "video" else None)

    inference_kwargs = {
        "image": frame_paths,
        "process_res": args.da3_process_res,
        "process_res_method": args.da3_process_res_method,
    }
    if ref_view_strategy is not None:
        inference_kwargs["ref_view_strategy"] = ref_view_strategy

    prediction = model.inference(**inference_kwargs)

    depth = _to_numpy(_prediction_value(prediction, "depth"))
    if depth is None:
        raise RuntimeError("DA3 inference did not return `depth`.")
    if depth.ndim == 2:
        depth = depth[None, ...]
    if depth.ndim != 3:
        raise ValueError(f"Expected DA3 depth shape [N,H,W], got {depth.shape}")

    conf = _to_numpy(_prediction_value(prediction, "conf"))
    if conf is not None and conf.ndim == 2:
        conf = conf[None, ...]

    intrinsics = _to_numpy(_prediction_value(prediction, "intrinsics"))
    if intrinsics is None:
        raise RuntimeError("DA3 inference did not return `intrinsics`.")
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[None, ...]
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected DA3 intrinsics shape [N,3,3], got {intrinsics.shape}")

    extrinsics_raw = _to_numpy(_prediction_value(prediction, "extrinsics"))
    if extrinsics_raw is None:
        raise RuntimeError("DA3 inference did not return `extrinsics`.")
    extrinsics_w2c = ensure_homogeneous_extrinsics(extrinsics_raw)

    processed_images = _to_numpy(_prediction_value(prediction, "processed_images"))
    processed_h, processed_w = _resolve_processed_hw(processed_images, depth)

    num_frames = len(frame_paths)
    if depth.shape[0] != num_frames:
        raise RuntimeError(f"DA3 returned {depth.shape[0]} depth maps for {num_frames} frames.")
    if intrinsics.shape[0] != num_frames:
        raise RuntimeError(f"DA3 returned {intrinsics.shape[0]} intrinsics for {num_frames} frames.")
    if extrinsics_w2c.shape[0] != num_frames:
        raise RuntimeError(f"DA3 returned {extrinsics_w2c.shape[0]} extrinsics for {num_frames} frames.")
    if conf is not None and conf.shape[0] != num_frames:
        raise RuntimeError(f"DA3 returned {conf.shape[0]} confidence maps for {num_frames} frames.")

    return {
        "depth": depth.astype(np.float32),
        "conf": None if conf is None else conf.astype(np.float32),
        "intrinsics": intrinsics.astype(np.float32),
        "extrinsics_w2c": extrinsics_w2c.astype(np.float32),
        "processed_h": processed_h,
        "processed_w": processed_w,
        "ref_view_strategy": ref_view_strategy,
    }


def _load_intrinsics_matrix_txt(path: str) -> np.ndarray:
    raw = np.asarray(np.loadtxt(path, dtype=np.float32), dtype=np.float32)
    if raw.shape == (3, 3):
        intrinsics = raw
    else:
        values = raw.reshape(-1)
        if values.size < 4:
            raise ValueError(
                f"Expected at least 4 values in intrinsics file `{path}`, got {values.size}."
            )
        intrinsics = np.array(
            [
                [float(values[0]), 0.0, float(values[2])],
                [0.0, float(values[1]), float(values[3])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    if intrinsics.shape != (3, 3):
        raise ValueError(f"Intrinsics file `{path}` must resolve to a 3x3 matrix, got {intrinsics.shape}.")
    if not np.all(np.isfinite(intrinsics)):
        raise ValueError(f"Intrinsics file `{path}` contains non-finite values.")
    return intrinsics.astype(np.float32)


def _load_extrinsics_matrix_txt(path: str) -> np.ndarray:
    raw = np.asarray(np.loadtxt(path, dtype=np.float32), dtype=np.float32)
    if raw.shape in {(3, 4), (4, 4)}:
        matrix = raw
    else:
        values = raw.reshape(-1)
        if values.size == 12:
            matrix = values.reshape(3, 4)
        elif values.size == 16:
            matrix = values.reshape(4, 4)
        else:
            raise ValueError(
                f"Extrinsics file `{path}` must contain a 3x4 or 4x4 matrix, got {values.size} values."
            )

    matrix = ensure_homogeneous_extrinsics(matrix)[0]
    if matrix.shape != (4, 4):
        raise ValueError(f"Extrinsics file `{path}` must resolve to a 4x4 matrix, got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Extrinsics file `{path}` contains non-finite values.")
    return matrix.astype(np.float32)


def _load_user_camera_parameters(
    frame_paths: list[str],
    args: SequenceDepthInferenceArgs,
) -> dict[str, list[str] | np.ndarray]:
    intrinsics_paths = _collect_camera_parameter_paths(
        args.camera_intrinsics_dir,
        expected_count=len(frame_paths),
        label="Camera intrinsics",
    )
    extrinsics_paths = _collect_camera_parameter_paths(
        args.camera_extrinsics_dir,
        expected_count=len(frame_paths),
        label="Camera extrinsics",
    )

    intrinsics = np.stack([_load_intrinsics_matrix_txt(path) for path in intrinsics_paths], axis=0)
    extrinsics_w2c = np.stack([_load_extrinsics_matrix_txt(path) for path in extrinsics_paths], axis=0)
    return {
        "intrinsics_original": intrinsics.astype(np.float32),
        "extrinsics_w2c": extrinsics_w2c.astype(np.float32),
        "intrinsics_paths": intrinsics_paths,
        "extrinsics_paths": extrinsics_paths,
    }


def _build_frame_args(args: SequenceDepthInferenceArgs) -> DepthInferenceArgs:
    return DepthInferenceArgs(
        input_image_path=args.input_path,
        input_depth_path=args.input_depth_path if args.model_type == "InfiniDepth_DepthSensor" else None,
        depth_output_dir=None,
        pcd_output_dir=None,
        save_pcd=args.save_frame_pcd,
        model_type=args.model_type,
        depth_model_path=args.depth_model_path,
        moge2_pretrained=args.moge2_pretrained,
        input_size=args.input_size,
        output_size=args.output_size,
        output_resolution_mode=args.output_resolution_mode,
        upsample_ratio=args.upsample_ratio,
        enable_skyseg_model=args.enable_skyseg_model,
        sky_model_ckpt_path=args.sky_model_ckpt_path,
    )


def _resolve_da3_intrinsics_original(
    da3_outputs: dict[str, np.ndarray | Optional[np.ndarray] | Optional[str] | int],
    frame_index: int,
    org_h: int,
    org_w: int,
) -> np.ndarray:
    return scale_intrinsics_matrix_np(
        da3_outputs["intrinsics"][frame_index],
        src_h=int(da3_outputs["processed_h"]),
        src_w=int(da3_outputs["processed_w"]),
        dst_h=org_h,
        dst_w=org_w,
    )


def _build_output_intrinsics_from_original(
    intrinsics_org: np.ndarray,
    org_h: int,
    org_w: int,
    output_h: int,
    output_w: int,
) -> np.ndarray:
    return scale_intrinsics_matrix_np(
        intrinsics_org,
        src_h=org_h,
        src_w=org_w,
        dst_h=output_h,
        dst_w=output_w,
    )


def _build_da3_ransac_override(
    frame_da3_depth: np.ndarray,
    frame_conf: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(frame_da3_depth, dtype=np.float32)
    mask = np.isfinite(depth) & (depth > 1e-6)
    if frame_conf is not None:
        conf = np.asarray(frame_conf, dtype=np.float32)
        mask &= np.isfinite(conf) & (conf > 0.0)
    return depth, mask.astype(np.float32)


def _should_use_da3_ransac_reference(model_type: str) -> bool:
    return model_type == "InfiniDepth"


def _should_apply_da3_post_scale_align(args: SequenceDepthInferenceArgs) -> bool:
    return args.model_type == "InfiniDepth_DepthSensor" and args.align_to_da3_depth


def _write_json(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def _use_user_camera_parameters(args: SequenceDepthInferenceArgs) -> bool:
    return args.camera_intrinsics_dir is not None and args.camera_extrinsics_dir is not None


def _validate_args(args: SequenceDepthInferenceArgs, input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for sequence inference in this script.")
    if args.model_type == "InfiniDepth_DepthSensor" and args.input_depth_path is None:
        raise ValueError(
            "`InfiniDepth_DepthSensor` requires `--input_depth_path` for sequence inference."
        )

    has_camera_intrinsics = args.camera_intrinsics_dir is not None
    has_camera_extrinsics = args.camera_extrinsics_dir is not None
    if has_camera_intrinsics != has_camera_extrinsics:
        raise ValueError(
            "Provide both `--camera_intrinsics_dir` and `--camera_extrinsics_dir`, or leave both unset."
        )


@torch.no_grad()
def main(args: SequenceDepthInferenceArgs) -> None:
    input_path = Path(args.input_path).expanduser().resolve()
    _validate_args(args, input_path)

    input_mode = _detect_input_mode(input_path, args.input_mode)
    use_user_camera_parameters = _use_user_camera_parameters(args)
    if use_user_camera_parameters and input_mode == "video":
        raise ValueError("Explicit camera parameter mode only supports image inputs; video input is not supported.")

    output_paths = resolve_sequence_output_paths(str(input_path), args.output_root)

    frame_paths, source_fps, source_type = _prepare_rgb_inputs(input_path, input_mode, output_paths)
    use_input_depth = args.model_type == "InfiniDepth_DepthSensor"
    depth_frame_paths, depth_source_fps, depth_source_type = _prepare_depth_inputs(
        args.input_depth_path if use_input_depth else None,
        input_mode,
        output_paths,
        expected_count=len(frame_paths),
    )

    camera_source = "user" if use_user_camera_parameters else "da3"
    user_camera_outputs = None
    da3_outputs = None
    if use_user_camera_parameters:
        user_camera_outputs = _load_user_camera_parameters(frame_paths, args)
        print(
            "Using explicit camera parameters from "
            f"{Path(args.camera_intrinsics_dir).expanduser().resolve()} and "
            f"{Path(args.camera_extrinsics_dir).expanduser().resolve()}. Skip DA3."
        )
    else:
        DepthAnything3 = _import_depth_anything3()
        da3_model = DepthAnything3.from_pretrained(args.da3_model).to("cuda")
        if hasattr(da3_model, "eval"):
            da3_model.eval()
        print(f"Loaded DA3 model: {args.da3_model}")

        da3_outputs = _run_da3_inference(da3_model, frame_paths, args, source_type)
        cache_payload = {
            "depth": da3_outputs["depth"],
            "intrinsics": da3_outputs["intrinsics"],
            "extrinsics_w2c": da3_outputs["extrinsics_w2c"],
        }
        if da3_outputs["conf"] is not None:
            cache_payload["conf"] = da3_outputs["conf"]
        np.savez_compressed(output_paths.da3_cache_path, **cache_payload)
        print(f"Saved DA3 cache -> {output_paths.da3_cache_path}")

    frame_args = _build_frame_args(args)
    depth_model, device = load_depth_model(frame_args)

    merged_frame_pcds = []
    frame_meta_paths: list[str] = []
    da3_ransac_reference_enabled = (not use_user_camera_parameters) and _should_use_da3_ransac_reference(args.model_type)
    da3_post_scale_align_enabled = (not use_user_camera_parameters) and _should_apply_da3_post_scale_align(args)

    for frame_index, frame_path in enumerate(frame_paths):
        print(f"Processing frame {frame_index + 1}/{len(frame_paths)}: {frame_path}")
        org_h, org_w = _read_image_size(frame_path)
        frame_depth_path = None if depth_frame_paths is None else depth_frame_paths[frame_index]
        frame_conf = None
        override_gt_depth = None
        override_gt_depth_mask = None

        if use_user_camera_parameters:
            frame_intrinsics_org = user_camera_outputs["intrinsics_original"][frame_index]
            frame_w2c = user_camera_outputs["extrinsics_w2c"][frame_index]
        else:
            frame_intrinsics_org = _resolve_da3_intrinsics_original(
                da3_outputs,
                frame_index,
                org_h,
                org_w,
            )
            frame_w2c = da3_outputs["extrinsics_w2c"][frame_index]
            frame_conf = None if da3_outputs["conf"] is None else da3_outputs["conf"][frame_index]

        if da3_ransac_reference_enabled:
            override_gt_depth, override_gt_depth_mask = _build_da3_ransac_override(
                da3_outputs["depth"][frame_index],
                frame_conf,
            )

        result = run_depth_inference(
            frame_args,
            model=depth_model,
            device=device,
            input_image_path=frame_path,
            input_depth_path=frame_depth_path,
            fx_org=float(frame_intrinsics_org[0, 0]),
            fy_org=float(frame_intrinsics_org[1, 1]),
            cx_org=float(frame_intrinsics_org[0, 2]),
            cy_org=float(frame_intrinsics_org[1, 2]),
            override_gt_depth=override_gt_depth,
            override_gt_depth_mask=override_gt_depth_mask,
        )
        frame_intrinsics_output = _build_output_intrinsics_from_original(
            frame_intrinsics_org,
            org_h=org_h,
            org_w=org_w,
            output_h=result.output_h,
            output_w=result.output_w,
        )

        if da3_post_scale_align_enabled:
            scale_align_depth_result(
                result,
                da3_outputs["depth"][frame_index],
                reference_conf=frame_conf,
                confidence_threshold=args.da3_scale_align_conf_threshold,
                min_valid_pixels=args.da3_scale_align_min_valid_pixels,
            )
            print(
                "Applied DA3 depth scale alignment: "
                f"scale={result.depth_scale_align_factor:.6f}, valid_pixels={result.depth_scale_align_valid_pixels}"
            )

        frame_name = f"frame_{frame_index:06d}"
        depth_raw_path = str(Path(output_paths.frame_depth_dir) / f"{frame_name}.npy")
        depth_vis_path = str(Path(output_paths.frame_depth_vis_dir) / f"{frame_name}.png")
        pcd_path = str(Path(output_paths.frame_pcd_dir) / f"{frame_name}.ply")
        meta_path = str(Path(output_paths.frame_meta_dir) / f"{frame_name}.json")

        frame_pcd = save_depth_inference_result(
            result,
            depth_vis_path=depth_vis_path,
            depth_raw_path=depth_raw_path,
            pcd_path=pcd_path if args.save_frame_pcd else None,
            save_pcd=args.save_frame_pcd,
            pcd_extrinsics_w2c=frame_w2c,
            pcd_intrinsics_override=frame_intrinsics_output,
        )

        if args.save_merged_pcd:
            if frame_pcd is None:
                frame_pcd = build_point_cloud_from_depth_result(
                    result,
                    pcd_extrinsics_w2c=frame_w2c,
                    pcd_intrinsics_override=frame_intrinsics_output,
                )
            merged_frame_pcds.append(frame_pcd)

        frame_c2w = np.linalg.inv(frame_w2c)
        frame_meta = {
            "frame_id": frame_index,
            "frame_name": frame_name,
            "source_path": frame_path,
            "source_type": source_type,
            "input_depth_path": frame_depth_path,
            "depth_source_type": None if frame_depth_path is None else depth_source_type,
            "input_resolution": [result.org_h, result.org_w],
            "inference_resolution": [result.input_h, result.input_w],
            "output_resolution": [result.output_h, result.output_w],
            "camera_source": camera_source,
            "camera_intrinsics_original": frame_intrinsics_org.tolist(),
            "camera_intrinsics_output": frame_intrinsics_output.tolist(),
            "pcd_intrinsics_source": camera_source,
            "model_intrinsics_source": result.intrinsics_source,
            "model_intrinsics_output": result.output_intrinsics_matrix().tolist(),
            "extrinsics_w2c": frame_w2c.tolist(),
            "extrinsics_c2w": frame_c2w.tolist(),
            "frame_pcd_path": pcd_path if args.save_frame_pcd else None,
            "depth_raw_path": depth_raw_path,
            "depth_vis_path": depth_vis_path,
        }
        if use_user_camera_parameters:
            frame_meta["camera_intrinsics_path"] = user_camera_outputs["intrinsics_paths"][frame_index]
            frame_meta["camera_extrinsics_path"] = user_camera_outputs["extrinsics_paths"][frame_index]
        else:
            da3_post_scale_align_applied = (
                da3_post_scale_align_enabled
                and result.depth_scale_align_valid_pixels >= int(args.da3_scale_align_min_valid_pixels)
                and np.isfinite(result.depth_scale_align_factor)
                and result.depth_scale_align_factor > 0.0
            )
            frame_meta.update(
                {
                    "da3_process_resolution": [int(da3_outputs["processed_h"]), int(da3_outputs["processed_w"])],
                    "da3_intrinsics_original": frame_intrinsics_org.tolist(),
                    "da3_intrinsics_output": frame_intrinsics_output.tolist(),
                    "da3_conf_mean": None if frame_conf is None else float(np.mean(frame_conf)),
                    "da3_ransac_reference_enabled": da3_ransac_reference_enabled,
                    "da3_post_scale_align_enabled": da3_post_scale_align_enabled,
                    "da3_post_scale_align_applied": da3_post_scale_align_applied,
                    "da3_post_scale_align_factor": result.depth_scale_align_factor if da3_post_scale_align_enabled else 1.0,
                    "da3_post_scale_align_valid_pixels": result.depth_scale_align_valid_pixels if da3_post_scale_align_enabled else 0,
                    "da3_scale_align_factor": result.depth_scale_align_factor if da3_post_scale_align_enabled else 1.0,
                    "da3_scale_align_valid_pixels": result.depth_scale_align_valid_pixels if da3_post_scale_align_enabled else 0,
                }
            )
        _write_json(meta_path, frame_meta)
        frame_meta_paths.append(meta_path)

        del result

    merged_pcd_path = None
    merged_point_count = 0
    if args.save_merged_pcd:
        merged_pcd = merge_point_clouds(merged_frame_pcds)
        merged_point_count = len(merged_pcd.points)
        merged_pcd_path = output_paths.merged_pcd_path
        save_point_cloud(
            merged_pcd,
            merged_pcd_path,
            filter_flying_points=False,
        )
        print(f"Saved merged point cloud -> {merged_pcd_path}")

    manifest = {
        "input_path": str(input_path),
        "input_depth_path": None if not use_input_depth or args.input_depth_path is None else str(Path(args.input_depth_path).expanduser().resolve()),
        "input_mode": input_mode,
        "source_type": source_type,
        "depth_source_type": depth_source_type if use_input_depth else None,
        "num_frames": len(frame_paths),
        "video_fps": source_fps,
        "depth_video_fps": depth_source_fps if use_input_depth else None,
        "output_root": output_paths.root_dir,
        "camera_source": camera_source,
        "model_type": args.model_type,
        "frame_depth_dir": output_paths.frame_depth_dir,
        "frame_depth_vis_dir": output_paths.frame_depth_vis_dir,
        "frame_pcd_dir": output_paths.frame_pcd_dir if args.save_frame_pcd else None,
        "frame_meta_dir": output_paths.frame_meta_dir,
        "merged_pcd_path": merged_pcd_path,
        "merged_point_count": merged_point_count,
        "frame_meta_paths": frame_meta_paths,
    }
    if use_user_camera_parameters:
        manifest.update(
            {
                "camera_intrinsics_dir": str(Path(args.camera_intrinsics_dir).expanduser().resolve()),
                "camera_extrinsics_dir": str(Path(args.camera_extrinsics_dir).expanduser().resolve()),
                "align_to_da3_depth": False,
                "da3_model": None,
                "da3_process_res": None,
                "da3_process_res_method": None,
                "da3_ref_view_strategy": None,
                "da3_depth_ransac_conditioning_enabled": False,
                "da3_post_scale_align_enabled": False,
                "da3_scale_align_conf_threshold": None,
                "da3_scale_align_min_valid_pixels": None,
                "da3_cache_path": None,
            }
        )
    else:
        manifest.update(
            {
                "camera_intrinsics_dir": None,
                "camera_extrinsics_dir": None,
                "da3_model": args.da3_model,
                "da3_process_res": args.da3_process_res,
                "da3_process_res_method": args.da3_process_res_method,
                "da3_ref_view_strategy": da3_outputs["ref_view_strategy"],
                "align_to_da3_depth": args.align_to_da3_depth,
                "da3_depth_ransac_conditioning_enabled": da3_ransac_reference_enabled,
                "da3_post_scale_align_enabled": da3_post_scale_align_enabled,
                "da3_scale_align_conf_threshold": args.da3_scale_align_conf_threshold,
                "da3_scale_align_min_valid_pixels": args.da3_scale_align_min_valid_pixels,
                "da3_cache_path": output_paths.da3_cache_path,
            }
        )
    _write_json(output_paths.manifest_path, manifest)
    print(f"Saved manifest -> {output_paths.manifest_path}")


if __name__ == "__main__":
    main(tyro.cli(SequenceDepthInferenceArgs))
