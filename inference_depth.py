from dataclasses import dataclass
from typing import Literal, Optional
import os
import torch
import tyro

from InfiniDepth.utils.inference_utils import (
    OUTPUT_RESOLUTION_MODES,
    apply_sky_mask_to_depth,
    build_scaled_intrinsics_matrix,
    prepare_metric_depth_inputs,
    resolve_depth_output_paths,
    resolve_camera_intrinsics_for_inference,
    resolve_output_size_from_mode,
    run_optional_sky_mask,
)
from InfiniDepth.utils.io_utils import load_image, depth_to_disparity, plot_depth, save_sampled_point_clouds
from InfiniDepth.utils.model_utils import build_model
from InfiniDepth.utils.sampling_utils import SAMPLING_METHODS


@dataclass
class DepthInferenceArgs:
    # Inputs
    input_image_path: str
    input_depth_path: Optional[str] = None

    # Outputs
    depth_output_dir: Optional[str] = None
    pcd_output_dir: Optional[str] = None
    save_pcd: bool = True

    # Model
    model_type: str = "InfiniDepth_DepthSensor"  # [InfiniDepth, InfiniDepth_DepthSensor]
    depth_model_path: str = "checkpoints/depth/infinidepth_depthsensor.ckpt"
    moge2_pretrained: str = "checkpoints/moge-2-vitl-normal/model.pt"  # Metric depth via MoGe-2 (used when input_depth_path is None)

    # Camera intrinsics
    fx_org: Optional[float] = None
    fy_org: Optional[float] = None
    cx_org: Optional[float] = None
    cy_org: Optional[float] = None

    # Data Resolution
    input_size: tuple[int, int] = (768, 1024)
    output_size: tuple[int, int] = (768, 1024)
    output_resolution_mode: Literal["upsample", "original", "specific"] = "upsample"
    upsample_ratio: int = 1

    # Optional sky segmentation
    enable_skyseg_model: bool = False
    sky_model_ckpt_path: str = "checkpoints/sky/skyseg.onnx"


@torch.no_grad()
def main(args: DepthInferenceArgs) -> None:
    if args.output_resolution_mode not in OUTPUT_RESOLUTION_MODES:
        raise ValueError(
            f"Unsupported output_resolution_mode: {args.output_resolution_mode}. "
            f"Choose from {OUTPUT_RESOLUTION_MODES}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference in this script.")

    model = build_model(
        args.model_type,
        model_path=args.depth_model_path,
    )
    print(f"Loaded model: {model.__class__.__name__}")

    device = torch.device("cuda")

    org_img, image, (org_h, org_w) = load_image(args.input_image_path, args.input_size)
    image = image.to(device)

    if args.model_type == "InfiniDepth_DepthSensor":
        assert args.input_depth_path is not None and os.path.exists(args.input_depth_path), "InfiniDepth_DepthSensor requires a valid input depth map for depth completion. Please provide --input_depth_path."

    gt_depth, prompt_depth, gt_depth_mask, use_gt_depth, moge2_intrinsics = prepare_metric_depth_inputs(
        input_depth_path=args.input_depth_path,
        input_size=args.input_size,
        image=image,
        device=device,
        moge2_pretrained=args.moge2_pretrained,
    )
    if use_gt_depth and args.input_depth_path is not None:
        print(f"metric depth from `{args.input_depth_path}`")
    else:
        print(f"metric depth from `{args.moge2_pretrained}`")

    fx_org, fy_org, cx_org, cy_org, intrinsics_source = resolve_camera_intrinsics_for_inference(
        fx_org=args.fx_org,
        fy_org=args.fy_org,
        cx_org=args.cx_org,
        cy_org=args.cy_org,
        org_h=org_h,
        org_w=org_w,
        image=image,
        moge2_pretrained=args.moge2_pretrained,
        moge2_intrinsics=moge2_intrinsics,
    )
    if intrinsics_source == "moge2":
        print(
            "Camera intrinsics are partially/fully missing. "
            f"Using MoGe-2 estimated intrinsics in original space: fx={fx_org:.2f}, fy={fy_org:.2f}, cx={cx_org:.2f}, cy={cy_org:.2f}"
        )
    elif intrinsics_source == "default":
        print(
            "Camera intrinsics are partially/fully missing. "
            f"Using image-size defaults in original space: fx={fx_org:.2f}, fy={fy_org:.2f}, cx={cx_org:.2f}, cy={cy_org:.2f}"
        )

    gt = depth_to_disparity(gt_depth)
    prompt = depth_to_disparity(prompt_depth)

    _, _, h, w = image.shape
    fx, fy, cx, cy, k = build_scaled_intrinsics_matrix(
        fx_org=fx_org,
        fy_org=fy_org,
        cx_org=cx_org,
        cy_org=cy_org,
        org_h=org_h,
        org_w=org_w,
        h=h,
        w=w,
        device=image.device,
    )
    print(f"Scaled Intrinsics: fx {fx:.2f}, fy {fy:.2f}, cx {cx:.2f}, cy {cy:.2f}")

    sky_mask = run_optional_sky_mask(
        image=image,
        enable_skyseg_model=args.enable_skyseg_model,
        sky_model_ckpt_path=args.sky_model_ckpt_path,
    )

    h_sample, w_sample = resolve_output_size_from_mode(
        output_resolution_mode=args.output_resolution_mode,
        org_h=org_h,
        org_w=org_w,
        h=h,
        w=w,
        output_size=args.output_size,
        upsample_ratio=args.upsample_ratio,
    )

    query_2d_uniform_coord = SAMPLING_METHODS["2d_uniform"]((h_sample, w_sample)).unsqueeze(0).to(device)
    pred_2d_uniform_depth, _ = model.inference(
        image=image,
        query_coord=query_2d_uniform_coord,
        gt_depth=gt,
        gt_depth_mask=gt_depth_mask,
        prompt_depth=prompt,
        prompt_mask=prompt>0,
    )
    pred_depthmap = pred_2d_uniform_depth.permute(0,2,1).view(1,1,h_sample,w_sample) 

    pred_depthmap, pred_2d_uniform_depth = apply_sky_mask_to_depth(
        pred_depthmap=pred_depthmap,
        pred_2d_uniform_depth=pred_2d_uniform_depth,
        sky_mask=sky_mask,
        h_sample=h_sample,
        w_sample=w_sample,
        sky_depth_value=200.0,
    )

    output_paths = resolve_depth_output_paths(
        input_image_path=args.input_image_path,
        model_type=args.model_type,
        output_resolution_mode=args.output_resolution_mode,
        upsample_ratio=args.upsample_ratio,
        h_sample=h_sample,
        w_sample=w_sample,
        depth_output_dir=args.depth_output_dir,
        pcd_output_dir=args.pcd_output_dir,
    )

    plot_depth(org_img, pred_depthmap, output_paths.depth_path)
    if args.save_pcd:
        save_sampled_point_clouds(
            query_2d_uniform_coord.squeeze().cpu(),
            pred_2d_uniform_depth.squeeze().cpu(),
            image.squeeze().cpu(),
            fx,
            fy,
            cx,
            cy,
            output_paths.pcd_path,
        )


if __name__ == "__main__":
    main(tyro.cli(DepthInferenceArgs))
