RGB_DIR=example_data/multi-view/waymo/image
DEPTH_DIR=example_data/multi-view/waymo/depth
INTRINSICS_DIR=example_data/multi-view/waymo/intrinsics
EXTRINSICS_DIR=example_data/multi-view/waymo/extrinsics

python inference_multi_view_depth.py \
    --input_path="${RGB_DIR}" \
    --input_depth_path="${DEPTH_DIR}" \
    --camera_intrinsics_dir="${INTRINSICS_DIR}" \
    --camera_extrinsics_dir="${EXTRINSICS_DIR}" \
    --model_type=InfiniDepth_DepthSensor \
    --depth_model_path=checkpoints/depth/infinidepth_depthsensor.ckpt \
