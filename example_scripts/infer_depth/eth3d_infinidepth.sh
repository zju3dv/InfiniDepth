python inference_depth.py \
  --input_image_path=example_data/image/eth3d_office.png \
  --model_type=InfiniDepth \
  --depth_model_path=checkpoints/depth/infinidepth.ckpt \
  --fx_org=866.39 \
  --fy_org=866.04 \
  --cx_org=791.5 \
  --cy_org=523.81 \
  --output_resolution_mode=upsample \
  --upsample_ratio=2 \
