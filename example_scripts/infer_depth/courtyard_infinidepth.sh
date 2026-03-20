python inference_depth.py \
  --input_image_path=example_data/image/courtyard.jpg \
  --model_type=InfiniDepth \
  --depth_model_path=checkpoints/depth/infinidepth.ckpt \
  --output_resolution_mode=upsample \
  --upsample_ratio=2 \
