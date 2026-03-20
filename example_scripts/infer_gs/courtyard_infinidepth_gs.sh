python inference_gs.py \
  --input_image_path=example_data/image/courtyard.jpg \
  --model_type=InfiniDepth \
  --depth_model_path=checkpoints/depth/infinidepth.ckpt \
  --gs_model_path=checkpoints/gs/infinidepth_gs.ckpt 
