python inference_gs.py \
  --input_image_path=example_data/image/waymo_002.png \
  --input_depth_path=example_data/depth/waymo_002.npy \
  --model_type=InfiniDepth_DepthSensor \
  --depth_model_path=checkpoints/depth/infinidepth_depthsensor.ckpt \
  --gs_model_path=checkpoints/gs/infinidepth_depthsensor_gs.ckpt \
  --fx_org=2083.91 \
  --fy_org=2083.91 \
  --cx_org=957.29 \
  --cy_org=650.57 \
