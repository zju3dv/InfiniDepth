python inference_gs.py \
  --input_image_path=example_data/image/eth3d_office.png \
  --input_depth_path=example_data/depth/eth3d_office.npz \
  --model_type=InfiniDepth_DC \
  --depth_model_path=checkpoints/depth/infinidepth_dc.ckpt \
  --gs_model_path=checkpoints/gs/infinidepth_dc_gs.ckpt \
  --fx_org=866.39 \
  --fy_org=866.04 \
  --cx_org=791.5 \
  --cy_org=523.81 \
  
