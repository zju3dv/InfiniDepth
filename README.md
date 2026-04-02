<div align="center">

<h1>
  <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Telescope.png" alt="Telescope" width="40" height="40" />
  InfiniDepth: Arbitrary-Resolution and Fine-Grained Depth Estimation with Neural Implicit Fields
</h1>

<div align="center">
  <a href="https://arxiv.org/abs/2601.03252">
    <img src="https://img.shields.io/badge/arXiv-Paper-blue?logo=arxiv&logoColor=blue">
  </a>
  <a href="https://huggingface.co/spaces/ritianyu/InfiniDepth">
    <img src="https://img.shields.io/badge/HuggingFace-Demo-yellow?logo=huggingface&logoColor=yellow">
  </a>
  <a href="https://huggingface.co/datasets/ritianyu/game_4k_data">
    <img src="https://img.shields.io/badge/HuggingFace-Dataset-orange?logo=huggingface&logoColor=orange">
  </a>
  <a href="assets/wechat.jpg">
    <img src="https://img.shields.io/badge/微信-WeChat-green?logo=wechat&logoColor=green">
  </a>
</div>
  
<p align="center">
  <a href="https://ritianyu.github.io/">Hao Yu*</a> •
  <a href="https://haotongl.github.io/">Haotong Lin*</a> •
  <a href="https://github.com/PLUS-WAVE">Jiawei Wang*</a> •
  <a href="https://github.com/Dustbin-Li">Jiaxin Li</a> •
  <a href="https://wangyida.github.io/">Yida Wang</a> •
  <a href="#">Xueyang Zhang</a> •
  <a href="https://ywang-zju.github.io/">Yue Wang</a> <br>
  <a href="https://www.xzhou.me/">Xiaowei Zhou</a> •
  <a href="https://csse.szu.edu.cn/staff/ruizhenhu/">Ruizhen Hu</a> •
  <a href="https://pengsida.net/">Sida Peng</a>
</p>

</div>

<div align="center">

<img src="assets/demo.gif" alt="InfiniDepth Demo" width="90%" />

</div>

## 📢 News
> **[2026-03]** 🎉 Inference code of InfiniDepth (RGB Only & Depth Sensor Augmentation) is available now!

> **[2026-02]** 🎉 InfiniDepth has been accepted to CVPR 2026! Code coming soon!


## ✨ What can InfiniDepth do?
InfiniDepth supports three practical capabilities for single-image 3D perception and reconstruction:

| Capability | Input | Output |
| --- | --- | --- |
| Monocular & Arbitrary-Resolution Depth Estimation | RGB Image | Arbitrary-Resolution Depth Map |
| Monocular View Synthesis | RGB Image | 3D Gaussian Splatting (3DGS) |
| Depth Sensor Augmentation (Monocular Metric Depth Estimation) | RGB Image + Depth Sensor | Metric Depth + 3D Gaussian Splatting (3DGS) |

## ⚙️ Installation

Please see [INSTALL.md](INSTALL.md) for manual installation.
 
## 🤗 Hugging Face Space Demo

If you want to test InfiniDepth before running local CLI inference, start with the hosted demo:

- Hugging Face Space: https://huggingface.co/spaces/ritianyu/InfiniDepth

This repo also includes a Gradio Space entrypoint at `app.py`:

- Input: RGB image (required), depth map (optional)
- Task Switch: `Depth` / `3DGS`
- Model Switch: `InfiniDepth` / `InfiniDepth_DepthSensor`

### Local run

```bash
python app.py
```

### Notes

- In this demo, `InfiniDepth_DepthSensor` requires a depth map input; RGB-only inference should use `InfiniDepth`.
- Supported depth formats in the demo upload: `.png`, `.npy`, `.npz`, `.h5`, `.hdf5`, `.exr`.


## 🚀 Inference

### Quick Command Index

| If you want ... | Recommended command |
| --- | --- |
| Relative Depth from Single RGB Image | `bash example_scripts/infer_depth/courtyard_infinidepth.sh` |
| 3D Gaussian from Single RGB Image | `bash example_scripts/infer_gs/courtyard_infinidepth_gs.sh` |
| Metric Depth from RGB + Depth Sensor | `bash example_scripts/infer_depth/eth3d_infinidepth_depthsensor.sh` |
| 3D Gaussian from RGB + Depth Sensor | `bash example_scripts/infer_gs/eth3d_infinidepth_depthsensor_gs.sh` |
| Multi-View / Video Depth + Global Point Cloud | `bash example_scripts/infer_depth/waymo_multi_view_infinidepth.sh` | 

<details>
<summary><strong> 1. Relative Depth from Single RGB Image</strong> (<code>inference_depth.py</code>)</summary>

Use this when you want a relative depth map from a single RGB image and, optionally, a point cloud export.

**Required input**

- `RGB image`

**Required checkpoints**

- `checkpoints/depth/infinidepth.ckpt`
- `checkpoints/moge-2-vitl-normal/model.pt` recover metric scale for point cloud export

**Optional checkpoint**

- `checkpoints/sky/skyseg.onnx` additional sky filtering

**Recommended command**

```bash
python inference_depth.py \
  --input_image_path=example_data/image/courtyard.jpg \
  --model_type=InfiniDepth \
  --depth_model_path=checkpoints/depth/infinidepth.ckpt \
  --output_resolution_mode=upsample \
  --upsample_ratio=2
```

Replace `example_data/image/courtyard.jpg` with your own image path.

**For the example above, outputs are written to**

- `example_data/pred_depth/` for the colorized depth map
- `example_data/pred_pcd/` for the exported point cloud when `--save_pcd=True`

**Example scripts**

```bash
bash example_scripts/infer_depth/courtyard_infinidepth.sh
bash example_scripts/infer_depth/camera_infinidepth.sh
bash example_scripts/infer_depth/eth3d_infinidepth.sh
bash example_scripts/infer_depth/waymo_infinidepth.sh
```

**Most useful options**

| Argument | What it controls |
| --- | --- |
| `--output_resolution_mode` | Choose `upsample`, `original`, or `specific`. |
| `--upsample_ratio` | Used when `output_resolution_mode=upsample`. |
| `--output_size` | Explicit output size `(H,W)` when `output_resolution_mode=specific`. |
| `--save_pcd` | Export a point cloud alongside the depth map. |
| `--fx_org --fy_org --cx_org --cy_org` | Camera intrinsics in the original image resolution. |

</details>

<details>
<summary><strong>2. 3D Gaussian + Novel-View Video from Single RGB Image</strong> (<code>inference_gs.py</code>)</summary>

Use this when you want a 3D Gaussian export from a single RGB image and an optional novel-view video.

**Required input**

- `RGB image`

**Required checkpoints**

- `checkpoints/depth/infinidepth.ckpt`
- `checkpoints/gs/infinidepth_gs.ckpt`
- `checkpoints/moge-2-vitl-normal/model.pt` recover metric scale for 3D Gaussian export

**Optional checkpoint**

- `checkpoints/sky/skyseg.onnx` additional sky filtering

**Recommended command**

```bash
python inference_gs.py \
  --input_image_path=example_data/image/courtyard.jpg \
  --model_type=InfiniDepth \
  --depth_model_path=checkpoints/depth/infinidepth.ckpt \
  --gs_model_path=checkpoints/gs/infinidepth_gs.ckpt
```

Replace `example_data/image/courtyard.jpg` with your own image path.

**For the example above, outputs are written to**

- `example_data/pred_gs/InfiniDepth_courtyard_gaussians.ply`
- `example_data/pred_gs/InfiniDepth_courtyard_novel_orbit.mp4`

If `--render_size` is omitted, the novel-view video is rendered at the original input image resolution.

**Example scripts**

```bash
bash example_scripts/infer_gs/courtyard_infinidepth_gs.sh
bash example_scripts/infer_gs/camera_infinidepth_gs.sh
bash example_scripts/infer_gs/fruit_infinidepth_gs.sh
bash example_scripts/infer_gs/eth3d_infinidepth_gs.sh
```

**Most useful options**

| Argument | What it controls |
| --- | --- |
| `--render_novel_video` | Turn novel-view rendering on or off. |
| `--render_size` | Output video resolution `(H,W)`. |
| `--novel_trajectory` | Camera motion type: `orbit` or `swing`. |
| `--sample_point_num` | Number of sampled points used for gaussian construction. |
| `--enable_skyseg_model` | Enable sky masking before gaussian sampling. |
| `--sample_sky_mask_dilate_px` | Dilate the sky mask before filtering. |

> The exported `.ply` files can be visualized in 3D viewers such as [SuperSplat](https://superspl.at/).

</details>

<details>
<summary><strong>3. Depth Sensor Augmentation (Metric Depth and 3D Gaussian from RGB + Depth Sensor)</strong></summary>

Use this mode when you have an RGB image plus metric depth from a depth sensor.

**Required inputs**

- `RGB image`
- `Sparse depth` in `.png`, `.npy`, `.npz`, `.h5`, `.hdf5`, or `.exr`

**Required checkpoints**

- `checkpoints/depth/infinidepth_depthsensor.ckpt`
- `checkpoints/moge-2-vitl-normal/model.pt`
- `checkpoints/gs/infinidepth_depthsensor_gs.ckpt`

**Required flags**

- `--model_type=InfiniDepth_DepthSensor`
- `--input_depth_path=...`


**Metric Depth Inference Command**

```bash
python inference_depth.py \
  --input_image_path=example_data/image/eth3d_office.png \
  --input_depth_path=example_data/depth/eth3d_office.npz \
  --model_type=InfiniDepth_DepthSensor \
  --depth_model_path=checkpoints/depth/infinidepth_depthsensor.ckpt \
  --fx_org=866.39 \
  --fy_org=866.04 \
  --cx_org=791.5 \
  --cy_org=523.81 \
  --output_resolution_mode=upsample \
  --upsample_ratio=1
```

**3D Gaussian Inference Command**

```bash
python inference_gs.py \
  --input_image_path=example_data/image/eth3d_office.png \
  --input_depth_path=example_data/depth/eth3d_office.npz \
  --model_type=InfiniDepth_DepthSensor \
  --depth_model_path=checkpoints/depth/infinidepth_depthsensor.ckpt \
  --gs_model_path=checkpoints/gs/infinidepth_depthsensor_gs.ckpt \
  --fx_org=866.39 \
  --fy_org=866.04 \
  --cx_org=791.5 \
  --cy_org=523.81
```

**Example scripts**

```bash
bash example_scripts/infer_depth/eth3d_infinidepth_depthsensor.sh
bash example_scripts/infer_depth/waymo_infinidepth_depthsensor.sh
bash example_scripts/infer_gs/eth3d_infinidepth_depthsensor_gs.sh
bash example_scripts/infer_gs/waymo_infinidepth_depthsensor_gs.sh
```

**Most useful options**

| Argument | What it controls |
| --- | --- |
| `--fx_org --fy_org --cx_org --cy_org` | Strongly recommended when you know the sensor intrinsics. |
| `--output_resolution_mode` | Output behavior for `inference_depth.py`. |
| `--render_size` | Video resolution for `inference_gs.py`. |
| `--output_ply_dir` | Custom output directory for gaussian export. |

</details>

<details>
<summary><strong>4. Multi-View / Video Depth + Global Point Cloud</strong> (<code>inference_multi_view_depth.py</code>)</summary>

Use this when you want sequence-level depth inference from an RGB image folder or video, plus per-frame aligned point clouds and one merged global point cloud. By default the script runs DA3 once on the whole sequence, then aligns each InfiniDepth depth map to the corresponding DA3 depth map before export. When you already know the camera intrinsics and extrinsics, you can instead provide them directly and skip DA3 entirely.

**Required inputs**

- `RGB image directory`, `single RGB image`, or `video`
- `Sparse depth` directory / single file / depth video when `--model_type=InfiniDepth_DepthSensor`

**Required checkpoints / dependencies**

- `checkpoints/depth/infinidepth.ckpt` for RGB-only inference
- `checkpoints/depth/infinidepth_depthsensor.ckpt` for RGB + depth sensor inference
- `checkpoints/moge-2-vitl-normal/model.pt` recover metric scale for RGB-only frame inference
- `depth-anything-3` installed in the current environment when using the default DA3-based sequence mode; default DA3 model is `depth-anything/DA3-LARGE-1.1`

**Optional checkpoint**

- `checkpoints/sky/skyseg.onnx` additional sky filtering

**RGB-Only Multi-View / Video Command**

```bash
python inference_multi_view_depth.py \
  --input_path=example_data/multi-view/waymo/image \
  --model_type=InfiniDepth \
  --depth_model_path=checkpoints/depth/infinidepth.ckpt \
```

**RGB + Depth Sensor Multi-View / Video Command**

```bash
python inference_multi_view_depth.py \
  --input_path=example_data/multi-view/waymo/image \
  --input_depth_path=example_data/multi-view/waymo/depth \
  --model_type=InfiniDepth_DepthSensor \
  --depth_model_path=checkpoints/depth/infinidepth_depthsensor.ckpt \
```

For video input, replace `--input_path` with a video file. When `--model_type=InfiniDepth_DepthSensor`, `--input_depth_path` can also be a depth video and must contain the same number of frames as the RGB input.

**Explicit Camera-Parameter Multi-View Command**

```bash
python inference_multi_view_depth.py \
  --input_path=example_data/multi-view/waymo/image \
  --camera_intrinsics_dir=/path/to/intrinsics \
  --camera_extrinsics_dir=/path/to/extrinsics \
  --model_type=InfiniDepth \
  --depth_model_path=checkpoints/depth/infinidepth.ckpt \
```

The explicit camera mode expects Waymo-style text files under `intrinsics/` and `extrinsics/`. Files are sorted lexicographically and matched one-to-one against the sorted RGB image list, so the number of camera files must exactly match the number of images. In this mode the script skips DA3 loading, DA3 cache export, DA3 RANSAC conditioning, and DA3 post scale alignment. This mode currently supports image inputs only, not video.

**For the RGB-only example above, outputs are written to**

- `example_data/multi-view/waymo/pred_sequence/image/frames/depth/` for aligned raw depth maps
- `example_data/multi-view/waymo/pred_sequence/image/frames/depth_vis/` for colorized depth maps
- `example_data/multi-view/waymo/pred_sequence/image/frames/pcd/` for per-frame aligned point clouds
- `example_data/multi-view/waymo/pred_sequence/image/frames/meta/` for per-frame camera and alignment metadata
- `example_data/multi-view/waymo/pred_sequence/image/da3/sequence_pose.npz` for cached DA3 predictions
- `example_data/multi-view/waymo/pred_sequence/image/merged/sequence_merged.ply` for the merged global point cloud

**Example scripts**

```bash
bash example_scripts/infer_depth/waymo_multi_view_infinidepth.sh
bash example_scripts/infer_depth/waymo_multi_view_infinidepth_depthsensor.sh
bash example_scripts/infer_depth/waymo_multi_view_infinidepth_explicit_camera.sh
```

**Most useful options**

| Argument | What it controls |
| --- | --- |
| `--input_path` | RGB image directory, single image, or video path. |
| `--input_depth_path` | Depth directory, single depth file, or depth video; required for `InfiniDepth_DepthSensor`. |
| `--camera_intrinsics_dir --camera_extrinsics_dir` | Enable explicit camera mode from sorted Waymo-style txt directories. Image inputs only; file counts must match the RGB frame count. |
| `--input_mode` | Force `images` or `video` instead of auto detection. |
| `--align_to_da3_depth` | Align each InfiniDepth depth map to the corresponding DA3 depth map before export. Ignored in explicit camera mode. |
| `--save_frame_pcd` | Save one aligned point cloud per frame. |
| `--save_merged_pcd` | Save the merged global point cloud across the whole sequence. |
| `--da3_scale_align_conf_threshold` | Minimum DA3 confidence used during per-frame scale estimation. |
| `--output_root` | Override the default `pred_sequence/<sequence_name>/` output directory. |

</details>

<details>
<summary><strong>5. Common Argument Conventions</strong></summary>

| Argument | Used in | Description |
| --- | --- | --- |
| `--input_image_path` | depth + gs | Path to the input RGB image. |
| `--input_path` | multi-view | Path to an RGB image directory, single image, or video. |
| `--input_depth_path` | depth + gs + multi-view | Optional metric depth prompt; required for `InfiniDepth_DepthSensor`. In multi-view mode this can be a depth directory, single depth file, or depth video. |
| `--camera_intrinsics_dir --camera_extrinsics_dir` | multi-view | Optional sequence camera parameter directories. When both are set, multi-view inference skips DA3 and uses the provided sorted txt files directly. |
| `--model_type` | depth + gs + multi-view | `InfiniDepth` for RGB-only, `InfiniDepth_DepthSensor` for RGB + sparse depth. |
| `--depth_model_path` | depth + gs | Path to the depth checkpoint. |
| `--gs_model_path` | gs only | Path to the gaussian predictor checkpoint. |
| `--moge2_pretrained` | depth + gs | MoGe-2 checkpoint used when `--input_depth_path` is missing. |
| `--fx_org --fy_org --cx_org --cy_org` | depth + gs | Camera intrinsics in original image resolution. Missing values fall back to MoGe-2 estimates or image-size defaults. |
| `--input_size` | depth + gs | Network input size `(H,W)` used during inference. |
| `--enable_skyseg_model` | depth + gs + multi-view | Enable sky masking before depth or gaussian sampling. |
| `--sky_model_ckpt_path` | depth + gs | Path to the sky segmentation ONNX checkpoint. |

**Depth output modes**

- `--output_resolution_mode=upsample`: output size = `input_size * upsample_ratio`
- `--output_resolution_mode=original`: output size = original input image size
- `--output_resolution_mode=specific`: output size = `output_size`

**Default output directories**

| Script | Default directory |
| --- | --- |
| `inference_depth.py` depth images | `pred_depth/` next to your input data folder |
| `inference_depth.py` point clouds | `pred_pcd/` next to your input data folder |
| `inference_gs.py` gaussians and videos | `pred_gs/` next to your input data folder |
| `inference_multi_view_depth.py` sequence outputs | `pred_sequence/<sequence_name>/` next to your input data folder |

</details>

## 🙏 Acknowledgments

We thank <a href="https://yuanhongyu.xyz/" target="_blank">Yuanhong Yu</a>, <a href="https://gangweix.github.io/" target="_blank">Gangwei Xu</a>, <a href="https://github.com/ghy0324" target="_blank">Haoyu Guo</a>  and <a href="https://hugoycj.github.io/" target="_blank">Chongjie Ye</a> for their insightful discussions and valuable suggestions, and <a href="https://zhenx.me/" target="_blank">Zhen Xu</a> for his dedicated efforts in curating the synthetic data.


## 📖 Citation

If you find InfiniDepth useful in your research, please consider citing:

```bibtex
@article{yu2026infinidepth,
    title={InfiniDepth: Arbitrary-Resolution and Fine-Grained Depth Estimation with Neural Implicit Fields},
    author={Hao Yu, Haotong Lin, Jiawei Wang, Jiaxin Li, Yida Wang, Xueyang Zhang, Yue Wang, Xiaowei Zhou, Ruizhen Hu and Sida Peng},
    booktitle={arXiv preprint},
    year={2026}
}
```
---

<div align="center">

<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Hand%20gestures/Folded%20Hands%20Light%20Skin%20Tone.png" alt="Thanks" width="25" height="25" />

**Thank you for your interest in InfiniDepth!**

<sub>⭐ Star this repo if you find it interesting!</sub>

</div>
