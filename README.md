<div align="center">

<h1>
  <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Telescope.png" alt="Telescope" width="40" height="40" />
  InfiniDepth: Arbitrary-Resolution and Fine-Grained Depth Estimation with Neural Implicit Fields
</h1>


<p align="center">
  <a href="https://arxiv.org/abs/2601.03252">
    <img src="https://img.shields.io/badge/arXiv-2601.03252-b31b1b.svg?style=for-the-badge" alt="arXiv">
  </a>
  <a href="https://zju3dv.github.io/InfiniDepth">
    <img src="https://img.shields.io/badge/Project-Page-green.svg?style=for-the-badge" alt="Project Page">
  </a>
  <a href="https://huggingface.co/spaces/ritianyu/InfiniDepth">
    <img src="https://img.shields.io/badge/Hugging%20Face-Demo-blue.svg?style=for-the-badge" alt="Hugging Face Demo">
  </a>
  
</p>

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
> **[2026-03]** 🎉 Inference code of InfiniDepth (RGB Only & Sparse Depth Completion) is available now!

> **[2026-02]** 🎉 InfiniDepth has been accepted to CVPR 2026! Code coming soon!


## ✨ Highlights

<div align="center">

<table>
<tr>
<td width="33%" align="center" valign="top">

<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Globe%20Showing%20Americas.png" alt="Arbitrary Resolution Depth Map" width="80" height="80" />

### 🎨 Arbitrary-Resolution
```
4K • 8K • 16K • Beyond
```

</td>
<td width="33%" align="center" valign="top">

<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Microscope.png" alt="Accurate Metric Depth" width="80" height="80" />

### 📐 Accurate Metric Depth


```
Sparse Depth Prompts → Dense Accuracy
```


</td>
<td width="33%" align="center" valign="top">

<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Single-View Novel View Synthesis" width="80" height="80" />

### 📷 Single-View NVS

```
Hight-Quality • Large Viewpoint Shifts
```

</td>
</tr>
</table>

</div>

## 🛠️ Environment & Checkpoints

### 1) Create environment ([miniforge](https://github.com/conda-forge/miniforge) is recommended)

If using `conda`, replace `mamba` with `conda` in the following commands, AND change channel to `conda-forge` for gxx installation:
```bash
mamba create -n infinidepth python=3.10
mamba activate infinidepth

# Optional: When gsplat compilation fails due to g++ version or CUDA toolkit issues.
# mamba install gxx=10
# mamba install nvidia/label/cuda-12.8.0::cuda-toolkit -c nvidia/label/cuda-12.8.0
# export CUDA_HOME=$CONDA_PREFIX
```

### 2) Install dependencies

```bash
# Create environment
pip install uv

# Install PyTorch with CUDA 12.8
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 xformers==0.0.33.post1 --index-url https://download.pytorch.org/whl/cu128

# Install package dependencies
uv pip install -r requirements.txt

# Install MoGe (For scale restoration)
uv pip install git+https://github.com/microsoft/MoGe.git

# Install gsplat (For Gaussian Splatting)
uv pip install git+https://github.com/nerfstudio-project/gsplat.git
```


### 3) Download checkpoints

Download checkpoints and then place them under `checkpoints/depth`, `checkpoints/gs`, `checkpoints/moge-2-vitl-normal`, and `checkpoints/sky`.

#### Model Zoo

| Category | Model | Use Case | Download |
|---|---|---|---|
| Depth | `InfiniDepth` | Relative Depth Inference (RGB) | [infinidepth.ckpt](https://huggingface.co/ritianyu/InfiniDepth/tree/main) |
| Depth | `InfiniDepth_DC` | Metric Depth Inference (RGB + Sparse Depth) | [infinidepth_dc.ckpt](https://huggingface.co/ritianyu/InfiniDepth/tree/main) |
| Depth | `MoGe-2` | For scale restoration | [model.pt](https://huggingface.co/Ruicheng/moge-2-vitl-normal) |
| GS | `InfiniDepth_GS` | Gaussian Inference (RGB) | [infinidepth_gs.ckpt](https://huggingface.co/ritianyu/InfiniDepth/tree/main) |
| GS | `InfiniDepth_DC_GS` | Gaussian inference (RGB + Sparse Depth)| [infinidepth_dc_gs.ckpt](https://huggingface.co/ritianyu/InfiniDepth/tree/main) |
| Sky | `Sky_Mask` | Sky Segmentation | [skyseg.onnx](https://huggingface.co/ritianyu/InfiniDepth/tree/main) |

## 🤗 Hugging Face Space Demo

This repo includes a Gradio Space entrypoint at `app.py` for depth inference:

- input: RGB image (required), depth map (optional)
- outputs: colorized depth map + interactive point cloud + downloadable `.ply`
- model switch: `InfiniDepth` / `InfiniDepth_DC`

### Local run

```bash
python app.py
```

### Notes

- In this demo, `InfiniDepth_DC` requires a depth map input; RGB-only inference should use `InfiniDepth`.
- Supported depth formats in the demo upload: `.png`, `.npy`, `.npz`, `.h5`, `.hdf5`, `.exr`.


## 🧪 Inference Instruction

Run from repository root:

```bash
cd /path/to/InfiniDepth
```

### 0) Quick Start: Run on One Image Immediately

After environment setup and checkpoint download, you can run Gaussian inference on a single RGB image right away:

```bash
python inference_gs.py \
  --input_image_path=example_data/image/courtyard.jpg \
  --model_type=InfiniDepth \
  --depth_model_path=checkpoints/depth/infinidepth.ckpt \
  --gs_model_path=checkpoints/gs/infinidepth_gs.ckpt
```

To use your own image, just replace `example_data/image/courtyard.jpg` with your file path.

By default this command will:

- save the exported gaussian `.ply` to `example_data/pred_gs/`
- render a novel-view video to `example_data/pred_gs/` at the input image's original resolution

If you want to quickly test depth-only inference first:

```bash
python inference_depth.py \
  --input_image_path=example_data/image/courtyard.jpg \
  --model_type=InfiniDepth \
  --depth_model_path=checkpoints/depth/infinidepth.ckpt \
  --output_resolution_mode=upsample \
  --upsample_ratio=2
```

### 1) Depth Inference (`inference_depth.py`)

Use the provided example scripts:

```bash
bash example_scripts/infer_depth/eth3d_infinidepth_dc.sh   # RGB + Depth mode
bash example_scripts/infer_depth/eth3d_infinidepth.sh      # RGB mode
```

Or run `inference_depth.py` directly:

```bash
python inference_depth.py \
  --input_image_path=... \
  --model_type=InfiniDepth \
  --depth_model_path=checkpoints/depth/infinidepth.ckpt \
  --fx_org (optional) \ 
  --fy_org (optional) \ 
  --cx_org (optional) \ 
  --cy_org (optional) \
  --output_resolution_mode=... \ (optional)
  --output_size=... \ (optional)
  --upsample_ratio=... \ (optional)
```
Output resolution is controlled by three arguments:

- `--output_resolution_mode=upsample`: output size = `input_size * upsample_ratio` (with `upsample_ratio >= 1`)
- `--output_resolution_mode=original`: output size = original input image size
- `--output_resolution_mode=specific`: output size = `output_size` (explicit `(H,W)`)

Use `--model_type=InfiniDepth_DC` together with `--input_depth_path` for sparse depth completion.

### 2) Gaussian Inference (`inference_gs.py`)

Use the provided example scripts:

```bash
bash example_scripts/infer_gs/eth3d_infinidepth_gs.sh       # RGB mode
```

Or run `inference_gs.py` directly:

```bash
python inference_gs.py \
  --input_image_path=... \
  --depth_model_path=checkpoints/depth/infinidepth.ckpt \
  --gs_model_path=checkpoints/gs/infinidepth_gs.ckpt \
  --fx_org (optional) \ 
  --fx_org (optional) \ 
  --fy_org (optional) \ 
  --cx_org (optional) \ 
  --cy_org (optional) \ 
```
Use `--model_type=InfiniDepth_DC` together with `--input_depth_path` and the DC checkpoints for sparse depth completion + gaussian export.
Unless `--render_size=(H,W)` is specified, the exported novel-view video is rendered at the original input image resolution.


> The exported gs files (.ply) can be visualized in 3D viewers like [SuperSplat](https://superspl.at/).

### Common argument conventions:
| Argument | Used in | Description |
|---|---|---|
| `--input_image_path` | depth + gs | Path to input RGB image (required). |
| `--input_depth_path` | depth + gs | Optional metric depth input. If missing, MoGe-2 is used to estimate metric depth prompt. |
| `--model_type` | depth + gs | Depth backbone type: `InfiniDepth` (RGB-only relative depth) or `InfiniDepth_DC` (metric depth with sparse prompt). |
| `--depth_model_path` | depth + gs | Checkpoint path for depth model. |
| `--moge2_pretrained` | depth + gs | MoGe-2 checkpoint path used when `--input_depth_path` is not provided. |
| `--fx_org --fy_org --cx_org --cy_org` | depth + gs | Camera intrinsics in original image resolution. If any is missing, defaults are used (`fx=fy=max(H,W)`, `cx=W/2`, `cy=H/2`). |
| `--input_size` | depth + gs | Network input size as `(H,W)`. All model inference runs on this resized resolution; it does not control the default GS video render resolution. |
| `--enable_skyseg_model` | depth + gs | Whether to enable sky segmentation mask before depth / gaussian sampling logic. |
| `--sky_model_ckpt_path` | depth + gs | Sky segmentation ONNX checkpoint path. |

Depth-only arguments (`inference_depth.py`):

| Argument | Description |
|---|---|
| `--depth_output_dir` | Output directory for colorized depth images. |
| `--pcd_output_dir` | Output directory for exported point cloud (`.ply`). |
| `--save_pcd` | Whether to export point cloud. |
| `--output_resolution_mode` | Depth output mode: `upsample`, `original`, or `specific`. |
| `--upsample_ratio` | Used when `output_resolution_mode=upsample`; output size = `input_size * upsample_ratio`. |
| `--output_size` | Used when `output_resolution_mode=specific`; explicit output `(H,W)`. |

Gaussian-only arguments (`inference_gs.py`):

| Argument | Description |
|---|---|
| `--gs_model_path` | Checkpoint path for gaussian prediction model (InfiniDepth_GS). |
| `--sample_point_num` | Number of sampled points for sparse 3D gaussian construction. Larger values improve density but increase memory/time. |
| `--coord_deterministic_sampling` | Use deterministic coordinate sampling for reproducibility. |
| `--sample_sky_mask_dilate_px` | Dilation radius (pixels) for sky mask before GS sampling. |
| `--output_ply_dir` | Output directory for gaussian `.ply`. |
| `--output_ply_name` | Output filename for gaussian `.ply`. |
| `--render_novel_video` | Whether to render a novel-view video from exported gaussians. |
| `--novel_video_path` | Optional output path for novel-view video. |
| `--novel_trajectory` | Camera path type: `orbit` or `swing`. |
| `--novel_num_frames` | Number of frames in novel-view video. |
| `--novel_video_fps` | FPS of novel-view video. |
| `--novel_radius` | Camera translation amplitude for novel-view motion. |
| `--novel_vertical` | Vertical motion amplitude (mainly for `orbit`). |
| `--novel_forward` | Forward/backward motion amplitude. |
| `--render_size` | Novel-view render size `(H,W)`. If omitted / `None`, defaults to the original input image size and only affects the exported video resolution when explicitly provided. |
| `--novel_bg_color` | Background color as `(R,G,B)` in `[0,1]`. |

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
