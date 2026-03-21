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
| Depth | `InfiniDepth_DepthSensor` | Metric Depth Inference (RGB + Sparse Depth) | [infinidepth_depthsensor.ckpt](https://huggingface.co/ritianyu/InfiniDepth/tree/main) |
| Depth | `MoGe-2` | For scale restoration | [model.pt](https://huggingface.co/Ruicheng/moge-2-vitl-normal) |
| GS | `InfiniDepth_GS` | Gaussian Inference (RGB) | [infinidepth_gs.ckpt](https://huggingface.co/ritianyu/InfiniDepth/tree/main) |
| GS | `InfiniDepth_DepthSensor_GS` | Gaussian inference (RGB + Sparse Depth)| [infinidepth_depthsensor_gs.ckpt](https://huggingface.co/ritianyu/InfiniDepth/tree/main) |
| Sky | `Sky_Mask` | Sky Segmentation | [skyseg.onnx](https://huggingface.co/ritianyu/InfiniDepth/tree/main) |