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

#### Option A: Resolve everything from `pyproject.toml` with `uv sync` (recommended)

This repository's `pyproject.toml` already declares every dependency, including the PyTorch CUDA 12.8 index and the git sources for MoGe, gsplat, and t4-devkit. With `uv` installed, you can create a virtual environment and resolve all dependencies in a single command.

```bash
# Install uv (only if you don't have it yet)
pip install uv

# Read pyproject.toml / uv.lock, create a .venv, and install all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

Notes:
- `uv sync` automatically creates `.venv` and resolves PyTorch (cu128) and the git dependencies (MoGe / gsplat / t4-devkit) according to `[tool.uv.sources]` and `[[tool.uv.index]]` in `pyproject.toml`.
- If you prefer to install into an existing conda/mamba environment instead of a new `.venv`, pass `uv sync --active` to target the currently active interpreter.
- `gsplat` builds CUDA extensions at install time, so make sure `nvcc` / CUDA Toolkit 12.8 and a compatible `g++` are available (see the optional commands in "1) Create environment").
- Run `uv lock --upgrade` when you want to refresh the lock file.

#### Option B: Install packages individually with `uv pip`

If you'd rather install packages one by one (similar to a plain `pip` workflow) without going through `pyproject.toml`, use the following steps.

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