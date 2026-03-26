## Large-Scale Training Pipeline (splat_buffer branch)

This branch extends the original 3DGS codebase with a two-stage pipeline designed to train on massive, **dense MVS point clouds (100M+ Gaussians)** that cannot fit in GPU VRAM. The key insight is to keep all Gaussian parameters in CPU pinned memory and stream only the visible subset to the GPU for each camera batch, while overlapping data movement with training via a double-buffer design.

The full design rationale is documented in [`DESIGN.md`](DESIGN.md).

---

### Stage 1 — Lazy-Decode Camera Pipeline

**Problem**: loading thousands of high-resolution images into GPU memory at startup wastes RAM/VRAM. Most decoded tensors sit idle while only one is consumed per iteration.

**Solution**:
- **Compressed bytes in RAM** — each image file is read once into memory as raw JPEG/PNG bytes; no decoded pixels are stored at rest.
- **Lazy decode on demand** — `CachedCamera` holds compressed bytes and decodes to a float tensor only when the batch loader requests it (`scene/cached_camera.py`).
- **Parallel CPU decode** — a `ThreadPoolExecutor` pre-decodes a configurable batch in parallel before the GPU needs it.
- **Double-buffered H2D transfer** — `GPUImageBufferPacked` maintains two GPU image slots. While the training loop consumes Slot A, Slot B is being filled via an async H2D copy on a dedicated prefetch CUDA stream. When Slot A is exhausted the buffers swap (`scene/GPUImageBuffer.py`).

**Result**: disk I/O happens once at startup; CPU RAM holds only compressed bytes (10–50× smaller than decoded); the PCIe bus is saturated for the image side of training.

| File | Role |
|------|------|
| `scene/cached_camera.py` | `CachedCamera` — compressed bytes, lazy decode, per-resolution cache |
| `scene/GPUImageBuffer.py` | `GPUImageBufferPacked` — double-buffered packed-batch async H2D transfer |

---

### Stage 2 — Dynamic Gaussian Loading with KNN Batching

**Problem**: a 100M-Gaussian scene requires ~4.8 GB just for parameters, plus Adam state (another ~9.6 GB), far exceeding GPU memory. For any single camera view only a small fraction of Gaussians are visible.

**Architecture overview**:

```
┌─────────────────────────────────────────────────────┐
│  ONE-TIME PRECOMPUTATION (GPU-assisted)             │
│  1. Downsample 100M → 1M points (sampling_ply.py)  │
│  2. Per-camera GPU frustum visibility              │
│  3. KNN overlap graph (IoU-based, CPU-parallel)    │
│  4. Non-overlapping batch pairing for double buf.  │
│  5. Full-scale Gaussian index recomputation        │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  RUNTIME (train_stage2.py)                         │
│  CPUGaussianStore  ←──── all 100M params + Adam   │
│  GaussianSwapBuffer ──── double-buffered GPU slots │
│    Slot A: train B cameras (forward+backward+step) │
│    Slot B: prefetch next batch async               │
│  finish_batch(): writeback params+Adam → CPU swap  │
└─────────────────────────────────────────────────────┘
```

#### Key components

| File | Class | Purpose |
|------|-------|---------|
| `scene/spatial_block_index.py` | `SpatialBlockIndex` | Voxel-grid spatial blocking; prevents OOM during precomputation by processing the full point cloud in chunks |
| `scene/visibility_precomputer.py` | `VisibilityPrecomputer` | GPU frustum culling on the 1M downsampled set; produces per-camera boolean visibility masks |
| `scene/camera_batch_scheduler.py` | `CameraBatchScheduler` | Builds KNN overlap graph, forms batches, pairs non-overlapping batches for the double buffer, recomputes full-scale Gaussian indices |
| `scene/cpu_gaussian_store.py` | `CPUGaussianStore` | All Gaussian params **and Adam optimizer state** in CPU pinned memory; exposes `gather` / `scatter` / `gather_adam` / `scatter_adam` |
| `scene/gpu_gaussian_slice.py` | `GPUGaussianSlice` | Pre-allocated GPU slab; exposes the same interface as `GaussianModel`; builds a warm Adam optimizer from gathered CPU state on each swap |
| `scene/gaussian_swap_buffer.py` | `GaussianSwapBuffer` | Double-buffer managing coupled camera + Gaussian batches; owns `sample_count` and `skipped_batches` |
| `train_stage2.py` | — | Stage-2 training entry point; orchestrates precomputation, swap buffer, and the per-batch training loop |

#### SH degree

Fixed at degree 0 (constant colour per Gaussian — `features_dc` only). No densification, no SH degree upgrade, no exposure optimizer.

#### Running Stage-2 training

```shell
conda run -n GauUscene python train_stage2.py \
    -s <path to COLMAP dataset> \
    -m <output model path> \
    --iterations 30000
```

---

### CPU-Cached Adam Optimizer (most recent addition)

**Problem**: the original Stage-2 design used SGD and discarded optimizer state after each batch. This means every time a Gaussian subset is re-loaded to GPU, its Adam moments are reset to zero — equivalent to restarting training for those Gaussians.

**Solution** — Adam state persisted in `CPUGaussianStore`:

- `_adam_exp_avg` and `_adam_exp_avg_sq` (first and second moments) are stored in CPU pinned memory alongside the parameters, at the same layout and shape.
- A global step counter (`_adam_step`) per param group is maintained on CPU.
- On every slot swap, `GPUGaussianSlice` calls `gather_adam()` to fetch the stored moments for the current batch's Gaussians and **injects them directly into a new `torch.optim.Adam` instance** — the optimizer starts warm, not cold.
- After training, `scatter_adam()` writes the updated moments back to CPU.

**Zero-gradient Adam for non-batch Gaussians** (`apply_zero_grad_adam`):

Gaussians not in the active GPU batch receive no gradient for the duration of that batch. However, Adam's first moment continues to decay, and the residual momentum would still drive a parameter update — exactly what the full-GPU training loop (`train_fix.py`) does. To replicate this behaviour without moving those Gaussians to GPU:

```
For n zero-gradient steps starting at Adam step t₀:
  m_{t₀+n} = β₁ⁿ · m_{t₀}            (pure decay, no new gradient)
  v_{t₀+n} = β₂ⁿ · v_{t₀}            (pure decay, no new gradient)

Cumulative parameter delta (geometric series):
  r = β₁ / √β₂
  geom_coeff = r · (1 − rⁿ) / (1 − r)
  Δθ = −lr · (m_{t₀} / (√v_{t₀} + ε)) · geom_coeff  (bias-correction omitted for clarity)
```

This CPU-side update runs **concurrently** with GPU training on the active batch via a background `ThreadPoolExecutor`, adding zero wall-clock overhead on most iterations.

**Memory budget at 100M Gaussians**:

| Region | Size |
|--------|------|
| Parameters (xyz, f_dc, scaling, rotation, opacity) | ~4.8 GB pinned |
| Adam exp_avg (×5 param groups) | ~4.8 GB pinned |
| Adam exp_avg_sq (×5 param groups) | ~4.8 GB pinned |
| **Total CPU pinned** | **~14.4 GB** |

GPU holds only the active slice (~5–10M Gaussians per slot × 2 slots ≈ <1 GB for params).

---

### Rasterizer — Absorbed as Regular Source Code

The `diff-gaussian-rasterization` submodule has been absorbed into the repository as tracked source files (commit `bc9c1ea`). Local modifications (high-watermark persistent CUDA buffers, version 0.0.2) are committed directly rather than pinned to an external commit reference.

---

## Step-by-step Tutorial

Jonathan Stephens made a fantastic step-by-step tutorial for setting up Gaussian Splatting on your machine, along with instructions for creating usable datasets from videos. If the instructions below are too dry for you, go ahead and check it out [here](https://www.youtube.com/watch?v=UXtuigy_wYc).

## Colab

User [camenduru](https://github.com/camenduru) was kind enough to provide a Colab template that uses this repo's source (status: August 2023!) for quick and easy access to the method. Please check it out [here](https://github.com/camenduru/gaussian-splatting-colab).

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```

## Overview

The codebase has 4 main components:
- A PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs
- A network viewer that allows to connect to and visualize the optimization process
- An OpenGL-based real-time viewer to render trained models in real-time.
- A script to help you turn your own images into optimization-ready SfM data sets

The components have different requirements w.r.t. both hardware and software. They have been tested on Windows 10 and Ubuntu Linux 22.04. Instructions for setting up and running each of them are found in the sections below.




## Optimizer

The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)
- Please see FAQ for smaller VRAM configurations

### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used Visual Studio 2019 for Windows)
- CUDA SDK 11 for PyTorch extensions, install *after* Visual Studio (we used 11.8, **known issues with 11.6**)
- C++ Compiler and CUDA SDK must be compatible

### Setup

#### Local Setup

Our default, provided install method is based on Conda package and environment management:
```shell
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate gaussian_splatting
```
Please note that this process assumes that you have CUDA SDK **11** installed, not **12**. For modifications, see below.

Tip: Downloading packages and creating a new environment with Conda can require a significant amount of disk space. By default, Conda will use the main system hard drive. You can avoid this by specifying a different package download location and an environment on a different drive:

```shell
conda config --add pkgs_dirs <Drive>/<pkg_path>
conda env create --file environment.yml --prefix <Drive>/<env_path>/gaussian_splatting
conda activate <Drive>/<env_path>/gaussian_splatting
```

#### Modifications

If you can afford the disk space, we recommend using our environment files for setting up a training environment identical to ours. If you want to make modifications, please note that major version changes might affect the results of our method. However, our (limited) experiments suggest that the codebase works just fine inside a more up-to-date environment (Python 3.8, PyTorch 2.0.0, CUDA 12). Make sure to create an environment where PyTorch and its CUDA runtime version match and the installed CUDA SDK has no major version difference with PyTorch's CUDA version.

#### Known Issues

Some users experience problems building the submodules on Windows (```cl.exe: File not found``` or similar). Please consider the workaround for this problem from the FAQ.

### Running

To run the optimizer, simply use

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>
<br>

Note that similar to MipNeRF360, we target images at resolutions in the 1-1.6K pixel range. For convenience, arbitrary-size inputs can be passed and will be automatically resized if their width exceeds 1600 pixels. We recommend to keep this behavior, but you may force training to use your higher-resolution images by setting ```-r 1```.

The MipNeRF360 scenes are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/). You can find our SfM data sets for Tanks&Temples and Deep Blending [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip). If you do not provide an output model directory (```-m```), trained models are written to folders with randomized unique names inside the ```output``` directory. At this point, the trained models may be viewed with the real-time viewer (see further below).

### Evaluation
By default, the trained models use all available images in the dataset. To train them while withholding a test set for evaluation, use the ```--eval``` flag. This way, you can render training/test sets and produce error metrics as follows:
```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # Train with train/test split
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

If you want to evaluate our [pre-trained models](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip), you will have to download the corresponding source data sets and indicate their location to ```render.py``` with an additional ```--source_path/-s``` flag. Note: The pre-trained models were created with the release codebase. This code base has been cleaned up and includes bugfixes, hence the metrics you get from evaluating them will differ from those in the paper.
```shell
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
python metrics.py -m <path to pre-trained model>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for render.py</span></summary>

  #### --model_path / -m 
  Path to the trained model directory you want to create renderings for.
  #### --skip_train
  Flag to skip rendering the training set.
  #### --skip_test
  Flag to skip rendering the test set.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 

  **The below parameters will be read automatically from the model path, based on what was used for training. However, you may override them by providing them explicitly on the command line.** 

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Changes the resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. ```1``` by default.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --convert_SHs_python
  Flag to make pipeline render with computed SHs from PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline render with computed 3D covariance from PyTorch instead of ours.

</details>

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for metrics.py</span></summary>

  #### --model_paths / -m 
  Space-separated list of model paths for which metrics should be computed.
</details>
<br>
