# Dynamic Gaussian Splatting Training Pipeline — Design Document

## Overview

This document captures the design for a two-stage optimization of the Gaussian Splatting training pipeline. The goal is to train scenes with very large point clouds (millions of Gaussians from dense MVS) that cannot fit entirely in GPU memory, while keeping the GPU fully utilized.

---

## Stage 1: Lazy-Decode Camera Pipeline (Completed)

### Problem

Loading thousands of high-resolution images into GPU memory at initialization is wasteful. Most images sit idle in decoded form (H×W×C float tensors), consuming RAM and VRAM while only one is needed per iteration.

### Solution

- **Compressed bytes in RAM**: Each image file is read once into memory as raw bytes (JPEG/PNG). No decoded pixels are stored at rest.
- **Lazy decode on demand**: `CachedCamera` holds compressed bytes and only decodes + resizes to a torch tensor when the batch loader requests it.
- **Shared blobs across resolutions**: For coarse-to-fine training (×8 → ×4 → ×2 → ×1), the same compressed bytes are reused. Each resolution scale produces a different `CachedCamera` that points to the same blob but decodes to a different target size.
- **Parallel CPU decode**: A shared `ThreadPoolExecutor` pre-decodes a batch of images in parallel before packing them into pinned memory.
- **Double-buffered GPU transfer**: `GPUImageBufferPacked` maintains two GPU slots. While the training loop consumes images from Slot A, Slot B is being filled via async H2D copy on a prefetch CUDA stream. When Slot A is exhausted, buffers swap.

### Key Files

| File | Role |
|------|------|
| `scene/cached_camera.py` | `CachedCamera` — stores compressed bytes, lazy decode, per-resolution cache |
| `utils/camera_utils.py` | `loadCam()` — reads file once, builds `CachedImageBlob` + `CachedCamera` |
| `scene/GPUImageBuffer.py` | `GPUImageBufferPacked` — double-buffered packed batch H2D transfer |
| `scene/__init__.py` | `Scene` — preloads bytes once, creates thread pool, passes to buffer |

### Result

- Disk I/O happens once at startup.
- CPU RAM holds only compressed bytes (10–50× smaller than decoded).
- GPU sees decoded uint8 tensors via async batched copy.
- Bus is saturated for the image/camera side.

---

## Stage 2: Dynamic Gaussian Loading with KNN Batching

### Problem

The dense MVS point cloud produces ~100M Gaussians. The full parameter set (positions, SH coefficients, scales, rotations, opacities) plus Adam optimizer states (2× parameter size) exceed GPU memory. However, for any single camera view, only a fraction of Gaussians are visible.

### Key Assumptions

1. **No densification or pruning.** The Gaussian set is fixed throughout training (dense MVS input).
2. **Static visibility.** Each camera sees a fixed set of Gaussians. This mapping is computed once before training and never changes.
3. **Spatial locality.** Cameras looking at similar parts of the scene share most of their visible Gaussians.

### Core Idea

Use a **downsampled point cloud** (~1M from ~100M) on GPU for fast overlap computation. Build a KNN graph of cameras by Gaussian overlap. For each batch, find a non-overlapping partner batch for the double buffer. At training time, use the precomputed schedule to load the right Gaussians (at full 100M scale) without runtime frustum culling.

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   ONE-TIME PRECOMPUTATION (on GPU)                     │
│                                                                        │
│  1. Downsample 100M → 1M points                                       │
│  2. Visibility on downsampled set: per-camera visible indices (GPU)    │
│  3. Overlap matrix → KNN graph (N cameras × B neighbours)             │
│  4. Per-batch downsampled Gaussian union (expanded FOV)                │
│  5. Non-overlapping batch pairing (warn if impossible)                 │
│  6. Full-scale Gaussian index recomputation per batch                  │
│  7. Per-camera lookup table:                                           │
│       { batch_neighbours, unrelated_batches, gaussian_indices }        │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         RUNTIME TRAINING                               │
│                                                                        │
│  CPU Gaussian Store (pinned memory, full 100M, params only)            │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  All params: _xyz, _features, _scaling, _rotation, _opacity   │     │
│  │  No optimizer states (stateless SGD, discarded per batch)     │     │
│  │  Fixed layout — Gaussian i always lives at offset i           │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                        │
│  Double Buffer (2 GPU slots)                                           │
│  ┌──────────────┐    ┌──────────────┐                                  │
│  │   Slot A      │    │   Slot B      │                                │
│  │  Gaussians    │    │  Gaussians    │                                │
│  │  + Images     │    │  + Images     │                                │
│  └──────────────┘    └──────────────┘                                  │
│                                                                        │
│  Scheduling:                                                           │
│  1. Pick camera A → look up batch_neighbours → load batch Gaussians   │
│  2. Pick next from unrelated_batches → prefetch into other slot       │
│  3. Train on active slot while prefetch runs async                     │
│  4. Writeback active slot → swap → repeat                             │
│                                                                        │
│  Timeline:                                                             │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │ Train Batch A  │  │ Train Batch B  │  │ Train Batch C  │  ...       │
│  │ (A is unrelated│  │ (B is unrelated│  │ (C is unrelated│            │
│  │  to Batch B)   │  │  to Batch C)   │  │  to Batch D)   │            │
│  └────────────────┘  └────────────────┘  └────────────────┘            │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │ Prefetch B     │  │ Writeback A    │  │ Writeback B    │            │
│  │ (no conflict   │  │ Prefetch C     │  │ Prefetch D     │            │
│  │  with A)       │  │ (no conflict)  │  │ (no conflict)  │            │
│  └────────────────┘  └────────────────┘  └────────────────┘            │
│       prefetch stream     prefetch stream     prefetch stream          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

### Precomputation Phase

#### Step 1: Downsample Point Cloud

The raw MVS point cloud has ~100M points. Downsample to ~1M for all overlap/KNN computation.

- Use voxel grid downsampling or farthest point sampling.
- No mapping back to full-scale indices is needed (Step 6 recomputes visibility at full scale independently).
- Load the 1M downsampled points onto GPU as a `(1M, 3)` float32 tensor.

#### Step 2: Per-Camera Visibility Table (Downsampled, on GPU → CPU boolean)

For each camera `i`, compute `visible_ds[i]` = **boolean array of shape (1M,)** indicating which downsampled Gaussians are visible.

**Algorithm** (one camera at a time on GPU, result transferred to CPU):

```
All 1M downsampled points live on GPU as points_gpu: (1M, 4) homogeneous

For each camera i:
    1. Get W2C transform (4×4), upload to GPU
    2. Project all points: p_cam = W2C @ points_gpu^T       (GPU matmul)
    3. Filter: z > znear
    4. Perspective project with EXPANDED FOV (add margin)
    5. Filter: within expanded image bounds
    6. Produce visible_ds[i]: (1M,) bool tensor on GPU
    7. Transfer to CPU as np.ndarray bool (1 MB per camera)
```

With 1M points this is a single `(4,4) @ (4, 1M)` matmul per camera — milliseconds on GPU.

**Output**: `visible_ds: np.ndarray` of shape `(N_cameras, 1M)`, dtype bool, stored on CPU.
Total memory: N × 1 MB (e.g., 2 GB for N=2000; use `np.packbits` to reduce to N × 125 KB if needed).

#### Step 3: Camera Overlap KNN Graph (CPU, parallelized)

Build an `N × N` IoU overlap matrix and extract top-B neighbours per camera.

**Algorithm**:

```
visible_ds: (N, 1M) bool array on CPU

Pairwise IoU (parallelized with ThreadPoolExecutor):
    For each pair (i, j), i < j:
        intersection = (visible_ds[i] & visible_ds[j]).sum()
        union        = (visible_ds[i] | visible_ds[j]).sum()
        iou[i, j] = iou[j, i] = intersection / union   (0.0 if union == 0)

For each camera i:
    Sort iou[i, :] descending, exclude self → top B-1 neighbours

Store as knn_table: (N, B) int array
    knn_table[i] = [camera_i, neighbour_1, ..., neighbour_{B-1}]
```

**Parallelization**: the N*(N-1)/2 pairs are distributed across a `ThreadPoolExecutor`. At N=2000,
~2M pairs × ~0.1ms each ≈ 200s single-threaded; parallelized over available CPU cores this reduces
to seconds. NumPy bitwise ops release the GIL, so true parallelism is achieved.

**Output**: `knn_table: np.ndarray` of shape `(N_cameras, B)`.

#### Step 4: Per-Batch Gaussian Union (Downsampled)

For each row in `knn_table` (i.e., each batch of B cameras):

```
batch_gaussians_ds[i] = visible_ds[knn_table[i][0]]
for cam in knn_table[i][1:]:
    batch_gaussians_ds[i] |= visible_ds[cam]   # bitwise OR over (1M,) bool arrays
```

**Output**: `batch_gaussians_ds: np.ndarray` of shape `(N_cameras, 1M)`, dtype bool —
one union boolean mask per anchor camera.

#### Step 5: Non-Overlapping Batch Pairing

Two batches are **unrelated** if and only if:
1. They share **no cameras** (no camera id appears in both batches), AND
2. They share **no Gaussians** (their downsampled Gaussian unions are disjoint).

Condition (1) is necessary because shared cameras imply shared Gaussians.

```
For each batch i:
    cameras_i = set(knn_table[i])
    unrelated[i] = []
    for each batch j ≠ i:
        cameras_j = set(knn_table[j])
        if cameras_i ∩ cameras_j != ∅:
            continue                     # shared camera → skip
        if not (batch_gaussians_ds[i] & batch_gaussians_ds[j]).any():   # bitwise AND, check disjoint
            unrelated[i].append(j)

    if len(unrelated[i]) == 0:
        WARN: "No fully unrelated batch found for batch {i}."
        # Fallback: pick the batch with least overlap (excluding shared-camera batches)
        candidates = [j for j if cameras_i ∩ set(knn_table[j]) == ∅]
        if candidates:
            overlaps = {j: (batch_gaussians_ds[i] & batch_gaussians_ds[j]).sum() for j in candidates}
            unrelated[i] = [min(overlaps, key=overlaps.get)]
        else:
            # Even camera-disjoint batches not found; pick global least overlap
            overlaps = {j: (batch_gaussians_ds[i] & batch_gaussians_ds[j]).sum() for j ≠ i}
            unrelated[i] = [min(overlaps, key=overlaps.get)]
        # Mark this pair as having a potential dirty write
```

**Output**: `Dict[int, List[int]]` — `batch_id → list of unrelated batch ids`.

**Note**: When a batch has no fully unrelated partner, a warning is issued. The system still works — the dirty write (concurrent read/write to shared CPU memory) is tolerated. In practice this is rare for spatially structured scenes.

#### Step 6: Full-Scale Gaussian Index Recomputation

Recompute visible Gaussian indices at **full 100M scale**. The 100M points are loaded to GPU
**once** and kept resident throughout; all per-camera projections reuse that buffer.

**Memory budget during this step (24 GB GPU)**:
- `points_full_gpu: (100M, 4) float32` = 1.6 GB — loaded once, stays resident on GPU
- `union_mask: (100M,) bool` on GPU = 100 MB — one per batch, allocated/freed each batch
- `cam_mask: (100M,) bool` on GPU = 100 MB — one per camera within a batch, freed after OR
- Peak per batch: ~1.8 GB. Comfortably within 24 GB.

**Why not cache per-camera bool masks on CPU?**
Caching all N cameras' masks on CPU would require N × 12.5 MB (bit-packed) = **25 GB for N=2000** —
too large. Instead, process per-batch: accumulate the union directly on GPU, transfer only the
final union mask (100 MB), discard it after converting to a compact index array.

**PCIe bandwidth**:
- H2D (100M points): **1.6 GB × 1** — single transfer for entire Step 6
- D2H (union masks): 100 MB × N_batches (e.g., ~100 GB for N=1000 batches)

**Tradeoff**: cameras appear in ~B batches (as anchor + neighbour), so GPU projections = N_cameras × B
vs. N_cameras for a caching approach. At B=8, N=1000: 8000 GPU projections at ~ms each — negligible.
GPU compute is cheap; 25 GB of CPU RAM is not.

**Algorithm**:

```
points_full_gpu: (100M, 4) float32  ← loaded to GPU ONCE, stays resident

For each batch i (cameras = knn_table[i]):
    union_mask = zeros(100M, bool) on GPU           ← 100 MB, freshly allocated

    For each camera cam in knn_table[i]:            ← B iterations
        p_cam = w2c[cam] @ points_full_gpu.T        → (4, 100M) on GPU
        cam_mask = z_filter & perspective & in_bounds(p_cam)   → (100M,) bool
        union_mask |= cam_mask                      ← in-place OR, cam_mask freed

    union_cpu = union_mask.cpu().numpy()            ← 100 MB D2H transfer
    gaussian_indices[i] = np.nonzero(union_cpu)[0] ← compact sorted int32 array
    del union_mask                                  ← free 100 MB on GPU
```

**Output**: `gaussian_indices: Dict[int, np.ndarray]` — `batch_id → sorted full-scale Gaussian indices`.

#### Step 7: Per-Camera Lookup Table

Assemble the final schedule as a dictionary per camera:

```python
@dataclass
class BatchInfo:
    anchor_camera_id: int
    batch_neighbours: List[int]       # B-1 other camera ids in this batch
    unrelated_batches: List[int]      # batch ids with no (or minimal) Gaussian overlap
    gaussian_indices: np.ndarray      # full-scale sorted Gaussian indices for this batch
    has_dirty_overlap: bool           # True if no fully clean partner exists
    n_gaussians: int                  # len(gaussian_indices) for quick budget check
```

**Output**: `Dict[int, BatchInfo]` — `camera_id → its batch info`.

#### Step 8: Camera Sample-Rate Balancing Table

Each camera `i` appears in multiple batches (as anchor or as neighbour). Track how often each camera has been trained:

```python
sample_count: np.ndarray  # shape (N_cameras,), initialized to 0
```

**Anchor selection policy**: instead of uniform random, prefer cameras with the lowest `sample_count`. If all equal, pick randomly. After training a batch, increment `sample_count[cam]` for every camera in that batch.

This ensures all cameras converge to roughly equal training frequency over time.

---

### Runtime Components

#### GPU Memory Budget

Target GPU: **24 GB VRAM**. Budget per slot:

| Region | Budget |
|--------|--------|
| Gaussian params per slot (×2 for double buffer) | 4 GB × 2 = 8 GB |
| Rasterizer workspace + gradients + images | ~16 GB |

Per-Gaussian memory (SH0, params only — no optimizer states stored):

| Field | Bytes |
|-------|-------|
| xyz (3 floats) | 12 |
| features_dc (3 floats) | 12 |
| scaling (3 floats) | 12 |
| rotation (4 floats) | 16 |
| opacity (1 float) | 4 |
| **Total** | **56 bytes** |

4 GB / 56 bytes ≈ **~70M Gaussians per slot**. In practice batches are 5–10M, well within budget.

**Over-budget handling**: during precomputation, check each batch's `n_gaussians`. If a batch exceeds the 4 GB slot cap, **warn the user and skip** that batch at training time. At the end of training, report how many batches were skipped.

#### SH Degree

**Fixed at degree 0** (constant colour per Gaussian — `features_dc` only, no `features_rest`).
`active_sh_degree` tracking, `oneupSHdegree()`, and the `features_rest` tensor are all removed.
`get_features()` always returns `features_dc`. `GaussianModel` is fully replaced by
`CPUGaussianStore` + `GPUGaussianSlice` — no metadata shell is kept alive.

The `exposure_optimizer` (per-camera tone mapping) is also dropped.

#### Stateless Optimizer (SGD, No Persistent States)

**Decision**: optimizer states are **not persisted** across batch visits. Each time a Gaussian
subset is loaded to GPU, a fresh `torch.optim.SGD` is created for the slice. After training B
cameras, only the updated **parameters** are written back to CPU. Optimizer states are discarded.

**Learning rate**: maintained by the training loop on CPU. The current LR value is set directly on
each slot's fresh SGD optimizer at every step (`pg['lr'] = lr_schedule(global_step)`). No
`update_learning_rate()` method is needed.

**Rationale**:
- Each batch trains only B=8 steps — Adam barely warms up before the Gaussians are swapped out.
- Discarding states saves 3× transfer bandwidth and 3× CPU pinned memory (~11 GB saved at 100M scale).
- SGD with a scheduled LR provides the main learning rate control.

#### `CPUGaussianStore`

All Gaussian parameters in CPU **pinned memory** as contiguous tensors. **No optimizer states.
No `features_rest` (SH degree 0 only).**

- `_xyz`: `(N, 3)` float32, pinned
- `_features_dc`: `(N, 1, 3)` float32, pinned
- `_scaling`: `(N, 3)` float32, pinned
- `_rotation`: `(N, 4)` float32, pinned
- `_opacity`: `(N, 1)` float32, pinned

Total CPU pinned memory at 100M Gaussians: **~4.8 GB**.

Fixed layout: Gaussian `i` is always at row `i`. No reindexing ever.

**Gather/scatter**: for a batch of ~5–10M indices out of 100M, use `tensor[indices]` (PyTorch CPU fancy indexing). This is a single vectorized gather per parameter tensor — fast for contiguous source tensors even with random access patterns. No special parallelization needed beyond PyTorch's internal threading.

**Invisible Gaussians**: Gaussians that appear in no camera's visible set are simply ignored. They waste a small amount of CPU memory but never transfer to GPU or receive gradients.

#### `GPUGaussianSlice`

A subset of Gaussians on GPU, exposing the same interface as `GaussianModel` (SH degree 0 only):
- `get_xyz`, `get_scaling`, `get_rotation`, `get_opacity`, `get_features` — all work as usual.
- The renderer sees no difference. No `active_sh_degree` — `get_features()` always returns `features_dc`.

**Pre-allocated GPU buffers (no dealloc/realloc on swap):**
Each slot pre-allocates fixed-size GPU tensors sized for the largest possible batch
(slot_budget / 56 bytes ≈ 70M rows). On swap, the next batch's params are written
**in-place** into the existing allocation. Only `valid_length` changes.

```
slot._xyz:         (MAX_K, 3)  float32  nn.Parameter  — pre-allocated once
slot._features_dc: (MAX_K, 1, 3) float32 nn.Parameter
slot._scaling:     (MAX_K, 3)  float32  nn.Parameter
slot._rotation:    (MAX_K, 4)  float32  nn.Parameter
slot._opacity:     (MAX_K, 1)  float32  nn.Parameter

slot.valid_length: int   ← current batch size K ≤ MAX_K
slot.cpu_indices:  np.ndarray  ← global Gaussian indices for writeback scatter
```

Renderer and optimizer operate on `slot._xyz[:valid_length]` etc. — views into the slab,
no extra allocation. A **fresh SGD optimizer** is built over the `[:valid_length]` views each
swap. No state carries over between batch visits.

#### `GaussianSwapBuffer`

Double-buffered loader that extends the existing `GPUImageBufferPacked` pattern.
Cameras and Gaussians are loaded together as a coupled unit.

**CPUCacher thread**: a dedicated background thread that listens for CUDA events. When the GPU
signals a slot is done (training complete), the thread:
1. **Writeback**: `slot._xyz[:K].cpu()` → scatter into `CPUGaussianStore` at `cpu_indices`. Params only — optimizer states discarded.
2. **In-place overwrite**: gather next batch from `CPUGaussianStore` into pinned CPU slabs, then copy into `slot._xyz[:new_K]` etc. directly — no dealloc/realloc. Update `slot.valid_length = new_K`.
3. Record a new CUDA event marking the slot as ready for training.

**Prefetch** (CPUCacher thread, async H2D on prefetch CUDA stream):
1. Gather rows from `CPUGaussianStore` at the next batch's `gaussian_indices` into pinned slabs.
2. Async copy pinned slabs → `slot._xyz[:new_K]` etc. (in-place, no new GPU allocation).
3. Simultaneously decode + pack camera images for the batch (Stage 1 pipeline).
4. Record CUDA event marking slot ready.

**Writeback** (CPUCacher thread, triggered by CUDA event on training completion):
1. D2H: `slot._xyz[:K]` → pinned slab → scatter into `CPUGaussianStore[cpu_indices]`.
2. No optimizer states transferred — discarded on GPU.

**Camera scheduling via modified `GPUImageBufferPacked`**:

`GPUImageBufferPacked` is extended minimally to support KNN-aware batch scheduling. The core
`_pack_and_copy` H2D logic is **unchanged**. Only `_next_batch()` is replaced:

```
Original: pull next B cameras from random-shuffled flat order
Modified: pull cameras for a specific BatchInfo (anchor + neighbours)
```

When filling slot A with batch i:
1. Look up `batch_infos[anchor_i]` → get the B camera ids for this batch.
2. Read `batch_infos[anchor_i].unrelated_batches` → randomly pick one unrelated batch j.
3. Schedule slot B to be filled with `batch_infos[anchor_j]`'s cameras.

Both slots' camera packing uses the same `_pack_and_copy` path — no duplication.
`sample_count` and anchor selection live in `GaussianSwapBuffer`; the modified
`GPUImageBufferPacked` receives the pre-chosen camera list for each slot fill.

**Scheduling logic** (owned by `GaussianSwapBuffer`):
1. Pick anchor camera (prefer lowest `sample_count`) → look up `BatchInfo`.
2. Check `n_gaussians` ≤ slot budget. If over budget, skip and log; pick another anchor.
3. Fill active slot: cameras from `batch_infos[anchor].batch_neighbours` via modified `GPUImageBufferPacked`. Load Gaussians in-place into GPU slab.
4. From `batch_infos[anchor].unrelated_batches`, randomly pick next anchor → fill other slot.
5. Train on active slot (B forward+backward+step iterations).
6. Caller calls `finish_batch()` → CPUCacher thread does async writeback + in-place overwrite of freed slot.
7. Update `sample_count` for all B cameras in the completed batch.

**`pop()`** returns a coupled pair: `(camera_views, gaussian_slice)`.
**`finish_batch()`** signals CPUCacher thread. Must be called after the last `optimizer.step()`. Optimizer states are discarded.

---

### Modified Training Loop (Pseudocode)

```python
# ── Precomputation ──
points_ds = downsample(full_points, target=1_000_000)
visibility_ds = VisibilityPrecomputer(cameras, points_ds, fov_margin=0.1, device="cuda")
knn_table = build_knn_overlap(visibility_ds, batch_size=8)
batch_infos = build_batch_infos(knn_table, visibility_ds, full_points, cameras)
#   batch_infos: Dict[int, BatchInfo]
#     .batch_neighbours     → [B-1 camera ids]
#     .unrelated_batches    → [batch ids with no Gaussian overlap AND no shared cameras]
#     .gaussian_indices     → full-scale sorted indices
#     .n_gaussians          → quick budget check
#     .has_dirty_overlap    → bool

# ── Setup ──
cpu_store = CPUGaussianStore(full_points, pinned=True)
B = 8  # cameras per batch
swap_buffer = GaussianSwapBuffer(
    cpu_store, batch_infos, cameras,
    device="cuda",
    slot_budget_bytes=4 * 1024**3,   # 4 GB per slot
)
# swap_buffer owns: sample_count, skipped_batches (internal state)

# Prefill both slots (pick first anchor, then an unrelated batch)
swap_buffer.prefetch_initial()

global_step = 0
while global_step < total_iters:
    # Pop returns the batch already on GPU
    camera_views, gaussians = swap_buffer.pop()
    #   camera_views: list of B PackedCameraView (images decoded + on GPU)
    #   gaussians:    GPUGaussianSlice (fresh SGD optimizer, no carried state)

    # B gradient steps per batch (one per camera)
    for cam in camera_views:
        lr = lr_schedule(global_step)
        for pg in gaussians.optimizer.param_groups:
            pg['lr'] = lr

        render_pkg = render(cam, gaussians, pipe, background, ...)
        loss = compute_loss(render_pkg, cam, ...)
        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad()
        global_step += 1

    # Explicit writeback (params only) + swap
    swap_buffer.finish_batch()
    #   1. Async writeback active slot Gaussian params → CPU (no optimizer states)
    #   2. Pick next anchor (lowest sample_count), skip if over budget
    #   3. Async prefetch next batch into other slot
    #   4. Swap active/inactive slots
    #   5. Update sample_count for all B cameras in the completed batch

print(f"Training complete. Skipped {swap_buffer.skipped_batches} over-budget batches.")
```

### Iteration Counting

One **iteration** (= one `global_step`) = one image loss + one `optimizer.step()`. A batch of B cameras produces B iterations. The `--iterations` flag counts total gradient steps. The outer loop runs until `global_step >= total_iters`.

### State Ownership

- `sample_count` and `skipped_batches` live inside `GaussianSwapBuffer`.
- `sample_count` is exposed as a read-only property for logging.
- `skipped_batches` is exposed as a read-only property for the final report.

### Checkpoint Saving

To save a checkpoint, call `swap_buffer.finish_batch()` first to flush all GPU-side updates to CPU, then write `CPUGaussianStore` to PLY. No optimizer states need saving.

### Coarse-to-Fine Resolution

The Gaussian batch size per slot is **fixed** across all resolution stages (sized for the worst case at full resolution). At lower resolutions, images are smaller and the rasterizer uses less VRAM, but no dynamic adjustment is made — simplicity over optimality.

---

### Benefits of This Design

| Property | Benefit |
|----------|---------|
| Downsampled overlap computation on GPU | Fast precomputation even with 100M points |
| KNN-based camera batching | Cameras in one batch share most Gaussians → small union |
| Non-overlapping batch pairing for double buffer | Writeback and prefetch touch different CPU memory |
| Graceful fallback for dirty overlap | System still works; user is warned |
| Fixed Gaussian indices (no densification) | No reindexing, simple scatter/gather |
| One-time precomputation | Zero runtime frustum culling overhead |
| Full-scale index recomputation only once | Accurate at 100M level, cheap at 1M level |
| Per-camera lookup table | O(1) scheduling decisions at runtime |
| Coupled camera + Gaussian batches | Single `pop()` gives everything needed |

---

### Proposed File Layout

| File | Class | Purpose |
|------|-------|---------|
| `scene/visibility_precomputer.py` | `VisibilityPrecomputer` | GPU-accelerated backprojection on downsampled points → per-camera visible index sets |
| `scene/camera_batch_scheduler.py` | `CameraBatchScheduler` | KNN overlap graph + batch formation + non-overlap pairing + full-scale recomputation |
| `scene/cpu_gaussian_store.py` | `CPUGaussianStore` | All params in pinned CPU memory, no optimizer states (100M scale) |
| `scene/gpu_gaussian_slice.py` | `GPUGaussianSlice` | GPU-resident subset, same interface as `GaussianModel` |
| `scene/gaussian_swap_buffer.py` | `GaussianSwapBuffer` | Double-buffer for coupled camera + Gaussian batches with scheduling |

---

### Implementation Order

1. `VisibilityPrecomputer` — GPU-accelerated, self-contained: cameras + downsampled points → per-camera visible index sets.
2. `CameraBatchScheduler` — KNN graph + batch plans + non-overlap pairing + full-scale Gaussian index recomputation.
3. `CPUGaussianStore` — refactor `GaussianModel` storage into pinned CPU tensors at full 100M scale.
4. `GPUGaussianSlice` — GPU subset exposing same interface as `GaussianModel`.
5. `GaussianSwapBuffer` — double buffer coupling cameras and Gaussians, with scheduling from `BatchInfo`.
6. Wire into training loop — replace `gaussians` + `camera_loader` with the coupled buffer.
