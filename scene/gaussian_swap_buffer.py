import random
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from scene.camera_batch_scheduler import BatchInfo
from scene.cpu_gaussian_store import CPUGaussianStore
from scene.gpu_gaussian_slice import GPUGaussianSlice
from scene.GPUImageBuffer import GPUImageBufferPacked
from scene.GPUHelper.pachedCamera import PackedCameraView
from scene.spatial_block_index import SpatialBlockIndex


class GaussianSwapBuffer:
    """
    Double-buffered coupled loader for Gaussians + camera images.

    Owns two GPUGaussianSlice slots and uses the (minimally modified)
    GPUImageBufferPacked for camera H2D packing.

    Scheduling:
      - The "active" slot is the one currently being trained on.
      - The "inactive" slot holds the NEXT batch, already loaded and ready.
      - finish_batch() writebacks the active slot, picks the next batch
        from the active batch's unrelated_batches, prefetches it into the
        (now-freed) active slot, then swaps.

    pop() returns (camera_views, gaussian_slice) for the active slot.
    finish_batch() must be called after the last optimizer.step().

    Thread model: single-threaded Python with async CUDA streams.
      - H2D copies run on self.prefetch_stream.
      - Training runs on the default (compute) stream.
      - prefetch_stream.synchronize() ensures D2H writeback is complete
        before CPU scatter / gather.
    """

    def __init__(
        self,
        cpu_store:         CPUGaussianStore,
        batch_infos:       Dict[int, BatchInfo],
        cameras:           List[Any],             # all CachedCamera objects
        block_index:       SpatialBlockIndex,
        device:            str = "cuda",
        slot_budget_bytes: int = 4 * 1024 ** 3,  # 4 GB per slot
        batch_size:        int = 8,
        lr_dict:           Optional[Dict[str, float]] = None,
        decode_executor=None,
    ):
        self.cpu_store   = cpu_store
        self.batch_infos = batch_infos
        self.cameras     = cameras
        self.block_index = block_index
        self.device      = torch.device(device)
        self.batch_size  = batch_size
        self.lr_dict     = lr_dict or {}

        # ---- GPU Gaussian slots ----------------------------------------
        bytes_per_gaussian = 56
        max_gaussians = slot_budget_bytes // bytes_per_gaussian
        self.slots = [
            GPUGaussianSlice(max_gaussians, device=device),
            GPUGaussianSlice(max_gaussians, device=device),
        ]

        # ---- CUDA infrastructure ----------------------------------------
        self.prefetch_stream = torch.cuda.Stream(device=self.device)
        # One "ready" event per slot: fires when H2D load is complete
        self.slot_ready = [
            torch.cuda.Event(enable_timing=False),
            torch.cuda.Event(enable_timing=False),
        ]

        # ---- Camera image buffer (minimally modified GPUImageBufferPacked) --
        # We drive camera scheduling ourselves; pass cameras but disable
        # internal random scheduling via explicit_cameras mode.
        self._img_buf = GPUImageBufferPacked(
            cameras,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            loop=True,
            decode_executor=decode_executor,
        )

        # ---- Scheduling state -------------------------------------------
        # anchor_in_slot[s] = anchor camera index currently loaded in slot s
        self.anchor_in_slot: List[int] = [-1, -1]
        # camera views loaded per slot (list of PackedCameraView)
        self.cam_views: List[List[PackedCameraView]] = [[], []]
        # pinned CPU slab keepalives per slot — freed when the slot is reloaded
        self.cam_keepalive: List[list] = [[], []]

        self.active = 0                             # slot index currently being trained

        self.sample_count   = np.zeros(len(cameras), dtype=np.int64)
        # Per-batch selection count — how many times each anchor has been loaded.
        # Used by _pick_anchor_global and _pick_unrelated to prefer under-sampled
        # batches, ensuring uniform coverage even when some batches have small
        # unrelated_batches lists that would otherwise be skewed toward certain anchors.
        self.batch_count    = np.zeros(len(batch_infos), dtype=np.int64)
        self.skipped_batches = 0

        # ---- Background writeback thread --------------------------------
        # One thread per in-flight writeback; joined at the next finish_batch().
        self._wb_thread: Optional[threading.Thread] = None

        # ---- Over-budget set -------------------------------------------
        self._over_budget = {
            i for i, info in batch_infos.items()
            if info.n_gaussians > max_gaussians
        }
        if self._over_budget:
            print(f"[GaussianSwapBuffer] {len(self._over_budget)} batches are "
                  f"over-budget and will be skipped at training time.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prefetch_initial(self) -> None:
        """
        Synchronously fill both slots before training begins.
        Slot 0 = lowest-sample-count anchor.
        Slot 1 = random unrelated batch to slot 0's anchor.
        """
        anchor0 = self._pick_anchor_global()
        anchor1 = self._pick_unrelated(anchor0)

        print(f"[GaussianSwapBuffer] Initial slots: anchor {anchor0}, anchor {anchor1}")
        self._load_slot_sync(0, anchor0)
        self._load_slot_sync(1, anchor1)
        self.active = 0

    def pop(self) -> Tuple[List[PackedCameraView], GPUGaussianSlice]:
        """
        Return (camera_views, gaussian_slice) for the active slot.
        Blocks until the active slot's H2D copies are complete.
        """
        s = self.active
        # Wait for the active slot's H2D to be done on the default stream
        torch.cuda.current_stream(self.device).wait_event(self.slot_ready[s])
        # Create Adam optimizer now (lazy: inactive slot has no optimizer → saves ~3.4 GB)
        self.slots[s].ensure_optimizer()
        return self.cam_views[s], self.slots[s]

    def finish_batch(self) -> None:
        """
        Call after the last optimizer.step() for the active batch.

        Pipeline (no GPU idle between slots):
          1. Record a CUDA event marking when training on slot s is done.
          2. Switch active to the other slot immediately — it was prefetched
             and is ready; the compute stream will wait only for slot_ready[1-s].
          3. Launch a background thread that:
               a. synchronizes on the training-done event (waits for GPU)
               b. D2H slot s params → CPU scatter (writeback)
               c. empty_cache
               d. async H2D next batch into slot s (_load_slot_async)
             This runs entirely while the compute stream trains on the other slot.
          4. Join the background thread from the *previous* finish_batch() call
             (two slots ago) before launching the new one, ensuring we never
             overlap two writebacks for the same slot.
        """
        s    = self.active
        info = self.batch_infos[self.anchor_in_slot[s]]

        # Join the writeback thread from the previous finish_batch() to ensure
        # slot s's prior writeback+reload is fully done before we start a new one.
        if self._wb_thread is not None:
            self._wb_thread.join()
            exc_holder = getattr(self._wb_thread, '_exc_holder', [])
            if exc_holder:
                raise RuntimeError(
                    f"Background writeback thread failed: {exc_holder[0]}"
                ) from exc_holder[0]
            self._wb_thread = None

        # Record event on the compute stream: marks when training on slot s finishes.
        train_done = torch.cuda.Event()
        train_done.record(torch.cuda.current_stream(self.device))

        # Update sample count for the just-trained batch.
        for cam_idx in info.batch_camera_ids:
            self.sample_count[cam_idx] += 1

        # Pick the next anchor for slot s (decided now, on the main thread).
        next_anchor = self._pick_unrelated(self.anchor_in_slot[s])

        # Switch active slot NOW — training on slot (1-s) can begin immediately
        # via pop(); the compute stream will only wait for slot_ready[1-s] (a
        # stream-level dependency, not a CPU stall).
        self.active = 1 - s

        # Capture references needed by the background thread.
        _s           = s
        _slot        = self.slots[s]
        _cpu_idx     = self.slots[s].cpu_indices   # numpy array, immutable reference
        _cpu_store   = self.cpu_store
        _device_idx  = torch.cuda.current_device()  # integer index, safe across threads
        _exc_holder: List[BaseException] = []

        def _writeback_and_reload() -> None:
            try:
                # Ensure CUDA context is initialised on this thread.
                torch.cuda.set_device(_device_idx)
                # Block until the compute stream has finished training on slot _s.
                train_done.synchronize()
                # D2H: copy trained params to CPU and scatter back to the store.
                params = _slot.collect_params()
                _cpu_store.scatter(_cpu_idx, params)
                del params
                torch.cuda.empty_cache()
                # Async H2D: load the next batch into the now-free slot _s.
                self._load_slot_async(_s, next_anchor)
            except Exception as e:
                _exc_holder.append(e)

        self._wb_thread = threading.Thread(target=_writeback_and_reload, daemon=True)
        self._wb_thread._exc_holder = _exc_holder
        self._wb_thread.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_min_count(self, candidates: list) -> int:
        """Among candidates, return one with the minimum batch_count (random tie-break)."""
        counts = self.batch_count[candidates]
        min_c  = counts.min()
        least  = [c for c, cnt in zip(candidates, counts) if cnt == min_c]
        return random.choice(least)

    def _pick_anchor_global(self) -> int:
        """Pick the batch anchor with the lowest batch_count (ties broken randomly)."""
        valid = [i for i in self.batch_infos if i not in self._over_budget]
        if not valid:
            raise RuntimeError("All batches are over-budget. Cannot train.")
        return self._pick_min_count(valid)

    def _pick_unrelated(self, anchor: int) -> int:
        """
        Pick the least-chosen batch from anchor's unrelated_batches that is not
        over-budget.  Falls back to any non-over-budget batch if all unrelated
        ones are over-budget.
        """
        info = self.batch_infos[anchor]
        candidates = [
            j for j in info.unrelated_batches
            if j not in self._over_budget
        ]
        if candidates:
            return self._pick_min_count(candidates)
        # Fallback: any non-over-budget batch, least-chosen first
        fallback = [i for i in self.batch_infos if i not in self._over_budget]
        return self._pick_min_count(fallback) if fallback else anchor

    def _gather_cameras(self, anchor: int) -> List[Any]:
        """Return the list of CachedCamera objects for this batch."""
        info = self.batch_infos[anchor]
        return [self.cameras[i] for i in info.batch_camera_ids]

    def _load_slot_sync(self, s: int, anchor: int) -> None:
        """Synchronously load Gaussian + camera data into slot s."""
        info     = self.batch_infos[anchor]
        cam_list = self._gather_cameras(anchor)
        stream   = torch.cuda.current_stream(self.device)

        # Free previous keepalive (if any)
        self.cam_keepalive[s] = []

        # Gaussian H2D
        indices = self.block_index.get_block_indices(info.batch_blocks)
        params = self.cpu_store.gather(indices)
        with torch.no_grad():
            self.slots[s].load_from(params, indices, self.lr_dict)

        # Camera pack + H2D
        self.cam_views[s] = self._img_buf._pack_cameras_explicit(
            cam_list, stream=stream, keepalive=self.cam_keepalive[s]
        )
        self.slot_ready[s].record(stream)
        stream.synchronize()    # wait for H2D before marking ready

        self.anchor_in_slot[s] = anchor
        self.batch_count[anchor] += 1

    def _load_slot_async(self, s: int, anchor: int) -> None:
        """
        Async prefetch: gather from CPU store, H2D on prefetch_stream.
        Records slot_ready[s] on the prefetch_stream when done.
        The default (training) stream waits for slot_ready[s] via pop().
        """
        info     = self.batch_infos[anchor]
        cam_list = self._gather_cameras(anchor)

        # Free previous keepalive now (H2D for slot s is complete because
        # finish_batch() synced the compute stream before calling us).
        self.cam_keepalive[s] = []

        # CPU gather — runs here (overlaps with GPU training on default stream)
        indices = self.block_index.get_block_indices(info.batch_blocks)
        params = self.cpu_store.gather(indices)

        with torch.cuda.stream(self.prefetch_stream):
            # Gaussian H2D (async on prefetch_stream)
            self.slots[s].load_from(params, indices, self.lr_dict)

            # Camera pack + H2D (async on prefetch_stream)
            self.cam_views[s] = self._img_buf._pack_cameras_explicit(
                cam_list, stream=self.prefetch_stream,
                keepalive=self.cam_keepalive[s]
            )

            self.slot_ready[s].record(self.prefetch_stream)

        self.anchor_in_slot[s] = anchor
        self.batch_count[anchor] += 1
        if anchor in self._over_budget:
            self.skipped_batches += 1
