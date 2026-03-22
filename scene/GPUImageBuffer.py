import random
import torch
from typing import List, Any, Tuple
from scene.GPUHelper.pachedCamera import PackedCameraView


class _PackedSlot:
    __slots__ = ("views", "pos", "ready", "pinned_keepalive")
    def __init__(self):
        self.views: List[PackedCameraView] = []
        self.pos: int = 0
        self.ready = torch.cuda.Event(enable_timing=False)
        self.pinned_keepalive: List[torch.Tensor] = []  # keep pinned slabs alive

    def reset(self):
        self.views.clear()
        self.pos = 0
        self.pinned_keepalive.clear()

class GPUImageBufferPacked:
    """
    Double-buffer (2 slots) + packed batch copy for images and masks.

    pop() returns one camera view; image/mask are CUDA views into
    packed slabs.
    """
    def __init__(
        self,
        cameras_cpu: List[Any],
        *,
        device: str = "cuda",
        batch_size: int = 8,
        shuffle: bool = True,
        loop: bool = True,
        # choose your GPU storage dtype for images:
        # - torch.uint8 is smallest (best for IO), convert later in compute
        # - torch.float16 saves vs float32
        img_gpu_dtype: torch.dtype = torch.uint8,
        mask_gpu_dtype: torch.dtype = torch.uint8,
        decode_executor = None,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.loop = loop
        self.shuffle = shuffle

        self.img_gpu_dtype = img_gpu_dtype
        self.mask_gpu_dtype = mask_gpu_dtype
        self.decode_executor = decode_executor

        self.cameras = cameras_cpu
        self.order = list(range(len(cameras_cpu)))
        if shuffle:
            random.shuffle(self.order)
        self._next = 0

        self.prefetch_stream = torch.cuda.Stream(device=self.device)

        self.slots = [_PackedSlot(), _PackedSlot()]
        self.active = 0

        # prefill both slots
        self._launch_fill(self.slots[self.active])
        self._launch_fill(self.slots[1 - self.active])

    def _next_batch(self) -> List[Any]:
        out = []
        n = len(self.order)
        if n == 0:
            return out
        while len(out) < self.batch_size:
            if self._next >= n:
                if not self.loop:
                    break
                self._next = 0
                if self.shuffle:
                    random.shuffle(self.order)
            out.append(self.cameras[self.order[self._next]])
            self._next += 1
        return out

    def _infer_hw(self, cams: List[Any]) -> Tuple[int,int]:
        # assumes fixed resolution in dataset
        h = int(cams[0].image_height)
        w = int(cams[0].image_width)
        return h, w

    def _maybe_same_hw(self, cams: List[Any], h: int, w: int) -> bool:
        for c in cams:
            if int(c.image_height) != h or int(c.image_width) != w:
                return False
        return True

    def _ensure_uint8_image_cpu(self, img: torch.Tensor) -> torch.Tensor:
        """
        Convert CPU image to uint8 [0..255] for transfer efficiency.
        Expect img shape [3,H,W] float [0..1] or uint8 already.
        """
        if img.dtype == torch.uint8:
            return img
        # assume float 0..1
        return (img.clamp(0, 1) * 255.0).to(torch.uint8)

    def _ensure_uint8_mask_cpu(self, m: torch.Tensor) -> torch.Tensor:
        if m.dtype == torch.uint8:
            return m
        # if mask is float 0/1, convert
        return (m.clamp(0, 1) * 255.0).to(torch.uint8)

    def _pack_and_copy(self, cams: List[Any], slot: _PackedSlot):
        """
        Pack batch into pinned slabs, then one H2D copy per slab.
        Build PackedCameraView list indexing into the CUDA slabs.
        """
        slot.reset()
        if len(cams) == 0:
            return

        # If cameras support lazy decoding, optionally pre-decode in parallel.
        if self.decode_executor is not None:
            futs = []
            for c in cams:
                if hasattr(c, "ensure_decoded"):
                    futs.append(self.decode_executor.submit(c.ensure_decoded))
            for f in futs:
                f.result()
        else:
            for c in cams:
                if hasattr(c, "ensure_decoded"):
                    c.ensure_decoded()

        H, W = self._infer_hw(cams)
        if not self._maybe_same_hw(cams, H, W):
            raise ValueError("Mixed resolutions in one packed batch. Use bucketing by (H,W) first.")

        B = len(cams)

        # ---- Allocate pinned CPU slabs ----
        # Store CPU as uint8 if we want IO-efficient transfer
        img_cpu = torch.empty((B, 3, H, W), dtype=torch.uint8, pin_memory=True)
        msk_cpu = torch.empty((B, 1, H, W), dtype=torch.uint8, pin_memory=True)

        # Optional: pack small matrices too (one copy)
        wv_cpu = torch.empty((B, 4, 4), dtype=torch.float32, pin_memory=True)
        pj_cpu = torch.empty((B, 4, 4), dtype=torch.float32, pin_memory=True)
        fp_cpu = torch.empty((B, 4, 4), dtype=torch.float32, pin_memory=True)
        cc_cpu = torch.empty((B, 3), dtype=torch.float32, pin_memory=True)

        # Keep pinned slabs alive until GPU finished copies
        slot.pinned_keepalive.extend([img_cpu, msk_cpu, wv_cpu, pj_cpu, fp_cpu, cc_cpu])

        # ---- Fill slabs (CPU copy into pinned memory) ----
        for i, cam in enumerate(cams):
            img_i = self._ensure_uint8_image_cpu(cam.original_image.cpu())
            msk_i = self._ensure_uint8_mask_cpu(cam.alpha_mask.cpu())

            img_cpu[i].copy_(img_i, non_blocking=False)
            msk_cpu[i].copy_(msk_i, non_blocking=False)

            # matrices
            wv_cpu[i].copy_(cam.world_view_transform.cpu().to(torch.float32))
            pj_cpu[i].copy_(cam.projection_matrix.cpu().to(torch.float32))
            fp_cpu[i].copy_(cam.full_proj_transform.cpu().to(torch.float32))
            cc_cpu[i].copy_(cam.camera_center.cpu().to(torch.float32))

            # Free float32 decoded tensors — data is now in the pinned slab.
            if hasattr(cam, "free_decoded"):
                cam.free_decoded()

        # ---- One big async H2D copy per slab (prefetch stream) ----
        with torch.cuda.stream(self.prefetch_stream):
            img_gpu = img_cpu.to(self.device, non_blocking=True)
            msk_gpu = msk_cpu.to(self.device, non_blocking=True)

            # cast on GPU if you want (still one slab)
            if self.img_gpu_dtype != img_gpu.dtype:
                img_gpu = img_gpu.to(self.img_gpu_dtype)
            if self.mask_gpu_dtype != msk_gpu.dtype:
                msk_gpu = msk_gpu.to(self.mask_gpu_dtype)

            wv_gpu = wv_cpu.to(self.device, non_blocking=True)
            pj_gpu = pj_cpu.to(self.device, non_blocking=True)
            fp_gpu = fp_cpu.to(self.device, non_blocking=True)
            cc_gpu = cc_cpu.to(self.device, non_blocking=True)

            slot.ready.record(self.prefetch_stream)

        # ---- Build per-camera views (no extra GPU copy) ----
        views: List[PackedCameraView] = []
        for i, cam in enumerate(cams):
            views.append(PackedCameraView(
                uid=cam.uid,
                colmap_id=cam.colmap_id,
                image_name=cam.image_name,
                FoVx=cam.FoVx,
                FoVy=cam.FoVy,
                znear=cam.znear,
                zfar=cam.zfar,
                image_width=cam.image_width,
                image_height=cam.image_height,

                original_image=img_gpu[i],   # view
                alpha_mask=msk_gpu[i],       # view

                world_view_transform=wv_gpu[i],
                projection_matrix=pj_gpu[i],
                full_proj_transform=fp_gpu[i],
                camera_center=cc_gpu[i],
                cpu_camera=cam,
            ))
        slot.views = views
        slot.pos = 0

    def _launch_fill(self, slot: _PackedSlot):
        cams = self._next_batch()
        if len(cams) == 0:
            slot.reset()
            return
        self._pack_and_copy(cams, slot)

    # ------------------------------------------------------------------
    # Stage-2 extension: explicit camera list, caller-supplied stream
    # ------------------------------------------------------------------

    def _pack_cameras_explicit(
        self,
        cams: List[Any],
        stream: torch.cuda.Stream,
        keepalive: list,
    ) -> List["PackedCameraView"]:
        """
        Pack and H2D transfer a caller-chosen list of cameras on `stream`.
        Returns the list of PackedCameraView objects (GPU tensors live in slabs
        kept alive by the returned views via pinned_keepalive on a temp slot).

        Used by GaussianSwapBuffer to drive KNN-based camera scheduling while
        reusing all existing H2D packing logic unchanged.
        """
        tmp = _PackedSlot()

        # Decode images (CPU side, same logic as _pack_and_copy)
        if self.decode_executor is not None:
            futs = [
                self.decode_executor.submit(c.ensure_decoded)
                for c in cams if hasattr(c, "ensure_decoded")
            ]
            for f in futs:
                f.result()
        else:
            for c in cams:
                if hasattr(c, "ensure_decoded"):
                    c.ensure_decoded()

        if not cams:
            return []

        H, W = self._infer_hw(cams)
        B    = len(cams)

        img_cpu = torch.empty((B, 3, H, W), dtype=torch.uint8, pin_memory=True)
        msk_cpu = torch.empty((B, 1, H, W), dtype=torch.uint8, pin_memory=True)
        wv_cpu  = torch.empty((B, 4, 4),    dtype=torch.float32, pin_memory=True)
        pj_cpu  = torch.empty((B, 4, 4),    dtype=torch.float32, pin_memory=True)
        fp_cpu  = torch.empty((B, 4, 4),    dtype=torch.float32, pin_memory=True)
        cc_cpu  = torch.empty((B, 3),       dtype=torch.float32, pin_memory=True)

        for i, cam in enumerate(cams):
            img_cpu[i].copy_(self._ensure_uint8_image_cpu(cam.original_image.cpu()), non_blocking=False)
            msk_cpu[i].copy_(self._ensure_uint8_mask_cpu(cam.alpha_mask.cpu()),      non_blocking=False)
            wv_cpu[i].copy_(cam.world_view_transform.cpu().to(torch.float32))
            pj_cpu[i].copy_(cam.projection_matrix.cpu().to(torch.float32))
            fp_cpu[i].copy_(cam.full_proj_transform.cpu().to(torch.float32))
            cc_cpu[i].copy_(cam.camera_center.cpu().to(torch.float32))
            # Free float32 decoded tensors — data is now in the pinned slab.
            if hasattr(cam, "free_decoded"):
                cam.free_decoded()

        with torch.cuda.stream(stream):
            img_gpu = img_cpu.to(self.device, non_blocking=True)
            msk_gpu = msk_cpu.to(self.device, non_blocking=True)
            if self.img_gpu_dtype != img_gpu.dtype:
                img_gpu = img_gpu.to(self.img_gpu_dtype)
            if self.mask_gpu_dtype != msk_gpu.dtype:
                msk_gpu = msk_gpu.to(self.mask_gpu_dtype)
            wv_gpu = wv_cpu.to(self.device, non_blocking=True)
            pj_gpu = pj_cpu.to(self.device, non_blocking=True)
            fp_gpu = fp_cpu.to(self.device, non_blocking=True)
            cc_gpu = cc_cpu.to(self.device, non_blocking=True)

        # Keep pinned slabs alive until H2D completes (caller owns keepalive list).
        keepalive.extend([img_cpu, msk_cpu, wv_cpu, pj_cpu, fp_cpu, cc_cpu])

        views: List[PackedCameraView] = []
        for i, cam in enumerate(cams):
            views.append(PackedCameraView(
                uid=cam.uid,
                colmap_id=cam.colmap_id,
                image_name=cam.image_name,
                FoVx=cam.FoVx,
                FoVy=cam.FoVy,
                znear=cam.znear,
                zfar=cam.zfar,
                image_width=cam.image_width,
                image_height=cam.image_height,
                original_image=img_gpu[i],
                alpha_mask=msk_gpu[i],
                world_view_transform=wv_gpu[i],
                projection_matrix=pj_gpu[i],
                full_proj_transform=fp_gpu[i],
                camera_center=cc_gpu[i],
                cpu_camera=cam,
            ))

        # Attach keepalive to the list so caller keeps slabs alive
        return views

    def pop(self) -> PackedCameraView:
        slot = self.slots[self.active]

        # empty -> refill
        if slot.pos >= len(slot.views):
            self._launch_fill(slot)
            if len(slot.views) == 0:
                raise StopIteration

        # first consume from this slot: wait for its async copies
        if slot.pos == 0:
            torch.cuda.current_stream(device=self.device).wait_event(slot.ready)

            # ensure other slot is filled while we compute on this one
            other = self.slots[1 - self.active]
            if len(other.views) == 0 or other.pos >= len(other.views):
                self._launch_fill(other)

        out = slot.views[slot.pos]
        slot.pos += 1

        # if slot exhausted, swap buffers
        if slot.pos >= len(slot.views):
            self.active = 1 - self.active

        return out

    def __iter__(self):
        return self

    def __next__(self):
        return self.pop()
