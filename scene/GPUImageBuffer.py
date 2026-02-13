import collections
import random
import torch
from typing import List, Optional, Any, Dict, Tuple
from scene.GPUHelper.pachedCamera import PackedCameraView
import torch.nn.functional as F


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


def resize_semantic_to_hw(sem: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    sem: CPU tensor, shape [H,W] or [C,H,W]
    Returns: CPU tensor resized to [H,W] or [C,H,W] with same dtype.
    - integer / bool -> nearest
    - float -> bilinear
    """
    if sem.ndim == 2:
        # [H,W] -> [1,1,H,W] for interpolate
        sem_in = sem[None, None]
        is_discrete = (not sem.is_floating_point()) or (sem.dtype == torch.bool)
        mode = "nearest" if is_discrete else "bilinear"
        out = F.interpolate(
            sem_in.float() if mode != "nearest" else sem_in,
            size=(H, W),
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )
        out = out[0, 0]
        if sem.dtype != out.dtype:
            # restore dtype for discrete maps
            out = out.to(sem.dtype) if is_discrete else out.to(sem.dtype)
        return out

    if sem.ndim == 3:
        # [C,H,W] -> [1,C,H,W]
        sem_in = sem[None]
        is_discrete = (not sem.is_floating_point()) or (sem.dtype == torch.bool)
        mode = "nearest" if is_discrete else "bilinear"
        out = F.interpolate(
            sem_in.float() if mode != "nearest" else sem_in,
            size=(H, W),
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )
        out = out[0]
        if sem.dtype != out.dtype:
            out = out.to(sem.dtype) if is_discrete else out.to(sem.dtype)
        return out

    raise ValueError(f"Unsupported semantic ndim={sem.ndim}")

class GPUImageBufferPacked:
    """
    Double-buffer (2 slots) + packed batch copy for images/masks/semantic.

    pop() returns one camera view; image/mask/semantic are CUDA views into
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
        semantic_gpu_dtype: Optional[torch.dtype] = None,  # if None, keep semantic dtype
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.loop = loop
        self.shuffle = shuffle

        self.img_gpu_dtype = img_gpu_dtype
        self.mask_gpu_dtype = mask_gpu_dtype
        self.semantic_gpu_dtype = semantic_gpu_dtype

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

        H, W = self._infer_hw(cams)
        if not self._maybe_same_hw(cams, H, W):
            raise ValueError("Mixed resolutions in one packed batch. Use bucketing by (H,W) first.")

        B = len(cams)

        # ---- Decide semantic shape (optional) ----
        sem0 = getattr(cams[0], "semantic", None)
        have_sem = sem0 is not None

        sem_shape = None
        sem_dtype = None
        if have_sem:
            if not torch.is_tensor(sem0):
                sem0 = torch.as_tensor(sem0)
            # support [H,W] or [C,H,W]
            if sem0.ndim == 2:
                sem_shape = (B, H, W)
            elif sem0.ndim == 3:
                Csem = sem0.shape[0]
                sem_shape = (B, Csem, H, W)
            else:
                raise ValueError(f"Unsupported semantic ndim={sem0.ndim}")
            sem_dtype = sem0.dtype

        # ---- Allocate pinned CPU slabs ----
        # Store CPU as uint8 if we want IO-efficient transfer
        img_cpu = torch.empty((B, 3, H, W), dtype=torch.uint8, pin_memory=True)
        msk_cpu = torch.empty((B, 1, H, W), dtype=torch.uint8, pin_memory=True)

        sem_cpu = None
        if have_sem:
            sem_cpu = torch.empty(sem_shape, dtype=sem_dtype, pin_memory=True)

        # Optional: pack small matrices too (one copy)
        wv_cpu = torch.empty((B, 4, 4), dtype=torch.float32, pin_memory=True)
        pj_cpu = torch.empty((B, 4, 4), dtype=torch.float32, pin_memory=True)
        fp_cpu = torch.empty((B, 4, 4), dtype=torch.float32, pin_memory=True)
        cc_cpu = torch.empty((B, 3), dtype=torch.float32, pin_memory=True)

        # (Optional) depth slabs
        have_inv = getattr(cams[0], "invdepthmap", None) is not None
        inv_cpu = None
        dm_cpu = None
        if have_inv:
            inv0 = cams[0].invdepthmap
            if not (torch.is_tensor(inv0) and inv0.ndim == 3 and inv0.shape[0] == 1):
                # expected [1,H,W]
                raise ValueError("Expected invdepthmap shape [1,H,W] tensor on CPU.")
            inv_cpu = torch.empty((B, 1, H, W), dtype=inv0.dtype, pin_memory=True)
            dm0 = getattr(cams[0], "depth_mask", None)
            if dm0 is not None:
                dm_cpu = torch.empty((B, 1, H, W), dtype=dm0.dtype, pin_memory=True)

        # Keep pinned slabs alive until GPU finished copies
        slot.pinned_keepalive.extend([img_cpu, msk_cpu, wv_cpu, pj_cpu, fp_cpu, cc_cpu])
        if sem_cpu is not None:
            slot.pinned_keepalive.append(sem_cpu)
        if inv_cpu is not None:
            slot.pinned_keepalive.append(inv_cpu)
        if dm_cpu is not None:
            slot.pinned_keepalive.append(dm_cpu)

        # ---- Fill slabs (CPU copy into pinned memory) ----
        for i, cam in enumerate(cams):
            img_i = self._ensure_uint8_image_cpu(cam.original_image.cpu())
            msk_i = self._ensure_uint8_mask_cpu(cam.alpha_mask.cpu())

            img_cpu[i].copy_(img_i, non_blocking=False)
            msk_cpu[i].copy_(msk_i, non_blocking=False)

            # semantic (resize to image H,W if needed)
            if have_sem:
                sem_i = getattr(cam, "semantic", None)
                if sem_i is None:
                    raise ValueError("Semantic missing in a batch that expects semantic.")
                if not torch.is_tensor(sem_i):
                    sem_i = torch.as_tensor(sem_i)
            
                sem_i = sem_i.cpu()
            
                # sem_i can be [H,W] or [C,H,W]; resize if mismatch
                if sem_i.ndim == 2:
                    sh, sw = int(sem_i.shape[0]), int(sem_i.shape[1])
                elif sem_i.ndim == 3:
                    sh, sw = int(sem_i.shape[-2]), int(sem_i.shape[-1])
                else:
                    raise ValueError(f"Unsupported semantic ndim={sem_i.ndim}")
            
                if (sh != H) or (sw != W):
                    sem_i = resize_semantic_to_hw(sem_i, H, W)
            
                # Now shapes should match sem_cpu[i]
                sem_cpu[i].copy_(sem_i, non_blocking=False)

            # matrices
            wv_cpu[i].copy_(cam.world_view_transform.cpu().to(torch.float32))
            pj_cpu[i].copy_(cam.projection_matrix.cpu().to(torch.float32))
            fp_cpu[i].copy_(cam.full_proj_transform.cpu().to(torch.float32))
            cc_cpu[i].copy_(cam.camera_center.cpu().to(torch.float32))

            if have_inv:
                inv_cpu[i].copy_(cam.invdepthmap.cpu())
                if dm_cpu is not None and getattr(cam, "depth_mask", None) is not None:
                    dm_cpu[i].copy_(cam.depth_mask.cpu())

        # ---- One big async H2D copy per slab (prefetch stream) ----
        with torch.cuda.stream(self.prefetch_stream):
            img_gpu = img_cpu.to(self.device, non_blocking=True)
            msk_gpu = msk_cpu.to(self.device, non_blocking=True)

            # cast on GPU if you want (still one slab)
            if self.img_gpu_dtype != img_gpu.dtype:
                img_gpu = img_gpu.to(self.img_gpu_dtype)
            if self.mask_gpu_dtype != msk_gpu.dtype:
                msk_gpu = msk_gpu.to(self.mask_gpu_dtype)

            sem_gpu = None
            if have_sem:
                sem_gpu = sem_cpu.to(self.device, non_blocking=True)
                if self.semantic_gpu_dtype is not None and sem_gpu.dtype != self.semantic_gpu_dtype:
                    sem_gpu = sem_gpu.to(self.semantic_gpu_dtype)

            wv_gpu = wv_cpu.to(self.device, non_blocking=True)
            pj_gpu = pj_cpu.to(self.device, non_blocking=True)
            fp_gpu = fp_cpu.to(self.device, non_blocking=True)
            cc_gpu = cc_cpu.to(self.device, non_blocking=True)

            inv_gpu = None
            dm_gpu = None
            if have_inv:
                inv_gpu = inv_cpu.to(self.device, non_blocking=True)
                if dm_cpu is not None:
                    dm_gpu = dm_cpu.to(self.device, non_blocking=True)

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
                depth_reliable=getattr(cam, "depth_reliable", False),

                original_image=img_gpu[i],   # view
                alpha_mask=msk_gpu[i],       # view
                semantic=None if sem_gpu is None else sem_gpu[i],

                world_view_transform=wv_gpu[i],
                projection_matrix=pj_gpu[i],
                full_proj_transform=fp_gpu[i],
                camera_center=cc_gpu[i],

                invdepthmap=None if inv_gpu is None else inv_gpu[i],
                depth_mask=None if dm_gpu is None else dm_gpu[i],
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
