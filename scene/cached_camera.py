import io
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from PIL import Image

from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch


@dataclass(frozen=True)
class CachedImageBlob:
    """
    Immutable, read-only image payload kept in RAM as compressed bytes.
    Safe to share across threads.
    """
    image_path: str
    image_bytes: bytes
    orig_size: Tuple[int, int]  # (W, H)


class CachedCamera(torch.nn.Module):
    """
    Camera that keeps the compressed image in RAM and decodes lazily per requested resolution.

    - Stores compressed bytes (JPEG/PNG/etc) in memory.
    - Computes target (W,H) without decoding.
    - Decodes -> resize -> torch only when original_image/alpha_mask is first accessed.
    - Keeps a small per-instance cache for the decoded tensors (keyed by resolution).
    """

    def __init__(
        self,
        *,
        uid: int,
        colmap_id: int,
        R: np.ndarray,
        T: np.ndarray,
        FoVx: float,
        FoVy: float,
        image_name: str,
        blob: CachedImageBlob,
        resolution: Tuple[int, int],  # (W, H)
        data_device: str = "cuda",
        train_test_exp: bool = False,
        is_test_dataset: bool = False,
        is_test_view: bool = False,
        trans: np.ndarray = np.array([0.0, 0.0, 0.0]),
        scale: float = 1.0,
    ):
        super().__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception:
            self.data_device = torch.device("cuda")

        self._blob = blob
        self._resolution = (int(resolution[0]), int(resolution[1]))  # (W,H)

        # Exposed dimensions without decoding.
        self.image_width = int(self._resolution[0])
        self.image_height = int(self._resolution[1])

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        # Keep transforms on target device; images stay lazily decoded on CPU first.
        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        )
        self.projection_matrix = (
            getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy)
            .transpose(0, 1)
            .to(self.data_device)
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self._train_test_exp = train_test_exp
        self._is_test_dataset = is_test_dataset
        self._is_test_view = is_test_view

        # Decoded cache: (W,H) -> (original_image_cpu_float01, alpha_mask_cpu_float01)
        self._decoded: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    def ensure_decoded(self) -> None:
        if self._resolution in self._decoded:
            return

        # Decode from in-RAM compressed bytes.
        with Image.open(io.BytesIO(self._blob.image_bytes)) as im:
            im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")

            resized = PILtoTorch(im, self._resolution)  # [C,H,W], float32 in [0..1]
            rgb = resized[:3, ...].clamp(0.0, 1.0).contiguous()
            if resized.shape[0] == 4:
                alpha = resized[3:4, ...].clamp(0.0, 1.0).contiguous()
            else:
                alpha = torch.ones((1, rgb.shape[1], rgb.shape[2]), dtype=rgb.dtype)

        if self._train_test_exp and self._is_test_view:
            if self._is_test_dataset:
                alpha[..., : alpha.shape[-1] // 2] = 0
            else:
                alpha[..., alpha.shape[-1] // 2 :] = 0

        self._decoded[self._resolution] = (rgb, alpha)

    def free_decoded(self) -> None:
        """
        Drop all decoded image tensors from the in-memory cache.
        Call this after the image data has been copied into a pinned slab
        (and enqueued for H2D) so the float32 CPU copies don't accumulate.
        """
        self._decoded.clear()

    @property
    def original_image(self) -> torch.Tensor:
        self.ensure_decoded()
        return self._decoded[self._resolution][0]

    @property
    def alpha_mask(self) -> torch.Tensor:
        self.ensure_decoded()
        return self._decoded[self._resolution][1]

