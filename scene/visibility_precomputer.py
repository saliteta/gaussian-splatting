import math
import numpy as np
import torch
from typing import List, Any


class VisibilityPrecomputer:
    """
    GPU-accelerated per-camera visibility on a downsampled point cloud.

    For each camera, projects all M downsampled points through the camera frustum
    (with an optional FOV margin) and produces a (M,) bool visibility mask.
    All masks are collected into visible_ds: np.ndarray of shape (N, M), dtype bool,
    stored on CPU. One row per camera.

    world_view_transform convention: W2C^T (column-major).
    Projection: p_cam = points_h @ world_view_transform  (row-vector form).
    p_cam[:, 2] = depth in camera space.
    """

    def __init__(
        self,
        cameras: List[Any],
        points_ds: torch.Tensor,    # (M, 3) float32, CPU or GPU
        fov_margin: float = 0.1,
        device: str = "cuda",
    ):
        self.cameras = cameras
        self.fov_margin = fov_margin
        self.device = torch.device(device)
        self.N = len(cameras)

        # Build (M, 4) homogeneous coords on GPU — stays resident throughout compute()
        pts = points_ds.to(self.device, dtype=torch.float32)
        M = pts.shape[0]
        self.M = M
        ones = torch.ones(M, 1, device=self.device, dtype=torch.float32)
        self.points_h = torch.cat([pts, ones], dim=1)   # (M, 4)

    def compute(self) -> np.ndarray:
        """
        Returns visible_ds: np.ndarray of shape (N, M), dtype bool.
        Each row i is True where downsampled Gaussian j is visible from camera i.
        """
        visible_ds = np.zeros((self.N, self.M), dtype=bool)
        for i, cam in enumerate(self.cameras):
            visible_ds[i] = self._project_one(cam)
            if (i + 1) % 100 == 0 or i == self.N - 1:
                print(f"  [VisibilityPrecomputer] {i + 1}/{self.N} cameras done")
        return visible_ds

    @torch.no_grad()
    def _project_one(self, cam) -> np.ndarray:
        """
        Project all M downsampled points through `cam`.
        Returns (M,) bool np.ndarray: True = visible.

        world_view_transform is W2C^T, so:
            p_cam = points_h @ world_view_transform   (M,4) @ (4,4) -> (M,4)
        p_cam[:,2] is z in camera space (depth).
        """
        w2c_t = cam.world_view_transform.to(self.device, dtype=torch.float32)   # (4,4)
        tanfovx = math.tan(cam.FoVx * 0.5)
        tanfovy = math.tan(cam.FoVy * 0.5)
        znear   = cam.znear
        margin  = 1.0 + self.fov_margin

        p_cam = self.points_h @ w2c_t           # (M, 4)
        z     = p_cam[:, 2]                     # depth in camera space

        valid  = z > znear
        eps    = 1e-8
        ndc_x  = p_cam[:, 0] / (z.clamp(min=eps) * tanfovx)
        ndc_y  = p_cam[:, 1] / (z.clamp(min=eps) * tanfovy)
        in_fov = (ndc_x.abs() <= margin) & (ndc_y.abs() <= margin)

        mask = (valid & in_fov).cpu().numpy()
        return mask

    def free(self):
        """Release the GPU buffer when no longer needed."""
        del self.points_h
        torch.cuda.empty_cache()
