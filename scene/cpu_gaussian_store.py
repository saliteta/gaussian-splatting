import numpy as np
import torch
from typing import Dict


class CPUGaussianStore:
    """
    All Gaussian parameters in CPU pinned memory.

    SH degree 0 only: stores xyz, features_dc, scaling, rotation, opacity.
    No features_rest, no optimizer states.

    Layout: after optional reorder(), Gaussian i is at row i in the
    permuted order. Writeback scatter uses the same indices as the gather.
    """

    FIELDS = ('_xyz', '_features_dc', '_scaling', '_rotation', '_opacity')

    def __init__(self, gaussian_model):
        """
        Copy all parameters from an existing GaussianModel into CPU pinned memory.
        After construction the GaussianModel can be discarded — its GPU tensors are
        no longer needed.
        """
        self.N = gaussian_model._xyz.shape[0]

        def _pin(t: torch.Tensor) -> torch.Tensor:
            return t.detach().cpu().contiguous().pin_memory()

        self._xyz         = _pin(gaussian_model._xyz)           # (N, 3)
        self._features_dc = _pin(gaussian_model._features_dc)   # (N, 1, 3)
        self._scaling     = _pin(gaussian_model._scaling)        # (N, 3)
        self._rotation    = _pin(gaussian_model._rotation)       # (N, 4)
        self._opacity     = _pin(gaussian_model._opacity)        # (N, 1)

        mb = sum(
            getattr(self, f).nbytes for f in self.FIELDS
        ) / 1024 ** 2
        print(f"[CPUGaussianStore] {self.N:,} Gaussians, {mb:.0f} MB pinned CPU memory.")

    def reorder(self, perm: np.ndarray) -> None:
        """
        Permute all fields in-place according to `perm` (int64 array of length N).
        Must be called before any gather()/scatter() so indices stay consistent.
        """
        idx = torch.from_numpy(perm).long()
        for field in self.FIELDS:
            t = getattr(self, field)
            setattr(self, field, t[idx].contiguous().pin_memory())

    @property
    def xyz_cpu(self) -> torch.Tensor:
        """Full (N, 3) point cloud on CPU — for visibility precomputation."""
        return self._xyz

    def gather(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Gather rows at global `indices` into new pinned CPU tensors.
        Returns dict field → pinned Tensor of shape (K, ...).
        K = len(indices).
        """
        idx = torch.from_numpy(indices).long()
        return {
            '_xyz':         self._xyz[idx].pin_memory(),
            '_features_dc': self._features_dc[idx].pin_memory(),
            '_scaling':     self._scaling[idx].pin_memory(),
            '_rotation':    self._rotation[idx].pin_memory(),
            '_opacity':     self._opacity[idx].pin_memory(),
        }

    def scatter(self, indices: np.ndarray, params: Dict[str, torch.Tensor]) -> None:
        """
        Scatter updated CPU tensors back into the store at global `indices`.
        params: dict field → CPU Tensor (output of GPUGaussianSlice.collect_params).
        """
        idx = torch.from_numpy(indices).long()
        with torch.no_grad():
            self._xyz[idx]         = params['_xyz'].cpu()
            self._features_dc[idx] = params['_features_dc'].cpu()
            self._scaling[idx]     = params['_scaling'].cpu()
            self._rotation[idx]    = params['_rotation'].cpu()
            self._opacity[idx]     = params['_opacity'].cpu()
