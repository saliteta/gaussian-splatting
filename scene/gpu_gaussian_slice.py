import numpy as np
import torch
from torch import nn
from typing import Dict, Optional

from utils.general_utils import strip_symmetric, build_scaling_rotation


class GPUGaussianSlice(nn.Module):
    """
    Pre-allocated GPU buffers for a Gaussian batch. SH degree 0 only.

    Exposes the same interface as GaussianModel so the renderer sees no difference.

    Design:
    - Two fixed-size GPU storage buffers (not parameters) are allocated at init.
    - On each load_from(), parameters are created as nn.Parameter VIEWS of the
      first [:K] rows of those buffers. No new GPU memory is allocated on swap.
    - A fresh SGD optimizer is built over the [:K] views each load.
    - valid_length and cpu_indices are updated in-place.
    """

    active_sh_degree: int = 0
    max_sh_degree:    int = 0

    def __init__(self, max_gaussians: int, device: str = "cuda"):
        super().__init__()
        self.device        = torch.device(device)
        self.max_gaussians = max_gaussians

        # Pre-allocated GPU storage — NOT nn.Parameters; views become Parameters.
        self._xyz_buf         = torch.empty(max_gaussians, 3,    device=self.device)
        self._features_dc_buf = torch.empty(max_gaussians, 1, 3, device=self.device)
        self._scaling_buf     = torch.empty(max_gaussians, 3,    device=self.device)
        self._rotation_buf    = torch.empty(max_gaussians, 4,    device=self.device)
        self._opacity_buf     = torch.empty(max_gaussians, 1,    device=self.device)

        self.valid_length: int              = 0
        self.cpu_indices:  np.ndarray       = np.empty(0, dtype=np.int64)
        self.optimizer:    Optional[torch.optim.Optimizer] = None

        # Parameter attributes; set properly by load_from()
        self._xyz         = nn.Parameter(self._xyz_buf[:0])
        self._features_dc = nn.Parameter(self._features_dc_buf[:0])
        self._scaling     = nn.Parameter(self._scaling_buf[:0])
        self._rotation    = nn.Parameter(self._rotation_buf[:0])
        self._opacity     = nn.Parameter(self._opacity_buf[:0])

    # ------------------------------------------------------------------
    # GaussianModel-compatible properties
    # ------------------------------------------------------------------
    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz                                # (K, 3), no activation

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)                # (K, 3)

    @property
    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotation, dim=-1)   # (K, 4)

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)            # (K, 1)

    @property
    def get_features(self) -> torch.Tensor:
        # SH degree 0: features_dc only — shape (K, 1, 3)
        return self._features_dc

    @property
    def get_features_dc(self) -> torch.Tensor:
        return self._features_dc                       # (K, 1, 3)

    @property
    def get_features_rest(self) -> torch.Tensor:
        # SH degree 0: empty features_rest
        return self._features_dc[:0]                   # (0, 1, 3)

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        L   = build_scaling_rotation(scaling_modifier * self.get_scaling, self._rotation)
        cov = L @ L.transpose(1, 2)
        return strip_symmetric(cov)

    # ------------------------------------------------------------------
    # Load / writeback
    # ------------------------------------------------------------------
    @torch.no_grad()
    def load_from(
        self,
        params: Dict[str, torch.Tensor],
        indices: np.ndarray,
        lr_dict: Dict[str, float],
    ) -> None:
        """
        In-place load of a new Gaussian batch from CPU pinned tensors.

        params:   dict from CPUGaussianStore.gather() — CPU pinned tensors
        indices:  global Gaussian indices for writeback scatter
        lr_dict:  {param_name → initial lr} for the fresh SGD optimizer
                  keys: 'xyz', 'f_dc', 'scaling', 'rotation', 'opacity'

        H2D copies run on the CURRENT stream (caller is responsible for
        synchronisation via CUDA events before using the data in a forward pass).
        """
        K = params['_xyz'].shape[0]
        assert K <= self.max_gaussians, (
            f"Batch size {K} exceeds pre-allocated max {self.max_gaussians}"
        )

        self.valid_length = K
        self.cpu_indices  = indices

        # In-place H2D into pre-allocated buffers (non_blocking: caller syncs)
        self._xyz_buf[:K].copy_(params['_xyz'],         non_blocking=True)
        self._features_dc_buf[:K].copy_(params['_features_dc'], non_blocking=True)
        self._scaling_buf[:K].copy_(params['_scaling'], non_blocking=True)
        self._rotation_buf[:K].copy_(params['_rotation'], non_blocking=True)
        self._opacity_buf[:K].copy_(params['_opacity'], non_blocking=True)

        # Create parameter VIEWS of exactly [:K] rows — no new GPU allocation
        self._xyz         = nn.Parameter(self._xyz_buf[:K])
        self._features_dc = nn.Parameter(self._features_dc_buf[:K])
        self._scaling     = nn.Parameter(self._scaling_buf[:K])
        self._rotation    = nn.Parameter(self._rotation_buf[:K])
        self._opacity     = nn.Parameter(self._opacity_buf[:K])

        # Fresh SGD optimizer — no state carried over from previous batch
        self.optimizer = torch.optim.SGD([
            {'params': [self._xyz],         'lr': lr_dict.get('xyz',      1.6e-4), 'name': 'xyz'},
            {'params': [self._features_dc], 'lr': lr_dict.get('f_dc',     2.5e-3), 'name': 'f_dc'},
            {'params': [self._scaling],     'lr': lr_dict.get('scaling',  5.0e-3), 'name': 'scaling'},
            {'params': [self._rotation],    'lr': lr_dict.get('rotation', 1.0e-3), 'name': 'rotation'},
            {'params': [self._opacity],     'lr': lr_dict.get('opacity',  5.0e-2), 'name': 'opacity'},
        ])

    @torch.no_grad()
    def collect_params(self) -> Dict[str, torch.Tensor]:
        """
        Copy current GPU params ([:valid_length]) to CPU tensors for writeback.
        Caller must ensure GPU training is complete before calling this
        (e.g., via cuda Event.synchronize()).
        """
        K = self.valid_length
        return {
            '_xyz':         self._xyz_buf[:K].cpu(),
            '_features_dc': self._features_dc_buf[:K].cpu(),
            '_scaling':     self._scaling_buf[:K].cpu(),
            '_rotation':    self._rotation_buf[:K].cpu(),
            '_opacity':     self._opacity_buf[:K].cpu(),
        }
