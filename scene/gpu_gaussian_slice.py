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

        # Level 2 memory control: reusable screenspace buffers for render().
        # render() normally calls torch.zeros_like(xyz) + 0 each forward pass,
        # allocating and freeing K×3×4 bytes (~460 MB at 38 M Gaussians) 32×
        # per batch — a prime fragmentation source.  These permanent buffers are
        # sliced [:K] and detached each call, so no new CUDA memory is ever
        # requested.  _sp_grad_buf is pre-wired as screenspace_points.grad to
        # prevent PyTorch allocating a separate grad tensor during backward.
        self._sp_buf      = torch.zeros(max_gaussians, 3, device=self.device)
        self._sp_grad_buf = torch.zeros(max_gaussians, 3, device=self.device)

        self.valid_length: int              = 0
        self.cpu_indices:  np.ndarray       = np.empty(0, dtype=np.int64)
        self.optimizer:    Optional[torch.optim.Optimizer] = None
        self._lr_dict:     Dict[str, float] = {}   # stored for lazy optimizer creation
        self._adam_state:  Optional[Dict]   = None # CPU adam state stashed until ensure_optimizer()

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
        adam_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> None:
        """
        In-place load of a new Gaussian batch from CPU pinned tensors.

        params:     dict from CPUGaussianStore.gather() — CPU pinned tensors
        indices:    global Gaussian indices for writeback scatter
        lr_dict:    {param_name → lr} passed to the Adam optimizer
                    keys: 'xyz', 'f_dc', 'scaling', 'rotation', 'opacity'
        adam_state: optional dict from CPUGaussianStore.gather_adam().
                    When provided, the optimizer is warm-started with the
                    restored exp_avg / exp_avg_sq / step instead of zeros.

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
        self._xyz_buf[:K].copy_(params['_xyz'],                  non_blocking=True)
        self._features_dc_buf[:K].copy_(params['_features_dc'],  non_blocking=True)
        self._scaling_buf[:K].copy_(params['_scaling'],          non_blocking=True)
        self._rotation_buf[:K].copy_(params['_rotation'],        non_blocking=True)
        self._opacity_buf[:K].copy_(params['_opacity'],          non_blocking=True)

        # Create parameter VIEWS of exactly [:K] rows — no new GPU allocation
        self._xyz         = nn.Parameter(self._xyz_buf[:K])
        self._features_dc = nn.Parameter(self._features_dc_buf[:K])
        self._scaling     = nn.Parameter(self._scaling_buf[:K])
        self._rotation    = nn.Parameter(self._rotation_buf[:K])
        self._opacity     = nn.Parameter(self._opacity_buf[:K])

        # Drop old optimizer to free its GPU moment tensors.  A new optimizer
        # is created lazily in ensure_optimizer() when this slot goes active.
        # _adam_state is stashed here (CPU pinned) and injected at that point.
        self.optimizer    = None
        self._lr_dict     = lr_dict
        self._adam_state  = adam_state   # None → cold start; dict → warm start

    def ensure_optimizer(self) -> None:
        """
        Create the Adam optimizer for the current parameter views.

        Called by GaussianSwapBuffer.pop() the moment this slot becomes active.
        Lazy creation keeps the prefetched (inactive) slot free of moment tensors
        (~3.4 GB for 30 M Gaussians).

        If _adam_state was set by load_from(), the saved exp_avg / exp_avg_sq /
        step are injected directly into optimizer.state so training resumes with
        warm momentum rather than cold-starting from zero.

        Hyperparameters match automated3DGS: betas=(0.9, 0.999), eps=1e-15.
        """
        if self.optimizer is not None:
            return

        lr = self._lr_dict
        self.optimizer = torch.optim.Adam([
            {'params': [self._xyz],         'lr': lr.get('xyz',      1.6e-4), 'name': 'xyz'},
            {'params': [self._features_dc], 'lr': lr.get('f_dc',     2.5e-3), 'name': 'f_dc'},
            {'params': [self._scaling],     'lr': lr.get('scaling',  1.0e-3), 'name': 'scaling'},
            {'params': [self._rotation],    'lr': lr.get('rotation', 1.0e-3), 'name': 'rotation'},
            {'params': [self._opacity],     'lr': lr.get('opacity',  5.0e-2), 'name': 'opacity'},
        ], betas=(0.9, 0.999), eps=1e-15)

        if self._adam_state is not None:
            # Map param-group name → the Parameter object held by this slice.
            name_to_param = {
                'xyz':      self._xyz,
                'f_dc':     self._features_dc,
                'scaling':  self._scaling,
                'rotation': self._rotation,
                'opacity':  self._opacity,
            }
            for name, p in name_to_param.items():
                if name not in self._adam_state:
                    continue
                s = self._adam_state[name]
                # H2D for moment tensors — non_blocking is safe here because
                # pop() already waited for slot_ready (param H2D done), and
                # the compute stream will not touch these until optimizer.step().
                self.optimizer.state[p] = {
                    'step':        s['step'],   # must stay on CPU as float32 scalar
                    'exp_avg':     s['exp_avg'].to(self.device, non_blocking=True),
                    'exp_avg_sq':  s['exp_avg_sq'].to(self.device, non_blocking=True),
                }
            self._adam_state = None   # free CPU pinned memory

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

    @torch.no_grad()
    def collect_optimizer_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Copy current Adam optimizer state ([:valid_length]) to CPU tensors.

        Returns a dict keyed by param-group name matching CPUGaussianStore.ADAM_NAMES:
            {
                'xyz': {'step': tensor, 'exp_avg': tensor, 'exp_avg_sq': tensor},
                ...
            }
        Returns an empty dict if the optimizer has not been stepped yet
        (state is only populated after the first optimizer.step() call).
        """
        if self.optimizer is None:
            return {}

        name_to_param = {
            'xyz':      self._xyz,
            'f_dc':     self._features_dc,
            'scaling':  self._scaling,
            'rotation': self._rotation,
            'opacity':  self._opacity,
        }
        result = {}
        for name, p in name_to_param.items():
            s = self.optimizer.state.get(p)
            if not s:
                continue
            result[name] = {
                'step':        s['step'].cpu(),
                'exp_avg':     s['exp_avg'].cpu(),
                'exp_avg_sq':  s['exp_avg_sq'].cpu(),
            }
        return result
