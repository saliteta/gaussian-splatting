import numpy as np
import torch
from typing import Dict


class CPUGaussianStore:
    """
    All Gaussian parameters in CPU pinned memory, plus Adam optimizer state.

    SH degree 0 only: stores xyz, features_dc, scaling, rotation, opacity.

    Adam state (exp_avg, exp_avg_sq per Gaussian; step scalar per param) is
    stored alongside the parameters so that GPUGaussianSlice can restore a
    warm optimizer on every slot swap instead of starting from zero.

    Layout: after optional reorder(), Gaussian i is at row i in the
    permuted order. Writeback scatter uses the same indices as the gather.
    """

    FIELDS = ('_xyz', '_features_dc', '_scaling', '_rotation', '_opacity')

    # Mapping: Adam param-group name → store field name
    # Used to build / restore optimizer state in GPUGaussianSlice.
    ADAM_NAMES = ('xyz', 'f_dc', 'scaling', 'rotation', 'opacity')
    _ADAM_TO_FIELD = {
        'xyz':     '_xyz',
        'f_dc':    '_features_dc',
        'scaling': '_scaling',
        'rotation':'_rotation',
        'opacity': '_opacity',
    }

    def __init__(self, gaussian_model):
        """
        Copy all parameters from an existing GaussianModel into CPU pinned memory.
        Adam state buffers are initialised to zero (cold start).
        After construction the GaussianModel can be discarded.
        """
        self.N = gaussian_model._xyz.shape[0]

        def _pin(t: torch.Tensor) -> torch.Tensor:
            return t.detach().cpu().contiguous().pin_memory()

        def _zero_pin(*shape) -> torch.Tensor:
            return torch.zeros(*shape, dtype=torch.float32).pin_memory()

        self._xyz         = _pin(gaussian_model._xyz)           # (N, 3)
        self._features_dc = _pin(gaussian_model._features_dc)   # (N, 1, 3)
        self._scaling     = _pin(gaussian_model._scaling)        # (N, 3)
        self._rotation    = _pin(gaussian_model._rotation)       # (N, 4)
        self._opacity     = _pin(gaussian_model._opacity)        # (N, 1)

        # Adam moment buffers — same shape as the corresponding parameter tensors.
        # exp_avg  = first moment  (momentum),  β1 = 0.9
        # exp_avg_sq = second moment (variance), β2 = 0.999
        N = self.N
        self._adam_exp_avg: Dict[str, torch.Tensor] = {
            'xyz':      _zero_pin(N, 3),
            'f_dc':     _zero_pin(N, 1, 3),
            'scaling':  _zero_pin(N, 3),
            'rotation': _zero_pin(N, 4),
            'opacity':  _zero_pin(N, 1),
        }
        self._adam_exp_avg_sq: Dict[str, torch.Tensor] = {
            'xyz':      _zero_pin(N, 3),
            'f_dc':     _zero_pin(N, 1, 3),
            'scaling':  _zero_pin(N, 3),
            'rotation': _zero_pin(N, 4),
            'opacity':  _zero_pin(N, 1),
        }
        # Step counter: one float32 CPU scalar (0-dim) per param group.
        # PyTorch Adam requires step to be a CPU float32/64 scalar — it must
        # NOT be moved to CUDA or stored as int64.
        self._adam_step: Dict[str, torch.Tensor] = {
            name: torch.zeros((), dtype=torch.float32)
            for name in self.ADAM_NAMES
        }

        param_mb = sum(getattr(self, f).nbytes for f in self.FIELDS) / 1024 ** 2
        adam_mb  = sum(
            v.nbytes
            for d in (self._adam_exp_avg, self._adam_exp_avg_sq)
            for v in d.values()
        ) / 1024 ** 2
        print(f"[CPUGaussianStore] {self.N:,} Gaussians, "
              f"{param_mb:.0f} MB params + {adam_mb:.0f} MB Adam state "
              f"= {param_mb + adam_mb:.0f} MB pinned CPU memory.")

    def reorder(self, perm: np.ndarray) -> None:
        """
        Permute all fields (params + Adam moments) in-place according to `perm`.
        Must be called before any gather()/scatter() so indices stay consistent.
        """
        idx = torch.from_numpy(perm).long()
        for field in self.FIELDS:
            t = getattr(self, field)
            setattr(self, field, t[idx].contiguous().pin_memory())
        for name in self.ADAM_NAMES:
            self._adam_exp_avg[name]    = self._adam_exp_avg[name][idx].contiguous().pin_memory()
            self._adam_exp_avg_sq[name] = self._adam_exp_avg_sq[name][idx].contiguous().pin_memory()

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

    def gather_adam(self, indices: np.ndarray) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Gather Adam optimizer state for the given Gaussian indices.

        Returns a dict keyed by param-group name ('xyz', 'f_dc', ...):
            {
                'exp_avg':    pinned CPU tensor of shape (K, ...),
                'exp_avg_sq': pinned CPU tensor of shape (K, ...),
                'step':       float32 CPU scalar tensor (same for all Gaussians),
            }
        """
        idx = torch.from_numpy(indices).long()
        return {
            name: {
                'exp_avg':    self._adam_exp_avg[name][idx].pin_memory(),
                'exp_avg_sq': self._adam_exp_avg_sq[name][idx].pin_memory(),
                'step':       self._adam_step[name].clone(),
            }
            for name in self.ADAM_NAMES
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

    def scatter_adam(
        self,
        indices: np.ndarray,
        adam_state: Dict[str, Dict[str, torch.Tensor]],
    ) -> None:
        """
        Scatter Adam optimizer state back into the store at global `indices`.
        adam_state: output of GPUGaussianSlice.collect_optimizer_state().
        """
        idx = torch.from_numpy(indices).long()
        with torch.no_grad():
            for name, s in adam_state.items():
                if name not in self._adam_exp_avg:
                    continue
                self._adam_exp_avg[name][idx]    = s['exp_avg'].cpu()
                self._adam_exp_avg_sq[name][idx] = s['exp_avg_sq'].cpu()
                # step is a global scalar — take the latest value from the batch
                self._adam_step[name].copy_(s['step'].cpu())

    def apply_zero_grad_adam(
        self,
        non_batch_indices: np.ndarray,
        global_step: int,
        n_steps: int,
        lr_dict: Dict[str, float],
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-15,
    ) -> None:
        """
        Apply n_steps of zero-gradient Adam updates to non-batch Gaussians on CPU.

        When a Gaussian is not in the active GPU batch, it receives no gradient.
        However, Adam's momentum still decays and the residual first moment
        continues to drive parameter updates — exactly what train_fix does when
        all Gaussians live on GPU with g=0.

        Math for n zero-grad steps starting at step t0 (global_step before these):
          m_t0+n = beta1^n * m_t0                (pure decay, no new gradient)
          v_t0+n = beta2^n * v_t0                (pure decay, no new gradient)

        Cumulative parameter delta over those n steps (geometric series):
          r = beta1 / sqrt(beta2)
          geom_coeff = r*(1 - r^n)/(1 - r)      (sum of r^1 + r^2 + ... + r^n)
          bias_corr_1 = 1 - beta1^(t0 + 1)      (conservative: use first step)
          bias_corr_2 = 1 - beta2^(t0 + 1)
          Δθ ≈ -lr * geom_coeff * (m_t0/bias_corr_1) / (sqrt(v_t0/bias_corr_2) + eps)

        Using t0+1 for bias correction is conservative (slightly larger correction
        than the true per-step average) but avoids per-step loops on CPU.

        Both moments are updated in-place. The step counter is NOT advanced here
        because it is a global scalar shared with the GPU batch — the GPU batch
        owns step advancement.
        """
        import math

        if len(non_batch_indices) == 0 or n_steps <= 0:
            return

        idx    = torch.from_numpy(non_batch_indices).long()
        beta1_n = beta1 ** n_steps
        beta2_n = beta2 ** n_steps

        # r = beta1 / sqrt(beta2); geometric series coefficient for Δθ
        r = beta1 / math.sqrt(beta2)
        if abs(r - 1.0) < 1e-12:
            geom_coeff = float(n_steps)
        else:
            geom_coeff = r * (1.0 - r ** n_steps) / (1.0 - r)

        # Bias-correction using conservative step t0+1 (avoids per-step loop)
        t0 = global_step
        bc1 = 1.0 - beta1 ** (t0 + 1)
        bc2 = 1.0 - beta2 ** (t0 + 1)

        with torch.no_grad():
            for name, field in self._ADAM_TO_FIELD.items():  # Theta is the parameter tensor, m is the first moment, v is the second moment
                lr = lr_dict.get(name, 0.0)
                if lr == 0.0:
                    continue

                theta = getattr(self, field)       # (N, ...) pinned CPU
                m     = self._adam_exp_avg[name]   # (N, ...) pinned CPU
                v     = self._adam_exp_avg_sq[name]

                # Snapshot pre-decay moments (fancy indexing always returns a copy)
                m_pre = m[idx]   # (K, ...) fresh copy — safe to modify in-place
                v_pre = v[idx]   # (K, ...)

                # Scatter decayed moments back.  Fancy indexing returns a copy so
                # we must use __setitem__ to write back; .mul_() on the slice is a no-op.
                m[idx] = m_pre * beta1_n
                v[idx] = v_pre * beta2_n

                # Bias-corrected update (modifies m_pre / v_pre in-place — safe since
                # they are local copies not referenced by the store tensors).
                m_hat      = m_pre.div_(bc1)                    # (K, ...)
                v_hat_sqrt = v_pre.div_(bc2).sqrt_().add_(eps)  # (K, ...)
                # Accumulate parameter delta and scatter back
                delta = m_hat.mul_(geom_coeff).div_(v_hat_sqrt).mul_(lr)
                theta[idx] = theta[idx] - delta
