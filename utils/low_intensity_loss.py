from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from utils.loss_utils import ssim


@dataclass(frozen=True)
class LowIntensityLossConfig:
    enabled: bool = False

    # Overall multiplier for the low-intensity loss.
    overall_weight: float = 1.0

    # Loss terms:
    # w_rgb * RGB_error + w_ssim * (1 - SSIM) + w_intensity * RGB_error * (1 - intensity)
    rgb_weight: float = 0.8
    ssim_weight: float = 0.2
    intensity_weight: float = 3.0


def add_low_intensity_loss_args(parser):
    parser.add_argument("--low_intensity_loss", action="store_true", default=False)
    parser.add_argument("--low_intensity_weight", type=float, default=1.0)
    parser.add_argument("--low_intensity_rgb_weight", type=float, default=0.8)
    parser.add_argument("--low_intensity_ssim_weight", type=float, default=0.2)
    parser.add_argument("--low_intensity_intensity_weight", type=float, default=3.0)


def build_low_intensity_loss_config(args) -> LowIntensityLossConfig:
    return LowIntensityLossConfig(
        enabled=bool(args.low_intensity_loss),
        overall_weight=float(args.low_intensity_weight),
        rgb_weight=float(args.low_intensity_rgb_weight),
        ssim_weight=float(args.low_intensity_ssim_weight),
        intensity_weight=float(args.low_intensity_intensity_weight),
    )


def rgb_to_luma(image: torch.Tensor) -> torch.Tensor:
    if image.dim() != 3:
        raise ValueError(f"Expected CHW image, got shape {tuple(image.shape)}")

    if image.shape[0] >= 3:
        coeffs = image.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
        return (image[:3] * coeffs).sum(dim=0, keepdim=True)
    return image.mean(dim=0, keepdim=True)


def compute_low_intensity_mask(gt_image: torch.Tensor) -> torch.Tensor:
    intensity = rgb_to_luma(gt_image).clamp(0.0, 1.0)
    return (1.0 - intensity).clamp(0.0, 1.0)


def compute_low_intensity_loss(
    pred_image: torch.Tensor,
    gt_image: torch.Tensor,
    cfg: LowIntensityLossConfig,
    ssim_value: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    pred_image, gt_image: (3,H,W), expected in [0,1]
    Returns:
        total_loss: scalar tensor
        stats: dict with scalar tensors + weight map
    """
    if not cfg.enabled:
        zero = pred_image.new_zeros(())
        return zero, {
            "mask_mean": zero,
            "mask_max": zero,
            "rgb_term": zero,
            "ssim_term": zero,
            "intensity_term": zero,
        }

    pred_rgb = pred_image.clamp(0.0, 1.0)
    gt_rgb = gt_image.clamp(0.0, 1.0)

    rgb_error_map = (pred_rgb - gt_rgb).abs().mean(dim=0, keepdim=True)
    intensity_mask = compute_low_intensity_mask(gt_rgb)

    rgb_term = rgb_error_map.mean()
    intensity_term = (rgb_error_map * intensity_mask).mean()

    if ssim_value is None:
        ssim_value = ssim(pred_rgb, gt_rgb)
    ssim_term = 1.0 - ssim_value

    total = (
        cfg.rgb_weight * rgb_term
        + cfg.ssim_weight * ssim_term
        + cfg.intensity_weight * intensity_term
    )
    total = cfg.overall_weight * total

    stats = {
        "mask_mean": intensity_mask.mean().detach(),
        "mask_max": intensity_mask.max().detach(),
        "rgb_term": rgb_term.detach(),
        "ssim_term": ssim_term.detach(),
        "intensity_term": intensity_term.detach(),
    }
    return total, stats
