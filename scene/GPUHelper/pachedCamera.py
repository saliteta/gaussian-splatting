from dataclasses import dataclass
from typing import Optional, Any
import torch

@dataclass
class PackedCameraView:
    # meta
    uid: Any
    colmap_id: Any
    image_name: str
    FoVx: float
    FoVy: float
    znear: float
    zfar: float
    image_width: int
    image_height: int

    # CUDA views into packed slabs
    original_image: torch.Tensor          # (3,H,W) view
    alpha_mask: torch.Tensor              # (1,H,W) view

    # CUDA small tensors (packed too)
    world_view_transform: torch.Tensor    # (4,4)
    projection_matrix: torch.Tensor       # (4,4)
    full_proj_transform: torch.Tensor     # (4,4)
    camera_center: torch.Tensor           # (3,)

    cpu_camera: Optional[Any] = None
    cx: Optional[float] = None   # principal point x (scaled); None → use image_width/2
    cy: Optional[float] = None   # principal point y (scaled); None → use image_height/2
