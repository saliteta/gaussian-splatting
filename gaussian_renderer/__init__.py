import math
import torch
from gsplat import rasterization
from scene.GPUHelper.pachedCamera import PackedCameraView


def render(viewpoint_camera: PackedCameraView, pc, pipe, bg_color: torch.Tensor,
           scaling_modifier: float = 1.0, separate_sh: bool = False,
           override_color=None, use_trained_exp: bool = False):
    """
    Render the scene using gsplat.

    Drop-in replacement for the diff-gaussian-rasterization renderer.
    bg_color must be a (3,) GPU tensor.

    Key differences from the original renderer:
      - Uses gsplat.rasterization() instead of GaussianRasterizer.
      - Activations (exp, sigmoid, normalize) are applied in Python before
        the kernel call — gsplat expects already-activated values.
      - world_view_transform is stored as W2C^T (column-major for CUDA);
        gsplat expects the actual W2C matrix, so we transpose it here.
      - packed=True: intermediate results are sparse, saving memory for
        large scenes where each camera sees only a subset of Gaussians.
      - Output render_colors is (C, H, W, 3) NHWC; we permute to (3, H, W).
    """
    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)
    device = bg_color.device

    # --- Intrinsics matrix K from FoV + principal point ---
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    fx = W / (2.0 * tanfovx)
    fy = H / (2.0 * tanfovy)
    # Use the actual principal point if available; fall back to image centre.
    cx = getattr(viewpoint_camera, 'cx', None)
    cy = getattr(viewpoint_camera, 'cy', None)
    cx = cx if cx is not None else W / 2.0
    cy = cy if cy is not None else H / 2.0
    Ks = torch.tensor(
        [[fx,  0.0, cx],
         [0.0, fy,  cy],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)  # (1, 3, 3)

    # --- View matrix ---
    # world_view_transform is stored as W2C^T (column-major convention for the
    # original CUDA rasterizer).  gsplat expects the standard row-major W2C
    # matrix, so we transpose back.
    viewmats = viewpoint_camera.world_view_transform.T.contiguous().unsqueeze(0)  # (1, 4, 4)

    # --- Gaussian parameters (activations applied before kernel call) ---
    means     = pc.get_xyz                           # (N, 3)
    scales    = pc.get_scaling * scaling_modifier    # (N, 3)  exp already applied
    quats     = pc.get_rotation                      # (N, 4)  normalized
    opacities = pc.get_opacity.squeeze(-1)           # (N,)

    # --- Colors / SH ---
    if override_color is not None:
        colors        = override_color   # (N, 3) pre-computed RGB, sh_degree unused
        sh_degree_arg = None
    else:
        colors        = pc.get_features  # (N, K, 3) SH coefficients; K=1 for degree 0
        sh_degree_arg = pc.active_sh_degree

    # --- Rasterize ---
    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        near_plane=viewpoint_camera.znear,
        far_plane=viewpoint_camera.zfar,
        backgrounds=bg_color.unsqueeze(0),  # (1, 3)
        sh_degree=sh_degree_arg,
        packed=True,
        radius_clip=0.0,
    )

    # render_colors: (1, H, W, 3) NHWC → (3, H, W) for the rest of the pipeline
    rendered_image = render_colors[0].permute(2, 0, 1).clamp(0, 1)

    # Apply per-camera exposure (stage 1 only; stage 2 always passes use_trained_exp=False)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = (
            torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3])
            .permute(2, 0, 1)
            + exposure[:3, 3, None, None]
        )

    # Radii: gsplat returns (C, N) per-camera radii in packed mode.
    # Squeeze the camera dimension for single-camera renders.
    radii = info.get("radii", torch.zeros(means.shape[0], device=device, dtype=torch.int32))
    if radii.ndim == 2:
        radii = radii[0]  # (N,)

    return {
        "render":            rendered_image,
        "viewspace_points":  info.get("means2d", means.detach()[:, :2]),
        "visibility_filter": (radii > 0).nonzero(),
        "radii":             radii,
        "depth":             None,
    }
