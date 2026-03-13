import json
import math
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.sh_utils import eval_sh

try:
    from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
except ImportError:
    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    except ImportError as exc:
        raise SystemExit(
            "Failed to import a depth-capable rasterizer backend. "
            "Install diff_gaussian_rasterization_depth, or run this script from an environment "
            "where the depth-capable diff_gaussian_rasterization package is available."
        ) from exc


def render_depth(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, use_trained_exp: bool = False):
    screenspace_points = torch.zeros_like(
        pc.get_xyz,
        dtype=pc.get_xyz.dtype,
        requires_grad=True,
        device="cuda",
    ) + 0
    try:
        screenspace_points.retain_grad()
    except RuntimeError:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings_kwargs = {
        "image_height": int(viewpoint_camera.image_height),
        "image_width": int(viewpoint_camera.image_width),
        "tanfovx": tanfovx,
        "tanfovy": tanfovy,
        "bg": bg_color,
        "scale_modifier": 1.0,
        "viewmatrix": viewpoint_camera.world_view_transform,
        "projmatrix": viewpoint_camera.full_proj_transform,
        "sh_degree": pc.active_sh_degree,
        "campos": viewpoint_camera.camera_center,
        "prefiltered": False,
        "debug": pipe.debug,
    }
    raster_settings_fields = set(getattr(GaussianRasterizationSettings, "_fields", ()))
    if "kernel_size" in raster_settings_fields:
        raster_settings_kwargs["kernel_size"] = 0.0
    if "require_depth" in raster_settings_fields:
        raster_settings_kwargs["require_depth"] = True
    if "require_coord" in raster_settings_fields:
        raster_settings_kwargs["require_coord"] = False

    raster_settings = GaussianRasterizationSettings(**raster_settings_kwargs)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(1.0)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        shs = pc.get_features

    raster_outputs = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    if not isinstance(raster_outputs, tuple):
        raise RuntimeError(
            "The loaded diff_gaussian_rasterization backend does not expose depth outputs. "
            "Run this script with the depth-capable backend from the GauUscene/RaDe-GS environment."
        )

    if len(raster_outputs) == 8:
        rendered_image, radii, _coord, _mcoord, depth, median_depth, alpha, _normal = raster_outputs
    elif len(raster_outputs) == 5:
        rendered_image, depth, median_depth, alpha, radii = raster_outputs
    else:
        raise RuntimeError(
            f"Unsupported depth rasterizer output format with {len(raster_outputs)} values."
        )

    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = (
            torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1)
            + exposure[:3, 3, None, None]
        )

    return {
        "render": rendered_image.clamp(0, 1),
        "depth": depth,
        "median_depth": median_depth,
        "alpha": alpha,
        "radii": radii,
    }


def resolve_raw_depth_dir(source_path: str, raw_depth_dir: str | None) -> Path:
    candidates = []
    if raw_depth_dir is not None:
        candidates.append(Path(raw_depth_dir))
    candidates.extend(
        [
            Path(source_path) / "synthetic" / "raw_depths",
            Path(source_path) / "synthetic" / "raw_depth",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a raw depth directory. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def resize_depth_map(depth_map: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if depth_map.shape == (target_h, target_w):
        return depth_map
    depth_tensor = torch.from_numpy(depth_map).float()[None, None]
    resized = F.interpolate(depth_tensor, size=(target_h, target_w), mode="bicubic", align_corners=False)
    return resized[0, 0].cpu().numpy()


def compute_depth_metrics(pred: np.ndarray, gt: np.ndarray, alpha: np.ndarray) -> dict | None:
    valid_mask = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (pred > 0) & (alpha > 1e-6)
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        return None

    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    abs_err = np.abs(pred_valid - gt_valid)
    sq_err = (pred_valid - gt_valid) ** 2
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)

    return {
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt(sq_err.mean())),
        "AbsRel": float((abs_err / gt_valid).mean()),
        "Delta1": float((ratio < 1.25).mean()),
        "Delta2": float((ratio < 1.25**2).mean()),
        "Delta3": float((ratio < 1.25**3).mean()),
        "ValidPixels": valid_count,
    }


def aggregate_metrics(per_view_metrics: dict[str, dict]) -> dict:
    if not per_view_metrics:
        return {}

    metric_names = list(next(iter(per_view_metrics.values())).keys())
    aggregated = {}
    for metric_name in metric_names:
        values = [metrics[metric_name] for metrics in per_view_metrics.values()]
        aggregated[metric_name] = float(np.mean(values))
    return aggregated


VIRIDIS_ANCHORS = np.array(
    [
        [68, 1, 84],
        [71, 44, 122],
        [59, 81, 139],
        [44, 113, 142],
        [33, 144, 141],
        [39, 173, 129],
        [92, 200, 99],
        [170, 220, 50],
        [253, 231, 37],
    ],
    dtype=np.float32,
)

ERROR_ANCHORS = np.array(
    [
        [0, 0, 255],
        [64, 128, 255],
        [255, 255, 255],
        [255, 128, 64],
        [255, 0, 0],
    ],
    dtype=np.float32,
)


def interpolate_colormap(normalized_map: np.ndarray, valid_mask: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    normalized_map = np.clip(normalized_map, 0.0, 1.0)
    anchor_x = np.linspace(0.0, 1.0, anchors.shape[0], dtype=np.float32)
    mapped = np.zeros((*normalized_map.shape, 3), dtype=np.uint8)
    for channel_idx in range(3):
        mapped[..., channel_idx] = np.interp(
            normalized_map,
            anchor_x,
            anchors[:, channel_idx],
        ).astype(np.uint8)
    mapped[~valid_mask] = 0
    return mapped


def viridis_like_colormap(normalized_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return interpolate_colormap(normalized_map, valid_mask, VIRIDIS_ANCHORS)


def blue_red_colormap(normalized_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return interpolate_colormap(normalized_map, valid_mask, ERROR_ANCHORS)


def build_colorbar(height: int, min_depth: float, max_depth: float) -> Image.Image:
    bar_width = 72
    label_pad = 8
    canvas = Image.new("RGB", (bar_width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    gradient = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    gradient_rgb = viridis_like_colormap(np.repeat(gradient, 20, axis=1), np.ones((height, 20), dtype=bool))
    gradient_img = Image.fromarray(gradient_rgb, mode="RGB")
    canvas.paste(gradient_img, (label_pad, 0))

    draw.text((32, 0), f"{max_depth:.2f}", fill=(0, 0, 0))
    draw.text((32, max(0, height // 2 - 6)), f"{0.5 * (min_depth + max_depth):.2f}", fill=(0, 0, 0))
    draw.text((32, max(0, height - 14)), f"{min_depth:.2f}", fill=(0, 0, 0))
    return canvas


def build_error_colorbar(height: int, max_error: float) -> Image.Image:
    bar_width = 72
    label_pad = 8
    canvas = Image.new("RGB", (bar_width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    gradient = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    gradient_rgb = blue_red_colormap(np.repeat(gradient, 20, axis=1), np.ones((height, 20), dtype=bool))
    gradient_img = Image.fromarray(gradient_rgb, mode="RGB")
    canvas.paste(gradient_img, (label_pad, 0))

    draw.text((32, 0), f"{max_error:.2f}", fill=(0, 0, 0))
    draw.text((32, max(0, height // 2 - 6)), f"{0.5 * max_error:.2f}", fill=(0, 0, 0))
    draw.text((32, max(0, height - 14)), "0.00", fill=(0, 0, 0))
    return canvas


def save_depth_comparison_visualization(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    alpha_map: np.ndarray,
    output_path: Path,
    pred_label: str = "Pred",
    gt_label: str = "Raw",
):
    pred_valid = np.isfinite(pred_depth) & (pred_depth > 0) & (alpha_map > 1e-6)
    gt_valid = np.isfinite(gt_depth) & (gt_depth > 0)
    valid_union = pred_valid | gt_valid

    if not valid_union.any():
        width = pred_depth.shape[1] * 2 + 92
        height = pred_depth.shape[0] + 24
        canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), "No valid depth pixels", fill=(0, 0, 0))
        canvas.save(output_path)
        return

    min_depth = float(np.min(np.concatenate([pred_depth[pred_valid], gt_depth[gt_valid]])))
    max_depth = float(np.max(np.concatenate([pred_depth[pred_valid], gt_depth[gt_valid]])))
    if not np.isfinite(min_depth) or not np.isfinite(max_depth) or max_depth <= min_depth:
        max_depth = min_depth + 1.0

    pred_norm = (pred_depth - min_depth) / (max_depth - min_depth)
    gt_norm = (gt_depth - min_depth) / (max_depth - min_depth)
    pred_rgb = Image.fromarray(viridis_like_colormap(pred_norm, pred_valid), mode="RGB")
    gt_rgb = Image.fromarray(viridis_like_colormap(gt_norm, gt_valid), mode="RGB")

    header_h = 24
    separator_w = 6
    colorbar = build_colorbar(pred_depth.shape[0], min_depth, max_depth)
    canvas_w = pred_rgb.width + separator_w + gt_rgb.width + separator_w + colorbar.width
    canvas_h = header_h + pred_rgb.height
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    canvas.paste(pred_rgb, (0, header_h))
    canvas.paste(gt_rgb, (pred_rgb.width + separator_w, header_h))
    canvas.paste(colorbar, (pred_rgb.width + separator_w + gt_rgb.width + separator_w, header_h))

    draw.text((8, 6), pred_label, fill=(0, 0, 0))
    draw.text((pred_rgb.width + separator_w + 8, 6), gt_label, fill=(0, 0, 0))
    draw.text((pred_rgb.width + separator_w + gt_rgb.width + separator_w + 8, 6), "Depth", fill=(0, 0, 0))

    canvas.save(output_path)


def save_depth_error_visualization(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    alpha_map: np.ndarray,
    output_path: Path,
):
    valid_mask = (
        np.isfinite(pred_depth)
        & np.isfinite(gt_depth)
        & (pred_depth > 0)
        & (gt_depth > 0)
        & (alpha_map > 1e-6)
    )

    error_map = np.abs(pred_depth - gt_depth)
    if not valid_mask.any():
        canvas = Image.new("RGB", (pred_depth.shape[1] + 72, pred_depth.shape[0] + 24), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), "No valid depth pixels", fill=(0, 0, 0))
        canvas.save(output_path)
        return

    max_error = float(error_map[valid_mask].max())
    if not np.isfinite(max_error) or max_error <= 0:
        max_error = 1.0

    normalized_error = error_map / max_error
    error_rgb = Image.fromarray(blue_red_colormap(normalized_error, valid_mask), mode="RGB")
    colorbar = build_error_colorbar(pred_depth.shape[0], max_error)

    header_h = 24
    separator_w = 6
    canvas_w = error_rgb.width + separator_w + colorbar.width
    canvas_h = header_h + error_rgb.height
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    canvas.paste(error_rgb, (0, header_h))
    canvas.paste(colorbar, (error_rgb.width + separator_w, header_h))
    draw.text((8, 6), "Abs Error", fill=(0, 0, 0))
    draw.text((error_rgb.width + separator_w + 8, 6), "Error", fill=(0, 0, 0))
    canvas.save(output_path)


def save_rgb_image(rgb_tensor: torch.Tensor, output_path: Path):
    rgb = rgb_tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb_uint8, mode="RGB").save(output_path)


def render_split_depth(
    model_path: Path,
    split_name: str,
    iteration: int,
    views,
    gaussians: GaussianModel,
    pipeline,
    background: torch.Tensor,
    train_test_exp: bool,
    raw_depth_dir: Path,
):
    base_dir = model_path / split_name / f"ours_{iteration}"
    depth_dir = base_dir / "depth"
    median_depth_dir = base_dir / "median_depth"
    alpha_dir = base_dir / "alpha"
    depth_vis_dir = base_dir / "depth_vis"
    rendered_depth_dir = model_path / "rendered" / "depth"
    rendered_rgb_dir = model_path / "rendered" / "rgb"
    for out_dir in (depth_dir, median_depth_dir, alpha_dir, depth_vis_dir, rendered_depth_dir, rendered_rgb_dir):
        out_dir.mkdir(parents=True, exist_ok=True)

    per_view_depth = {}
    per_view_median_depth = {}
    resized_gt_notice_printed = False

    for view in tqdm(views, desc=f"Depth rendering ({split_name})"):
        render_pkg = render_depth(view, gaussians, pipeline, background, use_trained_exp=train_test_exp)
        pred_depth = render_pkg["depth"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        pred_median_depth = render_pkg["median_depth"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        pred_alpha = render_pkg["alpha"].squeeze(0).detach().cpu().numpy().astype(np.float32)

        image_stem = Path(view.image_name).stem
        np.save(depth_dir / f"{image_stem}_depth.npy", pred_depth)
        np.save(median_depth_dir / f"{image_stem}_median_depth.npy", pred_median_depth)
        np.save(alpha_dir / f"{image_stem}_alpha.npy", pred_alpha)
        save_rgb_image(render_pkg["render"], rendered_rgb_dir / f"{split_name}_{image_stem}_rgb.png")
        if np.isfinite(pred_depth).any():
            pred_only_gt = np.where(np.isfinite(pred_depth), pred_depth, 0.0)
            save_depth_comparison_visualization(
                pred_depth,
                pred_only_gt,
                pred_alpha,
                depth_vis_dir / f"{image_stem}_depth.png",
                pred_label="Pred",
                gt_label="Pred",
            )

        gt_depth_path = raw_depth_dir / f"{image_stem}_depth.npy"
        if not gt_depth_path.exists():
            continue

        gt_depth = np.load(gt_depth_path).astype(np.float32)
        if gt_depth.shape != pred_depth.shape:
            if not resized_gt_notice_printed:
                print(
                    f"Resizing raw depth maps from {gt_depth.shape} to {pred_depth.shape} for evaluation."
                )
                resized_gt_notice_printed = True
            gt_depth = resize_depth_map(gt_depth, pred_depth.shape)

        save_depth_comparison_visualization(
            pred_depth,
            gt_depth,
            pred_alpha,
            rendered_depth_dir / f"{split_name}_{image_stem}_depth_compare.png",
        )
        save_depth_error_visualization(
            pred_depth,
            gt_depth,
            pred_alpha,
            rendered_depth_dir / f"{split_name}_{image_stem}_depth_error.png",
        )

        depth_metrics = compute_depth_metrics(pred_depth, gt_depth, pred_alpha)
        if depth_metrics is not None:
            per_view_depth[image_stem] = depth_metrics

        median_depth_metrics = compute_depth_metrics(pred_median_depth, gt_depth, pred_alpha)
        if median_depth_metrics is not None:
            per_view_median_depth[image_stem] = median_depth_metrics

    results = {
        "depth": aggregate_metrics(per_view_depth),
        "median_depth": aggregate_metrics(per_view_median_depth),
    }
    per_view = {
        "depth": per_view_depth,
        "median_depth": per_view_median_depth,
    }

    with open(base_dir / "depth_metrics.json", "w") as fp:
        json.dump(results, fp, indent=2)
    with open(base_dir / "depth_per_view.json", "w") as fp:
        json.dump(per_view, fp, indent=2)

    print(f"{split_name} depth metrics:")
    for metric_group, metric_values in results.items():
        if not metric_values:
            print(f"  {metric_group}: no matching raw depth files found.")
            continue
        print(f"  {metric_group}:")
        for metric_name, metric_value in metric_values.items():
            print(f"    {metric_name}: {metric_value:.6f}")


def evaluate_depth(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    raw_depth_dir: str | None,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=[1.0])
        raw_depth_root = resolve_raw_depth_dir(dataset.source_path, raw_depth_dir)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_split_depth(
                Path(dataset.model_path),
                "train",
                scene.loaded_iter,
                scene.train_cameras[1.0],
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
                raw_depth_root,
            )

        if not skip_test:
            render_split_depth(
                Path(dataset.model_path),
                "test",
                scene.loaded_iter,
                scene.test_cameras[1.0],
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
                raw_depth_root,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Render and evaluate depth maps")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--raw_depth_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Evaluating depth for " + args.model_path)
    safe_state(args.quiet)

    evaluate_depth(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.raw_depth_dir,
    )
