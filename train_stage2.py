"""
Stage-2 training loop: dynamic Gaussian loading with KNN batching.

Key differences from train_fix.py:
  - GaussianModel is replaced by CPUGaussianStore + GPUGaussianSlice
  - 100M+ Gaussians streamed via GaussianSwapBuffer (double-buffered)
  - No densification, no SH degree upgrade, no exposure optimizer
  - SH degree fixed at 0
  - B cameras per batch; one optimizer.step() per camera
  - LR for xyz follows exponential schedule; others are flat SGD
  - Checkpoints flush GPU state via finish_batch() then save CPUGaussianStore to PLY
"""

import os
import sys
import uuid
import torch
import numpy as np
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from tqdm import trange

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.cpu_gaussian_store import CPUGaussianStore
from scene.spatial_block_index import SpatialBlockIndex
from scene.visibility_precomputer import VisibilityPrecomputer
from scene.camera_batch_scheduler import CameraBatchScheduler
from scene.gaussian_swap_buffer import GaussianSwapBuffer
from utils.general_utils import safe_state, get_expon_lr_func
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.system_utils import mkdir_p

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def image_to_float01(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    if image.dtype == torch.uint8:
        return image.to(device=device, dtype=torch.float32) / 255.0
    return image.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)


def mask_to_float01(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    if mask.dtype == torch.uint8:
        return mask.to(device=device, dtype=torch.float32) / 255.0
    return mask.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)


def build_loss(rendered_image, viewpoint_cam, background, opt):
    device = rendered_image.device
    gt_rgb   = image_to_float01(viewpoint_cam.original_image, device)
    alpha    = mask_to_float01(viewpoint_cam.alpha_mask, device)
    bg_img   = background[:, None, None].expand_as(gt_rgb)
    gt_image = gt_rgb * alpha + bg_img * (1.0 - alpha)
    pred     = rendered_image.to(torch.float32)

    Ll1 = l1_loss(pred, gt_image)
    if FUSED_SSIM_AVAILABLE:
        ssim_val = fused_ssim(pred.unsqueeze(0), gt_image.unsqueeze(0))
    else:
        ssim_val = ssim(pred, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
    return loss, Ll1


def save_cpu_store_ply(cpu_store: CPUGaussianStore, path: str):
    """Save CPUGaussianStore to PLY in GaussianModel-compatible format (SH degree 0)."""
    from plyfile import PlyData, PlyElement
    mkdir_p(os.path.dirname(path))

    xyz      = cpu_store._xyz.numpy()                                    # (N, 3)
    normals  = np.zeros_like(xyz)
    f_dc     = cpu_store._features_dc.numpy()                            # (N, 1, 3)
    f_dc_out = f_dc.transpose(0, 2, 1).reshape(xyz.shape[0], -1)        # (N, 3)
    opacity  = cpu_store._opacity.numpy()                                # (N, 1)
    scale    = cpu_store._scaling.numpy()                                # (N, 3)
    rotation = cpu_store._rotation.numpy()                               # (N, 4)

    attrs = (
        ['x', 'y', 'z', 'nx', 'ny', 'nz']
        + [f'f_dc_{i}' for i in range(f_dc_out.shape[1])]
        + ['opacity']
        + [f'scale_{i}' for i in range(scale.shape[1])]
        + [f'rot_{i}' for i in range(rotation.shape[1])]
    )
    dtype    = [(a, 'f4') for a in attrs]
    data     = np.concatenate([xyz, normals, f_dc_out, opacity, scale, rotation], axis=1)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, data))
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
    print(f"  Saved {xyz.shape[0]:,} Gaussians → {path}")


def downsample_points(xyz_cpu: torch.Tensor, target: int) -> torch.Tensor:
    """Random uniform downsample to `target` points (or all if fewer)."""
    N = xyz_cpu.shape[0]
    if N <= target:
        return xyz_cpu
    idx = torch.randperm(N)[:target]
    return xyz_cpu[idx]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint,
             ds_target: int = 1_000_000,
             batch_size: int = 8,
             slot_budget_gb: float = 4.0,
             fov_margin: float = 0.1):

    device = torch.device("cuda")

    # -----------------------------------------------------------------------
    # 1. Load scene + initialise GaussianModel (for create_from_pcd)
    # -----------------------------------------------------------------------
    gaussians = GaussianModel(sh_degree=0)          # SH0 only
    scene     = Scene(dataset, gaussians, resolution_scales=[1.0])

    spatial_lr_scale = gaussians.spatial_lr_scale   # save before discarding GM

    if checkpoint:
        print(f"[Stage2] Loading checkpoint: {checkpoint}")
        # Checkpoints are PLY files; load via GaussianModel then transfer
        gaussians.load_ply(checkpoint, dataset.train_test_exp)

    # -----------------------------------------------------------------------
    # 1b. Hard-cap point cloud to 40M Gaussians before transfer
    # -----------------------------------------------------------------------
    MAX_GAUSSIANS = 30_000_000
    P = gaussians._xyz.shape[0]
    if P > MAX_GAUSSIANS:
        print(f"[Stage2] Downsampling {P:,} → {MAX_GAUSSIANS:,} Gaussians "
              f"(hard cap before CPU transfer)...")
        perm = torch.randperm(P, device=gaussians._xyz.device)[:MAX_GAUSSIANS]
        with torch.no_grad():
            gaussians._xyz          = gaussians._xyz[perm].contiguous()
            gaussians._features_dc  = gaussians._features_dc[perm].contiguous()
            gaussians._scaling      = gaussians._scaling[perm].contiguous()
            gaussians._rotation     = gaussians._rotation[perm].contiguous()
            gaussians._opacity      = gaussians._opacity[perm].contiguous()
        del perm
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 2. Transfer to CPUGaussianStore (GaussianModel GPU tensors no longer needed)
    # -----------------------------------------------------------------------
    print("[Stage2] Transferring Gaussians to CPU pinned store...")
    cpu_store = CPUGaussianStore(gaussians)

    # Free GaussianModel GPU memory
    del gaussians
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 2b. Build spatial block index + reorder cpu_store
    # -----------------------------------------------------------------------
    print("[Stage2] Building SpatialBlockIndex and reordering CPU store...")
    xyz_full    = cpu_store.xyz_cpu.numpy().copy()  # (P, 3) before reorder
    block_index = SpatialBlockIndex(xyz_full, resolution=10)
    cpu_store.reorder(block_index.spatial_order)

    # -----------------------------------------------------------------------
    # 3. Downsample point cloud for visibility precomputation
    # -----------------------------------------------------------------------
    print(f"[Stage2] Downsampling {cpu_store.N:,} → {ds_target:,} points for "
          f"visibility precomputation...")
    points_ds = downsample_points(torch.from_numpy(xyz_full), ds_target)

    # -----------------------------------------------------------------------
    # 4. VisibilityPrecomputer (Step 2)
    # -----------------------------------------------------------------------
    cameras_all = scene.train_cameras[1.0]          # CachedCamera list at scale 1.0
    N_cams      = len(cameras_all)

    print(f"[Stage2] Step 2: computing downsampled visibility for {N_cams} cameras...")
    vp         = VisibilityPrecomputer(cameras_all, points_ds, fov_margin=fov_margin)
    visible_ds = vp.compute()                       # (N, M) bool on CPU
    vp.free()

    # Assign block ids to each downsampled point
    ds_blocks = block_index.assign_blocks(points_ds.numpy())  # (M,) int32

    # -----------------------------------------------------------------------
    # 5. CameraBatchScheduler (Steps 3-7)
    # -----------------------------------------------------------------------
    decode_workers  = int(getattr(dataset, "camera_decode_workers", 8) or 0)
    decode_executor = ThreadPoolExecutor(max_workers=max(1, decode_workers))

    scheduler = CameraBatchScheduler(
        cameras    = cameras_all,
        visible_ds = visible_ds,
        batch_size = batch_size,
    )
    batch_infos = scheduler.build(
        ds_blocks         = ds_blocks,
        block_index       = block_index,
        slot_budget_bytes = int(slot_budget_gb * 1024 ** 3),
    )

    # -----------------------------------------------------------------------
    # 6. LR schedules (CPU-side, set on optimizer each step)
    # -----------------------------------------------------------------------
    xyz_lr_fn = get_expon_lr_func(
        lr_init       = opt.position_lr_init * spatial_lr_scale,
        lr_final      = opt.position_lr_final * spatial_lr_scale,
        lr_delay_mult = opt.position_lr_delay_mult,
        max_steps     = opt.position_lr_max_steps,
    )
    # Initial LR dict for fresh SGD optimizers created in GPUGaussianSlice
    lr_dict = {
        'xyz':      opt.position_lr_init * spatial_lr_scale,
        'f_dc':     opt.feature_lr,
        'scaling':  opt.scaling_lr,
        'rotation': opt.rotation_lr,
        'opacity':  opt.opacity_lr,
    }

    # -----------------------------------------------------------------------
    # 7. GaussianSwapBuffer
    # -----------------------------------------------------------------------
    print("[Stage2] Building GaussianSwapBuffer...")
    swap_buffer = GaussianSwapBuffer(
        cpu_store         = cpu_store,
        batch_infos       = batch_infos,
        cameras           = cameras_all,
        block_index       = block_index,
        slot_budget_bytes = int(slot_budget_gb * 1024 ** 3),
        batch_size        = batch_size,
        lr_dict           = lr_dict,
        decode_executor   = decode_executor,
    )
    swap_buffer.prefetch_initial()

    # -----------------------------------------------------------------------
    # 8. Misc setup
    # -----------------------------------------------------------------------
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(dataset.model_path)

    print(f"[Stage2] Starting training: {opt.iterations} total iterations, "
          f"batch_size={batch_size}")

    # -----------------------------------------------------------------------
    # 9. Training loop
    # -----------------------------------------------------------------------
    global_step = 0

    with trange(opt.iterations, desc="Stage2 training") as pbar:
        while global_step < opt.iterations:
            # Pop: returns (camera_views, gaussian_slice) for the active slot
            camera_views, gaussians = swap_buffer.pop()

            print(f"camera_views: {len(camera_views)}")
            print(f"gaussians: {gaussians.valid_length}")

            # 4 rounds × B cameras per slot — amortises slot-switch overhead
            for _round in range(4):
                if global_step >= opt.iterations:
                    break
                for cam in camera_views:
                    if global_step >= opt.iterations:
                        break

                    # Update xyz LR on the slice's fresh SGD optimizer
                    for pg in gaussians.optimizer.param_groups:
                        if pg['name'] == 'xyz':
                            pg['lr'] = xyz_lr_fn(global_step)

                    render_pkg = render(
                        cam, gaussians, pipe, background,
                        use_trained_exp=False,
                        separate_sh=False,
                    )
                    image = render_pkg["render"]

                    loss, Ll1 = build_loss(image, cam, background, opt)
                    loss.backward()

                    # Free rasterizer outputs immediately — render_pkg["viewspace_points"]
                    # has retain_grad() called inside render(), so its .grad (K×3 float32)
                    # would otherwise persist until the next render_pkg assignment.
                    del render_pkg, image

                    with torch.no_grad():
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    pbar.update(1)

                    # Logging
                    if tb_writer:
                        tb_writer.add_scalar('train/l1_loss',    Ll1.item(),  global_step)
                        tb_writer.add_scalar('train/total_loss', loss.item(), global_step)

                    del loss, Ll1

                    # Checkpoint (save current CPU state; finish_batch already flushed GPU)
                    if global_step in checkpoint_iterations:
                        _flush_and_save(cpu_store, dataset.model_path,
                                        global_step, is_checkpoint=True)

            # Writeback active slot → CPU, prefetch next batch → freed slot, swap
            swap_buffer.finish_batch()

            # Periodic eval
            if global_step in testing_iterations:
                _eval(scene, gaussians, pipe, background, tb_writer, global_step)

            # Save output PLY
            if global_step in saving_iterations:
                _flush_and_save(cpu_store, dataset.model_path,
                                global_step, is_checkpoint=False)

    # Final save
    _flush_and_save(cpu_store, dataset.model_path,
                    global_step, is_checkpoint=False)

    print(f"\n[Stage2] Training complete.")
    print(f"  Skipped over-budget batches: {swap_buffer.skipped_batches}")
    print(f"  Sample counts — min: {swap_buffer.sample_count.min()}, "
          f"max: {swap_buffer.sample_count.max()}, "
          f"mean: {swap_buffer.sample_count.mean():.1f}")

    decode_executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _flush_and_save(cpu_store, model_path, iteration, is_checkpoint):
    """Flush GPU→CPU (finish_batch already called in loop) and save PLY."""
    subdir = "checkpoint" if is_checkpoint else "point_cloud"
    path   = os.path.join(model_path, subdir, f"iteration_{iteration}", "point_cloud.ply")
    print(f"\n[Stage2] Saving {'checkpoint' if is_checkpoint else 'output'} "
          f"at iteration {iteration}...")
    save_cpu_store_ply(cpu_store, path)


def _eval(scene, gaussians, pipe, background, tb_writer, iteration):
    """Quick PSNR eval on test cameras (uses the current active GPUGaussianSlice)."""
    torch.cuda.empty_cache()
    test_cams = scene.getTestCameras(scale=1.0)
    l1_test, psnr_test, n = 0.0, 0.0, 0

    with torch.no_grad():
        try:
            while True:
                vc = test_cams.pop()
                img = render(vc, gaussians, pipe, background,
                             use_trained_exp=False, separate_sh=False)["render"]
                img = img.clamp(0, 1)
                gt  = image_to_float01(vc.original_image, img.device)
                l1_test   += l1_loss(img, gt).mean().double()
                psnr_test += psnr(img, gt).mean().double()
                n += 1
        except StopIteration:
            pass

    if n > 0:
        psnr_test /= n
        l1_test   /= n
        print(f"\n[ITER {iteration}] Test — L1: {l1_test:.4f}, PSNR: {psnr_test:.2f}")
        if tb_writer:
            tb_writer.add_scalar('eval/l1_loss', float(l1_test),   iteration)
            tb_writer.add_scalar('eval/psnr',    float(psnr_test), iteration)

    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def prepare_output(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", str(uuid.uuid4())[:10])
    os.makedirs(args.model_path, exist_ok=True)
    print(f"Output folder: {args.model_path}")
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))


if __name__ == "__main__":
    parser = ArgumentParser(description="Stage-2 training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip",               type=str,   default="127.0.0.1")
    parser.add_argument("--port",             type=int,   default=6009)
    parser.add_argument("--debug_from",       type=int,   default=-1)
    parser.add_argument("--detect_anomaly",   action="store_true", default=False)
    parser.add_argument("--test_iterations",  nargs="+",  type=int,
                        default=[7_000, 30_000])
    parser.add_argument("--save_iterations",  nargs="+",  type=int,
                        default=[7_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str,   default=None)
    parser.add_argument("--quiet",            action="store_true")

    # Stage-2 specific
    parser.add_argument("--ds_target",        type=int,   default=1_000_000,
                        help="Downsampled point count for visibility precomputation")
    parser.add_argument("--batch_size",       type=int,   default=8,
                        help="Cameras per Gaussian batch (B)")
    parser.add_argument("--slot_budget_gb",   type=float, default=4.0,
                        help="GPU memory budget per Gaussian slot (GB)")
    parser.add_argument("--fov_margin",       type=float, default=0.1,
                        help="FOV expansion margin for visibility frustum")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    prepare_output(args)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        dataset              = lp.extract(args),
        opt                  = op.extract(args),
        pipe                 = pp.extract(args),
        testing_iterations   = args.test_iterations,
        saving_iterations    = args.save_iterations,
        checkpoint_iterations= args.checkpoint_iterations,
        checkpoint           = args.start_checkpoint,
        ds_target            = args.ds_target,
        batch_size           = args.batch_size,
        slot_budget_gb       = args.slot_budget_gb,
        fov_margin           = args.fov_margin,
    )
    print("\nTraining complete.")
