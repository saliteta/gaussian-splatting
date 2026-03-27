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


class FusedLoss(torch.autograd.Function):
    """
    Fused L1 + SSIM loss with minimal backward memory footprint.

    Default autograd saves between forward and backward:
        pred      ~100 MB  (saved by L1 and SSIM backward nodes)
        gt_image  ~100 MB  (saved by SSIM backward node)
        SSIM intermediates  ~900 MB  (mu1, mu2, sigma*, etc.)
        ─────────────────────────────────────────────────────
        Total     ~1100 MB held until loss.backward() completes

    This Function saves only:
        int8 sign(pred - gt)   25 MB   (4× smaller than float32 diff)
        ssim_grad w.r.t. pred  100 MB  (precomputed via mini-backward in forward)
        ─────────────────────────────────────────────────────
        Total     ~125 MB  — ~9× reduction

    The SSIM gradient is computed with a temporary sub-graph inside forward():
    all SSIM intermediates are freed before the main loss.backward() is called.
    gt_image is never retained in the autograd graph.
    """

    @staticmethod
    def forward(ctx, pred, gt_image, lambda_dssim):
        # ---- L1 ----
        diff    = pred - gt_image                      # (3,H,W) float32 — temporary
        l1_sign = diff.sign().to(torch.int8)           # save as int8: 4× smaller
        l1_val  = diff.abs().mean()
        del diff                                        # free immediately

        # ---- SSIM: mini-forward + backward inside a throw-away sub-graph ----
        # Running backward here computes d(SSIM)/d(pred) and frees all SSIM
        # intermediates before the main loss.backward() runs.
        pred_leaf = pred.detach().requires_grad_(True)
        with torch.enable_grad():
            if FUSED_SSIM_AVAILABLE:
                sv = fused_ssim(pred_leaf.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                sv = ssim(pred_leaf, gt_image)
            if sv.numel() > 1:
                sv = sv.mean()
            sv.backward()                              # fills pred_leaf.grad
        ssim_grad   = pred_leaf.grad                   # (3,H,W) float32 — only survivor
        ssim_scalar = sv.item()
        del pred_leaf, sv                              # free sub-graph + all intermediates

        # Save ONLY int8 sign + ssim_grad — gt_image is NOT saved
        ctx.save_for_backward(l1_sign, ssim_grad)
        ctx.lambda_dssim = lambda_dssim
        ctx.numel = float(pred.numel())

        loss_val = (1.0 - lambda_dssim) * l1_val.item() + lambda_dssim * (1.0 - ssim_scalar)
        return pred.new_tensor(loss_val), l1_val.detach()

    @staticmethod
    def backward(ctx, grad_loss, _grad_l1):
        l1_sign, ssim_grad = ctx.saved_tensors
        lam   = ctx.lambda_dssim
        numel = ctx.numel
        # d(loss)/d(pred) = (1-lam)*sign(pred-gt)/N  +  lam*(-d(SSIM)/d(pred))
        grad_pred = grad_loss * (
            (1.0 - lam) * l1_sign.float() / numel
            - lam * ssim_grad
        )
        return grad_pred, None, None   # gt_image, lambda_dssim → no gradient


def build_loss(rendered_image, viewpoint_cam, background, opt):
    device   = rendered_image.device
    gt_rgb   = image_to_float01(viewpoint_cam.original_image, device)
    alpha    = mask_to_float01(viewpoint_cam.alpha_mask, device)
    bg_img   = background[:, None, None].expand_as(gt_rgb)
    gt_image = gt_rgb * alpha + bg_img * (1.0 - alpha)
    del gt_rgb, alpha                                  # free 133 MB before FusedLoss

    loss, Ll1 = FusedLoss.apply(
        rendered_image.to(torch.float32),
        gt_image,
        opt.lambda_dssim,
    )
    del gt_image                                       # not retained by FusedLoss — free now
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


def _ravel_hash(arr: np.ndarray) -> np.ndarray:
    """Fortran-order hash for integer coordinate rows."""
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1
    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def voxel_downsample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Voxel-grid downsample: one random point kept per occupied voxel.
    Returns a boolean index array into xyz (or use xyz[mask]).
    """
    discrete = np.floor(xyz / voxel_size).astype(np.int64)
    key      = _ravel_hash(discrete)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, _, count = np.unique(key_sort, return_inverse=True, return_counts=True)
    idx_select  = np.cumsum(np.insert(count, 0, 0)[:-1]) + np.random.randint(0, count.max(), count.size) % count
    return idx_sort[idx_select]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint,
             voxel_size: float = 0.08,
             iou_sample: int = 500_000,
             batch_size: int = 8,
             slot_budget_gb: float = 2.0,
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
    # 3. Downsample point cloud — two levels:
    #    points_ds  (voxel, 16M): used for ds_blocks → accurate block coverage
    #    points_iou (random subset of points_ds, ~500K): used for visibility
    #               IoU / KNN only — spatial approximation is sufficient
    # -----------------------------------------------------------------------
    print(f"[Stage2] Voxel-downsampling {cpu_store.N:,} points "
          f"(voxel_size={voxel_size}) for block assignment...")
    ds_idx    = voxel_downsample(xyz_full, voxel_size)
    points_ds = xyz_full[ds_idx]                       # (M_full, 3) numpy, for ds_blocks
    print(f"[Stage2] Kept {len(ds_idx):,} points after voxel downsample.")

    # Coarse subsample for visibility / IoU (no spatial precision needed)
    M_full = len(points_ds)
    if M_full > iou_sample:
        iou_idx   = np.random.choice(M_full, size=iou_sample, replace=False)
        points_iou = points_ds[iou_idx]
        print(f"[Stage2] IoU subsample: {iou_sample:,} points from {M_full:,} "
              f"(ratio {iou_sample/M_full:.2%})")
    else:
        points_iou = points_ds
        print(f"[Stage2] IoU subsample: using all {M_full:,} points (below iou_sample cap).")

    # -----------------------------------------------------------------------
    # 4. VisibilityPrecomputer — runs on coarse IoU points only
    # -----------------------------------------------------------------------
    cameras_all = scene.train_cameras[1.0]          # CachedCamera list at scale 1.0
    N_cams      = len(cameras_all)

    print(f"[Stage2] Computing visibility for {N_cams} cameras "
          f"on {len(points_iou):,} IoU points...")
    vp         = VisibilityPrecomputer(cameras_all,
                                       torch.from_numpy(points_iou),
                                       fov_margin=fov_margin)
    visible_ds = vp.compute()   # (N, ceil(M_iou/8)) uint8, bit-packed
    M_iou      = vp.M
    vp.free()

    # Block assignment: use IoU points (500K) so dimensions match visible_ds.
    # Block IDs (0..999) are the same space regardless of which point set is
    # used — block_starts/block_ends still give accurate full-res Gaussian counts.
    # With 500K random points across 16M, every non-trivial block gets sampled.
    ds_blocks = block_index.assign_blocks(points_iou)   # (M_iou,) int32

    # -----------------------------------------------------------------------
    # 5. CameraBatchScheduler (Steps 3-7)
    # -----------------------------------------------------------------------
    decode_workers  = int(getattr(dataset, "camera_decode_workers", 8) or 0)
    decode_executor = ThreadPoolExecutor(max_workers=max(1, decode_workers))

    scheduler = CameraBatchScheduler(
        cameras    = cameras_all,
        visible_ds = visible_ds,
        M          = M_iou,
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
    # Each Gaussian is in the active GPU batch only a fraction of all steps.
    # The xyz LR exponential decay is keyed on global_step, so it decays
    # 1/coverage_ratio × too fast relative to per-Gaussian real gradient steps.
    # We correct by scaling position_lr_max_steps up by the same factor so the
    # LR at global_step matches what the Gaussian actually deserves.
    total_gaussian_slots = sum(info.n_gaussians for info in batch_infos.values())
    n_valid_batches      = len([i for i in batch_infos
                                if batch_infos[i].n_gaussians
                                   <= int(slot_budget_gb * 1024**3) // 56])
    avg_batch_gaussians  = total_gaussian_slots / max(1, len(batch_infos))
    batch_coverage       = avg_batch_gaussians / max(1, cpu_store.N)
    lr_steps_scale       = max(1.0, 1.0 / batch_coverage)
    print(f"[Stage2] Batch coverage: {batch_coverage:.1%}  "
          f"→ scaling position_lr_max_steps by {lr_steps_scale:.1f}×")

    xyz_lr_fn = get_expon_lr_func(
        lr_init       = opt.position_lr_init * spatial_lr_scale,
        lr_final      = opt.position_lr_final * spatial_lr_scale,
        lr_delay_mult = opt.position_lr_delay_mult,
        max_steps     = int(opt.position_lr_max_steps * lr_steps_scale),
    )
    # LR dict passed to GPUGaussianSlice; Adam state (exp_avg/exp_avg_sq/step)
    # is persisted in CPUGaussianStore and restored on each slot swap.
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
    slot_steps  = 0   # optimizer steps taken on the current GPU slot

    # Sorted milestone queues — fire once when global_step first crosses each value.
    # Using sorted lists + a low-water-mark pointer so we never fire twice even if
    # global_step jumps over the exact milestone (which can happen because we
    # increment by 1 per camera, and slot boundaries may not align with milestones).
    pending_saves         = sorted(set(saving_iterations))
    pending_checks        = sorted(set(checkpoint_iterations))
    pending_evals         = sorted(set(testing_iterations))

    def _fire_crossed(pending: list, step: int, action) -> None:
        """Pop and call action(ms) for every milestone ms <= step."""
        while pending and pending[0] <= step:
            action(pending.pop(0))

    with trange(opt.iterations, desc="Stage2 training") as pbar:
        while global_step < opt.iterations:
            camera_views, gaussians = swap_buffer.pop()

            # 4 rounds × B cameras per slot — amortises slot-switch overhead
            for _round in range(4):
                if global_step >= opt.iterations:
                    break

                # Update xyz LR once per round (not per camera) to reduce Python
                # overhead inside the tight per-camera GPU loop.
                for pg in gaussians.optimizer.param_groups:
                    if pg['name'] == 'xyz':
                        pg['lr'] = xyz_lr_fn(global_step)

                n_this_round = 0
                for cam in camera_views:
                    if global_step >= opt.iterations:
                        break

                    render_pkg = render(
                        cam, gaussians, pipe, background,
                        use_trained_exp=False,
                        separate_sh=False,
                    )
                    image = render_pkg["render"]

                    loss, Ll1 = build_loss(image, cam, background, opt)

                    loss.backward()
                    del render_pkg, image

                    with torch.no_grad():
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)

                    if tb_writer:
                        tb_writer.add_scalar('train/l1_loss',    Ll1.item(),  global_step)
                        tb_writer.add_scalar('train/total_loss', loss.item(), global_step)
                    del loss, Ll1

                    global_step  += 1
                    slot_steps   += 1
                    n_this_round += 1

                # Batch tqdm update once per round
                pbar.update(n_this_round)

            # --- Milestone checks after the slot (not inside the per-camera loop) ---
            # Fire saves/checkpoints/evals for any milestone we've crossed since the
            # last slot.  Each milestone fires at most once (popped from the queue).
            _fire_crossed(pending_checks, global_step,
                          lambda ms: _flush_and_save(cpu_store, dataset.model_path,
                                                     ms, is_checkpoint=True))
            _fire_crossed(pending_saves,  global_step,
                          lambda ms: _flush_and_save(cpu_store, dataset.model_path,
                                                     ms, is_checkpoint=False))
            _fire_crossed(pending_evals,  global_step,
                          lambda ms: _eval(scene, gaussians, pipe, background,
                                          tb_writer, ms))

            # Read current LRs from the active optimizer (xyz may have changed via schedule)
            current_lr = {pg['name']: pg['lr'] for pg in gaussians.optimizer.param_groups}
            swap_buffer.finish_batch(
                global_step     = global_step,
                n_steps         = slot_steps,
                current_lr_dict = current_lr,
            )
            slot_steps = 0

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
    if test_cams is None or len(getattr(test_cams, 'cameras', [])) == 0:
        print(f"[ITER {iteration}] No test cameras available, skipping eval.")
        return
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
    parser.add_argument("--voxel_size",       type=float, default=0.08,
                        help="Voxel size (metres) for block-assignment point cloud downsampling")
    parser.add_argument("--iou_sample",       type=int,   default=500_000,
                        help="Points randomly subsampled from voxel set for IoU/KNN visibility (coarse)")
    parser.add_argument("--batch_size",       type=int,   default=8,
                        help="Cameras per Gaussian batch (B)")
    parser.add_argument("--slot_budget_gb",   type=float, default=1.5,
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
        checkpoint              = args.start_checkpoint,
        voxel_size              = args.voxel_size,
        iou_sample           = args.iou_sample,
        batch_size           = args.batch_size,
        slot_budget_gb       = args.slot_budget_gb,
        fov_margin           = args.fov_margin,
    )
    print("\nTraining complete.")
