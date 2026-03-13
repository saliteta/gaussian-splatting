#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from dataclasses import dataclass
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from pathlib import Path
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from typing import List
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.GPUImageBuffer import GPUImageBufferPacked
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


COARSE_TO_FINE_SCALES = (8.0, 4.0, 2.0, 1.0)


@dataclass(frozen=True)
class TrainingStage:
    index: int
    scale: float
    start_iteration: int
    end_iteration: int

    @property
    def label(self) -> str:
        return "1" if self.scale == 1.0 else f"/{int(self.scale)}"


def build_training_stages(total_iterations: int, stage_iterations: int) -> List[TrainingStage]:
    if total_iterations < 1:
        raise ValueError("Total iterations must be at least 1.")
    if stage_iterations < 1:
        raise ValueError("resolution_stage_iterations must be at least 1.")

    stages = []
    for idx, scale in enumerate(COARSE_TO_FINE_SCALES):
        start_iteration = idx * stage_iterations + 1
        if start_iteration > total_iterations:
            break

        is_last_scale = idx == len(COARSE_TO_FINE_SCALES) - 1
        end_iteration = total_iterations if is_last_scale else min(total_iterations, (idx + 1) * stage_iterations)
        stages.append(TrainingStage(idx, scale, start_iteration, end_iteration))

        if end_iteration >= total_iterations:
            break

    return stages


def get_stage_for_iteration(iteration: int, stages: List[TrainingStage]) -> TrainingStage:
    for stage in stages:
        if stage.start_iteration <= iteration <= stage.end_iteration:
            return stage
    return stages[-1]


def get_stage_local_iteration(iteration: int, stage: TrainingStage) -> int:
    return iteration - stage.start_iteration + 1


def describe_training_stages(stages: List[TrainingStage]):
    print("Coarse-to-fine schedule:")
    for stage in stages:
        print(
            f"  iterations {stage.start_iteration}-{stage.end_iteration}: resolution {stage.label}"
        )


def image_to_float01(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    if image.dtype == torch.uint8:
        return image.to(device=device, dtype=torch.float32) / 255.0
    return image.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)


def mask_to_float01(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    if mask.dtype == torch.uint8:
        return mask.to(device=device, dtype=torch.float32) / 255.0
    return mask.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)


def composite_with_background(rgb: torch.Tensor, alpha: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    return rgb * alpha + background * (1.0 - alpha)


def build_training_loss_images(
    rendered_image: torch.Tensor,
    viewpoint_cam,
    default_background: torch.Tensor,
    use_random_background: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = rendered_image.device
    gt_rgb = image_to_float01(viewpoint_cam.original_image, device=device)
    alpha_mask = mask_to_float01(viewpoint_cam.alpha_mask, device=device)

    if use_random_background:
        random_background = torch.rand_like(gt_rgb)
        pred_image = rendered_image.to(torch.float32) + random_background * (1.0 - alpha_mask)
        gt_image = composite_with_background(gt_rgb, alpha_mask, random_background)
        return pred_image, gt_image

    default_background_image = default_background[:, None, None].expand_as(gt_rgb)
    gt_image = composite_with_background(gt_rgb, alpha_mask, default_background_image)
    return rendered_image.to(torch.float32), gt_image

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    training_stages = build_training_stages(opt.iterations, opt.resolution_stage_iterations)
    describe_training_stages(training_stages)
    scene = Scene(dataset, gaussians, resolution_scales=[stage.scale for stage in training_stages])
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    black_background = torch.zeros_like(background)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    next_iteration = first_iter + 1
    current_stage = get_stage_for_iteration(next_iteration, training_stages)
    if current_stage.index > 0 and next_iteration == current_stage.start_iteration:
        gaussians.reset_densification_state()
    print(
        f"Starting training at resolution {current_stage.label} "
        f"(iterations {current_stage.start_iteration}-{current_stage.end_iteration})."
    )
    if opt.random_background:
        print("Training loss uses per-pixel random background compositing.")

    camera_loader: GPUImageBufferPacked = scene.getTrainCameras(current_stage.scale)
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iteration_stage = get_stage_for_iteration(iteration, training_stages)
        if iteration_stage.index != current_stage.index:
            current_stage = iteration_stage
            del camera_loader
            torch.cuda.empty_cache()
            camera_loader = scene.getTrainCameras(current_stage.scale)
            gaussians.reset_densification_state()
            print(
                f"\n[ITER {iteration}] Switched to resolution {current_stage.label}. "
                f"Densification window restarted."
            )
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        viewpoint_cam = camera_loader.pop()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_background = black_background if opt.random_background else background
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            render_background,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=SPARSE_ADAM_AVAILABLE,
        )
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        image_for_loss, gt_for_loss = build_training_loss_images(
            image,
            viewpoint_cam,
            background,
            opt.random_background,
        )

        Ll1 = l1_loss(image_for_loss, gt_for_loss)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image_for_loss.unsqueeze(0), gt_for_loss.unsqueeze(0))
        else:
            ssim_value = ssim(image_for_loss, gt_for_loss)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Res": current_stage.label})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            #training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            stage_local_iteration = get_stage_local_iteration(iteration, current_stage)
            stage_length = current_stage.end_iteration - current_stage.start_iteration + 1
            stage_densify_until = min(opt.densify_until_iter, stage_length)
            if stage_local_iteration <= stage_densify_until:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if stage_local_iteration > opt.densify_from_iter and stage_local_iteration % opt.densification_interval == 0:
                    size_threshold = 20 if stage_local_iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if stage_local_iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and stage_local_iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras(scale=1.0)}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_rgb = image_to_float01(viewpoint.original_image, device=image.device)
                    alpha_mask = mask_to_float01(viewpoint.alpha_mask, device=image.device)
                    gt_image = composite_with_background(
                        gt_rgb,
                        alpha_mask,
                        renderArgs[1][:, None, None].expand_as(gt_rgb),
                    )
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    _ = prepare_output_and_logger(args)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
