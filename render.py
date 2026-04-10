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

import torch
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import GaussianModel

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.uint8:
        return t.permute(1, 2, 0).cpu().numpy()
    return t.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()


def _save_jpeg(arr: np.ndarray, path: str, quality: int = 95) -> None:
    Image.fromarray(arr).save(path, format="JPEG", quality=quality)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path    = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path,    exist_ok=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, pipeline, background,
                               use_trained_exp=train_test_exp,
                               separate_sh=separate_sh)["render"]
            gt = view.original_image[0:3]

            if train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]
                gt        = gt[...,        gt.shape[-1]        // 2:]

            pool.submit(_save_jpeg, _to_uint8(rendering),
                        os.path.join(render_path, f"{idx:05d}.jpg"))
            pool.submit(_save_jpeg, _to_uint8(gt),
                        os.path.join(gts_path,    f"{idx:05d}.jpg"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color   = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                       scene.train_cameras[1.0], gaussians, pipeline,
                       background, dataset.train_test_exp, separate_sh)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                       scene.test_cameras[1.0], gaussians, pipeline,
                       background, dataset.train_test_exp, separate_sh)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test",  action="store_true")
    parser.add_argument("--quiet",      action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)
