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

import json
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from scene.cameras import Camera
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.general_utils import PILtoTorch
###
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips


def render_set(
    model_path: str,
    name: str,
    iteration: int,
    views: list[Camera],
    gaussians: GaussianModel,
    pipeline: PipelineParams,
    background,
    resol=1,
):
    render_path = os.path.join(
        model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(
        model_path, name, f"ours_{iteration}", "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    PSNR = []
    SSIM = []
    LPIPS = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :].cuda()
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        torchvision.utils.save_image(
            rendering,
            os.path.join(
                render_path, f'{view.image_name}.png',
            ),
        )
        torchvision.utils.save_image(
            gt,
            os.path.join(
                gts_path, f'{view.image_name}.png',
            ),
        )
        PSNR.append(psnr(rendering.unsqueeze(0), gt.unsqueeze(0)))
        SSIM.append(ssim(rendering.unsqueeze(0), gt.unsqueeze(0)))
        LPIPS.append(lpips(rendering.unsqueeze(
            0), gt.unsqueeze(0), net_type='vgg'))

    psnr_mean = torch.tensor(PSNR).mean().item()
    ssim_mean = torch.tensor(SSIM).mean().item()
    lpips_mean = torch.tensor(LPIPS).mean().item()

    print('PSNR : {:>12.7f}'.format(psnr_mean))
    print('SSIM : {:>12.7f}'.format(ssim_mean))
    print('LPIPS : {:>12.7f}'.format(lpips_mean))

    with open(os.path.join(model_path, f'metrics_{name}_{iteration}.json'), 'w') as f:
        json.dump({
            'PSNR': psnr_mean,
            'SSIM': ssim_mean,
            'LPIPS': lpips_mean,
        },
            f,
            indent=4,
        )


def render_sets(
    dataset_params: ModelParams,
    iteration: int,
    pipeline_params: PipelineParams,
    skip_train: bool,
    skip_test: bool,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset_params.sh_degree)
        scene = Scene(
            dataset_params, gaussians,
            load_iteration=iteration, shuffle=False,
        )

        bg_color = [1, 1, 1] if dataset_params.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset_params.model_path, "train", scene.loaded_iter,
                scene.getTrainCameras(), gaussians, pipeline_params, background,
            )

        if not skip_test:
            render_set(
                dataset_params.model_path, "test", scene.loaded_iter,
                scene.getTestCameras(), gaussians, pipeline_params, background,
            )


def build_parser():
    parser = ArgumentParser(description="Testing script parameters")
    ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    parser.add_argument("--iteration", nargs="+", default=[-1], type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(arg_list=None):
    parser = build_parser()
    args = get_combined_args(
        parser,
        cmdlne_string=arg_list,
    )
    print(f"{args = }")

    print("Rendering " + args.model_path)
    safe_state(getattr(args, "quiet", False))

    # Build param helpers without using argparse (use provided args)
    model = ModelParams(ArgumentParser(add_help=False), sentinel=True)
    pipeline = PipelineParams(ArgumentParser(add_help=False))

    dataset_args = model.extract(args)
    print(f"{dataset_args = }")
    pipeline_args = pipeline.extract(args)
    print(f"{pipeline_args = }")

    iterations = args.iteration if isinstance(
        args.iteration, (list, tuple)) else [args.iteration]
    skip_train = getattr(args, "skip_train", False)
    skip_test = getattr(args, "skip_test", False)

    for it in iterations:
        render_sets(
            dataset_args,
            it,
            pipeline_args,
            skip_train,
            skip_test,
        )


if __name__ == "__main__":
    main()
