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
import logging
import os
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
from torch import nn
import copy

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from debug import SimpleLogger


def training(
    dataset_args,
    optimize_args,
    pipeline_args,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    args: Namespace = None,
):
    logger = SimpleLogger(
        filename=os.path.join(args.model_path, "training.log"),
        level=logging.DEBUG,
    )
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset_args)
    gaussians = GaussianModel(dataset_args.sh_degree)
    scene = Scene(dataset_args, gaussians)
    gaussians.training_setup(optimize_args)
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, optimize_args)

    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    train_cameras_all = scene.getTrainCameras().copy()
    from utils.camera_utils import (
        print_camera_list,
    )

    print_camera_list(
        train_cameras_all,
        save_image_dir=os.path.join(
            args.model_path, f"./drop_gaussian_train_camera_images/"
        ),
    )
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(
        range(first_iter, optimize_args.iterations), desc="Training progress"
    )
    first_iter += 1
    bg_mask = None
    loss_accum = 0
    pseudo_stack = None
    for iteration in range(first_iter, optimize_args.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            # print(f"[Training] Replenishing viewpoint stack at iteration {iteration}.")
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(
            randint(0, len(viewpoint_stack) - 1)
        )
        gt_image = viewpoint_cam.original_image.cuda()

        # Render
        if (iteration - 1) == debug_from:
            pipeline_args.debug = True

        if iteration % 200 == 0:
            logger.debug(f"[{iteration}] {gaussians.get_xyz.shape[0] = }")

        bg = (
            torch.rand((3), device="cuda")
            if optimize_args.random_background
            else background
        )
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipeline_args,
            bg,
            is_train=True,
            iteration=iteration,
            dropout_algorithm=args.dropout_algorithm,
            logger=logger,
        )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = Ll1 + optimize_args.lambda_dssim * (1.0 - ssim_value)

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if iteration > optimize_args.densify_from_iter:
                loss_accum += loss.clone().detach().item()

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == optimize_args.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                dataset_args,
                tb_writer,
                iteration,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipeline_args, background),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < optimize_args.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > optimize_args.densify_from_iter
                    and iteration % optimize_args.densification_interval == 0
                ):
                    size_threshold = None
                    gaussians.densify_and_prune(
                        optimize_args.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % optimize_args.opacity_reset_interval == 0 or (
                    dataset_args.white_background
                    and iteration == optimize_args.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < optimize_args.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )


def prepare_output_and_logger(
    args,
):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    args,
    tb_writer,
    iteration,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        # tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(
            "train_loss_patches/total_loss", loss.item(), iteration
        )
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(len(scene.getTrainCameras()))
                ],
            },
        )

        for config in validation_configs:
            render_path = os.path.join(
                args.model_path,
                config["name"],
                "ours_{}".format(iteration),
                "renders",
            )
            gts_path = os.path.join(
                args.model_path,
                config["name"],
                "ours_{}".format(iteration),
                "gt",
            )
            os.makedirs(render_path, exist_ok=True)
            os.makedirs(gts_path, exist_ok=True)

            metric_save_path = os.path.join(
                args.model_path, config["name"], "ours_{}".format(iteration)
            )
            os.makedirs(metric_save_path, exist_ok=True)

            metrics = {}
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_pkg = renderFunc(
                        viewpoint, scene.gaussians, *renderArgs
                    )
                    image = render_pkg["render"]
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(
                                    viewpoint.image_name
                                ),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    torchvision.utils.save_image(
                        image,
                        os.path.join(
                            render_path, viewpoint.image_name + ".png"
                        ),
                    )
                    torchvision.utils.save_image(
                        gt_image,
                        os.path.join(gts_path, viewpoint.image_name + ".png"),
                    )
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                metrics["l1"] = float(l1_test.item())
                metrics["psnr"] = float(psnr_test.item())
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                logging.info(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss",
                        l1_test,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr",
                        psnr_test,
                        iteration,
                    )
            with open(os.path.join(metric_save_path, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram",
                scene.gaussians.get_opacity,
                iteration,
            )
            tb_writer.add_scalar(
                "total_points",
                scene.gaussians.get_xyz.shape[0],
                iteration,
            )
        torch.cuda.empty_cache()


import copy
import socket  # added


def _pick_available_port(
    ip: str,
    preferred_port: int,
) -> int:
    """Return preferred_port if free; otherwise return a nearby available port, falling back to an ephemeral one."""

    def can_bind(port: int) -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Allow quick reuse during rapid restarts
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((ip, port))
            return True
        except OSError:
            return False
        finally:
            s.close()

    if can_bind(preferred_port):
        return preferred_port

    # Try a small range above the preferred port
    for p in range(preferred_port + 1, preferred_port + 100):
        if can_bind(p):
            return p

    # Fallback: let OS choose an ephemeral port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((ip, 0))
        return s.getsockname()[1]
    finally:
        s.close()


def build_parser():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[
            5000,
            10000,
        ],
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[10000],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--checkpoint_iterations",
        nargs="+",
        type=int,
        default=[],
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument(
        "--dropout_algorithm",
        type=str,
        default="largest",
        choices=[
            "origin",
            "indices",
        ],
    )
    return parser


def main(
    arg_list=None,
):
    parser = build_parser()
    if arg_list is not None:
        parsed_args = parser.parse_args(arg_list)
    else:
        parsed_args = parser.parse_args(sys.argv[1:])

    args = parsed_args

    print(f"{args = }")

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(getattr(args, "quiet", False))

    # Start GUI server, configure and run training
    ip = getattr(args, "ip", "127.0.0.1")
    requested_port = getattr(args, "port", 6009)
    selected_port = _pick_available_port(ip, requested_port)
    if selected_port != requested_port:
        print(
            f"Requested port {requested_port} is occupied. Using available port {selected_port} instead."
        )
    network_gui.init(ip, selected_port)
    torch.autograd.set_detect_anomaly(getattr(args, "detect_anomaly", False))

    # Build param helpers (no argparse here; we just extract from provided args)
    from argparse import ArgumentParser

    lp = ModelParams(ArgumentParser(add_help=False))
    op = OptimizationParams(ArgumentParser(add_help=False))
    pp = PipelineParams(ArgumentParser(add_help=False))

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args,
    )

    print("\nTraining complete.")
    network_gui.destroy()


if __name__ == "__main__":
    main()
