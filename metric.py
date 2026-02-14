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
import os
import logging
from glob import glob
from argparse import ArgumentParser
import re
from pathlib import Path


def compute_mean_metrics(
    path: str,
    name: str,
    iterations,
    scenes=[],
):
    def get_scene_dir_list():
        return [os.path.join(path, scene) for scene in scenes]

    scene_dir_list = get_scene_dir_list()

    iterations = (
        iterations if isinstance(iterations, (list, tuple)) else [iterations]
    )

    for it in iterations:
        PSNR = 0.0
        SSIM = 0.0
        LPIPS = 0.0
        count = 0

        for d in scene_dir_list:
            metrics_file = os.path.join(d, f"metrics_{name}_{it}.json")
            if not os.path.exists(metrics_file):
                logging.warning(f"Missing metrics file: {metrics_file}")
                continue

            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                PSNR += metrics.get("PSNR", 0.0)
                SSIM += metrics.get("SSIM", 0.0)
                LPIPS += metrics.get("LPIPS", 0.0)

            count += 1

        if count == 0:
            logging.warning(f"No metrics found for iteration {it} in {path}")
            continue

        PSNR /= count
        SSIM /= count
        LPIPS /= count

        metric_path = os.path.join(path, f"metrics_mean_{name}_{it}.json")
        with open(metric_path, "w") as f:
            json.dump(
                {
                    "PSNR": PSNR,
                    "SSIM": SSIM,
                    "LPIPS": LPIPS,
                },
                f,
                indent=4,
            )

        print(f"Iteration {it}:")
        print(PSNR)
        print(SSIM)
        print(LPIPS)


def build_parser():
    parser = ArgumentParser(description="Compute mean metrics across scenes")
    parser.add_argument("--path", "-s", required=True, type=str)
    parser.add_argument("--name", "-n", required=True, type=str)
    parser.add_argument(
        "--iteration", "-i", nargs="+", default=[10000], type=int
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--scenes", nargs="+", default=[])
    return parser


def main(
    arg_list=None,
):
    parser = build_parser()
    if arg_list is not None:
        args = parser.parse_args(arg_list)
    else:
        args = parser.parse_args()
    logging.basicConfig(
        level=(
            logging.ERROR if getattr(args, "quiet", False) else logging.WARNING
        ),
        format="%(levelname)s: %(message)s",
    )
    compute_mean_metrics(
        args.path,
        args.name,
        args.iteration,
        args.scenes,
    )


if __name__ == "__main__":
    main()
