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
import logging
from glob import glob
from argparse import ArgumentParser
import shutil
import re

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("metrics")
parser.add_argument("--path", "-s", required=True, type=str)
parser.add_argument("--iteration", "-i", nargs="+", default=[10000], type=int)
args = parser.parse_args()

dir_lst = glob(os.path.join(args.path, '*'))

iterations = args.iteration if isinstance(args.iteration, (list, tuple)) else [args.iteration]

for it in iterations:
    PSNR = 0.0
    SSIM = 0.0
    LPIPS = 0.0
    count = 0

    for d in dir_lst:
        metrics_file = os.path.join(d, f'metrics_{it}.txt')
        if not os.path.exists(metrics_file):
            logging.warning(f"Missing metrics file: {metrics_file}")
            continue

        with open(metrics_file, 'r') as f:
            l = f.readline()
            psnr = re.sub(r'[^0-9.\-eE]', '', l)
            print(d, psnr)
            PSNR += float(psnr)

            l = f.readline()
            ssim = re.sub(r'[^0-9.\-eE]', '', l)
            SSIM += float(ssim)

            l = f.readline()
            lpips = re.sub(r'[^0-9.\-eE]', '', l)
            LPIPS += float(lpips)

        count += 1

    if count == 0:
        logging.warning(f"No metrics found for iteration {it} in {args.path}")
        continue

    PSNR /= count
    SSIM /= count
    LPIPS /= count

    metric_path = os.path.join(args.path, f'metrics_mean_{it}.txt')
    with open(metric_path, 'w') as f:
        f.write(f'PSNR : {PSNR}\n')
        f.write(f'SSIM : {SSIM}\n')
        f.write(f'LPIPS : {LPIPS}\n')

    print(f'Iteration {it}:')
    print(PSNR)
    print(SSIM)
    print(LPIPS)