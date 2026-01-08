import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pathlib import Path

import sys

import subprocess

import json

from argparse import Namespace

from train import main as train_main
from render import main as render_main


if __name__ == "__main__":

    dataset_path = Path("./DropGaussian_Data/nerf_llff_data")
    # dataset_path = Path("./DropGaussian_Data/mipnerf360")

    dropout_algorithm = "origin"
    dropout_algorithm = "indices"

    method_name = f"{dataset_path.name}-{dropout_algorithm}"

    # output_dir = Path("./output") / dataset_path.name
    output_dir = Path("./output") / method_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_list = [scene_path.name for scene_path in dataset_path.iterdir()]
    print(f"{scene_list = }")

    if dataset_path.name == "nerf_llff_data":
        view_list = [3, 6, 9]
    elif dataset_path.name == "mipnerf360":
        view_list = [12, 24]

    listen_port = 6910 \
        if dataset_path.name == "mipnerf360" else 6909

    repeat_number = 4

    resolution = 8

    resolution_output_dir = output_dir / f"resolution_{resolution}"
    resolution_output_dir.mkdir(parents=True, exist_ok=True)

    save_iterations = [6000, 8000, 10000]

    for view in view_list:
        print(f"{view = }")

        view_output_dir = resolution_output_dir / f"view_{view}"
        view_output_dir.mkdir(parents=True, exist_ok=True)

        for repeat_index in range(repeat_number):
            print(f"{repeat_index = }")

            repeat_output_dir = view_output_dir / f"output_{repeat_index}"
            repeat_output_dir.mkdir(parents=True, exist_ok=True)

            for scene_path in dataset_path.iterdir():
                print(f"{scene_path = }")
                scene_name = scene_path.name

                model_path = repeat_output_dir / f"{scene_name}"
                print(f"{model_path = }")
                subprocess.run([
                    sys.executable, "train.py",
                    "--source_path", scene_path,
                    "--model_path", model_path,
                    "--eval",
                    "--resolution", f"{resolution}",
                    "--n_views", f"{view}",
                    "--test_iterations", "5000", "6000", "8000", "10000",
                    "--save_iterations", *[str(it) for it in save_iterations],
                    "--port", str(listen_port),
                    "--dropout_algorithm", dropout_algorithm,
                ])
                # train_main(
                #     Namespace(
                #         source_path=str(scene_path),
                #         model_path=str(model_path),
                #         eval=True,
                #         resolution=resolution,
                #         n_views=view,

                #         ip="127.0.0.1",
                #         port=listen_port,
                #         debug_from=-1,
                #         detect_anomaly=False,
                #         test_iterations=[5000, 6000, 8000, 10000],
                #         save_iterations=save_iterations,
                #         quiet=False,
                #         checkpoint_iterations=[],
                #         start_checkpoint=None,
                #     )
                # )
                subprocess.run([
                    sys.executable, "render.py",
                    "--model_path", model_path,
                    "--resolution", f"{resolution}",
                    "--iteration", *[str(it) for it in save_iterations],
                    "--skip_train",
                ])
                # render_main(
                #     Namespace(
                #         model_path=str(model_path),
                #         resolution=resolution,

                #         iteration=save_iterations,
                #         skip_train=False,
                #         skip_test=False,
                #         quiet=False,
                #     )
                # )
            subprocess.run([
                sys.executable, "metric.py",
                "--path", repeat_output_dir,
                "--iteration", *[str(it) for it in save_iterations],
            ])
