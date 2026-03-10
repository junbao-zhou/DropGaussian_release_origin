from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
import os

from pathlib import Path

import sys

import subprocess
import concurrent.futures as cf
import multiprocessing as mp
import inspect


def get_gpu_id(
    gpu_id_list: list[int],
    target_index: int,
) -> int:
    gpu_id = gpu_id_list[target_index % len(gpu_id_list)]
    return gpu_id


def _print_gpu_state(
    gpu_id: int,
    tag: str,
):
    """Best-effort console check of GPU memory + active compute processes."""

    print(f"\n[{tag}] nvidia-smi (gpu {gpu_id})")
    subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    print("")


def init_worker_gpu(
    gpu_id: int,
):
    # Stable device ordering + restrict visibility to a single GPU for this worker process
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Print PID so you can confirm a fresh process per task when max_tasks_per_child=1
    print(
        f"[worker init] pid={os.getpid()} gpu_id={gpu_id} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    _print_gpu_state(gpu_id, "worker init")


def _make_executor(
    max_workers: int,
    initializer: Callable | None = None,
    initargs: tuple | None = None,
) -> cf.ProcessPoolExecutor:
    ctx = mp.get_context("spawn")
    kwargs = {"max_workers": max_workers, "mp_context": ctx}

    # If supported (Python 3.11+), ensure each task gets a fresh process.
    # This improves CUDA resource release behavior and avoids cross-task CUDA reuse pitfalls.
    sig = inspect.signature(cf.ProcessPoolExecutor)
    if "max_tasks_per_child" in sig.parameters:
        print(
            "Using max_tasks_per_child=1 for better CUDA resource management."
        )
        kwargs["max_tasks_per_child"] = 1

    if initializer is not None:
        kwargs["initializer"] = initializer
    if initargs is not None:
        kwargs["initargs"] = initargs

    return cf.ProcessPoolExecutor(**kwargs)


def run_multiple_tasks_multiple_gpus(
    gpu_id_list: list[int],
    func_list: list[Callable],
    arg_list: list[list] | None = None,
) -> list:
    if arg_list is None:
        arg_list = [[]] * len(func_list)

    results = []

    with ExitStack() as stack:
        executors = [
            stack.enter_context(
                _make_executor(
                    max_workers=1,
                    initializer=init_worker_gpu,
                    initargs=(gpu_id,),
                )
            )
            for gpu_id in gpu_id_list
        ]
        future_to_gpu: dict[cf.Future, int] = {}
        for func_id, (func, args) in enumerate(zip(func_list, arg_list)):
            gpu_id = gpu_id_list[func_id % len(gpu_id_list)]
            executor = executors[func_id % len(executors)]
            fut = executor.submit(func, *args)
            future_to_gpu[fut] = gpu_id

        for completed_future in cf.as_completed(future_to_gpu.keys()):
            gpu_id = future_to_gpu[completed_future]
            results.append(completed_future.result())
            _print_gpu_state(gpu_id, "completed")

    return results


def scene_train_and_render(
    train_arg_list: list[str],
    render_arg_list: list[str],
):
    from train import main as train_main
    from render import main as render_main

    train_main(
        arg_list=train_arg_list,
    )
    render_main(
        arg_list=render_arg_list,
    )


def main():
    GPU_ID_LIST = [0, 1, 2, 3, 4, 5, 6, 7]

    dataset_path = Path("./DropGaussian_Data/nerf_llff_data")
    # dataset_path = Path("./DropGaussian_Data/mipnerf360")

    output_dir = Path("./output") / dataset_path.name
    output_dir = Path("./output_debug") / dataset_path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_list = [scene_path.name for scene_path in dataset_path.iterdir()]
    # scene_list = ["fortress", "leaves", "trex"]
    print(f"{scene_list = }")

    if dataset_path.name == "nerf_llff_data":
        view_list = [3, 6, 9]
        view_list = [3]
    elif dataset_path.name == "mipnerf360":
        view_list = [12, 24]

    base_listen_port = 6910 if dataset_path.name == "mipnerf360" else 6909

    repeat_number = 4
    # repeat_number = 1

    resolution = 8

    resolution_output_dir = output_dir / f"resolution_{resolution}"
    resolution_output_dir.mkdir(parents=True, exist_ok=True)

    save_iterations = [6000, 8000, 10000]

    dropout_algorithm = "origin"
    dropout_algorithm = "indices"

    method_name = f"{dropout_algorithm}"

    for view in view_list:
        print(f"{view = }")

        view_output_dir = resolution_output_dir / f"view_{view}"
        view_output_dir.mkdir(parents=True, exist_ok=True)

        method_output_dir = view_output_dir / method_name
        method_output_dir.mkdir(parents=True, exist_ok=True)

        for repeat_index in range(repeat_number):
            print(f"{repeat_index = }")

            existing_repeat_output_dirs = list(
                method_output_dir.glob(f"output_*")
            )
            existing_max_repeat_index = max(
                [-1] +
                [
                    int(d.name.split("_")[-1])
                    for d in existing_repeat_output_dirs
                    if d.name.split("_")[-1].isdigit()
                ]
            )

            repeat_output_dir = method_output_dir / f"output_{existing_max_repeat_index + 1}"
            repeat_output_dir.mkdir(parents=True, exist_ok=True)

            scene_task_list = []

            actual_scene_path_list = [
                scene_path
                for scene_path in dataset_path.iterdir()
                if scene_path.name in scene_list
            ]
            print(f"{actual_scene_path_list = }")
            for scene_index, scene_path in enumerate(actual_scene_path_list):
                print(f"{scene_path = }")

                model_path = repeat_output_dir / f"{scene_path.name}"
                print(f"{model_path = }")
                scene_task = partial(
                    scene_train_and_render,
                    train_arg_list=[
                        "--source_path",
                        str(scene_path),
                        "--model_path",
                        str(model_path),
                        "--eval",
                        "--resolution",
                        f"{resolution}",
                        "--n_views",
                        f"{view}",
                        "--test_iterations",
                        "5000",
                        "6000",
                        "8000",
                        "10000",
                        "--save_iterations",
                        *[str(it) for it in save_iterations],
                        "--port",
                        str(base_listen_port + scene_index),
                        "--dropout_algorithm",
                        dropout_algorithm,
                    ],
                    render_arg_list=[
                        "--model_path",
                        str(model_path),
                        "--resolution",
                        f"{resolution}",
                        "--iteration",
                        *[str(it) for it in save_iterations],
                        "--reserve_iteration",
                        str(save_iterations[-1]),
                    ],
                )

                scene_task_list.append(scene_task)

            run_multiple_tasks_multiple_gpus(
                gpu_id_list=GPU_ID_LIST,
                func_list=scene_task_list,
            )

            from metric import main as metric_main

            for name in ["train", "test"]:
                metric_main(
                    arg_list=[
                        "--path",
                        str(repeat_output_dir),
                        "--name",
                        name,
                        "--iteration",
                        *[str(it) for it in save_iterations],
                        "--scenes",
                        *scene_list,
                    ],
                )


if __name__ == "__main__":
    main()
