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

from pathlib import Path
from PIL import Image
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, _to_numpy
from utils.graphics_utils import fov2focal
import scipy
import matplotlib.pyplot as plt
from torch import nn
import copy

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    if cam_info.mask is not None:
        loaded_mask = PILtoTorch(cam_info.mask, resolution)
        if loaded_mask.shape[0] == 4:
            loaded_mask = loaded_mask[:3]
    else:
        loaded_mask = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, bounds=cam_info.bounds,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)


def cameraList_from_camInfos(
    cam_infos,
    resolution_scale,
    args,
) -> list[Camera]:
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def intrinsics_from_camera(
    camera: Camera,
) -> np.ndarray:
    """Compute pinhole intrinsics from FoV and image size (LLFF/3DGS-style)."""
    fx = fov2focal(camera.FoVx, camera.image_width)
    fy = fov2focal(camera.FoVy, camera.image_height)
    cx = 0.5 * camera.image_width
    cy = 0.5 * camera.image_height
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ],
        dtype=camera.R.dtype,
    )
    return K

def print_camera(
    cam: Camera,
    save_image_dir: str | Path | None = None,
):
    print(f"Camera: {getattr(cam, 'image_name', 'cam')}")
    print(f" {cam.R = }")
    print(f" {cam.T = }")
    print(f" {cam.FoVx = }")
    print(f" {cam.FoVy = }")
    print(f" {cam.camera_center = }")
    print(f" {cam.znear = }")
    print(f" {cam.zfar = }")
    print(f" {cam.trans = }")
    print(f" {cam.scale = }")
    print(f" {cam.bounds = }")
    print(f" {cam.world_view_transform = }")
    print(f" {cam.projection_matrix = }")
    print(f" {cam.full_proj_transform = }")
    intrinsics = intrinsics_from_camera(cam)
    print(f" {intrinsics = }")
    if save_image_dir is not None:
        save_image_dir = Path(save_image_dir)
        save_image_dir.mkdir(parents=True, exist_ok=True)
        img_path = save_image_dir / f"cam_{getattr(cam, 'image_name', 'cam')}.png"
        img_np = _to_numpy(cam.original_image).transpose(1, 2, 0)
        img_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_u8).save(img_path)

def print_camera_list(
    cam_list,
    save_image_dir: str | Path | None = None,
):
    for cam in cam_list:
        print_camera(cam, save_image_dir=save_image_dir)
        print("-----")