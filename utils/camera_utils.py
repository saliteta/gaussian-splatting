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

from scene.cameras import Camera
from scene.cached_camera import CachedCamera, CachedImageBlob
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import io
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm   
WARNED = False


class CachedCameras: 

    def __init__(self):
        # Different Resolution will get a List of NN.Module which is a Camera with different resolution
        self._cameras = {}
        self.current_resolution_scale = 1.0

    def getCamera(self, id) -> Camera:
        return self._cameras[id]

    def addCamera(self, id, camera: Camera):
        self.cameras[id] = camera

    @property
    def cached_resolution_scale(self) -> List[Camera]:
        return self._cameras[self.current_resolution_scale]

    def set_resolution_scale(self, resolution_scale: float):
        self.current_resolution_scale = resolution_scale

def loadCam(
    args,
    id,
    cam_info,
    resolution_scale,
    is_test_dataset,
    *,
    preloaded: Optional[Dict[str, Tuple[bytes, Tuple[int, int]]]] = None,
) -> Camera:
    # Keep compressed bytes in RAM; decode lazily inside CachedCamera.
    # We still open once to read header (size) if needed, but do not keep PIL objects around.
    if preloaded is not None and str(cam_info.image_path) in preloaded:
        image_bytes, (orig_w, orig_h) = preloaded[str(cam_info.image_path)]
    else:
        with open(cam_info.image_path, "rb") as f:
            image_bytes = f.read()
        with Image.open(io.BytesIO(image_bytes)) as im:
            orig_w, orig_h = im.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            # Enforce original resolution (no implicit 1.6K cap).
            global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    blob = CachedImageBlob(
        image_path=str(cam_info.image_path),
        image_bytes=image_bytes,
        orig_size=(orig_w, orig_h),
    )

    # Note: Camera expects resolution as (W,H) as used throughout this repo.
    return CachedCamera(
        uid=id,
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image_name=cam_info.image_name,
        blob=blob,
        resolution=resolution,
        data_device=args.data_device,
        train_test_exp=args.train_test_exp,
        is_test_dataset=is_test_dataset,
        is_test_view=cam_info.is_test,
    )

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_test_dataset)->List[Camera]:
    camera_list = []

    for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos), desc=f"Loading Cameras with resolution scale {resolution_scale}"):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_test_dataset))

    return camera_list


def cameraList_from_camInfos_preloaded(
    cam_infos,
    resolution_scale,
    args,
    is_test_dataset,
    *,
    preloaded: Dict[str, Tuple[bytes, Tuple[int, int]]],
) -> List[Camera]:
    camera_list: List[Camera] = []
    for id, c in tqdm(
        enumerate(cam_infos),
        total=len(cam_infos),
        desc=f"Loading Cameras with resolution scale {resolution_scale}",
    ):
        camera_list.append(
            loadCam(
                args,
                id,
                c,
                resolution_scale,
                is_test_dataset,
                preloaded=preloaded,
            )
        )
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
