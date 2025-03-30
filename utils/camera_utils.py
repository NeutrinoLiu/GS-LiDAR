#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
import torch
import cv2
from scene.cameras import Camera
import numpy as np
from scene.scene_utils import CameraInfo
from tqdm import tqdm
from torchvision.utils import save_image

def downsample_depth_map(depth_map, scale_factor=2):
    """
    Downsample a depth map numpy array to a lower resolution using OpenCV.
    
    Args:
        depth_map (numpy.ndarray): Input depth map with shape (H, W) or (1, H, W)
        scale_factor (int, optional): Factor by which to reduce dimensions. Default is 2.
    
    Returns:
        numpy.ndarray: Downsampled depth map with same number of dimensions as input
    """
    # Make sure input is a numpy array
    depth_map = np.asarray(depth_map)
    
    # Handle different input shapes
    original_shape_len = len(depth_map.shape)
    
    if original_shape_len == 3:  # Shape is (1, H, W)
        depth_map = depth_map[0]  # Extract the 2D map
        add_channel_dim = True
    else:
        add_channel_dim = False
    
    # Calculate new dimensions
    h, w = depth_map.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    
    # Resize the depth map
    # INTER_NEAREST or INTER_LINEAR are typically good for depth maps
    # INTER_AREA can be better for downsampling
    downsampled = cv2.resize(depth_map, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Restore the channel dimension if original input had it
    if add_channel_dim:
        downsampled = np.expand_dims(downsampled, axis=0)
    
    return downsampled

class Perturb:
    PERTURB_IDX = [66, 88, 33, 111,]
    PERTURB_INTENTSITY_max = 0.1
    PERTURB_INTENTSITY = {}
    @staticmethod
    def perturb_depth(id, depth):
        if id in Perturb.PERTURB_IDX and id not in Perturb.PERTURB_INTENTSITY:
            # generate a noise with the same shape as depth
            noise = np.random.uniform(-Perturb.PERTURB_INTENTSITY_max, Perturb.PERTURB_INTENTSITY_max, depth.shape)
            Perturb.PERTURB_INTENTSITY[id] = noise
            print(f">>>>> Perturb camera {id} with noise")
        if id in Perturb.PERTURB_INTENTSITY:
            # print(f"pts_depth means std before: {np.mean(depth)}, {np.std(depth)}")
            if Perturb.PERTURB_INTENTSITY[id].shape != depth.shape:
                # downsample the noise
                # depth is in shape (1, H, W)
                noise = downsample_depth_map(Perturb.PERTURB_INTENTSITY[id], scale_factor=2)
                Perturb.PERTURB_INTENTSITY[id] = noise
            scaler = np.clip(Perturb.PERTURB_INTENTSITY[id] + 1, 0.5, 1.5)
            depth = depth * scaler
            # print(f"pts_depth means std after: {np.mean(depth)}, {np.std(depth)}")
        return depth

def loadCam(args, id, cam_info: CameraInfo, resolution_scale):
    orig_h, orig_w = args.hw

    if args.resolution == -1:
        global_down = 1
    else:
        global_down = orig_w / args.resolution

    scale = float(global_down) * float(resolution_scale)
    resolution = (int(orig_w / scale), int(orig_h / scale))

    vfov = args.vfov
    hfov = args.hfov
    if cam_info.pointcloud_camera is not None:
        intensity = cam_info.intensity
        if intensity is None:
            intensity = np.ones_like(cam_info.pointcloud_camera)[:, 0]

        w = resolution[0]
        h = resolution[1]

        pts_depth = np.zeros([1, h, w])
        pts_intensity = np.zeros([1, h, w])
        point_camera = cam_info.pointcloud_camera
        x = point_camera[:, 0]
        y = point_camera[:, 1]
        z = point_camera[:, 2]
        phi = np.arctan2(x, z)
        theta = np.arctan2(np.sqrt(x ** 2 + z ** 2), -y)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        VFOV_max = np.pi / 2 - vfov[0] * np.pi / 180
        VFOV_min = np.pi / 2 - vfov[1] * np.pi / 180
        HFOV_max = hfov[1] * np.pi / 180
        HFOV_min = hfov[0] * np.pi / 180

        theta = (theta - VFOV_min) * h / (VFOV_max - VFOV_min)
        phi = (phi - HFOV_min) * w / (HFOV_max - HFOV_min)
        uvz = np.stack((theta, phi, r, intensity), 1)

        uvz = uvz[uvz[:, 0] >= -0.5]
        uvz = uvz[uvz[:, 0] < h - 0.5]
        uvz = uvz[uvz[:, 1] >= -0.5]
        uvz = uvz[uvz[:, 1] < w - 0.5]
        uv = uvz[:, :2]
        uv = np.around(uv).astype(int)

        for i in range(uv.shape[0]):
            x, y = uv[i]
            if pts_depth[0, x, y] == 0:
                pts_depth[0, x, y] = uvz[i, 2]
                pts_intensity[0, x, y] = uvz[i, 3]
            elif uvz[i, 2] < pts_depth[0, x, y]:
                pts_depth[0, x, y] = uvz[i, 2]
                pts_intensity[0, x, y] = uvz[i, 3]

        # pts_depth = Perturb.perturb_depth(cam_info.uid, pts_depth)
        pts_depth = torch.from_numpy(pts_depth).float().cuda()
        pts_intensity = torch.from_numpy(pts_intensity).float().cuda()
    else:
        pts_depth = None
        pts_intensity = None

    return Camera(
        colmap_id=cam_info.uid,
        uid=id,
        R=cam_info.R,
        T=cam_info.T,
        vfov=vfov,
        hfov=hfov,
        data_device=args.data_device,
        timestamp=cam_info.timestamp,
        resolution=resolution,
        pts_depth=pts_depth,
        pts_intensity=pts_intensity,
        towards=cam_info.towards
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos)):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list
