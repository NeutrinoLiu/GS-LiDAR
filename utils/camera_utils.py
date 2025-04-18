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
import time

def subsample_pointcloud(points, M, alpha=0.0005):
    """
    Subsample a point cloud to achieve roughly uniform density within each grid cell.
    
    Parameters:
    -----------
    points : ndarray, shape (N, 3)
        The input point cloud with N points in 3D space
    M : int
        Target number of points after subsampling
    alpha : float, default=0.001
        Parameter to determine grid size as alpha * diameter of original point cloud
        
    Returns:
    --------
    ndarray, shape (M', 3)
        Subsampled point cloud with approximately M points
    """
    if points.shape[0] <= M:
        # If the original point cloud has fewer points than M, return all points
        return np.arange(points.shape[0])
    
    # Calculate the diameter of the point cloud
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    diameter = np.linalg.norm(max_coords - min_coords)
    
    print(f"min_coords: {min_coords}, max_coords: {max_coords}, diameter: {diameter}")

    # Calculate the grid size
    grid_size = alpha * diameter
    
    # Calculate the number of grid cells in each dimension
    grid_dims = np.ceil((max_coords - min_coords) / grid_size).astype(int)
    
    # Create a dictionary to store points in each grid cell
    grid_dict = {}
    
    print(f"start to subsample point cloud, grid_dims: {grid_dims}, original points: {points.shape[0]}, target points: {M}")
    start = time.time()

    # Assign each point to a grid cell
    for i, point in enumerate(points):
        # Calculate the grid cell indices
        grid_idx = tuple(np.floor((point - min_coords) / grid_size).astype(int))
        
        # Add the point index to the corresponding grid cell
        if grid_idx in grid_dict:
            grid_dict[grid_idx].append(i)
        else:
            grid_dict[grid_idx] = [i]
    
    # Calculate the number of points to sample from each non-empty grid cell
    num_cells = len(grid_dict)
    points_per_cell = max(1, int(np.ceil(M / num_cells)))
    
    # Sample points from each grid cell
    sampled_indices = []
    exact_points_per_cell = []
    for cell_indices in grid_dict.values():
        # If the cell has fewer points than points_per_cell, take all points
        if len(cell_indices) <= points_per_cell:
            sampled_indices.extend(cell_indices)
            exact_points_per_cell.append(len(cell_indices))
        else:
            # Randomly sample points_per_cell points from the cell
            sampled_indices.extend(np.random.choice(cell_indices, points_per_cell, replace=False))
            exact_points_per_cell.append(points_per_cell)

    print(f"mean points per cell: {np.mean(exact_points_per_cell)}, std points per cell: {np.std(exact_points_per_cell)}")
    print(f"preliminary subsampled points: {len(sampled_indices)}")
    print(f"subsample time: {time.time() - start}")

    # If we have more points than M, randomly subsample to get exactly M points
    if len(sampled_indices) > M:
        sampled_indices = np.random.choice(sampled_indices, M, replace=False)
    
    # Return the subsampled point cloud
    return sampled_indices

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
    PERTURB_IDX = [5]
    PERTURB_INTENTSITY_max = 0.2
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
        towards=cam_info.towards,
        sequence_id=cam_info.sequence_id,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos)):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list
