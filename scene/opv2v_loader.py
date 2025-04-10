#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
import glob
import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from torchvision.utils import save_image
import torch.nn.functional as F
import json
from matplotlib import cm
from utils.system_utils import save_ply
from utils.camera_utils import subsample_pointcloud

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "AdvCollaborativePerception")))

from attack import GeneralAttacker


def rotation_matrix(roll, yaw, pitch):
    R = np.array([[np.cos(yaw)*np.cos(pitch), 
                   np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), 
                   np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                  [np.sin(yaw)*np.cos(pitch), 
                   np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), 
                   np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                  [-np.sin(pitch), 
                   np.cos(pitch)*np.sin(roll), 
                   np.cos(pitch)*np.cos(roll)]])
    return R


def get_c2w_matrix(calib):
    R = rotation_matrix(*(np.array(calib["lidar_pose"][3:]) * np.pi / 180))
    
    translation_vector = np.array(calib["lidar_pose"][:3])
    
    c2w_matrix = np.eye(4)
    c2w_matrix[:3, :3] = R
    c2w_matrix[:3, 3] = translation_vector
    
    return c2w_matrix


def range_to_ply(depth, filename, vfov=(-31.96, 10.67), hfov=(-90, 90)):
    panorama_height, panorama_width = depth.shape[-2:]

    theta, phi = torch.meshgrid(torch.arange(panorama_height, device='cuda'),
                                torch.arange(panorama_width, device='cuda'), indexing="ij")

    vertical_degree_range = vfov[1] - vfov[0]
    theta = (90 - vfov[1] + theta / panorama_height * vertical_degree_range) * torch.pi / 180

    horizontal_degree_range = hfov[1] - hfov[0]
    phi = (hfov[0] + phi / panorama_width * horizontal_degree_range) * torch.pi / 180

    dx = torch.sin(theta) * torch.sin(phi)
    dz = torch.sin(theta) * torch.cos(phi)
    dy = -torch.cos(theta)

    directions = torch.stack([dx, dy, dz], dim=0)
    directions = F.normalize(directions, dim=0)
    points_xyz = directions * depth

    save_ply(points_xyz.reshape(3, -1).permute(1, 0), filename)

    return


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_scale_factor=True):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    if fix_scale_factor:
        scale_factor = 1 / 10
    else:
        # Just make sure it's it in the [-1, 1]^3 cube
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform, scale_factor


def readOPV2VInfo_Spoof_Remove(args):
    ga = GeneralAttacker()

    path = args.source_path
    eval = args.eval
    num_pts = args.num_pts
    time_duration = args.time_duration
    debug_cuda = args.debug_cuda

    normal_lidar, attack_lidar, general_info, attack_info = ga.attack(
        attacker_type=args.attacker_type,
        dense=args.dense,
        sync=args.sync,
        advshape=args.advshape,
        attack_id=args.attack_id,
        attack_frame_ids=args.attack_frame_ids
    )

    assert args.vfov is not None and args.hfov is not None

    frame_data = general_info["frame_ids"]

    frames = len(frame_data)

    # static
    s_frame_id = frame_data[0]
    e_frame_id = frame_data[-1]
    val_frame_ids = args.val_frames
    stride = args.frame_stride
    args.frames = frames

    cars = general_info["vehicle_ids"]
    
    def parse_one_car(sequence_id, lidar_data):

        point_list = []
        points_time = []
        cam_infos = []

        for frame_idx in tqdm(range(frames), desc="Reading OPV2V data"):

            points = lidar_data[frame_idx][sequence_id]["lidar"]
            points[:, 3] = 1.0 # Assign a default intensity of 1.0
            intensity = points[:, 3]
            points = points[:, :3]

            # 把自车的lidar点去掉
            condition = (np.linalg.norm(points, axis=1) > 2.5)  # & (intensity > 0)
            indices = np.where(condition)
            points = points[indices]
            intensity = intensity[indices]

            lidar2globals = get_c2w_matrix(lidar_data[frame_idx][sequence_id]["lidar_pose"])
            points_homo = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
            points = (points_homo @ lidar2globals.T)[:, :3]
            point_list.append(points)

            timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * frame_idx / (frames - 1)
            point_time = np.full_like(points[:, :1], timestamp)
            points_time.append(point_time)

            idx = frame_idx
            w2l = np.array([0, -1, 0, 0,
                            0, 0, -1, 0,
                            1, 0, 0, 0,
                            0, 0, 0, 1]).reshape(4, 4) @ np.linalg.inv(lidar2globals)
            R = np.transpose(w2l[:3, :3])
            T = w2l[:3, 3]
            points_cam = points @ R + T

            # 前180度
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T,
                                        timestamp=timestamp,
                                        pointcloud_camera=points_cam,
                                        intensity=intensity,
                                        towards='forward',
                                        sequence_id=sequence_id))

            # 后180度
            R_back = R @ np.array([-1, 0, 0,
                                0, 1, 0,
                                0, 0, -1]).reshape(3, 3)
            T_back = T * np.array([-1, 1, -1])
            points_cam_back = points @ R_back + T_back
            cam_infos.append(CameraInfo(uid=idx + frames, R=R_back, T=T_back,
                                        timestamp=timestamp, pointcloud_camera=points_cam_back, intensity=intensity,
                                        towards='backward',
                                        sequence_id=sequence_id))

        pointcloud = np.concatenate(point_list, axis=0)
        pointcloud_timestamp = np.concatenate(points_time, axis=0)

        w2cs = np.zeros((len(cam_infos), 4, 4))
        Rs = np.stack([c.R for c in cam_infos], axis=0)
        Ts = np.stack([c.T for c in cam_infos], axis=0)
        w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
        w2cs[:, :3, 3] = Ts
        w2cs[:, 3, 3] = 1
        c2ws = unpad_poses(np.linalg.inv(w2cs))

        return pointcloud, pointcloud_timestamp, c2ws, cam_infos

    wild_cards = ["*", "all"]
    total_agents = 0
    if args.sequence_id not in wild_cards:
        pointcloud_all, pointcloud_timestamp_all, c2ws_all, cam_infos_all = parse_one_car(sequence_id, attack_lidar)
        total_agents = 1
    else:
        pointcloud_all = []
        pointcloud_timestamp_all = []
        c2ws_all = []
        cam_infos_all = []
        for sequence_id in cars:
            total_agents += 1
            pointcloud, pointcloud_timestamp, c2ws, cam_infos = parse_one_car(sequence_id, attack_lidar)
            pointcloud_all.append(pointcloud)
            pointcloud_timestamp_all.append(pointcloud_timestamp)
            c2ws_all.append(c2ws)
            cam_infos_all.extend(cam_infos)
        
    pointcloud_all = np.concatenate(pointcloud_all, axis=0)
    pointcloud_timestamp_all = np.concatenate(pointcloud_timestamp_all, axis=0)
    c2ws_all = np.concatenate(c2ws_all, axis=0)

    num_pts = min(num_pts, pointcloud_all.shape[0])
    # indices = np.random.choice(pointcloud_all.shape[0], num_pts, replace=False)
    indices = subsample_pointcloud(pointcloud_all, num_pts)
    pointcloud_all = pointcloud_all[indices]
    pointcloud_timestamp_all = pointcloud_timestamp_all[indices]
    
    print(f"Total {len(cam_infos_all)} cameras from {total_agents} agents.")

    if not args.test_only:
        c2ws_all, transform_all, scale_factor_all = transform_poses_pca(c2ws_all, args.dynamic)
        np.savez(os.path.join(args.model_path, 'transform_poses_pca.npz'), transform=transform_all, scale_factor=scale_factor_all)
        c2ws_all = pad_poses(c2ws_all)
    else:
        data = np.load(os.path.join(args.model_path, 'transform_poses_pca.npz'))
        transform_all = data['transform']
        scale_factor_all = data['scale_factor'].item()
        c2ws_all = np.diag(np.array([1 / scale_factor_all] * 3 + [1])) @ transform_all @ pad_poses(c2ws_all)
        c2ws_all[:, :3, 3] *= scale_factor_all

    for idx, cam_info in enumerate(cam_infos_all):
        c2w = c2ws_all[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        cam_info.pointcloud_camera[:] *= scale_factor_all

    pointcloud_all = (np.pad(pointcloud_all, ((0, 0), (0, 1)), constant_values=1) @ transform_all.T)[:, :3]
    args.scale_factor = float(scale_factor_all)

    mod = args.cam_num

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos_all) if (idx // mod + s_frame_id) not in val_frame_ids]
        test_cam_infos = [c for idx, c in enumerate(cam_infos_all) if (idx // mod + s_frame_id) in val_frame_ids]
    else:
        train_cam_infos = cam_infos_all
        test_cam_infos = [c for idx, c in enumerate(cam_infos_all) if (idx // mod + s_frame_id) in val_frame_ids]

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1

    ply_path = os.path.join(args.model_path, "points3d.ply")
    if not args.test_only:
        dummy_rgbs = np.random.random((pointcloud_all.shape[0], 3)) * 255.0
        storePly(ply_path, pointcloud_all, dummy_rgbs, pointcloud_timestamp_all)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    time_interval = (time_duration[1] - time_duration[0]) / (frames - 1)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_interval=time_interval)

    return scene_info    


def readOPV2VInfo(args):
    path = args.source_path
    eval = args.eval
    num_pts = args.num_pts
    time_duration = args.time_duration
    debug_cuda = args.debug_cuda

    opv2v_mode = "test"
    opv2v_mode_transform = "test_transform"

    assert args.vfov is not None and args.hfov is not None

    scenario = args.scenario
    assert scenario is not None, "Please specify the scenario name."
    sequence_id = args.sequence_id
    assert sequence_id is not None, "Please specify the sequence id."

    # static
    s_frame_id = args.frame_start
    e_frame_id = args.frame_end
    val_frame_ids = args.val_frames
    stride = args.frame_stride
    frames = (e_frame_id - s_frame_id) // stride + 1 # frame id stride 2
    args.frames = frames
    
    def parse_one_car(sequence_id):
        with open(os.path.join(path, opv2v_mode_transform, scenario, f"transforms_{scenario}_{sequence_id}.json"), "r") as file:
            data = json.load(file)

        poses = data["frames"]

        # frames = e_frame_id + 1 - s_frame_id
        lidar_dir = os.path.join(path, opv2v_mode, scenario, sequence_id)

        point_list = []
        points_time = []
        cam_infos = []

        for frame_idx in tqdm(range(frames), desc="Reading OPV2V data"):
            lidar_idx = frame_idx * stride + s_frame_id # frame id stride 2
            # lidar_idx = frame_idx + s_frame_id
            points = np.fromfile(os.path.join(lidar_dir, "%06d.bin" % lidar_idx), dtype=np.float32).reshape((-1, 4))
            intensity = points[:, 3]
            points = points[:, :3]

            # 把自车的lidar点去掉
            condition = (np.linalg.norm(points, axis=1) > 2.5)  # & (intensity > 0)
            indices = np.where(condition)
            points = points[indices]
            intensity = intensity[indices]

            lidar2globals = np.array(poses[frame_idx]["lidar2world"])

            points_homo = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
            points = (points_homo @ lidar2globals.T)[:, :3]
            point_list.append(points)

            timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * frame_idx / (frames - 1)
            point_time = np.full_like(points[:, :1], timestamp)
            points_time.append(point_time)

            idx = frame_idx
            w2l = np.array([0, -1, 0, 0,
                            0, 0, -1, 0,
                            1, 0, 0, 0,
                            0, 0, 0, 1]).reshape(4, 4) @ np.linalg.inv(lidar2globals)
            R = np.transpose(w2l[:3, :3])
            T = w2l[:3, 3]
            points_cam = points @ R + T

            # 前180度
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T,
                                        timestamp=timestamp,
                                        pointcloud_camera=points_cam,
                                        intensity=intensity,
                                        towards='forward',
                                        sequence_id=sequence_id))

            # 后180度
            R_back = R @ np.array([-1, 0, 0,
                                0, 1, 0,
                                0, 0, -1]).reshape(3, 3)
            T_back = T * np.array([-1, 1, -1])
            points_cam_back = points @ R_back + T_back
            cam_infos.append(CameraInfo(uid=idx + frames, R=R_back, T=T_back,
                                        timestamp=timestamp, pointcloud_camera=points_cam_back, intensity=intensity,
                                        towards='backward',
                                        sequence_id=sequence_id))

        pointcloud = np.concatenate(point_list, axis=0)
        pointcloud_timestamp = np.concatenate(points_time, axis=0)

        w2cs = np.zeros((len(cam_infos), 4, 4))
        Rs = np.stack([c.R for c in cam_infos], axis=0)
        Ts = np.stack([c.T for c in cam_infos], axis=0)
        w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
        w2cs[:, :3, 3] = Ts
        w2cs[:, 3, 3] = 1
        c2ws = unpad_poses(np.linalg.inv(w2cs))

        return pointcloud, pointcloud_timestamp, c2ws, cam_infos

    wild_cards = ["*", "all"]
    total_agents = 0
    if args.sequence_id not in wild_cards:
        pointcloud_all, pointcloud_timestamp_all, c2ws_all, cam_infos_all = parse_one_car(sequence_id)
        total_agents = 1
    else:
        pointcloud_all = []
        pointcloud_timestamp_all = []
        c2ws_all = []
        cam_infos_all = []
        sub_dirs = sorted(glob.glob(os.path.join(path, opv2v_mode, scenario, "*")))
        for sub_dir in sub_dirs:
            if not os.path.isdir(sub_dir):
                continue
            total_agents += 1
            sequence_id = os.path.basename(sub_dir)
            pointcloud, pointcloud_timestamp, c2ws, cam_infos = parse_one_car(sequence_id)
            pointcloud_all.append(pointcloud)
            pointcloud_timestamp_all.append(pointcloud_timestamp)
            c2ws_all.append(c2ws)
            cam_infos_all.extend(cam_infos)
        
    pointcloud_all = np.concatenate(pointcloud_all, axis=0)
    pointcloud_timestamp_all = np.concatenate(pointcloud_timestamp_all, axis=0)
    c2ws_all = np.concatenate(c2ws_all, axis=0)

    num_pts = min(num_pts, pointcloud_all.shape[0])
    # indices = np.random.choice(pointcloud_all.shape[0], num_pts, replace=False)
    indices = subsample_pointcloud(pointcloud_all, num_pts)
    pointcloud_all = pointcloud_all[indices]
    pointcloud_timestamp_all = pointcloud_timestamp_all[indices]
    
    print(f"Total {len(cam_infos_all)} cameras from {total_agents} agents.")

    if not args.test_only:
        c2ws_all, transform_all, scale_factor_all = transform_poses_pca(c2ws_all, args.dynamic)
        np.savez(os.path.join(args.model_path, 'transform_poses_pca.npz'), transform=transform_all, scale_factor=scale_factor_all)
        c2ws_all = pad_poses(c2ws_all)
    else:
        data = np.load(os.path.join(args.model_path, 'transform_poses_pca.npz'))
        transform_all = data['transform']
        scale_factor_all = data['scale_factor'].item()
        c2ws_all = np.diag(np.array([1 / scale_factor_all] * 3 + [1])) @ transform_all @ pad_poses(c2ws_all)
        c2ws_all[:, :3, 3] *= scale_factor_all

    for idx, cam_info in enumerate(cam_infos_all):
        c2w = c2ws_all[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        cam_info.pointcloud_camera[:] *= scale_factor_all

    pointcloud_all = (np.pad(pointcloud_all, ((0, 0), (0, 1)), constant_values=1) @ transform_all.T)[:, :3]
    args.scale_factor = float(scale_factor_all)

    mod = args.cam_num

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos_all) if (idx // mod + s_frame_id) not in val_frame_ids]
        test_cam_infos = [c for idx, c in enumerate(cam_infos_all) if (idx // mod + s_frame_id) in val_frame_ids]
    else:
        train_cam_infos = cam_infos_all
        test_cam_infos = [c for idx, c in enumerate(cam_infos_all) if (idx // mod + s_frame_id) in val_frame_ids]

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1

    ply_path = os.path.join(args.model_path, "points3d.ply")
    if not args.test_only:
        dummy_rgbs = np.random.random((pointcloud_all.shape[0], 3)) * 255.0
        storePly(ply_path, pointcloud_all, dummy_rgbs, pointcloud_timestamp_all)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    time_interval = (time_duration[1] - time_duration[0]) / (frames - 1)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_interval=time_interval)

    return scene_info
