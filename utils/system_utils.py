#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
import os
import torch
from matplotlib import cm
import numpy as np
import open3d as o3d

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


class Timing:
    """
    From https://github.com/sxyu/svox2/blob/ee80e2c4df8f29a407fda5729a494be94ccf9234/svox2/utils.py#L611
    
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        print(self.name, "elapsed", self.start.elapsed_time(self.end), "ms")



def save_ply(points, filename, rgbs=None):
    if type(points) in [torch.Tensor, torch.nn.parameter.Parameter]:
        points = points.detach().cpu().numpy()
    if type(rgbs) in [torch.Tensor, torch.nn.parameter.Parameter]:
        rgbs = rgbs.detach().cpu().numpy()

    if rgbs is None:
        rgbs = np.ones_like(points[:, [0]])
    if rgbs.shape[1] == 1:
        colormap = cm.get_cmap('turbo')
        rgbs = colormap(rgbs[:, 0])[:, :3]

    pcd = o3d.geometry.PointCloud()

    # 将 xyz 和 rgb 数据添加到点云中
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)  # 将 RGB 转换到 [0, 1] 范围
    # 保存为 PLY 文件
    o3d.io.write_point_cloud(filename, pcd)
