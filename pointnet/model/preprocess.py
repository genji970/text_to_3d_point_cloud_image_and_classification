import torch
import torch.nn as nn
import open3d as o3d
import numpy as np

from pointnet.master.config import config

if config['cuda_on']:
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

def load_point_cloud(ply_path, num_points=2048):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]

    # [1, 6, 2048]로 변경하여 Conv2d 입력과 일치시킴
    points = torch.cat([torch.tensor(points, dtype=torch.float32), torch.zeros((2048, 3))], dim=1)
    points = points.transpose(0, 1).unsqueeze(0).to(device)
    return points