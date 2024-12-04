# src/datasets/pointcloud_dataset.py

"""Helper to initialize the point cloud dataset"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
from typing import Tuple

class PointCloudDataset(Dataset):
    def __init__(self, data_dir: str, num_points: int = 1024, transform=None):
        """
        Initializes the dataset.

        Args:
            data_dir (str): Directory containing point cloud files.
            num_points (int): Number of points per point cloud.
            transform (callable, optional): Optional transform to be applied.
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.transform = transform
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.ply')]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a point cloud and its labels.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Points and labels tensors.
        """
        filename = self.files[idx]
        filepath = os.path.join(self.data_dir, filename)

        # Load point cloud
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
        # Assuming labels are stored as colors (modify as needed)
        labels = np.zeros(points.shape[0], dtype=np.int64)

        if self.transform:
            points = self.transform(points)

        # Convert to tensors
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels)

        return points, labels