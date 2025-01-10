## src/datasets/mixed_pointcloud_dataset.py

"""A dataset class that loads (points, labels) from a .npy or .npz (shape (N,4) => x,y,z,label or (points=(N,3), labels=(N,)))."""

import os
import torch
import numpy as np
from torch.utils.data import Dataset


class MixedPointCloudDataset(Dataset):
    def __init__(self, file_list, num_points=4096):
        """
        file_list: list of paths to .npy/.npz with final processed data
        """
        self.file_list = file_list
        self.num_points = num_points

    def __len__(self):
        return len(self.file_list)
        

    def __getitem__(self, idx):
        path = self.file_list[idx]
        if path.endswith(".npy"):
            data = np.load(path)           # shape (N,4) or (N,3) + separate labels
            points = data[:, :3]
            labels = data[:, 3].astype(int)
        else:
            # e.g. if .npz => keys {points, labels}
            content = np.load(path)
            points = content["points"]
            labels = content["labels"].astype(int)


        # if more points than num_points -> random sample
        if points.shape[0] > self.num_points:
            idx_sample = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[idx_sample]
            labels = labels[idx_sample]

        # Convert to tensors
        points = torch.tensor(points, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        return points, labels
        

