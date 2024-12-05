# src/datasets/shapenet_dataset.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import json

class ShapeNetDataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=2048, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.transform = transform

        # Load file paths
        self.datapath = []
        with open(os.path.join(root_dir, 'train_test_split', f'shuffled_{split}_file_list.json'), 'r') as f:
            file_list = json.load(f)
            for item in file_list:
                self.datapath.append((item[0], os.path.join(root_dir, item[0], item[1])))

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, idx):
        category, file_path = self.datapath[idx]
        point_set = np.loadtxt(file_path + '.pts').astype(np.float32)
        labels = np.loadtxt(file_path + '.seg').astype(np.int64) - 1  # Labels start from 1

        # Sample points
        choice = np.random.choice(len(labels), self.num_points, replace=True)
        point_set = point_set[choice, :]
        labels = labels[choice]

        # Normalize
        point_set = point_set - np.mean(point_set, axis=0)
        norm = np.max(np.linalg.norm(point_set, axis=1))
        point_set = point_set / norm

        # Apply transforms if any
        if self.transform:
            point_set = self.transform(point_set)

        return torch.from_numpy(point_set), torch.from_numpy(labels)