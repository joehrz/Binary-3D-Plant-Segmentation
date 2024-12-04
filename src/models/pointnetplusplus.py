# src/models/pointnetplusplus.py

"""Implementaion of the pointnet++ model"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Import PointNet++ utility modules
from src.models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes: int):
        super(PointNetPlusPlus, self).__init__()

        # Set Abstraction layers with MSG
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radii=[0.1, 0.2],
            nsamples=[32, 64],
            mlps=[[3, 32, 32, 64], [3, 64, 64, 128]]  # Input channels remain 3
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radii=[0.2, 0.4],
            nsamples=[64, 128],
            mlps=[[195, 128, 128, 256], [195, 256, 256, 512]]  # Adjusted input channels to 195
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[771, 512, 1024],  # Adjusted input channels to 771
            group_all=True
        )

        # Feature Propagation layers
        self.fp3 = PointNetFeaturePropagation(in_channel=1792, mlp=[512, 512])  # Correct
        self.fp2 = PointNetFeaturePropagation(in_channel=704, mlp=[512, 256])   # Correct
        self.fp1 = PointNetFeaturePropagation(in_channel=256, mlp=[256, 128])   # Corrected to 256

        # Fully connected layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            xyz (torch.Tensor): Input point cloud data of shape (B, N, 3).

        Returns:
            torch.Tensor: Segmentation scores for each point.
        """
        B, N, _ = xyz.shape

        # Input Transformation
        l0_xyz = xyz.transpose(2, 1).contiguous()  # Shape: (B, 3, N)
        l0_points = None  # No additional features at input

        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        # Fully connected layers
        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.dropout(x)
        x = self.conv2(x)

        x = x.transpose(2, 1).contiguous()  # Shape: (B, N, num_classes)
        x = F.log_softmax(x, dim=-1)

        return x
