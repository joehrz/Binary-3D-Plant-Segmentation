# src/models/pointnet2_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate Euclidean distance between each pair of points.

    Args:
        src (torch.Tensor): Source points of shape (B, N, C)
        dst (torch.Tensor): Destination points of shape (B, M, C)

    Returns:
        torch.Tensor: Distance matrix of shape (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # Shape: (B, N, M)
    dist += torch.sum(src ** 2, -1).unsqueeze(-1)  # Shape: (B, N, 1)
    dist += torch.sum(dst ** 2, -1).unsqueeze(-2)  # Shape: (B, 1, M)
    return dist

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Indexing into point cloud data.

    Args:
        points (torch.Tensor): Input points data, (B, N, C)
        idx (torch.Tensor): Indices to sample, (B, S) or (B, S, nsample)

    Returns:
        torch.Tensor: Sampled points data, (B, S, C) or (B, S, nsample, C)
    """
    device = points.device
    B, N, C = points.shape

    # Handle both 2D and 3D idx tensors
    if idx.dim() == 2:
        S = idx.shape[1]
        idx = idx + (torch.arange(B, device=device) * N).view(B, 1)
        idx = idx.view(-1)
        points = points.reshape(B * N, C)
        sampled_points = points[idx]
        sampled_points = sampled_points.view(B, S, C)
    elif idx.dim() == 3:
        S, nsample = idx.shape[1], idx.shape[2]
        idx = idx + (torch.arange(B, device=device) * N).view(B, 1, 1)
        idx = idx.view(-1)
        points = points.reshape(B * N, C)
        sampled_points = points[idx]
        sampled_points = sampled_points.view(B, S, nsample, C)
    else:
        raise ValueError("Invalid idx shape: {}".format(idx.shape))
    return sampled_points

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest point sampling.

    Args:
        xyz (torch.Tensor): Point cloud data, (B, N, 3)
        npoint (int): Number of points to sample

    Returns:
        torch.Tensor: Indices of sampled points, (B, npoint)
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Grouping operation in a local region.

    Args:
        radius (float): Search radius
        nsample (int): Maximum number of features in the local region
        xyz (torch.Tensor): All points, (B, N, 3)
        new_xyz (torch.Tensor): Query points, (B, S, 3)

    Returns:
        torch.Tensor: Grouped indices, (B, S, nsample)
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, S, 1)  # (B, S, N)
    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, points: torch.Tensor, returnfps: bool = False):
    """
    Sampling and grouping operation.

    Args:
        npoint (int): Number of points to sample
        radius (float): Search radius
        nsample (int): Number of points in each local region
        xyz (torch.Tensor): All points, (B, N, 3)
        points (torch.Tensor): Point features, (B, N, D)
        returnfps (bool): Whether to return the sampled point indices

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: new_xyz, new_points, [optional] fps_idx
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # (B, npoint)
    new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B, npoint, nsample)
    grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
    grouped_xyz -= new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)  # (B, npoint, nsample, D)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (B, npoint, nsample, 3+D)
    else:
        new_points = grouped_xyz
    if returnfps:
        return new_xyz, new_points, fps_idx
    else:
        return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, mlp: List[int], group_all: bool = False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        last_channel = mlp[0]
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, kernel_size=1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the PointNetSetAbstraction layer.

        Args:
            xyz (torch.Tensor): Input points position data, (B, 3, N)
            points (torch.Tensor): Input points data, (B, D, N)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: new_xyz, new_points
        """
        xyz = xyz.permute(0, 2, 1)  # (B, N, 3)
        if points is not None:
            points = points.permute(0, 2, 1)  # (B, N, D)
        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, xyz.shape[2], device=xyz.device)
            new_points = xyz.unsqueeze(1)  # (B, 1, N, 3)
            if points is not None:
                new_points = torch.cat([new_points, points.unsqueeze(1)], dim=-1)  # (B, 1, N, 3+D)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # Apply MLP
        new_points = new_points.permute(0, 3, 2, 1)  # (B, C, nsample, npoint)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]  # (B, C, npoint)
        return new_xyz.permute(0, 2, 1), new_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]]):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radii = radii
        self.nsamples = nsamples
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            # Define MLPs for each scale
            mlp_layers = []
            last_channel = mlps[i][0]
            for out_channel in mlps[i][1:]:
                mlp_layers.append(nn.Conv2d(last_channel, out_channel, kernel_size=1))
                mlp_layers.append(nn.BatchNorm2d(out_channel))
                mlp_layers.append(nn.ReLU())
                last_channel = out_channel
            self.mlps.append(nn.Sequential(*mlp_layers))


    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the PointNetSetAbstractionMsg layer.

        Args:
            xyz (torch.Tensor): Input points position data, (B, 3, N)
            points (torch.Tensor): Input points data, (B, D, N)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: new_xyz, new_points
        """
        B, _, N = xyz.shape
        S = self.npoint

        # Sample new points using farthest point sampling
        if S is not None:
            fps_idx = farthest_point_sample(xyz.permute(0, 2, 1).contiguous(), S)  # (B, npoint)
            new_xyz = index_points(xyz.permute(0, 2, 1).contiguous(), fps_idx).permute(0, 2, 1)  # (B, 3, npoint)
        else:
            new_xyz = xyz

        new_points_list = []
        for i in range(len(self.radii)):
            radius = self.radii[i]
            nsample = self.nsamples[i]

            # Perform grouping
            idx = query_ball_point(radius, nsample, xyz.permute(0, 2, 1), new_xyz.permute(0, 2, 1))
            grouped_xyz = index_points(xyz.permute(0, 2, 1), idx)  # (B, npoint, nsample, 3)
            grouped_xyz = grouped_xyz - new_xyz.permute(0, 2, 1).unsqueeze(2)  # (B, npoint, nsample, 3)

            if points is not None:
                grouped_points = index_points(points.permute(0, 2, 1), idx)  # (B, npoint, nsample, D)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (B, npoint, nsample, 3+D)
            else:
                grouped_points = grouped_xyz  # (B, npoint, nsample, 3)

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # (B, C, nsample, npoint)
            new_points = self.mlps[i](grouped_points)  # Apply MLP
            new_points = torch.max(new_points, 2)[0]  # Max pooling over nsample
            new_points_list.append(new_points)

        new_points_concat = torch.cat(new_points_list, dim=1)  # Concatenate features from different scales
        return new_xyz, new_points_concat

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel: int, mlp: List[int]):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor, points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        """
        Feature Propagation Module.

        Args:
            xyz1 (torch.Tensor): Input points position data, (B, 3, N)
            xyz2 (torch.Tensor): Input points position data, (B, 3, S)
            points1 (torch.Tensor): Input points data, (B, D, N)
            points2 (torch.Tensor): Input points data, (B, D, S)

        Returns:
            torch.Tensor: Upsampled point features, (B, D_new, N)
        """
        xyz1 = xyz1.permute(0, 2, 1)  # (B, N, 3)
        xyz2 = xyz2.permute(0, 2, 1)  # (B, S, 3)
        points2 = points2.permute(0, 2, 1)  # (B, S, D)
        dists = square_distance(xyz1, xyz2)  # (B, N, S)
        dists, idx = dists.sort(dim=-1)
        dists = dists[:, :, :3]
        idx = idx[:, :, :3]  # (B, N, 3)
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)  # (B, N, D)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)  # (B, N, D_old)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points  # (B, N, D)
        new_points = new_points.permute(0, 2, 1)  # (B, D_new, N)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points


