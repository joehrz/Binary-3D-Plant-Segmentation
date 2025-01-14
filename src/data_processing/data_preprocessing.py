# src/data_processing/data_preprocessing.py

'''Downsample, adjust and normalize point cloud data'''

import numpy as np
import open3d as o3d
from typing import Tuple
import os


def voxel_down_sample_with_indices(pcd: o3d.geometry.PointCloud, voxel_size: float):
    """
    Downsamples the point cloud using a voxel grid and returns the indices of the selected points.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        voxel_size (float): The voxel size for downsampling.

    Returns:
        downsampled_pcd (o3d.geometry.PointCloud): The downsampled point cloud.
        indices (np.ndarray): Array of indices of the selected points in the original point cloud.
    """
    # Compute the voxel grid boundaries
    min_bound = pcd.get_min_bound() - voxel_size * 0.5
    max_bound = pcd.get_max_bound() + voxel_size * 0.5

    # Perform voxel downsampling and trace the points
    downsampled_pcd, _, point_indices = pcd.voxel_down_sample_and_trace(
        voxel_size, min_bound, max_bound, False
    )

    # Extract the indices of the first point in each voxel
    indices = []
    for idx_list in point_indices:
        if len(idx_list) > 0:
            indices.append(idx_list[0])
    indices = np.array(indices, dtype=int)

    return downsampled_pcd, indices


def adjust_point_count_with_indices(pcd: o3d.geometry.PointCloud, labels: np.ndarray, num_points: int):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Separate plant and noise indices
    plant_indices = np.where(labels == 1)[0]
    noise_indices = np.where(labels == 0)[0]

    total_indices = len(plant_indices) + len(noise_indices)

    # Calculate the proportion of each class
    plant_ratio = len(plant_indices) / total_indices if total_indices > 0 else 0
    noise_ratio = len(noise_indices) / total_indices if total_indices > 0 else 0

    # Ensure minimum number of noise points (e.g., 10% of num_points)
    min_noise_points = max(1, int(0.1 * num_points))
    num_noise_points = max(min_noise_points, int(num_points * noise_ratio))

    # Adjust if not enough noise points are available
    num_noise_points = min(num_noise_points, len(noise_indices))
    num_plant_points = num_points - num_noise_points

    # Adjust if not enough plant points are available
    num_plant_points = min(num_plant_points, len(plant_indices))

    # Sample indices from each class
    selected_plant_indices = np.random.choice(
        plant_indices, num_plant_points, replace=(len(plant_indices) < num_plant_points)
    )
    selected_noise_indices = np.random.choice(
        noise_indices, num_noise_points, replace=(len(noise_indices) < num_noise_points)
    )

    # Combine and shuffle indices
    selected_indices = np.concatenate([selected_plant_indices, selected_noise_indices])
    np.random.shuffle(selected_indices)

    # Extract points, colors, and labels
    adjusted_points = points[selected_indices]
    adjusted_colors = colors[selected_indices]
    adjusted_labels = labels[selected_indices]

    # Create new point cloud
    adjusted_pcd = o3d.geometry.PointCloud()
    adjusted_pcd.points = o3d.utility.Vector3dVector(adjusted_points)
    adjusted_pcd.colors = o3d.utility.Vector3dVector(adjusted_colors)

    return adjusted_pcd, adjusted_labels


def normalize_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Normalizes the point cloud to fit within a unit sphere centered at the origin.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.

    Returns:
        o3d.geometry.PointCloud: Normalized point cloud.
    """
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_distance = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points /= max_distance
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def preprocess_point_clouds(input_dir: str, output_dir: str, voxel_size: float, num_points: int) -> None:
    """
    Preprocesses point clouds by downsampling, adjusting point count, normalizing, and saving.

    Args:
        input_dir (str): Directory containing original point clouds.
        output_dir (str): Directory to save the preprocessed point clouds.
        voxel_size (float): Voxel size for downsampling.
        num_points (int): Desired number of points per point cloud.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.ply'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load point cloud
            pcd = o3d.io.read_point_cloud(input_path)

            # Downsample
            pcd = downsample_point_cloud(pcd, voxel_size)

            # Adjust point count
            pcd = adjust_point_count(pcd, num_points)

            # Normalize
            pcd = normalize_point_cloud(pcd)

            # Save preprocessed point cloud
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"Saved preprocessed point cloud to {output_path}")
