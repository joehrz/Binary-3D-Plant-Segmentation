# src/data_processing/data_preprocessing.py

"""
Downsample, adjust, and normalize point cloud data while preserving labels.
"""

import numpy as np
import open3d as o3d
import os
from typing import Tuple


def voxel_down_sample_with_indices(
    pcd: o3d.geometry.PointCloud, voxel_size: float
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Downsamples the point cloud using a voxel grid and returns the indices
    of the selected points in the original cloud.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        voxel_size (float): The voxel size for downsampling.

    Returns:
        downsampled_pcd (o3d.geometry.PointCloud): The downsampled point cloud.
        indices (np.ndarray): Array of indices of the selected points in the original point cloud.
    """
    # Compute approximate voxel grid boundaries
    min_bound = pcd.get_min_bound() - voxel_size * 0.5
    max_bound = pcd.get_max_bound() + voxel_size * 0.5

    # Downsample + trace which points survive
    downsampled_pcd, _, point_indices = pcd.voxel_down_sample_and_trace(
        voxel_size, min_bound, max_bound, False
    )

    # Extract the index of the first point in each voxel
    indices = []
    for idx_list in point_indices:
        if len(idx_list) > 0:
            indices.append(idx_list[0])
    indices = np.array(indices, dtype=int)

    return downsampled_pcd, indices


def adjust_point_count_with_indices(
    pcd: o3d.geometry.PointCloud, labels: np.ndarray, num_points: int
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Adjusts the total number of points to 'num_points', preserving the ratio
    of plant vs. non-plant as much as possible. If needed, oversamples
    the minority class or ensures a minimum number of non-plant points.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud.
        labels (np.ndarray): Binary labels (0 or 1) corresponding to each point.
        num_points (int): Desired total number of points.

    Returns:
        adjusted_pcd (o3d.geometry.PointCloud): New point cloud with 'num_points' points.
        adjusted_labels (np.ndarray): Labels array of length 'num_points'.
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    plant_indices = np.where(labels == 1)[0]
    noise_indices = np.where(labels == 0)[0]

    total_plant = len(plant_indices)
    total_noise = len(noise_indices)
    total_points = total_plant + total_noise

    if total_points == 0:
        # Edge case: empty cloud => just return
        return pcd, labels

    # Optional: enforce a min ratio of non-plant (e.g., 10%)
    min_noise_points = max(1, int(0.1 * num_points))

    # Estimate how many noise vs. plant to pick
    # (This is a flexible approach; adapt logic as needed.)
    noise_ratio = total_noise / total_points
    est_noise_count = int(num_points * noise_ratio)
    est_noise_count = max(min_noise_points, est_noise_count)
    est_noise_count = min(est_noise_count, total_noise)

    est_plant_count = num_points - est_noise_count
    est_plant_count = min(est_plant_count, total_plant)

    # Sample indices
    selected_plant_indices = np.random.choice(
        plant_indices, est_plant_count, replace=(total_plant < est_plant_count)
    )
    selected_noise_indices = np.random.choice(
        noise_indices, est_noise_count, replace=(total_noise < est_noise_count)
    )

    selected_indices = np.concatenate([selected_plant_indices, selected_noise_indices])
    np.random.shuffle(selected_indices)

    # Extract final points, colors, labels
    adjusted_points = points[selected_indices]
    adjusted_labels = labels[selected_indices]
    if colors is not None:
        adjusted_colors = colors[selected_indices]
    else:
        adjusted_colors = None

    # Create new point cloud
    adjusted_pcd = o3d.geometry.PointCloud()
    adjusted_pcd.points = o3d.utility.Vector3dVector(adjusted_points)
    if adjusted_colors is not None:
        adjusted_pcd.colors = o3d.utility.Vector3dVector(adjusted_colors)

    return adjusted_pcd, adjusted_labels


def normalize_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Normalizes the point cloud so that it fits inside a unit sphere
    centered at the origin. Subtract the centroid and scale by max distance.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.

    Returns:
        pcd (o3d.geometry.PointCloud): Normalized point cloud.
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    centroid = np.mean(points, axis=0)
    points -= centroid
    max_distance = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if max_distance > 1e-12:
        points /= max_distance
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def preprocess_point_clouds(
    input_dir: str,
    output_dir: str,
    voxel_size: float,
    num_points: int,
    file_ext: str = ".npz"
) -> None:
    """
    Preprocesses point clouds by:
      1) Downsampling (voxel grid)
      2) Adjusting total point count
      3) Normalizing
      4) Saving final data

    We assume each input file has:
       - 'points'  => shape (N,3)
       - 'labels'  => shape (N,)   (binary, 0 or 1)
       - (optional) 'colors' => shape (N,3) if you want to keep original color

    The final result is saved as .npz with 'points', 'labels', (and 'colors' if present).

    Args:
        input_dir (str): Directory with .npz files containing point data + labels.
        output_dir (str): Where to save the preprocessed .npz files.
        voxel_size (float): Voxel size for downsampling.
        num_points (int): Desired number of points after adjusting.
        file_ext (str): Extension of input files (default ".npz").
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith(file_ext)]
    if not files:
        print(f"No {file_ext} files found in {input_dir}.")
        return

    for fname in files:
        input_path = os.path.join(input_dir, fname)
        data = np.load(input_path)

        points = data["points"]
        labels = data["labels"]
        colors = data["colors"] if "colors" in data.files else None

        # Rebuild pcd for open3d
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # === 1) Downsample (voxel grid) ===
        pcd_down, idx_down = voxel_down_sample_with_indices(pcd, voxel_size)
        labels_down = labels[idx_down]  # filter labels by chosen indices

        # === 2) Adjust point count (preserve ratio of plant vs. non-plant) ===
        pcd_adj, labels_adj = adjust_point_count_with_indices(pcd_down, labels_down, num_points)

        # === 3) Normalize ===
        pcd_norm = normalize_point_cloud(pcd_adj)

        # Convert back to arrays
        final_points = np.asarray(pcd_norm.points)
        final_labels = labels_adj
        if pcd_norm.has_colors():
            final_colors = np.asarray(pcd_norm.colors)
        else:
            final_colors = None

        # === 4) Save final .npz ===
        out_path = os.path.join(output_dir, fname)
        if final_colors is not None:
            np.savez(out_path, points=final_points, labels=final_labels, colors=final_colors)
        else:
            np.savez(out_path, points=final_points, labels=final_labels)

        print(f"Preprocessed => {out_path}  [points: {len(final_points)}]")
