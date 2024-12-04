## src/utils/data_utils.py

'''Load and save point cloud data'''

import open3d as o3d
import numpy as np
from typing import Tuple

def load_point_cloud(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a point cloud from a .ply file.

    Args:
        file_path (str): Path to the .ply file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Points of shape (N, 3) and colors of shape (N, 3), or None if not available.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return points, colors

def save_point_cloud(points: np.ndarray, colors: np.ndarray, file_path: str) -> None:
    """
    Saves a point cloud to a .ply file.

    Args:
        points (np.ndarray): Points of shape (N, 3).
        colors (np.ndarray): Colors of shape (N, 3), or None.
        file_path (str): Path to save the .ply file.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file_path, pcd)

