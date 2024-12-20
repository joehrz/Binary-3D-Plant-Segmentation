## src/data_processing/data_augmentation.py

'''Data augmentation to add noise to the point cloud data'''

import os
import numpy as np
import open3d as o3d
from typing import List, Tuple, Dict, Any


def add_random_noise_points(
    points: np.ndarray,
    colors: np.ndarray,
    num_noise_points: int,
    color_options: List[Tuple[float, float, float]] = [
        (0.0, 0.0, 1.0),  # Blue
        (0.0, 0.0, 0.0),  # Black
        (1.0, 1.0, 1.0)   # White
    ],
    extend_ratio: float = 0.1,
    noise_below_ratio: float = 0.7  # Ratio of noise points below the plant
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adds random noise points around the original point cloud to simulate non-plant elements.

    Args:
        points (np.ndarray): Original point cloud data of shape (N, 3).
        colors (np.ndarray): Original colors of shape (N, 3).
        num_noise_points (int): Total number of noise points to add.
        color_options (List[Tuple[float, float, float]]): List of RGB color tuples to assign to noise points.
        extend_ratio (float): Ratio to extend the bounding box of the original points to place noise points.
        noise_below_ratio (float): Proportion of noise points to place below the plant.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            combined_points: Combined point cloud data of shape (N + num_noise_points, 3).
            combined_colors: Combined colors of shape (N + num_noise_points, 3).
    """
    # Compute the bounding box of the original points
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)

    # Extend the bounding box to include space around and above the plant
    extension = (max_xyz - min_xyz) * extend_ratio
    min_xyz_extended = min_xyz - extension
    max_xyz_extended = max_xyz + extension

    # Extend the bounding box further downward to include more space below the plant
    min_xyz_below = min_xyz_extended.copy()
    min_xyz_below[2] -= extension[2] * 2  # Extend more in the negative Z direction (assuming Z is up)

    # Determine the number of noise points below and above/around the plant
    num_noise_below = int(num_noise_points * noise_below_ratio)
    num_noise_above = num_noise_points - num_noise_below

    # Generate noise points below the plant
    noise_points_below = np.random.uniform(
        low=min_xyz_below,
        high=np.array([max_xyz_extended[0], max_xyz_extended[1], min_xyz_extended[2]]),
        size=(num_noise_below, 3)
    )

    # Generate noise points around and above the plant
    noise_points_above = np.random.uniform(
        low=min_xyz_extended,
        high=max_xyz_extended,
        size=(num_noise_above, 3)
    )

    # Combine noise points
    noise_points = np.vstack((noise_points_below, noise_points_above))

    # Assign colors to the noise points from the given color options
    noise_colors = np.array(color_options)[
        np.random.choice(len(color_options), size=num_noise_points)
    ]

    # Preserve original colors
    if colors is not None and len(colors) == len(points):
        original_colors = colors
    else:
        # If colors are not available, assign default color (e.g., green)
        original_colors = np.full((points.shape[0], 3), [0.0, 1.0, 0.0])  # RGB for green

    # Labels: 1 for plant points, 0 for noise points
    plant_labels = np.ones(len(points), dtype=int)
    noise_labels = np.zeros(num_noise_points, dtype=int)
    
    # Combine original points and noise points
    combined_points = np.vstack((points, noise_points))
    combined_colors = np.vstack((original_colors, noise_colors))
    labels = np.concatenate((plant_labels, noise_labels))


    return combined_points, combined_colors, labels

def process_point_clouds(
    input_dir: str,
    output_dir: str,
    noise_params: Dict[str, Any]
) -> None:
    """
    Processes point clouds by adding random noise points and saving the result.

    Args:
        input_dir (str): Directory containing original point clouds.
        output_dir (str): Directory to save the noisy point clouds.
        noise_params (Dict[str, Any]): Parameters for the noise function.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.ply'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load point cloud
            pcd = o3d.io.read_point_cloud(input_path)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None

            # Add random noise points
            combined_points, combined_colors, _ = add_random_noise_points(
                points,
                colors,
                **noise_params  # Pass the parameters as keyword arguments
            )

            # Create new point cloud
            noisy_pcd = o3d.geometry.PointCloud()
            noisy_pcd.points = o3d.utility.Vector3dVector(combined_points)
            noisy_pcd.colors = o3d.utility.Vector3dVector(combined_colors)

            # Save noisy point cloud
            o3d.io.write_point_cloud(output_path, noisy_pcd)
            print(f"Saved noisy point cloud to {output_path}")


if __name__ == '__main__':
    input_directory = 'data/raw/plant_only'
    output_directory = 'data/raw/plant_only_noisy'
    noise_parameters = {
        'num_noise_points': 1000,  # Adjust as needed
        'color_options': [
            (0.0, 0.0, 1.0),  # Blue
            (0.0, 0.0, 0.0),  # Black
            (1.0, 1.0, 1.0)   # White
        ],
        'extend_ratio': 0.1,     # Adjust as needed
        'noise_below_ratio': 0.7 # 70% of noise points below the plant
    }

    process_point_clouds(input_directory, output_directory, noise_parameters)