## src/utils/augmentation_utils.py

'''Rotate and apply jitter to point cloud data'''

import numpy as np

def rotate_point_cloud(points: np.ndarray, angle: float, axis: str = 'y') -> np.ndarray:
    """
    Rotates the point cloud around a specified axis.

    Args:
        points (np.ndarray): Original points of shape (N, 3).
        angle (float): Rotation angle in radians.
        axis (str): Axis to rotate around ('x', 'y', or 'z').

    Returns:
        np.ndarray: Rotated points of shape (N, 3).
    """
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, c, -s],
                                    [0, s, c]])
    elif axis == 'y':
        rotation_matrix = np.array([[c, 0, s],
                                    [0, 1, 0],
                                    [-s, 0, c]])
    elif axis == 'z':
        rotation_matrix = np.array([[c, -s, 0],
                                    [s, c, 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    rotated_points = points @ rotation_matrix.T
    return rotated_points

def jitter_point_cloud(points: np.ndarray, sigma: float = 0.005, clip: float = 0.02) -> np.ndarray:
    """
    Applies random jitter to the point cloud.

    Args:
        points (np.ndarray): Original points of shape (N, 3).
        sigma (float): Standard deviation of the jitter.
        clip (float): Clipping value for the jitter.

    Returns:
        np.ndarray: Jittered points of shape (N, 3).
    """
    jitter = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    jittered_points = points + jitter
    return jittered_points