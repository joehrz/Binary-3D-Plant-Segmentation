## src/scripts/batch_threshold_dbscan.py


"""
This script performs semi-automatic labeling of plant vs. non-plant:
  1) Reads raw .ply files from a specified directory.
  2) Computes ExG + Otsu threshold to identify potential plant points.
  3) Applies DBSCAN to remove small/noisy clusters, labeling remaining clusters as plant.
  4) Colors the .ply file for easy visualization (green=plant, red=non-plant).
  5) Saves both a .ply (for manual inspection) and a .npz (with points, original colors, and final labels).

Usage:
    python src/scripts/batch_threshold_dbscan.py
"""

import os
import numpy as np
import open3d as o3d
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def process_point_cloud(
    input_path: str,
    output_ply_path: str,
    output_npz_path: str,
    dbscan_eps: float = 0.02,
    dbscan_min_samples: int = 20,
    min_cluster_size: int = 10000,
    debug: bool = False
) -> None:
    """
    Process a single raw .ply point cloud, label each point as plant(1) or non-plant(0),
    and save the results for visualization and future use.

    Steps:
      1) Load .ply point cloud with RGB.
      2) Compute ExG index => 2*g - r - b.
      3) Otsu threshold on ExG (optionally smoothed) to form an initial "plant" mask.
      4) DBSCAN to remove small/noisy clusters, keep only clusters >= min_cluster_size.
      5) Write a .ply (colored green=1, red=0) + a .npz with {points, orig_colors, labels}.

    Args:
        input_path (str): Full path to the input .ply file containing RGB.
        output_ply_path (str): Path where the color-labeled .ply will be saved.
        output_npz_path (str): Path where the final data (.npz) is saved (points, orig_colors, labels).
        dbscan_eps (float): DBSCAN radius parameter for clustering plant points.
        dbscan_min_samples (int): Minimum samples for DBSCAN to consider a cluster.
        min_cluster_size (int): Minimum cluster size to keep in the final plant mask.
        debug (bool): If True, prints additional debugging info.

    Returns:
        None. Writes output to disk.
    """
    # 1. Load the raw point cloud
    pcd = o3d.io.read_point_cloud(input_path)
    if not pcd.has_colors():
        raise ValueError(f"Point cloud '{input_path}' does not contain RGB information.")

    points = np.asarray(pcd.points)         # shape: (N, 3)
    orig_colors = np.asarray(pcd.colors)    # shape: (N, 3) in [0, 1]

    if debug:
        print(f"Loaded '{input_path}' with {len(points)} points.")

    # 2. Compute ExG => 2*g - r - b
    sum_rgb = np.sum(orig_colors, axis=1) + 1e-6
    r = orig_colors[:, 0] / sum_rgb
    g = orig_colors[:, 1] / sum_rgb
    b = orig_colors[:, 2] / sum_rgb
    ExG = 2 * g - r - b

    # 3. Otsu threshold on ExG (optionally smoothed)
    ExG_smooth = gaussian_filter1d(ExG, sigma=1)
    otsu_thresh = threshold_otsu(ExG_smooth)
    if debug:
        print(f"Otsu threshold for ExG: {otsu_thresh:.4f}")

    plant_mask = (ExG_smooth >= otsu_thresh)

    if debug:
        print(f"Initial plant points (before DBSCAN): {np.sum(plant_mask)} of {len(plant_mask)}")

    # 4. DBSCAN on the filtered plant points
    filtered_indices = np.where(plant_mask)[0]
    filtered_points = points[filtered_indices]
    scaled_filtered = StandardScaler().fit_transform(filtered_points)

    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels_db = db.fit_predict(scaled_filtered)

    unique_labels, counts = np.unique(labels_db, return_counts=True)
    n_clusters = np.sum(unique_labels >= 0)  # ignoring noise label -1
    n_noise = (labels_db == -1).sum()
    if debug:
        print(f"DBSCAN => clusters: {n_clusters}, noise points: {n_noise}")

    # Keep clusters >= min_cluster_size
    significant_clusters = unique_labels[(unique_labels != -1) & (counts >= min_cluster_size)]
    significant_mask = np.isin(labels_db, significant_clusters)

    # 5. Map back to original indices => final "plant" points
    significant_indices_filtered = np.where(significant_mask)[0]
    significant_indices_original = filtered_indices[significant_indices_filtered]

    final_labels = np.zeros(len(points), dtype=int)
    final_labels[significant_indices_original] = 1

    if debug:
        num_plants = np.sum(final_labels == 1)
        num_non_plants = len(final_labels) - num_plants
        print(f"Final labeling => plant: {num_plants}, non-plant: {num_non_plants}")

    # Overwrite color for visualization (red=0, green=1)
    viz_colors = np.zeros((len(points), 3))
    viz_colors[final_labels == 1] = [0, 1, 0]  # green
    viz_colors[final_labels == 0] = [1, 0, 0]  # red

    labeled_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labeled_pcd.colors = o3d.utility.Vector3dVector(viz_colors)

    o3d.io.write_point_cloud(output_ply_path, labeled_pcd, write_ascii=True)
    print(f"[Saved labeled .ply =>] {output_ply_path}")

    np.savez(
        output_npz_path,
        points=points,         # shape: (N, 3)
        colors=orig_colors,    # shape: (N, 3)
        labels=final_labels    # shape: (N,)
    )
    print(f"[Saved .npz =>] {output_npz_path}")


def batch_threshold_dbscan(
    raw_dir: str = "data/raw",
    out_dir: str = "data/manually_adjustments",
    dbscan_eps: float = 0.02,
    dbscan_min_samples: int = 20,
    min_cluster_size: int = 10000,
    debug: bool = False
) -> None:
    """
    Batch process all .ply files in 'raw_dir' using ExG + Otsu threshold + DBSCAN.
    Saves labeled .ply + .npz in 'out_dir'.

    Args:
        raw_dir (str): Path to directory with .ply files.
        out_dir (str): Directory to store labeled .ply + .npz.
        dbscan_eps (float): DBSCAN eps parameter (distance threshold for cluster expansion).
        dbscan_min_samples (int): Minimum samples for DBSCAN to form a cluster.
        min_cluster_size (int): Minimum cluster size to retain as "plant."
        debug (bool): If True, prints additional logging info.
    """
    os.makedirs(out_dir, exist_ok=True)
    ply_files = [f for f in os.listdir(raw_dir) if f.endswith('.ply')]
    if not ply_files:
        print(f"No .ply files found in '{raw_dir}'. Nothing to process.")
        return

    print("===== Starting batch_threshold_dbscan =====")
    print(f"Input directory: '{raw_dir}'")
    print(f"Output directory: '{out_dir}'\n")

    for fname in ply_files:
        in_path = os.path.join(raw_dir, fname)
        base_name, _ = os.path.splitext(fname)
        out_ply = os.path.join(out_dir, base_name + "_labeled.ply")
        out_npz = os.path.join(out_dir, base_name + "_labeled.npz")

        print(f"Processing => {in_path}")
        process_point_cloud(
            input_path=in_path,
            output_ply_path=out_ply,
            output_npz_path=out_npz,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            min_cluster_size=min_cluster_size,
            debug=debug
        )
        print("-" * 60)

    print("\n===== Finished batch_threshold_dbscan =====")