## scr/scripts/adjust_labels.py

""" This script:

    Takes the combined labeled .ply + .npy (with green=plant, red=non-plant)
    Also takes the Meshlab-edited plant-only .ply (where wrong points were removed).
    Finds which original plant points are “missing” in the new plant-only file → sets their label to 0.
    Saves the final updated labels as .npz (and optionally a final .ply). """


import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def adjust_labels_after_meshlab(
    combined_ply_path,
    combined_labels_path,
    plant_only_fixed_path,
    output_npz_path,
    match_threshold=1e-5
):
    """
    Compare the original combined labeled cloud (N points, label=0/1) to the
    Meshlab-edited plant-only cloud. If a 'plant' point was removed in Meshlab,
    we set that point's label to 0 in the final label array.

    Args:
        combined_ply_path (str): Path to combined .ply (with colors).
        combined_labels_path (str): Path to .npy => shape (N,) with 0/1 labels.
        plant_only_fixed_path (str): Meshlab-edited .ply for plant points only.
        output_npz_path (str): Where to save final { points, labels } as .npz.
        match_threshold (float): Max distance for a point to be considered "kept."
    """
    # 1. Load the combined point cloud
    print(f"Loading combined cloud: {combined_ply_path}")
    pcd_combined = o3d.io.read_point_cloud(combined_ply_path)
    combined_points = np.asarray(pcd_combined.points)
    combined_labels = np.load(combined_labels_path)

    if len(combined_points) != len(combined_labels):
        raise ValueError("Mismatch in length: points vs. labels.")

    # Indices of originally labeled "plant" points
    plant_indices = np.where(combined_labels == 1)[0]
    plant_points_original = combined_points[plant_indices]  # shape (M,3)

    print(f"Loading Meshlab-fixed plant-only cloud: {plant_only_fixed_path}")
    pcd_plant_fixed = o3d.io.read_point_cloud(plant_only_fixed_path)
    plant_only_fixed_points = np.asarray(pcd_plant_fixed.points)

    # 2. Build KD tree for the fixed plant-only points
    tree = cKDTree(plant_only_fixed_points)
    # For each originally-plant points, find the nearest neighbor in fixed cloud
    _, nn_dist = tree.query(plant_points_original, k=1, n_jobs=-1)
    # Points with distance > match_threshold => must've been deleted in Meshlab
    retained_mask = nn_dist < match_threshold

    removed_count = np.sum(~retained_mask)
    print(f"Originally had {len(plant_points_original)} plant points.")
    print(f"Meshlab-deleted or missing: {removed_count}")

    # 3. Convert local indices back to global indices
    removed_indices_global = plant_indices[~retained_mask]
    final_labels = combined_labels.copy()
    final_labels[removed_indices_global] = 0

    # (Optional) create a final color-coded .ply for visualization
    final_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(combined_points))
    final_colors = np.zeros((len(final_pcd.points), 3))
    final_colors[final_labels == 1] = [0,1,0]  # green
    final_colors[final_labels == 0] = [1,0,0]  # red
    final_pcd.colors = o3d.utility.Vector3dVector(final_colors)

    # Save final npz
    print(f"Saving final .npz => {output_npz_path}")
    np.savez(output_npz_path, points=combined_points, labels=final_labels)

    # Also save a final .ply if you wish
    final_ply_path = output_npz_path.replace('.npz', '_final.ply')
    o3d.io.write_point_cloud(final_ply_path, final_pcd)
    print(f"Saved final .ply => {final_ply_path}")

if __name__ == "__main__":
    # Example usage
    BASE = "data/manually_adjustments"
    combined_ply = os.path.join(BASE, "combined_cloud_labelled.ply")
    combined_labels = os.path.join(BASE, "combined_cloud_labelled_labels.npy")
    plant_fixed_ply = os.path.join(BASE, "plant_only_fixed.ply")
    output_npz = os.path.join(BASE, "final_adjusted_labels.npz")

    adjust_labels_after_meshlab(
        combined_ply_path=combined_ply,
        combined_labels_path=combined_labels,
        plant_only_fixed_path=plant_fixed_ply,
        output_npz_path=output_npz,
        match_threshold=1e-5
    )