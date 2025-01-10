## src/scripts/batch_threshold_dbscan.py

""" This script:

    Loops over .ply files in data/raw.
    Computes ExG + Otsu thresholding + DBSCAN to label “plant” vs. “non-plant.”
    Colors the result (green=plant, red=non-plant) and saves .ply + a .npy label array to data/manually_adjustments/. """

import os
import numpy as np
import open3d as o3d
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def process_point_cloud(
    input_path,
    output_path,
    white_threshold=0.3,
    dbscan_eps=0.02,
    dbscan_min_samples=20,
    min_cluster_size=10000,
    debug=False
):
    """
    Computes the ExG + Otsu thresholding + DBSCAN to label “plant” vs. “non-plant 1 => plant, 0 => non-plant.
    Colors the result (green=plant, red=non-plant) and saves .ply + a .npy label array to data/manually_adjustments/.

    Args:
        input_path (str): Path to raw .ply (with colors).
        output_path (str): Path to .npy => shape (N,) with 0/1 labels.
        white_threshold (float): Threshold for white color
        dbscan_eps (float): 
        dbscan_min_samples (int): 
        min_cluster_size (int):
    """

    # 1. Load the raw point cloud
    pcd = o3d.io.read_point_cloud(input_path)
    if not pcd.has_colors():
        raise ValueError(f"Point cloud '{input_path}' has no RGB colors.")

    points = np.asarray(pcd.points)   # (N, 3)
    colors = np.asarray(pcd.colors)  # (N, 3) in [0,1]

    # 2. Normalize RGB => r,g,b in [0,1], then compute ExG
    sum_rgb = np.sum(colors, axis=1) + 1e-6
    r = colors[:, 0] / sum_rgb
    g = colors[:, 1] / sum_rgb
    b = colors[:, 2] / sum_rgb
    ExG = 2*g - r - b

    # 3. Otsu threshold on ExG
    ExG_smooth = gaussian_filter1d(ExG, sigma=1)
    otsu_thresh = threshold_otsu(ExG_smooth)

    # 4. Exclude "white" points
    white_mask = (r > white_threshold) & (g > white_threshold) & (b > white_threshold)

    # 5. Initial plant mask
    plant_mask = ExG_smooth >= otsu_thresh
    final_mask = plant_mask & (~white_mask)

    # 6. DBSCAN on the filtered plant points
    filtered_indices = np.where(final_mask)[0]
    filtered_points = points[filtered_indices]
    filtered_points_scaled = StandardScaler().fit_transform(filtered_points)
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels_db = db.fit_predict(filtered_points_scaled)
    unique_labels, counts = np.unique(labels_db, return_counts=True)

    # 7. Keep clusters above min_cluster_size (and exclude noise label -1)
    significant_clusters = unique_labels[
        (unique_labels != -1) & (counts >= min_cluster_size)
    ]
    significant_mask = np.isin(labels_db, significant_clusters)

    # 8. Map back to original indices
    significant_indices_filtered = np.where(significant_mask)[0]
    significant_indices_original = filtered_indices[significant_indices_filtered]

    # 9. Build the final label array for all points
    #    1 => plant, 0 => non-plant
    final_labels = np.zeros(len(points), dtype=int)
    final_labels[significant_indices_original] = 1

    # 10. Color them green or red in the combined cloud
    new_colors = np.zeros((len(points), 3))
    new_colors[final_labels == 1] = [0,1,0]  # green
    new_colors[final_labels == 0] = [1,0,0]  # red

    pcd.colors = o3d.utility.Vector3dVector(new_colors)

    # 11. Save the combined cloud with color-coded labels
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
    print(f"[Saved labeled .ply =>] {output_path}")

    # 12. Also save the label array (shape: (N,)) as .npy
    label_npy_path = output_path.replace('.ply', '_labels.npy')
    np.save(label_npy_path, final_labels)
    print(f"[Saved label array =>] {label_npy_path}")


def batch_threshold_dbscan(
    raw_dir="data/raw",
    out_dir="data/manually_adjustments",
    white_threshold=0.3,
    dbscan_eps=0.02,
    dbscan_min_samples=20,
    min_cluster_size=10000,
    debug=False
):
    os.makedirs(out_dir, exist_ok=True)
    ply_files = [f for f in os.listdir(raw_dir) if f.endswith('.ply')]
    if not ply_files:
        print(f"No .ply in {raw_dir}!")
        return
    
    for fname in ply_files:
        in_path = os.path.join(raw_dir, fname)
        out_path = os.path.join(out_dir, fname)
        print(f"Processing => {in_path}")
        process_point_cloud(
            in_path, out_path,
            white_threshold=white_threshold,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            min_cluster_size=min_cluster_size,
            debug=debug
        )

if __name__ == "__main__":
    # Simple usage example
    batch_threshold_dbscan()