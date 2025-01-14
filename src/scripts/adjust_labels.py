# src/scripts/adjust_labels.py

"""
This script batch-processes multiple labeled .ply/.npz files by comparing each
to two possible Meshlab-edited clouds:
  1) <basename>_labeled_plant_only_fixed.ply
     - Contains only correct green points. We remove any green that is not in here.
  2) <basename>_labeled_nonplant_only_fixed.ply
     - Contains only those red points that should actually be green. We flip them to green.

Dictionary Approach (exact coordinate matching):
  - We assume NO transforms or scaling in Meshlab.
  - For each point, we round coords to reduce float noise (decimals=6 by default).
  - Use sets to find which points remain or should be added as green.

Nomenclature:
  For each base name => <basename>_labeled.ply, <basename>_labeled.npz
    Possibly:
      <basename>_labeled_plant_only_fixed.ply      (Optional)
      <basename>_labeled_nonplant_only_fixed.ply   (Optional)

Steps (batch_adjust_labels):
  1) In 'manual_dir', find all .npz with suffix_labeled (e.g. '_labeled.npz').
  2) For each file => define:
       <basename>_labeled.ply
       <basename>_labeled.npz
       <basename>_labeled_plant_only_fixed.ply      (if present => remove green not in here)
       <basename>_labeled_nonplant_only_fixed.ply   (if present => add green from here)
  3) We call 'adjust_labels_for_file(...)' to:
       - Load combined (points, labels).
       - If _plant_only_fixed exists => remove missing green
       - If _nonplant_only_fixed exists => flip those red to green
       - Save a final .npz + color-coded .ply.

SKIP ALREADY-PROCESSED:
  If we detect <basename>_final.npz is present, we skip re-processing.

IMPORTANT NOTE:
  - This dictionary approach ONLY works if Meshlab didn't alter or transform
    coordinates. If you see everything turn red or can't add green, likely
    coords changed.
"""

import os
import numpy as np
import open3d as o3d

# -----------------------------------------------------------------------------
# Dictionary-based method to remove green
# -----------------------------------------------------------------------------
def dictionary_remove_green(
    combined_points: np.ndarray,
    labels: np.ndarray,
    plant_points_fixed: np.ndarray,
    decimals: int = 6
) -> np.ndarray:
    """
    For each point originally labeled green (labels==1), keep it if the point's coords
    are found in 'plant_points_fixed'. Otherwise set label=0.

    We assume 'plant_points_fixed' has the correct green points only.
    """
    final_labels = labels.copy()

    plant_indices = np.where(labels == 1)[0]
    plant_points_original = combined_points[plant_indices]
    print(f"[RemoveGreen] originally-labeled green => {len(plant_indices)}")

    # Build a set for coords in plant_points_fixed
    fixed_set = set()
    for pt in plant_points_fixed:
        key = (
            round(pt[0], decimals),
            round(pt[1], decimals),
            round(pt[2], decimals)
        )
        fixed_set.add(key)

    removed_count = 0
    for idx, pindex in enumerate(plant_indices):
        pt = plant_points_original[idx]
        key = (
            round(pt[0], decimals),
            round(pt[1], decimals),
            round(pt[2], decimals)
        )
        if key not in fixed_set:
            final_labels[pindex] = 0
            removed_count += 1

    print(f"[RemoveGreen] => {removed_count} of {len(plant_indices)} green points removed.\n")
    return final_labels


# -----------------------------------------------------------------------------
# Dictionary-based method to add green
# -----------------------------------------------------------------------------
def dictionary_add_green(
    combined_points: np.ndarray,
    labels: np.ndarray,
    nonplant_points_fixed: np.ndarray,
    decimals: int = 6
) -> np.ndarray:
    """
    For each point originally labeled red (labels==0), if that point's coords
    appear in 'nonplant_points_fixed', we flip label to 1 (green).

    We assume 'nonplant_points_fixed' is a set of red points that the user
    decided should be green.
    """
    final_labels = labels.copy()

    red_indices = np.where(labels == 0)[0]
    red_points_original = combined_points[red_indices]
    print(f"[AddGreen] originally-labeled red => {len(red_indices)}")

    # Build a set for coords in nonplant_points_fixed
    fixed_set = set()
    for pt in nonplant_points_fixed:
        key = (
            round(pt[0], decimals),
            round(pt[1], decimals),
            round(pt[2], decimals)
        )
        fixed_set.add(key)

    added_count = 0
    for idx, rindex in enumerate(red_indices):
        pt = red_points_original[idx]
        key = (
            round(pt[0], decimals),
            round(pt[1], decimals),
            round(pt[2], decimals)
        )
        if key in fixed_set:
            # Flip label => 1
            final_labels[rindex] = 1
            added_count += 1

    print(f"[AddGreen] => {added_count} of {len(red_indices)} red points turned green.\n")
    return final_labels

# -----------------------------------------------------------------------------
def adjust_labels_for_file(
    labeled_ply_path: str,
    labeled_npz_path: str,
    plant_fixed_ply: str,       # e.g. <basename>_labeled_plant_only_fixed.ply
    nonplant_fixed_ply: str,    # e.g. <basename>_labeled_nonplant_only_fixed.ply
    output_npz_path: str,
    decimals: int = 6
) -> None:
    """
    Adjust labels in a single file using dictionary-based approach for BOTH:
      1) Removing incorrectly-labeled green using 'plant_fixed_ply' (if present)
      2) Adding incorrectly-labeled red => green using 'nonplant_fixed_ply' (if present)

    Steps:
      - Load original cloud (points, labels).
      - If 'plant_fixed_ply' exists => remove any green not in that file.
      - If 'nonplant_fixed_ply' exists => add green for any red points in that file.
      - Save final .npz + color-coded .ply
    """
    print(f"\n[INFO] Adjusting labels => {labeled_ply_path}")
    print(f"  Labeled NPZ => {labeled_npz_path}")
    print(f"  Plant-Fixed PLY => {plant_fixed_ply if os.path.exists(plant_fixed_ply) else '(missing)'}")
    print(f"  NonPlant-Fixed PLY => {nonplant_fixed_ply if os.path.exists(nonplant_fixed_ply) else '(missing)'}")

    # 1) Load original
    pcd_combined = o3d.io.read_point_cloud(labeled_ply_path)
    combined_points = np.asarray(pcd_combined.points)

    data = np.load(labeled_npz_path)
    labels = data["labels"]

    if len(combined_points) != len(labels):
        raise ValueError(f"Mismatch: {len(combined_points)} points vs {len(labels)} labels")

    # 2) If 'plant_fixed_ply' exists => remove green not in that file
    final_labels = labels.copy()
    if os.path.exists(plant_fixed_ply):
        pcd_plant_fixed = o3d.io.read_point_cloud(plant_fixed_ply)
        plant_points_fixed = np.asarray(pcd_plant_fixed.points)
        if len(plant_points_fixed) == 0:
            print("[WARNING] plant_only_fixed file is empty => all green -> 0")
            plant_indices = np.where(final_labels == 1)[0]
            final_labels[plant_indices] = 0
        else:
            final_labels = dictionary_remove_green(
                combined_points, final_labels, plant_points_fixed, decimals
            )
    else:
        print("  => No 'plant_only_fixed' file found. Skipping remove-green step.")

    # 3) If 'nonplant_fixed_ply' exists => add green for any red in that file
    if os.path.exists(nonplant_fixed_ply):
        pcd_nonplant_fixed = o3d.io.read_point_cloud(nonplant_fixed_ply)
        nonplant_points_fixed = np.asarray(pcd_nonplant_fixed.points)
        if len(nonplant_points_fixed) == 0:
            print("[WARNING] nonplant_only_fixed file is empty => no red->green changed")
        else:
            final_labels = dictionary_add_green(
                combined_points, final_labels, nonplant_points_fixed, decimals
            )
    else:
        print("  => No 'nonplant_only_fixed' file found. Skipping add-green step.")

    # 4) Save final result
    final_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(combined_points))
    final_colors = np.zeros((len(final_pcd.points), 3))
    final_colors[final_labels == 1] = [0,1,0]  # green
    final_colors[final_labels == 0] = [1,0,0]  # red
    final_pcd.colors = o3d.utility.Vector3dVector(final_colors)

    print(f"  => Saving final .npz => {output_npz_path}")
    np.savez(output_npz_path, points=combined_points, labels=final_labels)

    final_ply_path = output_npz_path.replace(".npz", "_final.ply")
    o3d.io.write_point_cloud(final_ply_path, final_pcd)
    print(f"  => Final .ply => {final_ply_path}")


def batch_adjust_labels(
    manual_dir: str = "data/manually_adjustments",
    suffix_labeled: str = "_labeled",
    suffix_plant_fixed: str = "_labeled_plant_only_fixed",
    suffix_nonplant_fixed: str = "_labeled_nonplant_only_fixed",
    decimals: int = 6
) -> None:
    """
    Batch dictionary-based approach for all .npz with suffix_labeled in 'manual_dir'.

    For each base_name, we look for:
      <base_name>_labeled.ply
      <base_name>_labeled.npz
      optional:
        <base_name>_labeled_plant_only_fixed.ply
        <base_name>_labeled_nonplant_only_fixed.ply
    We'll produce <base_name>_final.npz (plus _final.ply).

    SKIP if <base_name>_final.npz already exists.

    Steps:
      1) If plant_only_fixed => we remove green not in that file
      2) If nonplant_only_fixed => we add green for red points found in that file
    """
    npz_files = [
        f for f in os.listdir(manual_dir)
        if f.endswith(suffix_labeled + ".npz")
    ]
    if not npz_files:
        print(f"[INFO] No .npz found in '{manual_dir}' with suffix '{suffix_labeled}'. Nothing to do.")
        return

    print("[INFO] Starting dictionary-based label adjustments (green->red + red->green).")

    for npz_file in npz_files:
        base_name = npz_file.replace(suffix_labeled + ".npz", "")

        labeled_ply_path = os.path.join(manual_dir, base_name + suffix_labeled + ".ply")
        labeled_npz_path = os.path.join(manual_dir, npz_file)

        # Optionally: plant_only & nonplant_only
        plant_fixed_ply    = os.path.join(manual_dir, base_name + suffix_plant_fixed + ".ply")
        nonplant_fixed_ply = os.path.join(manual_dir, base_name + suffix_nonplant_fixed + ".ply")

        output_npz_path = os.path.join(manual_dir, base_name + "_final.npz")

        # 1) skip if final already exists
        if os.path.exists(output_npz_path):
            print(f"[SKIP] {output_npz_path} already exists.")
            continue

        # 2) check input existence
        if not os.path.exists(labeled_ply_path):
            print(f"  [WARNING] Missing => {labeled_ply_path}. Skipping.")
            continue
        if not os.path.exists(labeled_npz_path):
            print(f"  [WARNING] Missing => {labeled_npz_path}. Skipping.")
            continue

        # 3) do the combined logic
        adjust_labels_for_file(
            labeled_ply_path=labeled_ply_path,
            labeled_npz_path=labeled_npz_path,
            plant_fixed_ply=plant_fixed_ply,
            nonplant_fixed_ply=nonplant_fixed_ply,
            output_npz_path=output_npz_path,
            decimals=decimals
        )

    print("[INFO] Done adjusting labels for all files.\n")


if __name__ == "__main__":
    # Example usage:
    batch_adjust_labels(
        manual_dir="data/manually_adjustments",
        suffix_labeled="_labeled",
        suffix_plant_fixed="_labeled_plant_only_fixed",
        suffix_nonplant_fixed="_labeled_nonplant_only_fixed",
        decimals=6
    )
