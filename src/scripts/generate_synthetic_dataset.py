#!/usr/bin/env python3

import os
import numpy as np

def load_plant_point_cloud(file_path: str) -> np.ndarray:
    """
    Load a real plant from a .txt with lines 'x y z'.
    """
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split()
            if len(vals) < 3:
                continue
            x, y, z = vals[0], vals[1], vals[2]
            points.append([float(x), float(y), float(z)])
    return np.array(points)


# ---------------------------
# 1. Geometry/Noise Functions
# ---------------------------
def generate_box(num_points, x_min, x_max, y_min, y_max, z_min, z_max):
    x = np.random.uniform(x_min, x_max, num_points)
    y = np.random.uniform(y_min, y_max, num_points)
    z = np.random.uniform(z_min, z_max, num_points)
    return np.column_stack((x, y, z))

def generate_table(num_points, center, radius, z_value=0.0):
    angles = np.random.uniform(0, 2*np.pi, num_points)
    radii  = np.sqrt(np.random.uniform(0, 1, num_points)) * radius
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    z = np.full(num_points, z_value)
    return np.column_stack((x, y, z))

def generate_random_plane(num_points, x_min, x_max, y_min, y_max, z_value, z_jitter=0.0):
    x = np.random.uniform(x_min, x_max, num_points)
    y = np.random.uniform(y_min, y_max, num_points)
    z = np.random.normal(loc=z_value, scale=z_jitter, size=num_points)
    return np.column_stack((x, y, z))

def generate_random_noise(num_points, region_box_min, region_box_max):
    x_min, y_min, z_min = region_box_min
    x_max, y_max, z_max = region_box_max
    x = np.random.uniform(x_min, x_max, num_points)
    y = np.random.uniform(y_min, y_max, num_points)
    z = np.random.uniform(z_min, z_max, num_points)
    return np.column_stack((x, y, z))

def add_gaussian_noise_to_points(points, sigma=0.002):
    noise = np.random.normal(loc=0.0, scale=sigma, size=points.shape)
    return points + noise

def generate_scanning_artifacts(plant_pts, artifact_ratio=0.1, artifact_sigma=0.005):
    num_plant = plant_pts.shape[0]
    num_artifacts = int(artifact_ratio * num_plant)
    if num_artifacts == 0:
        return np.empty((0, 3))
    seed_indices = np.random.choice(num_plant, size=num_artifacts, replace=False)
    seed_points = plant_pts[seed_indices, :]
    offsets = np.random.normal(loc=0.0, scale=artifact_sigma, size=seed_points.shape)
    return seed_points + offsets

def position_plant(pts, pot_center=(0,0), plant_base_z=1.0):
    min_pt_ = np.min(pts, axis=0)
    max_pt_ = np.max(pts, axis=0)
    center_xy = (min_pt_[:2] + max_pt_[:2]) / 2.0
    min_z_ = min_pt_[2]
    translate_xy = np.array(pot_center) - center_xy
    translate_z = plant_base_z - min_z_
    translation = np.array([translate_xy[0], translate_xy[1], translate_z])
    return pts + translation


# ---------------------------
# 2. Single-file Generation
# ---------------------------
def generate_synthetic_dataset_for_file(
    plant_file: str,
    out_file: str,
    artifact_ratio: float = 0.2,
    artifact_sigma: float = 0.05,
    plant_noise_sigma: float = 0.01,
    pot_diameter_range: tuple = (1.2, 2.0),
    pot_height_range: tuple = (1.2, 2.0),
    table_offset_range: float = 0.15,
    partial_occlusion_prob: float = 0.3,
    partial_occlusion_height_range: tuple = (0.6, 0.9)
) -> None:
    """
    Generates one synthetic dataset (points+labels) from a single plant_file, saving an NPZ.

    Domain Randomization changes:
      - Random pot diameter/height
      - Random table center
      - Partial occlusion chance
      - Heavier plant noise
      - More random scene noise
    """
    # 1) Load real plant
    plant_points = load_plant_point_cloud(plant_file)
    plant_labels = np.ones((plant_points.shape[0],), dtype=int)  # label=1

    # 2) bounding box
    min_pt = np.min(plant_points, axis=0)
    max_pt = np.max(plant_points, axis=0)
    bbox_size = max_pt - min_pt

    # --- domain randomization: pot size
    diameter_factor = np.random.uniform(*pot_diameter_range)
    height_factor   = np.random.uniform(*pot_height_range)

    pot_diameter = diameter_factor * max(bbox_size[0], bbox_size[1])
    pot_height   = height_factor * bbox_size[2]

    # soil thickness
    soil_thickness = 0.05 * pot_height
    # table diameter
    table_diameter = 3.0 * pot_diameter

    # pot bounding
    pot_radius = pot_diameter / 2.0
    pot_x_min, pot_x_max = -pot_radius, +pot_radius
    pot_y_min, pot_y_max = -pot_radius, +pot_radius
    pot_z_min, pot_z_max = 0.0, pot_height

    # pot points
    pot_pts = generate_box(
        3000,
        pot_x_min, pot_x_max,
        pot_y_min, pot_y_max,
        pot_z_min, pot_z_max
    )

    # soil
    soil_pts = generate_random_plane(
        2000,
        pot_x_min, pot_x_max,
        pot_y_min, pot_y_max,
        z_value=pot_z_max,
        z_jitter=0.01 * pot_height
    )

    # table offset
    table_center_x = np.random.uniform(-table_offset_range, table_offset_range)
    table_center_y = np.random.uniform(-table_offset_range, table_offset_range)
    table_radius = table_diameter / 2.0
    base_table_z = -0.05 * pot_height
    table_z_offset = np.random.uniform(-0.02 * pot_height, 0.02 * pot_height)
    table_z = base_table_z + table_z_offset

    table_pts = generate_table(
        3000,
        center=(table_center_x, table_center_y),
        radius=table_radius,
        z_value=table_z
    )

    # 6) heavier Gaussian noise
    plant_points_noisy = add_gaussian_noise_to_points(plant_points, sigma=plant_noise_sigma)

    # 7) partial occlusion
    if np.random.rand() < partial_occlusion_prob:
        cutoff_fraction = np.random.uniform(*partial_occlusion_height_range)
        occlusion_cutoff = min_pt[2] + cutoff_fraction * (max_pt[2] - min_pt[2])
        keep_mask = (plant_points_noisy[:, 2] <= occlusion_cutoff)
        plant_points_noisy = plant_points_noisy[keep_mask]
        plant_labels = plant_labels[keep_mask]

    # 8) position plant
    plant_points_noisy = position_plant(
        plant_points_noisy,
        pot_center=(0, 0),
        plant_base_z=pot_z_max
    )

    # 9) scanning artifacts
    artifact_pts = generate_scanning_artifacts(
        plant_points_noisy,
        artifact_ratio=artifact_ratio,
        artifact_sigma=artifact_sigma
    )

    # 10) random scene noise
    noise_expansion_factor = np.random.uniform(0.5, 1.0)
    noise_min = np.array([
        pot_x_min - pot_radius * noise_expansion_factor,
        pot_y_min - pot_radius * noise_expansion_factor,
        table_z - 0.5 * pot_height * noise_expansion_factor
    ])
    noise_max = np.array([
        pot_x_max + pot_radius * noise_expansion_factor,
        pot_y_max + pot_radius * noise_expansion_factor,
        pot_z_max + 0.5 * pot_height * noise_expansion_factor
    ])
    scene_noise_pts = generate_random_noise(2000, noise_min, noise_max)

    # combine non-plant
    pot_lbls     = np.zeros((pot_pts.shape[0],), dtype=int)
    soil_lbls    = np.zeros((soil_pts.shape[0],), dtype=int)
    table_lbls   = np.zeros((table_pts.shape[0],), dtype=int)
    artifact_lbls= np.zeros((artifact_pts.shape[0],), dtype=int)
    scene_lbls   = np.zeros((scene_noise_pts.shape[0],), dtype=int)

    non_plant_points = np.vstack([
        pot_pts,
        soil_pts,
        table_pts,
        artifact_pts,
        scene_noise_pts
    ])
    non_plant_labels = np.concatenate([
        pot_lbls,
        soil_lbls,
        table_lbls,
        artifact_lbls,
        scene_lbls
    ])

    # final
    all_points = np.vstack([plant_points_noisy, non_plant_points])
    all_labels = np.concatenate([plant_labels, non_plant_labels])

    # save
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez_compressed(out_file, points=all_points, labels=all_labels)
    print(f"[INFO] Synthetic dataset saved: {out_file} with domain randomization")


# ---------------------------
# 3. Multi-file API for main
# ---------------------------
def generate_synthetic_dataset(
    plant_file: str,
    output_file: str,
    artifact_ratio: float = 0.1,
    artifact_sigma: float = 0.005,
    plant_noise_sigma: float = 0.002
) -> None:
    """
    If 'plant_file' is a directory, we generate synthetic dataset for each *.txt inside it,
    saving to 'output_file' directory as <basename>_synthetic.npz.
    If 'plant_file' is single .txt, we generate only one .npz in 'output_file'.
    """
    if os.path.isdir(plant_file):
        if not os.path.exists(output_file):
            os.makedirs(output_file)

        for fname in os.listdir(plant_file):
            if fname.lower().endswith(".txt"):
                in_path = os.path.join(plant_file, fname)
                base_name = os.path.splitext(fname)[0]
                out_path = os.path.join(output_file, f"{base_name}_synthetic.npz")

                # Call the domain-randomized generator
                generate_synthetic_dataset_for_file(
                    plant_file=in_path,
                    out_file=out_path,
                    artifact_ratio=artifact_ratio,
                    artifact_sigma=artifact_sigma,
                    plant_noise_sigma=plant_noise_sigma
                )
    else:
        # single file
        if os.path.isdir(output_file):
            base_name = os.path.splitext(os.path.basename(plant_file))[0]
            out_path = os.path.join(output_file, f"{base_name}_synthetic.npz")
        else:
            out_path = output_file

        generate_synthetic_dataset_for_file(
            plant_file=plant_file,
            out_file=out_path,
            artifact_ratio=artifact_ratio,
            artifact_sigma=artifact_sigma,
            plant_noise_sigma=plant_noise_sigma
        )
