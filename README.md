# Wheat-Point-Cloud-Binary-Segmentation

A **binary semantic segmentation** pipeline that filters **plant** vs. **non-plant** points in 3D wheat point clouds. We treat “plant” as class 1 and “non-plant” as class 0, effectively learning to segment plant structures from soil, background, or other artifacts.

## 1. Introduction

Many real-world agricultural scans contain **millions** of points, with only a subset representing the actual **plant** structures. This project provides a **streamlined** pipeline to **automatically** label plant vs. non-plant points (binary semantic segmentation), **manually correct** mistakes, **downsample/normalize** the data for neural network input, and **train** a deep learning model (e.g., PointNet++). Ultimately, you obtain a robust classifier to handle new 3D point clouds, filtering out non-plant regions.

### Why Binary Semantic Segmentation?

- **Simplifies** the problem to just two classes: plant (1) vs. non-plant (0).  
- **Enhances** subsequent analysis (e.g., measuring plant geometry or extracting morphological traits) by removing background clutter.  
- **Scalable** for large farmland or greenhouse point clouds with complex backgrounds.

## 2. Pipeline Overview

1. **Auto Labeling**: Use color-based **ExG** threshold + **DBSCAN** clustering to guess which points are plant vs. non-plant.  
2. **Manual Edits**: Load the resulting color-coded point cloud (`_labeled.ply`) into Meshlab to correct mislabeled points (especially false positives). Save those edits.  
3. **Label Adjustment**: Reconcile your Meshlab changes with the original labeled data, flipping incorrect green points to red (and optionally red to green if needed).  
4. **Preprocessing**: Downsample (e.g., 4K points) and normalize each cloud for consistent neural network input.  
5. **Split**: Divide the dataset into train/val/test sets.  
6. **Train**: Learn a **PointNet++** (or other 3D net) to classify each point as plant vs. non-plant.  
7. **Evaluate**: Measure accuracy, precision, recall, IoU, etc. on unseen test data.

## 3. Core Steps

### Step 1: Batch Process (ExG + Otsu Threshold + DBSCAN)

- **Script**: `src/scripts/batch_threshold_dbscan.py`  
- **Input**: `.ply` files in `data/raw/`.  
- **Output**: `_labeled.ply` and `.npz` files in `data/manually_adjustments/`, with plant=green, non-plant=red.

    ```bash
    python src/main.py --batch_process


### Step 2: Manual Fix in Meshlab

    Load any _labeled.ply in Meshlab.
    Delete or fix incorrectly green-labeled points.
    Save as <basename>_labeled_plant_only_fixed.ply.

### Step 3: Adjust Labels

- **Script**: src/scripts/adjust_labels.py
    Compares Meshlab edits to the original labeled data.
    Points removed in Meshlab => label=0 (red).
    (Optional) Also flips mislabeled red=>green if there is _labeled_nonplant_only_fixed.ply.
    
    
    ```bash
    python src/main.py --adjust_labels --config src/configs/default_config.yaml

### Step 4: Preprocess (Downsample + Normalize)
- **Function**: src/data_processing/data_preprocessing.py
- **Reads**: final labeled .npz from data/manually_adjustments/, reduces points to e.g. 4096, normalizes, saves to data/processed/.


    ```bash
    python src/main.py --preprocess --config src/configs/default_config.yaml

### Step 6: Training

- **Script**: src/training/train.py
    Uses PointNet++ (or any 3D segmentation model) with num_classes=2 for binary segmentation.
    Trains on the train/val splits, saves best model.
    
    ```bash
    python src/main.py --train --config src/configs/default_config.yaml


### Step 7: Evaluation

- **Script**: src/training/evaluate.py
    Loads the best model, runs on test set, prints metrics (precision, recall, F1, IoU).
    
    ```bash
    python src/main.py --evaluate --config src/config/default_config.yaml

## 3. Configuration

All major parameters (paths, DBSCAN eps, training hyperparams) are in:
    
    ```bash
    src/configs/default_config.yaml

data:
  raw_dir: "data/raw"
  manual_dir: "data/manually_adjustments"
  processed_dir: "data/processed"
  split_ratios: [0.7, 0.15, 0.15]

preprocessing:
  voxel_size: 0.02
  num_points: 4096

training:
  batch_size: 16
  num_epochs: 50
  learning_rate: 0.001
  ...

## 4. Installation & Dependencies

    Python 3.7+ recommended
    PyTorch (for training)
    Other libraries: numpy, open3d, scikit-learn, scipy, etc.

Install via
    ```bash
    pip install -r requirements.txt

## 5. Advanced Notes

    ExG + DBSCAN is a heuristic. Tweak thresholds if color-based detection is inaccurate.
    Manual Fix in Meshlab primarily corrects false positives (green). Optionally fix false negatives with _labeled_nonplant_only_fixed.ply.
    Normalization & Downsampling ensures consistent input sizes for the neural net.