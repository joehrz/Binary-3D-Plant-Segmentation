#!/usr/bin/env python3

# src/main.py
"""Main script to run the entire point cloud segmentation pipeline."""

import os
import sys
import argparse

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.utils.logger import setup_logger
from src.data_processing.data_augmentation import preprocess_point_clouds
#from src.data_processing.data_preprocessing import preprocess_point_clouds
from src.data_processing.dataset_splitting import split_dataset
from src.training.train import train_model
from src.training.evaluate import evaluate_model
from src.configs.config import Config

from src.scripts.batch_threshold_dbscan import batch_threshold_dbscan
#from src.scripts.adjust_labels import adjust_labels_after_meshlab
from src.scripts.adjust_labels import batch_adjust_labels

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Point Cloud Segmentation Pipeline")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                        help="Path to the configuration file.")

    # Steps
    parser.add_argument("--batch_process", action="store_true",
                        help="(1) Run ExG+DBSCAN labeling on raw data.")
    parser.add_argument("--adjust_labels", action="store_true",
                        help="(2) Adjust final labels based on Meshlab-edited plant-only ply.")
    parser.add_argument("--preprocess", action="store_true",
                        help="(3) Downsample/normalize final labeled data.")
    parser.add_argument("--split", action="store_true",
                        help="(4) Split dataset into train/val/test.")
    parser.add_argument("--train", action="store_true",
                        help="(5) Train the segmentation model.")
    parser.add_argument("--evaluate", action="store_true",
                        help="(6) Evaluate the segmentation model.")

    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)

    # Setup logging
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", "pipeline.log")
    logger = setup_logger("pipeline_logger", log_file)
    logger.info("=== Point Cloud Segmentation Pipeline Started ===")

    # 1) Batch Process => data/raw -> data/manually_adjustments
    if args.batch_process:
        batch_threshold_dbscan(
            raw_dir=config.data.raw_dir,            # e.g. "data/raw"
            out_dir=config.data.manual_dir,         # e.g. "data/manually_adjustments"
            dbscan_eps=0.02,
            dbscan_min_samples=20,
            min_cluster_size=10000,
            debug=False
        )

    # 2) Adjust Labels => fix mislabeled plant points after Meshlab
    if args.adjust_labels:
        batch_adjust_labels(
            manual_dir="data/manually_adjustments",
            suffix_labeled="_labeled",
            suffix_plant_fixed="_labeled_plant_only_fixed",
            suffix_nonplant_fixed="_labeled_nonplant_only_fixed",
            decimals=6
        )

    # 3) Preprocess => downsample + normalize => data/processed
    if args.preprocess:
        logger.info("Starting data preprocessing...")

        # Possibly the final .npz from adjust_labels is "final_adjusted_labels.npz"
        # or multiple .npz files in data/manually_adjustments
        preprocess_point_clouds(
            input_dir=config.data.manual_dir,   # e.g. "data/manually_adjustments"
            output_dir=config.data.processed_dir,  # e.g. "data/processed"
            voxel_size=0.02,
            num_points=4096,
            file_ext="_final.npz"
        )
        logger.info("Data preprocessing completed.")

    # 4) Dataset Splitting => data/processed -> data/processed/splits
    if args.split:
        logger.info("Starting dataset splitting...")
        # If you want to split the final preprocessed data:
        input_dir = config.data.processed_dir
        output_dir = os.path.join(config.data.processed_dir, "splits")
        split_ratios = tuple(config.data.split_ratios)

        split_dataset(input_dir, output_dir, split_ratios)
        logger.info("Dataset splitting completed.")

    # 5) Training
    if args.train:
        logger.info("Starting model training...")
        train_model(config)
        logger.info("Model training completed.")

    # 6) Evaluation
    if args.evaluate:
        logger.info("Starting model evaluation...")
        evaluate_model(config)
        logger.info("Model evaluation completed.")

    logger.info("=== Point Cloud Segmentation Pipeline Finished ===")

if __name__ == '__main__':
    main()
