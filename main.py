#!/usr/bin/env python3
"""Main script to run the entire point cloud segmentation pipeline."""

import os
import sys
import argparse
import copy

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.utils.logger import setup_logger
from src.data_processing.data_augmentation import preprocess_point_clouds
from src.data_processing.dataset_splitting import split_dataset
from src.training.train import train_model
from src.training.evaluate import evaluate_model
from src.configs.config import Config

from src.scripts.batch_threshold_dbscan import batch_threshold_dbscan
from src.scripts.adjust_labels import batch_adjust_labels
from src.scripts.generate_synthetic_dataset import generate_synthetic_dataset

def main():
    parser = argparse.ArgumentParser(description="Point Cloud Segmentation Pipeline")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                        help="Path to the configuration file.")

    # Real data steps
    parser.add_argument("--batch_process", action="store_true")
    parser.add_argument("--adjust_labels", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")

    # Synthetic data steps
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic dataset from real plant data.")
    parser.add_argument("--synthetic_preprocess", action="store_true",
                        help="Preprocess the newly created synthetic .npz files.")
    parser.add_argument("--synthetic_split", action="store_true",
                        help="Split the processed synthetic dataset into train/val/test.")
    parser.add_argument("--synthetic_train", action="store_true",
                        help="Train the model using synthetic splits (pure synthetic).")

    args = parser.parse_args()
    config = Config(args.config)

    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", "pipeline.log")
    logger = setup_logger("pipeline_logger", log_file)
    logger.info("=== Point Cloud Segmentation Pipeline Started ===")

    # ---------------------
    # Real data steps
    # ---------------------
    if args.batch_process:
        batch_threshold_dbscan(
            raw_dir=config.data.raw_dir,
            out_dir=config.data.manual_dir,
            dbscan_eps=0.02,
            dbscan_min_samples=20,
            min_cluster_size=10000,
            debug=False
        )

    if args.adjust_labels:
        batch_adjust_labels(
            manual_dir=config.data.manual_dir,
            suffix_labeled="_labeled",
            suffix_plant_fixed="_labeled_plant_only_fixed",
            suffix_nonplant_fixed="_labeled_nonplant_only_fixed",
            decimals=6
        )

    if args.preprocess:
        logger.info("Starting data preprocessing (real data)...")
        preprocess_point_clouds(
            input_dir=config.data.manual_dir,
            output_dir=config.data.processed_dir,
            voxel_size=config.preprocessing.voxel_size,
            num_points=config.model.num_points,
            file_ext="_final.npz"
        )
        logger.info("Real data preprocessing completed.")

    if args.split:
        logger.info("Splitting real data into train/val/test...")
        split_dataset(
            input_dir=config.data.processed_dir,
            output_dir=config.data.splits_dir,
            split_ratios=tuple(config.data.split_ratios)
        )
        logger.info("Real data splitting completed.")

    if args.train:
        logger.info("Starting model training on real data...")
        train_model(config)
        logger.info("Model training completed.")

    if args.evaluate:
        logger.info("Evaluating on real data test set...")
        evaluate_model(config)
        logger.info("Model evaluation completed.")

    # ---------------------
    # Synthetic data steps
    # ---------------------
    if args.synthetic:
        logger.info("Generating synthetic dataset from real plant data...")
        generate_synthetic_dataset(
            plant_file=config.data.synthetic.plant_file,
            output_file=config.data.synthetic.output_file,
            artifact_ratio=config.data.synthetic.artifact_ratio,
            artifact_sigma=config.data.synthetic.artifact_sigma,
            plant_noise_sigma=config.data.synthetic.plant_noise_sigma
        )
        logger.info("Synthetic dataset generation completed.")

    if args.synthetic_preprocess:
        logger.info("Preprocessing synthetic .npz files...")
        preprocess_point_clouds(
            input_dir=config.data.synthetic.output_file,     # e.g. data/Sorghum_Plants_Point_Cloud_Data/synthetic
            output_dir=config.data.synthetic.processed_dir,  # e.g. data/Sorghum_Plants_Point_Cloud_Data/synthetic_proc
            voxel_size=config.preprocessing.voxel_size,
            num_points=config.model.num_points,
            file_ext="_synthetic.npz"  # or "" to catch all npz
        )
        logger.info("Synthetic data preprocessing completed.")

    if args.synthetic_split:
        logger.info("Splitting synthetic data into train/val/test...")
        split_dataset(
            input_dir=config.data.synthetic.processed_dir,
            output_dir=config.data.synthetic.splits_dir,
            split_ratios=tuple(config.data.split_ratios)
        )
        logger.info("Synthetic data splitting completed.")

    if args.synthetic_train:
        logger.info("Starting training on synthetic data...")
        synthetic_cfg = copy.deepcopy(config)
        # Overwrite the normal splits with synthetic_splits
        synthetic_cfg.data.splits = synthetic_cfg.data.synthetic_splits
        train_model(synthetic_cfg)
        logger.info("Model training on synthetic data completed.")

    logger.info("=== Point Cloud Segmentation Pipeline Finished ===")


if __name__ == '__main__':
    main()
