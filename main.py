#!/usr/bin/env python3

## src/main.py

"""Main script to run the entire point cloud segmentation pipeline."""

import os
import sys
import argparse

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import custom modules
from src.utils.logger import setup_logger
from src.data_processing.data_augmentation import process_point_clouds
from src.data_processing.data_preprocessing import preprocess_point_clouds
from src.data_processing.dataset_splitting import split_dataset
from src.training.train import train_model
from src.training.evaluate import evaluate_model
from src.configs.config import Config

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Point Cloud Segmentation Pipeline")
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the raw data.')
    parser.add_argument('--split', action='store_true', help='Split the dataset.')
    parser.add_argument('--train', action='store_true', help='Train the segmentation model.')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the segmentation model.')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)

    # Setup logging
    log_file = os.path.join('logs', 'pipeline.log')
    logger = setup_logger('pipeline_logger', log_file)
    logger.info("=== Point Cloud Segmentation Pipeline Started ===")

    # Step 1: Data Augmentation (Adding Noise)
    if args.preprocess:
        logger.info("Starting data preprocessing...")

        # Process plant-only data
        input_dir = config.data.raw.plant_only_dir
        noisy_output_dir = config.data.raw.plant_only_noisy_dir
        noise_params = config.preprocessing.noise_params

        process_point_clouds(input_dir, noisy_output_dir, noise_params)

        # Preprocess noisy data
        processed_output_dir = config.data.processed.plant_only_noisy_dir
        voxel_size = config.preprocessing.voxel_size
        num_points = config.preprocessing.num_points

        preprocess_point_clouds(noisy_output_dir, processed_output_dir, voxel_size, num_points)

        logger.info("Data preprocessing completed.")

    # Step 2: Dataset Splitting
    if args.split:
        logger.info("Starting dataset splitting...")
        input_dir = config.data.processed.plant_only_noisy_dir
        output_dir = os.path.join('data', 'processed', 'splits')
        split_ratios = tuple(config.data.split_ratios)

        split_dataset(input_dir, output_dir, split_ratios)
        logger.info("Dataset splitting completed.")

    # Step 3: Training
    if args.train:
        logger.info("Starting model training...")
        train_model(config)
        logger.info("Model training completed.")

    # Step 4: Evaluation
    if args.evaluate:
        logger.info("Starting model evaluation...")
        evaluate_model(config)
        logger.info("Model evaluation completed.")

    logger.info("=== Point Cloud Segmentation Pipeline Finished ===")

if __name__ == '__main__':
    main()
