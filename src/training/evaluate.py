# src/training/evaluate.py

import os
import torch
import glob
from torch.utils.data import DataLoader

from src.models.pointnetplusplus import PointNetPlusPlus
from src.datasets.mixed_pointcloud_dataset import MixedPointCloudDataset
from src.training.metrics import calculate_metrics
from src.configs.config import Config

def evaluate_model(config: Config) -> None:
    """
    Evaluates the trained model on the test set by computing precision, recall,
    F1-score, and IoU for label=1 (plant vs. background=0).

    Args:
        config (Config): Configuration object with evaluation parameters.
    """

    # (1) Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating on device: {device}")

    # (2) Find the test directory.
    # Store real test data under 'config.data.splits.test_dir',
    # or if used 'config.data.splits_dir' + "/test" subfolder,
    # adapt as needed:
    test_dir = config.data.splits.test_dir  
    # e.g. "data/wheat_data/splits/test"

    test_dir = os.path.join(config.data.splits_dir, "test")
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.npz")))
    if len(test_files) == 0:
        print(f"[WARNING] No test files found in {test_dir}")
        return

    # (3) Create dataset & dataloader
    test_dataset = MixedPointCloudDataset(
        file_list=test_files,
        num_points=config.model.num_points
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4  # or 0 if no parallel workers
    )
    print(f"[INFO] Loaded {len(test_dataset)} test samples.")

    # (4) Load model + checkpoint
    model = PointNetPlusPlus(num_classes=config.training.num_classes).to(device)
    best_model_path = os.path.join(config.model.save_dir, "best_model_train.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {best_model_path}")
    
    # use model.load.state_dict
    print(f"[INFO] Loading model checkpoint from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # (5) Compute metrics
    precision, recall, f1_score, iou = calculate_metrics(
        model, test_loader, device, config.training.num_classes
    )

    # (6) Print or log results
    print("\n==== Test Metrics ====")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1_score:.4f}")
    print(f"IoU       : {iou:.4f}\n")