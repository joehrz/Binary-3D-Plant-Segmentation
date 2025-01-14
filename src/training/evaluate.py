# src/training/evaluate.py

import os
import torch
from torch.utils.data import DataLoader

from src.models.pointnetplusplus import PointNetPlusPlus
from src.datasets.pointcloud_dataset import PointCloudDataset
from src.training.metrics import calculate_metrics
from src.configs.config import Config

def evaluate_model(config: Config) -> None:
    """
    Evaluates the trained model on the test set by computing precision, recall,
    F1-score, and IoU for label=1 (plant vs. background=0).

    Args:
        config (Config): Configuration object with evaluation parameters.
    """
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating on device: {device}")

    # Dataset & Dataloader
    test_dataset = PointCloudDataset(
        config.data.processed.splits.test_dir,
        num_points=config.model.num_points
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False
    )
    print(f"[INFO] Loaded {len(test_dataset)} test samples.")


    # Load Model
    model = PointNetPlusPlus(num_classes=config.training.num_classes).to(device)
    best_model_path = os.path.join(config.model.save_dir, "best_model.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {best_model_path}")
    
    model.load.state_dict(torch.load(best_model_path))
    model.eval()

    # Calculate Metrics
    precision, recall, f1_score, iou = calculate_metrics(
        model, test_loader, device, config.training.num_classes
    )

    # Print results
    print(f"\n==== Test Metrics ====")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1_score:.4f}")
    print(f"IoU       : {iou:.4f}\n")
