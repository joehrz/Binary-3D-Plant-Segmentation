# src/training/evaluate.py

import os
import torch
from torch.utils.data import DataLoader

from src.models.pointnetplusplus import PointNetPlusPlus
from src.datasets.pointcloud_dataset import PointCloudDataset
from src.utils.metrics import calculate_metrics
from src.configs.config import Config

def evaluate_model(config: Config) -> None:
    """
    Evaluates the trained model on the test set.

    Args:
        config (Config): Configuration object with evaluation parameters.
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    test_dataset = PointCloudDataset(config.data.processed.splits.test_dir, num_points=config.model.num_points)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

    # Model
    model = PointNetPlusPlus(num_classes=config.training.num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(config.model.save_dir, 'best_model.pth')))
    model.eval()

    # Metrics
    precision, recall, f1_score, iou = calculate_metrics(model, test_loader, device, config.training.num_classes)

    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}, IoU: {iou:.4f}")
