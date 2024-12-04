# src/training/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.pointnetplusplus import PointNetPlusPlus
from src.datasets.pointcloud_dataset import PointCloudDataset
from src.utils.metrics import calculate_iou
from src.configs.config import Config

def train_model(config: Config) -> None:
    """
    Trains the segmentation model.

    Args:
        config (Config): Configuration object with training parameters.
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    train_dataset = PointCloudDataset(config.data.processed.splits.train_dir, num_points=config.model.num_points)
    val_dataset = PointCloudDataset(config.data.processed.splits.val_dir, num_points=config.model.num_points)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = PointNetPlusPlus(num_classes=config.training.num_classes).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.training.scheduler_step_size, gamma=config.training.scheduler_gamma)

    best_iou = 0.0

    for epoch in range(config.training.num_epochs):
        model.train()
        running_loss = 0.0

        for points, labels in train_loader:
            points = points.to(device)  # Shape: (B, N, 3)
            labels = labels.to(device)  # Shape: (B, N)

            optimizer.zero_grad()
            outputs = model(points)  # Shape: (B, N, num_classes)
            outputs = outputs.view(-1, config.training.num_classes)
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            iou = calculate_iou(model, val_loader, device, config.training.num_classes)

        print(f"Epoch [{epoch+1}/{config.training.num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val IoU: {iou:.4f}")

        # Save the best model
        if iou > best_iou:
            best_iou = iou
            model_save_path = os.path.join(config.model.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")

    print("Training completed.")
