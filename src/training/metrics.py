# src/training/metrics.py

import torch
from torch.utils.data import DataLoader
from typing import Tuple

def calculate_iou(model: torch.nn.Module, data_loader: DataLoader, device: torch.device, num_classes: int) -> float:
    """
    Calculates the mean Intersection over Union (IoU) over the dataset.

    Args:
        model (torch.nn.Module): Trained model.
        data_loader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run computations.
        num_classes (int): Number of classes.

    Returns:
        float: Mean IoU score.
    """
    ious = []

    for points, labels in data_loader:
        points = points.to(device)
        labels = labels.to(device)

        outputs = model(points)
        preds = outputs.argmax(dim=2)

        intersection = ((preds == labels) & (labels > 0)).sum().item()
        union = ((preds > 0) | (labels > 0)).sum().item()
        if union == 0:
            iou = 0.0
        else:
            iou = intersection / union
        ious.append(iou)

    mean_iou = sum(ious) / len(ious)
    return mean_iou

def calculate_metrics(model: torch.nn.Module, data_loader: DataLoader, device: torch.device, num_classes: int) -> Tuple[float, float, float, float]:
    """
    Calculates precision, recall, F1-score, and IoU.

    Args:
        model (torch.nn.Module): Trained model.
        data_loader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run computations.
        num_classes (int): Number of classes.

    Returns:
        Tuple[float, float, float, float]: Precision, recall, F1-score, and IoU.
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    ious = []

    for points, labels in data_loader:
        points = points.to(device)
        labels = labels.to(device)

        outputs = model(points)
        preds = outputs.argmax(dim=2)

        tp = ((preds == labels) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        true_positive += tp
        false_positive += fp
        false_negative += fn

        intersection = ((preds == labels) & (labels == 1)).sum().item()
        union = ((preds == 1) | (labels == 1)).sum().item()
        if union == 0:
            iou = 0.0
        else:
            iou = intersection / union
        ious.append(iou)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = sum(ious) / len(ious)

    return precision, recall, f1_score, mean_iou
