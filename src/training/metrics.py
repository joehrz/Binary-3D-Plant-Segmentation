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

def calculate_metrics(
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        num_classes: int
)-> Tuple[float, float, float, float]:
    """
    Calculates precision, recall, F1-score, and IoU for a binary segmentation task
    (plant=1, background=0).

    It performs a single pass over `data_loader`, accumulating:
      - true positives (tp)
      - false positives (fp)
      - false negatives (fn)
      - intersection & union for label=1

    Precision = tp / (tp + fp)  [plant points predicted correctly / all predicted plant]
    Recall    = tp / (tp + fn)  [plant points predicted correctly / all real plant points]
    F1        = 2 * precision * recall / (precision + recall)
    IoU       = intersection / union, where intersection & union consider label=1

    Args:
        model (torch.nn.Module): Trained segmentation model.
        data_loader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run computations (cpu/gpu).
        num_classes (int): Number of classes. (For binary segmentation, typically 2.)

    Returns:
        Tuple[float, float, float, float]:
            (precision, recall, f1_score, iou) for the plant class (label=1).
    """
    model.eval()

    true_positive = 0
    false_positive = 0
    false_negative = 0

    # For a "micro IoU" approach across the entire dataset"
    intersection_global = 0
    union_global = 0

    with torch.no_grad():
        for points, labels in data_loader:
            points = points.to(device)
            labels = labels.to(device) # shape: (B, N)

            # Forward pass
            outputs = model(points)     # shape: (B, N, num_classes)
            preds = outputs.argmax(dim=2)  # shape: (B, N)

            # --------------------------------
            # Accumulate confusion matrix stats
            # --------------------------------
            # Here we treat "1" as the positive (plant) class.

            tp_batch = ((preds == 1) & (labels == 1)).sum().item()
            fp_batch = ((preds == 1) & (labels == 0)).sum().item()
            fn_batch = ((preds == 0) & (labels == 1)).sum().item()

            true_positive += tp_batch
            false_positive += fp_batch
            false_negative += fn_batch


            # --------------------------------
            # Intersection & Union for IoU
            # --------------------------------
            # intersection = plant points that match
            # union = all plant points in either preds or labels
            intersection = ((preds == 1) & (labels == 1)).sum().item()
            union = ((preds == 1) | (labels == 1)).sum().item()

            intersection_global += intersection
            union_global += union

    # Precision / Recall / F1
    precision = 0.0
    recall = 0.0
    if (true_positive + false_positive) > 0:
        precision = true_positive / (true_positive + false_positive)
    if (true_positive + false_negative) > 0:
        recall = true_positive / (true_positive + false_negative)

    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0

    # IoU across entire dataset
    if union_global == 0:
        iou = 0.0
    else:
        iou = intersection_global / union_global

    return precision, recall, f1_score, iou

def calculate_iou(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int
) -> float:
    """
    A convenience function if you only want the IoU. This internally calls
    `calculate_metrics` and returns just the IoU for label=1.

    Args:
        model (torch.nn.Module): Trained segmentation model.
        data_loader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run computations.
        num_classes (int): Number of classes.

    Returns:
        float: IoU for the "plant" class (label=1).
    """
    model.eval()
    _, _, _, iou = calculate_metrics(model, data_loader, device, num_classes)
    return iou
    

