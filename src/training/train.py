# src/training/train.py

import os
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.models.pointnetplusplus import PointNetPlusPlus
from src.datasets.mixed_pointcloud_dataset import MixedPointCloudDataset
from src.datasets.augmented_dataset_wrapper import AugmentedDatasetWrapper
from src.training.metrics import calculate_iou
from src.configs.config import Config


def plot_training_curve(train_losses, val_ious, out_path="training_curve.png"):
    """
    Saves a plot of training loss and validation IoU over epochs to out_path.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    # Plot training loss
    plt.plot(epochs, train_losses, label='Train Loss', color='red')
    # Plot validation IoU
    plt.plot(epochs, val_ious, label='Val IoU', color='blue')

    plt.xlabel("Epoch")
    plt.ylabel("Loss / IoU")
    plt.title("Training Curve: Loss and Val IoU")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Training curve saved to {out_path}")


def train_model(config: Config) -> None:
    """
    Trains the segmentation model using PointNet++.
    Adds a unique name to the saved checkpoint file so it doesn't overwrite
    other training runs (e.g. synthetic vs. real datasets).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Training on device: {device}")

    # -----------------------------
    # 1. Prepare file lists for train/val
    # -----------------------------
    train_dir = config.data.splits.train_dir
    val_dir   = config.data.splits.val_dir

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.npz")))
    val_files   = sorted(glob.glob(os.path.join(val_dir, "*.npz")))

    # We'll extract a short name from train_dir to include in the checkpoint filename.
    # For instance, if train_dir="data/synthetic_proc/splits/train", we might use just "train".
    train_dir_name = os.path.basename(os.path.normpath(train_dir))  # e.g. "train"

    # -----------------------------
    # 2. Base dataset objects
    # -----------------------------
    base_train_dataset = MixedPointCloudDataset(
        file_list=train_files,
        num_points=config.model.num_points
    )
    base_val_dataset = MixedPointCloudDataset(
        file_list=val_files,
        num_points=config.model.num_points
    )

    # Optionally apply augmentation for train
    augment_train = getattr(config.training, "augment_train", True)
    if augment_train:
        train_dataset = AugmentedDatasetWrapper(
            base_dataset=base_train_dataset,
            augment=True,
            rotate_range=(-np.pi, np.pi),
            flip_prob=0.5,
            scale_range=(0.9, 1.1),
            partial_dropout_prob=0.3
        )
        print("[INFO] On-the-fly augmentation is enabled for TRAIN dataset.")
    else:
        train_dataset = base_train_dataset

    val_dataset = base_val_dataset

    # -----------------------------
    # 3. Create DataLoaders
    # -----------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"[INFO] Loaded {len(train_dataset)} training samples, "
          f"{len(val_dataset)} validation samples.")

    # -----------------------------
    # 4. Build model
    # -----------------------------
    model = PointNetPlusPlus(num_classes=config.training.num_classes).to(device)

    # Optionally load checkpoint
    resume_ckpt = getattr(config.training, "resume_checkpoint", None)
    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"[INFO] Loading pre-trained weights from {resume_ckpt}")
        model.load_state_dict(torch.load(resume_ckpt, map_location=device))

    # -----------------------------
    # 5. Loss/optimizer
    # -----------------------------
    criterion = nn.NLLLoss()  # or nn.CrossEntropyLoss if removing log_softmax in model
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training.scheduler_step_size,
        gamma=config.training.scheduler_gamma
    )

    # -----------------------------
    # 6. Training loop + logging
    # -----------------------------
    best_iou = 0.0
    best_epoch = 0
    no_improve_epochs = 0
    patience = 10

    train_loss_history = []
    val_iou_history = []

    for epoch in range(config.training.num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (points, labels) in enumerate(train_loader):
            points, labels = points.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)  # shape: (B, N, num_classes)
            outputs = outputs.view(-1, config.training.num_classes)
            labels  = labels.view(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Average training loss for this epoch
        epoch_loss = running_loss / len(train_loader)
        train_loss_history.append(epoch_loss)

        scheduler.step()

        # Validation IoU
        model.eval()
        with torch.no_grad():
            val_iou = calculate_iou(model, val_loader, device, config.training.num_classes)

        val_iou_history.append(val_iou)

        print(f"Epoch [{epoch + 1}/{config.training.num_epochs}] "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val IoU: {val_iou:.4f}")

        # Checkpointing
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch + 1

            # Instead of always "best_model.pth", incorporate train_dir_name
            # e.g., "best_model_train.pth"
            ckpt_name = f"best_model_{train_dir_name}.pth"
            os.makedirs(config.model.save_dir, exist_ok=True)
            model_save_path = os.path.join(config.model.save_dir, ckpt_name)

            torch.save(model.state_dict(), model_save_path)
            print(f"  => New best IoU: {best_iou:.4f}. Model saved to {model_save_path}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early Stopping
        if no_improve_epochs >= patience:
            print(f"[Early Stopping] Validation IoU did not improve for {patience} epochs.")
            break

    print(f"Training completed. Best IoU: {best_iou:.4f} at epoch {best_epoch}.")

    # -----------------------------
    # 7. Plot the training curve
    # -----------------------------
    out_path = os.path.join(config.model.save_dir, f"training_curve_{train_dir_name}.png")
    plot_training_curve(train_loss_history, val_iou_history, out_path=out_path)

