# src/training/train.py

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from src.models.pointnetplusplus import PointNetPlusPlus
from src.datasets.mixed_pointcloud_dataset import MixedPointCloudDataset
from src.training.metrics import calculate_iou
from src.configs.config import Config

def train_model(config: Config) -> None:
    """
    Trains the segmentation model using PointNet++.

    Steps:
        1) Load train & validation datasets into Dataloaders.
        2) Initialize PointNet++ model, optimizer, and learning rate scheduler.
        3) For each epoch:
             - Train the model on the entire training set.
             - Evaluate on the validation set (IoU).
             - Update best model if current val IoU is higher.

    Optional enhancements:
        - Log training IoU per epoch (requires an additional function call).
        - Early stopping if val IoU does not improve for X consecutive epochs.
        - Gradient clipping to avoid exploding gradients.

    Args:
        config (Config): Configuration object with training parameters, e.g.:
            config.data.processed.splits.train_dir
            config.data.processed.splits.val_dir
            config.model.num_points
            config.training.num_epochs
            config.training.batch_size
            config.training.learning_rate
            config.training.num_classes
            config.training.scheduler_step_size
            config.training.scheduler_gamma
            config.model.save_dir
    """
    # -----------------------------
    # 1. Device configuration
    # -----------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Training on device: {device}")

    # -----------------------------
    # 2. Dataset & DataLoader
    # -----------------------------
    # Collect actual file paths
    train_dir = config.data.splits.train_dir
    val_dir   = config.data.splits.val_dir

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.npz")))
    val_files   = sorted(glob.glob(os.path.join(val_dir, "*.npz")))

    train_dataset = MixedPointCloudDataset(
        train_files,
        num_points=config.model.num_points
    )


    val_dataset = MixedPointCloudDataset(
        val_files,
        num_points=config.model.num_points
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,     # Adjust based on system
        pin_memory=True    # Potential speedup if enough RAM
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
    # 3. Model, Loss, Optimizer
    # -----------------------------
    model = PointNetPlusPlus(num_classes=config.training.num_classes).to(device)

    criterion = nn.NLLLoss()  # or nn.CrossEntropyLoss if you remove log_softmax in model
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training.scheduler_step_size,
        gamma=config.training.scheduler_gamma
    )
  
    
    # -----------------------------
    # 4. Training loop
    # -----------------------------
    best_iou = 0.0
    best_epoch = 0
    no_improve_epochs = 0     # For optional early stopping
    patience = 10             # Stop if val iou doesn't improve for X epochs

    for epoch in range(config.training.num_epochs):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        # Optionally track training IoU
        # train_iou_running = 0.0

        for batch_idx, (points, labels) in enumerate(train_loader):
            points = points.to(device)   # shape: (B, N, 3)
            labels = labels.to(device)   # shape: (B, N)

            optimizer.zero_grad()
            outputs = model(points)      # shape: (B, N, num_classes)
            outputs = outputs.view(-1, config.training.num_classes)
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            loss.backward()

            # clip_grad_norm_(model.parameters(), max_norm=1.0)  # optional gradient clipping
            optimizer.step()

            running_loss += loss.item()
            # If you want training IoU on the fly:
            # preds = outputs.argmax(dim=-1)
            # train_iou_batch = compute_batch_iou(preds, labels, config.training.num_classes)
            # train_iou_running += train_iou_batch

        # Average training loss
        epoch_loss = running_loss / len(train_loader)

        # Optionally, compute average training IoU
        # train_iou_avg = train_iou_running / len(train_loader)

        scheduler.step()

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            val_iou = calculate_iou(
                model, val_loader, device, config.training.num_classes
            )

        print(f"Epoch [{epoch + 1}/{config.training.num_epochs}] "
              f"Train Loss: {epoch_loss:.4f}, "
              # f"Train IoU: {train_iou_avg:.4f}, "  # if you computed train iou
              f"Val IoU: {val_iou:.4f}")

        # --------------------
        # 5. Checkpoint
        # --------------------
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch + 1
            os.makedirs(config.model.save_dir, exist_ok=True)
            model_save_path = os.path.join(config.model.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"  => New best IoU: {best_iou:.4f}. Model saved to {model_save_path}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # --------------------
        # 6. Early Stopping (Optional)
        # --------------------
        if no_improve_epochs >= patience:
            print(f"[Early Stopping] Validation IoU did not improve for {patience} epochs.")
            break

    print(f"Training completed. Best IoU: {best_iou:.4f} at epoch {best_epoch}.")
