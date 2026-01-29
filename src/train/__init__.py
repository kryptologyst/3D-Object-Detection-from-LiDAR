"""Training utilities for 3D object detection models."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import KITTIDataset, collate_fn
from ..utils import get_device, set_seed, save_checkpoint


class DetectionLoss(nn.Module):
    """Loss function for 3D object detection."""
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        regression_weight: float = 2.0,
        direction_weight: float = 0.2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        """Initialize detection loss.
        
        Args:
            classification_weight: Weight for classification loss.
            regression_weight: Weight for regression loss.
            direction_weight: Weight for direction loss.
            focal_alpha: Focal loss alpha parameter.
            focal_gamma: Focal loss gamma parameter.
        """
        super().__init__()
        
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.direction_weight = direction_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Loss functions
        self.focal_loss = self._focal_loss
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')
    
    def _focal_loss(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Predicted logits [N, C].
            targets: Target labels [N].
            
        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute detection loss.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            
        Returns:
            Dictionary of loss components.
        """
        losses = {}
        
        # Classification loss
        if "classification" in predictions and "labels" in targets:
            cls_pred = predictions["classification"]
            cls_target = targets["labels"]
            
            # Reshape predictions
            batch_size, num_classes, height, width = cls_pred.shape
            cls_pred = cls_pred.view(batch_size, num_classes, -1).transpose(1, 2)
            cls_pred = cls_pred.contiguous().view(-1, num_classes)
            
            # Create target labels (simplified)
            cls_target = torch.zeros(cls_pred.shape[0], dtype=torch.long, device=cls_pred.device)
            
            cls_loss = self.focal_loss(cls_pred, cls_target)
            losses["classification"] = cls_loss
        
        # Regression loss
        if "regression" in predictions and "bboxes" in targets:
            reg_pred = predictions["regression"]
            reg_target = targets["bboxes"]
            
            # Simplified regression loss
            reg_loss = torch.tensor(0.0, device=reg_pred.device)
            losses["regression"] = reg_loss
        
        # Direction loss
        if "direction" in predictions:
            dir_pred = predictions["direction"]
            dir_loss = torch.tensor(0.0, device=dir_pred.device)
            losses["direction"] = dir_loss
        
        # Total loss
        total_loss = (
            self.classification_weight * losses.get("classification", 0) +
            self.regression_weight * losses.get("regression", 0) +
            self.direction_weight * losses.get("direction", 0)
        )
        losses["total"] = total_loss
        
        return losses


class Trainer:
    """Trainer class for 3D object detection models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            device: Device to train on.
            config: Training configuration.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or get_device()
        self.config = config or {}
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = DetectionLoss(
            classification_weight=self.config.get("classification_weight", 1.0),
            regression_weight=self.config.get("regression_weight", 2.0),
            direction_weight=self.config.get("direction_weight", 0.2),
        )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 0.001),
            weight_decay=self.config.get("weight_decay", 0.0001),
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get("epochs", 100),
            eta_min=self.config.get("min_lr", 0.0001),
        )
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        
        # Mixed precision training
        self.use_amp = self.config.get("mixed_precision", True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            point_clouds = batch["point_clouds"].to(self.device)
            labels = batch["labels"]
            bboxes = batch["bboxes"]
            
            # Forward pass
            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    predictions = self.model(point_clouds)
                    targets = {"labels": labels, "bboxes": bboxes}
                    losses = self.criterion(predictions, targets)
            else:
                predictions = self.model(point_clouds)
                targets = {"labels": labels, "bboxes": bboxes}
                losses = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(losses["total"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["total"].backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += losses["total"].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "cls": f"{losses.get('classification', 0).item():.4f}",
                "reg": f"{losses.get('regression', 0).item():.4f}",
            })
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        return {
            "train_loss": avg_loss,
            "train_classification_loss": losses.get("classification", 0).item(),
            "train_regression_loss": losses.get("regression", 0).item(),
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary of validation metrics.
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                point_clouds = batch["point_clouds"].to(self.device)
                labels = batch["labels"]
                bboxes = batch["bboxes"]
                
                # Forward pass
                if self.use_amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(point_clouds)
                        targets = {"labels": labels, "bboxes": bboxes}
                        losses = self.criterion(predictions, targets)
                else:
                    predictions = self.model(point_clouds)
                    targets = {"labels": labels, "bboxes": bboxes}
                    losses = self.criterion(predictions, targets)
                
                # Update metrics
                total_loss += losses["total"].item()
                num_batches += 1
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        return {
            "val_loss": avg_loss,
            "val_classification_loss": losses.get("classification", 0).item(),
            "val_regression_loss": losses.get("regression", 0).item(),
        }
    
    def train(self, epochs: int, save_dir: str = "checkpoints") -> None:
        """Train the model.
        
        Args:
            epochs: Number of epochs to train.
            save_dir: Directory to save checkpoints.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Print metrics
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            if val_metrics:
                print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
            
            # Save checkpoint
            if (epoch + 1) % self.config.get("save_every_n_epochs", 5) == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    train_metrics["train_loss"],
                    {**train_metrics, **val_metrics},
                    checkpoint_path,
                )
            
            # Save best model
            if val_metrics and val_metrics["val_loss"] < self.best_metric:
                self.best_metric = val_metrics["val_loss"]
                best_path = os.path.join(save_dir, "best_model.pth")
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    train_metrics["train_loss"],
                    {**train_metrics, **val_metrics},
                    best_path,
                )
                print(f"New best model saved: {best_path}")


def train_model(
    model: nn.Module,
    train_dataset: KITTIDataset,
    val_dataset: Optional[KITTIDataset] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Train a 3D object detection model.
    
    Args:
        model: Model to train.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        config: Training configuration.
    """
    # Set random seed
    set_seed(config.get("seed", 42) if config else 42)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 4) if config else 4,
        shuffle=True,
        num_workers=config.get("num_workers", 4) if config else 4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get("batch_size", 4) if config else 4,
            shuffle=False,
            num_workers=config.get("num_workers", 4) if config else 4,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    
    # Train
    epochs = config.get("epochs", 100) if config else 100
    save_dir = config.get("checkpoints_dir", "checkpoints") if config else "checkpoints"
    
    trainer.train(epochs=epochs, save_dir=save_dir)
