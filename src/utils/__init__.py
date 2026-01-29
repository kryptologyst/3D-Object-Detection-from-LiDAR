"""Utility functions for 3D object detection from LiDAR."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cuda", "mps", "cpu").
        
    Returns:
        PyTorch device object.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        device: Device to load checkpoint on.
        
    Returns:
        Dictionary containing checkpoint information.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    checkpoint_path: str,
) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save.
        optimizer: Optimizer to save.
        epoch: Current epoch.
        loss: Current loss value.
        metrics: Dictionary of metrics.
        checkpoint_path: Path to save checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics,
    }
    
    torch.save(checkpoint, checkpoint_path)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batched data loading.
    
    Args:
        batch: List of samples from dataset.
        
    Returns:
        Batched data dictionary.
    """
    # Handle variable-length point clouds
    point_clouds = [sample["point_cloud"] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    bboxes = [sample["bboxes"] for sample in batch]
    
    # Pad point clouds to same length
    max_points = max(pc.shape[0] for pc in point_clouds)
    
    padded_point_clouds = []
    point_cloud_masks = []
    
    for pc in point_clouds:
        if pc.shape[0] < max_points:
            padding = torch.zeros(max_points - pc.shape[0], pc.shape[1])
            padded_pc = torch.cat([pc, padding], dim=0)
            mask = torch.cat([torch.ones(pc.shape[0]), torch.zeros(max_points - pc.shape[0])])
        else:
            padded_pc = pc
            mask = torch.ones(max_points)
        
        padded_point_clouds.append(padded_pc)
        point_cloud_masks.append(mask)
    
    return {
        "point_clouds": torch.stack(padded_point_clouds),
        "point_cloud_masks": torch.stack(point_cloud_masks),
        "labels": labels,
        "bboxes": bboxes,
    }


def calculate_iou_3d(
    bbox1: torch.Tensor, bbox2: torch.Tensor
) -> torch.Tensor:
    """Calculate 3D IoU between two bounding boxes.
    
    Args:
        bbox1: First bounding box tensor [x, y, z, w, h, l, yaw].
        bbox2: Second bounding box tensor [x, y, z, w, h, l, yaw].
        
    Returns:
        IoU values.
    """
    # Convert to corner format for easier IoU calculation
    # This is a simplified implementation
    # In practice, you'd want to use proper 3D IoU calculation
    
    # For now, return a placeholder
    return torch.tensor(0.0)


def apply_nms_3d(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.1,
    score_threshold: float = 0.1,
    max_detections: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply 3D Non-Maximum Suppression.
    
    Args:
        boxes: Bounding boxes tensor [N, 7].
        scores: Confidence scores tensor [N].
        iou_threshold: IoU threshold for NMS.
        score_threshold: Score threshold for filtering.
        max_detections: Maximum number of detections to keep.
        
    Returns:
        Tuple of (filtered_boxes, filtered_scores, filtered_indices).
    """
    # Filter by score threshold
    valid_mask = scores > score_threshold
    valid_boxes = boxes[valid_mask]
    valid_scores = scores[valid_mask]
    valid_indices = torch.where(valid_mask)[0]
    
    if len(valid_boxes) == 0:
        return torch.empty(0, 7), torch.empty(0), torch.empty(0, dtype=torch.long)
    
    # Sort by scores
    sorted_indices = torch.argsort(valid_scores, descending=True)
    sorted_boxes = valid_boxes[sorted_indices]
    sorted_scores = valid_scores[sorted_indices]
    sorted_valid_indices = valid_indices[sorted_indices]
    
    # Apply NMS (simplified implementation)
    keep_indices = []
    for i in range(len(sorted_boxes)):
        if len(keep_indices) >= max_detections:
            break
            
        keep = True
        for j in keep_indices:
            iou = calculate_iou_3d(sorted_boxes[i], sorted_boxes[j])
            if iou > iou_threshold:
                keep = False
                break
        
        if keep:
            keep_indices.append(i)
    
    if keep_indices:
        keep_indices = torch.tensor(keep_indices)
        return (
            sorted_boxes[keep_indices],
            sorted_scores[keep_indices],
            sorted_valid_indices[keep_indices],
        )
    else:
        return torch.empty(0, 7), torch.empty(0), torch.empty(0, dtype=torch.long)
