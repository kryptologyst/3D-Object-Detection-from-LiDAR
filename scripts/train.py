#!/usr/bin/env python3
"""Main training script for 3D object detection from LiDAR."""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import torch
from omegaconf import OmegaConf

from src.data import KITTIDataset
from src.models import PointPillars, SECOND, CenterPoint
from src.train import train_model
from src.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train 3D object detection model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to dataset directory",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["pointpillars", "second", "centerpoint"],
        default="pointpillars",
        help="Model architecture to use",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs",
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        return OmegaConf.to_container(config, resolve=True)
    else:
        print(f"Config file not found: {config_path}")
        print("Using default configuration...")
        return {}


def create_model(model_name: str, config: Dict[str, Any]) -> torch.nn.Module:
    """Create model based on configuration.
    
    Args:
        model_name: Name of the model architecture.
        config: Model configuration.
        
    Returns:
        Model instance.
    """
    if model_name == "pointpillars":
        return PointPillars(
            voxel_size=config.get("voxel_size", [0.2, 0.2, 4.0]),
            point_cloud_range=config.get("point_cloud_range", [-50.0, -50.0, -3.0, 50.0, 50.0, 1.0]),
            max_points_per_voxel=config.get("max_points_per_voxel", 32),
            max_voxels=config.get("max_voxels", 20000),
            num_classes=config.get("num_classes", 3),
            num_anchors=config.get("num_anchors", 2),
        )
    elif model_name == "second":
        return SECOND(
            voxel_size=config.get("voxel_size", [0.05, 0.05, 0.1]),
            point_cloud_range=config.get("point_cloud_range", [-50.0, -50.0, -3.0, 50.0, 50.0, 1.0]),
            max_points_per_voxel=config.get("max_points_per_voxel", 35),
            max_voxels=config.get("max_voxels", 20000),
            num_classes=config.get("num_classes", 3),
            num_anchors=config.get("num_anchors", 2),
        )
    elif model_name == "centerpoint":
        return CenterPoint(
            voxel_size=config.get("voxel_size", [0.05, 0.05, 0.1]),
            point_cloud_range=config.get("point_cloud_range", [-50.0, -50.0, -3.0, 50.0, 50.0, 1.0]),
            max_points_per_voxel=config.get("max_points_per_voxel", 35),
            max_voxels=config.get("max_voxels", 20000),
            num_classes=config.get("num_classes", 3),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.learning_rate
    config["device"] = args.device
    config["seed"] = args.seed
    config["output_dir"] = args.output_dir
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = KITTIDataset(
        data_dir=args.data_dir,
        split="train",
        max_points=config.get("max_points", 16384),
        augmentation=config.get("augmentation", {}),
    )
    
    val_dataset = KITTIDataset(
        data_dir=args.data_dir,
        split="val",
        max_points=config.get("max_points", 16384),
        augmentation={"enabled": False},  # No augmentation for validation
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Create model
    print(f"Creating {args.model} model...")
    model_config = config.get("model", {})
    model = create_model(args.model, model_config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Train model
    print("Starting training...")
    train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
