#!/usr/bin/env python3
"""Evaluation script for 3D object detection from LiDAR."""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import torch
from omegaconf import OmegaConf

from src.data import KITTIDataset
from src.models import PointPillars, SECOND, CenterPoint
from src.eval import evaluate_model, create_evaluation_report
from src.utils import get_device, set_seed, load_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate 3D object detection model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
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
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation",
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
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate on",
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
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config["batch_size"] = args.batch_size
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
    
    # Create test dataset
    print(f"Creating {args.split} dataset...")
    test_dataset = KITTIDataset(
        data_dir=args.data_dir,
        split=args.split,
        max_points=config.get("max_points", 16384),
        augmentation={"enabled": False},  # No augmentation for evaluation
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Create model
    print(f"Creating {args.model} model...")
    model_config = config.get("model", {})
    model = create_model(args.model, model_config)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Evaluate model
    print("Starting evaluation...")
    metrics = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        config=config,
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Create evaluation report
    report_path = os.path.join(args.output_dir, "evaluation_report.txt")
    create_evaluation_report(metrics, report_path)
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
