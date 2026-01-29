"""Dataset classes for 3D object detection from LiDAR."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from . import PointCloudProcessor, BBoxProcessor


class KITTIDataset(Dataset):
    """KITTI dataset for 3D object detection."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_points: int = 16384,
        augmentation: Optional[Dict[str, Any]] = None,
    ):
        """Initialize KITTI dataset.
        
        Args:
            data_dir: Path to KITTI dataset directory.
            split: Dataset split ("train", "val", "test").
            max_points: Maximum number of points per point cloud.
            augmentation: Augmentation configuration.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_points = max_points
        self.augmentation = augmentation or {}
        
        # Initialize processors
        self.point_cloud_processor = PointCloudProcessor(max_points=max_points)
        self.bbox_processor = BBoxProcessor()
        
        # Load data paths
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Dict[str, str]]:
        """Load sample paths from dataset directory.
        
        Returns:
            List of sample dictionaries with file paths.
        """
        samples = []
        
        # KITTI structure: data_dir/velodyne/ and data_dir/label_2/
        velodyne_dir = self.data_dir / "velodyne"
        label_dir = self.data_dir / "label_2"
        
        if not velodyne_dir.exists():
            # Create synthetic data if KITTI not available
            print("KITTI dataset not found. Creating synthetic data...")
            self._create_synthetic_data()
            velodyne_dir = self.data_dir / "synthetic" / "point_clouds"
            label_dir = self.data_dir / "synthetic" / "annotations"
        
        # Get all point cloud files
        point_cloud_files = sorted(velodyne_dir.glob("*.bin"))
        if not point_cloud_files:
            point_cloud_files = sorted(velodyne_dir.glob("*.pcd"))
        
        for pc_file in point_cloud_files:
            sample_id = pc_file.stem
            
            # Find corresponding label file
            label_file = label_dir / f"{sample_id}.txt"
            
            if label_file.exists() or self.split == "test":
                samples.append({
                    "point_cloud": str(pc_file),
                    "annotation": str(label_file) if label_file.exists() else None,
                    "sample_id": sample_id,
                })
        
        return samples
    
    def _create_synthetic_data(self) -> None:
        """Create synthetic data for testing."""
        from . import create_synthetic_data
        
        synthetic_dir = self.data_dir / "synthetic"
        create_synthetic_data(
            num_samples=100,
            num_points_per_sample=1000,
            num_objects_per_sample=3,
            output_dir=str(synthetic_dir),
        )
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item.
        
        Args:
            idx: Sample index.
            
        Returns:
            Sample dictionary with point cloud, labels, and bboxes.
        """
        sample = self.samples[idx]
        
        # Load point cloud
        point_cloud = self.point_cloud_processor.process_point_cloud(sample["point_cloud"])
        
        # Load annotations
        labels = []
        bboxes = []
        
        if sample["annotation"] and os.path.exists(sample["annotation"]):
            bboxes, labels = self.bbox_processor.process_annotations(sample["annotation"])
        
        # Apply augmentations
        if self.augmentation.get("enabled", False) and self.split == "train":
            point_cloud, bboxes = self._apply_augmentation(point_cloud, bboxes)
        
        return {
            "point_cloud": point_cloud,
            "labels": labels,
            "bboxes": bboxes,
            "sample_id": sample["sample_id"],
        }
    
    def _apply_augmentation(
        self, point_cloud: torch.Tensor, bboxes: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Apply data augmentation.
        
        Args:
            point_cloud: Point cloud tensor.
            bboxes: List of bounding box tensors.
            
        Returns:
            Augmented point cloud and bounding boxes.
        """
        # Random rotation
        if "rotation_range" in self.augmentation:
            angle = np.random.uniform(*self.augmentation["rotation_range"])
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=torch.float32)
            
            # Rotate points
            point_cloud[:, :3] = torch.matmul(point_cloud[:, :3], rotation_matrix.T)
            
            # Rotate bounding boxes
            for i, bbox in enumerate(bboxes):
                bbox[6] += angle  # Update yaw angle
        
        # Random translation
        if "translation_range" in self.augmentation:
            translation = np.random.uniform(
                -self.augmentation["translation_range"][1],
                self.augmentation["translation_range"][1],
                size=3
            )
            point_cloud[:, :3] += torch.tensor(translation, dtype=torch.float32)
            
            # Translate bounding boxes
            for i, bbox in enumerate(bboxes):
                bbox[:3] += torch.tensor(translation, dtype=torch.float32)
        
        # Random scaling
        if "scale_range" in self.augmentation:
            scale = np.random.uniform(*self.augmentation["scale_range"])
            point_cloud[:, :3] *= scale
            
            # Scale bounding boxes
            for i, bbox in enumerate(bboxes):
                bbox[3:6] *= scale  # Scale dimensions
        
        # Random horizontal flip
        if self.augmentation.get("flip_probability", 0) > 0:
            if np.random.random() < self.augmentation["flip_probability"]:
                point_cloud[:, 1] *= -1  # Flip y-axis
                
                # Flip bounding boxes
                for i, bbox in enumerate(bboxes):
                    bbox[1] *= -1  # Flip y position
                    bbox[6] *= -1  # Flip yaw angle
        
        return point_cloud, bboxes


class WaymoDataset(Dataset):
    """Waymo dataset for 3D object detection."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_points: int = 16384,
        augmentation: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Waymo dataset.
        
        Args:
            data_dir: Path to Waymo dataset directory.
            split: Dataset split ("train", "val", "test").
            max_points: Maximum number of points per point cloud.
            augmentation: Augmentation configuration.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_points = max_points
        self.augmentation = augmentation or {}
        
        # Initialize processors
        self.point_cloud_processor = PointCloudProcessor(max_points=max_points)
        
        # Load data paths
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Dict[str, str]]:
        """Load sample paths from dataset directory.
        
        Returns:
            List of sample dictionaries with file paths.
        """
        samples = []
        
        # Waymo structure: data_dir/training/velodyne/ and data_dir/training/label/
        velodyne_dir = self.data_dir / "training" / "velodyne"
        label_dir = self.data_dir / "training" / "label"
        
        if not velodyne_dir.exists():
            print("Waymo dataset not found. Using synthetic data...")
            return self._create_synthetic_samples()
        
        # Get all point cloud files
        point_cloud_files = sorted(velodyne_dir.glob("*.bin"))
        
        for pc_file in point_cloud_files:
            sample_id = pc_file.stem
            
            # Find corresponding label file
            label_file = label_dir / f"{sample_id}.txt"
            
            if label_file.exists() or self.split == "test":
                samples.append({
                    "point_cloud": str(pc_file),
                    "annotation": str(label_file) if label_file.exists() else None,
                    "sample_id": sample_id,
                })
        
        return samples
    
    def _create_synthetic_samples(self) -> List[Dict[str, str]]:
        """Create synthetic samples for testing."""
        samples = []
        
        # Create a few synthetic samples
        for i in range(50):
            samples.append({
                "point_cloud": f"synthetic_{i:06d}.bin",
                "annotation": f"synthetic_{i:06d}.txt",
                "sample_id": f"synthetic_{i:06d}",
            })
        
        return samples
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item.
        
        Args:
            idx: Sample index.
            
        Returns:
            Sample dictionary with point cloud, labels, and bboxes.
        """
        sample = self.samples[idx]
        
        # For synthetic data, generate random point cloud
        if sample["point_cloud"].startswith("synthetic_"):
            point_cloud = self._generate_synthetic_point_cloud()
            labels = ["Car", "Pedestrian"]
            bboxes = [
                torch.tensor([0, 0, 0, 2, 1, 4, 0], dtype=torch.float32),
                torch.tensor([5, 5, 0, 0.8, 1.8, 0.6, 0], dtype=torch.float32),
            ]
        else:
            # Load real data
            point_cloud = self.point_cloud_processor.process_point_cloud(sample["point_cloud"])
            
            # Load annotations
            labels = []
            bboxes = []
            
            if sample["annotation"] and os.path.exists(sample["annotation"]):
                bboxes, labels = self.bbox_processor.process_annotations(sample["annotation"])
        
        return {
            "point_cloud": point_cloud,
            "labels": labels,
            "bboxes": bboxes,
            "sample_id": sample["sample_id"],
        }
    
    def _generate_synthetic_point_cloud(self) -> torch.Tensor:
        """Generate synthetic point cloud for testing."""
        # Generate random points
        points = np.random.randn(self.max_points, 4) * 10
        points[:, 2] = np.abs(points[:, 2])  # Ensure positive z
        
        return torch.tensor(points, dtype=torch.float32)
