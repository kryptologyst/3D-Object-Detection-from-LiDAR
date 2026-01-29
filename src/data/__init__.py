"""Data loading and preprocessing utilities for 3D object detection."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset


class PointCloudProcessor:
    """Processes point cloud data for 3D object detection."""
    
    def __init__(
        self,
        max_points: int = 16384,
        voxel_size: List[float] = [0.05, 0.05, 0.1],
        point_cloud_range: List[float] = [-50.0, -50.0, -3.0, 50.0, 50.0, 1.0],
    ):
        """Initialize point cloud processor.
        
        Args:
            max_points: Maximum number of points to keep.
            voxel_size: Voxel size for downsampling [x, y, z].
            point_cloud_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        """
        self.max_points = max_points
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
    
    def load_point_cloud(self, file_path: Union[str, Path]) -> np.ndarray:
        """Load point cloud from file.
        
        Args:
            file_path: Path to point cloud file (.pcd, .ply, .bin).
            
        Returns:
            Point cloud as numpy array [N, 3] or [N, 4] if intensity available.
        """
        file_path = Path(file_path)
        
        if file_path.suffix == ".pcd":
            pcd = o3d.io.read_point_cloud(str(file_path))
            points = np.asarray(pcd.points)
            
            # Add intensity if available
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                intensity = np.mean(colors, axis=1, keepdims=True)
                points = np.concatenate([points, intensity], axis=1)
            else:
                # Add dummy intensity
                intensity = np.zeros((points.shape[0], 1))
                points = np.concatenate([points, intensity], axis=1)
                
        elif file_path.suffix == ".bin":
            # KITTI format: x, y, z, intensity
            points = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, 4)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return points
    
    def filter_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Filter point cloud based on range.
        
        Args:
            points: Point cloud [N, 3] or [N, 4].
            
        Returns:
            Filtered point cloud.
        """
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        
        return points[mask]
    
    def downsample_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Downsample point cloud using voxel grid.
        
        Args:
            points: Point cloud [N, 3] or [N, 4].
            
        Returns:
            Downsampled point cloud.
        """
        if len(points) == 0:
            return points
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Add intensity if available
        if points.shape[1] > 3:
            intensity = points[:, 3:4]
        else:
            intensity = np.zeros((points.shape[0], 1))
        
        # Voxel downsampling
        voxel_size = self.voxel_size[0]  # Use x dimension voxel size
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        
        if len(downsampled_pcd.points) == 0:
            return points
        
        downsampled_points = np.asarray(downsampled_pcd.points)
        
        # Re-add intensity (simplified - take mean intensity in each voxel)
        if points.shape[1] > 3:
            # This is a simplified approach - in practice you'd want more sophisticated intensity handling
            downsampled_intensity = np.zeros((downsampled_points.shape[0], 1))
            downsampled_points = np.concatenate([downsampled_points, downsampled_intensity], axis=1)
        
        return downsampled_points
    
    def random_sample_points(self, points: np.ndarray) -> np.ndarray:
        """Randomly sample points to fixed size.
        
        Args:
            points: Point cloud [N, 3] or [N, 4].
            
        Returns:
            Sampled point cloud.
        """
        if len(points) <= self.max_points:
            # Pad with zeros if needed
            if len(points) < self.max_points:
                padding = np.zeros((self.max_points - len(points), points.shape[1]))
                points = np.concatenate([points, padding], axis=0)
            return points
        
        # Random sampling
        indices = np.random.choice(len(points), self.max_points, replace=False)
        return points[indices]
    
    def process_point_cloud(self, file_path: Union[str, Path]) -> torch.Tensor:
        """Complete point cloud processing pipeline.
        
        Args:
            file_path: Path to point cloud file.
            
        Returns:
            Processed point cloud tensor.
        """
        # Load point cloud
        points = self.load_point_cloud(file_path)
        
        # Filter by range
        points = self.filter_point_cloud(points)
        
        # Downsample
        points = self.downsample_point_cloud(points)
        
        # Random sample to fixed size
        points = self.random_sample_points(points)
        
        return torch.tensor(points, dtype=torch.float32)


class BBoxProcessor:
    """Processes 3D bounding boxes for object detection."""
    
    def __init__(self, num_classes: int = 3):
        """Initialize bbox processor.
        
        Args:
            num_classes: Number of object classes.
        """
        self.num_classes = num_classes
    
    def load_annotations(self, annotation_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load annotations from file.
        
        Args:
            annotation_path: Path to annotation file.
            
        Returns:
            List of annotation dictionaries.
        """
        annotations = []
        
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 15:  # KITTI format
                    annotation = {
                        'type': parts[0],
                        'truncated': float(parts[1]),
                        'occluded': int(parts[2]),
                        'alpha': float(parts[3]),
                        'bbox_2d': [float(x) for x in parts[4:8]],
                        'dimensions': [float(x) for x in parts[8:11]],  # h, w, l
                        'location': [float(x) for x in parts[11:14]],  # x, y, z
                        'rotation_y': float(parts[14]),
                    }
                    annotations.append(annotation)
        
        return annotations
    
    def convert_to_3d_bbox(self, annotation: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Convert annotation to 3D bounding box format.
        
        Args:
            annotation: Annotation dictionary.
            
        Returns:
            3D bounding box tensor [x, y, z, w, h, l, yaw] or None if invalid.
        """
        # Map class names to indices
        class_mapping = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        
        if annotation['type'] not in class_mapping:
            return None
        
        # Extract 3D box parameters
        x, y, z = annotation['location']
        h, w, l = annotation['dimensions']
        yaw = annotation['rotation_y']
        
        # Create 3D bounding box tensor
        bbox_3d = torch.tensor([x, y, z, w, h, l, yaw], dtype=torch.float32)
        
        return bbox_3d
    
    def process_annotations(self, annotation_path: Union[str, Path]) -> Tuple[List[torch.Tensor], List[int]]:
        """Process all annotations for a sample.
        
        Args:
            annotation_path: Path to annotation file.
            
        Returns:
            Tuple of (bboxes, labels).
        """
        annotations = self.load_annotations(annotation_path)
        
        bboxes = []
        labels = []
        
        for annotation in annotations:
            bbox_3d = self.convert_to_3d_bbox(annotation)
            if bbox_3d is not None:
                bboxes.append(bbox_3d)
                labels.append(annotation['type'])
        
        return bboxes, labels


def create_synthetic_data(
    num_samples: int = 100,
    num_points_per_sample: int = 1000,
    num_objects_per_sample: int = 5,
    output_dir: str = "data/synthetic",
) -> None:
    """Create synthetic LiDAR data for testing.
    
    Args:
        num_samples: Number of synthetic samples to create.
        num_points_per_sample: Number of points per point cloud.
        num_objects_per_sample: Number of objects per sample.
        output_dir: Output directory for synthetic data.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/point_clouds", exist_ok=True)
    os.makedirs(f"{output_dir}/annotations", exist_ok=True)
    
    processor = PointCloudProcessor()
    bbox_processor = BBoxProcessor()
    
    for i in range(num_samples):
        # Generate random point cloud
        points = np.random.randn(num_points_per_sample, 4) * 10
        points[:, 2] = np.abs(points[:, 2])  # Ensure positive z (height)
        
        # Add some objects (simplified)
        for j in range(num_objects_per_sample):
            # Random object position
            obj_x = np.random.uniform(-20, 20)
            obj_y = np.random.uniform(-20, 20)
            obj_z = np.random.uniform(0, 2)
            
            # Add points around object
            obj_points = np.random.randn(50, 4) * 2
            obj_points[:, 0] += obj_x
            obj_points[:, 1] += obj_y
            obj_points[:, 2] += obj_z
            
            points = np.concatenate([points, obj_points], axis=0)
        
        # Save point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        o3d.io.write_point_cloud(f"{output_dir}/point_clouds/{i:06d}.pcd", pcd)
        
        # Create dummy annotation
        with open(f"{output_dir}/annotations/{i:06d}.txt", 'w') as f:
            f.write("Car 0 0 0 0 0 0 0 2 1 2 0 0 0 0\n")
    
    print(f"Created {num_samples} synthetic samples in {output_dir}")
