"""Tests for 3D object detection models."""

import pytest
import torch
import numpy as np

from src.models import PointPillars, SECOND, CenterPoint, PointNetBackbone
from src.data import KITTIDataset, PointCloudProcessor, BBoxProcessor
from src.utils import get_device, set_seed, collate_fn


class TestModels:
    """Test cases for model architectures."""
    
    def test_pointpillars_forward(self):
        """Test PointPillars forward pass."""
        model = PointPillars(
            voxel_size=[0.2, 0.2, 4.0],
            point_cloud_range=[-50.0, -50.0, -3.0, 50.0, 50.0, 1.0],
            max_points_per_voxel=32,
            max_voxels=20000,
            num_classes=3,
            num_anchors=2,
        )
        
        # Create dummy input
        batch_size = 2
        num_points = 1000
        point_clouds = torch.randn(batch_size, num_points, 4)
        
        # Forward pass
        outputs = model(point_clouds)
        
        # Check output structure
        assert "classification" in outputs
        assert "regression" in outputs
        assert "direction" in outputs
        
        # Check output shapes
        assert outputs["classification"].shape[0] == batch_size
        assert outputs["regression"].shape[0] == batch_size
        assert outputs["direction"].shape[0] == batch_size
    
    def test_second_forward(self):
        """Test SECOND forward pass."""
        model = SECOND(
            voxel_size=[0.05, 0.05, 0.1],
            point_cloud_range=[-50.0, -50.0, -3.0, 50.0, 50.0, 1.0],
            max_points_per_voxel=35,
            max_voxels=20000,
            num_classes=3,
            num_anchors=2,
        )
        
        # Create dummy input
        batch_size = 2
        num_points = 1000
        point_clouds = torch.randn(batch_size, num_points, 4)
        
        # Forward pass
        outputs = model(point_clouds)
        
        # Check output structure
        assert "classification" in outputs
        assert "regression" in outputs
        assert "direction" in outputs
    
    def test_centerpoint_forward(self):
        """Test CenterPoint forward pass."""
        model = CenterPoint(
            voxel_size=[0.05, 0.05, 0.1],
            point_cloud_range=[-50.0, -50.0, -3.0, 50.0, 50.0, 1.0],
            max_points_per_voxel=35,
            max_voxels=20000,
            num_classes=3,
        )
        
        # Create dummy input
        batch_size = 2
        num_points = 1000
        point_clouds = torch.randn(batch_size, num_points, 4)
        
        # Forward pass
        outputs = model(point_clouds)
        
        # Check output structure
        assert "center" in outputs
        assert "regression" in outputs
    
    def test_pointnet_backbone(self):
        """Test PointNet backbone."""
        backbone = PointNetBackbone(
            input_channels=4,
            hidden_channels=[64, 128, 256],
            output_channels=256,
        )
        
        # Create dummy input
        batch_size = 2
        num_points = 1000
        point_clouds = torch.randn(batch_size, num_points, 4)
        
        # Forward pass
        features = backbone(point_clouds)
        
        # Check output shape
        assert features.shape == (batch_size, 256)


class TestDataProcessing:
    """Test cases for data processing."""
    
    def test_point_cloud_processor(self):
        """Test point cloud processing."""
        processor = PointCloudProcessor(
            max_points=1000,
            voxel_size=[0.1, 0.1, 0.1],
            point_cloud_range=[-10.0, -10.0, -2.0, 10.0, 10.0, 2.0],
        )
        
        # Create dummy point cloud
        points = np.random.randn(2000, 4) * 5
        
        # Process point cloud
        processed_points = processor.process_point_cloud("dummy_path")
        
        # Check output
        assert isinstance(processed_points, torch.Tensor)
        assert processed_points.shape[0] == 1000  # max_points
        assert processed_points.shape[1] == 4    # x, y, z, intensity
    
    def test_bbox_processor(self):
        """Test bounding box processing."""
        processor = BBoxProcessor(num_classes=3)
        
        # Create dummy annotation
        annotation = {
            'type': 'Car',
            'location': [1.0, 2.0, 0.5],
            'dimensions': [1.8, 1.6, 4.0],
            'rotation_y': 0.1,
        }
        
        # Convert to 3D bbox
        bbox = processor.convert_to_3d_bbox(annotation)
        
        # Check output
        assert bbox is not None
        assert isinstance(bbox, torch.Tensor)
        assert bbox.shape == (7,)  # x, y, z, w, h, l, yaw
    
    def test_collate_fn(self):
        """Test custom collate function."""
        # Create dummy batch
        batch = [
            {
                "point_cloud": torch.randn(1000, 4),
                "labels": ["Car", "Pedestrian"],
                "bboxes": [torch.randn(7), torch.randn(7)],
            },
            {
                "point_cloud": torch.randn(1500, 4),
                "labels": ["Cyclist"],
                "bboxes": [torch.randn(7)],
            },
        ]
        
        # Apply collate function
        collated = collate_fn(batch)
        
        # Check output structure
        assert "point_clouds" in collated
        assert "point_cloud_masks" in collated
        assert "labels" in collated
        assert "bboxes" in collated
        
        # Check shapes
        assert collated["point_clouds"].shape[0] == 2  # batch size
        assert collated["point_clouds"].shape[1] == 1500  # max points
        assert collated["point_cloud_masks"].shape == (2, 1500)


class TestUtils:
    """Test cases for utility functions."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate random numbers
        torch_rand = torch.rand(5)
        np_rand = np.random.rand(5)
        
        # Set seed again and generate same numbers
        set_seed(42)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        # Check reproducibility
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)


class TestDataset:
    """Test cases for dataset classes."""
    
    def test_kitti_dataset_synthetic(self):
        """Test KITTI dataset with synthetic data."""
        # Create dataset with non-existent path to trigger synthetic data creation
        dataset = KITTIDataset(
            data_dir="non_existent_path",
            split="train",
            max_points=1000,
        )
        
        # Check dataset length
        assert len(dataset) > 0
        
        # Get a sample
        sample = dataset[0]
        
        # Check sample structure
        assert "point_cloud" in sample
        assert "labels" in sample
        assert "bboxes" in sample
        assert "sample_id" in sample
        
        # Check point cloud shape
        assert sample["point_cloud"].shape[0] == 1000  # max_points
        assert sample["point_cloud"].shape[1] == 4    # x, y, z, intensity


if __name__ == "__main__":
    pytest.main([__file__])
