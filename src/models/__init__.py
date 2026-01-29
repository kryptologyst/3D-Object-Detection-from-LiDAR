"""Model architectures for 3D object detection from LiDAR."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import (
    DetectionHead,
    PillarFeatureNet,
    PointNetLayer,
    SparseConvBlock,
    TNet,
    VoxelFeatureExtractor,
)


class PointNetBackbone(nn.Module):
    """PointNet backbone for point cloud feature extraction."""
    
    def __init__(
        self,
        input_channels: int = 4,
        hidden_channels: List[int] = [64, 128, 256],
        output_channels: int = 256,
    ):
        """Initialize PointNet backbone.
        
        Args:
            input_channels: Number of input channels (x, y, z, intensity).
            hidden_channels: Hidden layer channels.
            output_channels: Output channels.
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Input transformation
        self.input_transform = TNet(input_channels)
        
        # PointNet layers
        self.pointnet_layers = nn.ModuleList()
        prev_channels = input_channels
        
        for hidden_channels in hidden_channels:
            self.pointnet_layers.append(
                PointNetLayer(
                    prev_channels,
                    [hidden_channels // 2, hidden_channels],
                    hidden_channels,
                    use_tnet=True,
                )
            )
            prev_channels = hidden_channels
        
        # Final layer
        self.final_layer = nn.Sequential(
            nn.Conv1d(prev_channels, output_channels, 1),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
        )
        
        # Global max pooling
        self.maxpool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Point cloud tensor [B, N, C] or [B, C, N].
            
        Returns:
            Global features [B, output_channels].
        """
        # Ensure input is [B, C, N]
        if x.dim() == 3 and x.shape[1] != self.input_channels:
            x = x.transpose(1, 2)
        
        # Apply input transformation
        trans = self.input_transform(x)
        x = torch.bmm(trans, x)
        
        # Apply PointNet layers
        for layer in self.pointnet_layers:
            x = layer(x)
        
        # Final layer
        x = self.final_layer(x)
        
        # Global max pooling
        x = self.maxpool(x).squeeze(-1)  # [B, output_channels]
        
        return x


class PointPillars(nn.Module):
    """PointPillars model for 3D object detection."""
    
    def __init__(
        self,
        voxel_size: List[float] = [0.2, 0.2, 4.0],
        point_cloud_range: List[float] = [-50.0, -50.0, -3.0, 50.0, 50.0, 1.0],
        max_points_per_voxel: int = 32,
        max_voxels: int = 20000,
        num_classes: int = 3,
        num_anchors: int = 2,
    ):
        """Initialize PointPillars model.
        
        Args:
            voxel_size: Voxel size [x, y, z].
            point_cloud_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
            max_points_per_voxel: Maximum points per voxel.
            max_voxels: Maximum number of voxels.
            num_classes: Number of object classes.
            num_anchors: Number of anchors per location.
        """
        super().__init__()
        
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Voxel feature extractor
        self.voxel_feature_extractor = VoxelFeatureExtractor(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
        )
        
        # Pillar feature network
        self.pillar_feature_net = PillarFeatureNet(
            num_features=64,
            max_points_per_pillar=max_points_per_voxel,
        )
        
        # Backbone network
        self.backbone = nn.Sequential(
            SparseConvBlock(64, 64, 3, 1, 1),
            SparseConvBlock(64, 64, 3, 1, 1),
            SparseConvBlock(64, 64, 3, 2, 1),  # Downsample
            
            SparseConvBlock(64, 128, 3, 1, 1),
            SparseConvBlock(128, 128, 3, 1, 1),
            SparseConvBlock(128, 128, 3, 2, 1),  # Downsample
            
            SparseConvBlock(128, 256, 3, 1, 1),
            SparseConvBlock(256, 256, 3, 1, 1),
            SparseConvBlock(256, 256, 3, 2, 1),  # Downsample
            
            SparseConvBlock(256, 512, 3, 1, 1),
            SparseConvBlock(512, 512, 3, 1, 1),
        )
        
        # Detection head
        self.detection_head = DetectionHead(
            in_channels=512,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )
        
        # Calculate grid size
        self.grid_size = [
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
        ]
    
    def forward(self, point_clouds: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            point_clouds: Point cloud tensor [B, N, 4].
            
        Returns:
            Dictionary containing detection outputs.
        """
        batch_size = point_clouds.shape[0]
        
        # Process each point cloud in the batch
        batch_features = []
        
        for i in range(batch_size):
            # Extract voxel features
            voxel_features, voxel_coords, voxel_num_points = self.voxel_feature_extractor(
                point_clouds[i]
            )
            
            if len(voxel_features) == 0:
                # Empty point cloud - create dummy features
                features = torch.zeros(1, 64, device=point_clouds.device)
            else:
                # Extract pillar features
                features = self.pillar_feature_net(
                    voxel_features, voxel_coords, voxel_num_points
                )
            
            batch_features.append(features)
        
        # Stack features
        batch_features = torch.cat(batch_features, dim=0)  # [B, 64]
        
        # Reshape to spatial format
        spatial_features = batch_features.view(batch_size, 64, 1, 1)
        
        # Apply backbone
        backbone_features = self.backbone(spatial_features)
        
        # Detection head
        cls_out, reg_out, dir_out = self.detection_head(backbone_features)
        
        return {
            "classification": cls_out,
            "regression": reg_out,
            "direction": dir_out,
        }


class SECOND(nn.Module):
    """SECOND model for 3D object detection."""
    
    def __init__(
        self,
        voxel_size: List[float] = [0.05, 0.05, 0.1],
        point_cloud_range: List[float] = [-50.0, -50.0, -3.0, 50.0, 50.0, 1.0],
        max_points_per_voxel: int = 35,
        max_voxels: int = 20000,
        num_classes: int = 3,
        num_anchors: int = 2,
    ):
        """Initialize SECOND model.
        
        Args:
            voxel_size: Voxel size [x, y, z].
            point_cloud_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
            max_points_per_voxel: Maximum points per voxel.
            max_voxels: Maximum number of voxels.
            num_classes: Number of object classes.
            num_anchors: Number of anchors per location.
        """
        super().__init__()
        
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Voxel feature extractor
        self.voxel_feature_extractor = VoxelFeatureExtractor(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
        )
        
        # Sparse convolution backbone
        self.backbone = nn.Sequential(
            SparseConvBlock(4, 32, 3, 1, 1),
            SparseConvBlock(32, 32, 3, 1, 1),
            SparseConvBlock(32, 64, 3, 2, 1),  # Downsample
            
            SparseConvBlock(64, 64, 3, 1, 1),
            SparseConvBlock(64, 64, 3, 1, 1),
            SparseConvBlock(64, 128, 3, 2, 1),  # Downsample
            
            SparseConvBlock(128, 128, 3, 1, 1),
            SparseConvBlock(128, 128, 3, 1, 1),
            SparseConvBlock(128, 256, 3, 2, 1),  # Downsample
            
            SparseConvBlock(256, 256, 3, 1, 1),
            SparseConvBlock(256, 256, 3, 1, 1),
        )
        
        # Detection head
        self.detection_head = DetectionHead(
            in_channels=256,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )
    
    def forward(self, point_clouds: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            point_clouds: Point cloud tensor [B, N, 4].
            
        Returns:
            Dictionary containing detection outputs.
        """
        batch_size = point_clouds.shape[0]
        
        # Process each point cloud in the batch
        batch_features = []
        
        for i in range(batch_size):
            # Extract voxel features
            voxel_features, voxel_coords, voxel_num_points = self.voxel_feature_extractor(
                point_clouds[i]
            )
            
            if len(voxel_features) == 0:
                # Empty point cloud - create dummy features
                features = torch.zeros(1, 4, device=point_clouds.device)
            else:
                # Use voxel features directly
                features = voxel_features.mean(dim=1)  # [N, 4]
            
            batch_features.append(features)
        
        # Stack features
        batch_features = torch.cat(batch_features, dim=0)  # [B, 4]
        
        # Reshape to spatial format
        spatial_features = batch_features.view(batch_size, 4, 1, 1)
        
        # Apply backbone
        backbone_features = self.backbone(spatial_features)
        
        # Detection head
        cls_out, reg_out, dir_out = self.detection_head(backbone_features)
        
        return {
            "classification": cls_out,
            "regression": reg_out,
            "direction": dir_out,
        }


class CenterPoint(nn.Module):
    """CenterPoint model for 3D object detection."""
    
    def __init__(
        self,
        voxel_size: List[float] = [0.05, 0.05, 0.1],
        point_cloud_range: List[float] = [-50.0, -50.0, -3.0, 50.0, 50.0, 1.0],
        max_points_per_voxel: int = 35,
        max_voxels: int = 20000,
        num_classes: int = 3,
    ):
        """Initialize CenterPoint model.
        
        Args:
            voxel_size: Voxel size [x, y, z].
            point_cloud_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
            max_points_per_voxel: Maximum points per voxel.
            max_voxels: Maximum number of voxels.
            num_classes: Number of object classes.
        """
        super().__init__()
        
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.num_classes = num_classes
        
        # Voxel feature extractor
        self.voxel_feature_extractor = VoxelFeatureExtractor(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
        )
        
        # Backbone network
        self.backbone = nn.Sequential(
            SparseConvBlock(4, 64, 3, 1, 1),
            SparseConvBlock(64, 64, 3, 1, 1),
            SparseConvBlock(64, 128, 3, 2, 1),  # Downsample
            
            SparseConvBlock(128, 128, 3, 1, 1),
            SparseConvBlock(128, 128, 3, 1, 1),
            SparseConvBlock(128, 256, 3, 2, 1),  # Downsample
            
            SparseConvBlock(256, 256, 3, 1, 1),
            SparseConvBlock(256, 256, 3, 1, 1),
            SparseConvBlock(256, 512, 3, 2, 1),  # Downsample
            
            SparseConvBlock(512, 512, 3, 1, 1),
            SparseConvBlock(512, 512, 3, 1, 1),
        )
        
        # Center head
        self.center_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),  # Center heatmap
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 7, 1),  # x, y, z, w, h, l, yaw
        )
    
    def forward(self, point_clouds: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            point_clouds: Point cloud tensor [B, N, 4].
            
        Returns:
            Dictionary containing detection outputs.
        """
        batch_size = point_clouds.shape[0]
        
        # Process each point cloud in the batch
        batch_features = []
        
        for i in range(batch_size):
            # Extract voxel features
            voxel_features, voxel_coords, voxel_num_points = self.voxel_feature_extractor(
                point_clouds[i]
            )
            
            if len(voxel_features) == 0:
                # Empty point cloud - create dummy features
                features = torch.zeros(1, 4, device=point_clouds.device)
            else:
                # Use voxel features directly
                features = voxel_features.mean(dim=1)  # [N, 4]
            
            batch_features.append(features)
        
        # Stack features
        batch_features = torch.cat(batch_features, dim=0)  # [B, 4]
        
        # Reshape to spatial format
        spatial_features = batch_features.view(batch_size, 4, 1, 1)
        
        # Apply backbone
        backbone_features = self.backbone(spatial_features)
        
        # Center head
        center_out = self.center_head(backbone_features)
        
        # Regression head
        reg_out = self.reg_head(backbone_features)
        
        return {
            "center": center_out,
            "regression": reg_out,
        }
