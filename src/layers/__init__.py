"""Custom layers for 3D object detection models."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetLayer(nn.Module):
    """PointNet layer with T-Net transformation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        use_tnet: bool = True,
    ):
        """Initialize PointNet layer.
        
        Args:
            input_dim: Input dimension.
            hidden_dims: Hidden layer dimensions.
            output_dim: Output dimension.
            use_tnet: Whether to use T-Net transformation.
        """
        super().__init__()
        
        self.use_tnet = use_tnet
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(prev_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Conv1d(prev_dim, output_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        # T-Net transformation
        if use_tnet:
            self.tnet = TNet(input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, C, N].
            
        Returns:
            Output tensor [B, C', N].
        """
        if self.use_tnet:
            # Apply T-Net transformation
            trans = self.tnet(x)
            x = torch.bmm(trans, x)
        
        # Apply MLP
        x = self.mlp(x)
        
        return x


class TNet(nn.Module):
    """T-Net transformation network."""
    
    def __init__(self, input_dim: int):
        """Initialize T-Net.
        
        Args:
            input_dim: Input dimension.
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # MLP for transformation matrix
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        
        # Global max pooling
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        
        # Transformation matrix head
        self.transform_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, input_dim * input_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, C, N].
            
        Returns:
            Transformation matrix [B, C, C].
        """
        # Apply MLP
        x = self.mlp(x)
        
        # Global max pooling
        x = self.maxpool(x).squeeze(-1)  # [B, 1024]
        
        # Generate transformation matrix
        transform = self.transform_head(x)  # [B, C*C]
        transform = transform.view(-1, self.input_dim, self.input_dim)
        
        # Add identity matrix for stability
        identity = torch.eye(self.input_dim, device=x.device, dtype=x.dtype)
        transform = transform + identity
        
        return transform


class VoxelFeatureExtractor(nn.Module):
    """Voxel feature extraction for point clouds."""
    
    def __init__(
        self,
        voxel_size: List[float],
        point_cloud_range: List[float],
        max_points_per_voxel: int = 35,
        max_voxels: int = 20000,
    ):
        """Initialize voxel feature extractor.
        
        Args:
            voxel_size: Voxel size [x, y, z].
            point_cloud_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
            max_points_per_voxel: Maximum points per voxel.
            max_voxels: Maximum number of voxels.
        """
        super().__init__()
        
        self.voxel_size = torch.tensor(voxel_size)
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        
        # Calculate grid size
        self.grid_size = ((self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size).long()
    
    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Voxelize point cloud.
        
        Args:
            points: Point cloud [N, 4] (x, y, z, intensity).
            
        Returns:
            Tuple of (voxel_features, voxel_coords, voxel_num_points).
        """
        # Shift points to start from origin
        shifted_points = points[:, :3] - self.point_cloud_range[:3]
        
        # Calculate voxel coordinates
        voxel_coords = torch.floor(shifted_points / self.voxel_size).long()
        
        # Filter out points outside range
        valid_mask = torch.all(
            (voxel_coords >= 0) & (voxel_coords < self.grid_size), dim=1
        )
        
        if not valid_mask.any():
            # Return empty tensors if no valid points
            return (
                torch.empty(0, self.max_points_per_voxel, 4),
                torch.empty(0, 3),
                torch.empty(0),
            )
        
        points = points[valid_mask]
        voxel_coords = voxel_coords[valid_mask]
        
        # Create voxel indices
        voxel_indices = (
            voxel_coords[:, 0] * self.grid_size[1] * self.grid_size[2] +
            voxel_coords[:, 1] * self.grid_size[2] +
            voxel_coords[:, 2]
        )
        
        # Group points by voxel
        unique_voxel_indices, inverse_indices = torch.unique(voxel_indices, return_inverse=True)
        
        # Limit number of voxels
        if len(unique_voxel_indices) > self.max_voxels:
            # Randomly sample voxels
            sampled_indices = torch.randperm(len(unique_voxel_indices))[:self.max_voxels]
            unique_voxel_indices = unique_voxel_indices[sampled_indices]
            
            # Create mask for sampled voxels
            mask = torch.isin(inverse_indices, sampled_indices)
            points = points[mask]
            voxel_coords = voxel_coords[mask]
            voxel_indices = voxel_indices[mask]
            
            # Recompute inverse indices
            _, inverse_indices = torch.unique(voxel_indices, return_inverse=True)
        
        # Convert back to voxel coordinates
        voxel_coords_final = torch.zeros(len(unique_voxel_indices), 3, dtype=torch.long)
        voxel_coords_final[:, 0] = unique_voxel_indices // (self.grid_size[1] * self.grid_size[2])
        voxel_coords_final[:, 1] = (unique_voxel_indices % (self.grid_size[1] * self.grid_size[2])) // self.grid_size[2]
        voxel_coords_final[:, 2] = unique_voxel_indices % self.grid_size[2]
        
        # Create voxel features
        voxel_features = torch.zeros(len(unique_voxel_indices), self.max_points_per_voxel, 4)
        voxel_num_points = torch.zeros(len(unique_voxel_indices), dtype=torch.long)
        
        for i in range(len(unique_voxel_indices)):
            voxel_mask = inverse_indices == i
            voxel_points = points[voxel_mask]
            
            num_points = min(len(voxel_points), self.max_points_per_voxel)
            voxel_features[i, :num_points] = voxel_points[:num_points]
            voxel_num_points[i] = num_points
        
        return voxel_features, voxel_coords_final, voxel_num_points


class PillarFeatureNet(nn.Module):
    """Pillar feature network for PointPillars."""
    
    def __init__(
        self,
        num_features: int = 64,
        max_points_per_pillar: int = 32,
    ):
        """Initialize pillar feature network.
        
        Args:
            num_features: Number of output features.
            max_points_per_pillar: Maximum points per pillar.
        """
        super().__init__()
        
        self.max_points_per_pillar = max_points_per_pillar
        
        # Point feature extraction
        self.point_net = nn.Sequential(
            nn.Conv1d(9, 64, 1),  # x, y, z, intensity, x_c, y_c, z_c, x_p, y_p
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, num_features, 1),
        )
        
        # Max pooling
        self.maxpool = nn.AdaptiveMaxPool1d(1)
    
    def forward(
        self,
        pillar_features: torch.Tensor,
        pillar_coords: torch.Tensor,
        pillar_num_points: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            pillar_features: Pillar features [N, P, 4].
            pillar_coords: Pillar coordinates [N, 2].
            pillar_num_points: Number of points per pillar [N].
            
        Returns:
            Pillar features [N, C].
        """
        # Create additional features
        batch_size = pillar_features.shape[0]
        
        # Center of pillar
        pillar_center = pillar_coords.float()
        
        # Point offsets from pillar center
        point_offsets = pillar_features[:, :, :3] - pillar_center.unsqueeze(1)
        
        # Point offsets from pillar mean
        pillar_mean = torch.sum(pillar_features[:, :, :3], dim=1, keepdim=True) / pillar_num_points.unsqueeze(1).float()
        point_offsets_mean = pillar_features[:, :, :3] - pillar_mean
        
        # Concatenate all features
        enhanced_features = torch.cat([
            pillar_features,  # x, y, z, intensity
            point_offsets,    # x_c, y_c, z_c
            point_offsets_mean,  # x_p, y_p, z_p
        ], dim=-1)
        
        # Apply point net
        enhanced_features = enhanced_features.transpose(1, 2)  # [N, 9, P]
        features = self.point_net(enhanced_features)  # [N, C, P]
        
        # Max pooling
        features = self.maxpool(features).squeeze(-1)  # [N, C]
        
        return features


class SparseConvBlock(nn.Module):
    """Sparse convolution block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        """Initialize sparse conv block.
        
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size.
            stride: Stride.
            padding: Padding.
        """
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DetectionHead(nn.Module):
    """Detection head for 3D object detection."""
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 2,
        num_reg_channels: int = 7,  # x, y, z, w, h, l, yaw
    ):
        """Initialize detection head.
        
        Args:
            in_channels: Input channels.
            num_classes: Number of object classes.
            num_anchors: Number of anchors per location.
            num_reg_channels: Number of regression channels.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_reg_channels = num_reg_channels
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes * num_anchors, 1),
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_reg_channels * num_anchors, 1),
        )
        
        # Direction head
        self.dir_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2 * num_anchors, 1),  # 2 directions
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input feature map.
            
        Returns:
            Tuple of (classification, regression, direction) outputs.
        """
        cls_out = self.cls_head(x)
        reg_out = self.reg_head(x)
        dir_out = self.dir_head(x)
        
        return cls_out, reg_out, dir_out
