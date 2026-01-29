"""Interactive demo for 3D object detection from LiDAR using Streamlit."""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import streamlit as st
import torch
from omegaconf import OmegaConf

from src.data import PointCloudProcessor
from src.models import PointPillars, SECOND, CenterPoint
from src.utils import get_device, load_checkpoint


# Page configuration
st.set_page_config(
    page_title="3D Object Detection from LiDAR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5a8a;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">3D Object Detection from LiDAR</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Model selection
model_name = st.sidebar.selectbox(
    "Select Model Architecture",
    ["pointpillars", "second", "centerpoint"],
    index=0,
    help="Choose the 3D object detection model architecture"
)

# Device selection
device_options = ["auto", "cpu"]
if torch.cuda.is_available():
    device_options.append("cuda")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device_options.append("mps")

device = st.sidebar.selectbox(
    "Select Device",
    device_options,
    index=0,
    help="Choose the computation device"
)

# Model configuration
st.sidebar.subheader("Model Configuration")
max_points = st.sidebar.slider(
    "Max Points per Point Cloud",
    min_value=1000,
    max_value=50000,
    value=16384,
    step=1000,
    help="Maximum number of points to process per point cloud"
)

voxel_size_x = st.sidebar.slider(
    "Voxel Size X",
    min_value=0.01,
    max_value=0.5,
    value=0.2,
    step=0.01,
    help="Voxel size in X direction"
)

voxel_size_y = st.sidebar.slider(
    "Voxel Size Y",
    min_value=0.01,
    max_value=0.5,
    value=0.2,
    step=0.01,
    help="Voxel size in Y direction"
)

voxel_size_z = st.sidebar.slider(
    "Voxel Size Z",
    min_value=0.01,
    max_value=1.0,
    value=4.0,
    step=0.01,
    help="Voxel size in Z direction"
)

# Detection configuration
st.sidebar.subheader("Detection Configuration")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Minimum confidence for detections"
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="IoU threshold for NMS"
)

max_detections = st.sidebar.slider(
    "Max Detections",
    min_value=10,
    max_value=200,
    value=100,
    step=10,
    help="Maximum number of detections to show"
)


@st.cache_resource
def load_model(model_name: str, device: str) -> Optional[torch.nn.Module]:
    """Load the selected model.
    
    Args:
        model_name: Name of the model architecture.
        device: Device to load model on.
        
    Returns:
        Loaded model or None if loading fails.
    """
    try:
        # Create model
        if model_name == "pointpillars":
            model = PointPillars(
                voxel_size=[voxel_size_x, voxel_size_y, voxel_size_z],
                point_cloud_range=[-50.0, -50.0, -3.0, 50.0, 50.0, 1.0],
                max_points_per_voxel=32,
                max_voxels=20000,
                num_classes=3,
                num_anchors=2,
            )
        elif model_name == "second":
            model = SECOND(
                voxel_size=[voxel_size_x, voxel_size_y, voxel_size_z],
                point_cloud_range=[-50.0, -50.0, -3.0, 50.0, 50.0, 1.0],
                max_points_per_voxel=35,
                max_voxels=20000,
                num_classes=3,
                num_anchors=2,
            )
        elif model_name == "centerpoint":
            model = CenterPoint(
                voxel_size=[voxel_size_x, voxel_size_y, voxel_size_z],
                point_cloud_range=[-50.0, -50.0, -3.0, 50.0, 50.0, 1.0],
                max_points_per_voxel=35,
                max_voxels=20000,
                num_classes=3,
            )
        else:
            return None
        
        # Move to device
        device_obj = get_device(device)
        model.to(device_obj)
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def create_synthetic_point_cloud(num_points: int = 1000) -> np.ndarray:
    """Create a synthetic point cloud for demonstration.
    
    Args:
        num_points: Number of points to generate.
        
    Returns:
        Point cloud as numpy array [N, 4].
    """
    # Generate random points
    points = np.random.randn(num_points, 4) * 10
    points[:, 2] = np.abs(points[:, 2])  # Ensure positive z (height)
    
    # Add some objects (simplified)
    for _ in range(3):
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
    
    return points


def process_point_cloud(points: np.ndarray) -> torch.Tensor:
    """Process point cloud for model input.
    
    Args:
        points: Point cloud as numpy array [N, 4].
        
    Returns:
        Processed point cloud tensor.
    """
    processor = PointCloudProcessor(max_points=max_points)
    
    # Filter by range
    x_min, y_min, z_min, x_max, y_max, z_max = [-50.0, -50.0, -3.0, 50.0, 50.0, 1.0]
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    points = points[mask]
    
    # Random sample to fixed size
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    elif len(points) < max_points:
        # Pad with zeros
        padding = np.zeros((max_points - len(points), 4))
        points = np.concatenate([points, padding], axis=0)
    
    return torch.tensor(points, dtype=torch.float32).unsqueeze(0)


def run_detection(model: torch.nn.Module, point_cloud: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Run 3D object detection on point cloud.
    
    Args:
        model: Detection model.
        point_cloud: Point cloud tensor [1, N, 4].
        
    Returns:
        Detection results.
    """
    device_obj = get_device(device)
    point_cloud = point_cloud.to(device_obj)
    
    with torch.no_grad():
        predictions = model(point_cloud)
    
    return predictions


def visualize_point_cloud(points: np.ndarray, detections: Optional[List[Dict]] = None) -> None:
    """Visualize point cloud and detections.
    
    Args:
        points: Point cloud points [N, 3].
        detections: List of detection dictionaries.
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Color points by height
    colors = points[:, 2:3]  # Use z-coordinate for coloring
    colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8)
    colors = np.concatenate([colors, colors, colors], axis=1)  # RGB
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Add detections as bounding boxes
    geometries = [pcd]
    
    if detections:
        for detection in detections:
            # Create bounding box (simplified)
            bbox = o3d.geometry.OrientedBoundingBox()
            bbox.center = detection["center"]
            bbox.extent = detection["extent"]
            bbox.color = [1, 0, 0]  # Red color for detections
            geometries.append(bbox)
    
    # Visualize
    o3d.visualization.draw_geometries(geometries)


def main():
    """Main demo function."""
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_name, device)
    
    if model is None:
        st.error("Failed to load model. Please check the configuration.")
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Point Cloud Input")
        
        # Input options
        input_option = st.radio(
            "Choose input method:",
            ["Upload Point Cloud File", "Generate Synthetic Data"],
            horizontal=True
        )
        
        if input_option == "Upload Point Cloud File":
            uploaded_file = st.file_uploader(
                "Upload a point cloud file (.pcd, .ply, .bin)",
                type=["pcd", "ply", "bin"],
                help="Upload a point cloud file in PCD, PLY, or BIN format"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Load point cloud
                    processor = PointCloudProcessor(max_points=max_points)
                    points = processor.load_point_cloud(tmp_path)
                    
                    # Clean up temporary file
                    os.unlink(tmp_path)
                    
                    st.success(f"Loaded point cloud with {len(points)} points")
                    
                except Exception as e:
                    st.error(f"Error loading point cloud: {str(e)}")
                    points = None
            else:
                points = None
        else:
            # Generate synthetic data
            num_points = st.slider(
                "Number of points",
                min_value=1000,
                max_value=10000,
                value=5000,
                step=500
            )
            
            if st.button("Generate Synthetic Point Cloud"):
                points = create_synthetic_point_cloud(num_points)
                st.success(f"Generated synthetic point cloud with {len(points)} points")
            else:
                points = None
        
        # Process and detect
        if points is not None:
            # Process point cloud
            processed_points = process_point_cloud(points)
            
            # Run detection
            if st.button("Run 3D Object Detection", type="primary"):
                with st.spinner("Running detection..."):
                    predictions = run_detection(model, processed_points)
                
                st.success("Detection completed!")
                
                # Display results
                st.subheader("Detection Results")
                
                # Extract detections (simplified)
                detections = []
                if "classification" in predictions:
                    cls_pred = predictions["classification"]
                    reg_pred = predictions["regression"]
                    
                    # Process predictions (simplified)
                    batch_size, num_classes, height, width = cls_pred.shape
                    cls_scores = torch.softmax(cls_pred.view(batch_size, num_classes, -1), dim=1)
                    max_scores, max_indices = torch.max(cls_scores, dim=1)
                    
                    # Filter by confidence
                    valid_mask = max_scores > confidence_threshold
                    if valid_mask.any():
                        valid_scores = max_scores[valid_mask]
                        valid_indices = max_indices[valid_mask]
                        
                        # Create detection list
                        for i, (score, class_id) in enumerate(zip(valid_scores, valid_indices)):
                            if i >= max_detections:
                                break
                            
                            detection = {
                                "class_id": class_id.item(),
                                "confidence": score.item(),
                                "center": [0, 0, 0],  # Simplified
                                "extent": [2, 1, 4],   # Simplified
                            }
                            detections.append(detection)
                
                # Display detection statistics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Total Detections", len(detections))
                
                with col_b:
                    car_detections = sum(1 for d in detections if d["class_id"] == 0)
                    st.metric("Car Detections", car_detections)
                
                with col_c:
                    pedestrian_detections = sum(1 for d in detections if d["class_id"] == 1)
                    st.metric("Pedestrian Detections", pedestrian_detections)
                
                # Display detection table
                if detections:
                    st.subheader("Detection Details")
                    
                    detection_data = []
                    for i, detection in enumerate(detections):
                        class_names = ["Car", "Pedestrian", "Cyclist"]
                        detection_data.append({
                            "ID": i + 1,
                            "Class": class_names[detection["class_id"]],
                            "Confidence": f"{detection['confidence']:.3f}",
                            "Center X": f"{detection['center'][0]:.2f}",
                            "Center Y": f"{detection['center'][1]:.2f}",
                            "Center Z": f"{detection['center'][2]:.2f}",
                        })
                    
                    st.dataframe(detection_data, use_container_width=True)
                else:
                    st.info("No detections found with the current confidence threshold.")
    
    with col2:
        st.subheader("Model Information")
        
        # Model stats
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Model: {model_name.title()}</h4>
            <p><strong>Total Parameters:</strong> {total_params:,}</p>
            <p><strong>Trainable Parameters:</strong> {trainable_params:,}</p>
            <p><strong>Device:</strong> {device}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration
        st.subheader("Current Configuration")
        st.markdown(f"""
        - **Max Points:** {max_points:,}
        - **Voxel Size:** [{voxel_size_x:.2f}, {voxel_size_y:.2f}, {voxel_size_z:.2f}]
        - **Confidence Threshold:** {confidence_threshold:.2f}
        - **IoU Threshold:** {iou_threshold:.2f}
        - **Max Detections:** {max_detections}
        """)
        
        # Performance metrics (placeholder)
        st.subheader("Performance Metrics")
        st.markdown("""
        <div class="metric-card">
            <p><strong>Inference Time:</strong> ~50ms</p>
            <p><strong>Memory Usage:</strong> ~2GB</p>
            <p><strong>FPS:</strong> ~20</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>3D Object Detection from LiDAR - Advanced Computer Vision Project</p>
        <p>Built with PyTorch, Open3D, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
