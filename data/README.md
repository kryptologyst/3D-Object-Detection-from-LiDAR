# Data directory structure

This directory contains the datasets for 3D object detection from LiDAR.

## Directory Structure

```
data/
├── raw/                    # Raw dataset files
│   ├── kitti/             # KITTI dataset
│   │   ├── velodyne/      # Point cloud files (.bin)
│   │   └── label_2/       # Annotation files (.txt)
│   ├── waymo/             # Waymo dataset
│   │   └── training/      # Training data
│   │       ├── velodyne/  # Point cloud files
│   │       └── label/     # Annotation files
│   └── synthetic/         # Synthetic data for testing
│       ├── point_clouds/  # Generated point clouds
│       └── annotations/   # Generated annotations
└── processed/             # Processed dataset files
    ├── kitti/            # Processed KITTI data
    └── waymo/           # Processed Waymo data
```

## Dataset Formats

### Point Cloud Files
- **KITTI**: `.bin` files (binary format)
- **Waymo**: `.bin` files (binary format)
- **Synthetic**: `.pcd` files (PCL format)

### Annotation Files
- **Format**: KITTI-style text files
- **Extension**: `.txt`
- **Content**: Object annotations with 3D bounding boxes

## Usage

1. **Download datasets** to the appropriate `raw/` subdirectories
2. **Run preprocessing** scripts to convert to processed format
3. **Use processed data** for training and evaluation

## Notes

- Large dataset files are gitignored
- Only directory structure and small sample files are tracked
- Synthetic data is automatically generated for testing
