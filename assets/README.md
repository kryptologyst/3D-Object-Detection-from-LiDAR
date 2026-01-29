# Assets Directory

This directory contains visualizations, results, and other generated assets.

## Directory Structure

```
assets/
├── visualizations/        # Generated visualizations
│   ├── point_clouds/      # Point cloud visualizations
│   ├── detections/        # Detection result visualizations
│   └── attention_maps/    # Attention map visualizations
├── results/               # Evaluation results
│   ├── metrics/           # Metric plots and tables
│   ├── comparisons/       # Model comparison results
│   └── reports/           # Evaluation reports
├── demos/                 # Demo outputs
│   ├── screenshots/       # Demo screenshots
│   └── videos/           # Demo videos
└── checkpoints/           # Model checkpoints (symlinked)
```

## Usage

- **Training scripts** automatically save visualizations here
- **Evaluation scripts** save results and plots here
- **Demo applications** save outputs here
- **Checkpoints** are symlinked from the main checkpoints directory

## File Types

- **Images**: `.png`, `.jpg`, `.svg`
- **Videos**: `.mp4`, `.gif`
- **Data**: `.json`, `.csv`, `.pkl`
- **Models**: `.pth`, `.pt` (symlinked)
