#!/usr/bin/env python3
"""Make scripts executable and set up the project."""

import os
import stat
from pathlib import Path


def make_executable(file_path: str) -> None:
    """Make a file executable."""
    if os.path.exists(file_path):
        current_permissions = os.stat(file_path).st_mode
        os.chmod(file_path, current_permissions | stat.S_IEXEC)
        print(f"✅ Made {file_path} executable")


def main():
    """Main setup function."""
    print("🔧 Setting up 3D Object Detection from LiDAR project...")
    
    # Make scripts executable
    scripts_dir = Path("scripts")
    if scripts_dir.exists():
        for script_file in scripts_dir.glob("*.py"):
            make_executable(str(script_file))
    
    # Make demo executable
    demo_file = Path("demo/streamlit_demo.py")
    if demo_file.exists():
        make_executable(str(demo_file))
    
    # Create necessary directories
    directories = [
        "data/raw/kitti/velodyne",
        "data/raw/kitti/label_2", 
        "data/raw/waymo/training/velodyne",
        "data/raw/waymo/training/label",
        "data/processed/kitti",
        "data/processed/waymo",
        "checkpoints",
        "logs",
        "outputs",
        "assets/visualizations",
        "assets/results", 
        "assets/demos",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Create .gitkeep files for empty directories
    gitkeep_dirs = [
        "data/raw/kitti/velodyne",
        "data/raw/kitti/label_2",
        "data/raw/waymo/training/velodyne", 
        "data/raw/waymo/training/label",
        "data/processed/kitti",
        "data/processed/waymo",
    ]
    
    for directory in gitkeep_dirs:
        gitkeep_file = os.path.join(directory, ".gitkeep")
        if not os.path.exists(gitkeep_file):
            with open(gitkeep_file, "w") as f:
                f.write("")
            print(f"📄 Created .gitkeep: {gitkeep_file}")
    
    print("\n🎉 Project setup completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run quick start: python scripts/quick_start.py")
    print("3. Launch demo: streamlit run demo/streamlit_demo.py")


if __name__ == "__main__":
    main()
