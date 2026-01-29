#!/usr/bin/env python3
"""Quick start script for 3D object detection from LiDAR."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main quick start function."""
    print("🚀 3D Object Detection from LiDAR - Quick Start")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "checkpoints",
        "logs",
        "assets",
        "outputs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but continuing...")
    
    # Create synthetic data
    if not run_command("python -c \"from src.data import create_synthetic_data; create_synthetic_data(num_samples=10)\"", "Creating synthetic data"):
        print("⚠️  Failed to create synthetic data")
    
    # Test model creation
    if not run_command("python -c \"from src.models import PointPillars; model = PointPillars(); print(f'Model created with {sum(p.numel() for p in model.parameters()):,} parameters')\"", "Testing model creation"):
        print("❌ Failed to create model")
        sys.exit(1)
    
    print("\n🎉 Quick start completed successfully!")
    print("\nNext steps:")
    print("1. 📊 Launch the interactive demo:")
    print("   streamlit run demo/streamlit_demo.py")
    print("\n2. 🏋️  Train a model:")
    print("   python scripts/train.py --model pointpillars --epochs 10")
    print("\n3. 📈 Evaluate a model:")
    print("   python scripts/evaluate.py --checkpoint checkpoints/best_model.pth")
    print("\n4. 📚 Read the documentation:")
    print("   See README.md for detailed instructions")


if __name__ == "__main__":
    main()
