#!/usr/bin/env python3
"""Benchmark script for 3D object detection models."""

import argparse
import time
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm

from src.models import PointPillars, SECOND, CenterPoint
from src.utils import get_device


def benchmark_model(
    model: torch.nn.Module,
    input_shape: tuple,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "auto",
) -> Dict[str, float]:
    """Benchmark a model for inference speed and memory usage.
    
    Args:
        model: Model to benchmark.
        input_shape: Input tensor shape (batch_size, num_points, features).
        num_iterations: Number of benchmark iterations.
        warmup_iterations: Number of warmup iterations.
        device: Device to run on.
        
    Returns:
        Dictionary containing benchmark results.
    """
    device_obj = get_device(device)
    model.to(device_obj)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device_obj)
    
    # Warmup
    print(f"Warming up for {warmup_iterations} iterations...")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
    
    # Benchmark
    print(f"Benchmarking for {num_iterations} iterations...")
    times = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_iterations), desc="Benchmarking"):
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            times.append(end_time - start_time)
    
    # Calculate statistics
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1.0 / avg_time
    
    # Memory usage
    if device_obj.type == "cuda":
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**3  # GB
    else:
        memory_allocated = 0.0
        memory_reserved = 0.0
    
    return {
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "fps": fps,
        "memory_allocated_gb": memory_allocated,
        "memory_reserved_gb": memory_reserved,
    }


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark 3D object detection models")
    parser.add_argument("--model", type=str, choices=["pointpillars", "second", "centerpoint"], 
                       default="pointpillars", help="Model to benchmark")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-points", type=int, default=16384, help="Number of points")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    
    args = parser.parse_args()
    
    print(f"🚀 Benchmarking {args.model} model")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of points: {args.num_points}")
    print(f"Iterations: {args.iterations}")
    print("=" * 60)
    
    # Create model
    if args.model == "pointpillars":
        model = PointPillars(
            voxel_size=[0.2, 0.2, 4.0],
            point_cloud_range=[-50.0, -50.0, -3.0, 50.0, 50.0, 1.0],
            max_points_per_voxel=32,
            max_voxels=20000,
            num_classes=3,
            num_anchors=2,
        )
    elif args.model == "second":
        model = SECOND(
            voxel_size=[0.05, 0.05, 0.1],
            point_cloud_range=[-50.0, -50.0, -3.0, 50.0, 50.0, 1.0],
            max_points_per_voxel=35,
            max_voxels=20000,
            num_classes=3,
            num_anchors=2,
        )
    elif args.model == "centerpoint":
        model = CenterPoint(
            voxel_size=[0.05, 0.05, 0.1],
            point_cloud_range=[-50.0, -50.0, -3.0, 50.0, 50.0, 1.0],
            max_points_per_voxel=35,
            max_voxels=20000,
            num_classes=3,
        )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Benchmark
    input_shape = (args.batch_size, args.num_points, 4)
    results = benchmark_model(
        model=model,
        input_shape=input_shape,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        device=args.device,
    )
    
    # Print results
    print("\n📊 Benchmark Results:")
    print("=" * 60)
    print(f"Average inference time: {results['avg_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms")
    print(f"Min inference time: {results['min_time_ms']:.2f} ms")
    print(f"Max inference time: {results['max_time_ms']:.2f} ms")
    print(f"Frames per second: {results['fps']:.2f}")
    
    if results['memory_allocated_gb'] > 0:
        print(f"Memory allocated: {results['memory_allocated_gb']:.2f} GB")
        print(f"Memory reserved: {results['memory_reserved_gb']:.2f} GB")
    
    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    main()
