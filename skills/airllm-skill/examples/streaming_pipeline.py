#!/usr/bin/env python3
"""
AirLLM Pattern Demo — Streaming Pipeline with Prefetching & Compression

This script demonstrates Patterns 1, 4, 5, 6, 7 from the AirLLM skill
in a self-contained example that processes "layers" from disk.

Usage:
    python streaming_pipeline.py

This creates fake layer files, then processes them using the streaming
pipeline with prefetching, showing memory usage at each step.
"""

import gc
import os
import sys
import json
import time
import pickle
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Pattern 6: Aggressive Memory Cleanup
# ═══════════════════════════════════════════════════════════════════

def clean_memory():
    """Three-level memory cleanup. Call after every eviction."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# Pattern 7: Crash-Safe Operations with Marker Files
# ═══════════════════════════════════════════════════════════════════

def save_with_marker(data: dict, path: Path, name: str):
    """Save data with .done marker for crash safety."""
    data_path = path / f"{name}.pkl"
    done_path = path / f"{name}.pkl.done"
    
    done_path.unlink(missing_ok=True)
    
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    
    done_path.touch()
    print(f"  ✅ Saved {name} ({data_path.stat().st_size / 1024:.0f} KB)")


def unit_exists(path: Path, name: str) -> bool:
    """Check if unit was successfully saved (data + marker)."""
    return (path / f"{name}.pkl").exists() and \
           (path / f"{name}.pkl.done").exists()


# ═══════════════════════════════════════════════════════════════════
# Pattern 3: One-Time Preprocessing (Layer Splitting)
# ═══════════════════════════════════════════════════════════════════

def create_fake_model(output_path: Path, n_layers: int = 8, 
                      layer_size: int = 1_000_000):
    """Simulate splitting a large model into per-layer files."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    layer_names = [f"layer_{i}" for i in range(n_layers)]
    
    # Check if already split (idempotency)
    if all(unit_exists(output_path, name) for name in layer_names):
        print(f"✅ Model already split at {output_path}")
        return layer_names
    
    print(f"📦 Splitting model into {n_layers} layers...")
    for name in layer_names:
        if unit_exists(output_path, name):
            print(f"  ⏭️  {name} already done, skipping")
            continue
        
        # Simulate layer weights (normally distributed, like real neural nets)
        weights = {
            'weight_a': np.random.randn(layer_size).astype(np.float16),
            'weight_b': np.random.randn(layer_size // 4).astype(np.float16),
            'bias': np.random.randn(1000).astype(np.float16),
        }
        
        save_with_marker(weights, output_path, name)
        del weights
        clean_memory()
    
    print(f"✅ Split complete: {n_layers} layer files\n")
    return layer_names


# ═══════════════════════════════════════════════════════════════════
# Pattern 4: Disk-Bottleneck Compression (simplified demo)
# ═══════════════════════════════════════════════════════════════════

def compress_weights(weights: dict) -> dict:
    """Simple 8-bit quantization for demo purposes.
    
    Real AirLLM uses bitsandbytes NF4 quantization.
    This demo uses basic min-max scaling to illustrate the concept.
    """
    compressed = {}
    for key, tensor in weights.items():
        t_min, t_max = tensor.min(), tensor.max()
        scale = (t_max - t_min) / 255.0
        quantized = ((tensor - t_min) / scale).astype(np.uint8)
        compressed[key] = {
            'data': quantized,  # 1 byte per value (was 2 bytes)
            'min': float(t_min),
            'max': float(t_max),
            'scale': float(scale),
            'shape': tensor.shape,
        }
    return compressed


def decompress_weights(compressed: dict) -> dict:
    """Decompress back to float16 for full-precision compute."""
    weights = {}
    for key, info in compressed.items():
        reconstructed = info['data'].astype(np.float32) * info['scale'] + info['min']
        weights[key] = reconstructed.astype(np.float16)
    return weights


# ═══════════════════════════════════════════════════════════════════
# Pattern 1 + 5: Streaming Pipeline with Prefetching
# ═══════════════════════════════════════════════════════════════════

def load_layer(layer_path: Path, name: str, 
               use_compression: bool = False) -> dict:
    """Load a layer from disk. Called in background thread for prefetching."""
    with open(layer_path / f"{name}.pkl", 'rb') as f:
        weights = pickle.load(f)
    
    if use_compression:
        weights = compress_weights(weights)
    
    return weights


def process_layer(data: np.ndarray, weights: dict, 
                  use_compression: bool = False) -> np.ndarray:
    """Simulate processing data through one layer."""
    if use_compression:
        weights = decompress_weights(weights)
    
    # Simulate a neural network layer: linear transform + activation
    w = weights['weight_a'][:len(data)]
    result = data * w[:len(data)] + weights['bias'][:len(data)]
    return np.tanh(result)  # activation function


def run_streaming_pipeline(layer_path: Path, layer_names: list[str],
                           input_data: np.ndarray,
                           prefetch: bool = True,
                           compress: bool = False) -> np.ndarray:
    """
    Pattern 1: Layer-wise streaming
    Pattern 5: Prefetching (optional)
    Pattern 4: Compression (optional)
    Pattern 6: Aggressive cleanup
    """
    
    result = input_data.copy()
    timings = []
    
    if prefetch and not compress:
        # ═══ WITH PREFETCHING ═══
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Start loading first layer immediately
            future = executor.submit(load_layer, layer_path, 
                                     layer_names[0], compress)
            
            for i, name in enumerate(layer_names):
                t_start = time.perf_counter()
                
                # Wait for current layer
                weights = future.result()
                t_loaded = time.perf_counter()
                
                # Start loading NEXT layer in background
                if i + 1 < len(layer_names):
                    future = executor.submit(load_layer, layer_path,
                                             layer_names[i + 1], compress)
                
                # Process current layer
                result = process_layer(result, weights, compress)
                t_done = time.perf_counter()
                
                # Evict + cleanup
                del weights
                clean_memory()
                
                timings.append({
                    'layer': name,
                    'load_ms': (t_loaded - t_start) * 1000,
                    'compute_ms': (t_done - t_loaded) * 1000,
                    'total_ms': (t_done - t_start) * 1000,
                })
    else:
        # ═══ WITHOUT PREFETCHING (sequential) ═══
        for i, name in enumerate(layer_names):
            t_start = time.perf_counter()
            
            weights = load_layer(layer_path, name, compress)
            t_loaded = time.perf_counter()
            
            result = process_layer(result, weights, compress)
            t_done = time.perf_counter()
            
            del weights
            clean_memory()
            
            timings.append({
                'layer': name,
                'load_ms': (t_loaded - t_start) * 1000,
                'compute_ms': (t_done - t_loaded) * 1000,
                'total_ms': (t_done - t_start) * 1000,
            })
    
    # Print timing report
    total_load = sum(t['load_ms'] for t in timings)
    total_compute = sum(t['compute_ms'] for t in timings)
    total = sum(t['total_ms'] for t in timings)
    
    mode = "prefetch" if (prefetch and not compress) else "sequential"
    comp = "+compressed" if compress else ""
    
    print(f"\n{'─'*50}")
    print(f"  Mode: {mode} {comp}")
    print(f"  Total load:    {total_load:.1f} ms")
    print(f"  Total compute: {total_compute:.1f} ms")
    print(f"  Total time:    {total:.1f} ms")
    print(f"  Output sample: {result[:3]}")
    print(f"{'─'*50}\n")
    
    return result


# ═══════════════════════════════════════════════════════════════════
# Main Demo
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  🚀 AirLLM Patterns Demo — Streaming Pipeline")
    print("=" * 60)
    print()
    
    # Setup
    work_dir = Path(tempfile.mkdtemp(prefix="airllm_demo_"))
    n_layers = 8
    data_size = 1000
    
    # Pattern 3: One-time split
    layer_names = create_fake_model(work_dir, n_layers=n_layers)
    
    # Create input data
    input_data = np.random.randn(data_size).astype(np.float16)
    print(f"Input shape: {input_data.shape}, sample: {input_data[:3]}\n")
    
    # Run 1: Sequential (no prefetching)
    print("🔄 Run 1: Sequential (no prefetching)")
    run_streaming_pipeline(work_dir, layer_names, input_data, 
                          prefetch=False, compress=False)
    
    # Run 2: With prefetching
    print("🚀 Run 2: With prefetching")
    run_streaming_pipeline(work_dir, layer_names, input_data,
                          prefetch=True, compress=False)
    
    # Run 3: With compression (sequential)
    print("📦 Run 3: With compression (sequential)")
    run_streaming_pipeline(work_dir, layer_names, input_data,
                          prefetch=False, compress=True)
    
    # Cleanup
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)
    
    print("✅ Demo complete! All patterns demonstrated.")
    print(f"\nPatterns used:")
    print(f"  1️⃣  Layer-wise streaming (one layer at a time)")
    print(f"  3️⃣  One-time preprocessing (split into layer files)")
    print(f"  4️⃣  Disk-bottleneck compression (8-bit quantization)")
    print(f"  5️⃣  Prefetching (ThreadPoolExecutor overlap)")
    print(f"  6️⃣  Aggressive memory cleanup (gc + cleanup)")
    print(f"  7️⃣  Crash-safe markers (.done files)")


if __name__ == "__main__":
    main()
