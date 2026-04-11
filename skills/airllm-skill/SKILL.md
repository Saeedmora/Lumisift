---
name: airllm-resource-constrained-inference
description: Apply AirLLM's proven engineering patterns to build systems that process massive data through resource-constrained hardware using layer-wise streaming, disk-bottleneck compression, prefetching pipelines, and aggressive memory management.
---

# AirLLM Engineering Patterns Skill

## Overview

This skill teaches you to apply **10 battle-tested engineering patterns** extracted from the AirLLM framework (14,500+ GitHub stars), which runs 70B-parameter LLMs on 4GB GPUs. These patterns are **universally applicable** to any system where data is too large to fit in memory.

## When To Use This Skill

Use these patterns when you encounter:
- **Memory-constrained processing**: Data/model doesn't fit in RAM/VRAM
- **I/O-bound workloads**: Disk or network transfer is the bottleneck, not compute
- **Sequential pipeline processing**: Data flows through stages (A→B→C→D)
- **Large file processing**: Working with files larger than available memory
- **Cross-platform systems**: Code must run on different hardware (CUDA, Apple Silicon, CPU-only)
- **Crash-safe long operations**: Multi-step processes that must survive interruption

---

## Pattern 1: Layer-Wise Streaming (Core Pattern)

### Problem
You need to process data through a pipeline of N stages, but loading all N stages simultaneously exceeds available memory.

### Solution
Load **one stage at a time**, process the data through it, evict it, and load the next. Only the **intermediate result** (much smaller than the stage itself) persists in memory.

### Implementation Template

```python
import gc
import torch

class StreamingPipeline:
    """Process data through stages without loading all stages at once."""
    
    def __init__(self, stage_paths: list[str]):
        self.stage_paths = stage_paths  # paths to serialized stages
        self.skeleton = self._create_skeleton()  # lightweight structure (Pattern 2)
    
    def forward(self, input_data):
        # Pattern 6: Nuclear reset — guarantee clean memory state
        self._reset_state()
        
        result = input_data
        for i, stage_path in enumerate(self.stage_paths):
            # A) LOAD stage from disk → memory
            stage_weights = self._load_stage(stage_path)
            
            # B) MATERIALIZE — fill the skeleton with real weights
            self._fill_stage(i, stage_weights)
            
            # C) PROCESS — run data through this stage
            result = self.skeleton.stages[i](result)
            
            # D) EVICT — free the stage from memory
            self._evict_stage(i)
            self._clean_memory()
        
        return result
    
    def _reset_state(self):
        """Pattern 6: Destroy and rebuild for guaranteed clean state."""
        if hasattr(self, 'skeleton'):
            del self.skeleton
        self._clean_memory()
        self.skeleton = self._create_skeleton()
    
    def _evict_stage(self, idx):
        """Move stage to virtual device (zero memory)."""
        # For PyTorch: stage.to("meta")
        # For general: del stage; replace with placeholder
        self.skeleton.stages[idx].to("meta")
    
    def _clean_memory(self):
        """Pattern 9: Three-level memory cleanup."""
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### Key Insight
> **Activations (intermediate results) are typically 1000-3000x smaller than stage weights.** This ratio is what makes streaming viable. Always verify this ratio for your specific use case before applying this pattern.

### Size Ratio Checklist
```
Stage weights:    ████████████████████████████████ (e.g., 1.7 GB per layer)
Activations:      █ (e.g., 0.5 MB)
Peak memory:      Stage + Activations + overhead ≈ 1 stage worth
```

---

## Pattern 2: Meta-Device Initialization (Zero-Memory Skeleton)

### Problem
Even creating an object allocates memory for all its components. For large models/structures, this alone causes out-of-memory crashes.

### Solution
Create a **skeleton** (blueprint) that tracks structure but allocates zero bytes for actual data.

### Implementation

```python
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoConfig

def create_skeleton(config_path: str):
    """Create a model skeleton that uses zero GPU memory."""
    config = AutoConfig.from_pretrained(config_path)
    
    with init_empty_weights():
        # All parameters exist as metadata only — no actual tensors allocated
        model = AutoModelForCausalLM.from_config(config)
    
    return model  # Inspectable, iterable, but 0 bytes of real memory

# Generic version (non-PyTorch):
class LazyArray:
    """Placeholder that records shape/dtype but allocates nothing."""
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self._data = None  # No allocation until explicitly filled
    
    def materialize(self, data):
        self._data = data
    
    def evict(self):
        self._data = None  # Free immediately
```

### When To Use
- Model inspection without loading weights
- Building computation graphs before execution
- Memory planning and estimation
- Testing pipeline structure without real data

---

## Pattern 3: One-Time Preprocessing (Shard Splitting)

### Problem
Source data is organized for storage efficiency (large files), not for processing efficiency (random access to individual components).

### Solution
Run a **one-time** reorganization step that splits data into individually-loadable units.

### Implementation

```python
import json
from pathlib import Path
from tqdm import tqdm

def split_into_units(source_path: Path, output_path: Path, 
                     unit_names: list[str], compression=None):
    """One-time: split large files into per-unit files."""
    
    # Idempotency check (Pattern 7)
    if _all_units_exist(output_path, unit_names):
        print(f"Already split at {output_path}")
        return output_path
    
    # Read the source index/manifest
    index = _read_index(source_path)
    
    for unit_name in tqdm(unit_names, desc="Splitting"):
        # Skip if this unit already completed (crash recovery)
        if _unit_exists(output_path, unit_name):
            continue
        
        # Extract this unit's data
        unit_data = _extract_unit(source_path, index, unit_name)
        
        # Optional: compress for faster I/O (Pattern 4)
        if compression:
            unit_data = compress(unit_data, method=compression)
        
        # Save with crash-safe marker (Pattern 7)
        _save_with_marker(unit_data, output_path, unit_name)
        
        # Free immediately
        del unit_data
        clean_memory()
    
    return output_path

def _save_with_marker(data, path, name):
    """Pattern 7: Atomic save with done-marker for crash safety."""
    filepath = path / f"{name}.dat"
    marker = path / f"{name}.dat.done"
    
    # Remove stale marker if exists (from failed previous run)
    marker.unlink(missing_ok=True)
    
    # Write the data
    save_data(data, filepath)
    
    # Only create marker AFTER successful save
    marker.touch()

def _unit_exists(path, name):
    """Both data file AND marker must exist."""
    return (path / f"{name}.dat").exists() and \
           (path / f"{name}.dat.done").exists()
```

---

## Pattern 4: Disk-Bottleneck Compression

### Problem
When I/O (disk read, network transfer) is the bottleneck — not compute — raw data transfer is the limiting factor.

### Solution
**Compress at rest, decompress at use.** Shrink stored data to reduce I/O time, then decompress on-the-fly (on GPU/CPU) where compute is fast and cheap.

### Key Insight
> **This is NOT traditional quantization.** Traditional quantization reduces compute precision. Disk-bottleneck compression reduces **transfer size** — the data is decompressed back to full precision before any computation happens. Zero compute accuracy loss.

### Implementation

```python
import bitsandbytes as bnb
import torch

def compress_for_storage(tensor: torch.Tensor, method='4bit') -> dict:
    """Compress a tensor for disk storage. Decompress before compute."""
    
    if method == '4bit':
        # NF4 quantile quantization — optimal for normally-distributed weights
        quantized, quant_state = bnb.functional.quantize_nf4(
            tensor.cuda(), blocksize=64
        )
        # Original: 67M params × 2 bytes = 134 MB
        # Compressed: 67M params × 0.5 bytes + metadata ≈ 38 MB (3.5x smaller)
        return {
            'data': quantized,         # uint8 (two 4-bit values per byte)
            'quant_state': quant_state  # scale factors, codebook, shape
        }
    
    elif method == '8bit':
        quantized, quant_state = bnb.functional.quantize_blockwise(
            tensor.cuda(), blocksize=2048
        )
        return {'data': quantized, 'quant_state': quant_state}

def decompress_for_compute(compressed: dict) -> torch.Tensor:
    """Decompress back to full precision. Call this right before GPU compute."""
    
    if '4bit' in str(type(compressed.get('quant_state', ''))):
        return bnb.functional.dequantize_nf4(
            compressed['data'].cuda(), 
            compressed['quant_state']
        )
    # Returns full float16 tensor — same precision as original!
    # Decompression takes ~1ms on GPU — negligible vs I/O savings

# Size savings calculator
def estimate_savings(original_size_mb: float, method='4bit') -> dict:
    ratio = 3.5 if method == '4bit' else 2.0
    compressed = original_size_mb / ratio
    ssd_speed = 3500  # MB/s for NVMe SSD
    return {
        'original_mb': original_size_mb,
        'compressed_mb': round(compressed, 1),
        'original_read_time': round(original_size_mb / ssd_speed, 3),
        'compressed_read_time': round(compressed / ssd_speed, 3),
        'speedup': f'{ratio}x'
    }
```

### Decision Matrix: When To Compress

| Bottleneck | Solution | Example |
|---|---|---|
| **Disk → Memory** is slow | ✅ Compress at rest | AirLLM (3.6x speedup) |
| **Compute** is slow | ❌ Don't compress (or use compute quantization) | Training |
| **Network** is slow | ✅ Compress for transfer | Model serving over API |
| **Both CPU and GPU** are idle | ✅ Compress — free decompression | Most inference |

---

## Pattern 5: Prefetching Pipeline

### Problem
Disk I/O and compute happen sequentially — the GPU sits idle while waiting for the next data chunk to load from disk.

### Solution
Use a background thread to **load the next chunk while the current chunk is being processed**.

### Implementation

```python
from concurrent.futures import ThreadPoolExecutor, Future
import torch

def streaming_with_prefetch(stages: list, data, load_fn, process_fn):
    """Overlap disk I/O with compute using a prefetch thread."""
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Kick off loading the FIRST stage immediately
        future: Future = executor.submit(load_fn, stages[0])
        
        for i, stage in enumerate(stages):
            # WAIT for current stage to finish loading
            stage_data = future.result()
            
            # START loading NEXT stage in background (if exists)
            if i + 1 < len(stages):
                future = executor.submit(load_fn, stages[i + 1])
            
            # PROCESS current stage while next one loads
            data = process_fn(data, stage_data)
            
            # FREE current stage
            del stage_data
            clean_memory()
    
    return data

# With pinned memory for faster CPU→GPU transfer:
def load_with_pin(path: str) -> dict:
    """Load data and pin memory for async DMA transfer."""
    data = load_from_disk(path)
    if torch.cuda.is_available():
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].pin_memory()
                # Pinned memory: GPU DMA engine copies directly
                # Without pinning: CPU must mediate every byte
    return data
```

### Visual Model
```
WITHOUT prefetching:         WITH prefetching:
[Load A][Compute A]          [Load A]
        [Load B][Compute B]  [Load B][Compute A]
                [Load C]...  [Load C][Compute B]
                                     [Compute C]
Total: 6 units               Total: 4 units (33% faster)
```

### Important: When NOT to prefetch
- When compression is enabled (decompression may need GPU → conflicts with compute)
- When memory is so tight that holding 2 stages would OOM
- When stages are tiny and load instantly

---

## Pattern 6: Aggressive Memory Management

### Problem
In memory-constrained environments, standard garbage collection isn't enough — freed memory may not actually be returned to the OS or GPU.

### Solution
Implement **three-level cleanup** and optionally use the **destroy-and-rebuild** pattern.

### Implementation

```python
import gc
import ctypes
import torch

def clean_memory():
    """Three-level memory cleanup. Call after every stage eviction."""
    
    # Level 1: Python garbage collection
    gc.collect()
    
    # Level 2: Force C allocator to return pages to OS
    # (Python gc frees objects, but libc may keep the pages in its free-list)
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass  # Not on Linux? Skip
    
    # Level 3: Release unused CUDA cached memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class ResetableProcessor:
    """Pattern: Destroy and rebuild for guaranteed clean state."""
    
    def process(self, input_data):
        # NUCLEAR RESET: delete everything and rebuild
        if hasattr(self, '_internal_state'):
            del self._internal_state
        clean_memory()
        self._internal_state = self._create_fresh_state()
        
        # Now process with guaranteed clean state
        return self._execute(input_data)
```

### Memory Cleanup Checklist
```
After evicting each stage:
  ✅ gc.collect()
  ✅ malloc_trim(0) (Linux)
  ✅ torch.cuda.empty_cache() (if GPU)
  ✅ del reference (remove Python reference)
  ✅ stage.to("meta") (if PyTorch, move to virtual device)

Before each new inference call:
  ✅ Destroy the entire state object
  ✅ Full clean_memory()
  ✅ Rebuild from scratch
```

---

## Pattern 7: Crash-Safe Operations with Marker Files

### Problem
Long-running operations (splitting, downloading, converting) may crash mid-way. On restart, you need to know what completed and what needs to be redone.

### Solution
Use **`.done` marker files** as completion sentinels.

### Implementation

```python
from pathlib import Path

def safe_process_unit(data, output_path: Path, unit_name: str) -> bool:
    """Process a unit with crash-safe markers."""
    
    data_path = output_path / f"{unit_name}.dat"
    done_path = output_path / f"{unit_name}.dat.done"
    
    # Already completed? Skip!
    if data_path.exists() and done_path.exists():
        return True  # Idempotent — safe to call multiple times
    
    # Remove stale files from failed previous attempt
    data_path.unlink(missing_ok=True)
    done_path.unlink(missing_ok=True)
    
    # Do the work
    save_data(data, data_path)
    
    # Mark complete ONLY after successful save
    done_path.touch()
    return True

def check_all_complete(output_path: Path, unit_names: list[str]) -> bool:
    """Check if all units were successfully processed."""
    return all(
        (output_path / f"{name}.dat").exists() and
        (output_path / f"{name}.dat.done").exists()
        for name in unit_names
    )
```

### Rules
1. **Create marker AFTER the data file** — never before
2. **Check marker AND data file** — both must exist
3. **Delete stale markers** before retry — prevents false positives
4. **Marker files are empty** — they're just sentinels, content doesn't matter

---

## Pattern 8: Strategy Pattern for Variants

### Problem
You need to support multiple variants (model architectures, data formats, hardware backends) that share 90%+ of their logic but differ in naming conventions or small behaviors.

### Solution
Define the **algorithm skeleton** in a base class. Subclasses override only the **vocabulary** (names, configs) and **specific hooks**.

### Implementation

```python
from abc import ABC, abstractmethod

class BaseStreamingProcessor(ABC):
    """Shared algorithm — subclasses customize vocabulary only."""
    
    def __init__(self, source_path):
        self.source_path = source_path
        self.component_names = self._get_component_names()
    
    @abstractmethod
    def _get_component_names(self) -> dict:
        """Override this to define component naming for your variant."""
        pass
    
    def process(self, input_data):
        """Shared algorithm — identical across all variants."""
        for name in self.component_names.values():
            data = self.load_component(name)
            input_data = self.run_component(data, input_data)
            self.evict_component()
        return input_data

# Variant A
class LlamaProcessor(BaseStreamingProcessor):
    def _get_component_names(self):
        return {
            'embed': 'model.embed_tokens',
            'layers': 'model.layers',
            'norm': 'model.norm',
            'head': 'lm_head'
        }

# Variant B — same algorithm, different names!
class ChatGLMProcessor(BaseStreamingProcessor):
    def _get_component_names(self):
        return {
            'embed': 'transformer.embedding.word_embeddings',
            'layers': 'transformer.encoder.layers',
            'norm': 'transformer.encoder.final_layernorm',
            'head': 'transformer.output_layer'
        }
```

---

## Pattern 9: Factory + Platform Abstraction

### Problem
Code must run on different platforms (CUDA GPUs, Apple Silicon, CPU-only) with completely different backends.

### Solution
Use **conditional imports at the module level** and a **factory method** to auto-detect the right implementation.

### Implementation

```python
# __init__.py — swap entire implementation at import time
from sys import platform

if platform == "darwin":
    from .backend_mlx import MLXEngine as Engine
else:
    from .backend_cuda import CUDAEngine as Engine

# auto_factory.py — auto-detect variant from config
import importlib

class AutoProcessor:
    REGISTRY = {
        "TypeA": ("module_a", "ProcessorA"),
        "TypeB": ("module_b", "ProcessorB"),
    }
    
    @classmethod
    def from_config(cls, config_path, **kwargs):
        config = load_config(config_path)
        variant = detect_variant(config)
        
        module_name, class_name = cls.REGISTRY.get(
            variant, ("module_default", "DefaultProcessor")
        )
        
        module = importlib.import_module(f".{module_name}", package=__package__)
        processor_class = getattr(module, class_name)
        return processor_class(config_path, **kwargs)
```

---

## Pattern 10: Graceful Degradation (Try-Chains)

### Problem
The optimal code path may not work on all systems (version mismatches, missing libraries, hardware differences).

### Solution
Try strategies in order of preference, catching failures and falling back gracefully.

### Implementation

```python
def initialize_with_best_backend(config):
    """Try best option first, fall back gracefully."""
    
    strategies = [
        ("Flash Attention", lambda: init_with_flash_attn(config)),
        ("SDPA", lambda: init_with_sdpa(config)),
        ("Vanilla", lambda: init_vanilla(config)),
    ]
    
    for name, strategy in strategies:
        try:
            result = strategy()
            print(f"✅ Using {name}")
            return result
        except (ImportError, ValueError, RuntimeError) as e:
            print(f"⚠️ {name} unavailable: {e}")
            continue
    
    raise RuntimeError("All initialization strategies failed!")
```

---

## Quick Decision Guide

```
Is your data too big for memory?
├── YES: Is it a sequential pipeline?
│   ├── YES → Pattern 1 (Layer-Wise Streaming)
│   │         + Pattern 2 (Meta Skeleton)
│   │         + Pattern 6 (Memory Cleanup)
│   └── NO → Consider chunked processing or MapReduce
│
├── Is I/O the bottleneck?
│   ├── YES → Pattern 4 (Disk-Bottleneck Compression)
│   │         + Pattern 5 (Prefetching)
│   └── NO → Focus on compute optimization instead
│
├── Do you need crash safety?
│   └── YES → Pattern 7 (Marker Files)
│             + Pattern 3 (Idempotent Preprocessing)
│
├── Multiple variants/platforms?
│   └── YES → Pattern 8 (Strategy) + Pattern 9 (Factory)
│
└── Version compatibility issues?
    └── YES → Pattern 10 (Graceful Degradation)
```

---

## Reference

These patterns are extracted from **AirLLM** (github.com/lyogavin/airllm):
- 14,500+ GitHub stars
- Runs 70B LLM on 4GB GPU, 405B on 8GB VRAM
- Apache-2.0 licensed

Key source files:
- `airllm_base.py` (643 lines) — Patterns 1, 2, 5, 6
- `utils.py` (404 lines) — Patterns 3, 4, 7
- `auto_model.py` (56 lines) — Pattern 9
- Model adapters — Pattern 8
- `persist/` package — Pattern 7, 9
