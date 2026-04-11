# AirLLM Patterns — Quick Reference Card

## 10 Patterns at a Glance

| # | Pattern | One-Liner | Key Code |
|---|---------|-----------|----------|
| 1 | **Layer-Wise Streaming** | Load 1 stage, process, evict, repeat | `for stage in stages: load→run→evict` |
| 2 | **Meta Skeleton** | Create structure with zero memory | `with init_empty_weights(): model = ...` |
| 3 | **One-Time Split** | Reorganize data for random access | `split_into_units() + idempotency check` |
| 4 | **Disk Compression** | Shrink files, decompress on-the-fly | `quantize_nf4() → save → load → dequantize_nf4()` |
| 5 | **Prefetching** | Overlap I/O and compute | `ThreadPoolExecutor + future.submit()` |
| 6 | **Memory Cleanup** | 3-level: gc + malloc_trim + cuda | `gc.collect(); malloc_trim(0); empty_cache()` |
| 7 | **Marker Files** | Crash-safe with `.done` sentinels | `save(data); Path(".done").touch()` |
| 8 | **Strategy Pattern** | Shared algo, variant-specific naming | `base_class.set_names() → subclass overrides` |
| 9 | **Factory + Platform** | Auto-detect variant and hardware | `importlib.import_module() + if platform==` |
| 10 | **Try-Chain** | Best first, graceful fallback | `try: fast() except: try: medium() except: slow()` |

## Common Combinations

```
Memory-constrained inference:  1 + 2 + 6
Fast inference:                1 + 4 + 5
Crash-safe preprocessing:     3 + 7
Multi-architecture support:   8 + 9
Maximum compatibility:         9 + 10
```

## Key Size Ratios (Llama-70B reference)

```
Full model:           140 GB (too big for any single GPU)
One layer (float16):  1.7 GB (fits in 4GB GPU with overhead)
One layer (4-bit):    0.48 GB (3.5x smaller → 3.5x faster I/O)
Activations:          0.5 MB (3000x smaller than one layer!)
```

## Critical Don'ts

- ❌ Don't compress activations (they're already tiny)
- ❌ Don't prefetch when compression is enabled (GPU conflict)
- ❌ Don't trust gc.collect() alone (use all 3 levels)
- ❌ Don't skip the .done marker (half-written files = corruption)
- ❌ Don't try to clean up state — destroy and rebuild instead
