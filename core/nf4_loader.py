"""
NF4 (Normal Float 4-bit) Model Loader
=======================================
Loads TinyLlama or any HuggingFace model with NF4 quantization
(QLoRA-style) using bitsandbytes. This reduces memory by ~75%
compared to float16 while maintaining near-lossless quality.

NF4 (Normal Float 4-bit):
  - Information-theoretically optimal 4-bit data type for normally
    distributed weights (Dettmers et al., 2023)
  - Double quantization: quantizes the quantization constants themselves
  - Typical VRAM: 1.1B model in ~0.7 GB instead of ~2.2 GB (fp16)

Usage:
    from core.nf4_loader import load_model_nf4, load_model_gguf_q4
    model, tokenizer = load_model_nf4("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
"""

import os
from typing import Optional, Tuple, Any

# ─── GGUF Q4 Loader (existing path, optimized) ──────────────────────────────

def load_model_gguf_q4(
    model_path: str = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx: int = 512,
    n_threads: Optional[int] = None,
) -> Any:
    """
    Load a GGUF model with Q4_K_M quantization via llama-cpp-python.

    Q4_K_M is a 4-bit quantization scheme using k-quants:
    - Mixed precision: important layers kept at higher precision
    - Typical size: ~660 MB for 1.1B params
    - Quality: near-fp16 on most benchmarks

    Args:
        model_path: Path to .gguf file
        n_ctx: Context window size
        n_threads: CPU threads (None = auto)

    Returns:
        Llama model instance
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python not installed. Run: pip install llama-cpp-python"
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    kwargs = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": 0,  # CPU-only for maximum compatibility
        "verbose": False,
        "use_mmap": True,   # Memory-mapped I/O for faster loading
        "use_mlock": False,  # Don't lock in RAM (allow swap)
    }
    if n_threads:
        kwargs["n_threads"] = n_threads

    return Llama(**kwargs)


# ─── NF4 Loader (HuggingFace + bitsandbytes) ────────────────────────────────

def load_model_nf4(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    compute_dtype: str = "bfloat16",
    double_quant: bool = True,
) -> Tuple[Any, Any]:
    """
    Load a HuggingFace model with NF4 quantization via bitsandbytes.

    NF4 (Normal Float 4-bit):
    - Optimal 4-bit quantization for normally distributed neural network weights
    - 2x memory reduction vs 8-bit, 4x vs fp16
    - With double quantization: quantizes the quantization constants too
    - Quality loss: <0.5% on most benchmarks vs fp16

    Requirements:
        pip install bitsandbytes transformers torch accelerate

    Args:
        model_name: HuggingFace model hub name
        compute_dtype: Compute dtype for 4-bit base model (bfloat16 recommended)
        double_quant: Enable double quantization (reduces memory by another ~0.4 bits/param)

    Returns:
        (model, tokenizer) tuple
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        raise ImportError(
            "Required packages not installed. Run:\n"
            "pip install bitsandbytes transformers torch accelerate"
        )

    # Resolve compute dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    cdtype = dtype_map.get(compute_dtype, torch.bfloat16)

    # NF4 quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,               # Enable 4-bit loading
        bnb_4bit_quant_type="nf4",       # NF4 data type (vs. fp4)
        bnb_4bit_compute_dtype=cdtype,   # Compute precision
        bnb_4bit_use_double_quant=double_quant,  # Double quantization
    )

    print(f"Loading {model_name} with NF4 quantization...")
    print(f"  Quant type: NF4 (Normal Float 4)")
    print(f"  Double quant: {double_quant}")
    print(f"  Compute dtype: {compute_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Print memory stats
    param_count = sum(p.numel() for p in model.parameters())
    mem_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    print(f"  Parameters: {param_count / 1e6:.1f}M")
    print(f"  Memory: {mem_bytes / 1024 / 1024:.1f} MB (NF4)")
    print(f"  vs fp16: ~{param_count * 2 / 1024 / 1024:.1f} MB")
    print(f"  Reduction: {(1 - mem_bytes / (param_count * 2)) * 100:.1f}%")

    return model, tokenizer


# ─── NF4-backed Evaluator ────────────────────────────────────────────────────

class NF4Evaluator:
    """
    7-axis evaluator using NF4-quantized TinyLlama for scoring.

    Memory footprint: ~700 MB (vs ~2.2 GB for fp16 TinyLlama)
    Quality: near-identical to fp16 on axis scoring tasks.
    """

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model = None
        self.tokenizer = None
        self.is_ready = False

        try:
            self.model, self.tokenizer = load_model_nf4(model_name)
            self.is_ready = True
            print("NF4Evaluator: Ready.")
        except Exception as e:
            print(f"NF4Evaluator: Failed to load ({e}). Use heuristic mode.")

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate text with the NF4 model."""
        if not self.is_ready:
            return ""

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─── Embedding Quantization ─────────────────────────────────────────────────

def quantize_embeddings_int8(embeddings):
    """
    Quantize float32 embeddings to int8 for 4x memory reduction.

    Process:
    1. Compute per-vector scale factor
    2. Quantize to int8 range [-127, 127]
    3. Store scale separately for dequantization

    Memory: 384-dim × float32 = 1536 bytes → 384-dim × int8 = 384 bytes + 4 bytes scale = 388 bytes
    Reduction: 74.7%

    Used for storage in AtomStore — dequantized for similarity computations.
    """
    import numpy as np

    embeddings = np.array(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    # Per-vector max absolute value
    scales = np.max(np.abs(embeddings), axis=1, keepdims=True)
    scales = np.where(scales == 0, 1.0, scales)  # Avoid division by zero

    # Quantize
    quantized = np.round(embeddings / scales * 127.0).astype(np.int8)

    return quantized, scales.flatten().astype(np.float32)


def dequantize_embeddings_int8(quantized, scales):
    """Restore float32 embeddings from int8 quantized form."""
    import numpy as np
    return (quantized.astype(np.float32) / 127.0) * scales.reshape(-1, 1)


# ─── Utility ────────────────────────────────────────────────────────────────

def get_model_info() -> dict:
    """Get information about available quantization backends."""
    info = {
        "gguf_q4": False,
        "nf4_bitsandbytes": False,
        "cuda_available": False,
    }

    try:
        from llama_cpp import Llama
        info["gguf_q4"] = True
    except ImportError:
        pass

    try:
        import bitsandbytes
        info["nf4_bitsandbytes"] = True
    except ImportError:
        pass

    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass

    return info
