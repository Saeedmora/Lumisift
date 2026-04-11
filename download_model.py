"""
Download TinyLlama GGUF Model
=============================
Downloads TinyLlama-1.1B-Chat model in GGUF format for local inference.
"""

import os
import sys

def download_model():
    """Download TinyLlama GGUF model from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface-hub...")
        os.system(sys.executable + " -m pip install huggingface-hub")
        from huggingface_hub import hf_hub_download
    
    # Model configuration
    repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    target_path = os.path.join(models_dir, filename)
    
    if os.path.exists(target_path):
        print("Model already exists at:", target_path)
        return target_path
    
    print("=" * 60)
    print("Downloading TinyLlama GGUF Model")
    print("=" * 60)
    print("Repository:", repo_id)
    print("File:", filename)
    print("Size: ~600-800 MB (4-bit quantization)")
    print("=" * 60)
    print()
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        print()
        print("Download complete!")
        print("Model saved to:", downloaded_path)
        return downloaded_path
    except Exception as e:
        print("Download failed:", e)
        print()
        print("Manual download instructions:")
        print("1. Go to: https://huggingface.co/" + repo_id)
        print("2. Download:", filename)
        print("3. Place in:", models_dir)
        return None


if __name__ == "__main__":
    download_model()
