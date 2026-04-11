"""
Logical Rooms — CLI Entry Point
=================================
Command-line interface for processing, training, evaluation, and export.

Usage:
    python main.py process  "Your text here" --domain security
    python main.py batch    input.txt --domain finance --output atoms.jsonl
    python main.py train    feedback.jsonl --epochs 5
    python main.py evaluate dataset.jsonl
    python main.py export   atoms.jsonl --format jsonl
    python main.py serve    --port 5000
    python main.py info
"""

import os
import sys
import json
import argparse

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_process(args):
    """Process a single text into atoms."""
    from core.pipeline import LogicalRoomsPipeline

    pipe = LogicalRoomsPipeline(use_llm=not args.no_llm, verbose=True)

    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("Error: Provide --text or --file")
        return

    atom = pipe.process(text, domain=args.domain)
    print(f"\n{'='*60}")
    print(f"  ATOM RESULT")
    print(f"{'='*60}")
    print(f"  ID:       {atom.id[:12]}")
    print(f"  Domain:   {atom.domain}")
    print(f"  Category: {atom.ontology_category.value}")
    print(f"  Tension:  {atom.tension:.3f}")
    print(f"  Tokens:   {atom.original_tokens} → {atom.compressed_tokens} ({atom.compression_ratio:.1%} saved)")
    print(f"\n  Axes:")
    for axis, value in atom.axes.items():
        bar = "█" * int(abs(value) * 20)
        sign = "+" if value >= 0 else "-"
        print(f"    {axis:12s} {sign}{abs(value):.3f}  {bar}")
    print(f"\n  Compressed: {atom.to_compressed_repr()}")
    print(f"{'='*60}\n")


def cmd_batch(args):
    """Batch process texts from a file."""
    from core.pipeline import LogicalRoomsPipeline

    pipe = LogicalRoomsPipeline(use_llm=not args.no_llm, verbose=True)

    # Read input
    with open(args.input, "r", encoding="utf-8") as f:
        if args.input.endswith(".jsonl"):
            texts = [json.loads(line)["text"] for line in f if line.strip()]
        else:
            texts = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(texts)} texts...")
    atoms = pipe.process_batch(texts, domain=args.domain)

    # Compress
    compressed = pipe.compress_context(texts, domain=args.domain)
    print(f"\nCompression: {compressed.original_tokens} → {compressed.compressed_tokens} tokens "
          f"({compressed.compression_ratio:.1%} saved)")
    print(f"Rooms: {compressed.rooms_used} | Surfaces: {compressed.surfaces_used} | Atoms: {compressed.atoms_created}")
    print(f"Time: {compressed.processing_time_ms:.1f}ms")

    # Export
    if args.output:
        data = pipe.export_atoms(atoms)
        with open(args.output, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\nExported {len(data)} atoms → {args.output}")

    stats = pipe.get_stats()
    print(f"\nPipeline stats: {json.dumps(stats, indent=2)}")


def cmd_train(args):
    """Fine-tune axis evaluation from feedback data."""
    from core.finetuning import AxisFineTuner
    from core.dataset import LogicalRoomsDataset

    tuner = AxisFineTuner(learning_rate=args.lr)

    # Load calibration if exists
    if os.path.exists("axis_calibration.json"):
        tuner.load_calibration()
        print("Loaded existing calibration.")

    # Load feedback dataset
    ds = LogicalRoomsDataset()
    count = ds.load(args.dataset)
    print(f"Loaded {count} feedback samples from {args.dataset}")

    # Convert to feedback records
    for sample in ds.samples:
        tuner.record_feedback(
            atom_id="training",
            text=sample.text,
            predicted_axes=sample.axes,  # Use as both predicted and target for bootstrapping
            corrected_axes=sample.axes,
            domain=sample.domain,
        )

    # Training loop
    for epoch in range(args.epochs):
        metrics = tuner.train_step()
        print(f"  Epoch {epoch + 1}/{args.epochs}: "
              f"error={metrics.mean_error_before:.4f} → {metrics.mean_error_after:.4f} "
              f"({metrics.improvement_pct:+.1f}%)")

    # Save calibration
    tuner.save_calibration()
    print(f"\nCalibration saved to axis_calibration.json")
    print(f"Summary: {json.dumps(tuner.get_summary(), indent=2, default=str)}")


def cmd_evaluate(args):
    """Evaluate pipeline on a dataset."""
    from core.pipeline import LogicalRoomsPipeline
    from core.dataset import LogicalRoomsDataset

    pipe = LogicalRoomsPipeline(use_llm=not args.no_llm, verbose=False)

    ds = LogicalRoomsDataset()
    count = ds.load(args.dataset)
    stats = ds.get_statistics()

    print(f"Dataset: {count} samples across {len(stats.domains)} domains")
    print(f"Domains: {stats.domains}")

    # Process and measure compression
    texts = [s.text for s in ds.samples]
    compressed = pipe.compress_context(texts, domain="evaluation")

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Texts processed:  {len(texts)}")
    print(f"  Original tokens:  {compressed.original_tokens}")
    print(f"  Compressed tokens: {compressed.compressed_tokens}")
    print(f"  Compression ratio: {compressed.compression_ratio:.1%}")
    print(f"  Rooms created:    {compressed.rooms_used}")
    print(f"  Processing time:  {compressed.processing_time_ms:.1f}ms")
    print(f"{'='*60}\n")


def cmd_export(args):
    """Export processed atoms or datasets."""
    from core.pipeline import LogicalRoomsPipeline

    pipe = LogicalRoomsPipeline(use_llm=not args.no_llm, verbose=True)

    with open(args.input, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    atoms = pipe.process_batch(texts, domain=args.domain)
    data = pipe.export_atoms(atoms)

    with open(args.output, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Exported {len(data)} atoms → {args.output}")


def cmd_serve(args):
    """Start the chat server."""
    print(f"Starting Logical Rooms server on port {args.port}...")
    from chat_server import app
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


def cmd_info(args):
    """Show system information and configuration."""
    print(f"\n{'='*60}")
    print(f"  LOGICAL ROOMS — System Information")
    print(f"{'='*60}")

    # Check dependencies
    deps = {
        "sentence-transformers": False,
        "llama-cpp-python": False,
        "faiss-cpu": False,
        "numpy": False,
        "networkx": False,
    }

    try:
        import sentence_transformers
        deps["sentence-transformers"] = True
    except ImportError:
        pass
    try:
        import llama_cpp
        deps["llama-cpp-python"] = True
    except ImportError:
        pass
    try:
        import faiss
        deps["faiss-cpu"] = True
    except ImportError:
        pass
    try:
        import numpy
        deps["numpy"] = True
    except ImportError:
        pass
    try:
        import networkx
        deps["networkx"] = True
    except ImportError:
        pass

    print("\n  Dependencies:")
    for dep, ok in deps.items():
        status = "✅" if ok else "❌"
        print(f"    {status} {dep}")

    # Check model
    model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    model_exists = os.path.exists(model_path)
    print(f"\n  Model:")
    print(f"    {'✅' if model_exists else '❌'} TinyLlama GGUF ({'found' if model_exists else 'not found'})")
    if model_exists:
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"       Size: {size_mb:.0f} MB")

    # Check calibration
    cal_exists = os.path.exists("axis_calibration.json")
    print(f"\n  Calibration:")
    print(f"    {'✅' if cal_exists else '⚪'} axis_calibration.json ({'found' if cal_exists else 'none yet'})")

    print(f"\n{'='*60}\n")
    print("  Quickstart:")
    print("    python main.py process --text 'Your text here'")
    print("    python main.py batch input.txt --output atoms.jsonl")
    print("    python main.py train feedback.jsonl --epochs 5")
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="logical-rooms",
        description="Logical Rooms — Semantic Compression & Fine-Tuning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py process --text "Critical security vulnerability detected"
  python main.py batch texts.txt --domain security --output atoms.jsonl
  python main.py train feedback.jsonl --epochs 10
  python main.py evaluate dataset.jsonl
  python main.py info
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── process ─────────────────────────────────────────────────────────
    p_process = subparsers.add_parser("process", help="Process a single text")
    p_process.add_argument("--text", "-t", type=str, help="Text to process")
    p_process.add_argument("--file", "-f", type=str, help="File containing text")
    p_process.add_argument("--domain", "-d", type=str, default="general", help="Domain tag")
    p_process.add_argument("--no-llm", action="store_true", help="Use heuristic mode only")
    p_process.set_defaults(func=cmd_process)

    # ── batch ───────────────────────────────────────────────────────────
    p_batch = subparsers.add_parser("batch", help="Batch process texts from file")
    p_batch.add_argument("input", type=str, help="Input file (txt or jsonl)")
    p_batch.add_argument("--domain", "-d", type=str, default="general", help="Domain tag")
    p_batch.add_argument("--output", "-o", type=str, help="Output JSONL file")
    p_batch.add_argument("--no-llm", action="store_true", help="Use heuristic mode only")
    p_batch.set_defaults(func=cmd_batch)

    # ── train ───────────────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Fine-tune axis evaluation")
    p_train.add_argument("dataset", type=str, help="Feedback dataset (JSONL/CSV)")
    p_train.add_argument("--epochs", "-e", type=int, default=5, help="Training epochs")
    p_train.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    p_train.set_defaults(func=cmd_train)

    # ── evaluate ────────────────────────────────────────────────────────
    p_eval = subparsers.add_parser("evaluate", help="Evaluate on a dataset")
    p_eval.add_argument("dataset", type=str, help="Evaluation dataset (JSONL/CSV)")
    p_eval.add_argument("--no-llm", action="store_true", help="Use heuristic mode only")
    p_eval.set_defaults(func=cmd_evaluate)

    # ── export ──────────────────────────────────────────────────────────
    p_export = subparsers.add_parser("export", help="Export processed atoms")
    p_export.add_argument("input", type=str, help="Input text file")
    p_export.add_argument("--output", "-o", type=str, required=True, help="Output JSONL file")
    p_export.add_argument("--domain", "-d", type=str, default="general", help="Domain tag")
    p_export.add_argument("--no-llm", action="store_true", help="Use heuristic mode only")
    p_export.set_defaults(func=cmd_export)

    # ── serve ───────────────────────────────────────────────────────────
    p_serve = subparsers.add_parser("serve", help="Start API server")
    p_serve.add_argument("--port", "-p", type=int, default=5000, help="Port")
    p_serve.add_argument("--debug", action="store_true", help="Debug mode")
    p_serve.set_defaults(func=cmd_serve)

    # ── info ────────────────────────────────────────────────────────────
    p_info = subparsers.add_parser("info", help="Show system info")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
