"""
Logical Rooms — Scientific AI Fine-Tuning Server
=================================================
Web interface for processing scientific articles through
the 7-axis semantic evaluation pipeline and generating
structured fine-tuning data.

Features:
  - Graceful degradation (heuristic -> LLM fallback)
  - Bounded session state (prevents OOM on long sessions)
  - Atomic file writes for crash-safe calibration
  - Memory cleanup after batch processing
"""

import gc
import os
import sys
import time
import json
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from core.atom import Atom, AXIS_NAMES, OntologyCategory, _count_tokens
from core.models import LogicalRoom
from core.embeddings import EmbeddingService
from core.axes_evaluator import SevenAxesEvaluator
from core.pipeline import LogicalRoomsPipeline
from core.finetuning import AxisFineTuner, AxisCalibration
from core.dataset import LogicalRoomsDataset
from core.projection_engine import ContextProjectionEngine
from core.self_optimization import SelfOptimizer
from core.atom_store import AtomStore

import numpy as np

# ─── App Setup ──────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"))
CORS(app)

# ─── Global State (bounded to prevent OOM) ─────────────────────────────────

MAX_SESSION_ATOMS = 500
MAX_SESSION_HISTORY = 50

pipeline = None
tuner = None
dataset = None
atom_store = None
session_atoms = []
session_history = []


def _clean_memory():
    """Release memory after heavy processing."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass  # Not on Linux


def initialize():
    """Initialize pipeline with graceful degradation (try best, fall back)."""
    global pipeline, tuner, dataset, atom_store

    strategies = [
        ("Heuristic Pipeline (fast, reliable)", lambda: LogicalRoomsPipeline(use_llm=False, verbose=True)),
        ("Full Pipeline (LLM + Embeddings)", lambda: LogicalRoomsPipeline(use_llm=True, verbose=True)),
    ]

    for name, strategy in strategies:
        try:
            pipeline = strategy()
            print(f"  [OK] {name}")
            break
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            continue

    if pipeline is None:
        raise RuntimeError("All pipeline initialization strategies failed!")

    # Initialize fine-tuning engine
    tuner = AxisFineTuner(learning_rate=0.05)
    if os.path.exists("axis_calibration.json"):
        tuner.load_calibration()
        print("  [OK] Loaded existing calibration")

    # Initialize dataset
    dataset = LogicalRoomsDataset(name="scientific_articles")
    print("  [OK] Fine-tuning engine ready")
    print("  [OK] Dataset manager ready")

    # Initialize persistent atom store
    atom_store = AtomStore()
    projects = atom_store.list_projects()
    if not projects:
        atom_store.create_project("default", domain="general")
    atom_store.set_active(projects[0]["name"] if projects else "default")
    print(f"  [OK] Atom store: project '{atom_store.active_project}' ({atom_store.get_atom_count()} atoms)")


# ─── Static Files ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE_DIR, "static"), "index.html")

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)


# ─── API: Article Processing ───────────────────────────────────────────────

@app.route("/api/process", methods=["POST"])
def process_article():
    """
    Process a pasted scientific article through the full pipeline.

    Agent Loop Pattern:
      1. Receive text -> 2. Evaluate axes -> 3. Create atoms -> 4. Build surfaces
      -> 5. Build compressed repr -> 6. Return structured result
    """
    start = time.time()

    try:
        data = request.json
        text = data.get("text", "").strip()
        domain = data.get("domain", "general")
        title = data.get("title", "Untitled Article")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Split into paragraphs for granular atom creation
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 20]
        if not paragraphs:
            paragraphs = [text]

        # Step 1: Process all paragraphs into atoms
        atoms = pipeline.process_batch(paragraphs, domain=domain)

        # Step 2: Build surfaces from atoms
        surfaces = pipeline.build_surfaces(atoms, strategy="similarity")

        # Step 3: Build compressed representations manually (avoids re-processing)
        compressed_parts = []
        for atom in atoms:
            compressed_parts.append(atom.to_compressed_repr())
        compressed_text = " ".join(compressed_parts)

        original_tokens = sum(a.original_tokens for a in atoms)
        compressed_tokens = _count_tokens(compressed_text)
        compression_ratio = 1.0 - (compressed_tokens / max(1, original_tokens))

        # Store in session (bounded)
        session_atoms.extend(atoms)
        if len(session_atoms) > MAX_SESSION_ATOMS:
            session_atoms[:] = session_atoms[-MAX_SESSION_ATOMS:]

        # Persist atoms to local database
        if atom_store:
            atom_store.save_atoms(atoms, source=f"ui:{title[:40]}")

        # Build response
        atoms_data = []
        for atom in atoms:
            atoms_data.append({
                "id": atom.id[:12],
                "text": atom.text[:200],
                "axes": {k: round(v, 3) for k, v in atom.axes.items()},
                "tension": round(atom.tension, 3),
                "category": atom.ontology_category.value if atom.ontology_category else "unknown",
                "domain": atom.domain,
                "confidence": round(atom.confidence, 3),
                "original_tokens": atom.original_tokens,
                "compressed_tokens": atom.compressed_tokens,
                "compression_ratio": round(atom.compression_ratio, 3) if atom.compression_ratio > 0 else 0,
            })

        # Axes distribution summary
        axes_dist = {}
        for axis in AXIS_NAMES:
            values = [a.axes.get(axis, 0) for a in atoms]
            axes_dist[axis] = {
                "mean": round(float(np.mean(values)), 3),
                "std": round(float(np.std(values)), 3),
                "min": round(float(np.min(values)), 3),
                "max": round(float(np.max(values)), 3),
            }

        elapsed = (time.time() - start) * 1000

        # Add to session history
        entry = {
            "id": str(uuid.uuid4())[:8],
            "title": title,
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
            "atoms_count": len(atoms),
            "compression_ratio": round(compression_ratio, 3),
            "processing_time_ms": round(elapsed, 1),
        }
        session_history.append(entry)
        if len(session_history) > MAX_SESSION_HISTORY:
            session_history[:] = session_history[-MAX_SESSION_HISTORY:]

        # Add to dataset for fine-tuning
        for atom in atoms:
            dataset.add_sample(
                text=atom.text,
                axes=atom.axes,
                domain=domain,
                metadata={"title": title, "atom_id": atom.id},
            )

        result = jsonify({
            "success": True,
            "title": title,
            "domain": domain,
            "atoms": atoms_data,
            "surfaces_count": len(surfaces),
            "rooms_count": 0,
            "compression": {
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "ratio": round(compression_ratio, 3),
                "compressed_text": compressed_text,
            },
            "axes_distribution": axes_dist,
            "processing_time_ms": round(elapsed, 1),
            "pipeline_stats": pipeline.get_stats(),
            "dataset_size": len(dataset),
        })

        # Release intermediate objects
        _clean_memory()
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        _clean_memory()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Record corrected axes for fine-tuning."""
    data = request.json
    atom_id = data.get("atom_id", "")
    text = data.get("text", "")
    predicted = data.get("predicted_axes", {})
    corrected = data.get("corrected_axes", {})
    domain = data.get("domain", "general")

    if not corrected:
        return jsonify({"error": "No corrected axes provided"}), 400

    record = tuner.record_feedback(
        atom_id=atom_id,
        text=text,
        predicted_axes=predicted,
        corrected_axes=corrected,
        domain=domain,
    )

    return jsonify({
        "success": True,
        "feedback_count": len(tuner.feedback),
        "mean_error": round(record.mean_error, 4),
    })


@app.route("/api/train", methods=["POST"])
def train_step():
    """Run a fine-tuning training step."""
    data = request.json or {}
    epochs = data.get("epochs", 1)

    results = []
    for i in range(epochs):
        metrics = tuner.train_step()
        results.append({
            "step": metrics.step,
            "samples": metrics.samples_used,
            "error_before": metrics.mean_error_before,
            "error_after": metrics.mean_error_after,
            "improvement_pct": metrics.improvement_pct,
            "per_axis": metrics.per_axis_error,
        })

    tuner.save_calibration("axis_calibration.json")

    return jsonify({
        "success": True,
        "epochs_completed": epochs,
        "results": results,
        "calibration": tuner.calibration.to_dict(),
        "summary": tuner.get_summary(),
    })


@app.route("/api/export", methods=["GET"])
def export_dataset():
    """Export current dataset as JSONL."""
    path = "exported_dataset.jsonl"
    count = dataset.export(path)
    return jsonify({
        "success": True,
        "exported": count,
        "path": path,
        "statistics": {
            "total_samples": dataset.get_statistics().total_samples,
            "domains": dataset.get_statistics().domains,
        },
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get current pipeline and training stats."""
    return jsonify({
        "pipeline": pipeline.get_stats(),
        "dataset": {
            "total_samples": len(dataset),
            "statistics": {
                "total_samples": dataset.get_statistics().total_samples,
                "domains": dataset.get_statistics().domains,
                "avg_text_length": dataset.get_statistics().avg_text_length,
            },
        },
        "finetuning": tuner.get_summary(),
        "session": {
            "total_atoms": len(session_atoms),
            "history": session_history[-10:],
        },
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    """Get processing history."""
    return jsonify({"history": session_history})


@app.route("/api/select", methods=["POST"])
def select_context():
    """Axes-driven context selection (the proven approach)."""
    data = request.json
    texts = data.get("texts", [])
    query = data.get("query", "")
    domain = data.get("domain", "general")
    top_k = data.get("top_k", 5)

    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    result = pipeline.select_context(texts, query=query, domain=domain, top_k=top_k)

    return jsonify({
        "success": True,
        "selected_text": result.compressed_text,
        "original_tokens": result.original_tokens,
        "selected_tokens": result.compressed_tokens,
        "reduction_pct": round(result.compression_ratio * 100, 1),
        "atoms_analyzed": result.atoms_created,
        "processing_time_ms": round(result.processing_time_ms, 1),
        "axes_summary": result.axes_summary,
    })


@app.route("/api/system", methods=["GET"])
def get_system_info():
    """Get system capabilities for UI status display."""
    has_tiktoken = False
    has_embeddings = False
    evaluator_mode = "unknown"

    try:
        import tiktoken
        has_tiktoken = True
    except ImportError:
        pass

    try:
        has_embeddings = pipeline.embedder is not None
    except Exception:
        pass

    try:
        evaluator_mode = "llm" if pipeline.evaluator.is_ready else "heuristic"
    except Exception:
        evaluator_mode = "heuristic"

    return jsonify({
        "tokenizer": "cl100k_base (BPE)" if has_tiktoken else "word_count (fallback)",
        "embeddings": "all-MiniLM-L6-v2" if has_embeddings else "mock",
        "evaluator": evaluator_mode,
        "version": "2.0.0",
        "capabilities": {
            "tiktoken": has_tiktoken,
            "embeddings": has_embeddings,
            "llm_evaluator": evaluator_mode == "llm",
            "fine_tuning": tuner is not None,
        },
    })


# ─── API: Benchmark Data ───────────────────────────────────────────────────

@app.route("/api/benchmark", methods=["GET"])
def get_benchmark():
    """Return PubMed benchmark results if available."""
    benchmark_path = os.path.join(BASE_DIR, "benchmark_data", "benchmark_results.json")
    if os.path.exists(benchmark_path):
        with open(benchmark_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No benchmark data found. Run pubmed_benchmark.py first."}), 404


@app.route("/api/training_data", methods=["GET"])
def get_training_data():
    """Return training data samples (paginated)."""
    training_path = os.path.join(BASE_DIR, "benchmark_data", "training_data.jsonl")
    if not os.path.exists(training_path):
        return jsonify({"error": "No training data found."}), 404

    offset = int(request.args.get("offset", 0))
    limit = min(int(request.args.get("limit", 20)), 100)

    samples = []
    total = 0
    with open(training_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            total += 1
            if offset <= i < offset + limit:
                samples.append(json.loads(line))

    return jsonify({
        "samples": samples,
        "total": total,
        "offset": offset,
        "limit": limit,
    })


# ─── API: Project Management ──────────────────────────────────────────────

@app.route("/api/projects", methods=["GET"])
def list_projects():
    """List all projects."""
    if not atom_store:
        return jsonify({"projects": []})
    return jsonify({"projects": atom_store.list_projects(), "active": atom_store.active_project})


@app.route("/api/projects", methods=["POST"])
def create_project():
    """Create a new project and optionally switch to it."""
    data = request.json or {}
    name = data.get("name", "").strip()
    domain = data.get("domain", "general")
    if not name:
        return jsonify({"error": "Project name required"}), 400

    meta = atom_store.create_project(name, domain=domain)
    if data.get("activate", True):
        atom_store.set_active(meta["name"])

    return jsonify({"project": meta, "active": atom_store.active_project})


@app.route("/api/projects/switch", methods=["POST"])
def switch_project():
    """Switch active project."""
    data = request.json or {}
    name = data.get("name", "")
    if not name:
        return jsonify({"error": "Project name required"}), 400

    ok = atom_store.set_active(name)
    if not ok:
        return jsonify({"error": f"Project '{name}' not found"}), 404

    return jsonify({
        "active": atom_store.active_project,
        "atoms": atom_store.get_atom_count(),
        "stats": atom_store.get_stats(),
    })


@app.route("/api/projects/<name>/atoms", methods=["GET"])
def get_project_atoms(name):
    """Get atoms for a specific project (paginated)."""
    # Temporarily switch if needed
    current = atom_store.active_project
    if name != current:
        if not atom_store.set_active(name):
            return jsonify({"error": f"Project '{name}' not found"}), 404

    offset = int(request.args.get("offset", 0))
    limit = min(int(request.args.get("limit", 50)), 200)
    atoms = atom_store.get_atoms(offset=offset, limit=limit)
    total = atom_store.get_atom_count()

    # Switch back
    if name != current and current:
        atom_store.set_active(current)

    return jsonify({"atoms": atoms, "total": total, "offset": offset, "limit": limit})


@app.route("/api/search", methods=["POST"])
def search_atoms():
    """Vector similarity search across stored atoms."""
    data = request.json or {}
    query = data.get("query", "").strip()
    top_k = min(int(data.get("top_k", 10)), 50)

    if not query:
        return jsonify({"error": "Query text required"}), 400
    if not atom_store or atom_store.get_atom_count() == 0:
        return jsonify({"results": [], "message": "No atoms stored yet"})

    # Embed query
    query_embedding = pipeline.embedder.embed(query)
    results = atom_store.search(query_embedding, top_k=top_k)

    return jsonify({"results": results, "query": query, "total_searched": atom_store.get_atom_count()})


@app.route("/api/store/stats", methods=["GET"])
def store_stats():
    """Get atom store statistics."""
    if not atom_store:
        return jsonify({"error": "Store not initialized"})
    return jsonify(atom_store.get_stats())


@app.route("/api/export/<fmt>", methods=["GET"])
def export_data(fmt):
    """Export atoms in various formats: jsonl, huggingface, openai."""
    if not atom_store or atom_store.get_atom_count() == 0:
        return jsonify({"error": "No atoms to export"}), 400

    try:
        if fmt == "jsonl":
            path = atom_store.export_training_jsonl()
        elif fmt == "huggingface":
            path = atom_store.export_huggingface()
        elif fmt == "openai":
            path = atom_store.export_openai_finetune()
        else:
            return jsonify({"error": f"Unknown format: {fmt}. Use: jsonl, huggingface, openai"}), 400

        return jsonify({"path": path, "atoms": atom_store.get_atom_count(), "format": fmt})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  LOGICAL ROOMS — Scientific Article UI")
    print("=" * 60)
    print()

    initialize()

    print()
    print("=" * 60)
    print("  Server running at http://localhost:5000")
    print("=" * 60)
    print()

    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
