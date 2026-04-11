"""
Pipeline Integration Tests
============================
End-to-end tests for the Logical Rooms pipeline, fine-tuning engine,
and dataset management.
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.atom import Atom, OntologyCategory, AXIS_NAMES, merge_atoms, calculate_atom_statistics
from core.surface import Surface, AssociationGraph
from core.models import LogicalRoom
from core.pipeline import LogicalRoomsPipeline
from core.finetuning import AxisFineTuner, AxisCalibration
from core.dataset import LogicalRoomsDataset
from core.axes_evaluator import SevenAxesEvaluator
from core.embeddings import EmbeddingService

import numpy as np


# ─── Atom Tests ─────────────────────────────────────────────────────────────

def test_atom_creation():
    """Test basic atom creation and properties."""
    emb = np.random.rand(384).astype(np.float32)
    atom = Atom(text="Test text for atom creation with enough words to exceed compression threshold", embedding=emb)

    assert atom.text.startswith("Test text")
    assert atom.embedding.shape == (384,)
    assert len(atom.axes) == 7
    assert atom.original_tokens > atom.compressed_tokens  # Enough words to compress
    assert atom.compressed_tokens == 8
    assert atom.compression_ratio > 0  # Positive for longer texts
    assert 0 <= atom.tension <= 1
    print("  [PASS] Atom creation")


def test_atom_serialization():
    """Test atom to_dict and from_dict."""
    emb = np.random.rand(384).astype(np.float32)
    atom = Atom(
        text="Serialization test",
        embedding=emb,
        axes={"temporal": 0.5, "relevance": 0.8, "risk": 0.3, "ontology": 0.2,
              "causality": -0.1, "visibility": 0.6, "trust": 0.9},
        domain="testing",
    )

    data = atom.to_dict()
    restored = Atom.from_dict(data)

    assert restored.text == atom.text
    assert restored.domain == atom.domain
    assert np.allclose(restored.embedding, atom.embedding, atol=1e-6)
    assert restored.axes["relevance"] == 0.8
    print("  ✅ Atom serialization")


def test_atom_similarity():
    """Test atom-to-atom similarity."""
    emb1 = np.random.rand(384).astype(np.float32)
    emb2 = emb1 + np.random.rand(384).astype(np.float32) * 0.1  # Similar

    a1 = Atom(text="Test A", embedding=emb1)
    a2 = Atom(text="Test B", embedding=emb2)

    sim = a1.similarity_to(a2)
    assert 0 <= sim <= 1
    assert sim > 0.5  # Should be similar
    print("  ✅ Atom similarity")


def test_merge_atoms():
    """Test atom merging."""
    atoms = [
        Atom(text="First atom text", embedding=np.random.rand(384).astype(np.float32)),
        Atom(text="Second atom text", embedding=np.random.rand(384).astype(np.float32)),
    ]
    merged = merge_atoms(atoms)

    assert "First" in merged.text
    assert merged.original_tokens == sum(a.original_tokens for a in atoms)
    assert merged.compressed_tokens == 8
    print("  ✅ Atom merging")


def test_ontology_category():
    """Test ontology category parsing."""
    assert OntologyCategory.from_string("human") == OntologyCategory.HUMAN
    assert OntologyCategory.from_string("TECHNOLOGY") == OntologyCategory.TECHNOLOGY
    assert OntologyCategory.from_string("invalid") == OntologyCategory.UNKNOWN
    assert 0 <= OntologyCategory.STRATEGY.numeric <= 1
    print("  ✅ OntologyCategory")


# ─── Surface Tests ──────────────────────────────────────────────────────────

def test_surface_creation():
    """Test surface from atoms."""
    atoms = [
        Atom(text=f"Atom {i}", embedding=np.random.rand(384).astype(np.float32))
        for i in range(3)
    ]
    surface = Surface.from_atoms(atoms, name="Test Surface")

    assert len(surface.atoms) == 3
    assert surface.centroid_embedding is not None
    assert surface.covariance_matrix is not None
    assert surface.total_original_tokens > 0
    print("  ✅ Surface creation")


def test_surface_serialization():
    """Test surface to_dict."""
    atoms = [
        Atom(text=f"Atom {i}", embedding=np.random.rand(384).astype(np.float32))
        for i in range(2)
    ]
    surface = Surface.from_atoms(atoms, name="Serialize Surface")
    data = surface.to_dict()

    assert data["name"] == "Serialize Surface"
    assert len(data["atom_ids"]) == 2
    print("  ✅ Surface serialization")


# ─── Room Tests ─────────────────────────────────────────────────────────────

def test_room_creation():
    """Test room creation and atom addition."""
    room = LogicalRoom(name="Test Room")
    emb = np.random.rand(384).astype(np.float32)
    atom = Atom(
        text="Test atom",
        embedding=emb,
        axes={"relevance": 0.8, "risk": 0.5, "trust": 0.3},
    )
    room.add_object(atom)

    assert len(room.objects) == 1
    assert room.centroid is not None
    assert room.tension > 0
    print("  ✅ Room creation")


# ─── Evaluator Tests ────────────────────────────────────────────────────────

def test_evaluator_deterministic():
    """Test that heuristic evaluation is deterministic."""
    evaluator = SevenAxesEvaluator(use_llm=False)

    text = "Critical security vulnerability with remote code execution"
    axes1, cat1 = evaluator.evaluate(text)
    axes2, cat2 = evaluator.evaluate(text)

    assert axes1 == axes2, "Heuristic evaluation should be deterministic"
    assert cat1 == cat2
    assert axes1["risk"] > 0  # Should detect risk keywords
    print("  ✅ Evaluator deterministic")


def test_evaluator_batch():
    """Test batch evaluation."""
    evaluator = SevenAxesEvaluator(use_llm=False)

    texts = ["Security alert", "Financial report", "AI strategy"]
    results = evaluator.evaluate_batch(texts)

    assert len(results) == 3
    for axes, cat in results:
        assert len(axes) == 7
        assert isinstance(cat, OntologyCategory)
    print("  ✅ Evaluator batch")


# ─── Pipeline Tests ─────────────────────────────────────────────────────────

def test_pipeline_process():
    """Test single-text processing."""
    pipe = LogicalRoomsPipeline(use_llm=False, verbose=False)
    atom = pipe.process("Critical security vulnerability detected", domain="security")

    assert atom.domain == "security"
    assert atom.original_tokens > 0
    assert len(atom.axes) == 7
    print("  ✅ Pipeline process")


def test_pipeline_batch():
    """Test batch processing."""
    pipe = LogicalRoomsPipeline(use_llm=False, verbose=False)
    texts = [
        "Buffer overflow in authentication module",
        "Revenue increased 23% year-over-year",
        "API endpoint accepts JSON payload",
    ]
    atoms = pipe.process_batch(texts, domain="mixed")

    assert len(atoms) == 3
    for atom in atoms:
        assert atom.domain == "mixed"
    print("  ✅ Pipeline batch")


def test_pipeline_compression():
    """Test full compression pipeline."""
    pipe = LogicalRoomsPipeline(use_llm=False, verbose=False)
    texts = [
        "Critical vulnerability allows remote code execution",
        "Firewall logs show unusual traffic patterns",
        "Two-factor authentication bypass discovered",
        "SQL injection in user search endpoint",
    ]

    result = pipe.compress_context(texts, query="security threats", domain="security")

    assert result.compression_ratio > 0
    assert result.compressed_tokens < result.original_tokens
    assert result.rooms_used > 0
    assert result.atoms_created == 4
    assert len(result.compressed_text) > 0
    print(f"  ✅ Pipeline compression ({result.compression_ratio:.0%} reduction)")


def test_pipeline_surfaces():
    """Test surface building strategies."""
    pipe = LogicalRoomsPipeline(use_llm=False, verbose=False)
    texts = [f"Test text number {i} for surface building" for i in range(6)]
    atoms = pipe.process_batch(texts)

    # Fixed strategy
    surfaces_fixed = pipe.build_surfaces(atoms, strategy="fixed")
    assert len(surfaces_fixed) > 0

    # Domain strategy
    pipe.reset()
    atoms_domain = pipe.process_batch(["Security alert", "Financial report"], domain="security")
    atoms_domain += pipe.process_batch(["Database schema"], domain="tech")
    surfaces_domain = pipe.build_surfaces(atoms_domain, strategy="domain")
    assert len(surfaces_domain) > 0

    print("  ✅ Pipeline surfaces")


def test_pipeline_stats():
    """Test pipeline statistics."""
    pipe = LogicalRoomsPipeline(use_llm=False, verbose=False)
    pipe.process("Test text", domain="test")

    stats = pipe.get_stats()
    assert stats["total_processed"] == 1
    assert stats["total_atoms"] == 1
    print("  ✅ Pipeline stats")


# ─── Fine-Tuning Tests ─────────────────────────────────────────────────────

def test_finetuning_feedback():
    """Test feedback recording."""
    tuner = AxisFineTuner()
    tuner.record_feedback(
        atom_id="test-123",
        text="Test text",
        predicted_axes={"risk": 0.3, "trust": 0.7},
        corrected_axes={"risk": 0.8, "trust": 0.2},
        domain="security",
    )

    assert len(tuner.feedback) == 1
    assert tuner.feedback[0].mean_error > 0
    print("  ✅ Fine-tuning feedback")


def test_finetuning_train_step():
    """Test training step."""
    tuner = AxisFineTuner(learning_rate=0.1)

    for i in range(5):
        tuner.record_feedback(
            atom_id=f"test-{i}",
            text=f"Training sample {i}",
            predicted_axes={"risk": 0.3 + i * 0.1, "trust": 0.7 - i * 0.1},
            corrected_axes={"risk": 0.8, "trust": 0.2},
            domain="security",
        )

    metrics = tuner.train_step()
    assert metrics.samples_used == 5
    assert metrics.step == 1
    print("  ✅ Fine-tuning train step")


def test_finetuning_calibration():
    """Test calibration save/load."""
    tuner = AxisFineTuner()
    tuner.calibration.bias["risk"] = 0.1
    tuner.calibration.scale["risk"] = 1.2

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name

    try:
        tuner.save_calibration(path)

        tuner2 = AxisFineTuner()
        tuner2.load_calibration(path)

        assert abs(tuner2.calibration.bias["risk"] - 0.1) < 0.001
        assert abs(tuner2.calibration.scale["risk"] - 1.2) < 0.001
        print("  ✅ Fine-tuning calibration I/O")
    finally:
        os.unlink(path)


def test_finetuning_export():
    """Test training data export."""
    tuner = AxisFineTuner()
    tuner.record_feedback(
        atom_id="export-test",
        text="Export test text",
        predicted_axes={"risk": 0.3},
        corrected_axes={"risk": 0.9},
    )

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name

    try:
        count = tuner.export_training_data(path, format="jsonl")
        assert count == 1

        with open(path, "r") as f:
            data = json.loads(f.readline())
            assert data["text"] == "Export test text"
        print("  ✅ Fine-tuning export")
    finally:
        os.unlink(path)


# ─── Dataset Tests ──────────────────────────────────────────────────────────

def test_dataset_basic():
    """Test dataset creation and samples."""
    ds = LogicalRoomsDataset(name="test")
    ds.add_sample("Test text 1", axes={"risk": 0.5}, domain="security")
    ds.add_sample("Test text 2", axes={"risk": 0.3}, domain="finance")

    assert len(ds) == 2
    stats = ds.get_statistics()
    assert stats.total_samples == 2
    assert "security" in stats.domains
    print("  ✅ Dataset basic")


def test_dataset_split():
    """Test dataset splitting."""
    ds = LogicalRoomsDataset()
    for i in range(20):
        domain = "security" if i % 2 == 0 else "finance"
        ds.add_sample(f"Sample {i}", axes={"risk": i / 20}, domain=domain)

    train, val, test = ds.split(train=0.8, val=0.1, test=0.1)

    assert len(train) > 0
    assert len(val) >= 0
    assert len(test) >= 0
    assert len(train) + len(val) + len(test) == 20
    print("  ✅ Dataset split")


def test_dataset_io():
    """Test dataset export/import."""
    ds = LogicalRoomsDataset()
    ds.add_sample("IO test", axes={"risk": 0.7, "trust": 0.3}, domain="test")

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name

    try:
        ds.export(path)

        ds2 = LogicalRoomsDataset()
        count = ds2.load(path)

        assert count == 1
        assert ds2.samples[0].text == "IO test"
        assert ds2.samples[0].axes["risk"] == 0.7
        print("  ✅ Dataset I/O")
    finally:
        os.unlink(path)


# ─── Runner ─────────────────────────────────────────────────────────────────

def run_all_tests():
    """Run all pipeline tests."""
    print("\n" + "=" * 60)
    print("  LOGICAL ROOMS — TEST SUITE")
    print("=" * 60 + "\n")

    test_groups = [
        ("Atom", [
            test_atom_creation,
            test_atom_serialization,
            test_atom_similarity,
            test_merge_atoms,
            test_ontology_category,
        ]),
        ("Surface", [
            test_surface_creation,
            test_surface_serialization,
        ]),
        ("Room", [
            test_room_creation,
        ]),
        ("Evaluator", [
            test_evaluator_deterministic,
            test_evaluator_batch,
        ]),
        ("Pipeline", [
            test_pipeline_process,
            test_pipeline_batch,
            test_pipeline_compression,
            test_pipeline_surfaces,
            test_pipeline_stats,
        ]),
        ("Fine-Tuning", [
            test_finetuning_feedback,
            test_finetuning_train_step,
            test_finetuning_calibration,
            test_finetuning_export,
        ]),
        ("Dataset", [
            test_dataset_basic,
            test_dataset_split,
            test_dataset_io,
        ]),
    ]

    total = 0
    passed = 0
    failed = 0

    for group_name, tests in test_groups:
        print(f"\n{group_name}:")
        for test_fn in tests:
            total += 1
            try:
                test_fn()
                passed += 1
            except Exception as e:
                failed += 1
                print(f"  ❌ {test_fn.__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    status = "ALL TESTS PASSED ✅" if failed == 0 else f"{failed} TESTS FAILED ❌"
    print(f"  {status}")
    print(f"{'=' * 60}\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
