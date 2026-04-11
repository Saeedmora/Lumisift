"""
Atom — Semantic Primitive Unit
===============================
The smallest unit of meaning in the Logical Rooms framework.
Like Chinese characters, each Atom encodes dense, multi-dimensional
information in a compact representation.

An Atom carries:
  - Raw text content
  - 384-D embedding vector
  - 7-axis semantic scores
  - Anomaly tracking for self-optimization
  - Serialization for dataset I/O
"""

import numpy as np
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

try:
    import tiktoken
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")  # GPT-4 / GPT-3.5 tokenizer
except ImportError:
    _TOKENIZER = None


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (BPE) if available, else word count."""
    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text))
    return len(text.split())


# ─── Canonical Ontology Categories ──────────────────────────────────────────

class OntologyCategory(Enum):
    """
    Ontological categories for the A4 (ontology) axis.
    Single source of truth — imported everywhere else.
    """
    HUMAN = "human"
    PROCESS = "process"
    TECHNOLOGY = "technology"
    INFORMATION = "information"
    STRATEGY = "strategy"
    UNKNOWN = "unknown"

    @property
    def numeric(self) -> float:
        """Normalized numeric value for axis calculations (0.0 – 1.0)."""
        mapping = {
            "human": 0.0,
            "process": 0.2,
            "technology": 0.4,
            "information": 0.6,
            "strategy": 0.8,
            "unknown": 0.5,
        }
        return mapping.get(self.value, 0.5)

    @classmethod
    def from_string(cls, s: str) -> "OntologyCategory":
        """Parse a category from a string (case-insensitive)."""
        try:
            return cls(s.lower().strip())
        except ValueError:
            return cls.UNKNOWN


# ─── Axis Constants ─────────────────────────────────────────────────────────

AXIS_NAMES = [
    "temporal",     # A1: -1 (outdated) → +1 (future-relevant)
    "relevance",    # A2:  0 → 1 (strategic weight)
    "risk",         # A3: -1 → +1 (threat / uncertainty)
    "ontology",     # A4:  0 → 1 (encoded category)
    "causality",    # A5: -1 (cause) → +1 (effect)
    "visibility",   # A6:  0 (internal) → 1 (public)
    "trust",        # A7:  0 → 1 (reliability)
]

DEFAULT_AXES: Dict[str, float] = {
    "temporal": 0.0,
    "relevance": 0.5,
    "risk": 0.0,
    "ontology": 0.0,
    "causality": 0.0,
    "visibility": 0.5,
    "trust": 0.5,
}


# ─── Atom ───────────────────────────────────────────────────────────────────

@dataclass
class Atom:
    """
    Atom = smallest semantic unit.

    Analogy: a Chinese character that packs multiple meanings into
    a single, dense glyph.  An Atom packs text + embedding + 7 axes
    into one reusable object.

    Attributes:
        text:           Raw content string.
        embedding:      384-D float32 vector.
        axes:           Dict of 7 axis scores.
        domain:         Domain tag (e.g. "security", "finance").
        id:             Unique UUID.
        room_id:        Parent room (set during projection).
        surface_id:     Parent surface (set during aggregation).
        anomaly_score:  Distance from room centroid on axes.
        is_anomaly:     Whether the atom is an outlier.
        confidence:     Evaluator confidence in the axes scores (0–1).
        original_tokens:    Word count of raw text.
        compressed_tokens:  Compressed representation size.
    """

    # Core
    text: str
    embedding: np.ndarray

    # 7 axes
    axes: Dict[str, float] = field(default_factory=lambda: DEFAULT_AXES.copy())

    # Metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    domain: str = "general"
    room_id: Optional[str] = None
    surface_id: Optional[str] = None
    ontology_category: OntologyCategory = OntologyCategory.UNKNOWN

    # Anomaly tracking
    anomaly_score: float = 0.0
    is_anomaly: bool = False

    # Evaluator confidence
    confidence: float = 1.0

    # Token stats
    original_tokens: int = 0
    compressed_tokens: int = 0

    def __post_init__(self):
        self.original_tokens = _count_tokens(self.text)
        self.compressed_tokens = _count_tokens(self.to_compressed_repr())

    # ── Vector helpers ──────────────────────────────────────────────────

    @property
    def axes_vector(self) -> np.ndarray:
        """Axes as a numpy vector in canonical order."""
        return np.array([self.axes.get(k, 0.0) for k in AXIS_NAMES], dtype=np.float32)

    @property
    def tension(self) -> float:
        """Meta-tension: T = Relevance × Risk⁺ × (1 − Trust)."""
        rel = self.axes.get("relevance", 0.5)
        risk = max(0.0, self.axes.get("risk", 0.0))
        trust = self.axes.get("trust", 0.5)
        return rel * risk * (1.0 - trust)

    @property
    def compression_ratio(self) -> float:
        """1 − (compressed / original).  Higher is better."""
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.original_tokens)

    # ── Similarity / anomaly ────────────────────────────────────────────

    def check_anomaly(self, room_mean_axes: np.ndarray, threshold: float = 0.5) -> bool:
        """Flag this atom as anomalous if its axes deviate from the room mean."""
        diff = float(np.linalg.norm(self.axes_vector - room_mean_axes))
        self.anomaly_score = diff
        self.is_anomaly = diff > threshold
        return self.is_anomaly

    def similarity_to(self, other: "Atom", alpha: float = 0.6) -> float:
        """
        Weighted similarity: α·cos(embed) + (1−α)·(1 − norm_axes_dist).
        """
        # Embedding cosine similarity
        e1, e2 = self.embedding, other.embedding
        denom = np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9
        cos_sim = float(np.dot(e1, e2) / denom)

        # Axes distance (normalized)
        a1, a2 = self.axes_vector, other.axes_vector
        max_dist = np.sqrt(len(AXIS_NAMES) * 4)  # range [-1,+1] → span 2
        axes_sim = 1.0 - (float(np.linalg.norm(a1 - a2)) / max_dist)

        return alpha * cos_sim + (1.0 - alpha) * axes_sim

    # ── Compressed representation ───────────────────────────────────────

    def to_compressed_repr(self) -> str:
        """Compact string for LLM context injection."""
        parts = "|".join(
            f"{k[:2].upper()}:{self.axes.get(k, 0):.1f}" for k in AXIS_NAMES
        )
        return f"[{parts}]"

    # ── Serialization ───────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding.tolist(),
            "axes": self.axes.copy(),
            "domain": self.domain,
            "ontology_category": self.ontology_category.value,
            "confidence": self.confidence,
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "room_id": self.room_id,
            "surface_id": self.surface_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Atom":
        """Deserialize from a dictionary."""
        atom = cls(
            text=data["text"],
            embedding=np.array(data["embedding"], dtype=np.float32),
            axes=data.get("axes", DEFAULT_AXES.copy()),
            id=data.get("id", str(uuid.uuid4())),
            domain=data.get("domain", "general"),
            ontology_category=OntologyCategory.from_string(
                data.get("ontology_category", "unknown")
            ),
            confidence=data.get("confidence", 1.0),
        )
        atom.anomaly_score = data.get("anomaly_score", 0.0)
        atom.is_anomaly = data.get("is_anomaly", False)
        atom.room_id = data.get("room_id")
        atom.surface_id = data.get("surface_id")
        return atom

    def __repr__(self):
        return (
            f"Atom(id={self.id[:8]}, domain={self.domain}, "
            f"tension={self.tension:.2f}, anomaly={self.anomaly_score:.2f})"
        )


# ─── Utility Functions ──────────────────────────────────────────────────────

def merge_atoms(atoms: List[Atom]) -> Atom:
    """
    Merge multiple atoms into a single, denser atom.
    Like composed Chinese characters: radicals → compound.
    """
    if not atoms:
        raise ValueError("At least one Atom is required")
    if len(atoms) == 1:
        return atoms[0]

    merged_embedding = np.mean([a.embedding for a in atoms], axis=0)
    merged_axes = {
        k: float(np.mean([a.axes.get(k, 0.0) for a in atoms]))
        for k in AXIS_NAMES
    }
    merged_text = " | ".join(a.text[:50] for a in atoms)
    merged_domain = atoms[0].domain  # inherit from first

    merged = Atom(
        text=merged_text,
        embedding=merged_embedding,
        axes=merged_axes,
        domain=merged_domain,
        confidence=float(np.mean([a.confidence for a in atoms])),
    )
    merged.original_tokens = sum(a.original_tokens for a in atoms)
    # compressed_tokens is set by __post_init__ via to_compressed_repr()
    return merged


def calculate_atom_statistics(atoms: List[Atom]) -> Dict[str, Any]:
    """Aggregate statistics over a collection of atoms."""
    if not atoms:
        return {"count": 0}

    total_orig = sum(a.original_tokens for a in atoms)
    total_comp = sum(a.compressed_tokens for a in atoms)

    return {
        "count": len(atoms),
        "total_original_tokens": total_orig,
        "total_compressed_tokens": total_comp,
        "compression_ratio": 1.0 - (total_comp / max(1, total_orig)),
        "avg_tension": float(np.mean([a.tension for a in atoms])),
        "anomaly_count": sum(1 for a in atoms if a.is_anomaly),
        "avg_anomaly_score": float(np.mean([a.anomaly_score for a in atoms])),
        "avg_confidence": float(np.mean([a.confidence for a in atoms])),
        "domains": list({a.domain for a in atoms}),
    }
