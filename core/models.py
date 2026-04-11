"""
Logical Room — Top-Level Context Cluster
==========================================
A Room groups related Atoms (via Surfaces) and supports
self-optimization with EMA updates, tension monitoring,
and automatic splitting.
"""

import numpy as np
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from core.atom import Atom, AXIS_NAMES, DEFAULT_AXES, _count_tokens


@dataclass
class LogicalRoom:
    """
    Top-level organizational unit in the Logical Rooms hierarchy.

    A Room:
      - Contains Atoms (optionally grouped into Surfaces)
      - Maintains an EMA-updated centroid and axes summary
      - Monitors tension and triggers reviews when exceeded
      - Splits when internal variance is too high
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Unnamed Room"
    objects: List[Atom] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    mean_axes: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    axes_summary: Dict[str, float] = field(default_factory=lambda: DEFAULT_AXES.copy())

    # Self-optimization parameters
    alpha: float = 0.1           # EMA learning rate
    tension_threshold: float = 0.3
    variance: Optional[np.ndarray] = None

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def tension(self) -> float:
        """Room-level tension from aggregated axes."""
        rel = self.axes_summary.get("relevance", 0.5)
        risk = max(0.0, self.axes_summary.get("risk", 0.0))
        trust = self.axes_summary.get("trust", 0.5)
        return rel * risk * (1.0 - trust)

    @property
    def should_trigger_review(self) -> bool:
        return self.tension > self.tension_threshold

    @property
    def compression_ratio(self) -> float:
        """Room-level compression ratio using real BPE token counts."""
        total_orig = sum(a.original_tokens for a in self.objects)
        if total_orig == 0:
            return 0.0
        compressed = _count_tokens(self.to_compressed_repr())
        return 1.0 - (compressed / total_orig)

    # ── Object management ───────────────────────────────────────────────

    def add_object(self, obj: Atom):
        """Add an atom and update room state."""
        self.objects.append(obj)
        obj.room_id = self.id
        self._update_centroid(obj.embedding)
        self._update_axes(obj.axes_vector, obj.axes)

    def update(self, v_new: np.ndarray, a_new: np.ndarray, axes_dict: Dict[str, float]):
        """EMA self-optimization update."""
        if self.centroid is not None:
            self.centroid = self.centroid + self.alpha * (v_new - self.centroid)
        else:
            self.centroid = v_new.copy()

        self.mean_axes = self.mean_axes + self.alpha * (a_new - self.mean_axes)

        for i, name in enumerate(AXIS_NAMES):
            self.axes_summary[name] = float(self.mean_axes[i])

        if self.variance is None:
            self.variance = np.zeros(7, dtype=np.float32)
        delta = a_new - self.mean_axes
        self.variance = self.variance + self.alpha * (delta ** 2 - self.variance)

    # ── Internal helpers ────────────────────────────────────────────────

    def _update_centroid(self, new_embedding: np.ndarray):
        if self.centroid is None:
            self.centroid = new_embedding.copy()
        else:
            n = len(self.objects)
            self.centroid = (self.centroid * (n - 1) + new_embedding) / n

    def _update_axes(self, axes_vector: np.ndarray, axes_dict: Dict[str, float]):
        n = len(self.objects)
        if n == 1:
            self.mean_axes = axes_vector.copy().astype(np.float32)
            self.axes_summary = axes_dict.copy()
        else:
            self.mean_axes = ((self.mean_axes * (n - 1) + axes_vector) / n).astype(np.float32)
            for k, v in axes_dict.items():
                current = self.axes_summary.get(k, 0.0)
                self.axes_summary[k] = (current * (n - 1) + v) / n

    # ── Splitting ───────────────────────────────────────────────────────

    def should_split(self, variance_threshold: float = 0.5) -> bool:
        if self.variance is None:
            return False
        return bool(np.max(self.variance) > variance_threshold)

    # ── Serialization ───────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "axes_summary": self.axes_summary.copy(),
            "tension": self.tension,
            "object_count": len(self.objects),
            "atom_ids": [o.id for o in self.objects],
        }

    def to_compressed_repr(self) -> str:
        """Compact string for LLM context injection."""
        abbr = ["TE", "RE", "RI", "ON", "CA", "VI", "TR"]
        axes_str = "|".join(
            f"{a}:{self.axes_summary.get(AXIS_NAMES[i], 0):.1f}"
            for i, a in enumerate(abbr)
        )
        return f"[Room:{self.name}|{len(self.objects)}atoms|{axes_str}]"

    def __repr__(self):
        return f"LogicalRoom(name='{self.name}', objects={len(self.objects)}, tension={self.tension:.2f})"
