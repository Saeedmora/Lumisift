"""
Surface — Aggregated Atom Cluster with Covariance
===================================================
A Surface groups related Atoms and computes inter-axis correlations.
Think of it as a compound Chinese character built from radicals.
"""

import numpy as np
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from core.atom import Atom, AXIS_NAMES, _count_tokens


@dataclass
class Surface:
    """
    Surface = a semantic "area" formed by aggregating Atoms.

    Properties:
        - Centroid embedding (mean of atom embeddings)
        - Mean axes vector + 7×7 covariance matrix
        - Associations to other surfaces
        - Serialization support
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Unnamed Surface"

    # Atoms
    atoms: List[Atom] = field(default_factory=list)

    # Aggregated representation
    centroid_embedding: Optional[np.ndarray] = None
    mean_axes: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    covariance_matrix: Optional[np.ndarray] = None  # 7×7

    # Room membership
    room_id: Optional[str] = None

    # Associations
    associations: Dict[str, float] = field(default_factory=dict)

    # Token stats
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0

    # ── Atom management ─────────────────────────────────────────────────

    def add_atom(self, atom: Atom):
        """Add an atom and re-compute aggregates."""
        self.atoms.append(atom)
        atom.surface_id = self.id
        self._update_aggregates()

    @classmethod
    def from_atoms(cls, atoms: List[Atom], name: str = "Surface") -> "Surface":
        """Factory: build a Surface from a pre-existing list of Atoms."""
        surface = cls(name=name)
        for atom in atoms:
            surface.atoms.append(atom)
            atom.surface_id = surface.id
        surface._update_aggregates()
        return surface

    # ── Internal aggregation ────────────────────────────────────────────

    def _update_aggregates(self):
        if not self.atoms:
            return

        embeddings = np.array([a.embedding for a in self.atoms])
        self.centroid_embedding = np.mean(embeddings, axis=0)

        axes_vectors = np.array([a.axes_vector for a in self.atoms])
        self.mean_axes = np.mean(axes_vectors, axis=0).astype(np.float32)

        if len(self.atoms) > 1:
            self.covariance_matrix = np.cov(axes_vectors.T)
        else:
            self.covariance_matrix = np.zeros((7, 7), dtype=np.float32)

        self.total_original_tokens = sum(a.original_tokens for a in self.atoms)
        # Measure the actual token count of the compressed representation
        self.total_compressed_tokens = _count_tokens(self.to_compressed_repr())

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def tension(self) -> float:
        """Aggregated tension from mean axes."""
        rel = float(self.mean_axes[1])   # relevance
        risk = max(0.0, float(self.mean_axes[2]))  # risk
        trust = float(self.mean_axes[6])  # trust
        return rel * risk * (1.0 - trust)

    @property
    def compression_ratio(self) -> float:
        if self.total_original_tokens == 0:
            return 0.0
        return 1.0 - (self.total_compressed_tokens / self.total_original_tokens)

    # ── Correlation analysis ────────────────────────────────────────────

    def get_axis_correlations(self) -> Dict[Tuple[str, str], float]:
        """Return significant (|r| > 0.3) pairwise axis correlations."""
        if self.covariance_matrix is None or len(self.atoms) < 2:
            return {}

        correlations = {}
        std = np.sqrt(np.diag(self.covariance_matrix))
        std = np.where(std == 0, 1.0, std)

        for i in range(7):
            for j in range(i + 1, 7):
                r = self.covariance_matrix[i, j] / (std[i] * std[j])
                if abs(r) > 0.3:
                    correlations[(AXIS_NAMES[i], AXIS_NAMES[j])] = round(float(r), 3)
        return correlations

    # ── Similarity ──────────────────────────────────────────────────────

    def similarity_to(self, other: "Surface", alpha: float = 0.6) -> float:
        """Weighted cosine-embedding + axes similarity."""
        if self.centroid_embedding is None or other.centroid_embedding is None:
            return 0.0

        e1, e2 = self.centroid_embedding, other.centroid_embedding
        cos_sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))

        a1, a2 = self.mean_axes, other.mean_axes
        max_dist = np.sqrt(7 * 4)
        axes_sim = 1.0 - (float(np.linalg.norm(a1 - a2)) / max_dist)

        return alpha * cos_sim + (1.0 - alpha) * axes_sim

    # ── Compressed representation ───────────────────────────────────────

    def to_compressed_repr(self) -> str:
        """Compact string for LLM context."""
        abbr = ["TE", "RE", "RI", "ON", "CA", "VI", "TR"]
        axes_str = "|".join(f"{a}:{self.mean_axes[i]:.1f}" for i, a in enumerate(abbr))
        return f"[{self.name}|{len(self.atoms)}atoms|{axes_str}]"

    # ── Serialization ───────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "room_id": self.room_id,
            "atom_ids": [a.id for a in self.atoms],
            "mean_axes": self.mean_axes.tolist(),
            "total_original_tokens": self.total_original_tokens,
            "total_compressed_tokens": self.total_compressed_tokens,
            "associations": self.associations.copy(),
        }

    def __repr__(self):
        return f"Surface(name='{self.name}', atoms={len(self.atoms)}, tension={self.tension:.2f})"


# ─── Association Graph ──────────────────────────────────────────────────────

class AssociationGraph:
    """
    Graph of associations between Surfaces.
    Enables network-level anomaly detection and concept navigation.
    """

    def __init__(self, alpha: float = 0.6, similarity_threshold: float = 0.4):
        self.surfaces: Dict[str, Surface] = {}
        self.edges: Dict[Tuple[str, str], float] = {}
        self.alpha = alpha
        self.similarity_threshold = similarity_threshold

    def add_surface(self, surface: Surface):
        self.surfaces[surface.id] = surface
        for other_id, other in self.surfaces.items():
            if other_id == surface.id:
                continue
            sim = surface.similarity_to(other, self.alpha)
            if sim >= self.similarity_threshold:
                key = tuple(sorted([surface.id, other_id]))
                self.edges[key] = sim
                surface.associations[other_id] = sim
                other.associations[surface.id] = sim

    def get_related_surfaces(self, surface_id: str, top_k: int = 5) -> List[Tuple[Surface, float]]:
        if surface_id not in self.surfaces:
            return []
        surface = self.surfaces[surface_id]
        related = [
            (self.surfaces[oid], sim)
            for oid, sim in surface.associations.items()
            if oid in self.surfaces
        ]
        related.sort(key=lambda x: -x[1])
        return related[:top_k]

    def get_graph_stats(self) -> Dict[str, Any]:
        return {
            "total_surfaces": len(self.surfaces),
            "total_edges": len(self.edges),
            "avg_associations": (
                float(np.mean([len(s.associations) for s in self.surfaces.values()]))
                if self.surfaces else 0
            ),
            "total_atoms": sum(len(s.atoms) for s in self.surfaces.values()),
            "total_original_tokens": sum(s.total_original_tokens for s in self.surfaces.values()),
            "total_compressed_tokens": sum(s.total_compressed_tokens for s in self.surfaces.values()),
        }

    def get_compression_stats(self) -> Dict[str, Any]:
        stats = self.get_graph_stats()
        orig = stats["total_original_tokens"]
        comp = stats["total_compressed_tokens"]
        if orig == 0:
            return {"compression_ratio": 0, "tokens_saved": 0}
        ratio = 1.0 - (comp / orig)
        return {
            "original_tokens": orig,
            "compressed_tokens": comp,
            "compression_ratio": round(ratio, 3),
            "tokens_saved": orig - comp,
            "tokens_saved_pct": round(ratio * 100, 1),
        }
