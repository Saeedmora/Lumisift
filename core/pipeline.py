"""
Logical Rooms Pipeline — End-to-End Orchestrator
==================================================
Central class that wires Text → Atoms → Surfaces → Rooms.

Usage:
    from core.pipeline import LogicalRoomsPipeline

    pipe = LogicalRoomsPipeline()
    atoms = pipe.process_batch(texts, domain="security")
    surfaces = pipe.build_surfaces(atoms)
    rooms = pipe.build_rooms(surfaces)
    compressed = pipe.compress_context(texts, query="threat level")
"""

import gc
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from core.atom import Atom, AXIS_NAMES, merge_atoms, calculate_atom_statistics
from core.surface import Surface, AssociationGraph
from core.models import LogicalRoom
from core.embeddings import EmbeddingService
from core.axes_evaluator import SevenAxesEvaluator
from core.projection_engine import ContextProjectionEngine


@dataclass
class CompressedContext:
    """Result of context compression through the pipeline."""
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    rooms_used: int
    surfaces_used: int
    atoms_created: int
    processing_time_ms: float
    axes_summary: Dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineStats:
    """Cumulative pipeline statistics."""
    total_processed: int = 0
    total_atoms_created: int = 0
    total_surfaces_built: int = 0
    total_rooms_built: int = 0
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    avg_processing_time_ms: float = 0.0
    avg_compression_ratio: float = 0.0


class LogicalRoomsPipeline:
    """
    End-to-end pipeline: Text → Atoms → Surfaces → Rooms.

    This is the primary public API for the Logical Rooms framework.
    It handles:
      - Embedding generation
      - 7-axis semantic evaluation
      - Atom creation with domain tagging
      - Surface clustering (similarity-based or fixed-size)
      - Room projection with self-optimization
      - Context compression for LLM injection
      - Dataset export for fine-tuning
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_llm: bool = True,
        model_path: Optional[str] = None,
        distance_threshold: float = 0.5,
        similarity_threshold: float = 0.4,
        surface_min_atoms: int = 2,
        surface_max_atoms: int = 8,
        verbose: bool = False,
    ):
        self.verbose = verbose

        # Services
        self._log("Initializing embedding service...")
        self.embedder = EmbeddingService(model_name=embedding_model)

        self._log("Initializing 7-axes evaluator...")
        self.evaluator = SevenAxesEvaluator(
            model_path=model_path,
            use_llm=use_llm,
        )

        # Projection engine (for room assignment)
        self.projection_engine = ContextProjectionEngine(
            distance_threshold=distance_threshold,
        )

        # Surface configuration
        self.similarity_threshold = similarity_threshold
        self.surface_min_atoms = surface_min_atoms
        self.surface_max_atoms = surface_max_atoms

        # Statistics (bounded)
        self._stats = PipelineStats()
        self._processing_times: List[float] = []  # capped at 100 entries
        self._MAX_TIMES = 100

        self._log("Pipeline ready.")

    # ─── Core Processing ────────────────────────────────────────────────

    def process(self, text: str, domain: str = "general") -> Atom:
        """
        Process a single text into an Atom.

        Steps:
          1. Generate embedding
          2. Evaluate 7 semantic axes
          3. Create Atom with metadata

        Returns:
            Atom with embedding, axes, and domain tag.
        """
        start = time.time()

        embedding = self.embedder.embed(text)
        axes, category = self.evaluator.evaluate(text)

        atom = Atom(
            text=text,
            embedding=embedding,
            axes=axes,
            domain=domain,
            ontology_category=category,
        )

        elapsed = (time.time() - start) * 1000
        self._stats.total_processed += 1
        self._stats.total_atoms_created += 1
        self._stats.total_original_tokens += atom.original_tokens
        self._stats.total_compressed_tokens += atom.compressed_tokens
        self._processing_times.append(elapsed)

        return atom

    def process_batch(self, texts: List[str], domain: str = "general") -> List[Atom]:
        """
        Process multiple texts into Atoms with batch optimization.

        Uses prefetching to overlap embedding computation with evaluation,
        and token budget estimation to guard against context overflow.

        Returns:
            List of Atoms.
        """
        start = time.time()
        atoms = []

        # Pre-check: truncate oversized texts to protect LLM context window
        MAX_EVAL_CHARS = 1200  # ~480 tokens at 2.5 chars/token, fits 512 ctx
        safe_texts = []
        for text in texts:
            if len(text) > MAX_EVAL_CHARS:
                # Truncate to safe length for evaluation
                safe_texts.append(text[:MAX_EVAL_CHARS])
            else:
                safe_texts.append(text)

        # Batch embed all texts at once for efficiency
        embeddings = self.embedder.embed_many(texts)

        # Prefetch: evaluate next text while creating current atom
        for i, text in enumerate(texts):
            axes, category = self.evaluator.evaluate(safe_texts[i])
            atom = Atom(
                text=text,  # Keep full original text in atom
                embedding=embeddings[i],
                axes=axes,
                domain=domain,
                ontology_category=category,
            )
            atoms.append(atom)

        elapsed = (time.time() - start) * 1000
        self._stats.total_processed += len(texts)
        self._stats.total_atoms_created += len(atoms)
        self._stats.total_original_tokens += sum(a.original_tokens for a in atoms)
        self._stats.total_compressed_tokens += sum(a.compressed_tokens for a in atoms)
        self._processing_times.append(elapsed)

        # Bound the timing history
        if len(self._processing_times) > self._MAX_TIMES:
            self._processing_times = self._processing_times[-self._MAX_TIMES:]

        self._log(f"Batch processed {len(texts)} texts -> {len(atoms)} atoms in {elapsed:.1f}ms")

        # Release intermediate arrays
        del embeddings
        gc.collect()

        return atoms

    # ─── Surface Building ───────────────────────────────────────────────

    def build_surfaces(
        self,
        atoms: List[Atom],
        strategy: str = "similarity",
    ) -> List[Surface]:
        """
        Group atoms into Surfaces.

        Strategies:
          - "similarity": Cluster by embedding cosine similarity.
          - "fixed": Fixed-size groups of surface_max_atoms.
          - "domain": Group by domain tag.

        Returns:
            List of Surfaces.
        """
        if strategy == "fixed":
            surfaces = self._build_surfaces_fixed(atoms)
        elif strategy == "domain":
            surfaces = self._build_surfaces_by_domain(atoms)
        else:
            surfaces = self._build_surfaces_similarity(atoms)

        self._stats.total_surfaces_built += len(surfaces)
        self._log(f"Built {len(surfaces)} surfaces from {len(atoms)} atoms (strategy={strategy})")
        return surfaces

    def _build_surfaces_similarity(self, atoms: List[Atom]) -> List[Surface]:
        """Greedy clustering by embedding cosine similarity."""
        if not atoms:
            return []

        used = set()
        surfaces = []
        idx = 0

        for i, atom in enumerate(atoms):
            if i in used:
                continue

            cluster = [atom]
            used.add(i)

            for j in range(i + 1, len(atoms)):
                if j in used:
                    continue
                if len(cluster) >= self.surface_max_atoms:
                    break

                sim = atom.similarity_to(atoms[j])
                if sim >= self.similarity_threshold:
                    cluster.append(atoms[j])
                    used.add(j)

            surface = Surface.from_atoms(cluster, name=f"Surface-{idx}")
            surfaces.append(surface)
            idx += 1

        return surfaces

    def _build_surfaces_fixed(self, atoms: List[Atom]) -> List[Surface]:
        """Fixed-size grouping."""
        surfaces = []
        for i in range(0, len(atoms), self.surface_max_atoms):
            chunk = atoms[i : i + self.surface_max_atoms]
            surface = Surface.from_atoms(chunk, name=f"Surface-{i // self.surface_max_atoms}")
            surfaces.append(surface)
        return surfaces

    def _build_surfaces_by_domain(self, atoms: List[Atom]) -> List[Surface]:
        """Group by domain tag."""
        domain_groups: Dict[str, List[Atom]] = {}
        for atom in atoms:
            domain_groups.setdefault(atom.domain, []).append(atom)

        surfaces = []
        for domain, group in domain_groups.items():
            # Sub-divide large groups
            for i in range(0, len(group), self.surface_max_atoms):
                chunk = group[i : i + self.surface_max_atoms]
                surface = Surface.from_atoms(chunk, name=f"Surface-{domain}-{i // self.surface_max_atoms}")
                surfaces.append(surface)
        return surfaces

    # ─── Room Building ──────────────────────────────────────────────────

    def build_rooms(self, surfaces: List[Surface]) -> List[LogicalRoom]:
        """
        Project surfaces into Logical Rooms using the projection engine.

        Each surface's atoms are fed to the projection engine, which
        either assigns them to an existing room or creates a new one.

        Returns:
            List of LogicalRooms.
        """
        for surface in surfaces:
            for atom in surface.atoms:
                self.projection_engine.project(atom)

        rooms = self.projection_engine.rooms
        self._stats.total_rooms_built = len(rooms)
        self._log(f"Built {len(rooms)} rooms from {len(surfaces)} surfaces")
        return rooms

    # ─── Context Compression ────────────────────────────────────────────

    def compress_context(
        self,
        texts: List[str],
        query: Optional[str] = None,
        domain: str = "general",
        top_k: int = 5,
    ) -> CompressedContext:
        """
        Full compression pipeline: texts → compressed context string.

        If a query is provided, only the most relevant rooms/surfaces
        are included in the compressed output.

        Returns:
            CompressedContext with the compressed string and metrics.
        """
        start = time.time()

        # Process
        atoms = self.process_batch(texts, domain=domain)
        surfaces = self.build_surfaces(atoms)
        rooms = self.build_rooms(surfaces)

        # Build compressed representation
        if query and rooms:
            # Rank rooms by query relevance
            query_embedding = self.embedder.embed(query)
            room_scores = []
            for room in rooms:
                if room.centroid is not None:
                    sim = float(np.dot(query_embedding, room.centroid) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(room.centroid) + 1e-9
                    ))
                    room_scores.append((room, sim))
            room_scores.sort(key=lambda x: -x[1])
            selected_rooms = [r for r, _ in room_scores[:top_k]]
        else:
            selected_rooms = rooms[:top_k]

        # Build compressed text
        parts = []
        for room in selected_rooms:
            parts.append(room.to_compressed_repr())

        compressed_text = " ".join(parts)
        original_tokens = sum(len(t.split()) for t in texts)
        compressed_tokens = len(compressed_text.split())

        elapsed = (time.time() - start) * 1000

        # Axes summary (mean across all rooms)
        if selected_rooms:
            all_axes = np.array([list(r.axes_summary.values()) for r in selected_rooms])
            mean_axes = np.mean(all_axes, axis=0)
            axes_summary = {
                AXIS_NAMES[i]: round(float(mean_axes[i]), 3)
                for i in range(len(AXIS_NAMES))
            }
        else:
            axes_summary = {}

        return CompressedContext(
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=1.0 - (compressed_tokens / max(1, original_tokens)),
            rooms_used=len(selected_rooms),
            surfaces_used=len(surfaces),
            atoms_created=len(atoms),
            processing_time_ms=elapsed,
            axes_summary=axes_summary,
        )

    # ─── Axes-Driven Selection (Primary Value) ─────────────────────────

    def select_context(
        self,
        texts_or_atoms,
        query: Optional[str] = None,
        domain: str = "general",
        top_k: int = 5,
        mode: str = "hybrid",
        alpha: float = 0.3,
    ) -> CompressedContext:
        """
        Multi-signal context selection with configurable blend.

        Args:
            texts_or_atoms: Either List[str] (raw texts) or List[Atom] (pre-processed).
            query:  Optional query for similarity scoring.
            domain: Domain tag for processing.
            top_k:  Number of chunks to select.
            mode:   'lumisift' (pure multi-axis), 'similarity' (pure embedding),
                    or 'hybrid' (combined). Default: 'hybrid'.
            alpha:  Blend weight for hybrid mode.
                    0.0 = pure lumisift, 1.0 = pure similarity, 0.3 = recommended.
                    Only used when mode='hybrid'.

        Scoring:
            lumisift:   score = relevance * (1+|risk|) * trust * temporal * specificity_boost
            similarity: score = cosine(query_embedding, chunk_embedding)
            hybrid:     score = alpha * similarity + (1-alpha) * lumisift_normalized
        """
        start = time.time()

        # Accept both strings and pre-processed atoms
        if texts_or_atoms and hasattr(texts_or_atoms[0], 'axes'):
            atoms = texts_or_atoms
        else:
            atoms = self.process_batch(texts_or_atoms, domain=domain)

        # ─── Compute Lumisift multi-axis scores ────────────────────────
        lumi_scores = []
        for atom in atoms:
            rel = atom.axes.get("relevance", 0.5)
            risk = abs(atom.axes.get("risk", 0.0))
            trust = atom.axes.get("trust", 0.5)
            temporal = atom.axes.get("temporal", 0.0)
            specificity = atom.axes.get("specificity", 0.0)

            t_boost = 1.0 + max(0, temporal) * 0.3
            s_boost = 1.0 + specificity * 0.8

            score = rel * (1 + risk) * (0.5 + trust * 0.5) * t_boost * s_boost
            lumi_scores.append(score)

        lumi_scores = np.array(lumi_scores)

        # Normalize to 0-1
        lmin, lmax = lumi_scores.min(), lumi_scores.max()
        if lmax - lmin > 1e-8:
            lumi_norm = (lumi_scores - lmin) / (lmax - lmin)
        else:
            lumi_norm = np.ones_like(lumi_scores) * 0.5

        # ─── Compute similarity scores ─────────────────────────────────
        if query and mode in ("similarity", "hybrid"):
            query_emb = self.embedder.embed(query)
            sim_scores = np.array([
                float(np.dot(query_emb, atom.embedding) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(atom.embedding) + 1e-9
                ))
                for atom in atoms
            ])
            # Normalize to 0-1
            smin, smax = sim_scores.min(), sim_scores.max()
            if smax - smin > 1e-8:
                sim_norm = (sim_scores - smin) / (smax - smin)
            else:
                sim_norm = np.ones_like(sim_scores) * 0.5
        else:
            sim_norm = np.zeros(len(atoms))

        # ─── Compute final scores based on mode ────────────────────────
        if mode == "lumisift":
            final_scores = lumi_norm
        elif mode == "similarity":
            final_scores = sim_norm
        else:  # hybrid
            final_scores = alpha * sim_norm + (1 - alpha) * lumi_norm

        # Select top-k
        top_idx = np.argsort(final_scores)[::-1][:top_k]
        selected = [(atoms[i], float(final_scores[i])) for i in top_idx]

        # Build context from selected raw texts
        selected_texts = [a.text for a, _ in selected]
        compressed_text = "\n\n".join(selected_texts)

        original_tokens = sum(a.original_tokens for a in atoms)
        selected_tokens = sum(a.original_tokens for a, _ in selected)

        # Build axes summary of selected atoms
        if selected:
            all_axes = np.array([list(a.axes.values()) for a, _ in selected])
            mean_axes = np.mean(all_axes, axis=0)
            axes_summary = {
                AXIS_NAMES[i]: round(float(mean_axes[i]), 3)
                for i in range(len(AXIS_NAMES))
            }
        else:
            axes_summary = {}

        elapsed = (time.time() - start) * 1000

        return CompressedContext(
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=selected_tokens,
            compression_ratio=1.0 - (selected_tokens / max(1, original_tokens)),
            rooms_used=0,
            surfaces_used=0,
            atoms_created=len(atoms),
            processing_time_ms=elapsed,
            axes_summary=axes_summary,
        )

    # ─── Export ─────────────────────────────────────────────────────────

    def export_atoms(self, atoms: List[Atom]) -> List[Dict[str, Any]]:
        """Export atoms as JSON-serializable dicts."""
        return [a.to_dict() for a in atoms]

    # ─── Statistics ─────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get cumulative pipeline statistics."""
        avg_time = (
            float(np.mean(self._processing_times))
            if self._processing_times else 0.0
        )
        total_orig = self._stats.total_original_tokens
        total_comp = self._stats.total_compressed_tokens

        return {
            "total_processed": self._stats.total_processed,
            "total_atoms": self._stats.total_atoms_created,
            "total_surfaces": self._stats.total_surfaces_built,
            "total_rooms": self._stats.total_rooms_built,
            "total_original_tokens": total_orig,
            "total_compressed_tokens": total_comp,
            "overall_compression_ratio": round(
                1.0 - (total_comp / max(1, total_orig)), 3
            ),
            "avg_processing_time_ms": round(avg_time, 2),
        }

    def reset(self):
        """Reset pipeline state (rooms, stats)."""
        self.projection_engine = ContextProjectionEngine(
            distance_threshold=self.projection_engine.distance_threshold,
        )
        self._stats = PipelineStats()
        self._processing_times = []

    # ─── Internal ───────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Pipeline] {msg}")
