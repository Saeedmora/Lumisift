"""
Atom Store — Persistent Local Database for Logical Rooms
=========================================================
Project-based storage for processed atoms with vector search.

Storage layout:
  data/projects/{project_name}/
    atoms.jsonl           # Full atom data (text, axes, metadata)
    embeddings.npy        # Embedding matrix for vector search
    calibration.json      # Project-specific axis calibration
    project.json          # Project metadata (created, stats)

Usage:
    store = AtomStore()
    store.create_project("protein_engineering")
    store.set_active("protein_engineering")
    store.save_atoms(atoms)
    results = store.search("directed evolution", top_k=5)
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "projects")


@dataclass
class StoredAtom:
    """Serializable atom record for persistent storage."""
    id: str
    text: str
    axes: Dict[str, float]
    domain: str = "general"
    category: str = "unknown"
    tension: float = 0.0
    confidence: float = 1.0
    original_tokens: int = 0
    compressed_tokens: int = 0
    source: str = ""          # e.g. "pubmed:40628259" or "manual"
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "axes": self.axes,
            "domain": self.domain,
            "category": self.category,
            "tension": round(self.tension, 4),
            "confidence": round(self.confidence, 4),
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "source": self.source,
            "created_at": self.created_at or datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StoredAtom":
        return cls(
            id=d["id"],
            text=d["text"],
            axes=d.get("axes", {}),
            domain=d.get("domain", "general"),
            category=d.get("category", "unknown"),
            tension=d.get("tension", 0.0),
            confidence=d.get("confidence", 1.0),
            original_tokens=d.get("original_tokens", 0),
            compressed_tokens=d.get("compressed_tokens", 0),
            source=d.get("source", ""),
            created_at=d.get("created_at", ""),
        )

    @classmethod
    def from_atom(cls, atom, source: str = "manual") -> "StoredAtom":
        """Convert a pipeline Atom to StoredAtom."""
        cat = atom.ontology_category.value if atom.ontology_category else "unknown"
        return cls(
            id=atom.id,
            text=atom.text,
            axes={k: round(v, 4) for k, v in atom.axes.items()},
            domain=atom.domain,
            category=cat,
            tension=atom.tension,
            confidence=atom.confidence,
            original_tokens=atom.original_tokens,
            compressed_tokens=atom.compressed_tokens,
            source=source,
            created_at=datetime.now().isoformat(),
        )


class AtomStore:
    """
    Persistent local database for processed atoms.

    Features:
    - Project-based organization
    - JSONL atom storage (human-readable, git-friendly)
    - Numpy embedding matrix for fast vector search
    - Auto-save on every operation
    - Crash-safe writes (atomic for critical files)
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or DATA_DIR
        self.active_project: Optional[str] = None
        self._atoms: List[StoredAtom] = []
        self._embeddings: Optional[np.ndarray] = None
        self._dirty = False

        os.makedirs(self.data_dir, exist_ok=True)

    # ─── Project Management ────────────────────────────────────────────

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects with metadata."""
        projects = []
        if not os.path.exists(self.data_dir):
            return projects

        for name in sorted(os.listdir(self.data_dir)):
            proj_dir = os.path.join(self.data_dir, name)
            if not os.path.isdir(proj_dir):
                continue

            meta_path = os.path.join(proj_dir, "project.json")
            meta = {"name": name, "atoms": 0, "created": ""}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta.update(json.load(f))
                except Exception:
                    pass

            # Count atoms
            atoms_path = os.path.join(proj_dir, "atoms.jsonl")
            if os.path.exists(atoms_path):
                with open(atoms_path, "r", encoding="utf-8") as f:
                    meta["atoms"] = sum(1 for line in f if line.strip())

            meta["name"] = name
            projects.append(meta)

        return projects

    def create_project(self, name: str, domain: str = "general") -> Dict[str, Any]:
        """Create a new project directory."""
        # Sanitize name
        safe_name = "".join(c for c in name if c.isalnum() or c in "-_").strip()
        if not safe_name:
            safe_name = "default"

        proj_dir = os.path.join(self.data_dir, safe_name)
        os.makedirs(proj_dir, exist_ok=True)

        meta = {
            "name": safe_name,
            "domain": domain,
            "created": datetime.now().isoformat(),
            "atoms": 0,
        }

        meta_path = os.path.join(proj_dir, "project.json")
        if not os.path.exists(meta_path):
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        return meta

    def set_active(self, name: str) -> bool:
        """Switch to a project, loading its atoms."""
        proj_dir = os.path.join(self.data_dir, name)
        if not os.path.isdir(proj_dir):
            return False

        # Save current project first
        if self.active_project and self._dirty:
            self._save()

        self.active_project = name
        self._atoms = []
        self._embeddings = None
        self._dirty = False

        # Load atoms
        atoms_path = os.path.join(proj_dir, "atoms.jsonl")
        if os.path.exists(atoms_path):
            with open(atoms_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._atoms.append(StoredAtom.from_dict(json.loads(line)))
                        except Exception:
                            continue

        # Load embeddings
        emb_path = os.path.join(proj_dir, "embeddings.npy")
        if os.path.exists(emb_path):
            try:
                self._embeddings = np.load(emb_path)
            except Exception:
                self._embeddings = None

        return True

    def delete_project(self, name: str) -> bool:
        """Delete a project and all its data."""
        import shutil
        proj_dir = os.path.join(self.data_dir, name)
        if os.path.isdir(proj_dir):
            shutil.rmtree(proj_dir)
            if self.active_project == name:
                self.active_project = None
                self._atoms = []
                self._embeddings = None
            return True
        return False

    # ─── Atom Operations ───────────────────────────────────────────────

    def save_atoms(self, atoms, embeddings: np.ndarray = None, source: str = "manual"):
        """
        Save pipeline Atom objects to the active project.

        Args:
            atoms: List of pipeline Atom objects
            embeddings: Optional numpy array of embeddings (len(atoms) × dim)
            source: Source identifier (e.g. "pubmed:12345", "manual", "batch")
        """
        if not self.active_project:
            self.create_project("default")
            self.set_active("default")

        for i, atom in enumerate(atoms):
            stored = StoredAtom.from_atom(atom, source=source)
            self._atoms.append(stored)

        # Handle embeddings
        if embeddings is not None and len(embeddings) > 0:
            if self._embeddings is None:
                self._embeddings = embeddings.astype(np.float32)
            else:
                self._embeddings = np.vstack([
                    self._embeddings,
                    embeddings.astype(np.float32),
                ])
        elif atoms and hasattr(atoms[0], 'embedding') and atoms[0].embedding is not None:
            # Extract embeddings from atoms
            new_emb = np.array([a.embedding for a in atoms], dtype=np.float32)
            if self._embeddings is None:
                self._embeddings = new_emb
            else:
                self._embeddings = np.vstack([self._embeddings, new_emb])

        self._dirty = True
        self._save()

    def get_atoms(self, offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """Return paginated atom list."""
        end = min(offset + limit, len(self._atoms))
        return [a.to_dict() for a in self._atoms[offset:end]]

    def get_atom_count(self) -> int:
        """Return total atom count for active project."""
        return len(self._atoms)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Vector similarity search across stored atoms.

        Returns top-k atoms ranked by cosine similarity.
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        # Normalize
        query_norm = query_embedding.flatten().astype(np.float32)
        qn = np.linalg.norm(query_norm)
        if qn > 0:
            query_norm = query_norm / qn

        # Compute cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        normalized = self._embeddings / norms

        similarities = normalized @ query_norm
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if idx < len(self._atoms):
                atom_dict = self._atoms[idx].to_dict()
                atom_dict["similarity"] = round(float(similarities[idx]), 4)
                results.append(atom_dict)

        return results

    # ─── Export ─────────────────────────────────────────────────────────

    def export_training_jsonl(self, path: str = None) -> str:
        """Export all atoms as JSONL training data."""
        if path is None:
            proj_dir = os.path.join(self.data_dir, self.active_project or "default")
            path = os.path.join(proj_dir, "training_export.jsonl")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for atom in self._atoms:
                entry = {
                    "text": atom.text,
                    "axes": atom.axes,
                    "domain": atom.domain,
                    "category": atom.category,
                    "tension": atom.tension,
                    "source": atom.source,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return path

    def export_huggingface(self, path: str = None) -> str:
        """Export in HuggingFace datasets format (JSON with 'train' split)."""
        if path is None:
            proj_dir = os.path.join(self.data_dir, self.active_project or "default")
            path = os.path.join(proj_dir, "hf_dataset.json")

        data = {
            "version": "1.0",
            "data": [
                {
                    "text": a.text,
                    "label": a.category,
                    "temporal": a.axes.get("temporal", 0),
                    "relevance": a.axes.get("relevance", 0),
                    "risk": a.axes.get("risk", 0),
                    "ontology": a.axes.get("ontology", 0),
                    "causality": a.axes.get("causality", 0),
                    "visibility": a.axes.get("visibility", 0),
                    "trust": a.axes.get("trust", 0),
                }
                for a in self._atoms
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return path

    def export_openai_finetune(self, path: str = None) -> str:
        """Export in OpenAI fine-tuning format (chat completions JSONL)."""
        if path is None:
            proj_dir = os.path.join(self.data_dir, self.active_project or "default")
            path = os.path.join(proj_dir, "openai_finetune.jsonl")

        with open(path, "w", encoding="utf-8") as f:
            for atom in self._atoms:
                axes_str = ", ".join(f"{k}: {v:+.2f}" for k, v in atom.axes.items())
                entry = {
                    "messages": [
                        {"role": "system", "content": "You are a scientific text analyzer. Evaluate the text across 7 semantic axes: temporal, relevance, risk, ontology, causality, visibility, trust. Return scores for each axis."},
                        {"role": "user", "content": atom.text},
                        {"role": "assistant", "content": f"Axes: {axes_str}\nCategory: {atom.category}\nTension: {atom.tension:.4f}"},
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return path

    # ─── Stats ─────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for the active project."""
        if not self._atoms:
            return {"atoms": 0, "project": self.active_project}

        domains = {}
        categories = {}
        for a in self._atoms:
            domains[a.domain] = domains.get(a.domain, 0) + 1
            categories[a.category] = categories.get(a.category, 0) + 1

        axes_means = {}
        axes_keys = ["temporal", "relevance", "risk", "ontology", "causality", "visibility", "trust"]
        for k in axes_keys:
            values = [a.axes.get(k, 0) for a in self._atoms]
            axes_means[k] = round(float(np.mean(values)), 4)

        return {
            "project": self.active_project,
            "atoms": len(self._atoms),
            "has_embeddings": self._embeddings is not None,
            "embedding_count": len(self._embeddings) if self._embeddings is not None else 0,
            "domains": domains,
            "categories": categories,
            "axes_means": axes_means,
        }

    # ─── Internal ──────────────────────────────────────────────────────

    def _save(self):
        """Persist atoms and embeddings to disk."""
        if not self.active_project:
            return

        proj_dir = os.path.join(self.data_dir, self.active_project)
        os.makedirs(proj_dir, exist_ok=True)

        # Save atoms as JSONL (atomic write)
        atoms_path = os.path.join(proj_dir, "atoms.jsonl")
        tmp_path = atoms_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for atom in self._atoms:
                f.write(json.dumps(atom.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, atoms_path)

        # Save embeddings
        if self._embeddings is not None:
            emb_path = os.path.join(proj_dir, "embeddings.npy")
            np.save(emb_path, self._embeddings)

        # Update project metadata
        meta_path = os.path.join(proj_dir, "project.json")
        meta = {"name": self.active_project, "atoms": len(self._atoms)}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta.update(json.load(f))
            except Exception:
                pass
        meta["atoms"] = len(self._atoms)
        meta["updated"] = datetime.now().isoformat()

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self._dirty = False
