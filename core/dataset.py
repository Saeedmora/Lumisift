"""
Dataset Management for Logical Rooms
======================================
Load, manage, split, and export datasets for fine-tuning
the axes evaluator and the broader Logical Rooms pipeline.

Supported formats:
  - JSONL (one JSON object per line — HuggingFace compatible)
  - CSV
  - In-memory Python lists

Usage:
    from core.dataset import LogicalRoomsDataset

    ds = LogicalRoomsDataset()
    ds.load("data/security_samples.jsonl")
    ds.add_sample("New threat detected", axes={"risk": 0.9}, domain="security")
    train, val, test = ds.split()
    ds.export("output/train.jsonl")
"""

import json
import os
import random
import csv
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


@dataclass
class DataSample:
    """A single labeled sample for training/evaluation."""
    text: str
    axes: Dict[str, float]
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "axes": self.axes.copy(),
            "domain": self.domain,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSample":
        return cls(
            text=data["text"],
            axes=data.get("axes", {}),
            domain=data.get("domain", "general"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DatasetStats:
    """Statistics about a dataset."""
    total_samples: int = 0
    domains: Dict[str, int] = field(default_factory=dict)
    avg_text_length: float = 0.0
    avg_axes: Dict[str, float] = field(default_factory=dict)
    axes_std: Dict[str, float] = field(default_factory=dict)


class LogicalRoomsDataset:
    """
    Dataset manager for Logical Rooms training data.

    Handles:
      - Loading from JSONL, CSV, or in-memory lists
      - Domain-stratified splitting (train/val/test)
      - Statistics computation
      - Export for fine-tuning pipelines
    """

    def __init__(self, name: str = "logical_rooms_dataset"):
        self.name = name
        self.samples: List[DataSample] = []
        self._created_at = datetime.now().isoformat()

    # ─── Loading ────────────────────────────────────────────────────────

    def load(self, path: str) -> int:
        """
        Load samples from a file. Auto-detects format by extension.

        Returns:
            Number of samples loaded.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        ext = os.path.splitext(path)[1].lower()

        if ext in (".jsonl", ".json"):
            return self._load_jsonl(path)
        elif ext == ".csv":
            return self._load_csv(path)
        else:
            raise ValueError(f"Unsupported format: {ext}. Use .jsonl or .csv")

    def _load_jsonl(self, path: str) -> int:
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Support both "axes" and "corrected" keys (from finetuning export)
                axes = data.get("axes") or data.get("corrected", {})
                sample = DataSample(
                    text=data["text"],
                    axes=axes,
                    domain=data.get("domain", "general"),
                    metadata=data.get("metadata", {}),
                )
                self.samples.append(sample)
                count += 1
        return count

    def _load_csv(self, path: str) -> int:
        axes_keys = ["temporal", "relevance", "risk", "ontology", "causality", "visibility", "trust"]
        count = 0

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                axes = {}
                for k in axes_keys:
                    if k in row:
                        try:
                            axes[k] = float(row[k])
                        except (ValueError, TypeError):
                            pass
                    # Also check "true_*" prefix (from finetuning export)
                    elif f"true_{k}" in row:
                        try:
                            axes[k] = float(row[f"true_{k}"])
                        except (ValueError, TypeError):
                            pass

                sample = DataSample(
                    text=row.get("text", ""),
                    axes=axes,
                    domain=row.get("domain", "general"),
                )
                self.samples.append(sample)
                count += 1
        return count

    # ─── Sample Management ──────────────────────────────────────────────

    def add_sample(
        self,
        text: str,
        axes: Dict[str, float],
        domain: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a single labeled sample."""
        self.samples.append(DataSample(
            text=text,
            axes=axes,
            domain=domain,
            metadata=metadata or {},
        ))

    def add_samples(self, samples: List[Dict[str, Any]]):
        """Add multiple samples from dictionaries."""
        for s in samples:
            self.add_sample(
                text=s["text"],
                axes=s.get("axes", {}),
                domain=s.get("domain", "general"),
                metadata=s.get("metadata", {}),
            )

    # ─── Splitting ──────────────────────────────────────────────────────

    def split(
        self,
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        seed: int = 42,
        stratify_by_domain: bool = True,
    ) -> Tuple["LogicalRoomsDataset", "LogicalRoomsDataset", "LogicalRoomsDataset"]:
        """
        Split into train/val/test datasets.

        Args:
            train: Fraction for training.
            val:   Fraction for validation.
            test:  Fraction for testing.
            seed:  Random seed for reproducibility.
            stratify_by_domain: Maintain domain proportions in each split.

        Returns:
            (train_ds, val_ds, test_ds) tuple.
        """
        assert abs(train + val + test - 1.0) < 0.01, "Splits must sum to 1.0"

        rng = random.Random(seed)

        if stratify_by_domain:
            return self._stratified_split(train, val, test, rng)
        else:
            return self._random_split(train, val, test, rng)

    def _random_split(
        self, train_f: float, val_f: float, test_f: float, rng: random.Random,
    ) -> Tuple["LogicalRoomsDataset", "LogicalRoomsDataset", "LogicalRoomsDataset"]:
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(n * train_f)
        n_val = int(n * val_f)

        train_ds = LogicalRoomsDataset(name=f"{self.name}_train")
        val_ds = LogicalRoomsDataset(name=f"{self.name}_val")
        test_ds = LogicalRoomsDataset(name=f"{self.name}_test")

        train_ds.samples = [self.samples[i] for i in indices[:n_train]]
        val_ds.samples = [self.samples[i] for i in indices[n_train:n_train + n_val]]
        test_ds.samples = [self.samples[i] for i in indices[n_train + n_val:]]

        return train_ds, val_ds, test_ds

    def _stratified_split(
        self, train_f: float, val_f: float, test_f: float, rng: random.Random,
    ) -> Tuple["LogicalRoomsDataset", "LogicalRoomsDataset", "LogicalRoomsDataset"]:
        # Group by domain
        domain_groups: Dict[str, List[int]] = {}
        for i, sample in enumerate(self.samples):
            domain_groups.setdefault(sample.domain, []).append(i)

        train_indices, val_indices, test_indices = [], [], []

        for domain, indices in domain_groups.items():
            rng.shuffle(indices)
            n = len(indices)
            n_train = max(1, int(n * train_f))
            n_val = max(0, int(n * val_f))

            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])

        train_ds = LogicalRoomsDataset(name=f"{self.name}_train")
        val_ds = LogicalRoomsDataset(name=f"{self.name}_val")
        test_ds = LogicalRoomsDataset(name=f"{self.name}_test")

        train_ds.samples = [self.samples[i] for i in train_indices]
        val_ds.samples = [self.samples[i] for i in val_indices]
        test_ds.samples = [self.samples[i] for i in test_indices]

        return train_ds, val_ds, test_ds

    # ─── Export ─────────────────────────────────────────────────────────

    def export(self, path: str, format: str = "jsonl") -> int:
        """
        Export dataset to file.

        Returns:
            Number of samples exported.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        if format == "jsonl":
            return self._export_jsonl(path)
        elif format == "csv":
            return self._export_csv(path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_jsonl(self, path: str) -> int:
        with open(path, "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        return len(self.samples)

    def _export_csv(self, path: str) -> int:
        axes_keys = ["temporal", "relevance", "risk", "ontology", "causality", "visibility", "trust"]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "domain"] + axes_keys)
            for sample in self.samples:
                row = [sample.text, sample.domain]
                row += [sample.axes.get(k, 0.0) for k in axes_keys]
                writer.writerow(row)
        return len(self.samples)

    # ─── Statistics ─────────────────────────────────────────────────────

    def get_statistics(self) -> DatasetStats:
        """Compute dataset statistics."""
        if not self.samples:
            return DatasetStats()

        # Domain distribution
        domains: Dict[str, int] = {}
        for s in self.samples:
            domains[s.domain] = domains.get(s.domain, 0) + 1

        # Text length
        lengths = [len(s.text.split()) for s in self.samples]

        # Axes means and stds
        axes_keys = ["temporal", "relevance", "risk", "ontology", "causality", "visibility", "trust"]
        axes_values = {k: [] for k in axes_keys}

        for s in self.samples:
            for k in axes_keys:
                if k in s.axes:
                    axes_values[k].append(s.axes[k])

        avg_axes = {
            k: round(float(np.mean(v)), 4) if v else 0.0
            for k, v in axes_values.items()
        }
        axes_std = {
            k: round(float(np.std(v)), 4) if v else 0.0
            for k, v in axes_values.items()
        }

        return DatasetStats(
            total_samples=len(self.samples),
            domains=domains,
            avg_text_length=round(float(np.mean(lengths)), 1),
            avg_axes=avg_axes,
            axes_std=axes_std,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return f"LogicalRoomsDataset(name='{self.name}', samples={stats.total_samples}, domains={list(stats.domains.keys())})"
