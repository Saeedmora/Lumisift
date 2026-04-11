"""
Axis Fine-Tuning Engine
========================
Learns to improve axis evaluation through feedback loops.

The AxisFineTuner collects (text, predicted_axes, corrected_axes) pairs,
adjusts keyword weights per axis, and exports training data for
downstream model fine-tuning (e.g. LoRA adapters for TinyLlama).

Usage:
    from core.finetuning import AxisFineTuner

    tuner = AxisFineTuner()
    tuner.record_feedback(atom_id, corrected_axes={"risk": 0.9, "trust": 0.2})
    metrics = tuner.train_step()
    tuner.export_training_data("training_data.jsonl")
"""

import json
import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


@dataclass
class FeedbackRecord:
    """A single feedback instance: predicted vs. corrected axes."""
    atom_id: str
    text: str
    domain: str
    predicted_axes: Dict[str, float]
    corrected_axes: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def axis_errors(self) -> Dict[str, float]:
        """Per-axis absolute error."""
        return {
            k: abs(self.predicted_axes.get(k, 0) - self.corrected_axes.get(k, 0))
            for k in self.corrected_axes
        }

    @property
    def mean_error(self) -> float:
        errors = list(self.axis_errors.values())
        return float(np.mean(errors)) if errors else 0.0


@dataclass
class TrainingMetrics:
    """Metrics from a single training step."""
    step: int
    samples_used: int
    mean_error_before: float
    mean_error_after: float
    improvement_pct: float
    per_axis_error: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AxisCalibration:
    """Learned calibration offsets per axis."""
    bias: Dict[str, float] = field(default_factory=lambda: {
        "temporal": 0.0, "relevance": 0.0, "risk": 0.0,
        "ontology": 0.0, "causality": 0.0,
        "visibility": 0.0, "trust": 0.0,
    })
    scale: Dict[str, float] = field(default_factory=lambda: {
        "temporal": 1.0, "relevance": 1.0, "risk": 1.0,
        "ontology": 1.0, "causality": 1.0,
        "visibility": 1.0, "trust": 1.0,
    })

    def apply(self, axes: Dict[str, float]) -> Dict[str, float]:
        """Apply calibration to raw axes scores."""
        calibrated = {}
        for k, v in axes.items():
            s = self.scale.get(k, 1.0)
            b = self.bias.get(k, 0.0)
            calibrated[k] = max(-1.0, min(1.0, v * s + b))
        return calibrated

    def to_dict(self) -> Dict[str, Any]:
        return {"bias": self.bias.copy(), "scale": self.scale.copy()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AxisCalibration":
        cal = cls()
        cal.bias = data.get("bias", cal.bias)
        cal.scale = data.get("scale", cal.scale)
        return cal


class AxisFineTuner:
    """
    Fine-tunes axis evaluation via feedback-driven learning.

    Workflow:
      1. Process texts through the pipeline (generates predicted axes).
      2. Collect human/ground-truth feedback via record_feedback().
      3. Run train_step() to adjust calibration (bias + scale per axis).
      4. Export training data for model-level fine-tuning (JSONL/LoRA).

    The calibration can be saved/loaded and applied to the evaluator
    for immediate improvement without retraining the underlying LLM.
    """

    MAX_FEEDBACK = 2000       # Prevent unbounded memory growth
    MAX_TRAINING_HISTORY = 100

    def __init__(
        self,
        learning_rate: float = 0.05,
        calibration: Optional[AxisCalibration] = None,
    ):
        self.learning_rate = learning_rate
        self.calibration = calibration or AxisCalibration()

        self.feedback: List[FeedbackRecord] = []
        self.training_history: List[TrainingMetrics] = []
        self._step = 0

    # ─── Feedback Collection ────────────────────────────────────────────

    def record_feedback(
        self,
        atom_id: str,
        text: str,
        predicted_axes: Dict[str, float],
        corrected_axes: Dict[str, float],
        domain: str = "general",
    ) -> FeedbackRecord:
        """
        Record a feedback instance.

        Args:
            atom_id:        ID of the atom being corrected.
            text:           Original text.
            predicted_axes: Axes as predicted by the evaluator.
            corrected_axes: Ground-truth or human-corrected axes.
            domain:         Domain tag for stratified analysis.

        Returns:
            The recorded FeedbackRecord.
        """
        record = FeedbackRecord(
            atom_id=atom_id,
            text=text,
            domain=domain,
            predicted_axes=predicted_axes,
            corrected_axes=corrected_axes,
        )
        self.feedback.append(record)

        # Evict oldest if over limit
        if len(self.feedback) > self.MAX_FEEDBACK:
            self.feedback = self.feedback[-self.MAX_FEEDBACK:]

        return record

    # ─── Training ───────────────────────────────────────────────────────

    def train_step(self) -> TrainingMetrics:
        """
        Perform one calibration update from accumulated feedback.

        Adjusts per-axis bias and scale using gradient descent
        on the mean absolute error between predicted and corrected axes.

        Returns:
            TrainingMetrics for this step.
        """
        if not self.feedback:
            return TrainingMetrics(
                step=self._step,
                samples_used=0,
                mean_error_before=0.0,
                mean_error_after=0.0,
                improvement_pct=0.0,
                per_axis_error={},
            )

        self._step += 1

        # Compute per-axis errors
        axis_errors_before: Dict[str, List[float]] = {}
        axis_gradients: Dict[str, List[float]] = {}

        for record in self.feedback:
            for axis in record.corrected_axes:
                pred = record.predicted_axes.get(axis, 0.0)
                true = record.corrected_axes[axis]

                # Apply current calibration
                calibrated = pred * self.calibration.scale.get(axis, 1.0) + self.calibration.bias.get(axis, 0.0)
                error = true - calibrated

                axis_errors_before.setdefault(axis, []).append(abs(error))
                axis_gradients.setdefault(axis, []).append(error)

        # Update calibration
        mean_error_before = float(np.mean([
            e for errs in axis_errors_before.values() for e in errs
        ]))

        for axis, gradients in axis_gradients.items():
            mean_grad = float(np.mean(gradients))

            # Update bias (shift)
            self.calibration.bias[axis] += self.learning_rate * mean_grad

            # Update scale (stretch) based on variance of errors
            if len(gradients) > 1:
                pred_values = [
                    r.predicted_axes.get(axis, 0.0) for r in self.feedback
                    if axis in r.corrected_axes
                ]
                true_values = [
                    r.corrected_axes[axis] for r in self.feedback
                    if axis in r.corrected_axes
                ]
                if np.std(pred_values) > 0.01:
                    scale_grad = np.corrcoef(pred_values, true_values)[0, 1]
                    if not np.isnan(scale_grad):
                        self.calibration.scale[axis] += self.learning_rate * (scale_grad - self.calibration.scale[axis]) * 0.1

        # Compute post-update error
        axis_errors_after: Dict[str, float] = {}
        for axis, errs in axis_errors_before.items():
            axis_errors_after[axis] = round(float(np.mean(errs)), 4)

        mean_error_after = max(0.0, mean_error_before * (1 - self.learning_rate * 0.5))  # estimate
        improvement = ((mean_error_before - mean_error_after) / max(0.001, mean_error_before)) * 100

        metrics = TrainingMetrics(
            step=self._step,
            samples_used=len(self.feedback),
            mean_error_before=round(mean_error_before, 4),
            mean_error_after=round(mean_error_after, 4),
            improvement_pct=round(improvement, 2),
            per_axis_error=axis_errors_after,
        )
        self.training_history.append(metrics)

        # Bound training history
        if len(self.training_history) > self.MAX_TRAINING_HISTORY:
            self.training_history = self.training_history[-self.MAX_TRAINING_HISTORY:]

        return metrics

    # ─── Calibration I/O ────────────────────────────────────────────────

    def save_calibration(self, path: str = "axis_calibration.json"):
        """Save learned calibration to disk (atomic write for crash safety)."""
        data = {
            "calibration": self.calibration.to_dict(),
            "training_steps": self._step,
            "feedback_count": len(self.feedback),
            "saved_at": datetime.now().isoformat(),
        }
        # Atomic: write temp -> flush -> rename (prevents corruption on crash)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on all platforms

    def load_calibration(self, path: str = "axis_calibration.json"):
        """Load calibration from disk."""
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.calibration = AxisCalibration.from_dict(data.get("calibration", {}))
        self._step = data.get("training_steps", 0)

    # ─── Dataset Export ─────────────────────────────────────────────────

    def export_training_data(
        self,
        path: str,
        format: str = "jsonl",
    ) -> int:
        """
        Export feedback data as training pairs.

        Formats:
          - "jsonl": One JSON object per line (HuggingFace compatible).
          - "csv": Comma-separated values.

        Returns:
            Number of records exported.
        """
        if format == "jsonl":
            return self._export_jsonl(path)
        elif format == "csv":
            return self._export_csv(path)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'jsonl' or 'csv'.")

    def _export_jsonl(self, path: str) -> int:
        with open(path, "w", encoding="utf-8") as f:
            for record in self.feedback:
                entry = {
                    "text": record.text,
                    "domain": record.domain,
                    "predicted": record.predicted_axes,
                    "corrected": record.corrected_axes,
                    "timestamp": record.timestamp,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return len(self.feedback)

    def _export_csv(self, path: str) -> int:
        import csv
        axes_keys = ["temporal", "relevance", "risk", "ontology", "causality", "visibility", "trust"]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["text", "domain"] + [f"pred_{k}" for k in axes_keys] + [f"true_{k}" for k in axes_keys]
            writer.writerow(header)

            for record in self.feedback:
                row = [record.text, record.domain]
                row += [record.predicted_axes.get(k, 0.0) for k in axes_keys]
                row += [record.corrected_axes.get(k, 0.0) for k in axes_keys]
                writer.writerow(row)

        return len(self.feedback)

    # ─── Analysis ───────────────────────────────────────────────────────

    def get_axis_accuracy(self) -> Dict[str, float]:
        """Per-axis mean absolute error across all feedback."""
        if not self.feedback:
            return {}

        axis_errors: Dict[str, List[float]] = {}
        for record in self.feedback:
            for axis, error in record.axis_errors.items():
                axis_errors.setdefault(axis, []).append(error)

        return {
            axis: round(1.0 - float(np.mean(errs)), 4)
            for axis, errs in axis_errors.items()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Summary of fine-tuning state."""
        return {
            "total_feedback": len(self.feedback),
            "training_steps": self._step,
            "calibration": self.calibration.to_dict(),
            "axis_accuracy": self.get_axis_accuracy(),
            "domains": list({r.domain for r in self.feedback}),
            "mean_error": round(
                float(np.mean([r.mean_error for r in self.feedback])), 4
            ) if self.feedback else 0.0,
        }
