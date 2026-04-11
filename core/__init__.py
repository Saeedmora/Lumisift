"""
Logical Rooms — Core Package
==============================
Public API exports for the Logical Rooms framework.
"""

from core.atom import (
    Atom,
    OntologyCategory,
    AXIS_NAMES,
    DEFAULT_AXES,
    merge_atoms,
    calculate_atom_statistics,
)
from core.surface import Surface, AssociationGraph
from core.models import LogicalRoom
from core.pipeline import LogicalRoomsPipeline, CompressedContext
from core.finetuning import AxisFineTuner, AxisCalibration
from core.dataset import LogicalRoomsDataset, DataSample
from core.axes_evaluator import SevenAxesEvaluator
from core.embeddings import EmbeddingService
from core.projection_engine import ContextProjectionEngine
from core.self_optimization import SelfOptimizer

__all__ = [
    # Data model
    "Atom",
    "Surface",
    "LogicalRoom",
    "OntologyCategory",
    "AXIS_NAMES",
    "DEFAULT_AXES",
    # Pipeline
    "LogicalRoomsPipeline",
    "CompressedContext",
    # Fine-tuning
    "AxisFineTuner",
    "AxisCalibration",
    # Dataset
    "LogicalRoomsDataset",
    "DataSample",
    # Evaluation
    "SevenAxesEvaluator",
    # Services
    "EmbeddingService",
    "ContextProjectionEngine",
    "SelfOptimizer",
    "AssociationGraph",
    # Utilities
    "merge_atoms",
    "calculate_atom_statistics",
]
