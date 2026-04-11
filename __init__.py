"""
Logical Rooms
==============
Semantic compression framework for LLM context optimization.
"""

__version__ = "1.0.0"
__author__ = "Logical Rooms Contributors"

from core.atom import Atom, OntologyCategory, AXIS_NAMES
from core.surface import Surface
from core.models import LogicalRoom
from core.pipeline import LogicalRoomsPipeline, CompressedContext
from core.finetuning import AxisFineTuner, AxisCalibration
from core.dataset import LogicalRoomsDataset

__all__ = [
    "Atom",
    "Surface",
    "LogicalRoom",
    "LogicalRoomsPipeline",
    "CompressedContext",
    "AxisFineTuner",
    "AxisCalibration",
    "LogicalRoomsDataset",
    "OntologyCategory",
    "AXIS_NAMES",
]
