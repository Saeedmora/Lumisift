"""
Context Projection Engine
==========================
Projects Atoms onto existing Rooms or creates new ones,
using weighted embedding + axes distance.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from core.atom import Atom
from core.models import LogicalRoom


@dataclass
class ProjectionResult:
    """Result of projecting an atom into the room space."""
    room: LogicalRoom
    is_new_room: bool
    distance: float
    embedding_distance: float
    axes_distance: float
    tension: float


class ContextProjectionEngine:
    """
    Projects atoms onto existing rooms or creates new ones.
    Uses weighted distance: D = λ·d_embed + (1−λ)·d_axes.
    """

    def __init__(
        self,
        rooms: Optional[List[LogicalRoom]] = None,
        distance_threshold: float = 0.5,
        lambda_weight: float = 0.6,
        alpha: float = 0.1,
    ):
        self.rooms = rooms or []
        self.distance_threshold = distance_threshold
        self.lambda_weight = lambda_weight
        self.alpha = alpha

    def project(self, obj: Atom) -> ProjectionResult:
        """Project an atom onto the room space."""
        if not self.rooms:
            new_room = self._create_room(obj)
            return ProjectionResult(
                room=new_room, is_new_room=True,
                distance=0.0, embedding_distance=0.0,
                axes_distance=0.0, tension=new_room.tension,
            )

        best_room = None
        min_distance = float("inf")
        best_d_s = 0.0
        best_d_a = 0.0

        for room in self.rooms:
            if room.centroid is None:
                continue
            d_s = float(np.linalg.norm(obj.embedding - room.centroid))
            d_a = float(np.linalg.norm(obj.axes_vector - room.mean_axes))
            D = self.lambda_weight * d_s + (1 - self.lambda_weight) * d_a

            if D < min_distance:
                min_distance = D
                best_room = room
                best_d_s = d_s
                best_d_a = d_a

        if min_distance > self.distance_threshold:
            new_room = self._create_room(obj)
            return ProjectionResult(
                room=new_room, is_new_room=True,
                distance=min_distance, embedding_distance=best_d_s,
                axes_distance=best_d_a, tension=new_room.tension,
            )
        else:
            best_room.add_object(obj)
            return ProjectionResult(
                room=best_room, is_new_room=False,
                distance=min_distance, embedding_distance=best_d_s,
                axes_distance=best_d_a, tension=best_room.tension,
            )

    def _create_room(self, obj: Atom) -> LogicalRoom:
        room = LogicalRoom(name=f"Room-{len(self.rooms) + 1}")
        room.alpha = self.alpha
        room.add_object(obj)
        self.rooms.append(room)
        return room

    def update_room_ema(self, room: LogicalRoom, obj: Atom):
        room.update(obj.embedding, obj.axes_vector, obj.axes)

    def get_triggered_rooms(self) -> List[LogicalRoom]:
        return [r for r in self.rooms if r.should_trigger_review]

    def get_splittable_rooms(self, variance_threshold: float = 0.5) -> List[LogicalRoom]:
        return [r for r in self.rooms if r.should_split(variance_threshold)]

    def find_similar_rooms(self, obj: Atom, k: int = 3) -> List[Tuple[LogicalRoom, float]]:
        if not self.rooms:
            return []
        distances = []
        for room in self.rooms:
            if room.centroid is None:
                continue
            d_s = float(np.linalg.norm(obj.embedding - room.centroid))
            d_a = float(np.linalg.norm(obj.axes_vector - room.mean_axes))
            D = self.lambda_weight * d_s + (1 - self.lambda_weight) * d_a
            distances.append((room, D))
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def get_room_stats(self) -> dict:
        if not self.rooms:
            return {"total_rooms": 0, "total_objects": 0}

        total_objects = sum(len(r.objects) for r in self.rooms)
        avg_tension = float(np.mean([r.tension for r in self.rooms]))
        triggered = len(self.get_triggered_rooms())

        all_axes = np.array([r.mean_axes for r in self.rooms])
        axis_means = np.mean(all_axes, axis=0) if len(all_axes) > 0 else np.zeros(7)

        from core.atom import AXIS_NAMES
        return {
            "total_rooms": len(self.rooms),
            "total_objects": total_objects,
            "avg_tension": round(avg_tension, 3),
            "triggered_rooms": triggered,
            "avg_axes": {
                name: round(float(axis_means[i]), 3)
                for i, name in enumerate(AXIS_NAMES)
            },
        }
