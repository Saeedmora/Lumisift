"""
Self-Optimization Module für Logical Rooms
==========================================
Handles room updates, tension monitoring, room splitting,
time-decay, query-tracking, and adaptive thresholds.
"""

import numpy as np
from typing import List, Optional, Callable, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

from core.atom import Atom, AXIS_NAMES
from core.models import LogicalRoom


@dataclass
class OptimizationEvent:
    """Represents an optimization event."""
    timestamp: datetime
    event_type: str  # "tension_trigger", "room_split", "ema_update", "decay", "reinforce"
    room_id: str
    details: dict


@dataclass
class AssociationEdge:
    """Weighted connection between two rooms."""
    room_a_id: str
    room_b_id: str
    weight: float
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    def reinforce(self, amount: float = 0.1):
        """Strengthen association (Hebbian learning)."""
        self.weight = min(1.0, self.weight + amount * (1 - self.weight))
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def decay(self, rate: float = 0.01, days: int = 1):
        """Weaken association over time."""
        self.weight *= (1 - rate) ** days
        return self.weight


class SelfOptimizer:
    """
    Manages self-optimization for Logical Rooms.
    
    Features:
    - Tension monitoring and room splitting
    - Time-based decay for associations
    - Query-based reinforcement (Hebbian learning)
    - Adaptive thresholds based on usage patterns
    """
    
    MAX_EVENTS = 1000          # Prevent unbounded event log growth
    MAX_QUERY_HISTORY = 500    # Bound query history

    def __init__(
        self,
        tension_threshold: float = 0.3,
        variance_threshold: float = 0.5,
        alpha: float = 0.1,
        decay_rate: float = 0.02,  # 2% decay per day
        reinforce_rate: float = 0.1,  # 10% reinforcement per co-access
        on_trigger: Optional[Callable[[LogicalRoom], None]] = None,
    ):
        self.tension_threshold = tension_threshold
        self.variance_threshold = variance_threshold
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.reinforce_rate = reinforce_rate
        self.on_trigger = on_trigger
        
        self.events: List[OptimizationEvent] = []
        self.associations: Dict[Tuple[str, str], AssociationEdge] = {}
        self.query_history: List[Dict] = []
        self.room_access_counts: Dict[str, int] = defaultdict(int)
        
        # Adaptive thresholds
        self.adaptive_tension = tension_threshold
        self.threshold_history: List[float] = [tension_threshold]

    def _bound_lists(self):
        """Enforce size limits on event and history lists."""
        if len(self.events) > self.MAX_EVENTS:
            self.events = self.events[-self.MAX_EVENTS:]
        if len(self.query_history) > self.MAX_QUERY_HISTORY:
            self.query_history = self.query_history[-self.MAX_QUERY_HISTORY:]
    
    # =========================================================================
    # Core Room Updates
    # =========================================================================
    
    def update_room(self, room: LogicalRoom, obj: Atom) -> Optional[OptimizationEvent]:
        """
        Update room with new object using EMA.
        Returns event if tension threshold exceeded.
        """
        # EMA update
        room.update(obj.embedding, obj.axes_vector, obj.axes)
        
        # Track room access
        self.room_access_counts[room.id] += 1
        
        self.events.append(OptimizationEvent(
            timestamp=datetime.now(),
            event_type="ema_update",
            room_id=room.id,
            details={"object_id": obj.id, "new_tension": room.tension}
        ))
        
        # Use adaptive threshold
        if room.tension > self.adaptive_tension:
            event = OptimizationEvent(
                timestamp=datetime.now(),
                event_type="tension_trigger",
                room_id=room.id,
                details={
                    "tension": room.tension,
                    "threshold": self.adaptive_tension,
                    "axes": room.axes_summary.copy()
                }
            )
            self.events.append(event)
            
            if self.on_trigger:
                self.on_trigger(room)
            
            self._bound_lists()
            return event
        
        self._bound_lists()
        return None
    
    # =========================================================================
    # Association Management
    # =========================================================================
    
    def get_or_create_association(self, room_a_id: str, room_b_id: str) -> AssociationEdge:
        """Get existing association or create new one."""
        # Normalize order for consistent key
        key = tuple(sorted([room_a_id, room_b_id]))
        
        if key not in self.associations:
            self.associations[key] = AssociationEdge(
                room_a_id=key[0],
                room_b_id=key[1],
                weight=0.1  # Initial weak connection
            )
        
        return self.associations[key]
    
    def reinforce_association(self, room_a_id: str, room_b_id: str, amount: float = None):
        """
        Strengthen association between two rooms (Hebbian learning).
        Called when rooms are accessed together in a query.
        """
        amount = amount or self.reinforce_rate
        edge = self.get_or_create_association(room_a_id, room_b_id)
        old_weight = edge.weight
        edge.reinforce(amount)
        
        self.events.append(OptimizationEvent(
            timestamp=datetime.now(),
            event_type="reinforce",
            room_id=f"{room_a_id[:8]}↔{room_b_id[:8]}",
            details={
                "old_weight": round(old_weight, 3),
                "new_weight": round(edge.weight, 3),
                "access_count": edge.access_count
            }
        ))
        
        return edge
    
    def apply_time_decay(self):
        """
        Apply time-based decay to all associations.
        Should be called periodically (e.g., daily or on startup).
        """
        now = datetime.now()
        decayed_count = 0
        removed_count = 0
        
        for key, edge in list(self.associations.items()):
            days_inactive = (now - edge.last_accessed).days
            
            if days_inactive > 0:
                old_weight = edge.weight
                new_weight = edge.decay(self.decay_rate, days_inactive)
                decayed_count += 1
                
                # Remove very weak associations
                if new_weight < 0.05:
                    del self.associations[key]
                    removed_count += 1
        
        if decayed_count > 0:
            self.events.append(OptimizationEvent(
                timestamp=now,
                event_type="decay",
                room_id="global",
                details={
                    "associations_decayed": decayed_count,
                    "associations_removed": removed_count,
                    "total_remaining": len(self.associations)
                }
            ))
        
        return decayed_count, removed_count
    
    def get_associated_rooms(self, room_id: str, min_weight: float = 0.2) -> List[Tuple[str, float]]:
        """Get rooms associated with given room, sorted by weight."""
        results = []
        
        for key, edge in self.associations.items():
            if room_id in key:
                other_id = key[0] if key[1] == room_id else key[1]
                if edge.weight >= min_weight:
                    results.append((other_id, edge.weight))
        
        return sorted(results, key=lambda x: -x[1])
    
    # =========================================================================
    # Query Tracking
    # =========================================================================
    
    def track_query(self, query: str, accessed_rooms: List[str], response_quality: float = None):
        """
        Track a query and the rooms it accessed.
        Reinforces associations between co-accessed rooms.
        """
        query_record = {
            "timestamp": datetime.now(),
            "query": query[:100],
            "rooms": accessed_rooms,
            "quality": response_quality
        }
        self.query_history.append(query_record)
        
        # Reinforce associations between co-accessed rooms
        if len(accessed_rooms) >= 2:
            from itertools import combinations
            for room_a, room_b in combinations(accessed_rooms, 2):
                self.reinforce_association(room_a, room_b)
        
        # Track individual room accesses
        for room_id in accessed_rooms:
            self.room_access_counts[room_id] += 1
        
        # Adapt thresholds based on query patterns
        self._update_adaptive_threshold()
        
        self._bound_lists()
        return query_record
    
    def _update_adaptive_threshold(self):
        """
        Adjust tension threshold based on recent activity.
        More activity = slightly higher tolerance for tension.
        """
        recent_queries = [q for q in self.query_history 
                        if datetime.now() - q["timestamp"] < timedelta(hours=24)]
        
        if len(recent_queries) > 10:
            # High activity: be slightly more tolerant
            self.adaptive_tension = min(0.5, self.tension_threshold * 1.1)
        elif len(recent_queries) < 3:
            # Low activity: be stricter
            self.adaptive_tension = max(0.2, self.tension_threshold * 0.9)
        else:
            # Normal activity: use base threshold
            self.adaptive_tension = self.tension_threshold
        
        self.threshold_history.append(self.adaptive_tension)
    
    # =========================================================================
    # Room Splitting
    # =========================================================================
    
    def check_split(self, room: LogicalRoom) -> Optional[List[LogicalRoom]]:
        """
        Check if room should split due to high variance.
        Returns new rooms if split occurred.
        """
        if not room.should_split(self.variance_threshold):
            return None
        
        if len(room.objects) < 4:
            return None  # Need enough objects to split
        
        # Simple k-means style split
        new_rooms = self._split_room(room)
        
        self.events.append(OptimizationEvent(
            timestamp=datetime.now(),
            event_type="room_split",
            room_id=room.id,
            details={
                "original_objects": len(room.objects),
                "new_rooms": len(new_rooms),
                "variance": float(np.max(room.variance)) if room.variance is not None else 0
            }
        ))
        
        return new_rooms
    
    def _split_room(self, room: LogicalRoom) -> List[LogicalRoom]:
        """Split a room into two based on object clustering."""
        objects = room.objects
        
        # Simple split by highest variance axis
        if room.variance is None:
            return [room]
        
        split_axis = int(np.argmax(room.variance))
        
        # Sort objects by their value on the split axis
        axis_names = AXIS_NAMES
        split_name = axis_names[split_axis]
        
        sorted_objects = sorted(objects, key=lambda o: o.axes.get(split_name, 0))
        mid = len(sorted_objects) // 2
        
        # Create two new rooms
        room1 = LogicalRoom(name=f"{room.name}-A")
        room2 = LogicalRoom(name=f"{room.name}-B")
        
        for obj in sorted_objects[:mid]:
            obj.room_id = None  # Reset
            room1.add_object(obj)
        
        for obj in sorted_objects[mid:]:
            obj.room_id = None
            room2.add_object(obj)
        
        return [room1, room2]
    
    # =========================================================================
    # Statistics & Reporting
    # =========================================================================
    
    def get_high_tension_rooms(self, rooms: List[LogicalRoom]) -> List[LogicalRoom]:
        """Get all rooms exceeding tension threshold."""
        return [r for r in rooms if r.tension > self.adaptive_tension]
    
    def get_optimization_summary(self) -> dict:
        """Get summary of optimization events."""
        triggers = [e for e in self.events if e.event_type == "tension_trigger"]
        splits = [e for e in self.events if e.event_type == "room_split"]
        updates = [e for e in self.events if e.event_type == "ema_update"]
        reinforces = [e for e in self.events if e.event_type == "reinforce"]
        decays = [e for e in self.events if e.event_type == "decay"]
        
        return {
            "total_events": len(self.events),
            "tension_triggers": len(triggers),
            "room_splits": len(splits),
            "ema_updates": len(updates),
            "reinforcements": len(reinforces),
            "decay_events": len(decays),
            "active_associations": len(self.associations),
            "queries_tracked": len(self.query_history),
            "adaptive_tension": round(self.adaptive_tension, 3),
            "latest_events": [
                {
                    "type": e.event_type,
                    "room": e.room_id[:12] if len(e.room_id) > 12 else e.room_id,
                    "time": e.timestamp.strftime("%H:%M:%S")
                }
                for e in self.events[-5:]
            ]
        }
    
    def get_association_stats(self) -> dict:
        """Get statistics about associations."""
        if not self.associations:
            return {"count": 0, "avg_weight": 0, "strongest": None}
        
        weights = [e.weight for e in self.associations.values()]
        strongest = max(self.associations.values(), key=lambda e: e.weight)
        
        return {
            "count": len(self.associations),
            "avg_weight": round(np.mean(weights), 3),
            "max_weight": round(max(weights), 3),
            "min_weight": round(min(weights), 3),
            "strongest": {
                "rooms": f"{strongest.room_a_id[:8]}↔{strongest.room_b_id[:8]}",
                "weight": round(strongest.weight, 3),
                "accesses": strongest.access_count
            }
        }


def compute_meta_tension(relevance: float, risk: float, trust: float) -> float:
    """
    Compute meta-tension: T = Relevance * Risk * (1 - Trust)
    
    High tension indicates content that is:
    - Highly relevant
    - High risk
    - Low trust (unverified)
    
    Returns value 0-1.
    """
    risk_positive = max(0, risk)  # Only positive risk contributes
    return relevance * risk_positive * (1 - trust)

