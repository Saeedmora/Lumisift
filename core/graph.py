import networkx as nx
from typing import List, Dict
from core.models import LogicalRoom

class GraphManager:
    def __init__(self):
        self.graph = nx.Graph()

    def add_room(self, room: LogicalRoom):
        """
        Adds a LogicalRoom as a node in the graph.
        """
        self.graph.add_node(
            room.id,
            label=room.name,
            type="room",
            axes=room.axes_summary
        )

    def add_relationship(self, room_id_a: str, room_id_b: str, weight: float, type: str = "semantic"):
        """
        Adds an edge between two rooms.
        """
        self.graph.add_edge(room_id_a, room_id_b, weight=weight, type=type)

    def find_analogies(self, room_id: str, threshold: float = 0.5) -> List[str]:
        """
        Finds connected nodes with high edge weights (simple analogy finder).
        """
        if room_id not in self.graph:
            return []
        
        neighbors = []
        for neighbor, attrs in self.graph[room_id].items():
            if attrs.get('weight', 0) > threshold:
                neighbors.append(neighbor)
        return neighbors

    def get_graph_data(self):
        return self.graph
