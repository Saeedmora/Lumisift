try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False
    print("Warning: pyvis not found. Visualization will be text-only.")

import networkx as nx

class Visualizer:
    def __init__(self, graph_manager):
        self.graph_manager = graph_manager

    def generate_html(self, output_file: str = "logical_rooms_graph.html"):
        """
        Generates an interactive HTML visualization of the graph.
        """
        nx_graph = self.graph_manager.get_graph_data()
        
        if not HAS_PYVIS:
            print("\n[Mock Visualization] Graph Data:")
            print(f"Nodes ({len(nx_graph.nodes)}):")
            for node, data in nx_graph.nodes(data=True):
                print(f" - {data.get('label', node)} (Axes: {data.get('axes')})")
            print(f"Edges ({len(nx_graph.edges)}):")
            for src, dst, data in nx_graph.edges(data=True):
                print(f" - {src} <-> {dst} (Weight: {data.get('weight')})")
            return

        try:
            net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
            
            # Convert NetworkX graph to PyVis
            # We need to ensure attributes are compatible
            for node, data in nx_graph.nodes(data=True):
                # Format title for hover effect
                axes_str = "<br>".join([f"{k}: {v}" for k,v in data.get('axes', {}).items()])
                title = f"{data.get('label', node)}<br>{axes_str}"
                
                color = "#00ffcc" # default cyan for rooms
                 
                net.add_node(node, label=data.get('label', str(node)), title=title, color=color)

            for src, dst, data in nx_graph.edges(data=True):
                net.add_edge(src, dst, value=data.get('weight', 0.1), title=f"Weight: {data.get('weight', 0.1)}")

            net.toggle_physics(True)
            net.show(output_file)
            print(f"Graph visualization saved to {output_file}")
            
        except Exception as e:
            print(f"\n[Error] PyVis visualization failed: {e}")
            print("Switching to Text-Only Visualization.")
            print("\n[Mock Visualization] Graph Data:")
            print(f"Nodes ({len(nx_graph.nodes)}):")
            for node, data in nx_graph.nodes(data=True):
                print(f" - {data.get('label', node)} (Axes: {data.get('axes')})")
            print(f"Edges ({len(nx_graph.edges)}):")
            for src, dst, data in nx_graph.edges(data=True):
                print(f" - {src} <-> {dst} (Weight: {data.get('weight')})")
