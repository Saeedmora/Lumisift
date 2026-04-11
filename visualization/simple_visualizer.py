import json
import os

class SimpleVisualizer:
    def __init__(self, graph_manager):
        self.graph_manager = graph_manager

    def generate_html(self, output_file: str = "logical_rooms_graph.html"):
        """
        Generates a standalone HTML file using Vis.js CDN.
        """
        g = self.graph_manager.get_graph_data()
        
        nodes = []
        for node_id, attrs in g.nodes(data=True):
            # Format title
            axes = attrs.get('axes', {})
            axes_str = "<br>".join([f"{k}: {v}" for k, v in axes.items()])
            title_html = f"<b>{attrs.get('label', node_id)}</b><br>{axes_str}"
            
            nodes.append({
                "id": node_id,
                "label": attrs.get('label', str(node_id)),
                "title": title_html,
                "color": "#00ffcc",
                "shape": "box"
            })

        edges = []
        for src, dst, attrs in g.edges(data=True):
            edges.append({
                "from": src,
                "to": dst,
                "value": attrs.get('weight', 1),
                "title": f"Weight: {attrs.get('weight', 0):.2f}"
            })

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Logical Rooms Graph</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {{ background-color: #222; color: white; font-family: sans-serif; }}
        #mynetwork {{
            width: 100%;
            height: 750px;
            border: 1px solid #444;
            background-color: #222222;
        }}
        .stats {{ padding: 20px; }}
    </style>
</head>
<body>
    <div class="stats">
        <h2>Logical Rooms Visualization</h2>
        <p>Nodes (Rooms): {len(nodes)} | Edges (Relationships): {len(edges)}</p>
    </div>
    <div id="mynetwork"></div>
    <script type="text/javascript">
        // create an array with nodes
        var nodes = new vis.DataSet({json.dumps(nodes)});

        // create an array with edges
        var edges = new vis.DataSet({json.dumps(edges)});

        // create a network
        var container = document.getElementById('mynetwork');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        var options = {{
            nodes: {{
                font: {{ color: 'white' }}
            }},
            physics: {{
                stabilization: false,
                barnesHut: {{
                    gravitationalConstant: -8000,
                    springConstant: 0.04,
                    springLength: 95
                }}
            }}
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>
"""
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"Graph visualization saved to {output_file} (HTML/JS generation successful)")
