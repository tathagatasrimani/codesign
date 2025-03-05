import json
import sys
import networkx as nx
import matplotlib.pyplot as plt

def parse_yosys_json(json_file):
    """Parses Yosys JSON netlist and converts it into a NetworkX graph, keeping only exact 'mult' and 'adder' nodes."""
    with open(json_file, "r") as f:
        yosys_data = json.load(f)

    G = nx.DiGraph()  # Directed graph for circuit

    # Extract original signal names from netnames
    signal_map = {}  # Maps bit IDs to original signal names
    for module_data in yosys_data["modules"].values():
        for sig_name, sig_info in module_data["netnames"].items():
            for bit in sig_info["bits"]:
                signal_map[bit] = sig_name  # Map bit ID to original signal name

    # Process all modules in the design
    for module_name, module_data in yosys_data["modules"].items():
        # Process cells (operations like adders, multipliers, etc.)
        for cell_name, cell_data in module_data["cells"].items():
            cell_type = cell_data["type"]
            if cell_type == "mult" or cell_type == "adder":
                G.add_node(cell_name, type=cell_type, label=cell_type)  # Node for operation

        # Process connections (nets as edges)
        net_to_cells = {}  # Map nets to connected cells

        for cell_name, cell_data in module_data["cells"].items():
            cell_type = cell_data["type"]
            if cell_type == "mult" or cell_type == "adder":
                for port, net_list in cell_data["connections"].items():
                    for net in net_list:
                        net_name = signal_map.get(net, f"net_{net}")  # Use original name if available
                        if net_name not in net_to_cells:
                            net_to_cells[net_name] = []
                        net_to_cells[net_name].append(cell_name)

        # Create direct edges between operations (removing explicit net nodes)
        for net, connected_cells in net_to_cells.items():
            if len(connected_cells) > 1:
                for i in range(len(connected_cells) - 1):
                    G.add_edge(connected_cells[i], connected_cells[i + 1], signal=net)


        # Filter out edges that loop back to the same node
        for u, v in list(G.edges()):
            if u == v:
                G.remove_edge(u, v)

    return G

# Load and parse Yosys JSON file
json_file = sys.argv[1]
if not json_file.endswith(".json"):
    raise ValueError("Input file must be a Yosys JSON file.")
G = parse_yosys_json(json_file)

# Export the graph to a GML file
output_gml_file_name = sys.argv[2]
if not output_gml_file_name.endswith(".gml"):
    output_gml_file_name += ".gml"
nx.write_gml(G, output_gml_file_name)

# Draw the graph
# plt.figure(figsize=(10, 7))
# pos = nx.spring_layout(G)  # Layout for visualization
# nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=800, font_size=8)

# # Add signal labels to edges
# edge_labels = {(u, v): d["signal"] for u, v, d in G.edges(data=True)}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

# plt.title("Filtered Circuit Netlist Graph (Multipliers and Adders Only)")
# plt.show()
