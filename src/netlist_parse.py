import json
import networkx as nx
import matplotlib.pyplot as plt

# This script parses a Yosys JSON netlist and converts it into a NetworkX graph.
def parse_yosys_json(json_file):
    """Parses Yosys JSON netlist and converts it into a NetworkX graph, keeping only exact 'mult' and 'add' nodes."""
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
            #if cell_type == "mult" or cell_type == "add":
            G.add_node(cell_name, type=cell_type, label=cell_type)  # Node for operation

        # Process connections (nets as edges)
        net_to_cells = {}  # Map nets to connected cells

        for cell_name, cell_data in module_data["cells"].items():
            cell_type = cell_data["type"]
            #if cell_type == "mult" or cell_type == "add":
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