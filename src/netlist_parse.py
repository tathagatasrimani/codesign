import json
import networkx as nx
import matplotlib.pyplot as plt

# This script parses a Yosys JSON netlist and converts it into a NetworkX graph.
def parse_yosys_json(json_file):
    """Parses Yosys JSON netlist and converts it into a NetworkX graph, keeping only exact 'mult' and 'add' nodes."""
    with open(json_file, "r") as f:
        yosys_data = json.load(f)

    G = nx.DiGraph()  # Directed graph for circuit

    
    # for module_name, module_data in yosys_data["modules"].items():
    #     print(f"Module: {module_name}")

    #     if module_name == "MatMult":
    #         # Process cells (operations like adders, multipliers, etc.)
    #         for cell_name, cell_data in module_data["cells"].items():
    #             print(f"  Cell: {cell_name}")

    top_level_module_type = "MatMult"  # Specify the top-level module name (which is also its type)
    #top_level_module_data = yosys_data["modules"][top_level_module_type]

    # do a DFS on the design hierarchy
    queue = [(top_level_module_type, top_level_module_type)]
        #### in general, the queue should contain tuples of (module_name, module_type)

    print("Queue initialized with top-level module:", queue)
    while queue:
        current_module_tuple = queue.pop(0)
        current_module_name, current_module_type = current_module_tuple

        ## get the current module data
        current_module = yosys_data["modules"][current_module_type]
        print(f"Processing module: {current_module_name} of type {current_module_type}")  

        # Process cells (modules instantiated in this module)
        for cell_name, cell_data in current_module["cells"].items():
            print(f"  Cell: {cell_name}")
            cell_type = cell_data["type"]
            if cell_type[0] == "$":
                ### This is a primitive cell, so we can ignore it for now.
                continue
            elif cell_type == "mgc_in_sync_v2" or cell_type == "mgc_io_sync_v2":
                continue
            elif cell_type == "mult" or cell_type == "add":
                G.add_node(cell_name, type=cell_type, label=cell_type)  # include c-cores in final graph.
            elif "MatMult_ccs_ram_sync_1R1W" in cell_type:
                G.add_node(cell_name, type=cell_type, label=cell_type)  # include mem in final graph.
            else:
                ## add the cell in this module to the queue
                queue.append((cell_name, cell_type))
        # Process connections (nets as edges)

    print("Queue after processing cells:", queue)

    return G


def main():
    ### for testing
    G = parse_yosys_json("tmp/benchmark/netlist.json")

    ## write the netlist to a file
    with open("tmp/benchmark/netlist.gml", "wb") as f:
        nx.write_gml(G, f)

if __name__ == "__main__":
    main()


                

    
    
    
    
    
    # Extract original signal names from netnames
    # signal_map = {}  # Maps bit IDs to original signal names
    # for module_data in yosys_data["modules"].values():
    #     for sig_name, sig_info in module_data["netnames"].items():
    #         for bit in sig_info["bits"]:
    #             signal_map[bit] = sig_name  # Map bit ID to original signal name

    

    # # Process all modules in the design
    # for module_name, module_data in yosys_data["modules"].items():
    #     # Process cells (operations like adders, multipliers, etc.)
    #     for cell_name, cell_data in module_data["cells"].items():
    #         cell_type = cell_data["type"]
    #         #if cell_type == "mult" or cell_type == "add":
    #         G.add_node(cell_name, type=cell_type, label=cell_type)  # Node for operation

    #     # Process connections (nets as edges)
    #     net_to_cells = {}  # Map nets to connected cells

    #     for cell_name, cell_data in module_data["cells"].items():
    #         cell_type = cell_data["type"]
    #         #if cell_type == "mult" or cell_type == "add":
    #         for port, net_list in cell_data["connections"].items():
    #             for net in net_list:
    #                 net_name = signal_map.get(net, f"net_{net}")  # Use original name if available
    #                 if net_name not in net_to_cells:
    #                     net_to_cells[net_name] = []
    #                 net_to_cells[net_name].append(cell_name)

    #     # Create direct edges between operations (removing explicit net nodes)
    #     for net, connected_cells in net_to_cells.items():
    #         if len(connected_cells) > 1:
    #             for i in range(len(connected_cells) - 1):
    #                 G.add_edge(connected_cells[i], connected_cells[i + 1], signal=net)


    #     # Filter out edges that loop back to the same node
    #     for u, v in list(G.edges()):
    #         if u == v:
    #             G.remove_edge(u, v)