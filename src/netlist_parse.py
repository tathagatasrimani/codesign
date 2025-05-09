import json
import subprocess
import networkx as nx
import matplotlib.pyplot as plt

## TODO: Need to add a way to connect the 

# This script parses a Yosys JSON netlist and converts it into a NetworkX graph.
def parse_yosys_json(json_file):
    """Parses Yosys JSON netlist and converts it into a NetworkX graph, keeping only exact 'mult' and 'add' nodes."""
    with open(json_file, "r") as f:
        yosys_data = json.load(f)

    final_graph = nx.DiGraph()  # Directed graph for the final circuit

    full_graph = nx.MultiDiGraph()  # Graph for all nodes in the design
    ## each graph node is a node in a module. Ex: MatMult_19 is node 19 in module MatMult. It can be an integer value or a string. 
    ## each graph edge is a connection between two nodes Ex: MatMult_19 -> MatMult_instance101insidematmult_20


    top_level_module_type = "MatMult"  # Specify the top-level module name (which is also its type)
    #top_level_module_data = yosys_data["modules"][top_level_module_type]

    # do a DFS on the design hierarchy
    queue = [(top_level_module_type, top_level_module_type, top_level_module_type)]
        #### in general, the queue should contain tuples of (module_name, module_type, hierarchy_path)

    print("Queue initialized with top-level module:", queue)
    while queue:
        current_module_tuple = queue.pop(0)
        current_module_name, current_module_type, hierarchy_path = current_module_tuple

        ## get the current module data
        current_module = yosys_data["modules"][current_module_type]
        print(f"Processing module: {current_module_name} of type {current_module_type}")  


        ## add nets to full graph for internal nets. 
        for net_name, net_data in current_module["netnames"].items():
            net_numbers = net_data["bits"]
            for net_number in net_numbers:
                #print(f"  Net: {net_name}, Number: {net_number}")
                full_graph.add_node(hierarchy_path + '#' + str(net_number), type="net")


        ## then add the connections from input/output ports to the internal nets for this module.
        for port_name, port_data in current_module["ports"].items():
            interal_node_numbers = port_data["bits"]
            port_direction = port_data["direction"]
            input_port = False
            if port_direction == "input":
                input_port = True


            port_bit = 0 # the port bit is the index within the port. Ex: a[0] is the MSB bit of port a (note not the LSB!!!)
            
            for internal_node_num in interal_node_numbers:
                #print(f"  Port: {port_name}, Number: {port_number}")

                if input_port:
                    full_graph.add_edge(hierarchy_path + port_name + '#' + str(port_bit), hierarchy_path + '#' + str(internal_node_num))
                else:
                    full_graph.add_edge(hierarchy_path + '#' + str(internal_node_num), hierarchy_path + port_name + '#' + str(port_bit))

                port_bit += 1

        # Process cells (modules instantiated in this module)
        for cell_name, cell_data in current_module["cells"].items():
            print(f"  Cell: {cell_name}")
            cell_type = cell_data["type"]
            
            if cell_type == "mgc_in_sync_v2" or cell_type == "mgc_io_sync_v2":
                continue
            elif cell_type == "mult" or cell_type == "add":
                final_graph.add_node(cell_name, type=cell_type)  # include c-cores in final graph.
            elif "MatMult_ccs_ram_sync_1R1W" in cell_type:
                final_graph.add_node(cell_name, type=cell_type)  # include mem in final graph.
            else:
                ## add the cell in this module to the queue
                if cell_type[0] != "$":
                    ### don't need to dig deeper into the primitive cells. 
                    queue.append((cell_name, cell_type, hierarchy_path + cell_name))

            ### add to the full_graph
            for port_name, net_numbers in cell_data["connections"].items():
                
                ## input/output is from the perspective of the cell.
                input_port = False

                ## the ccore modules don't have a port direction, so we need assign direction manually.
                if cell_data["type"] == "add" or cell_data["type"] == "mult":
                    if port_name == "a" or port_name == "b":
                        input_port = True
                
                elif cell_data["port_directions"][port_name] == "input":
                    input_port = True
                
                port_bit = 0 # the port bit is the index within the port. Ex: a[0] is the MSB bit of port a (note not the LSB!!!)

                ## the net numbers is an ordered list of net numbers from the current module connect to the cell port.
                for net_number in net_numbers:
                    #print(f"  Port: {port_name}, Net: {net_number}")
                    full_graph.add_node(hierarchy_path + cell_name + port_name + '#' + str(port_bit), type=cell_type)

                    ### make a connection between the internal net and the cell port.
                    if input_port:
                        full_graph.add_edge(hierarchy_path + '#' + str(net_number), hierarchy_path + cell_name + port_name + '#' + str(port_bit))
                    else:
                        full_graph.add_edge(hierarchy_path + cell_name + port_name + '#' + str(port_bit), hierarchy_path + '#' + str(net_number))

                    port_bit += 1

    print("Queue after processing cells:", queue)

    return final_graph, full_graph


def main():
    ### for testing
    # top_module_name = "MatMult"
    # cmd = ["yosys", "-p", f"read_verilog tmp/benchmark/build/{top_module_name}.v1/rtl.v; hierarchy -top MatMult; proc; write_json tmp/benchmark/netlist.json"]
    # p = subprocess.run(cmd, capture_output=True, text=True)
    # print(f"Yosys output: {p.stdout}")
    # if p.returncode != 0:
    #     raise Exception(f"Yosys failed with error: {p.stderr}")

    G, full_graph = parse_yosys_json("tmp/benchmark/netlist.json")

    ## write the netlist to a file
    with open("tmp/benchmark/netlist.gml", "wb") as f:
        nx.write_gml(G, f)

    ## write the full_graph to a file
    with open("tmp/benchmark/full_graph.gml", "wb") as f:
        nx.write_gml(full_graph, f)

if __name__ == "__main__":
    main()