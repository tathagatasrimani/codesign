import json
import os
import subprocess
import networkx as nx
import matplotlib.pyplot as plt

DEBUG_PRINT = True

def debug_print(*args, **kwargs):
    if DEBUG_PRINT:
        print(*args, **kwargs)

def check_if_nodes_are_in_graph(graph, nodes):
    """Check if all nodes in the list are present in the graph."""
    all_found = True   
    node_not_found = None
    for node in nodes:
        if node not in graph.nodes:
            all_found = False
            node_not_found = node
    assert all_found, f"Not all nodes are in the graph. Missing node: {node_not_found}"

# This script parses a Yosys JSON netlist and converts it into a NetworkX graph.
def parse_yosys_json(json_file, include_memories=True, top_level_module_type="MatMult"):
    """Parses Yosys JSON netlist and converts it into a NetworkX graph, keeping only CCORE cells and memories."""
    with open(json_file, "r") as f:
        yosys_data = json.load(f)

    final_graph = nx.DiGraph()  # Directed graph for the final circuit

    full_graph = nx.DiGraph()  # Graph for all nodes in the design
    ## each graph node is a node in a module. Ex: MatMult_19 is node 19 in module MatMult. It can be an integer value or a string. 
    ## each graph edge is a connection between two nodes Ex: MatMult_19 -> MatMult_instance101insidematmult_20

    # do a DFS on the design hierarchy
    queue = [(top_level_module_type, top_level_module_type, top_level_module_type)]
        #### in general, the queue should contain tuples of (module_name, module_type, hierarchy_path)

    debug_print("Queue initialized with top-level module:", queue)


    # add the top-level module's input/output ports to the full graph
    top_level_module_info = yosys_data["modules"][top_level_module_type]

    for port_name, port_data in top_level_module_info["ports"].items():
                
        ## input/output is from the perspective of the cell.
        input_port = False
        
        if port_data["direction"] == "input":
            input_port = True
        
        port_bit = 0 # the port bit is the index within the port. Ex: a[0] is the MSB bit of port a (note not the LSB!!!)

        ## the net numbers is an ordered list of net numbers from the current module connect to the cell port.
        for net_number in port_data["bits"]:
            debug_print(f"  Port: {port_name}, Net: {net_number}")
            full_graph.add_node(top_level_module_type + "$$" + port_name + '#' + str(port_bit), type="module_port", text_label= port_name + "[" + str(port_bit) + "]", hierarchy_path= top_level_module_type, in_final_graph=False)

            port_bit += 1

    all_proc_modules_found = {}

    while queue:
        current_module_tuple = queue.pop(0)
        current_module_name, current_module_type, hierarchy_path = current_module_tuple

        ## get the current module data
        current_module = yosys_data["modules"][current_module_type]
        debug_print(f"Processing module: {current_module_name} of type {current_module_type}")  

        ## add nets to full graph for internal nets. 
        for net_name, net_data in current_module["netnames"].items():
            net_numbers = net_data["bits"]
            index_num = 0
            for net_number in net_numbers:
                debug_print(f"  Net: {net_name}, Number: {net_number}")
                full_graph.add_node(hierarchy_path + '#' + str(net_number), type="internal_net", text_label= net_name + "[" + str(index_num) + "]", in_final_graph=False, hierarchy_path= hierarchy_path)
                index_num += 1


        ## then add the connections from input/output ports to the internal nets for this module.
        for port_name, port_data in current_module["ports"].items():
            interal_node_numbers = port_data["bits"]
            port_direction = port_data["direction"]
            input_port = False
            if port_direction == "input":
                input_port = True


            port_bit = 0 # the port bit is the index within the port. Ex: a[0] is the MSB bit of port a (note not the LSB!!!)
            
            for internal_node_num in interal_node_numbers:
                debug_print(f"  Port: {port_name}, Number: {port_bit}")

                check_if_nodes_are_in_graph(full_graph, [hierarchy_path + '#' + str(internal_node_num), hierarchy_path + "$$" + port_name + '#' + str(port_bit)])

                if input_port:
                    full_graph.add_edge(hierarchy_path + "$$" + port_name + '#' + str(port_bit), hierarchy_path + '#' + str(internal_node_num))
                else:
                    full_graph.add_edge(hierarchy_path + '#' + str(internal_node_num), hierarchy_path + "$$" + port_name + '#' + str(port_bit))

                port_bit += 1

        # Process cells (modules instantiated in this module)
        for cell_name, cell_data in current_module["cells"].items():
            debug_print(f"  Cell: {cell_name}")
            cell_type = cell_data["type"]
            in_final_graph = False
            
            if cell_type == "mgc_in_sync_v2" or cell_type == "mgc_io_sync_v2":
                continue
            elif cell_type == "mult" or cell_type == "add": ### TODO: Add more CORES Here. 
                final_graph.add_node(hierarchy_path + "**" + cell_name, type="c-core", module_type=cell_type, hierarchy_path=hierarchy_path + "**" + cell_name)  # include c-cores in final graph.
                in_final_graph = True
            elif "MatMult_ccs_ram_sync_1R1W" in cell_type:
                if include_memories:
                    final_graph.add_node(hierarchy_path + "**" + cell_name, type="memory", module_type=cell_type, hierarchy_path=hierarchy_path + "**" + cell_name)  # include mem in final graph.
                    in_final_graph = True
                else:
                    continue ## completely ignore the memory instance.
            elif cell_type[0] != "$":
                ## add the cell in this module to the queue 
                queue.append((cell_name, cell_type, hierarchy_path + "**" + cell_name))

            list_nodes_added = []
            ### add to the full_graph
            for port_name, net_numbers in cell_data["connections"].items():
                
                ## input/output is from the perspective of the cell.
                input_port = False

                

                ## the ccore modules don't have a port direction, so we need assign direction manually.
                if cell_type == "add" or cell_type == "mult": ## TODO: Add more CORES Here.
                    if port_name == "a" or port_name == "b":
                        input_port = True

                ### check if the there is no port direction in the cell data, assuming that we are not dealing with a C-CORE cell.
                elif "port_directions" not in cell_data:
                    ## This is a Catapult cell. We need to get the port direction from the catapult library.

                    ## parse this cell type if we haven't done so already.
                    if not os.path.isfile(f"src/tmp/benchmark/{cell_type}.json"):
                        cmd = ["yosys", "-p", f"read_verilog /nfs/cad/mentor/2024.2/Mgc_home/pkgs/siflibs/{cell_type}.v; hierarchy -top {cell_type}; proc; write_json src/tmp/benchmark/{cell_type}.json"]
                        p = subprocess.run(cmd, capture_output=True, text=True)
                        ##logger.info(f"Yosys output: {p.stdout}")
                        if p.returncode != 0:
                            raise Exception(f"Yosys failed with error: {p.stderr}")
                        
                    ## read the cell data from the JSON file
                    with open(f"src/tmp/benchmark/{cell_type}.json", "r") as f:
                        cell_data = json.load(f)["modules"][cell_type]

                        if cell_data["ports"][port_name]["direction"] == "input":
                            input_port = True

                    ## add the cell data to the yosys_data which will be parsed later.
                    if not cell_type in yosys_data["modules"]:
                        yosys_data["modules"][cell_type] = cell_data

                elif cell_data["port_directions"][port_name] == "input":
                    input_port = True

                in_out_direction = "input" if input_port else "output"
                
                port_bit = 0 # the port bit is the index within the port. Ex: a[0] is the MSB bit of port a (note not the LSB!!!)

                ## the net numbers is an ordered list of net numbers from the current module connect to the cell port.
                for net_number in net_numbers:
                    ## check if the net number isn't actually a net number and is a constant.
                    ## if it is a constnant, it will be a string instead of an int.
                    if isinstance(net_number, str):
                        port_bit += 1
                        continue

                    debug_print(f"  Port: {port_name}, Net: {net_number}")
                    full_graph.add_node(hierarchy_path + "**" + cell_name + "$$" + port_name + '#' + str(port_bit), type="module_port", text_label= port_name + "[" + str(port_bit) + "]", hierarchy_path= hierarchy_path + "**" + cell_name, in_final_graph=in_final_graph, direction=in_out_direction, cell_type=cell_type, orig_port_name=port_name, bit_pos=port_bit)
                    list_nodes_added.append(hierarchy_path + "**" + cell_name + "$$" + port_name + '#' + str(port_bit))

                    ### make a connection between the internal net and the cell port.
                    check_if_nodes_are_in_graph(full_graph, [hierarchy_path + '#' + str(net_number), hierarchy_path + "**" + cell_name + "$$" + port_name + '#' + str(port_bit)])
                    if input_port:
                        full_graph.add_edge(hierarchy_path + '#' + str(net_number), hierarchy_path + "**" + cell_name + "$$" + port_name + '#' + str(port_bit))
                    else:
                        full_graph.add_edge(hierarchy_path + "**" + cell_name + "$$" + port_name + '#' + str(port_bit), hierarchy_path + '#' + str(net_number))

                    port_bit += 1
            
            ## This code will make connection between the inputs/outputs of a automatically generated cell. 
            if cell_type[0] == "$":
                all_proc_modules_found[cell_type] = True
                ## this is an automatically generated cell by Yosys from running the proc command.
                ## we need to make connections between the inputs/outputs as needed based on the cell type.
                if cell_type == "$adff":
                    ## iterate through the added ports and make connections between the input and output ports "internally" in this module.
                    for curr_cell_node in list_nodes_added:
                        if full_graph.nodes[curr_cell_node]["orig_port_name"] == "D":
                            ## this is the input port, so we need to connect it to the output port.
                            output_port_node = curr_cell_node.split("$$")[0] + "$$" + "Q" + "#" + str(full_graph.nodes[curr_cell_node]["bit_pos"])
                            debug_print(f"  Adding edge between {curr_cell_node} and {output_port_node}")
                            check_if_nodes_are_in_graph(full_graph, [curr_cell_node, output_port_node])
                            full_graph.add_edge(curr_cell_node, output_port_node)
                elif cell_type == "$not":
                    ## iterate through the added ports and make connections between the input and output ports "internally" in this module.
                    for curr_cell_node in list_nodes_added:
                        if full_graph.nodes[curr_cell_node]["orig_port_name"] == "A":
                            ## this is the input port, so we need to connect it to the output port.
                            output_port_node = curr_cell_node.split("$$")[0] + "$$" + "Y" + "#" + str(full_graph.nodes[curr_cell_node]["bit_pos"])
                            debug_print(f"  Adding edge between {curr_cell_node} and {output_port_node}")
                            check_if_nodes_are_in_graph(full_graph, [curr_cell_node, output_port_node])
                            full_graph.add_edge(curr_cell_node, output_port_node)
                elif cell_type == "$logic_not" or cell_type == "$reduce_bool":
                    ## iterate through the added ports and make connections between the input and output ports "internally" in this module.
                    for curr_cell_node in list_nodes_added:
                        if full_graph.nodes[curr_cell_node]["orig_port_name"] == "A":
                            ## this is the input port, so we need to connect it to the output port.
                            output_port_node = curr_cell_node.split("$$")[0] + "$$" + "Y" + "#" + "0"
                            debug_print(f"  Adding edge between {curr_cell_node} and {output_port_node}")
                            check_if_nodes_are_in_graph(full_graph, [curr_cell_node, output_port_node])
                            full_graph.add_edge(curr_cell_node, output_port_node)
                elif cell_type == "$eq" or cell_type == "$ne":
                    ## iterate through the added ports and make connections between the input and output ports "internally" in this module.
                    for curr_cell_node in list_nodes_added:
                        if full_graph.nodes[curr_cell_node]["orig_port_name"] == "A":
                            ## this is the input port, so we need to connect it to the output port.
                            output_port_node = curr_cell_node.split("$$")[0] + "$$" + "Y" + "#0"
                            debug_print(f"  Adding edge between {curr_cell_node} and {output_port_node}")
                            check_if_nodes_are_in_graph(full_graph, [curr_cell_node, output_port_node])
                            full_graph.add_edge(curr_cell_node, output_port_node)
                elif cell_type == "$and" or cell_type == "$or" or cell_type == "$xor":
                    ## iterate through the added ports and make connections between the input and output ports "internally" in this module.
                    for curr_cell_node in list_nodes_added:
                        if full_graph.nodes[curr_cell_node]["orig_port_name"] == "A" or full_graph.nodes[curr_cell_node]["orig_port_name"] == "B":
                            ## this is the input port, so we need to connect it to the output port.
                            output_port_node = curr_cell_node.split("$$")[0] + "$$" + "Y" + "#" + str(full_graph.nodes[curr_cell_node]["bit_pos"])
                            debug_print(f"  Adding edge between {curr_cell_node} and {output_port_node}")
                            check_if_nodes_are_in_graph(full_graph, [curr_cell_node, output_port_node])
                            full_graph.add_edge(curr_cell_node, output_port_node)

                elif cell_type == "$mux":
                    ## iterate through the added ports and make connections between the input and output ports "internally" in this module.
                    for curr_cell_node in list_nodes_added:
                        if full_graph.nodes[curr_cell_node]["orig_port_name"] == "A" or full_graph.nodes[curr_cell_node]["orig_port_name"] == "B":
                            ## this is the input port, so we need to connect it to the output port.
                            output_port_node = curr_cell_node.split("$$")[0] + "$$" + "Y" + "#" + str(full_graph.nodes[curr_cell_node]["bit_pos"])
                            debug_print(f"  Adding edge between {curr_cell_node} and {output_port_node}")
                            check_if_nodes_are_in_graph(full_graph, [curr_cell_node, output_port_node])
                            full_graph.add_edge(curr_cell_node, output_port_node)
                        elif full_graph.nodes[curr_cell_node]["orig_port_name"] == "S":
                            ## This is the select port, so we need to connect it to ALL the output ports.
                            for curr_output_cell_node in list_nodes_added:
                                if full_graph.nodes[curr_output_cell_node]["orig_port_name"] == "Y":

                                    debug_print(f"  Adding edge between {curr_cell_node} and {output_port_node}")
                                    check_if_nodes_are_in_graph(full_graph, [curr_cell_node, curr_output_cell_node])
                                    full_graph.add_edge(curr_cell_node, curr_output_cell_node)
                elif cell_type == "$memrd_v2" or cell_type == "$meminit":
                    ## skip these cell types. It stores the state for the control state machine. We aren't modeling this in the final graph. 
                    pass 

    ### Raise error message if we have unhandled proc modules. This may happen as we test different designs. 
    current_handled_proc_modules = {'$and': True, '$not': True, '$eq': True, '$logic_not': True, '$reduce_bool': True, '$ne': True, '$or': True, '$adff': True, '$mux': True, '$xor': True, '$memrd_v2': True, '$meminit': True}
    
    for proc_module in all_proc_modules_found:
        if proc_module not in current_handled_proc_modules:
            debug_print(f"WARNING: Unhandled proc module type: {proc_module}. Please add handling for this module type.")
            

    debug_print("Queue after processing cells:", queue)


    ############ Now, add edges between the cells in the final graph.
    
    
    ## First, create a mapping from the hierarchy path to a list of single bit port nodes in the full graph for each C-CORE cell instance or memory instance.
    
    final_graph_node_hierarchy_paths_dict = {}
    for node, data in final_graph.nodes(data=True): 
        final_graph_node_hierarchy_paths_dict[data["hierarchy_path"]] = { "inputs" : [] , "outputs" : [] }
    
    
    
    cell_port_mapping = {}
    for node, data in full_graph.nodes(data=True):
        debug_print(f"Node: {node}, Data: {data}")
        if data['in_final_graph'] == True:
            if data['hierarchy_path'] in final_graph_node_hierarchy_paths_dict:
                if data['direction'] == "input":
                    final_graph_node_hierarchy_paths_dict[data['hierarchy_path']]["inputs"].append(node)
                elif data['direction'] == "output":
                    final_graph_node_hierarchy_paths_dict[data['hierarchy_path']]["outputs"].append(node)
                
            
        
    ### debug_print final_graph_node_hierarchy_paths_dict
    # debug_print("Final graph node hierarchy paths dictionary:")
    # for key, value in final_graph_node_hierarchy_paths_dict.items():
    #     debug_print(f"  {key}: {value}")
    #     debug_print("="*20)

    ##############################


    for node, data in final_graph.nodes(data=True):
        ## iterate through all of the corresponding OUTPUT port nodes in the full graph.
        for port_node in final_graph_node_hierarchy_paths_dict[data["hierarchy_path"]]["outputs"]:
            ### get a list of all reachable nodes from the output port nodes.
            reachable_nodes = nx.descendants(full_graph, port_node)
            debug_print(f"  Port node: {port_node}, Reachable nodes: {reachable_nodes}")

            ## go through each reachable node and see if it is an input port of a C-CORE cell instance or memory instance.
            for reachable_node in reachable_nodes:
                rechable_node_data = full_graph.nodes[reachable_node]
                debug_print(f"    Reachable node: {reachable_node}, Data: {rechable_node_data}")

                if rechable_node_data['in_final_graph'] == True and rechable_node_data['direction'] == "input":
                    ## see if the reachable node is one we are looking for. 
                    reachable_node_hierarchy_path = rechable_node_data['hierarchy_path']
                    if reachable_node_hierarchy_path in final_graph_node_hierarchy_paths_dict and reachable_node_hierarchy_path != data["hierarchy_path"]:
                        ## add this edge to the final graph.
                        final_graph.add_edge(data["hierarchy_path"], reachable_node_hierarchy_path)

    

    return final_graph, full_graph


def main():
    ### for testing
    top_module_name = "MatMult"
    ## cmd = ["yosys", "-p", f"read_verilog benchmarks/test_verilog_file.v; hierarchy -top MatMult; proc; write_json tmp/benchmark/netlist.json"] ## toy test case
    cmd = ["yosys", "-p", f"read_verilog tmp/benchmark/build/MatMult.v1/rtl.v; hierarchy -top {top_module_name}; proc; write_json tmp/benchmark/netlist.json"] # the real deal
    
    p = subprocess.run(cmd, capture_output=True, text=True)
    debug_print(f"Yosys output: {p.stdout}")
    if p.returncode != 0:
        raise Exception(f"Yosys failed with error: {p.stderr}")

    G, full_graph = parse_yosys_json("tmp/benchmark/netlist.json", top_level_module_type=top_module_name, include_memories=True)

    ## write the netlist to a file
    with open("tmp/benchmark/netlist.gml", "wb") as f:
        nx.write_gml(G, f)

    ## write the full_graph to a file
    with open("tmp/benchmark/full_graph.gml", "wb") as f:
        nx.write_gml(full_graph, f)


    

if __name__ == "__main__":
    main()