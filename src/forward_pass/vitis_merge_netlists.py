import os
import sys
import json
import networkx as nx


DEBUG = True

def debug_print(message):
    if DEBUG:
        print(message)


all_modules_visited = set()

def merge_netlists_vitis(root_dir, top_level_module_name, filter_nodes=True):
    """
    Main function to create a unified netlist for all files in the given directory.
    """
    full_netlist = nx.DiGraph()

    ## get the names of all the submodules (the subdirectories)
    submodules = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
    #debug_print(f"Submodules found: {submodules}")

    ## start with the top-level module
    if top_level_module_name not in submodules:
        print(f"Error: Top-level module {top_level_module_name} does not exist in {root_dir}.")
        return

    full_netlist = parse_module(root_dir, top_level_module_name)
    if full_netlist is None:
        print(f"Error: Failed to parse the top-level module {top_level_module_name}.")
        return
    
    ## print out the visited modules
    #debug_print(f"Visited modules: {all_modules_visited}")
    ## print out the number of visited modules
    #debug_print(f"Number of visited modules: {len(all_modules_visited)}")

    ## save the full netlist to a file
    output_file_path = os.path.join(root_dir, f"{top_level_module_name}_full_netlist_unfiltered.gml")
    nx.write_gml(full_netlist, output_file_path)
    #debug_print(f"Full netlist saved to {output_file_path}")

    ## filter the netlist to only include desired node types
    desired_node_types = {'add', 'mul', 'fmul'}
    filtered_netlist = filter_netlist(full_netlist, desired_node_types, filter_nodes)

    ## save the filtered netlist to a file
    output_filtered_file_path = os.path.join(root_dir, f"{top_level_module_name}_full_netlist.gml")
    nx.write_gml(filtered_netlist, output_filtered_file_path)

def filter_netlist(full_netlist, desired_node_types={"add", "mul", "fmul"}, filter_nodes=True, op_attr="fcode"):
    """
    Return a new DiGraph H that contains only the 'functional unit' nodes
    (bind[op_attr] in desired_node_types) from full_netlist, as well as any node with type 'CONTROL_NODE'.
    H has an edge u->v iff in full_netlist there exists a directed path u -> ... -> v whose internal nodes are all NON-target.
    Each node in H will have all attributes from the original netlist node.
    """
    # Identify target nodes and control nodes
    is_target = {
        n: (
            (filter_nodes and full_netlist.nodes[n].get('bind', {}).get(op_attr) in desired_node_types)
            or (full_netlist.nodes[n].get('type') == 'CONTROL_NODE')
        )
        for n in full_netlist.nodes
    }
    targets = [n for n, t in is_target.items() if t]

    # Initialize the pruned graph with the target nodes and all their attributes
    H = nx.DiGraph()
    for n in targets:
        H.add_node(n, **full_netlist.nodes[n])

    # For each target, walk outward through only non-target nodes.
    for u in targets:
        stack = list(full_netlist.successors(u))
        visited = set()

        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)

            if is_target.get(x, False) and x != u:
                # Copy edge attributes if present
                edge_attrs = full_netlist.get_edge_data(u, x, default={})
                H.add_edge(u, x, **edge_attrs)
                continue
            stack.extend(full_netlist.successors(x))

    return H



def parse_module(root_dir, current_module):

    debug_print(f"!!!!!!!!!!!!!!!!!!!!!!!Parsing module for netlist merge: {current_module}")

    ## check if the current module has already been visited
    # if it has, we can read the netlist from the file and return it.
    if current_module in all_modules_visited:
        #debug_print(f"Module {current_module} already visited. Returning existing netlist.")
        netlist_file_path = os.path.join(root_dir, f"{current_module}_full_netlist.gml")
        if not os.path.exists(netlist_file_path):
            print(f"Error: netlist file {netlist_file_path} does not exist.")
            exit(1)
            return None
        return nx.read_gml(netlist_file_path)

    ## add module to the visited modules list
    all_modules_visited.add(current_module)

    ## open the _netlist.gml file for the current module. Read it in as a NetworkX graph.
    netlist_file_path = os.path.join(root_dir, current_module, f"{current_module}.verbose_netlist.gml")

    append_s_to_path = False
    if not os.path.exists(netlist_file_path):
        netlist_file_path = os.path.join(root_dir, current_module + "s", f"{current_module}s.verbose_netlist.gml")
        if not os.path.exists(netlist_file_path):
            print(f"Error: netlist file {netlist_file_path} does not exist.")
            exit(1)
            return
        else:
            current_module += "s"

    full_netlist = nx.read_gml(netlist_file_path)

    ## read in the _modules.json file for the current module
    modules_file_path = os.path.join(root_dir, current_module, f"{current_module}.verbose_modules.json")
    if not os.path.exists(modules_file_path):
        modules_file_path = os.path.join(root_dir, current_module + "s", f"{current_module}s.verbose_modules.json")
        if not os.path.exists(modules_file_path):
            print(f"Error: Modules file {modules_file_path} does not exist.")
            exit(1)
            return

    with open(modules_file_path, 'r') as mf:
        module_dependences = json.load(mf)

    if module_dependences:
        debug_print(f"Instantiated modules for {current_module}: {module_dependences}")
        pass

    if module_dependences is None or len(module_dependences) == 0:
        #debug_print(f"No submodules found for {current_module}. Returning netlist as is.")
        pass

    ## get the netlists for each of the instantiated modules recursively
    submodule_netlists = {}
    for module_name in module_dependences:
        submodule_netlists[module_name] = parse_module(root_dir, module_name)

    ## merge the netlists of the submodules into the full netlist
    for submodule_name, submodule_netlist in submodule_netlists.items():
        if submodule_netlist is not None:
            #full_netlist = nx.compose(full_netlist, submodule_netlist)
            # print the number of nodes in the full_netlist before and after the merge
            debug_print(f"Before merging {submodule_name}, full netlist has {full_netlist.number_of_nodes()} nodes.")
            full_netlist_new = merge_netlists(root_dir, full_netlist, current_module, submodule_netlist, submodule_name)
            debug_print(f"After merging {submodule_name}, full netlist has {full_netlist_new.number_of_nodes()} nodes.")
            full_netlist = full_netlist_new
            #debug_print(f"Merged netlist for submodule {submodule_name} into full netlist.")

    ## write out the full netlist for the current module
    output_file_path = os.path.join(root_dir, current_module, f"{current_module}_full_netlist.gml")
    nx.write_gml(full_netlist, output_file_path)
    #debug_print(f"Full netlist for module {current_module} saved to {output_file_path}")

    return full_netlist

def merge_netlists(root_dir, full_netlist, current_module, submodule_netlist, submodule_name): 
    """
    Merge the submodule netlist into the full netlist.
    """
    debug_print(f"Merging netlist for submodule {submodule_name} into full netlist for module {current_module}.")


    # Read the submodule instance to module name mapping for this module
    module_instance_file = os.path.join(root_dir, current_module, f"{current_module}.verbose_instance_names.json")
    if not os.path.exists(module_instance_file):
        module_instance_file = os.path.join(root_dir, current_module + "s", f"{current_module}s.verbose_instance_names.json")
        if not os.path.exists(module_instance_file):
            print(f"Error: Instance to module file {module_instance_file} does not exist.")
            exit(1)
            return

    with open(module_instance_file, 'r') as f:
        module_instance_mapping = json.load(f)
    #debug_print(f"module instance mapping for {current_module}: {module_instance_mapping}")

    # Find all nodes in the full netlist that are call functions to the submodule
    # these nodes will have fcode="call"
    submodule_call_nodes = []
    for n, d in full_netlist.nodes(data=True):
        bind_node = d.get('bind')
        if isinstance(bind_node, dict):
            ## remove the part before the first _ and after the second to last _ to get the curr_node_submodule name
            ## for example, grp_VITIS_LOOP_5859_1_proc31_fu_82 -> VITIS_LOOP_5859_1_proc31
            curr_node_full_name = d.get('name')
            if curr_node_full_name is None:
                #debug_print(f"current node name full is None")
                exit(1)
                continue
            curr_node_submodule_name = '_'.join(curr_node_full_name.split('_')[1:-2])
            #debug_print(f"current node submodule name: {curr_node_submodule_name}")

            if bind_node.get('fcode') == 'call' and curr_node_submodule_name == submodule_name:
                submodule_call_nodes.append(n)


    debug_print(f"Number of nodes in full netlist before merge: {full_netlist.number_of_nodes()}")
    debug_print(f"Number of nodes in submodule netlist: {submodule_netlist.number_of_nodes()}")


    new_full_netlist = nx.compose(full_netlist, submodule_netlist)

    #debug_print(f"Composed full netlist with submodule netlist for {submodule_name}.")

    ## read in the stg file for this submodule to get the input/output port names
    stg_file_path = os.path.join(root_dir, submodule_name, f"{submodule_name}.verbose_stg.rpt")
    if not os.path.exists(stg_file_path): 
        stg_file_path = os.path.join(root_dir, submodule_name + "s", f"{submodule_name}s.verbose_stg.rpt")
        if not os.path.exists(stg_file_path):
            print(f"Error: STG file {stg_file_path} does not exist.")
            exit(1)
            return
    with open(stg_file_path, 'r') as sf:
        stg_lines = sf.readlines()

    pin_to_port = parse_stg_ports(stg_lines)

    ## TODO: Handle the netlist edges across the submodule boundaries

    ## first step is to find all of the predecessor and sucessor nodes of the first submodule call node (the rest are duplicates of the param info)
    if len(submodule_call_nodes) == 0:
        debug_print(f"No call nodes found for submodule {submodule_name} in module {current_module}.")
        return new_full_netlist
    
    first_call_node = submodule_call_nodes[0]

    ## the predecessors of the call node are the dependences that need to be reconnected after the merge
    predecessors = list(full_netlist.predecessors(first_call_node))

    ## find the edge data for each predecessor edge to get the pin number
    predecessor_edges = {}
    for pred in predecessors:
        edge_data = full_netlist.get_edge_data(pred, first_call_node)
        if edge_data is not None:
            predecessor_edges[pred] = edge_data
        else:
            debug_print(f"ERROR: No edge data found from predecessor {pred} to call node {first_call_node}.")
            exit(1)
            continue

    ## go through each predecessor and add an edge from it to another node in the graph with the same name field
    for pred in predecessors:
        pred_data = full_netlist.nodes[pred]
        pred_name = pred_data.get('name')
        if pred_name is None:
            debug_print(f"Predecessor node {pred} has no name field.")
            exit(1)
            continue

        # find the corresponding node in the submodule netlist
        target_node = None
        for n, d in submodule_netlist.nodes(data=True):
            if d.get('name') == pred_name:
                target_node = n
                break

        if target_node is not None:
            # add an edge from the predecessor to the target node in the new full netlist
            new_full_netlist.add_edge(pred, target_node)
            debug_print(f"Added edge from predecessor {pred} to target node {target_node} in submodule {submodule_name}.")
        else:
            debug_print(f"No matching node found in submodule {submodule_name} for predecessor {pred} with name {pred_name}.")
            ## we will need to use STG matching instead:
            # find the pin number for this predecessor node. This will be encoded in the edge data
            edge_data = predecessor_edges[pred]
            if edge_data is None:
                debug_print(f"No edge data found from predecessor {pred} to call node {first_call_node}.")
                exit(1)
                continue
            pin_num = edge_data.get('sink_pin')
            if pin_num is None:
                debug_print(f"No sink_pin number found in edge data from predecessor {pred} to call node {first_call_node}.")
                exit(1)
                continue
            port_name = pin_to_port.get(pin_num)
            if port_name is None:
                debug_print(f"No port name found for pin number {pin_num} in submodule {submodule_name}.")
                exit(1)
                continue
            # find the node in the submodule netlist with this port name
            target_node = None
            for n, d in submodule_netlist.nodes(data=True):
                if d.get('name') == port_name:
                    target_node = n
                    break
            if target_node is not None:
                new_full_netlist.add_edge(pred, target_node)
                debug_print(f"Added edge from predecessor {pred} to target node {target_node} in submodule {submodule_name} using STG port name {port_name}.")
            else:
                debug_print(f"No matching node found in submodule {submodule_name} for predecessor {pred} with STG port name {port_name}.")
                if port_name == "Return":
                    debug_print(f"Skipping Return port connection for predecessor {pred}.")
                    continue
                exit(1)
                continue

    return new_full_netlist

def parse_stg_ports(stg_lines):
    """
    Parse the STG file lines and return a mapping from pin number to port name.
    Pin numbers start at 0, and each Port line is assigned the next pin number.
    """
    pin_to_port = {}
    pin_num = 0
    for line in stg_lines:
        line = line.strip()
        # Match lines like: Port [ port_name ] ...
        if line.startswith("Port ["):
            # Extract port name between brackets
            import re
            match = re.match(r'Port\s*\[\s*([^\]]+)\s*\]', line)
            if match:
                port_name = match.group(1).strip()
                pin_to_port[pin_num] = port_name
                pin_num += 1
    return pin_to_port

# Example usage:
# with open(stg_file_path, 'r') as sf:
#     stg_lines = sf.readlines()
# pin_to_port = parse_stg_ports(stg_lines)

def main(root_dir, top_level_module_name, filter_nodes=True):
    merge_netlists_vitis(root_dir, top_level_module_name, filter_nodes)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python vitis_merge_netlists.py <root_directory> <top_level_module_name>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])