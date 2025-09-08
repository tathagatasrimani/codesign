import os
import sys
import json
import networkx as nx


DEBUG = True

def debug_print(message):
    if DEBUG:
        print(message)


all_modules_visited = set()

def merge_netlists_vitis(root_dir, top_level_module_name):
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
    output_file_path = os.path.join(root_dir, f"{top_level_module_name}_full_netlist.gml")
    nx.write_gml(full_netlist, output_file_path)
    #debug_print(f"Full netlist saved to {output_file_path}")


def parse_module(root_dir, current_module):

    debug_print(f"!!!!!!!!!!!!!!!!!!!!!!!Parsing module for netlist merge: {current_module}")

    ## check if the current module has already been visited
    # if it has, we can read the netlist from the file and return it.
    if current_module in all_modules_visited:
        #debug_print(f"Module {current_module} already visited. Returning existing netlist.")
        netlist_file_path = os.path.join(root_dir, current_module, f"{current_module}_full_netlist.gml")
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


    ## TODO: Handle the netlist edges across the submodule boundaries

    ## first step is to find all of the predecessor and sucessor nodes of the first submodule call node (the rest are duplicates of the param info)
    if len(submodule_call_nodes) == 0:
        debug_print(f"No call nodes found for submodule {submodule_name} in module {current_module}.")
        return full_netlist
    
    first_call_node = submodule_call_nodes[0]
    predecessors = list(full_netlist.predecessors(first_call_node))
    successors = list(full_netlist.successors(first_call_node))
    


    debug_print(f"Number of nodes in full netlist before merge: {full_netlist.number_of_nodes()}")
    debug_print(f"Number of nodes in submodule netlist: {submodule_netlist.number_of_nodes()}")


    new_full_netlist = nx.compose(full_netlist, submodule_netlist)

    #debug_print(f"Composed full netlist with submodule netlist for {submodule_name}.")

    # intermediate start and stop nodes in the submodule netlist should be preserved, as they are part of the submodule's internal flow.

    return new_full_netlist

def main(root_dir, top_level_module_name):
    merge_netlists_vitis(root_dir, top_level_module_name)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python vitis_merge_netlists.py <root_directory> <top_level_module_name>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])