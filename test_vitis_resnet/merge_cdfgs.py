import os
import sys
import json
import networkx as nx


DEBUG = True

def debug_print(message):
    if DEBUG:
        print(message)


all_modules_visited = set()


def main(root_dir, top_level_module_name):
    """
    Main function to create CDFG for all files in the given directory.
    """
    full_cdfg = nx.DiGraph()

    ## get the names of all the submodules (the subdirectories)
    submodules = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
    debug_print(f"Submodules found: {submodules}")

    ## start with the top-level module
    if top_level_module_name not in submodules:
        print(f"Error: Top-level module {top_level_module_name} does not exist in {root_dir}.")
        return
    
    full_cdfg = parse_module(root_dir, top_level_module_name)
    if full_cdfg is None:
        print(f"Error: Failed to parse the top-level module {top_level_module_name}.")
        return
    
    ## print out the visited modules
    debug_print(f"Visited modules: {all_modules_visited}")
    ## print out the number of visited modules
    debug_print(f"Number of visited modules: {len(all_modules_visited)}")
    
    ## save the full CDFG to a file
    output_file_path = os.path.join(root_dir, f"{top_level_module_name}_full_cdfg.gml")
    nx.write_gml(full_cdfg, output_file_path)
    debug_print(f"Full CDFG saved to {output_file_path}")


def parse_module(root_dir, current_module):

    debug_print(f"Parsing module: {current_module}")

    ## add module to the vitised modules list
    all_modules_visited.add(current_module)

    ## open the _cdfg.gml file for the current module. Read it in as a NetworkX graph.
    cdfg_file_path = os.path.join(root_dir, current_module, f"{current_module}.verbose_cdfg.gml")

    append_s_to_path = False
    if not os.path.exists(cdfg_file_path):
        cdfg_file_path = os.path.join(root_dir, current_module + "s", f"{current_module}s.verbose_cdfg.gml")
        if not os.path.exists(cdfg_file_path):
            print(f"Error: CDFG file {cdfg_file_path} does not exist.")
            exit(1)
            return
        else:
            current_module += "s"

    full_cdfg = nx.read_gml(cdfg_file_path)

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

    if module_dependences is None or len(module_dependences) == 0:
        debug_print(f"No submodules found for {current_module}. Returning CDFG as is.")

    ## get the CDFGs for each of the instantiated modules recursively
    submodule_cdfgs = {}
    for module_name in module_dependences:
        submodule_cdfgs[module_name] = parse_module(root_dir, module_name)

    ## merge the CDFGs of the submodules into the full CDFG
    for submodule_name, submodule_cdfg in submodule_cdfgs.items():
        if submodule_cdfg is not None:
            #full_cdfg = nx.compose(full_cdfg, submodule_cdfg)
            full_cdfg = merge_cdfgs(full_cdfg, submodule_cdfg, submodule_name)
            debug_print(f"Merged CDFG for submodule {submodule_name} into full CDFG.")

    ## write out the full CDFG for the current module for debugging purposes
    output_file_path = os.path.join(root_dir, current_module, f"{current_module}_full_cdfg.gml")
    nx.write_gml(full_cdfg, output_file_path)
    debug_print(f"Full CDFG for module {current_module} saved to {output_file_path}")

    return full_cdfg

def get_submodule_cycle_count(submodule_cdfg, sub_cdfg_start_node):
    """
    Get the cycle count of the submodule CDFG by counting the number of cycles between the start and exit nodes.
    """
    cycle_count = 0

    # Traverse from the start node to the exit node, counting cycles
    for node in nx.bfs_tree(submodule_cdfg, source=sub_cdfg_start_node):
        if submodule_cdfg.nodes[node].get('type') == 'CONTROL_NODE':
            if 'CYCLE_START' in node:
                cycle_count += 1

    debug_print(f"Submodule CDFG cycle count: {cycle_count}")
    return cycle_count


def remove_start_end_cycle_nodes(submodule_cdfg, sub_cdfg_start_node, sub_cdfg_exit_node, submodule_name):

    # start at the START_CDFG_ node in the graph and go through its successors. Look for the START_CYCLE node.
    start_cycle_node_to_remove = None
    for node in submodule_cdfg.successors(sub_cdfg_start_node):
        if submodule_cdfg.nodes[node].get('type') == 'CONTROL_NODE' and 'START_CYCLE' in node:
            start_cycle_node_to_remove = node
            break

    if start_cycle_node_to_remove is None:
        debug_print(f"Error: No START_CYCLE node found in submodule CDFG {submodule_name}.")
        ## print successors of the START_CDFG node
        successors = list(submodule_cdfg.successors(sub_cdfg_start_node))
        debug_print(f"Successors of START_CDFG node {sub_cdfg_start_node}: {successors}")

        exit(1)
    else:
        debug_print(f"Found START_CYCLE node to remove: {start_cycle_node_to_remove}")

        ## get all of the successors of the START_CYCLE node and connect them to the submodule CDFG start node
        successors = list(submodule_cdfg.successors(start_cycle_node_to_remove))
        debug_print(f"Successors of START_CYCLE node {start_cycle_node_to_remove}: {successors}")
        for succ in successors:
            submodule_cdfg.add_edge(sub_cdfg_start_node, succ)
        
        # remove the START_CYCLE node from the submodule CDFG
        submodule_cdfg.remove_node(start_cycle_node_to_remove)
        debug_print(f"Removed START_CYCLE node {start_cycle_node_to_remove} from submodule CDFG {submodule_name}.")
    
    # start at the END_CDFG_ node in the graph and go through its predecessors. Look for the END_CYCLE node.
    end_cycle_node_to_remove = None
    for node in submodule_cdfg.predecessors(sub_cdfg_exit_node):
        if submodule_cdfg.nodes[node].get('type') == 'CONTROL_NODE' and 'END_CYCLE' in node:
            end_cycle_node_to_remove = node
            break
    
    if end_cycle_node_to_remove is None:
        debug_print(f"Error: No END_CYCLE node found in submodule CDFG {submodule_name}.")
        exit(1)
    else:
        debug_print(f"Found END_CYCLE node to remove: {end_cycle_node_to_remove}")

        ## get all of the predecessors of the END_CYCLE node and connect them to the submodule CDFG exit node
        predecessors = list(submodule_cdfg.predecessors(end_cycle_node_to_remove))
        debug_print(f"Predecessors of END_CYCLE node {end_cycle_node_to_remove}: {predecessors}")
        for pred in predecessors:
            submodule_cdfg.add_edge(pred, sub_cdfg_exit_node)

        # remove the END_CYCLE node from the submodule CDFG
        submodule_cdfg.remove_node(end_cycle_node_to_remove)
        debug_print(f"Removed END_CYCLE node {end_cycle_node_to_remove} from submodule CDFG {submodule_name}.")

        return submodule_cdfg
    
def find_cycle_start_node(full_cdfg, start_search_node):
    """
    Find the cycle start node prior to the provided node.
    This is done by traversing the graph backwards from the provided call node.
    """
    node_queue = [start_search_node]
    visited = set()
    cycle_start_node = None

    while node_queue:
        current_node = node_queue.pop(0)
        if current_node in visited:
            continue
        visited.add(current_node)

        for pred in full_cdfg.predecessors(current_node):
            if full_cdfg.nodes[pred].get('type') == 'CONTROL_NODE' and 'CYCLE_START' in pred:
                cycle_start_node = pred
                break
            if pred not in visited:
                node_queue.append(pred)

    return cycle_start_node

def find_cycle_end_node(full_cdfg, start_search_node):
    """
    Find the cycle end node after the provided node.
    This is done by traversing the graph forwards from the provided call node.
    """
    node_queue = [start_search_node]
    visited = set()
    cycle_end_node = None

    while node_queue:
        current_node = node_queue.pop(0)
        if current_node in visited:
            continue
        visited.add(current_node)

        for succ in full_cdfg.successors(current_node):
            if full_cdfg.nodes[succ].get('type') == 'CONTROL_NODE' and 'CYCLE_END' in succ:
                cycle_end_node = succ
                break
            if succ not in visited:
                node_queue.append(succ)

    return cycle_end_node

def merge_cdfgs(full_cdfg, submodule_cdfg, submodule_name):
    """
    Merge the submodule CDFG into the full CDFG.
    """
    debug_print(f"Merging CDFG for submodule {submodule_name} into full CDFG.")

    # Find entry and exit nodes in the submodule CDFG (Should only be one of each)
    # Assume there is only one entry and one exit node (this should be the case)
    sub_cdfg_start_node = next(n for n, d in submodule_cdfg.nodes(data=True) if d.get('type') == 'CONTROL_NODE' and 'START_CDFG' in n)
    sub_cdfg_exit_node = next(n for n, d in submodule_cdfg.nodes(data=True) if d.get('type') == 'CONTROL_NODE' and 'END_CDFG' in n)

    if sub_cdfg_start_node is None or sub_cdfg_exit_node is None:
        debug_print(f"Error: Could not find START_CDFG or END_CDFG nodes in submodule CDFG {submodule_name}.")
        exit(1)


    ## First, remove the first START_CYCLE and the last END_CYCLE nodes from the submodule CDFG. 
    submodule_cdfg = remove_start_end_cycle_nodes(submodule_cdfg, sub_cdfg_start_node, sub_cdfg_exit_node, submodule_name)
    
    # Find all nodes in the full CDFG that are call functions to the submodule
    submodule_call_nodes = []
    for n, d in full_cdfg.nodes(data=True):
        fsm_node = d.get('fsm_node')
        if isinstance(fsm_node, dict):
            if fsm_node.get('operator') == 'call' and fsm_node.get('function') == submodule_name:
                submodule_call_nodes.append(n)

    # See if there are duplicate nodes in the full CDFG that match the submodule name
    if len(submodule_call_nodes) > 1:
        debug_print(f"Found multiple call nodes for submodule {submodule_name}: {submodule_call_nodes}")

        ## not implemented yet. TODO: implement this.
        exit(1)
    elif len(submodule_call_nodes) == 0:
        debug_print(f"No call nodes found for submodule {submodule_name}, so no work to do.")
        return full_cdfg

    # if there are, check if they are part of the same submodule call. (i.e. multi-cycle ops)
    # if they are, remove all of them as a group, recording the inputs of the first node and the outputs of the last node.

    first_node_submodule_call = submodule_call_nodes[0]
    last_node_submodule_call = submodule_call_nodes[-1]

    ## find the cycle start node prior to the first submodule call node. It's possible that it isn't a direct predecessor, so we need to traverse the graph.
    ## essentially, we are doing a reverse BFS from the first node to find the first cycle start node.
    cycle_start_node_before_submodule_call = find_cycle_start_node(full_cdfg, first_node_submodule_call)

    ## find the cycle end node after the last submodule call node.
    cycle_end_node_after_submodule_call = find_cycle_end_node(full_cdfg, last_node_submodule_call)

    ## record input edges of the first node
    input_edges = list(full_cdfg.in_edges(first_node_submodule_call))
    debug_print(f"Input edges for first node {first_node_submodule_call}: {input_edges}")

    ## record output edges of the last node
    output_edges = list(full_cdfg.out_edges(last_node_submodule_call))
    debug_print(f"Output edges for last node {last_node_submodule_call}: {output_edges}")

    # remove all nodes in the submodule_call_nodes list from the full CDFG
    full_cdfg.remove_nodes_from(submodule_call_nodes)
    debug_print(f"Removed nodes {submodule_call_nodes} from full CDFG.")

    ## then, insert the submodule CDFG in place of the removed nodes, connecting the inputs and outputs to the first and last nodes respectively.
    new_full_cdfg = nx.compose(full_cdfg, submodule_cdfg)

    # Reconnect input edges to entry node
    for src, _ in input_edges:
        new_full_cdfg.add_edge(src, sub_cdfg_start_node)

    # Reconnect output edges from exit node
    for _, dst in output_edges:
        new_full_cdfg.add_edge(sub_cdfg_exit_node, dst)

    # intermediate start and stop nodes in the submodule CDFG should be preserved, as they are part of the submodule's internal flow.
    
    ## add stall nodes to the full CDFG to keep the timing correct between the start and stop nodes of the original CDFG (on other branches).
    
    # first, calculate the cycle count - 1 of the submodule CDFG. It is 1 less because the first start node is removed.
    submodule_cycle_count = get_submodule_cycle_count(submodule_cdfg, sub_cdfg_start_node)

    ## find all paths between the CYCLE_START and CYCLE_END nodes in the full CDFG 

    if cycle_start_node_before_submodule_call is not None and cycle_end_node_after_submodule_call is not None:
        all_paths = list(nx.all_simple_paths(
            new_full_cdfg,
            source=cycle_start_node_before_submodule_call,
            target=cycle_end_node_after_submodule_call
        ))
        debug_print(f"Found {len(all_paths)} paths between {cycle_start_node_before_submodule_call} and {cycle_end_node_after_submodule_call}")
        exit(0)
    else:
        debug_print("Could not find CYCLE_START or CYCLE_END node for path search.")
        exit(1)

    # NOTE: The call could be multi-cycle. In this case we would need to merge the intermediate cycle start and stop nodes.
    

    return full_cdfg

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_cdfgs.py <root_directory> <top_level_module_name>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])