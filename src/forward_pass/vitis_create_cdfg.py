import os
import sys
import json
import networkx as nx
import logging

logger = logging.getLogger(__name__)

## Enables additional logging when set to True
DEBUG = False

def debug_print(message):
    if DEBUG:
        logger.info(message)


## NOTE: The data dependencies in this function are represeted as edges in the graph where the source node 
# is the operation that produces the data and the destination node is the operation that consumes the data.
def create_cdfg_one_file(fsm_data, state_transitions, stg_data, dir_name):

    debug_print(f"Creating CDFG for {dir_name} with {len(fsm_data)} states and {len(state_transitions)} transitions.")
    ## First, identify if there is a loop in the FSM.
    fsm_graph = nx.DiGraph()
    for state, next_state in state_transitions.items():
        if next_state == -1:
            continue
        fsm_graph.add_edge(state, next_state)

    dag_cycles = []
    cycles_present = False
    states_visited = set()

    if not nx.is_directed_acyclic_graph(fsm_graph):
        cycles_present = True
        debug_print(f"Detected a loop in the FSM: {dir_name}")
        ## print out which states are involved in the loop
        dag_cycles = list(nx.simple_cycles(fsm_graph))
        for cycle in dag_cycles:
            debug_print(f"Cycle detected: {cycle}")

    ## When there is a cycle, it means that we'll need to keep track of program
    ## variables to determine the CDFG. These will be stored here
    # control_vars = {}

    # Create the graph
    G = nx.DiGraph()

    curr_state = 1  # Start at state 1
    curr_step_count_local = 0 ## the step count (1 FSM state complete = 1 state != 1 clock cycle)
    ## it is "local" because it is only valid within the CDFG for this specific FSM file. 

    ## start at state 1
    if curr_state not in fsm_data:
        debug_print(f"State '{curr_state}' (start state) not found in {dir_name}. Cannot parse this file.")
        return

    if curr_state not in state_transitions:
        debug_print(f"State '{curr_state}' (start state) not found in {dir_name}. Cannot parse this file.")
        debug_print("State transitions file:", state_transitions)
        return

    prev_state_end_node = None

    ## Create a start and end node for the entire CDFG.
    cdfg_start_node_name = f"START_CDFG_{dir_name}"
    cdfg_end_node_name = f"END_CDFG_{dir_name}"
    G.add_node(cdfg_start_node_name, type='CONTROL_NODE')
    G.add_node(cdfg_end_node_name, type='CONTROL_NODE')

    instantiated_modules = []  ## This will keep track of the submodules that have been used

    # determine submodules independently of FSM
    for state in fsm_data:
        for fsm_node in fsm_data[state]:
            if fsm_node['operator'] == 'call':
                ## if the operator is a call, we will treat it as a submodule and add it to the graph
                module_name = fsm_node['function']
                if module_name not in instantiated_modules:
                    instantiated_modules.append(module_name)

    ## This loop will continue until we have followed the state transition flow through all states in the FSM.
    while curr_state in fsm_data:
        ## Iterate through all FSM nodes in the current state.
        #print(f"Processing FSM state: {curr_state} with step count: {curr_step_count_local}")

        ## if there are cycles in the FSM and this state is one of the states in the cycle, we will
        ## need to handle it differently
        curr_state_in_cycle = False
        if cycles_present:
            for cycle in dag_cycles:
                if curr_state in cycle:
                    curr_state_in_cycle = True
                    break

        if curr_state_in_cycle and curr_state in states_visited:
            break ## if we are in a cycle and we have already visited this state, we will break out of the loop

        states_visited.add(curr_state)


        ## create the start and end nodes for the current state
        start_node_name = f"START_CYCLE_{curr_state}_{curr_step_count_local}"
        end_node_name = f"END_CYCLE_{curr_state}_{curr_step_count_local}"

        G.add_node(start_node_name, state=curr_state, cycle=curr_step_count_local, type='CONTROL_NODE')
        G.add_node(end_node_name, state=curr_state, cycle=curr_step_count_local, type='CONTROL_NODE')

        if prev_state_end_node is not None:
            ## if there is a previous state end node, we will connect the start node of this state to the end node of the previous state
            G.add_edge(prev_state_end_node, start_node_name, type='CONTROL_DEPENDENCY')
            #debug_print(f"Added control edge from {prev_state_end_node} to {start_node_name} (previous state end to current state start)")
        else:
            ## if there is no previous state end node, we will connect the start node of this state to the CDFG start node
            G.add_edge(cdfg_start_node_name, start_node_name, type='CONTROL_DEPENDENCY')
            #debug_print(f"Added control edge from {cdfg_start_node_name} to {start_node_name} (CDFG start to current state start)")

        nodes_added_in_this_state = []

        ## Add nodes for each op in this state. 
        for fsm_node in fsm_data[curr_state]:
            #print(f"Processing FSM node: {fsm_node} in state {curr_state}")
            curr_node_name = f"{fsm_node['operator']}_op_{curr_step_count_local}_{fsm_node['destination']}"
            G.add_node(curr_node_name, fsm_node=fsm_node, start_state=int(curr_state), start_step_count=curr_step_count_local)
            nodes_added_in_this_state.append(curr_node_name)

        ## add edges based on the data dependencies
        ## go through the sources of all nodes added in this state and add a dependency edge to that node
        for node_name in nodes_added_in_this_state:
            fsm_node = G.nodes[node_name]['fsm_node']
            if 'sources' in fsm_node:
                for source in fsm_node['sources']:
                    ## if the source is directly parsable as an integer, then it is a constant and 
                    ## not a true dependency
                    if isinstance(source['source'], int):
                        continue
                    ## otherwise, if the source starts with an '@', we will assume it is also a constant 
                    ## and not a true dependency
                    elif isinstance(source['source'], str) and source['source'].startswith('@'):
                        continue
                    ## otherwise, if the source starts with a '%', then it is a variable and we will add it as a dependency
                    elif isinstance(source['source'], str) and source['source'].startswith('%'):
                        ## see if the source is currently in the list of nodes added in this state
                        for added_node in nodes_added_in_this_state:
                            if G.nodes[added_node]['fsm_node']['destination'] == source['source']:
                                G.add_edge(node_name, added_node, type='DATA_DEPENDENCY', var_name=source['source'])
                                #debug_print(f"Added edge from {node_name} to {added_node} based on source {source['source']}")

        ## Add control dependency edges
        for node_name in nodes_added_in_this_state:
            ## if the node doesn't rely on any previous data, we will connect it to the start node. 
            ## to determine this, we will need to go through the incoming edges of this node and see
            ## if any are data dependencies
            has_data_dependency = False
            for src, dst, attrs in G.in_edges(node_name, data=True):
                #debug_print(f"Checking edge from {src} to {dst} with attributes {attrs}")
                if attrs.get('type') == 'DATA_DEPENDENCY':
                    has_data_dependency = True
                    break
            if not has_data_dependency:
                G.add_edge(start_node_name, node_name, type='CONTROL_DEPENDENCY')
                #debug_print(f"Added control edge from {node_name} to {start_node_name} (no data dependencies)")
            #debug_print(f"Done checking......")
            ## If the node doesn't produce any data that is consumed by another node, we will connect it to the end node.
            ## to determine this, we will need to go through the outgoing edges of this node and see
            ## if any are data dependencies
            has_output_dependency = False
            for src, dst, attrs in G.out_edges(node_name, data=True):
                #debug_print(f"Checking edge from {src} to {dst} with attributes {attrs}")
                if attrs.get('type') == 'DATA_DEPENDENCY':
                    has_output_dependency = True
                    break
            if not has_output_dependency:
                G.add_edge(node_name, end_node_name, type='CONTROL_DEPENDENCY')
                #debug_print(f"Added control edge from {end_node_name} to {node_name} (no data produced)")

        ## go to the next state
        if curr_state in state_transitions:
            curr_state = state_transitions[curr_state]
            curr_step_count_local += 1
        else:
            print(f"No transition found for state {curr_state}. Ending processing.")
            break

        prev_state_end_node = end_node_name

    ## After processing all states, we will connect the end node of the last state to the CDFG end node.
    if prev_state_end_node is not None:
        G.add_edge(prev_state_end_node, cdfg_end_node_name, type='CONTROL_DEPENDENCY')
        #debug_print(f"Added control edge from {prev_state_end_node} to {cdfg_end_node_name} (last state end to CDFG end)")

    for node in G.nodes:
        ## We need to check each input to each node to see if it is a variable that is an input to the CDFG.
        if 'fsm_node' in G.nodes[node]:
            fsm_node = G.nodes[node]['fsm_node']

            ## All nodes that have an data dependency on an input to the CDFG should be connected to the start node.
            if 'sources' in fsm_node:
                for source in fsm_node['sources']:
                    if isinstance(source['source'], str) and source['source'].startswith('%'):
                        ## If the source is a variable that is an input to the CDFG, we will connect it to the start node.
                        source_var = source['source'].strip('%')
                        for input_var in stg_data.get('inputs', []):
                            if input_var['name'] == source_var:
                                G.add_edge(cdfg_start_node_name, node, type='INPUT_DEPENDENCY', var_name=source_var)
                                #debug_print(f"Added input dependency edge from {cdfg_start_node_name} to {node} for variable {source_var}")

            ## All nodes that produce an output that is an output of the CDFG should be connected to the end node.
            if 'destination' in fsm_node:
                output_var = fsm_node['destination'].strip('%')
                for output_var_info in stg_data.get('outputs', []):
                    if output_var_info['name'] == output_var:
                        G.add_edge(node, cdfg_end_node_name, type='OUTPUT_DEPENDENCY', var_name=output_var)
                        #debug_print(f"Added output dependency edge from {node} to {cdfg_end_node_name} for variable {output_var}")
            

    return G, instantiated_modules


def prune_to_functional_unit_graph(G: nx.DiGraph,
                      target_ops={"add", "mul", "call"},
                      op_attr="operator") -> nx.DiGraph:
    """
    Return a new DiGraph H that contains only the 'functional unit' nodes
    (operator in target_ops) from G, as well as any node with type 'CONTROL_NODE'.
    H has an edge u->v iff in G there exists a directed path u -> ... -> v whose internal nodes are all NON-target
    (i.e., no internal node has operator in target_ops or type 'CONTROL_NODE').
    Each node in H will have the full fsm_node dictionary as an attribute.
    """
    # Identify target nodes and control nodes
    is_target = {
        n: (
            G.nodes[n].get('fsm_node', {}).get(op_attr) in target_ops
            or G.nodes[n].get('type') == 'CONTROL_NODE'
        )
        for n in G.nodes
    }
    targets = [n for n, t in is_target.items() if t]

    # Initialize the pruned graph with the target nodes and full fsm_node dict
    H = nx.DiGraph()
    for n in targets:
        attrs = {}
        if 'fsm_node' in G.nodes[n]:
            attrs['fsm_node'] = G.nodes[n]['fsm_node']
        if 'start_state' in G.nodes[n]:
            attrs['start_state'] = G.nodes[n]['start_state']
        if 'start_step_count' in G.nodes[n]:
            attrs['start_step_count'] = G.nodes[n]['start_step_count']
        if 'type' in G.nodes[n]:
            attrs['type'] = G.nodes[n]['type']
        H.add_node(n, **attrs)

    # For each target, walk outward through only non-target nodes.
    for u in targets:
        stack = list(G.successors(u))
        visited = set()

        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)

            if is_target.get(x, False):
                H.add_edge(u, x)
                continue

            stack.extend(G.successors(x))

    return H

def create_cdfg_vitis(root_dir):
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        debug_print(f"Processing directory: {subdir_path}")

        # Find _fsm.json and _state_transitions.json files
        fsm_file = None
        transitions_file = None
        for fname in os.listdir(subdir_path):
            if fname.endswith('_fsm.json'):
                fsm_file = os.path.join(subdir_path, fname)
            elif fname.endswith('_state_transitions.json'):
                transitions_file = os.path.join(subdir_path, fname)
            elif fname.endswith('_STG_IN_OUT.json'):
                stg_file = os.path.join(subdir_path, fname)

        if not fsm_file or not transitions_file:
            continue

        

        # Load FSM and transitions, converting top-level keys to integers
        with open(fsm_file, 'r') as f:
            fsm_data_raw = json.load(f)
            fsm_data = {int(k): v for k, v in fsm_data_raw.items()}
        with open(transitions_file, 'r') as f:
            state_transitions_raw = json.load(f)
            state_transitions = {int(k): v for k, v in state_transitions_raw.items()}

        # Load the STG data
        with open(stg_file, 'r') as f:
            stg_data = json.load(f)

        result = create_cdfg_one_file(fsm_data, state_transitions, stg_data, subdir_path)

        if result is not None:
            G, module_dependences = result

        if G is None:
            debug_print(f"Graph creation failed for {fsm_file}. Skipping.")
            continue

        ## Save the full graph as a .gml file with the name based on the fsm file
        gml_file = fsm_file.replace('_fsm.json', '_unpruned_cdfg.gml')
        nx.write_gml(G, gml_file)
        debug_print(f"Created GML graph: {gml_file} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

        ## prune the graph down to only include the functional unit ops.
        pruned_G = prune_to_functional_unit_graph(G)

        # Save the pruned graph as a .gml file with the name based on the fsm file
        gml_file = fsm_file.replace('_fsm.json', '_cdfg.gml')
        nx.write_gml(pruned_G, gml_file)
        debug_print(f"Created GML graph: {gml_file} with {pruned_G.number_of_nodes()} nodes and {pruned_G.number_of_edges()} edges.")

        if not module_dependences:
            module_dependences = []

        module_file = fsm_file.replace('_fsm.json', '_modules.json')
        with open(module_file, 'w') as mf:
            json.dump(module_dependences, mf, indent=4)
        debug_print(f"Created module dependencies file: {module_file} with {len(module_dependences)} modules.")


def main(root_dir):
    create_cdfg_vitis(root_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_cdfg.py <root_directory>")
        sys.exit(1)
    main(sys.argv[1])