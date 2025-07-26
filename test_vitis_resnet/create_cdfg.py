import os
import sys
import json
import networkx as nx


DEBUG = True

def debug_print(message):
    if DEBUG:
        print(message)


## NOTE: The data dependencies in this function are represeted as edges in the graph where the source node 
# is the operation that produces the data and the destination node is the operation that consumes the data.
def create_cdfg_one_file(fsm_data, state_transitions, stg_data, dir_name):
    ## First, identify if there is a loop in the FSM.
    fsm_graph = nx.DiGraph()
    for state, next_state in state_transitions.items():
        if next_state == -1:
            continue
        fsm_graph.add_edge(state, next_state)

    dag_cycles = []
    cycles_present = False

    if not nx.is_directed_acyclic_graph(fsm_graph):
        cycles_present = True
        debug_print(f"Detected a loop in the FSM: {dir_name}")
        ## print out which states are involved in the loop
        dag_cycles = list(nx.simple_cycles(fsm_graph))
        for cycle in dag_cycles:
            debug_print(f"Cycle detected: {cycle}")
        return

    ## When there is a cycle, it means that we'll need to keep track of program
    ## variables to determine the CDFG. These will be stored here
    control_vars = {}

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

    ## This loop will continue until we have followed the state transition flow through all states in the FSM.
    while curr_state in fsm_data:
        ## Iterate through all FSM nodes in the current state.
        #print(f"Processing FSM state: {curr_state} with step count: {curr_step_count_local}")

        ## if there are cycles in the FSM and this state is one of the states in the cycle, we will
        ## need to handle it differently
        # curr_state_in_cycle = False
        # if cycles_present:
        #     for cycle in dag_cycles:
        #         if curr_state in cycle:
        #             curr_state_in_cycle = True
        #             break


        ## create the start and end nodes for the current state
        start_node_name = f"START_{curr_state}_{curr_step_count_local}"
        end_node_name = f"END_{curr_state}_{curr_step_count_local}"

        G.add_node(start_node_name, state=curr_state, cycle=curr_step_count_local, type='CONTROL_NODE')
        G.add_node(end_node_name, state=curr_state, cycle=curr_step_count_local, type='CONTROL_NODE')

        nodes_added_in_this_state = []

        for fsm_node in fsm_data[curr_state]:
            #print(f"Processing FSM node: {fsm_node} in state {curr_state}")
            
            # if curr_state_in_cycle:
            #     ## see if it is a phi op. If it is, do the operation.
            #     # if fsm_node["operator"] == "phi":
            #     #     if fsm_node["destination"] not in control_vars:
            #     #         control_vars[fsm_node["destination"]] = int(fsm_node["sources"][0]["source"])
            #     #     else:
            #     #         control_vars[fsm_node["destination"]] = control_vars[fsm_node["sources"][2]["source"]]
                
            #     # ## otherwise, see if it is an icmp_eq op. If it is, do that operation.
            #     # elif fsm_node["operator"] == "icmp_eq":
            #     #     control_vars[fsm_node["destination"]] = 1 if control_vars[fsm_node["sources"][0]["source"]] == control_vars[fsm_node["sources"][1]["source"]] else 0

            #     ## otherwise, see if it is
            # else:
            ## if we are on the first step of a multi step operation (or a single step operation), we will add the node to the graph
            # if fsm_node["steps_remaining"] == fsm_node["total_steps"]:
                curr_node_name = f"{fsm_node['operator']}_op_{curr_step_count_local}_{fsm_node['destination']}"
                G.add_node(curr_node_name, fsm_node=fsm_node, start_state=int(curr_state), start_step_count=curr_step_count_local)
                nodes_added_in_this_state.append(curr_node_name)
            # else:
            #     ## we need to find the node in the graph that corresponds to the current operation
            #     continue
            ## first, look at the sources for this operation
            # if "sources" in fsm_node:
            #     for source in fsm_node["sources"]:
            #         ## if the source is directly parsable as in integer, then it is a constant and 
            #         ## not a true dependency
            #         if isinstance(source, int):
            #             continue
            #         ## otherwise, if the source starts with an '@', we will assume it is also a constant 
            #         ## and not a true dependency
            #         elif isinstance(source, str) and source.startswith('@'):
            #             continue
            #         ## otherwise, if the source starts with a % then it is a variable and we will add it as a dependency
            #         elif isinstance(source, str) and source.startswith('%'):
            #             ## see if the source is currently in the graph
            #             if source in G:
            #                 G.add_edge(source, f"{curr_state}_op_{idx}")
                ## then, add the operation node itself

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
                                debug_print(f"Added edge from {node_name} to {added_node} based on source {source['source']}")

        ## Add control dependency edges
        for node_name in nodes_added_in_this_state:
            ## if the node doesn't rely on any previous data, we will connect it to the start node. 
            ## this means it has an in degree of 0
            if G.in_degree(node_name) == 0:
                G.add_edge(node_name, start_node_name, type='CONTROL_DEPENDENCY')
                debug_print(f"Added control edge from {node_name} to {start_node_name} (no data dependencies)")

            ## If the node doesn't produce any data that is consumed by another node, we will connect it to the end node.
            ## this means it has an out degree of 0
            if G.out_degree(node_name) == 0:
                G.add_edge(end_node_name, node_name, type='CONTROL_DEPENDENCY')
                debug_print(f"Added control edge from {end_node_name} to {node_name} (no data produced)")

        ## go to the next state
        if curr_state in state_transitions:
            curr_state = state_transitions[curr_state]
            curr_step_count_local += 1
        else:
            print(f"No transition found for state {curr_state}. Ending processing.")
            break

    ## TODO: At the end, create a start and end node for the entire CDFG. 

    ## TODO: All nodes that have an data dependency on an input to the CDFG should be connected to the start node.

    ## TODO: All nodes that produce an output that is an output of the CDFG should be connected to the end node.
    
    return G

def main(root_dir):
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        #print(f"Processing directory: {subdir_path}")

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

        G = create_cdfg_one_file(fsm_data, state_transitions, stg_data, subdir_path)

        if G is None:
            debug_print(f"Graph creation failed for {fsm_file}. Skipping.")
            continue

        # Save the graph as a .gml file with the name based on the fsm file
        gml_file = fsm_file.replace('_fsm.json', '_cdfg.gml')
        nx.write_gml(G, gml_file)
        #print(f"Created GML graph: {gml_file} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_cdfg.py <root_directory>")
        sys.exit(1)
    main(sys.argv[1])