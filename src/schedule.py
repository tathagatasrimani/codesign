import logging

logger = logging.getLogger(__name__)

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from . import sim_util

def get_longest_paths(G: nx.DiGraph, num_paths=5, num_unique_slacks=5):
    """
    Returns at most num_paths longest paths in G. num_unique_slacks can be optionally
    specified to remove any node from the graph which has too large of a slack.
    """
    # Topological sorting to process nodes in order
    top_order = list(nx.topological_sort(G))
    
    # Compute earliest arrival times (forward propagation)
    earliest = {node: 0 for node in G.nodes}
    earliest[top_order[0]] = 0  # Assuming first node is the start point
    
    for u in top_order:
        for v in G.successors(u):
            assert "weight" in G[u][v]
            weight = G[u][v].get("weight")
            earliest[v] = max(earliest[v], earliest[u] + weight)
    
    # Compute latest arrival times (backward propagation)
    latest = {node: float('inf') for node in G.nodes}
    latest[top_order[-1]] = earliest[top_order[-1]]  # Critical path end node
    
    for u in reversed(top_order):
        for v in G.successors(u):
            weight = G[u][v].get("weight")
            latest[u] = min(latest[u], latest[v] - weight)
    
    # Compute slacks
    slacks = {node: np.round(latest[node] - earliest[node], 5) for node in G.nodes}

    # only want to look at nodes with the worst slack. As a heuristic, take the k worst
    # unique slacks and disregard all other nodes
    sorted_slacks = sorted(slacks.keys(), key=lambda x: slacks[x])
    logger.info(f"slacks: {slacks}")
    unique_slacks = sorted(list(set(slacks.values())))
    logger.info(f"sorted slacks: {sorted_slacks}")
    k = min(num_unique_slacks, len(unique_slacks))
    nodes_to_remove = []
    for i in range(len(sorted_slacks)):
        if slacks[sorted_slacks[i]] > unique_slacks[k-1]:
            sorted_slacks = sorted_slacks[:i]
            nodes_to_remove = sorted_slacks[i:]
            break

    # create a copy of G with negative weights. Only consider nodes below some threshold of slack.
    # We will use this to run bellman ford, which can handle negative weights,
    # and find longest paths by negating the path lengths at the end.
    copy = G.copy()
    copy.add_node("start", weight=0)
    for node in nodes_to_remove:
        copy.remove_node(node)
    root_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]

    for edge in copy.edges():
        copy.edges[edge]["weight"] *= -1
    for node in root_nodes:
        logger.info(f"connecting start to {node}")
        copy.add_edge("start", node, weight=0)
    forward_paths = nx.single_source_bellman_ford(copy, "start")
    reverse_paths = nx.single_source_bellman_ford(copy.reverse(), "end")
    
    nodes_seen = set()
    i = 0
    paths = []
    for node in sorted_slacks:
        if node not in nodes_seen:
            # exclude "start" node and last node (already in reverse path) in forward path.
            # reverse the reverse path so that it goes from node -> "end"
            path = forward_paths[1][node][1:-1] + reverse_paths[1][node][::-1]
            length = forward_paths[0][node] + reverse_paths[0][node]
            for path_node in path:
                nodes_seen.add(path_node)
            paths.append([length, path])

    paths_sorted = sorted(paths, key=lambda x: x[0]) # sort by longest length
    for i in range(len(paths_sorted)):
        paths_sorted[i][0] *= -1
    
    # may only want a certain number of paths
    n = min(len(paths_sorted), num_paths)
    return paths_sorted[:n]

def sdc_schedule(graph, topo_order_by_elem, extra_constraints=[], add_resource_edges=False, debug=False):
    """
    Runs the convex optimization problem to minimize the longest path latency.
    Encodes data dependency constraints from CDFG and resource constraints using
    the SDC formulation published here https://www.csl.cornell.edu/~zhiruz/pdfs/sdc-dac2006.pdf.
    TODO: Add exact hw schema constraints as resource constraints. Right now we assume hardware is
    all to all.
    """
    constraints = []
    opt_vars = []
    graph_nodes = graph.nodes(data=True)
    id = 0
    if debug: sim_util.topological_layout_plot(graph)

    for node in graph_nodes:
        curr_var = cp.Variable(
            2, name=node[0]
        )  # first one is start time and last one is end time
        opt_vars.append(curr_var)
        # start time + latency = end time for each operating node
        node[1]["scheduling_id"] = id
        node[1]["allocation"] = ""
        id += 1
        constraints.append(curr_var[0] >= 0)
        if "cost" in node[1].keys():
            parents = list(graph.successors(node[0]))
            net_delay = 0
            num_net_delays = 0
            for parent in parents:
                edge_data = list(graph.edges([parent, node[0]]))[0]
                if "cost" in edge_data:
                    num_net_delays += 1
                    net_delay += edge_data["cost"]
            if num_net_delays: logger.info(f"adding net delay of {net_delay}")
            assert num_net_delays <= 1
            constraints.append(curr_var[0] + node[1]["cost"] + net_delay == curr_var[1])

    for node in graph_nodes:
        # constrain arithmetic ops to load from register right before they begin, and store right after to prevent value overwrites
        curr_var = opt_vars[node[1]["scheduling_id"]]
        successors = list(graph.successors(node[0]))
        op_count = 0
        for successor in successors:
            child = graph.nodes[successor]
            reg_to_arith_op = node[1]["function"] == "Regs" and child["function"] not in ["Regs", "end", "Buf"]
            main_mem_to_buf = node[1]["function"] == "MainMem" and child["function"] == "Buf"
            op_to_reg = node[1]["function"] not in ["Regs"] and child["function"] == "Regs"
            if reg_to_arith_op or op_to_reg or main_mem_to_buf:
                op_count += 1
                constraints += [curr_var[1] <= opt_vars[child["scheduling_id"]][0] + 0.00001, curr_var[1] >= opt_vars[child["scheduling_id"]][0] - 0.00001] # allow a slight variation to accommodate register topological order hacky fix
                logger.info(f"constraining {node[0]} to execute right before {successor}")
        assert op_count <= 1, f"each node should have at most 1 arithmetic op or reg child, violating node is {node[0]}"

    # data dependency constraints
    for u, v in graph.edges():
        source_id = int(graph_nodes[u]["scheduling_id"])
        dest_id = int(graph_nodes[v]["scheduling_id"])
        constraints.append(opt_vars[source_id][1] - opt_vars[dest_id][0] <= 0.0)

    resource_chain_edges = []
    resource_edge_graph = None
    if add_resource_edges or debug: resource_edge_graph = graph.copy()

    resource_constraints = []
    for elem in topo_order_by_elem:
        #print(elem)
        if elem.startswith("Buf"): continue
        for i in range(len(topo_order_by_elem[elem])-1):
            first_node = topo_order_by_elem[elem][i]
            next_node = topo_order_by_elem[elem][i+1]
            resource_constraints += [
                opt_vars[graph.nodes[first_node]["scheduling_id"]][1] <=
                opt_vars[graph.nodes[next_node]["scheduling_id"]][0]
            ]
            if debug: resource_chain_edges.append((first_node, next_node))
            if add_resource_edges or debug: 
                resource_edge_graph.add_edge(first_node, next_node, weight=graph.nodes[first_node]["cost"])
            graph.nodes[first_node]["allocation"] = elem
            graph.nodes[next_node]["allocation"] = elem
            logger.info(f"constraining {next_node} to be later than {first_node}")
    for extra_constraint in extra_constraints: # used for register ordering in first scheduling pass
        resource_constraints += [
            opt_vars[graph.nodes[extra_constraint[0]]["scheduling_id"]][0] + 0.00001
            <= opt_vars[graph.nodes[extra_constraint[1]]["scheduling_id"]][0]
        ]
        logger.info(f"constraining {extra_constraint[1]} to be later than {extra_constraint[0]} (register ordering constraint)")
    constraints += resource_constraints

    if debug:
        sim_util.topological_layout_plot(resource_edge_graph, extra_edges=resource_chain_edges)

    obj = cp.Minimize(opt_vars[graph_nodes["end"]["scheduling_id"]][0])
    prob = cp.Problem(obj, constraints)
    prob.solve()
    #print(prob.status)

    # assign start and end times to each node
    for node in graph_nodes:
        #print(node)
        #print(opt_vars[0])
        start_time, end_time = opt_vars[node[1]['scheduling_id']].value
        node[1]['start_time'] = np.round(start_time, 5)
        node[1]['end_time'] = np.round(end_time, 5)

    if debug:
        fig, ax = sim_util.plot_schedule_gantt(graph)
        plt.show()

    return resource_edge_graph

class node:
    def __init__(self, id, name, tp, module, delay, library):
        self.id = id
        self.name = name
        self.type = tp
        self.module = module
        self.delay = delay
        self.library = library

    def __str__(self):
        return f"========================\nNode id: {self.id}\nName: {self.name}\ntype: {self.type}\nmodule: {self.module}\n========================\n"

    def label(self):
        return f"{self.module}-{self.id}-{self.type}"
    
def convert_to_standard_dfg(graph: nx.DiGraph, module_map):
    """
    takes the output of parse_gnt_to_graph and converts
    it to our standard dfg format for use in the rest of the flow.
    We discard catapult-specific information and convert each node
    to one of our standard operators with the correct delay value.

    Arguments:
    - graph: output of parse_gnt_to_graph
    - node_objects: mapping of node -> node class object
    - module_map: mapping from ccore module to standard operator name
    """
    nodes_removed = []
    edges_to_remove = []
    for node in graph:
        node_data = graph.nodes[node]
        if node_data["module"] and node_data["module"][:node_data["module"].find('(')] in module_map:
            assert node_data["module"].find('(') != -1
            logger.info(f"{node} remaining in pruned graph")
        else:
            logger.info(f"removing {node}")
            nodes_removed.append(node)
        if graph.has_edge(node, node):
            edges_to_remove.append((node, node))

    logger.info(edges_to_remove)
    for edge in edges_to_remove:
        graph.remove_edge(edge[0], edge[1])

    for node in nodes_removed:
        if node in graph:
            predecessors = list(graph.predecessors(node))
            successors = list(graph.successors(node))
            
            # Create edges from predecessors to successors
            for pred in predecessors:
                for succ in successors:
                    if pred != succ:  # Avoid self-loops
                        logger.info(f"adding edge between {pred} and {succ}")
                        graph.add_edge(pred, succ)
            
            # Remove the node
            graph.remove_node(node)

    #sim_util.topological_layout_plot(graph)

    modified_graph = nx.DiGraph()
    for node in graph:
        node_data = graph.nodes[node]
        module_prefix = node_data["module"][:node_data["module"].find('(')]
        fn = module_map[module_prefix]
        modified_graph.add_node(
            node,
            id=node_data["id"],
            function=fn,
            cost=node_data["delay"],
            start_time=0,
            end_time=0,
            allocation="",
            library=node_data["library"]
        )
    modified_graph.add_node(
        "end",
        function="end"
    )
    for node in graph:
        node_data = graph.nodes[node]
        for child in graph.successors(node):
            modified_graph.add_edge(
                node, 
                child, 
                cost=0, # cost to be set after parasitic extraction 
                weight=modified_graph.nodes[node]["cost"]
            ) 
        if not len(list(graph.successors(node))):
            modified_graph.add_edge(
                node, "end",
                weight=modified_graph.nodes[node]["cost"]
            )

    

    nx.write_gml(modified_graph, "src/tmp/schedule.gml")
    logger.info(f"longest path length: {nx.dag_longest_path_length(modified_graph)}")
    logger.info(f"longest path: {nx.dag_longest_path(modified_graph)}")
    return modified_graph


def parse_gnt_to_graph(file_path):
    """
    Parses a .gnt file and creates a directed graph.
    
    :param file_path: Path to the .gnt file.
    :return: A NetworkX directed graph.
    """
    G = nx.DiGraph()

    ignore_types = ["{C-CORE", "ASSIGN"]

    nodes = {}
    node_successors = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line.startswith("set a("):
                continue
            node_id = line[line.find('(')+1:line.find(')')]
            tokens = line.split()
            node_name = ""
            if (line.find(" NAME ") != -1):
                for i in range(len(tokens)):
                    if (tokens[i] == "NAME"):
                        node_name = tokens[i+1]
                        break
                assert node_name != "", f"{line}"
            node_type = None
            if (line.find(" TYPE ") != -1):
                for i in range(len(tokens)):
                    if (tokens[i] == "TYPE"):
                        node_type = tokens[i+1]
                        break
                assert node_type, f"{line}"
            node_module = None
            if (line.find(" MODULE ") != -1):
                for i in range(len(tokens)):
                    if (tokens[i] == "MODULE"):
                        node_module = tokens[i+1]
                        break
                assert node_module, f"{line}"
            node_delay = 0
            if (line.find(" DELAY ") != -1): # todo: investigate if CYCLES need to be added to delay
                for i in range(len(tokens)):
                    if (tokens[i] == "DELAY"):
                        node_delay = float(tokens[i+1][1:-1]) # take out brackets with [1:-1]
                #print(node_delay)
            node_library = None
            if (line.find(" LIBRARY ") != -1):
                for i in range(len(tokens)):
                    if (tokens[i] == "LIBRARY"):
                        node_library = tokens[i+1]

            nodes[node_id] = node(node_id, node_name, node_type, node_module, node_delay, node_library)
        

        for line in lines:
            line = line.strip()
            if not line.startswith("set a("):
                continue
            node_id = line[line.find('(')+1:line.find(')')]
            tokens = line.split(' ')
            successors = []
            if (line.find(" SUCCS ") != -1):
                for i in range(len(tokens)):
                    if (tokens[i] == "SUCCS"):
                        for j in range(i+1, len(tokens)):
                            if (tokens[j] == "CYCLES"): break
                            if tokens[j] in nodes:
                                successors.append(nodes[tokens[j]])
                        break
                
                
            node_successors[node_id] = successors
        
        for n in nodes:
            G.add_node(
                nodes[n].label(),
                name=nodes[n].name,
                id=nodes[n].id,
                tp=nodes[n].type,
                module=nodes[n].module,
                delay=nodes[n].delay,
                library=nodes[n].library
            )
        for n in nodes:
            for successor in node_successors[nodes[n].id]:
                logger.info(f"adding edge between {nodes[n].label()} and {successor.label()}")
                if G.has_edge(successor.label(), nodes[n].label()):
                    G.remove_edge(successor.label(), nodes[n].label()) # no cycles, usually created by io constraints not data dependencies
                else:
                    G.add_edge(nodes[n].label(), successor.label())

        for n in nodes:
            if nodes[n].type in ignore_types:
                G.remove_node(nodes[n].label())
    
    return G


if __name__ == "__main__":
    G = parse_gnt_to_graph("src/tmp/benchmark/build/InputDoubleBufferless_4096comma_16comma_16greater_.v1/schedule.gnt")
    module_map = {
        "mgc_add": "Add",
        "mgc_mul": "Mult",
        "mgc_and": "And",
        "mgc_or": "Or",
        "ccs_ram_sync_1R1W_wport": "Buf",
        "ccs_ram_sync_1R1W_rport": "Buf"
    }
    nx.write_gml(G, "src/tmp/catapult_graph.gml", stringizer=lambda x: str(x))
    modified_G = convert_to_standard_dfg(G, module_map)