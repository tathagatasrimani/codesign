import logging

logger = logging.getLogger(__name__)

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from . import sim_util

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
    print(slacks)
    unique_slacks = sorted(list(set(slacks.values())))
    print(sorted_slacks)
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
        print(f"connecting start to {node}")
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