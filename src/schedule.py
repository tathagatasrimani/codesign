from collections import deque

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from global_constants import SEED
import sim_util


# format: cfg_node -> {states -> operations}
cfg_node_to_dfg_map = {}
operation_sets = {}

rng = np.random.default_rng(SEED)

def schedule_one_node(graph):
    """
    node does not get used here. WHY?!
    """
    node_ops = []
    op_set = set()
    queue = deque([[root, 0] for root in graph.roots])
    max_order = 0
    while len(queue) != 0:
        cur_node, order = queue.popleft()
        if cur_node not in op_set:
            op_set.add(cur_node)
        max_order = max(max_order, order)
        cur_node.order = max(order, cur_node.order)
        for child in cur_node.children:
            queue.append([child, order + 1])
    for i in range(max_order + 1):
        node_ops.append([])
    for cur_node in op_set:
        node_ops[cur_node.order].append(cur_node)
    return node_ops

def cfg_to_dfg(cfg, graphs, latency):
    """
    Literally just construct a nx.DiGraph from dfg_algo.Graph object

    graphs - is the output of dfg_algo.main(); it is a dictionary of cfg_node to dfg_algo.Graph objects
    latency - dict of operation -> latency
        this is used to add weights to the nodes in the nx.DiGraph
    returns:
        cfg_node_to_hw_map: dict of cfg_node -> {states -> operations}
    """
    for node in cfg:
        cfg_node_to_dfg_map[node] = nx.DiGraph()
        operation_sets[node] = set()

        queue = deque([[root, 0] for root in graphs[node].roots])
        max_order = 0
        while len(queue) != 0:
            cur_node, order = queue.popleft()
            if cur_node not in operation_sets[node]:
                if (
                    cur_node.operation is None
                ):  # if no operation, then we ignore in latency power calculation.
                    cfg_node_to_dfg_map[node] = nx.DiGraph()
                    break
                operation_sets[node].add(cur_node)
                cfg_node_to_dfg_map[node].add_node(
                    f"{cur_node.value};{cur_node.id}",
                    function=cur_node.operation,
                    idx=cur_node.id,
                    cost=latency[cur_node.operation],
                )

                for par in cur_node.parents:
                    try:
                        cfg_node_to_dfg_map[node].add_edge(
                            f"{par.value};{par.id}",
                            f"{cur_node.value};{cur_node.id}",
                            weight=latency[par.operation], # weight of edge is latency of parent
                        )
                    except KeyError:
                        print(f"KeyError: {par.operation} for {par.value};{par.id} -> {cur_node.value};{cur_node.id}")
                        cfg_node_to_dfg_map[node].add_edge(
                            par.value, cur_node.value, weight=latency[par.operation]
                        )

                for child in cur_node.children:
                    queue.append([child, order + 1])

    return cfg_node_to_dfg_map

def assign_time_of_execution(graph):
    """
    Calculates when each operation will take place,
    overwriting the "dist" attribute from before.
    """
    for node in graph.nodes:
        graph.nodes[node]["dist"] =  0
    graph = nx.reverse(graph)
    max_dist = 0
    end_node = list(nx.topological_generations(graph))[0][0]
    for node in graph.nodes:
        graph.nodes[node]["dist"] = nx.dijkstra_path_length(graph, end_node, node)
        max_dist = max(max_dist, graph.nodes[node]["dist"])
    for node in graph.nodes:
        # mirroring operation
        graph.nodes[node]["dist"] = (graph.nodes[node]["dist"] - max_dist) * -1
    graph = nx.reverse(graph)
    return graph

def assign_upstream_path_lengths(graph):
    """
    Assigns the longest path to each node in the graph.
    Currently ignores actual latencies of nodes.
    TODO: change this to dijkstra
    """
    for node in graph:
        graph.nodes[node]["dist"] =  0
    for i, generations in enumerate(nx.topological_generations(graph)):
        for node in generations:
            graph.nodes[node]["dist"] = max(i, graph.nodes[node]["dist"])
    
    return graph

def log_register_use(computation_graph, step):
    """
    Logs which registers' values are being used at discrete time intervals.

    Params:
    computation_graph: nx.DiGraph
        The computation graph containing operations to be examined
    step: int
        The discrete time step at which we log register use
    """
    in_use = {}

    for node in computation_graph:
        func = computation_graph.nodes[node]["function"]
        if not func == "Regs": continue
        #print(computation_graph.nodes[node]["allocation"], func, computation_graph.nodes[node]["dist"], node)
        first_time_step = (computation_graph.nodes[node]["dist"] // step) * step
        out_edge = list(computation_graph.out_edges(node))[0]
        end_time = computation_graph.nodes[node]["dist"] + computation_graph.edges[out_edge]["weight"]
        end_time_step = (end_time // step) * step
        i = first_time_step
        while i <= end_time_step:
            if i not in in_use:
                in_use[i] = []
            in_use[i].append(computation_graph.nodes[node]["allocation"])
            i += step
    keys = list(in_use.keys())
    keys.sort()
    in_use_sorted = {i: in_use[i] for i in keys}
    #sim_util.topological_layout_plot(computation_graph, reverse=True)
    return in_use_sorted
        


def schedule(computation_graph, hw_element_counts, hw_netlist):
    """
    Schedules the computation graph on the hardware netlist
    by determining the order of operations and the states in which
    they are executed. Includes the adding of stalls to account for
    data dependencies and in use elements.

    Allocation of operations to hardware elements is done in this function, in a greedy fashion.

    Params:
    computation_graph: nx.DiGraph
        The computation graph to be scheduled
    hw_element_counts: dict
        The number of hardware elements of each type (function)
    hw_netlist: nx.DiGraph
        The hardware netlist to be scheduled on
    """

    hw_element_counts["stall"] = np.inf

    # do topo sort from beginning and add dist attribute to each node
    # for longest path to get to it from a gen[0] node.

    # reset layers:
    for node in computation_graph.nodes:
        computation_graph.nodes[node]["layer"] = -np.inf
    computation_graph = assign_upstream_path_lengths(computation_graph)

    stall_counter = 0 # used to ensure unique stall names
    pushed = []
    # going through the computation graph from the end to the beginning and bubbling up operations
    generations = list(nx.topological_generations(nx.reverse(computation_graph)))
    layer = 0
    while layer < len(generations) or len(pushed) != 0:
        if layer >= len(generations):
            generation = []
        else:
            generation = generations[layer]
        generation += pushed
        pushed = []

        # if any horizontal dependencies, push them to the next layer
        for node in generation:
            out_nodes = list(map(lambda x: x[1], computation_graph.out_edges(node)))
            intersect = set(out_nodes).intersection(set(generation))
            if intersect:
                pushed.append(node)
            else:
                computation_graph.nodes[node]["allocation"] = ""
                computation_graph.nodes[node]["layer"] = -layer
        generation = [
            item for item in generation if item not in pushed
        ]

        nodes_in_gen = list(filter(lambda x: x[0] in generation, computation_graph.nodes.data()))
        funcs_in_gen, counts_in_gen = np.unique(
            list(map(lambda x: x[1]["function"], nodes_in_gen)), return_counts=True
        )

        for func, count in zip(funcs_in_gen, counts_in_gen): # for each function in the generation
            if func == "start" or func == "end":
                continue
            # if there are more operations of this type than there are hardware elements
            # then we need to add stalls, sort descending by distance from start
            # ie, the ones closest the start get stalled first
            if count > hw_element_counts[func]:
                func_nodes = list(filter(lambda x: x[1]["function"] == func, nodes_in_gen))
                func_nodes = sorted(func_nodes, key=lambda x: x[1]["dist"], reverse=True)

                start_idx = hw_element_counts[func]
                for idx in range(start_idx, count):
                    # an out edge in comp_dfg is an in_edge in the reversed_graph
                    out_edges = list(computation_graph.out_edges(func_nodes[idx][0]))

                    stall_name = f"stall_{layer}_{idx}_{func}_{stall_counter}"
                    stall_counter += 1
                    computation_graph.add_node(
                        stall_name,
                        function="stall",
                        cost=func_nodes[idx][1]["cost"],
                        layer= -layer,
                        allocation="",
                        dist=0
                    )
                    new_edges = []
                    for edge in out_edges:
                        new_edges.append((stall_name, edge[1], {"weight": computation_graph.edges[edge]["weight"]}))
                        new_edges.append((edge[0], stall_name, {"weight": computation_graph.edges[edge]["weight"]})) # edge[0] is same as func_nodes[idx][0]
                    computation_graph.add_edges_from(
                        new_edges
                    )
                    computation_graph.remove_edges_from(out_edges)

                    computation_graph.nodes[func_nodes[idx][0]]["layer"] = -(layer + 1) # bubble up
                    pushed.append(func_nodes[idx][0])

        hopeful_nodes = list(filter(
            lambda x: x[1]["layer"] >= -layer, computation_graph.nodes.data()
        ))
        processed_nodes = list(map(lambda x: x[0], hopeful_nodes))
        processed_graph = nx.subgraph(computation_graph, processed_nodes)

        curr_gen_nodes = list(filter(lambda x: x[1]["layer"] == -layer, computation_graph.nodes.data()))
        funcs_in_gen, counts_in_gen = np.unique(
            list(map(lambda x: x[1]["function"], curr_gen_nodes)), return_counts=True
        )

        for i, func in enumerate(funcs_in_gen):
            if func in ["start", "end", "stall"]:
                continue
            assert counts_in_gen[i] <= hw_element_counts[func]
            # do a greedy allocation of the nodes to the hardware elements
            comp_nodes = list(filter(lambda x: x[1]["function"] == func, curr_gen_nodes))
            hw_nodes = list(filter(lambda x: x[1]["function"] == func, hw_netlist.nodes.data()))
            for i in range(len(comp_nodes)):
                computation_graph.nodes[comp_nodes[i][0]]["allocation"] = hw_nodes[i][0]

        layer += 1
    computation_graph = assign_time_of_execution(computation_graph)
    in_use = log_register_use(computation_graph, 0.1)
    return in_use
