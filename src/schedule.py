from collections import deque

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import sim_util

# can't import hardware Model else will have circular imports
# import hardwareModel

# format: cfg_node -> {states -> operations}
cfg_node_to_dfg_map = {}
operation_sets = {}

rng = np.random.default_rng()


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


def schedule(computation_graph, hw_element_counts):
    """
    Schedules the computation graph on the hardware netlist
    by determining the order of operations and the states in which
    they are executed. Includes the adding of stalls to account for
    data dependencies and in use elements.
    """

    hw_element_counts["stall"] = np.inf
    print(f"hw_element_counts: {hw_element_counts}")

    # do topo sort from beginning and add dist attribute to each node
    # for longest path to get to it from a gen[0] node.

    gen_0 = list(nx.topological_generations(computation_graph))[0]
    # reset layers:
    for node in computation_graph.nodes:
        computation_graph.nodes[node]["layer"] = -np.inf

    stall_counter = 0 # used to ensure unique stall names
    pushed = []
    # going through the computation graph from the end to the beginning and bubbling up operations
    generations = list(nx.topological_generations(nx.reverse(computation_graph)))
    layer = 0
    while layer < len(generations) or len(pushed) != 0:
        # print(f"layer: {layer}; pushed: {pushed}")
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
        # print(f"generation before removing horizontal: {generation}")
        generation = [
            item for item in generation if item not in pushed
        ]
        # print(f"generation after removing horizontal: {generation}")

        nodes_in_gen = list(filter(lambda x: x[0] in generation, computation_graph.nodes.data()))
        # print(f"layers in gen: {list(map(lambda x: x[1]['layer'], nodes_in_gen))}")
        funcs_in_gen, counts_in_gen = np.unique(
            list(map(lambda x: x[1]["function"], nodes_in_gen)), return_counts=True
        )

        for func, count in zip(funcs_in_gen, counts_in_gen): # for each function in the generation
            # print(f"func: {func}; count: {count}")
            if func == "start" or func == "end":
                continue
            if count > hw_element_counts[func]:
                # print(f"more ops than hw elements for {func}")
                func_nodes = list(filter(lambda x: x[1]["function"] == func, nodes_in_gen))
                diff = count - hw_element_counts[func]
                # print(f"not enough resources for {func}; diff: {diff}")
                # print(f"nodes in gen of type {func}: {func_nodes}")

                start_idx = hw_element_counts[func]
                # TODO: pic this range based on upstream length. Calculated by Dijkstra
                for idx in range(start_idx, count):
                    # print(f"idx: {idx}; node: {func_nodes[idx][0]}; removing node from gen")
                    # generation.remove(func_nodes[idx][0])
                    # an out edge in comp_dfg is an in_edge in the reversed_graph
                    out_edges = list(computation_graph.out_edges(func_nodes[idx][0]))

                    stall_name = f"stall_{layer}_{idx}_{func}_{stall_counter}"
                    stall_counter += 1
                    # print(f"adding stall: {stall_name} for {func}")
                    computation_graph.add_node(
                        stall_name,
                        function="stall",
                        cost=func_nodes[idx][1]["cost"],
                        layer= -layer,
                    )
                    # print(f"adding stall: {stall_name}: {computation_graph.nodes[stall_name]}")
                    new_edges = []
                    for edge in out_edges:
                        new_edges.append((stall_name, edge[1]))
                        new_edges.append((edge[0], stall_name)) # edge[0] is same as func_nodes[idx][0]
                    computation_graph.add_edges_from(
                        new_edges
                    )
                    computation_graph.remove_edges_from(out_edges)

                    computation_graph.nodes[func_nodes[idx][0]]["layer"] = -(layer + 1) # bubble up
                    # print(f"node after bubbling: {func_nodes[idx][0]}: {computation_graph.nodes[func_nodes[idx][0]]}")
                    pushed.append(func_nodes[idx][0])

        # print(f"plotting processed graph")
        hopeful_nodes = list(filter(
            lambda x: x[1]["layer"] >= -layer, computation_graph.nodes.data()
        ))
        # print(f"layers: {[x[1]['layer'] for x in hopeful_nodes]}")
        processed_nodes = list(map(lambda x: x[0], hopeful_nodes))
        # print(f"processed_nodes: {processed_nodes}")
        processed_graph = nx.subgraph(computation_graph, processed_nodes)
        # try:
        #     sim_util.topological_layout_plot(processed_graph, reverse=False)
        # except nx.exception.NetworkXUnfeasible as e:
        #     print(f"Nx Exception: {e}")
        #     nodes_at_this_layer = list(map(lambda x: x[0], filter(lambda x: x[1]["layer"] == -layer, computation_graph.nodes.data())))
        #     print(f"nodes at this layer: {nodes_at_this_layer}")
        #     print(f"edges: {computation_graph.out_edges(nodes_at_this_layer)}")
        #     cycle_edges = nx.find_cycle(computation_graph)
        #     print(f"cycle: {cycle_edges}")
        #     nodes_1 = list(map(lambda x: x[0], cycle_edges))
        #     nodes_2 = list(map(lambda x: x[1], cycle_edges))
        #     nodes = nodes_1 + nodes_2
        #     print(f"nodes_1: {nodes_1}")
        #     cycle_graph = nx.DiGraph(nodes_1)
        #     cycle_graph.add_edges_from(cycle_edges)
        #     nx.draw(cycle_graph, with_labels=True)

        curr_gen_nodes = list(filter(lambda x: x[1]["layer"] == -layer, computation_graph.nodes.data()))
        funcs_in_gen, counts_in_gen = np.unique(
            list(map(lambda x: x[1]["function"], curr_gen_nodes)), return_counts=True
        )

        for i, func in enumerate(funcs_in_gen):
            if func in ["start", "end"]:
                continue
            assert counts_in_gen[i] <= hw_element_counts[func]

        # print(f"funcs_in_gen: {funcs_in_gen}; counts_in_gen: {counts_in_gen}")
        # print(f"hw_element_counts: {hw_element_counts}")

        layer += 1
