from collections import deque

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# can't import hardware Model else will have circular imports
# import hardwareModel

# format: cfg_node -> {states -> operations}
cfg_node_to_hw_map = {}
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
    computation_graph = nx.DiGraph()
    for node in cfg:
        cfg_node_to_hw_map[node] = nx.DiGraph()
        operation_sets[node] = set()

        queue = deque([[root, 0] for root in graphs[node].roots])
        max_order = 0
        while len(queue) != 0:
            cur_node, order = queue.popleft()
            if cur_node not in operation_sets[node]:
                if (
                    cur_node.operation is None
                ):  # if no operation, then we ignore in latency power calculation.
                    cfg_node_to_hw_map[node] = nx.DiGraph()
                    break
                operation_sets[node].add(cur_node)
                cfg_node_to_hw_map[node].add_node(
                    f"{cur_node.value};{cur_node.id}",
                    function=cur_node.operation,
                    idx=cur_node.id,
                    cost=latency[cur_node.operation],
                )

                for par in cur_node.parents:
                    try:
                        cfg_node_to_hw_map[node].add_edge(
                            f"{par.value};{par.id}",
                            f"{cur_node.value};{cur_node.id}",
                        )
                    except KeyError:
                        cfg_node_to_hw_map[node].add_edge(
                            par.value, cur_node.value, cost=0
                        )

                for child in cur_node.children:
                    queue.append([child, order + 1])

    return cfg_node_to_hw_map


def schedule(computation_graph, hw_element_counts):
    """
    Schedules the computation graph on the hardware netlist
    by determining the order of operations and the states in which
    they are executed. Includes the adding of stalls to account for
    data dependencies and in use elements.
    """

    hw_element_counts["stall"] = np.inf

    pushed = []
    # going through the computation graph from the end to the beginning amd bubbling up operations
    generations = list(nx.topological_generations(nx.reverse(computation_graph)))
    layer = 0
    while layer < len(generations) or len(pushed) != 0:
        if layer == len(generations):
            generation = []
        else:
            generation = generations[layer]
        generation += pushed
        pushed = []

        # if any horizontal dependencies, push them to the next layer
        for node in generation:
            computation_graph.nodes[node]["layer"] = -layer
            out_nodes = list(map(lambda x: x[1], computation_graph.out_edges(node)))
            intersect = set(out_nodes).intersection(set(generation))
            if intersect:
                pushed.append(node)
        generation = [
            item for item in generation if item not in pushed
        ]

        nodes = list(filter(lambda x: x[0] in generation, computation_graph.nodes.data()))
        funcs, counts = np.unique(list(map(lambda x: x[1]["function"], nodes)), return_counts=True)

        for func, count in zip(funcs, counts): # for each function in the generation
            if count > hw_element_counts[func]:
                func_nodes = list(filter(lambda x: x[1]["function"] == func, nodes))
                # diff = count - hw_element_counts[func]
                # print(f"not enough resources for {func}; diff: {diff}")

                start_idx = hw_element_counts[func]
                for idx in range(start_idx, count):
                    # print(f"idx: {idx}; node: {func_nodes[idx][0]}")
                    # an out edge in comp_dfg is an in_edge in the reversed_graph
                    out_edges = list(computation_graph.out_edges(func_nodes[idx][0]))
                    
                    assert len(out_edges) == 1
                    stall_name = f"stall_{layer}_{idx}_{func}_{rng.integers(0,100)}"
                    computation_graph.add_node(
                        stall_name,
                        function="stall",
                        cost=func_nodes[idx][1]["cost"],
                        layer=-1 * layer,
                    )
                    computation_graph.remove_edges_from(out_edges)
                    computation_graph.add_edges_from(
                        [(stall_name, out_edges[0][1]), (func_nodes[idx][0], stall_name)]
                    )
                    computation_graph.nodes[func_nodes[idx][0]]["layer"] = -1 * (
                        layer + 1
                    )
                    pushed.append(func_nodes[idx][0])

        layer = min(layer + 1, len(generations))
        pass
