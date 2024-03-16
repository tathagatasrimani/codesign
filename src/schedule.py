from collections import deque

import networkx as nx
import matplotlib.pyplot as plt

import hardwareModel

# format: cfg_node -> {states -> operations}
cfg_node_to_hw_map = {}
operation_sets = {}


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


def schedule(computation_graph, hw_netlist):
    """
    Schedules the computation graph on the hardware netlist
    by determining the order of operations and the states in which
    they are executed. Includes the adding of stalls to account for
    data dependencies and in use elements.
    """

    hw_element_counts = {}
    for func in hardwareModel.get_unique_funcs(hw_netlist):
        hw_element_counts[func] = hardwareModel.num_nodes_with_func(hw_netlist, func)

    for layer, generation in enumerate(
        reversed(list(nx.topological_generations(nx.reverse(computation_graph))))
    ):
        nodes = computation_graph[generation].data()

        pass
