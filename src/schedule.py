from collections import deque

import networkx as nx
import matplotlib.pyplot as plt

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


def schedule(cfg, graphs, latency):
    """

    graphs - is the output of dfg_algo.main(); it is a dictionary of cfg_node to dfg_algo.Graph objects
    latency - dict of operation -> latency
    returns:
        cfg_node_to_hw_map: dict of cfg_node -> {states -> operations}
    """
    print(f"in schedule")
    for node in cfg:
        print(f"\nnode id: {node.id}")
        cfg_node_to_hw_map[node] = nx.DiGraph()
        operation_sets[node] = set()
        graph = graphs[node]
        # print("graph: ", graph)
        # graph.gv_graph.render("graph" + str(node.id) + ".gv")
        queue = deque([[root, 0] for root in graph.roots])
        max_order = 0
        while len(queue) != 0:
            cur_node, order = queue.popleft()
            if cur_node not in operation_sets[node]:
                if cur_node.operation is None: # if no operation, then we ignore in latency power calculation.
                    cfg_node_to_hw_map[node] = nx.DiGraph()
                    break
                operation_sets[node].add(cur_node)
                cfg_node_to_hw_map[node].add_node(
                    f"{cur_node.value};{cur_node.id}",
                    function=cur_node.operation,
                    idx=cur_node.id,
                )
                print("order: ", order, "; curr node: ", cur_node)

                # print(
                #     f"node {cur_node.value} parents: {[str(par) for par in cur_node.parents]}"
                # )
                for par in cur_node.parents:
                    try:
                        cfg_node_to_hw_map[node].add_edge(
                            f"{par.value};{par.id}",
                            f"{cur_node.value};{cur_node.id}",
                            cost=latency[
                                cur_node.operation
                            ],  # cost of an edge is the latency of the downstream operation.
                        )
                    except KeyError:
                        cfg_node_to_hw_map[node].add_edge(
                            par.value, cur_node.value, cost=0
                        )
                # max_order = max(max_order, order)
                # cur_node.order = max(order, cur_node.order)
                for child in cur_node.children:
                    queue.append([child, order + 1])

        # for i in range(max_order + 1):
        #     cfg_node_to_hw_map[node].append([])
        # for cur_node in operation_sets[node]:
        #     cfg_node_to_hw_map[node][cur_node.order].append(cur_node)
        # for state in cfg_node_to_hw_map[node]:
        #     for op in state:
        #         print(op.order, op.operation)
        #     print('')
    n = len(cfg_node_to_hw_map.values())
    print(list(cfg_node_to_hw_map.values())[0].nodes.data())
    print(list(cfg_node_to_hw_map.values())[0].edges.data())
    # nx.draw(list(cfg_node_to_hw_map.values())[n - 2], with_labels=True)
    # plt.show()
    return cfg_node_to_hw_map
