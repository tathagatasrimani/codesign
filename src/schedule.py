from collections import deque

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import sim_util
import heapq

# can't import hardware Model else will have circular imports
# import hardwareModel

# format: cfg_node -> {states -> operations}
cfg_node_to_dfg_map = {}
operation_sets = {}

rng = np.random.default_rng()

def limited_topological_sorts(graph, max_sorts=10):
    in_degree = {node: 0 for node in graph.nodes()}
    for u, v in graph.edges():
        in_degree[v] += 1

    partial_order = []
    sorts_found = 0
    
    def visit():
        nonlocal sorts_found
        if sorts_found >= max_sorts:
            return

        all_visited = True
        for node in graph.nodes():
            if in_degree[node] == 0 and node not in partial_order:
                all_visited = False
                partial_order.append(node)
                for successor in graph.successors(node):
                    in_degree[successor] -= 1

                yield from visit()

                partial_order.pop()
                for successor in graph.successors(node):
                    in_degree[successor] += 1
        
        if all_visited:
            sorts_found += 1
            yield list(partial_order)

    return list(visit())

def longest_path_first_topological_sort(graph):
    # Calculate the longest path lengths to each node
    longest_path_length = {node: 0 for node in graph.nodes()}
    for node in reversed(list(nx.topological_sort(graph))):  # Use topological sort to order nodes
        for successor in graph.successors(node):
            longest_path_length[node] = max(longest_path_length[node], 1 + longest_path_length[successor])

    # Initialize a priority queue based on longest path lengths
    priority_queue = []
    in_degree = {node: 0 for node in graph.nodes()}
    for u, v in graph.edges():
        in_degree[v] += 1

    # Populate the priority queue with nodes having zero in-degree
    for node in graph.nodes():
        if in_degree[node] == 0:
            heapq.heappush(priority_queue, (-longest_path_length[node], node))  # Push with negative to simulate max heap

    topological_order = []
    while priority_queue:
        _, node = heapq.heappop(priority_queue)
        topological_order.append(node)
        for successor in graph.successors(node):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                heapq.heappush(priority_queue, (-longest_path_length[successor], successor))  # Maintain priority

    return topological_order

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

def schedule(graph, hw_element_counts):
    """
    Runs the convex optimization problem to minimize the longest path latency.
    Encodes data dependency constraints from CDFG and resource constraints using
    the SDC formulation published here https://www.csl.cornell.edu/~zhiruz/pdfs/sdc-dac2006.pdf.
    TODO: Add exact hw schema constraints as resource constraints. Right now we assume hardware is
    all to all.
    """
    constraints = []
    vars = []
    # graph nodes
    id = 0
    for node in graph_nodes:
        curr_var = cp.Variable(2, name = node[0]) # first one is start time and last one is end time
        # curr_var.name = node[0]
        vars.append(curr_var)
        # start time + latency = end time for each operating node
        node[1]['scheduling_id'] = id
        id += 1
        constraints.append(curr_var[0] >= 0)
        # constraints.append(curr_var[1] >= 0)
        if 'cost' in node[1].keys():
            constraints.append(curr_var[0] + node[1]['cost'] == curr_var[1])

    # print(graph_nodes)
    # nx.set_node_attributes(graph, graph_nodes)
    # data dependency constraints
    for u, v in graph.edges():
        # if 'idx'
        source_id = int(graph_nodes[u]['scheduling_id'])
        dest_id = int(graph_nodes[v]['scheduling_id'])
        constraints.append(vars[source_id][1] - vars[dest_id][0] <= 0.0)
    
    topological_order = longest_path_first_topological_sort(graph)
    resource_constraints = []
    for i in range(len(topological_order)):
        # curr_reg_count = {'Regs': 0, 'Add': 0, 'Mult': 0, 'Buf': 0, 'Eq': 0, 'stall': 0}
        curr_func_count = 0
        start_node = topological_order[i]
        if graph.nodes[start_node]['function'] not in curr_reg_count:
            continue
        # curr_reg_count[graph.nodes[start_node]['function']] += 1
        curr_func_count += 1
        for j in range(i + 1, len(topological_order)):
            curr_node = topological_order[j]
            if graph.nodes[curr_node]['function'] not in curr_reg_count.keys():
                    continue
            # curr_reg_count[graph.nodes[curr_node]['function']] += 1
            if graph.nodes[curr_node]['function'] == graph.nodes[start_node]['function']:
                curr_func_count += 1
                if curr_func_count > 2*hw_element_counts[graph.nodes[curr_node]['function']]:
                    break
                if curr_func_count > hw_element_counts[graph.nodes[curr_node]['function']]:
                    # add a constraint
                    resource_constraints.append(vars[graph.nodes[start_node]['scheduling_id']][0] - vars[graph.nodes[curr_node]['scheduling_id']][0] <= -graph.nodes[start_node]['cost'])
                    # break
        # print(curr_reg_count)
    constraints += resource_constraints
    obj = cp.Minimize(vars[graph_nodes['end']['scheduling_id']][0])
    # obj = cp.Minimize(all_nodes)
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return obj.value

def greedy_schedule(computation_graph, hw_element_counts):
    """
    Schedules the computation graph on the hardware netlist
    by determining the order of operations and the states in which
    they are executed. Includes the adding of stalls to account for
    data dependencies and in use elements.
    """

    hw_element_counts["stall"] = np.inf
    # print(f"hw_element_counts: {hw_element_counts}")

    # do topo sort from beginning and add dist attribute to each node
    # for longest path to get to it from a gen[0] node.

    # reset layers:
    for node in computation_graph.nodes:
        computation_graph.nodes[node]["layer"] = -np.inf
    assign_upstream_path_lengths(computation_graph)

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
                # print(f"func_nodes: {func_nodes}")
                func_nodes = sorted(func_nodes, key=lambda x: x[1]["dist"], reverse=True)
                # print(f"sorted func_nodes: {func_nodes}")

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
                        new_edges.append((stall_name, edge[1], {"weight": computation_graph.edges[edge]["weight"]}))
                        new_edges.append((edge[0], stall_name, {"weight": computation_graph.edges[edge]["weight"]})) # edge[0] is same as func_nodes[idx][0]
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
