from collections import deque
import heapq
import math
import logging
logger = logging.getLogger(__name__)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import pandas as pd

from global_constants import SEED
import sim_util

# format: cfg_node -> {states -> operations}
cfg_node_to_dfg_map = {}
operation_sets = {}

rng = np.random.default_rng(SEED)

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
        print(f"creating dfg for node {node}")
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
    Calculates start time of each operation in the graph.
    Adds node parameter 'start_time' to each node in the graph.
    Params:
        graph: nx.DiGraph
            The computation data flow graph
    """
    logger.info("Assigning time of execution")
    for node in graph.nodes:
        graph.nodes[node]["dist"] =  0
    graph = nx.reverse(graph)
    for gen in list(nx.topological_generations(graph))[1:]:
        max_weight = 0
        for node in gen:
            in_edge = list(graph.in_edges(node))[0]
            max_weight = max(max_weight, graph.edges[in_edge]["weight"])
        for node in gen:
            in_edge = list(graph.in_edges(node))[0]
            graph.edges[in_edge]["weight"] = max_weight
    max_dist = 0
    end_node = list(nx.topological_generations(graph))[0][0]
    for node in graph.nodes:
        graph.nodes[node]["start_time"] = nx.dijkstra_path_length(graph, end_node, node)
        max_dist = max(max_dist, graph.nodes[node]["start_time"])
    for node in graph.nodes:
        # mirroring operation
        graph.nodes[node]["start_time"] = (graph.nodes[node]["start_time"] - max_dist) * -1
    graph = nx.reverse(graph)
    return graph, max_dist

def assign_upstream_path_lengths_dijkstra(graph):
    """
    Assigns the longest path to each node in the graph.
    Uses Dijkstra to take operation latency into account.
    Params:
    graph: nx.DiGraph
        The computation graph
    """
    for node in graph.nodes:
        graph.nodes[node]["dist"] =  0
    q = deque()
    for node in list(nx.topological_generations(graph))[0]:
        q.append(node)
        while not len(q) == 0:
            curnode = q.popleft()
            graph.nodes[curnode]["dist"] = max(graph.nodes[curnode]["dist"], nx.dijkstra_path_length(graph, node, curnode))
            for child in graph.successors(curnode):
                q.append(child)
    return graph

def assign_upstream_path_lengths_old(graph):
    """
    Assigns the longest path to each node in the graph.
    Currently ignores actual latencies of nodes.
    """
    for node in graph:
        graph.nodes[node]["dist"] =  0
    for i, generations in enumerate(nx.topological_generations(graph)):
        for node in generations:
            graph.nodes[node]["dist"] = max(i, graph.nodes[node]["dist"])
    
    return graph

def sdc_schedule(graph, hw_element_counts):
    """
    Runs the convex optimization problem to minimize the longest path latency.
    Encodes data dependency constraints from CDFG and resource constraints using
    the SDC formulation published here https://www.csl.cornell.edu/~zhiruz/pdfs/sdc-dac2006.pdf.
    TODO: Add exact hw schema constraints as resource constraints. Right now we assume hardware is
    all to all.
    """
    constraints = []
    vars = []
    graph_nodes = graph.nodes(data=True)
    id = 0
    for node in graph_nodes:
        curr_var = cp.Variable(2, name = node[0]) # first one is start time and last one is end time
        vars.append(curr_var)
        # start time + latency = end time for each operating node
        node[1]['scheduling_id'] = id
        id += 1
        constraints.append(curr_var[0] >= 0)
        if 'cost' in node[1].keys():
            constraints.append(curr_var[0] + node[1]['cost'] == curr_var[1])

    # data dependency constraints
    for u, v in graph.edges():
        source_id = int(graph_nodes[u]['scheduling_id'])
        dest_id = int(graph_nodes[v]['scheduling_id'])
        constraints.append(vars[source_id][1] - vars[dest_id][0] <= 0.0)
    
    topological_order = longest_path_first_topological_sort(graph)
    resource_constraints = []
    for i in range(len(topological_order)):
        curr_func_count = 0
        start_node = topological_order[i]
        if graph.nodes[start_node]['function'] not in hw_element_counts:
            continue
        curr_func_count += 1
        for j in range(i + 1, len(topological_order)):
            curr_node = topological_order[j]
            if graph.nodes[curr_node]['function'] not in hw_element_counts:
                    continue
            if graph.nodes[curr_node]['function'] == graph.nodes[start_node]['function']:
                curr_func_count += 1
                if curr_func_count > 2*hw_element_counts[graph.nodes[curr_node]['function']]:
                    break
                if curr_func_count > hw_element_counts[graph.nodes[curr_node]['function']]:
                    # add a constraint
                    resource_constraints.append(vars[graph.nodes[start_node]['scheduling_id']][0] - vars[graph.nodes[curr_node]['scheduling_id']][0] <= -graph.nodes[start_node]['cost'])
    constraints += resource_constraints
    obj = cp.Minimize(vars[graph_nodes['end']['scheduling_id']][0])
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return obj.value

def write_df(in_use, hw_element_counts, execution_time, step):
    data = {round(i*step, 3): [0]*hw_element_counts["Regs"] for i in range(0, int(math.ceil(execution_time/step)))}
    for key in in_use:
        for elem in in_use[key]:
            data[key][int(elem[-1])] = 1
    df = pd.DataFrame(data=data).transpose()
    cols={i:f"reg{i}" for i in range(hw_element_counts["Regs"])}
    df = df.rename(columns=cols)
    df.index.name = "time"
    df.to_csv("codesign_log_dir/reg_use_table.csv")

def log_register_use(computation_graph, step, hw_element_counts, execution_time):
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
        first_time_step = (computation_graph.nodes[node]["dist"] // step) * step
        out_edge = list(computation_graph.out_edges(node))[0]
        end_time = computation_graph.nodes[node]["dist"] + computation_graph.edges[out_edge]["weight"]
        end_time_step = (end_time // step) * step
        i = round(first_time_step, 3)
        while i <= end_time_step:
            if i not in in_use:
                in_use[i] = []
            in_use[i].append(computation_graph.nodes[node]["allocation"])
            i = round(i + step, 3)
    keys = list(in_use.keys())
    keys.sort()
    in_use_sorted = {i: in_use[i] for i in keys}
    write_df(in_use_sorted, hw_element_counts, execution_time, step)
        
def greedy_schedule(computation_graph, hw_element_counts, hw_netlist, save_reg_use_table=False):
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
    computation_graph = assign_upstream_path_lengths_dijkstra(computation_graph)

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
                        size=func_nodes[idx][1]["size"] if "size" in func_nodes[idx][1] else 1,
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
    if save_reg_use_table:
        computation_graph, execution_time = assign_time_of_execution(computation_graph)
        log_register_use(computation_graph, 0.1, hw_element_counts, execution_time)
