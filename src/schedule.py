from collections import deque, defaultdict
import heapq
import math
import logging
from itertools import combinations

logger = logging.getLogger(__name__)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import pandas as pd
import networkx as nx

from .global_constants import SEED
from . import sim_util
from . import memory

# format: cfg_node -> {states -> operations}
cfg_node_to_dfg_map = {}
operation_sets = {}
processed = {}

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
    for node in reversed(
        list(nx.topological_sort(graph))
    ):  # Use topological sort to order nodes
        for successor in graph.successors(node):
            longest_path_length[node] = max(
                longest_path_length[node], 1 + longest_path_length[successor]
            )

    # Initialize a priority queue based on longest path lengths
    priority_queue = []
    in_degree = {node: 0 for node in graph.nodes()}
    for u, v in graph.edges():
        in_degree[v] += 1

    # Populate the priority queue with nodes having zero in-degree
    for node in graph.nodes():
        if in_degree[node] == 0:
            heapq.heappush(
                priority_queue, (-longest_path_length[node], node)
            )  # Push with negative to simulate max heap

    topological_order = []
    while priority_queue:
        _, node = heapq.heappop(priority_queue)
        topological_order.append(node)
        for successor in graph.successors(node):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                heapq.heappush(
                    priority_queue, (-longest_path_length[successor], successor)
                )  # Maintain priority

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
        processed[node] = set()

        queue = deque([[root, 0] for root in graphs[node].roots])
        while len(queue) != 0:
            cur_node, order = queue.popleft()
            #print(cur_node, order)
            if cur_node not in operation_sets[node]:
                if not (
                    cur_node.operation is None or cur_node.value.startswith("__")
                ):  # if no operation or an intrinsic python name, then we ignore in latency power calculation.
                    operation_sets[node].add(cur_node)
                    cfg_node_to_dfg_map[node].add_node(
                        f"{cur_node.value};{cur_node.id}",
                        function=cur_node.operation,
                        idx=cur_node.id,
                        cost=latency[cur_node.operation],
                        write=cur_node.write
                    )
                    for par in cur_node.parents:
                        # print("node", cur_node, "has parent", par)
                        # only add edges to parents which represent actual operations
                        if par in operation_sets[node]:
                            try:
                                cfg_node_to_dfg_map[node].add_edge(
                                    f"{par.value};{par.id}",
                                    f"{cur_node.value};{cur_node.id}",
                                    weight=latency[
                                        par.operation
                                    ],  # weight of edge is latency of parent
                                )
                            except KeyError:
                                raise Exception(
                                    f"KeyError: {par.operation} for {par.value};{par.id} -> {cur_node.value};{cur_node.id}"
                                )
                                cfg_node_to_dfg_map[node].add_edge(
                                    par.value, cur_node.value, weight=latency[par.operation]
                                )
                processed[node].add(cur_node)
                for child in cur_node.children:
                    add_to_queue = True
                    # We only want to add child to queue if all of its parent operations have finished.
                    for par in child.parents:
                        if par not in processed[node]:
                            add_to_queue = False
                            break
                    if add_to_queue: queue.append([child, order + 1])

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
        graph.nodes[node]["dist"] = 0
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
        graph.nodes[node]["start_time"] = (
            graph.nodes[node]["start_time"] - max_dist
        ) * -1
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
        graph.nodes[node]["dist"] = 0
    q = deque()
    for node in list(nx.topological_generations(graph))[0]:
        q.append(node)
        while not len(q) == 0:
            curnode = q.popleft()
            graph.nodes[curnode]["dist"] = max(
                graph.nodes[curnode]["dist"],
                nx.dijkstra_path_length(graph, node, curnode),
            )
            for child in graph.successors(curnode):
                q.append(child)
    return graph


def assign_upstream_path_lengths_old(graph):
    """
    Assigns the longest path to each node in the graph.
    Currently ignores actual latencies of nodes.
    """
    for node in graph:
        graph.nodes[node]["dist"] = 0
    for i, generations in enumerate(nx.topological_generations(graph)):
        for node in generations:
            graph.nodes[node]["dist"] = max(i, graph.nodes[node]["dist"])

    return graph

def process_register_topo(graph, node, topo_order_regs, topo_order_ops_grouped, seen, check_arith=False):
    """
    typical arithmetic op:      (Reg read)--->(op)<---(Reg read)
                                                |
                                                |
                                            (reg write)
    We want all three of these registers to be grouped in topological ordering so that
    resource edge chains do not create infeasible schedules.
    """
    topo_order_regs.append(node)
    topo_order_ops_grouped.append(node)
    seen.add(node)
    logger.info(f"processing register node {node} with check_arith {check_arith}")
    if check_arith:
        parents = list(graph.predecessors(node))
        assert len(parents) <= 1, f"reg op has more than one parent, {parents} -> {node}"
        for parent in parents:
            # if parent is an arithmetic operator, its input register operations should be grouped 
            # in topological order with the output register
            if graph.nodes[parent]["function"] not in ["Regs", "Buf"] and graph.nodes[node]["write"] and parent not in seen:
                topo_order_ops_grouped.append(parent)
                seen.add(parent)
                grandparents = graph.predecessors(parent)
                for grandparent in grandparents:
                    if graph.nodes[grandparent]["function"] == "Regs" and grandparent not in seen:
                        process_register_topo(graph, grandparent, topo_order_regs, topo_order_ops_grouped, seen) 

def get_topological_order(graph, mem_stage, hw_element_counts, hw_netlist, reg_chains=None, topo_order_override=None, buf_chain=None, mem_chain=None):
    assert mem_stage in ["Regs", "Buf", "MainMem"]
    topo_order_by_func = {}
    func_instances = {func: list(filter(lambda x: hw_netlist.nodes[x]["function"] == func, hw_netlist.nodes())) for func in hw_element_counts}
    print(func_instances)
    topo_order_by_elem = {elem: [] for elem in hw_netlist.nodes()}
    extra_constraints = []
    if mem_stage == "Regs":
        topological_order = longest_path_first_topological_sort(graph)
        topo_order_ops_grouped = []
        topo_order_regs= []
        all_regs = []
        seen = set()
        # set topological order for register dependency chain. We should ensure that
        # input registers to an arithmetic operation and the corresponding output register are grouped together
        # to avoid an infeasible scheduling problem.
        for node in topological_order[::-1]:
            if graph.nodes[node]["function"] == "Regs" and node not in seen:
                process_register_topo(graph, node, topo_order_regs, topo_order_ops_grouped, seen, check_arith=True)                   
            if graph.nodes[node]["function"] == "Regs":
                all_regs.append(node)
            if node not in seen:
                topo_order_ops_grouped.append(node)
                seen.add(node)
        assert len(all_regs) == len(topo_order_regs), "topo order did not include all register operations"
        assert len(topo_order_ops_grouped) == len(topological_order), f"grouped topo order did not include all ops. {topo_order_ops_grouped}, {topological_order}"
        topo_order_by_func = {}
        for op in topological_order:
            func = graph.nodes[op]["function"]
            if func == "end": continue
            if func not in topo_order_by_func:
                topo_order_by_func[func] = []
            topo_order_by_func[func].append(op)
        topo_order_by_func["Regs"] = topo_order_regs[::-1]
        # hacky fix to maintain register op ordering in the first scheduling pass. This constraint does not apply to future passes
        for i in range(len(topo_order_by_func["Regs"])-1):
            extra_constraints.append((topo_order_by_func["Regs"][i], topo_order_by_func["Regs"][i+1]))
    else:
        for op in topo_order_override:
            func = graph.nodes[op]["function"]
            if func in ["end", "Regs"]: continue
            if func not in topo_order_by_func:
                topo_order_by_func[func] = []
            topo_order_by_func[func].append(op)
        print(topo_order_by_func)
        print(type(topo_order_by_func))
        topo_order_by_func["Buf"] = buf_chain
        if mem_stage=="MainMem":
            topo_order_by_func["MainMem"] = mem_chain
        for i in range(len(reg_chains)):
            hw_element = func_instances["Regs"][i]
            for op in reg_chains[i]:
                topo_order_by_elem[hw_element].append(op)
    for func in topo_order_by_func:
        assert len(func_instances[func]) == hw_element_counts[func]
        for i in range(len(topo_order_by_func[func])):
            hw_element = func_instances[func][i%hw_element_counts[func]]
            topo_order_by_elem[hw_element].append(topo_order_by_func[func][i])
    return topo_order_by_elem, extra_constraints

def sdc_schedule(graph, topo_order_by_elem, extra_constraints=[], add_resource_edges=False, debug=True):
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
            constraints.append(curr_var[0] + node[1]["cost"] == curr_var[1])
    
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

    return resource_edge_graph


class LiveInterval:
    def __init__(self, start, end, variable, id, op_names=[]):
        self.start = start
        self.end = end
        self.variable = variable
        self.id = id
        self.op_names = op_names

    def extend(self, new_end):
        self.end = new_end
    
    def add_op(self, op):
        self.op_names.append(op)

    def name(self):
        return self.variable
    
    def split(self, time, graph):
        # create two new live intervals split at t=time
        new_op_names = list(filter(lambda x: graph.nodes[x]["start_time"] >= time, self.op_names))
        #print(new_op_names)
        new_op_names = sorted(new_op_names, key=lambda x: graph.nodes[x]["start_time"]) # I don't think this is necessary
        return LiveInterval(graph.nodes[new_op_names[0]]["start_time"], self.end, self.variable, self.id, new_op_names)

    def __str__(self):
        return f"{self.variable} ({self.op_names}): [{self.start}, {self.end}]"

    def __lt__(self, other):
        return self.start < other.start
    
    
def get_live_intervals(sorted_regs):
    active = {}
    intervals = []
    id = 0
    for reg_node in sorted_regs:
        #print(reg_node)
        var_name = reg_node[0][:reg_node[0].find(';')]
        if var_name in active:
            # if write, then we create a new interval for the variable and save the current one
            if reg_node[1]["write"]:
                intervals.append(active[var_name])
                active[var_name] = LiveInterval(reg_node[1]["start_time"], reg_node[1]["end_time"], var_name, id, [reg_node[0]])
                id = id + 1
            # if read, then we can extend the current interval
            else:
                active[var_name].extend(reg_node[1]["end_time"])
                active[var_name].add_op(reg_node[0])
        else:
            #print(f"creating interval for {var_name} with op {reg_node[0]}")
            # if no interval exists yet for the variable, create one
            active[var_name] = LiveInterval(reg_node[1]["start_time"], reg_node[1]["end_time"], var_name, id, [reg_node[0]])
            id = id + 1
    # add remaining active intervals
    for var in active:
        intervals.append(active[var])
    intervals = sorted(intervals, key=lambda x:x.start)
    return intervals


def overlaps(node_1, node_2):
    n1_start, n1_end = np.round(node_1["start_time"], 3), np.round(node_2["start_time"], 3)
    n2_start, n2_end = np.round(node_2["start_time"], 3), np.round(node_2["end_time"], 3)
    return n1_start < n2_end and n2_start < n1_end


def register_allocate(graph, hw_element_counts, hw_netlist):
    reg_nodes = filter(lambda x: x[1]["function"] == "Regs", graph.nodes(data=True))
    sorted_regs = sorted(reg_nodes, key=lambda x: x[1]["start_time"])
    intervals = get_live_intervals(sorted_regs)
    active = []
    allocation = {}
    op_allocation = {}

    for interval in intervals:
        logger.info(str(interval))

    num_registers = hw_element_counts["Regs"]
    reg_instances = list(filter(lambda x: x[1]["function"] == "Regs", hw_netlist.nodes(data=True)))

    least_recently_allocated = deque([i for i in range(num_registers)])
    while len(intervals) != 0:
        logger.info(f"intervals list is {[str(i) for i in intervals]}")
        interval = intervals[0]
        intervals = intervals[1:]
        # Expire old intervals
        active = [i for i in active if i.end > interval.start]
        logger.info(f"active intervals are {[str(i) for i in active]}")

        if len(active) == num_registers:
            # Spill a register
            spill_candidates = sorted(active, key=lambda i: i.end)[::-1] # from latest end time to earliest
            completed_spill = False
            for spill_candidate in spill_candidates:
                # ensure that no register operation in spill candidate is allocated at exactly the same time as the first one in
                # the current interval. That way, we can assign the current interval to spill_candidate's register
                first_reg_in_current_interval = graph.nodes[interval.op_names[0]]
                assert first_reg_in_current_interval["start_time"] == interval.start
                for candidate_reg_op in spill_candidate.op_names:
                    candidate_reg_data = graph.nodes[candidate_reg_op]
                    if overlaps(first_reg_in_current_interval, candidate_reg_data): # move to next spill candidate
                        logger.info(f"{candidate_reg_op} and {interval.op_names[0]} overlap")
                        break
                    elif candidate_reg_data["start_time"] >= first_reg_in_current_interval["end_time"]:
                        logger.info(f"choosing {spill_candidate} to spill to memory. Was previously allocated to {allocation[spill_candidate.variable]}")
                        active.remove(spill_candidate)
                        spilled_interval = spill_candidate.split(first_reg_in_current_interval["end_time"], graph)
                        logger.info(f"split interval to deallocate is {spilled_interval}")
                        logger.info(f"new interval allocated to that register is {interval}")
                        active.append(interval)
                        for op in spilled_interval.op_names:
                            op_allocation[op] = None
                        for op in interval.op_names:
                            op_allocation[op] = allocation[spill_candidate.variable]
                        allocation[interval.variable] = allocation[spill_candidate.variable]
                        allocation[spill_candidate.variable] = None  # Mark as spilled
                        completed_spill= True
                        # must re-insert spilled interval so that it is sorted by start time
                        for i in range(len(intervals)):
                            if intervals[i].start > spilled_interval.start:
                                if i == 0:
                                    intervals = [spilled_interval] + intervals
                                else:
                                    intervals = intervals[:i] + [spilled_interval] + intervals[i:]
                                break
                            if i == len(intervals) - 1:
                                intervals.append(spilled_interval)
                        break
                if completed_spill: break
            assert completed_spill, "Could not spill any active interval"
        else:
            # Allocate a register
            #print(allocation)
            free_registers = set([i for i in range(num_registers) if i not in [allocation.get(i.variable) for i in active]])
            i = 0
            free_register = None
            assert free_registers
            while i < num_registers: # round robin allocation
                candidate = least_recently_allocated.popleft()
                least_recently_allocated.append(candidate)
                if candidate in free_registers:
                    free_register = candidate
                    break
                i += 1
            assert free_register != None, f"{free_registers}, {least_recently_allocated}"
            allocation[interval.variable] = free_register
            #print(free_register)
            for op in interval.op_names:
                op_allocation[op] = free_register
            active.append(interval)
            logger.info(f"allocated {interval} to {reg_instances[free_register]}")
    #print(op_allocation)
    for op in op_allocation:
        graph.nodes[op]["allocation"] = reg_instances[op_allocation[op]][0]

    reg_ops_sorted = sorted([(op.split(';')[0], op) for op in op_allocation], key=lambda x: graph.nodes[x[1]]["start_time"])
    """for _, reg_op in reg_ops_sorted:
        print(reg_op, graph.nodes[reg_op]["start_time"], graph.nodes[reg_op]["end_time"])"""
    return op_allocation, reg_ops_sorted

# deals with Regs -> Buf and Buf -> MainMem
def add_higher_memory_accesses_to_scheduled_graph(graph, hw_element_counts, hw_netlist, op_allocation, lower_func, higher_func, lower_ops_sorted, lower_level_latency, higher_level_latency):
    # Assume all data is available in the higher level, add reads/writes to higher level based on lower level allocation

    num_lower_levels = hw_element_counts[lower_func]
    lower_level_instances = list(filter(lambda x: x[1]["function"] == lower_func, hw_netlist.nodes(data=True)))
    lower_level_ops_by_allocation = [[] for _ in range(num_lower_levels)]
    higher_level_ops = []

    #print(op_allocation)
    #print(lower_ops_sorted)


    higher_level_nodes = list(filter(lambda x: x[1]["function"] == higher_func, hw_netlist.nodes(data=True)))
    assert len(higher_level_nodes) == 1
    Lower_objects = [memory.Register() if lower_func == "Regs" else memory.Buffer(lower_level_instances[i][1]["size"]) for i in range(num_lower_levels)]
    higher_level_count = 0

    #########################################################
    # STATE ELEMENTS
    # each of these are indexed by a variable name
    #########################################################

    # is data in the higher_level updated for this variable?
    higher_level_updated = {}
    # if a last higher_level write exists (with most recent data), it is stored here
    last_higher_level_write = {}
    # is this variable in a lower_level?
    in_lower_level = {}
    # which lower_level is the variable currently in, or most recently in?
    which_lower_level = {}
    # if a previous lower_level write exists (with most recent data), it is stored here
    last_lower_level_write = {}

    for var_name, lower_level_op in lower_ops_sorted:
        # Initialize state elements
        higher_level_updated[var_name] = True
        last_higher_level_write[var_name] = None
        in_lower_level[var_name] = False
        which_lower_level[var_name] = None
        last_lower_level_write[var_name] = None

    for var_name, lower_level_op in lower_ops_sorted:
        
        # lower_level info corresponding to variable allocation
        lower_level_index = op_allocation[lower_level_op]
        Lower_object = Lower_objects[lower_level_index]
        lower_level_node = lower_level_instances[lower_level_index]
        lower_level_name = lower_level_node[0]

        evicted = None

        if not graph.nodes[lower_level_op]["write"]: # LOWER LEVEL READ
            logger.info(f"{lower_level_op} hit status in {lower_level_name} is {in_lower_level[var_name]}")
            if in_lower_level[var_name]:
                assert which_lower_level[var_name] == lower_level_name, f"We should not be reading a variable from a {lower_func} if it is already allocated to another one"
                assert Lower_object.check_hit(var_name)
            else: # not a lower_level hit
                assert not Lower_object.check_hit(var_name)

                # add higher_level read, lower_level write
                higher_level_read = f"{higher_func};{higher_level_count}"
                higher_level_count += 1
                graph.add_node(
                    higher_level_read,
                    function=higher_func,
                    allocation="",
                    cost=higher_level_latency,
                    size=16,
                    write=False
                )
                higher_level_ops.append((var_name, higher_level_read))
                new_lower_level_write = sim_util.get_unique_node_name(graph, lower_level_op)
                graph.add_node(
                    new_lower_level_write,
                    function=lower_func,
                    allocation=lower_level_name,
                    cost=lower_level_latency,
                    size=16,
                    write=True
                )
                lower_level_ops_by_allocation[lower_level_index].append(new_lower_level_write)
                # link higher_level read -> lower_level write -> this lower_level read op
                graph.add_edge(higher_level_read, new_lower_level_write, function=higher_func, weight=higher_level_latency)
                graph.add_edge(new_lower_level_write, lower_level_op, function=lower_func, weight=lower_level_latency)

                higher_level_write = None
                if higher_level_updated[var_name]:
                    if last_higher_level_write[var_name]:
                        higher_level_write = last_higher_level_write[var_name]
                        graph.add_edge(higher_level_write, higher_level_read, function=higher_func, weight=higher_level_latency)
                        logger.info(f"retrieving {higher_func} write {higher_level_write}")
                    else:
                        logger.info(f"no previous {higher_func} write existed")
                else:
                    assert last_lower_level_write[var_name] and last_higher_level_write[var_name]
                    higher_level_write = last_higher_level_write[var_name]
                    assert not graph.has_node(higher_level_write)
                    graph.add_node(
                        higher_level_write,
                        function=higher_func,
                        allocation="",
                        cost=higher_level_latency,
                        size=16,
                        write=True
                    )
                    logger.info(f"retrieving {higher_func} write {higher_level_write}")
                    old_lower_level_write = last_lower_level_write[var_name]
                    logger.info(f"adding edge {last_lower_level_write[var_name]}->{last_higher_level_write[var_name]}")
                    graph.add_edge(old_lower_level_write, higher_level_write, function=lower_func, weight=lower_level_latency)
                    graph.add_edge(higher_level_write, higher_level_read, function=higher_func, weight=higher_level_latency)
                    if graph.has_edge(old_lower_level_write, lower_level_op):
                        graph.remove_edge(old_lower_level_write, lower_level_op)
                        logger.info(f"removed edge {old_lower_level_write}->{lower_level_op}")

                evicted = Lower_object.write(var_name, lower_level_op)
                logger.info(f"writing {lower_level_op} to {lower_level_name}")

                # UPDATE STATE ELEMENTS
                higher_level_updated[var_name] = True
                last_higher_level_write[var_name] = higher_level_write
                in_lower_level[var_name] = True
                which_lower_level[var_name] = lower_level_name
                last_lower_level_write[var_name] = new_lower_level_write
        else: # LOWER LEVEL WRITE
            evicted = Lower_object.write(var_name, lower_level_op)

            # UPDATE STATE ELEMENTS
            higher_level_updated[var_name] = False
            last_higher_level_write[var_name] = None
            in_lower_level[var_name] = True
            which_lower_level[var_name] = lower_level_name
            last_lower_level_write[var_name] = lower_level_op

            logger.info(f"writing {lower_level_op} to {lower_level_name}")
        
        lower_level_ops_by_allocation[lower_level_index].append(lower_level_op)
        if evicted:
            assert len(evicted) == 1
            evicted_name = evicted[0]
            logger.info(f"{evicted_name} was evicted from {lower_level_name}, its {higher_func} updated status is {higher_level_updated[evicted_name]} and latest {lower_func} allocation was {which_lower_level[evicted_name]}")
            # if evicted variable was most recently in this lower_level and it is "dirty", must save some state for it in case a subsequent read is needed
            if not higher_level_updated[evicted_name] and which_lower_level[evicted_name] == lower_level_name:
                higher_level_write = f"{higher_func};{higher_level_count}"
                higher_level_count += 1
                higher_level_ops.append((evicted_name, higher_level_write))
                logger.info(f"creating {higher_func} write {higher_level_write} which would be linked from {last_lower_level_write[evicted_name]}")

                # UPDATE STATE ELEMENTS FOR EVICTED VAR
                last_higher_level_write[evicted_name] = higher_level_write
            if which_lower_level[evicted_name] == lower_level_name:
                in_lower_level[evicted_name] = False

    # some higher_level ops may have been created for dirty lower_level eviction but were never needed because that variable was never used again
    higher_level_ops_final = [op for op in higher_level_ops if graph.has_node(op[1])]
    sim_util.topological_layout_plot(graph)
    return lower_level_ops_by_allocation, higher_level_ops_final

def write_df(in_use, hw_element_counts, execution_time, step):
    data = {
        round(i * step, 3): [0] * hw_element_counts["Regs"]
        for i in range(0, int(math.ceil(execution_time / step)))
    }
    for key in in_use:
        for elem in in_use[key]:
            data[key][int(elem[-1])] = 1
    df = pd.DataFrame(data=data).transpose()
    cols = {i: f"reg{i}" for i in range(hw_element_counts["Regs"])}
    df = df.rename(columns=cols)
    df.index.name = "time"
    df.to_csv("logs/reg_use_table.csv")


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
        if not func == "Regs":
            continue
        first_time_step = (computation_graph.nodes[node]["dist"] // step) * step
        out_edge = list(computation_graph.out_edges(node))[0]
        end_time = (
            computation_graph.nodes[node]["dist"]
            + computation_graph.edges[out_edge]["weight"]
        )
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


def check_valid_hw(computation_graph, hw_netlist):
    """
    Checks if for every pair of operators, that isn't already bidirectionally connected,
    there exists at least one register that both operators are bidirectionally connected to.
    """
    hw_graph = nx.DiGraph()

    for node_id, attrs in hw_netlist.nodes(data=True):
        hw_graph.add_node(node_id, **attrs)

    for edge in hw_netlist.edges():
        hw_graph.add_edge(*edge)

    register_nodes = [
        node
        for node, data in hw_graph.nodes(data=True)
        if data.get("type") == "memory" and data.get("function") == "Regs"
    ]

    operator_nodes = [
        node for node, data in hw_graph.nodes(data=True) if data.get("type") == "pe"
    ]

    for op1, op2 in combinations(operator_nodes, 2):
        if hw_graph.has_edge(op1, op2) and hw_graph.has_edge(op2, op1):
            continue
        found_valid_register = False
        for register in register_nodes:
            if (
                hw_graph.has_edge(op1, register)
                and hw_graph.has_edge(register, op1)
                and hw_graph.has_edge(op2, register)
                and hw_graph.has_edge(register, op2)
            ):
                found_valid_register = True
                break

        if not found_valid_register:
            return False

    return True


def pre_schedule(computation_graph, hw_netlist, hw_latency):
    if not check_valid_hw(computation_graph, hw_netlist):
        raise ValueError(
            "Hardware netlist is not valid. Please ensure that every operator node is bi-directionally connected to every register node."
        )


    operator_edges = [
        (u, v, data)
        for u, v, data in computation_graph.edges(data=True)
        if computation_graph.nodes(data=True)[u]["function"]
        not in ["Regs", "Buf", "MainMem"]
        and computation_graph.nodes(data=True)[v]["function"]
        not in ["Regs", "Buf", "MainMem"]
    ]
    register_nodes = [
        node
        for node, data in computation_graph.nodes(data=True)
        if data["function"] == "Regs"
    ]

    for edge in computation_graph.edges(data=True):
        if edge[1] in register_nodes:
            reg_weight = edge[2]["weight"]
            break

    reg_ids = [int(name.split(";")[1]) for name in register_nodes]
    new_node_id = max(reg_ids) + 1
    tmp_op_id = 0

    for u, v, data in operator_edges:  # this shouldn't run after the first iteration
        if (u, v) not in hw_netlist.edges():
            new_node_name_r = f"tmp_op_reg_{tmp_op_id};{new_node_id}"
            new_node_name_w = f"tmp_op_reg_{tmp_op_id};{new_node_id+1}"
            computation_graph.add_node(new_node_name_w, function="Regs", cost=hw_latency["Regs"], write=True)
            computation_graph.add_node(new_node_name_r, function="Regs", cost=hw_latency["Regs"], write=False)
            
            computation_graph.remove_edge(u, v)
            computation_graph.add_edge(u, new_node_name_w, weight=reg_weight)
            computation_graph.add_edge(new_node_name_w, new_node_name_r, weight=reg_weight)
            computation_graph.add_edge(new_node_name_r, v, **data)

            new_node_id += 2
            tmp_op_id += 1


def greedy_schedule(
    computation_graph, hw_element_counts, hw_netlist, save_reg_use_table=False
):
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
    def greedy_schedule_main():
        nonlocal computation_graph, hw_element_counts, hw_netlist
        hw_element_counts["stall"] = np.inf
        # do topo sort from beginning and add dist attribute to each node
        # for longest path to get to it from a gen[0] node.
        # reset layers:
        for node in computation_graph.nodes:
            computation_graph.nodes[node]["layer"] = -np.inf
        computation_graph = assign_upstream_path_lengths_dijkstra(computation_graph)

        stall_counter = 0  # used to ensure unique stall names
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
            generation = [item for item in generation if item not in pushed]

            nodes_in_gen = list(
                filter(lambda x: x[0] in generation, computation_graph.nodes.data())
            )
            funcs_in_gen, counts_in_gen = np.unique(
                list(map(lambda x: x[1]["function"], nodes_in_gen)), return_counts=True
            )

            for func, count in zip(
                funcs_in_gen, counts_in_gen
            ):  # for each function in the generation
                if func == "start" or func == "end":
                    continue
                # if there are more operations of this type than there are hardware elements
                # then we need to add stalls, sort descending by distance from start
                # ie, the ones closest the start get stalled first
                if count > hw_element_counts[func]:
                    func_nodes = list(
                        filter(lambda x: x[1]["function"] == func, nodes_in_gen)
                    )
                    func_nodes = sorted(
                        func_nodes, key=lambda x: x[1]["dist"], reverse=True
                    )

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
                            layer=-layer,
                            allocation="",
                            size=(
                                func_nodes[idx][1]["size"]
                                if "size" in func_nodes[idx][1]
                                else 1
                            ),
                            dist=0,
                        )
                        new_edges = []
                        for edge in out_edges:
                            new_edges.append(
                                (
                                    stall_name,
                                    edge[1],
                                    {"weight": computation_graph.edges[edge]["weight"]},
                                )
                            )
                            new_edges.append(
                                (
                                    edge[0],
                                    stall_name,
                                    {"weight": computation_graph.edges[edge]["weight"]},
                                )
                            )  # edge[0] is same as func_nodes[idx][0]
                        computation_graph.add_edges_from(new_edges)
                        computation_graph.remove_edges_from(out_edges)

                        computation_graph.nodes[func_nodes[idx][0]]["layer"] = -(
                            layer + 1
                        )  # bubble up
                        pushed.append(func_nodes[idx][0])

            hopeful_nodes = list(
                filter(lambda x: x[1]["layer"] >= -layer, computation_graph.nodes.data())
            )
            processed_nodes = list(map(lambda x: x[0], hopeful_nodes))
            processed_graph = nx.subgraph(computation_graph, processed_nodes)

            curr_gen_nodes = list(
                filter(lambda x: x[1]["layer"] == -layer, computation_graph.nodes.data())
            )
            funcs_in_gen, counts_in_gen = np.unique(
                list(map(lambda x: x[1]["function"], curr_gen_nodes)), return_counts=True
            )

            for i, func in enumerate(funcs_in_gen):
                if func in ["start", "end", "stall"]:
                    continue
                assert counts_in_gen[i] <= hw_element_counts[func]
                # do a greedy allocation of the nodes to the hardware elements
                comp_nodes = list(
                    filter(lambda x: x[1]["function"] == func, curr_gen_nodes)
                )
                hw_nodes = list(
                    filter(lambda x: x[1]["function"] == func, hw_netlist.nodes.data())
                )
                for i in range(len(comp_nodes)):
                    computation_graph.nodes[comp_nodes[i][0]]["allocation"] = hw_nodes[i][0]

            layer += 1
        if save_reg_use_table:
            computation_graph, execution_time = assign_time_of_execution(computation_graph)
            log_register_use(computation_graph, 0.1, hw_element_counts, execution_time)
        # return computation_graph
    greedy_schedule_main()

    # take the output of the greedy schedule and look at end times of each node in every layer
    # max end time is the end time of all nodes in that layer and start time = max(et) - cost of the node
    generations = list(nx.topological_generations(computation_graph))
    current_time = 0
    for layer in range(len(generations)):
        max_layer_duration = 0

        # First pass: determine the maximum duration of the layer
        for node in generations[layer]:
            cost = computation_graph.nodes[node].get("cost", 0)
            max_layer_duration = max(max_layer_duration, cost)

        # print(max_layer_duration)
        # Second pass: assign start and end times
        for node in generations[layer]:
            cost = computation_graph.nodes[node].get("cost", 0)
            computation_graph.nodes[node]["start_time"] = current_time + max_layer_duration - cost
            computation_graph.nodes[node]["end_time"] = max_layer_duration

        # Move the current time forward by the duration of this layer
        current_time += max_layer_duration

    return computation_graph

