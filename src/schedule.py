import logging
import os
import heapq

logger = logging.getLogger(__name__)

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from . import sim_util



def calculate_similarity(G, p1, p2):
    # calculate the similarity between these sets using overlap coefficient
    p1_fn_counts = {}
    p1_funcs = []
    p2_fn_counts = {}
    for node in p1:
        fn = G.nodes[node]["function"] 
        if fn not in ["nop", "end"]:
            if fn not in p1_fn_counts:
                p1_fn_counts[fn] = 0
            p1_funcs.append(f"{fn};{p1_fn_counts[fn]}")
            p1_fn_counts[fn] += 1
    p2_funcs = []
    for node in p2:
        fn = G.nodes[node]["function"] 
        if fn not in ["nop", "end"]:
            if fn not in p2_fn_counts:
                p2_fn_counts[fn] = 0
            p2_funcs.append(f"{fn};{p2_fn_counts[fn]}")
            p2_fn_counts[fn] += 1
    
    p_shorter_funcs, p_longer_funcs = (set(p1_funcs), set(p2_funcs)) if len(p1_funcs) < len(p2_funcs) else (set(p2_funcs), set(p1_funcs))
    longer_path = len(p2_funcs) > len(p1_funcs)
    logger.info(f"{p2_funcs}, {p1_funcs}")
    
    intersect = p_longer_funcs.intersection(p_shorter_funcs)
    logger.info(f"intersection is {intersect}")

    logger.info(f"similarity is {len(intersect) / len(p_shorter_funcs)}, p2 longer path status is {longer_path}")
    return longer_path, len(intersect) / len(p_shorter_funcs)

def get_longest_paths(G: nx.DiGraph, num_paths=5, num_unique_slacks=100):
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
    slacks_seen = set()
    logger.info(f"sorted slacks: {sorted_slacks}")
    k = min(num_unique_slacks, len(unique_slacks))
    nodes_to_remove = []
    sorted_unique_slacks = []
    for i in range(len(sorted_slacks)):
        if slacks[sorted_slacks[i]] > unique_slacks[k-1]:
            sorted_slacks = sorted_slacks[:i]
            nodes_to_remove = sorted_slacks[i:]
            break
        if slacks[sorted_slacks[i]] not in slacks_seen:
            sorted_unique_slacks.append(sorted_slacks[i])
            slacks_seen.add(slacks[sorted_slacks[i]])


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
    for node in sorted_unique_slacks:
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

    logger.info(f"paths sorted: {paths_sorted}")
    logger.info(f"k is {k}")
    
    # may only want a certain number of paths
    n = min(len(paths_sorted), num_paths)

    paths_sorted_pruned = [paths_sorted[0]]
    paths_added = 1
    for i in range(1, k):
        next_path = paths_sorted[i]
        logger.info(f"considering path {next_path}")
        similar = False
        # only add a path to the final list if it is less than a certain amount similar to
        # all the other paths we are currently considering
        # if a path is a superset of another one, remove the first path and add this one in

        for i in range(len(paths_sorted_pruned)):
            NEXT_PATH_LONGER = True
            longer_path, similarity = calculate_similarity(G, paths_sorted_pruned[i][1], next_path[1])
            if similarity >= 0.8:
                if longer_path == NEXT_PATH_LONGER and similarity == 1:
                    # replace current path if next one is a superset of it
                    paths_sorted_pruned[i] = next_path
                    logger.info(f"replacing {paths_sorted_pruned[i]} with {next_path}")
                similar = True
        if not similar: 
            paths_sorted_pruned.append(next_path)
            paths_added += 1
        if paths_added == n: break

    return paths_sorted_pruned

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
        self.successors = []
        self.predecessors = []

    def __str__(self):
        return f"========================\nNode id: {self.id}\nName: {self.name}\ntype: {self.type}\nmodule: {self.module}\n========================\n"

    def label(self):
        return f"{self.module}-{self.id}-{self.type}"

class gnt_schedule_parser:
    def __init__(self, build_dir, module_map):
        self.filename = build_dir + "/schedule.gnt"
        self.bom_file = build_dir + "/rtl.rpt"
        self.line_map = {}
        self.loop_indices = []
        self.top_loop = None
        self.node_names = set()
        self.G = nx.DiGraph()
        self.modified_G = nx.DiGraph()
        self.ignore_types = ["ASSIGN", "TERMINATE"]
        self.latest_names = {}
        self.loop_graphs = {} # loop id -> saved graph
        self.module_map = module_map # mapping from ccore module to standard operator name
        self.extra_edges = [] # resource edges
        self.element_counts = {}

    def parse(self):
        self.parse_bom()
        self.parse_gnt_loops()
        self.G, _, _ = self.create_graph_for_loop(self.top_loop)
        #self.remove_loop_nodes()

    def convert(self):
        self.convert_to_standard_dfg()

    def parse_bom(self):
        file = open(self.bom_file, "r")
        lines = file.readlines()
        i = 0
        while (i < len(lines)):
            if lines[i].strip().startswith("[Lib: assembly]"):
                break
            i += 1
        i += 1
        while (i < len(lines)):
            if lines[i].strip().startswith("[Lib: nangate") or lines[i].strip().startswith("TOTAL AREA"):
                break
            data = lines[i].strip().split()
            if data:
                module_type = data[0].split('(')[0]
                if module_type in self.module_map:
                    if data[-1] == 0:
                        logger.warning(f"{module_type} has zero count post assign, setting to 1")
                    if self.module_map[module_type] not in self.element_counts:
                        self.element_counts[self.module_map[module_type]] = 0
                    # if multiple modules are mapped to the same module type, just add them up for count purposes
                    self.element_counts[self.module_map[module_type]] += max(int(data[-1]), 1)
            i += 1
        logger.info(str(self.element_counts))

    def convert_to_standard_dfg(self):
        """
        takes the output of parse_gnt_to_graph and converts
        it to our standard dfg format for use in the rest of the flow.
        We discard catapult-specific information and convert each node
        to one of our standard operators with the correct delay value.
        Also add resource dependencies.

        Arguments:
        - graph: output of parse_gnt_to_graph
        - node_objects: mapping of node -> node class object
        """
        for node in self.G:
            node_data = self.G.nodes[node]
            #print(node, node_data)
            assert node_data["module"] and node_data["module"][:node_data["module"].find('(')] in self.module_map

        #sim_util.topological_layout_plot(self.G)

        for node in self.G:
            node_data = self.G.nodes[node]
            module_prefix = node_data["module"][:node_data["module"].find('(')]
            fn = self.module_map[module_prefix]
            self.modified_G.add_node(
                node,
                id=node_data["id"],
                function=fn,
                cost=node_data["delay"],
                start_time=0,
                end_time=0,
                allocation="",
                library=node_data["library"]
            )
        self.modified_G.add_node(
            "end",
            function="end"
        )
        for node in self.G:
            node_data = self.G.nodes[node]
            for child in self.G.successors(node):
                self.modified_G.add_edge(
                    node, 
                    child, 
                    cost=0, # cost to be set after parasitic extraction 
                    weight=self.modified_G.nodes[node]["cost"]
                ) 
            if not len(list(self.G.successors(node))):
                self.modified_G.add_edge(
                    node, "end",
                    weight=self.modified_G.nodes[node]["cost"]
                )

        #sim_util.topological_layout_plot(self.modified_G)
        while not nx.is_directed_acyclic_graph(self.modified_G):
            cycle = nx.find_cycle(self.modified_G)
            logger.info(f"Graph is not a Directed Acyclic Graph (DAG). Cycle is {cycle}")
            for edge in cycle:
                self.modified_G.remove_edge(edge[0], edge[1])

        # add resource dependencies
        topo_order = longest_path_first_topological_sort(self.modified_G)
        topo_order_by_func = {func: [] for func in self.element_counts.keys()}
        for node in topo_order:
            node_data = self.modified_G.nodes[node]
            if node_data["function"] in self.element_counts:
                topo_order_by_func[node_data["function"]].append(node)
        #print(topo_order_by_func)
        topo_order_by_elem = {func: {i: [] for i in range(self.element_counts[func])} for func in self.element_counts.keys()}
        for func in topo_order_by_func:
            for i in range(len(topo_order_by_func[func])):
                topo_order_by_elem[func][i%self.element_counts[func]].append(topo_order_by_func[func][i])
        #print(topo_order_by_elem)
        for func in topo_order_by_elem:
            for elem in topo_order_by_elem[func]:
                for i in range(len(topo_order_by_elem[func][elem])-1):
                    assert self.modified_G.has_node(topo_order_by_elem[func][elem][i])
                    assert self.modified_G.has_node(topo_order_by_elem[func][elem][i+1])
                    self.modified_G.add_edge(
                        topo_order_by_elem[func][elem][i],
                        topo_order_by_elem[func][elem][i+1],
                        weight=0
                    )
                    logger.info(f"adding resource dependency between {topo_order_by_elem[func][elem][i]} and {topo_order_by_elem[func][elem][i+1]}")
                    self.extra_edges.append((topo_order_by_elem[func][elem][i], topo_order_by_elem[func][elem][i+1]))

        nx.write_gml(self.modified_G, "src/tmp/schedule.gml")
        logger.info(f"longest path length: {nx.dag_longest_path_length(self.modified_G)}")
        logger.info(f"longest path: {nx.dag_longest_path(self.modified_G)}")

    def get_unique_node_name(self, node_id):
        if node_id not in self.latest_names:
            self.latest_names[node_id] = 0
            name = f"{node_id};0"
        else:
            i = self.latest_names[node_id]
            name = f"{node_id};{i}"
            while name in self.node_names:
                i += 1
                name = f"{node_id};{i}"
                assert i < 100000000
            self.latest_names[node_id] = i
        self.node_names.add(name)
            
        return name

    def parse_gnt_loops(self):
        with open(self.filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if not line.startswith("set a("): continue

                node_id = line[line.find('(')+1:line.find(')')]
                self.line_map[node_id] = line
                tokens = line.split()
                node_type = None
                if (line.find(" TYPE ") != -1):
                    for i in range(len(tokens)):
                        if (tokens[i] == "TYPE"):
                            node_type = tokens[i+1]
                            break
                    assert node_type, f"{line}"
                if node_type != "LOOP": continue
                self.loop_indices.append(node_id)
            logger.info(f"loop indices: {self.loop_indices}")

            # ind -> successor inds
            loop_successors = {}

            # determine loop nesting
            for ind in self.loop_indices:
                line = self.line_map[ind]
                tokens = line.split()
                loop_successors[ind] = []
                if (line.find("{CHI ") != -1):
                    for i in range(len(tokens)):
                        if (tokens[i] == "{CHI"):
                            for j in range(i+1, len(tokens)):
                                if tokens[j].startswith('{'): tokens[j] = tokens[j][1:]
                                if tokens[j].endswith('}'): tokens[j] = tokens[j][:-1]
                                if (tokens[j] == "ITERATIONS"): break
                                if tokens[j] in self.loop_indices:
                                    loop_successors[ind].append(tokens[j])
                            break
            
            # determine top loop       
            all_loops = set(self.loop_indices)
            logger.info(f"loop hierarchy: {loop_successors}")
            for ind in self.loop_indices:
                for successor in loop_successors[ind]:
                    if successor in all_loops: all_loops.remove(successor)
            assert len(all_loops) == 1, list(all_loops)
            self.top_loop = list(all_loops)[0]

    def get_node_type(self, node_id):
        line = self.line_map[node_id]
        tokens = self.line_map[node_id].split(' ')
        node_type = ""
        if (line.find(" TYPE ") != -1):
            for i in range(len(tokens)):
                if (tokens[i] == "TYPE"):
                    node_type = tokens[i+1]
                    break
            assert node_type, f"{line}"
        return node_type

    def find_successors(self, node_id):
        tokens = self.line_map[node_id].split(' ')
        successors = []
        if (self.line_map[node_id].find(" SUCCS ") != -1):
            for i in range(len(tokens)):
                if (tokens[i] == "SUCCS"):
                    for j in range(i+1, len(tokens)):
                        if (tokens[j] == "CYCLES"): break
                        if tokens[j] in self.line_map and self.get_node_type(tokens[j]) not in self.ignore_types:
                            successors.append(tokens[j])
        logger.info(f"successors of {node_id} are {successors}")
        return successors

    def find_predecessors(self, node_id):
        tokens = self.line_map[node_id].split(' ')
        predecessors = []
        if (self.line_map[node_id].find(" PREDS ") != -1):
            for i in range(len(tokens)):
                if self.get_node_type(node_id) in self.ignore_types: continue
                if (tokens[i] == "PREDS"):
                    for j in range(i+1, len(tokens)):
                        if (tokens[j] == "SUCCS"): break
                        if tokens[j] in self.line_map and self.get_node_type(tokens[j]) not in self.ignore_types: 
                            predecessors.append(tokens[j])
        return predecessors

    def find_loop_children(self, loop_id):
        assert "{CHI" in self.line_map[loop_id]
        tokens = self.line_map[loop_id].split()

        ids = []
        for i in range(len(tokens)):
            if tokens[i] == "{CHI":
                for j in range(i+1, len(tokens)):
                    if tokens[j] == "ITERATIONS": break
                    if tokens[j].startswith("{"): tokens[j] = tokens[j][1:]
                    if tokens[j].endswith("}"): tokens[j] = tokens[j][:-1]
                    if self.get_node_type(tokens[j]) in self.ignore_types: continue
                    ids.append(tokens[j])
        return ids
    
    def get_num_iterations(self, loop_id):
        assert " ITERATIONS " in self.line_map[loop_id]
        tokens = self.line_map[loop_id].split()
        iters = tokens[tokens.index("ITERATIONS")+1]
        if iters == "Infinite": iters = "1"
        return int(iters)
    
    def connect_set_of_nodes(self, preds, succs, G):
        for pred in preds:
            for succ in succs:
                G.add_edge(pred, succ)
                #logger.info(f"connecting {pred} to {succ}")

    def get_node_info(self, node_id):
        line = self.line_map[node_id].strip()
        assert line.startswith("set a(")
        node_id = line[line.find('(')+1:line.find(')')]
        tokens = line.split()
        node_name = ""
        if (line.find(" NAME ") != -1):
            for i in range(len(tokens)):
                if (tokens[i] == "NAME"):
                    node_name = tokens[i+1]
                    break
            assert node_name != "", f"{line}"
        # print(node_name)
        node_type = self.get_node_type(node_id)
        node_module = ""
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
                    node_delay = float(tokens[i+1][1:]) # take out brackets with [1:-1]
            #print(node_delay)
        node_library = ""
        if (line.find(" LIBRARY ") != -1):
            for i in range(len(tokens)):
                if (tokens[i] == "LIBRARY"):
                    node_library = tokens[i+1]
        return node_name, node_type, node_module, node_delay, node_library
    
    # return labels of roots and leaves
    def create_graph_for_loop(self, loop_id):
        G = nx.DiGraph()
        if loop_id in self.loop_graphs:
            logger.info(f"using presaved version of graph for {loop_id}")
            G_saved = self.loop_graphs[loop_id]
            node_mapping = {}
            # maintain the same structure from the saved graph, but change all the node names
            for node in G_saved:
                assert ';' in node
                new_name = self.get_unique_node_name(node[:node.find(';')])
                node_mapping[node] = new_name
                logger.info(f"changed {node} from presaved graph to {new_name}")
                G.add_node(new_name, **G_saved.nodes[node])
            for node in G_saved:
                for successor in G_saved.successors(node):
                    G.add_edge(node_mapping[node], node_mapping[successor])
        else:
            node_ids = self.find_loop_children(loop_id)
            logger.info(f"loop children for {loop_id}: {node_ids}")
            names = {}
            successors = {}
            predecessors = {}

            # add node objects for each id in loop children
            for node_id in node_ids:
                names[node_id] = self.get_unique_node_name(node_id)
                predecessors[names[node_id]] = []
                successors[names[node_id]] = []
                node_name, node_type, node_module, node_delay, node_library = self.get_node_info(node_id)
                G.add_node(
                    names[node_id],
                    name=node_name,
                    id=node_id,
                    tp=node_type,
                    module=node_module,
                    delay=node_delay,
                    library=node_library
                )
            
            # add edges between nodes
            for node_id in node_ids:
                all_successors = self.find_successors(node_id)
                for successor in all_successors:
                    assert successor in names
                    predecessors[names[successor]].append(names[node_id])
                    successors[names[node_id]].append(names[successor])

            for node_id in node_ids:
                for successor in successors[names[node_id]]:
                    self_loop = successor in predecessors[successor] or names[node_id] in successors[names[node_id]]
                    single_loop = successor in successors[names[node_id]] and names[node_id] in successors[successor]

                    if not (self_loop or single_loop):
                        G.add_edge(names[node_id], successor)
                        logger.info(f"in original function, connecting {names[node_id]} with {successor}")
                    elif self_loop:
                        logger.warning(f"{node_id} contains self loop")
                    else: 
                        assert single_loop
                        if not G.nodes[names[node_id]]["module"]: # meaning its a ccore port
                            assert self.get_node_type(node_id) == "{C-CORE"
                            port_name = G.nodes[names[node_id]]["name"]
                            if port_name.find('.') != -1:
                                port_id = port_name.split('.')[1][0]
                                if port_id == "z": # output
                                    G.add_edge(successor, names[node_id])
                                    logger.info(f"in original function, connecting {names[node_id]} with {successor} as unit to c-core pair")
                                else:
                                    G.add_edge(names[node_id], successor)
                                    logger.info(f"in original function, connecting {names[node_id]} with {successor} as c-core to unit pair")



            
            # create nested loops
            for node_id in node_ids:
                if node_id in self.loop_indices:
                    iters = self.get_num_iterations(node_id)
                    logger.info(f"creating nested loop for {node_id} with {iters} iters")
                    loop_roots, loop_leaves = [], []
                    for _ in range(iters):
                        sub_G, loop_roots_i, loop_leaves_i = self.create_graph_for_loop(node_id)
                        # add all nodes and edges from subgraph into current graph
                        for node in sub_G:
                            logger.info(f"adding node {node} from subgraph")
                            G.add_node(node, **sub_G.nodes[node])
                        for node in sub_G:
                            for successor in sub_G.successors(node):
                                logger.info(f"adding edge between {node} and {successor} after subgraph")
                                G.add_edge(node, successor)
                        loop_roots.append(loop_roots_i)
                        loop_leaves.append(loop_leaves_i)
                    #connect incoming nodes to first iteration of loop, and outgoing nodes to outgoing nodes of last loop iteration
                    self.connect_set_of_nodes(predecessors[names[node_id]], loop_roots[0], G)
                    self.connect_set_of_nodes(loop_leaves[-1], successors[names[node_id]], G)
                    # connect intermediate loop iterations
                    for i in range(1, iters):
                        self.connect_set_of_nodes(loop_leaves[i-1], loop_roots[i], G)
            
            #prune graph for unnecessary node types
            nodes_removed = []
            edges_to_remove = []
            for node in G:
                node_data = G.nodes[node]
                #print(node, node_data)

                if node_data["module"] and node_data["module"][:node_data["module"].find('(')] in self.module_map:
                    assert node_data["module"].find('(') != -1
                    logger.info(f"{node} remaining in pruned graph")
                else:
                    logger.info(f"removing {node}")
                    nodes_removed.append(node)
                if G.has_edge(node, node):
                    edges_to_remove.append((node, node))

            logger.info(edges_to_remove)
            for edge in edges_to_remove:
                G.remove_edge(edge[0], edge[1])

            for node in nodes_removed:
                if node in G:
                    preds = list(G.predecessors(node))
                    succs = list(G.successors(node))
                    
                    if node[:node.find(';')] not in self.loop_indices:
                        # Create edges from predecessors to successors
                        for pred in preds:
                            for succ in succs:
                                if pred != succ:  # Avoid self-loops
                                    logger.info(f"adding edge between {pred} and {succ}")
                                    G.add_edge(pred, succ)
                    else:
                        logger.info(f"{node} is a loop; just remove it")
                    
                    logger.info(f"actually removing {node}")
                    # Remove the node
                    G.remove_node(node)

            # save graph in case we want to create one from this loop id again
            self.loop_graphs[loop_id] = G
        
        assert nx.is_directed_acyclic_graph(G), nx.find_cycle(G)
        topo_gens = list(nx.topological_generations(G))
        roots, leaves = ([], []) if not len(topo_gens) else (topo_gens[0], topo_gens[-1])
                
        return G, roots, leaves
    

if __name__ == "__main__":
    if os.path.exists("src/tmp/schedule.log"): os.remove("src/tmp/schedule.log")
    logging.basicConfig(filename=f"src/tmp/schedule.log", level=logging.INFO)
    module_map = {
        "add": "Add",
        "mult": "Mult",
        "ccs_ram_sync_1R1W_rwport": "Buf",
        "ccs_ram_sync_1R1W_rport": "Buf",
        "nop": "nop"
    }
    parser = gnt_schedule_parser("src/tmp/benchmark/build/MatMult.v1", module_map)
    parser.parse()
    print("finished parsing")
    nx.write_gml(parser.G, "src/tmp/test_graph.gml")
    #sim_util.topological_layout_plot(parser.G)
    parser.convert()
    print("finished converting")
    sim_util.topological_layout_plot(parser.modified_G, extra_edges=parser.extra_edges)
    nx.write_gml(parser.modified_G, "src/tmp/modified_test_graph.gml")
    lp = get_longest_paths(parser.modified_G)

    """logging.basicConfig(filename=f"src/tmp/schedule.log", level=logging.INFO)
    G = nx.read_gml("src/tmp/modified_test_graph.gml")
    lp = get_longest_paths(G)"""
    print(f"number of paths found: {len(lp)}")