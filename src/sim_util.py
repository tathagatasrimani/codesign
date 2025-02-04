import math
import ast
import glob
import os
import datetime
import logging
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from . import dfg_algo
from . import hw_symbols
from .global_constants import SEED

rng = np.random.default_rng(SEED)

def read_dataframe():
    """
    Reads a pandas DataFrame from a set file in params,
    corresponding to data about area, R, and C for specific tech nodes and standard cells.

    Returns:
    - DataFrame with a MultiIndex for tech node and standard cell.
    """
    return pd.read_csv("params/std_cell_data.csv", index_col=["tech node", "standard cell"])

# adds all mallocs and frees to vectors, and finds the next cfg node in the data path,
# returning the index of that node
def find_next_data_path_index(data_path, i, mallocs, frees):
    """
    Finds the next index in the data path that corresponds to a new computation node,
    updating malloc and free operations lists along the way.

    This function iterates through the data path of the simulation, starting from a given index,
    to identify the next point where a change in the hardware configuration or memory allocation occurs.
    It updates lists of memory allocation ('malloc') and deallocation ('free') operations encountered
    during this traversal.

    The data_path is a list of lists where each element is node, a memory operation, or a
    piece of instrumentation. For example, the following are valid elements of the data_path:
    ['10', '16'], ['pattern_seek_3'], ['malloc', '16', 'c_1', '(3, 3)'], ['free', 'c_1']

    Mallocs and Frees updated in place.

    Parameters:
    - i (int): The starting index in the data path from which the search begins.
    - mallocs (list): A list to be updated with memory allocation operations found during the traversal.
    Each element is a list representing a 'malloc' operation.
    - frees (list): A list to be updated with memory deallocation operations found during the traversal.
    Each element is a list representing a 'free' operation.

    Returns:
    - tuple: A tuple containing three elements:
        1. int: The index of the next configuration node in the data path.
        2. bool: A flag indicating whether a 'pattern_seek' operation was encountered.
        3. int: The maximum iterations to unroll if a 'pattern_seek' operation is encountered.
    """
    pattern_seek = False
    max_iters = 1
    if i == len(data_path):
        return i, pattern_seek, max_iters
    # print(f"i: {i}, len(self.data_path): {len(self.data_path)}, self.data_path: {self.data_path}")
    while len(data_path[i]) != 2:
        if len(data_path[i]) == 0:
            break
        elif len(data_path[i]) == 1:
            if data_path[i][0].startswith("pattern_seek"):
                pattern_seek = True
                max_iters = int(data_path[i][0][data_path[i][0].rfind("_") + 1 :])
        if data_path[i][0] == "malloc":
            mallocs.append(data_path[i])
        elif data_path[i][0] == "free":
            frees.append(data_path[i])
        i += 1
        if i == len(data_path):
            break
    return i, pattern_seek, max_iters


def find_nearest_power_2(num):
    """
    Finds the nearest memory size to scale based on the input.

    Parameters:
        num (int): The memory size to be scaled.

    Returns:
        int: The nearest appropriate memory size.
    """
    if num == 0:
        return 0
    if num < 512:
        return 512
    if num > 536870912:
        return 536870912
    return 2 ** math.ceil(math.log(num, 2))


def make_node(graph, id, name, ctx, opname):
    """
    Creates a node in the given graph with specified attributes.

    Parameters:
        graph (dfg_algo.Graph): The graph to which the node will be added.
        id (str): The identifier for the new node.
        name (str): The name of the new node.
        ctx (AST Context): The AST context of the node ~ the operation .
        opname (str): The operation name associated with the node.

    Returns:
        None
    """
    annotation = ""
    if ctx == ast.Load or ctx == ast.Store:
        annotation = "Register"
    dfg_node = dfg_algo.Node(name, opname, id, memory_links=set())
    graph.gv_graph.node(id, name + "\n" + annotation)
    graph.roots.add(dfg_node)
    graph.id_to_Node[id] = dfg_node


def make_edge(graph, source_id, target_id, annotation=""):
    source, target = graph.id_to_Node[source_id], graph.id_to_Node[target_id]
    graph.gv_graph.edge(source_id, target_id, label=annotation)
    target_node = graph.id_to_Node[target_id]
    if target_node in graph.roots:
        graph.roots.remove(target_node)
    source.children.append(target)
    target.parents.append(source)


def get_matching_bracket_count(name):
    """
    Counts matching brackets in a given name string.

    Parameters:
        name (str): The string in which to count brackets.

    Returns:
        int: The count of matching brackets in the name.
    """
    if name.find("[") == -1:
        return 0
    bracket_count = 0
    name = name[name.find("[") + 1 :]
    bracket_depth = 1
    while len(name) > 0:
        front_ind = name.find("[")
        back_ind = name.find("]")
        if back_ind == -1:
            break
        if front_ind != -1 and front_ind < back_ind:
            bracket_depth += 1
            name = name[front_ind + 1 :]
        else:
            bracket_depth -= 1
            name = name[back_ind + 1 :]
            if bracket_depth == 0:
                bracket_count += 1
    return bracket_count


def get_var_name_from_arr_access(arr_access):
    """
    Convert 'b[i][j]' -> 'b'.
    So that memory operations can be performed on the variable b.
    """
    bracket_ind = arr_access.find("[")
    if bracket_ind != -1:
        return arr_access[:bracket_ind]
    return arr_access


def update_computation_dfg_with_latency(computation_dfg, latency):
    """
    Updates the computation dfg with the latency of each operation.

    Parameters:
        computation_dfg (nx.Digraph): A list of operations in the computation dfg.
        latency (dict): A dictionary of operation names to their latencies.

    Returns:
        None;
        The computation dfg is updated in place.
    """
    for node in computation_dfg.nodes:
        func = computation_dfg.nodes[node]["function"]
        if func in latency:
            computation_dfg.nodes[node]["weight"] = latency[func]
    for edge in computation_dfg.edges:
        node = edge[0]
        func = computation_dfg.nodes.data()[node]["function"]
        if func == "stall":
            func = node.split("_")[3]
        computation_dfg.edges[edge]["weight"] = latency[func]
        


def get_dims(arr):
    """
    Extracts the dimensions from a given array representation.

    This function parses an array-like structure to determine its dimensions.
    It supports both tuple and array representations. The function is typically used in
    the context of memory operations to understand the shape and size of data structures,
    particularly when allocating memory.

    Parameters:
    - arr (list): A list representing the array or tuple. Each element in the list is a string
    representing the size of a dimension. The elements may be formatted as '(size,' for tuples or
    'size,' for arrays.

    Returns:
    - list: A list of integers where each integer represents the size of a dimension in the array.
    This list provides a multi-dimensional structure of the array or tuple.

    Example:
    - For an input like `['(3,', '4)']`, which represents a 2D tuple, the function would return `[3, 4]`.

    """
    dims = []
    if arr[0][0] == "(":  # processing tuple
        dims.append(int(arr[0][1 : arr[0].find(",")]))
        if len(arr) > 2:
            for dim in arr[1:-1]:
                dims.append(int(dim[:-1]))
        if len(arr) > 1:
            dims.append(int(arr[-1][:-1]))
    else:  # processing array
        dims.append(int(arr[0][1:-1]))
        if len(arr) > 2:
            for dim in arr[1:-1]:
                dims.append(int(dim[:-1]))
        if len(arr) > 1:
            dims.append(int(arr[-1][:-1]))
    return dims


def verify_can_execute(
    computation_graph, hw_spec_netlist, generation=None, should_update_arch=False
):
    """
    DEPRECATED?
    Determines whether or not the computation graph can be executed on the netlist
    specified in hw_spec.

    Topologically orders the computation graph (C) and checks if the subgraph of C determined
    by the ith and i+1th and i-1th order in the topo sort is monomorphic to the netlist.

    The way this works, if I have 10 addition operations to do in this node of the DFG, and 20 adders
    it should always allocate 10 adders to this node. <- TODO: VERIFY EXPERIMENTALLY

    if should_update_arch is true, does a graph compose onto the netlist, and returns the new netlist.
    This is done here to avoid looping over the topo ordering twice.
    TODO: seperate this out into a different function.

    Raises exception if the hardware cannot execute the computation graph.

    Parameters:
        computation_graph - nx.DiGraph of operations to be executed.
        hw_spec (HardwareModel): An object representing the current hardware allocation and specifications.
        generation (list): A list of nodes in the computation graph at the current generation.
                    If this is supplied, only do the check for this single generation.
        should_update_arch (bool): A flag indicating whether or not the hardware architecture should be updated.
                    Only set True from architecture search. Set false when running simulation.

    Returns:
        nx.Digraph: returns the temporary local digraph created from generation if generation is not None.
            if there's not monomorphicity, returns None instead.
        nx.Digraph: returns the new hardware netlist if should_update_arch is True.
        bool: True if the computation graph can be executed on the netlist, False otherwise.
            if neither generation nor should_update_arch is specified, returns True or False.
    """
    if generation is not None:
        generations = [generation]
    else:
        generations = nx.topological_generations(computation_graph)
    for gen in generations:
        temp_C = nx.DiGraph()
        for node in gen:
            temp_C.add_nodes_from([(node, computation_graph.nodes[node])])
            for child in computation_graph.successors(node):
                if child not in temp_C.nodes:
                    temp_C.add_nodes_from([(child, computation_graph.nodes[child])])
                temp_C.add_edge(node, child)
            for parent in computation_graph.predecessors(node):
                if parent not in temp_C.nodes:
                    temp_C.add_nodes_from([(parent, computation_graph.nodes[parent])])
                temp_C.add_edge(parent, node)
        dgm = nx.isomorphism.DiGraphMatcher(
            hw_spec_netlist,
            temp_C,
            node_match=lambda n1, n2: n1["function"] == n2["function"]
            or n2["function"] == None  # hw_graph can have no ops
            or n2["function"] == "stall",
        )
        if not dgm.subgraph_is_monomorphic():
            if should_update_arch:
                hw_spec_netlist = update_arch(temp_C, hw_spec_netlist)
                dgm = nx.isomorphism.DiGraphMatcher(
                    hw_spec_netlist,
                    temp_C,
                    node_match=lambda n1, n2: n1["function"] == n2["function"]
                    or n2["function"]
                    == None,  # hw_graph can have no ops, but netlist should not
                )
            if generation is not None:
                print(f"failed monomorphic")
                print(f"temp_c: {temp_C.nodes}")
                print(f"temp_c edges: {temp_C.edges}")
                print(f"temp_c_nodes: {temp_C.nodes.data(True)}")
                print(f"gen: {gen}")
                return None
            else:
                return False

        mapping = dgm.subgraph_monomorphisms_iter().__next__()
        for hw_node, op in mapping.items():
            if op in gen:  # only bind ops in the current generation
                hw_spec_netlist.nodes[hw_node]["allocation"].append(op.split(";")[0])
                computation_graph.nodes[op]["allocation"] = hw_node
                temp_C.nodes[op]["allocation"] = hw_node

    if should_update_arch:
        return hw_spec_netlist
    if generation is not None:
        return temp_C
    else:
        return True


def localize_memory(hw, computation_graph, total_computation_graph=None):
    """
    Updates memory in buffers (cache) to ensure that the data needed for the coming DFG node
    is in the cache.

    Sets a flag to indicate whether or not there was a cache hit or miss. This affects latency
    calculations.

    Currently not in use.
    """
    for node, data in dict(
        filter(lambda x: x[1]["function"] == "Regs", computation_graph.nodes.data())
    ).items():
        var_name = node.split(";")[0]
        cache_hit = False
        cache_hit = (
            hw.netlist.nodes[data["allocation"]]["var"] == var_name
        )  # no need to refetch from mem if var already in reg.

        mapped_edges = map(
            lambda edge: (edge[0], hw.netlist.nodes[edge[0]]),
            hw.netlist.in_edges(computation_graph.nodes[node]["allocation"], data=True),
        )

        in_bufs = list(
            filter(lambda node_data: node_data[1]["function"] == "Buf", mapped_edges)
        )
        for buf in in_bufs:
            cache_hit = buf[1]["memory_module"].find(var_name) or cache_hit
            if cache_hit:
                break
        if not cache_hit:
            # just choose one at random. Can make this smarter later.
            buf = rng.choice(in_bufs)  # in_bufs[0] # just choose one at random.

            size = -1 * buf[1]["memory_module"].read(
                var_name
            )  # size will be negative because cache miss.
            # add buf and mem nodes to indicate when explicit mem reads occur. ie. add a new one for each cache miss.
            if total_computation_graph is not None:
                active_graph = total_computation_graph
            else:
                active_graph = computation_graph
            buf_idx = len(
                list(
                    filter(
                        lambda x: x[1]["function"] == "Buf",
                        active_graph.nodes.data(),
                    )
                )
            )
            mem_idx = len(
                list(
                    filter(
                        lambda x: x[1]["function"] == "MainMem",
                        active_graph.nodes.data(),
                    )
                )
            )
            mem = list(
                filter(lambda x: x[1]["function"] == "MainMem", hw.netlist.nodes.data())
            )[0][0]
            active_graph.add_node(
                f"Buf{buf_idx}", function="Buf", allocation=buf[0], size=size
            )
            active_graph.add_node(
                f"Mem{mem_idx}", function="MainMem", allocation=mem, size=size
            )
            active_graph.add_edge(f"Mem{mem_idx}", f"Buf{buf_idx}", function="Mem")
            active_graph.add_edge(f"Buf{buf_idx}", node, function="Mem")


def add_cache_mem_access_to_dfg(
    scheduled_graph: nx.DiGraph, node_name, buf_latency: float, mem_latency: float, io_latency : float, mem_count, buf_count
):
    """
    Add cache and memory nodes to the scheduled graph to indicate when explicit memory reads occur.
    """
    node, data = scheduled_graph.nodes[node_name]
    size = 16 #data['size'] #-hardcoded for now; TODO: come back and fix
    scheduled_graph.add_node(
        f"Buf{buf_count}",
        function="Buf",
        allocation="",
        cost=buf_latency,
        size=size,
    )
    scheduled_graph.add_node(
        f"Mem{mem_count}",
        function="MainMem",
        allocation="",
        cost=mem_latency,
        size=size,
    )
    scheduled_graph.add_node(
        f"IO{mem_count}",
        function="OffChipIO",
        allocation="",
        cost=io_latency,
        size=size,
    )
    # weight of edge is latency of parent
    scheduled_graph.add_edge(f"Mem{mem_count}", f"IO{mem_count}", function="MainMem", weight=mem_latency)
    scheduled_graph.add_edge(f"IO{mem_count}", f"Buf{buf_count}", function="OffChipIO", weight=io_latency)
    scheduled_graph.add_edge(f"Buf{buf_count}", node, function="Buf", weight=buf_latency)

    return scheduled_graph


def find_upstream_node_in_graph(graph:nx.DiGraph, func:str, node:str, kill=True):
    """
    Find the upstream node in the graph with the specified function. Assumes all nodes have only one in edge.
    And that the desired function exists somewhere upstream.
    Returns the first upstream node with the desired function.
    Parameters:
        graph (nx.DiGraph): The graph to search.
        func (str): The function to search for.
        node (str): The node from which to search.
        kill (bool): raise an exception if no valid upstream node is found
    """
    func_in = []
    while len(func_in) == 0:
                in_edges = list(map(
                            lambda x: (x[0], graph.nodes[x[0]]),
                            graph.in_edges(node),
                        ))
                if len(in_edges) == 0:
                    if kill:
                        raise Exception(f"Could not find upstream node with function {func}")
                    else:
                        return None
                node = in_edges[0][0] # only need the name
                func_in = list(
                    filter(
                        lambda x: x[1]["function"] == func,
                        in_edges,
                    )
                ) # assuming only one upstream node
    return func_in[0]

def prune_buffer_and_mem_nodes(computation_graph: nx.DiGraph, hw_netlist: nx.DiGraph, sdc_schedule: bool = False):
    """
    Call after allocation to remove unnecessary buffer and memory nodes.
    Removes memory nodes when the data is already in the buffer.
    Removes buffer nodes when the data is already in the registers.
    """
    def check_buffer_reg_hit(reg_node):
        var_name = reg_node[0].split(";")[0]
        allocated_reg = reg_node[1]["allocation"]
        allocated_reg_data = hw_netlist.nodes[allocated_reg]
        logger.info(f"allocated reg data for {allocated_reg} is {allocated_reg_data}, var name is {var_name}")
        if not reg_node[1]["write"]:
            buf_in = find_upstream_node_in_graph(computation_graph, "Buf", reg_node[0]) 
            io_in = find_upstream_node_in_graph(computation_graph, "OffChipIO", buf_in[0])
            mem_in = find_upstream_node_in_graph(computation_graph, "MainMem", io_in[0])
            #print(hw_netlist.nodes[buf_in[1]["allocation"]]["memory_module"].memory.locations)
            if allocated_reg_data["var"] == var_name:
                # remove the buffer and memory nodes
                logger.info(f"register hit for {var_name} in {reg_node}")
                computation_graph.remove_node(buf_in[0])
                computation_graph.remove_node(mem_in[0])
                computation_graph.remove_node(io_in[0])
            elif hw_netlist.nodes[buf_in[1]["allocation"]]["memory_module"].find(var_name):
                # remove the memory node
                logger.info(f"cache hit for {var_name}")
                computation_graph.remove_node(mem_in[0])
                computation_graph.remove_node(io_in[0])
                size = int(hw_netlist.nodes[buf_in[1]["allocation"]]["memory_module"].read(var_name))
                computation_graph.nodes[buf_in[0]]["size"] = size
                computation_graph.nodes[reg_node[0]]["size"] = size
            else:
                # read from memory and add to cache
                logger.info(f"reading {var_name} from memory, adding to cache")
                size = int(-1*hw_netlist.nodes[buf_in[1]["allocation"]]["memory_module"].read(var_name))
                computation_graph.nodes[mem_in[0]]["size"] = size
                computation_graph.nodes[io_in[0]]["size"] = size
                computation_graph.nodes[buf_in[0]]["size"] = size
                computation_graph.nodes[reg_node[0]]["size"] = size
        hw_netlist.nodes[allocated_reg]["var"] = var_name
        logger.info(f"writing {var_name} to {allocated_reg}")
    logger.info("starting pruning process")
    layer = 0
    if sdc_schedule:
        regs_sorted = sorted(list(filter(lambda x: x[1]["function"] == "Regs", computation_graph.nodes.data())), key=lambda x: x[1]["start_time"])
        #print(regs_sorted)
        for reg_node in regs_sorted:
            check_buffer_reg_hit(reg_node)
    else:
        gen = list(filter(lambda x: x[1]["layer"] == layer, computation_graph.nodes.data()))
        #print(gen)
        while len(gen) != 0:
            reg_nodes = list(filter(lambda x: x[1]["function"] == "Regs", gen))
            for reg_node in reg_nodes:
                check_buffer_reg_hit(reg_node)
            layer += 1
            gen = list(
                filter(lambda x: x[1]["layer"] == layer, computation_graph.nodes.data())
            )
            #print(gen)
    return computation_graph


def update_arch(computation_graph, hw_netlist):
    """
    Updates the hardware architecture to include the computation graph.
    Based on graph composition. But need to rename nodes in computation graph s.t. nodes are not
    unnecessarily duplicated. For example, nodes in two different computation_graph
    maybe named '+;39' gets renamed to 'Add0', and 'a[i][j]' gets renamed to 'Reg0';
    This ensures the next time we're trying to compose '+;40' we can compose that onto 'Add0'
    instead of creating a new node.
    """

    c_graph_func_counts = {}
    mapping = {}
    for node in computation_graph.nodes:
        func = computation_graph.nodes.data()[node]["function"]
        if func not in c_graph_func_counts.keys():
            c_graph_func_counts[func] = 0
        mapping[node] = func + str(c_graph_func_counts[func])
        c_graph_func_counts[func] += 1
    comp_graph = nx.relabel_nodes(computation_graph, mapping, copy=True)

    composition = nx.compose(hw_netlist, comp_graph)
    for n in composition.nodes:
        composition.nodes[n]["allocation"] = []
    return composition


def rename_nodes(G, H, H_generations=None, curr_last_nodes=None, modified_regs=set()):
    """
    Rename nodes in H to avoid collisions in G.
    try to rename the values in H_generations here?
    currently it just gets calculated again after rename
    """

    relabelling = {}
    found_alignment = False
    # align
    if H_generations is not None and curr_last_nodes is not None:
        for elem in H_generations[0]:
            for elem_2 in curr_last_nodes:
                if elem.split(";")[0] == elem_2.split(";")[0]:
                    relabelling[elem] = elem_2
                    found_alignment = True
                    if G.nodes[elem_2]["function"] == "Regs" and elem_2 not in modified_regs:
                        parents = list(G.predecessors(elem_2))
                        assert len(parents) <= 1, f"more than 1 parent for Reg. {elem_2}: {G.nodes[elem_2]} has parents {[G.nodes[parent] for parent in parents]}"
                        name = get_unique_node_name(G, elem_2)
                        G.add_node(
                            name,
                            function=G.nodes[elem_2]["function"],
                            cost=G.nodes[elem_2]["cost"],
                            write=True
                        )
                        if parents:
                            G.remove_edge(parents[0], elem_2)
                            G.add_edge(parents[0], name, weight=G.nodes[elem_2]["cost"])
                        G.add_edge(name, elem_2, weight=G.nodes[elem_2]["cost"])
                        modified_regs.add(elem_2)
                    break

    for node in H.nodes:
        if node in relabelling:
            continue
        new_node = node
        while new_node in G or new_node in relabelling.values():
            # Rename the node to avoid collision
            new_node = get_unique_node_name(G, new_node)
        # G.nodes[new_node].update(H.nodes[node])
        relabelling[node] = new_node
    nx.relabel_nodes(H, relabelling, copy=False)
    return found_alignment


def get_unique_node_name(G, node):
    var_name, count = node.split(";")
    count = int(count)
    count += 1
    new_node = f"{var_name};{count}"
    while new_node in G:
        count += 1
        new_node = f"{var_name};{count}"
    return new_node


def compose_entire_computation_graph(
    cfg_node_to_dfg_map, id_to_node, data_path, data_path_vars, latency, plot=False
):
    """
    Composes a large DFG from the smaller DFGs.

    Parameters:
        cfg (CFG): The control flow graph of the program.
        cfg_node_to_hw_map (dict): A mapping of CFG nodes to hardware graphs represented by nx.DiGraphs.

    Returns:
        nx.DiGraph: The large DFG composed from the smaller DFGs.
    """
    computation_dfg = nx.DiGraph()
    curr_last_nodes = []
    mallocs = []
    modified_regs = set()
    i = find_next_data_path_index(data_path, 0, mallocs, [])[0]
    while i < len(data_path):
        node_id = data_path[i][0]
        vars = data_path_vars[i]

        node = id_to_node[node_id]
        dfg = cfg_node_to_dfg_map[node]

        if nx.is_empty(dfg):
            i = find_next_data_path_index(data_path, i + 1, mallocs, [])[0]
            continue

        # plug in index values for array accesses
        for node in dfg.nodes:
            var, id = node.split(";")
            if len(var) == 1:
                continue
            array = var.split("[")
            if len(array) == 1:  # not an array
                continue
            var_name = array[0]
            indices = [arr.split("]")[0] for arr in array[1:]]
            for idx in indices:
                if idx in vars.keys():
                    var_name += f"[{vars[idx]}]"
                else:
                    var_name += f"[{idx}]"
            var_name += f";{id}"
            dfg = nx.relabel_nodes(dfg, {node: var_name})

        generations = list(nx.topological_generations(dfg))
        found_alignment = rename_nodes(
            computation_dfg, dfg, generations, curr_last_nodes, modified_regs
        )
        print(modified_regs)
        computation_dfg = nx.compose(computation_dfg, dfg)

        curr_last_nodes = list(nx.topological_generations(nx.reverse(dfg.copy())))[0]
        #print("last nodes: ", curr_last_nodes)

        i = find_next_data_path_index(data_path, i + 1, mallocs, [])[0]

    last_gen = list(nx.topological_generations(nx.reverse(computation_dfg.copy())))[0]

    # create end node and connect all last nodes to it
    computation_dfg.add_node("end", function="end")
    for node in last_gen:
        computation_dfg.add_edge(
            node, "end", weight=latency[computation_dfg.nodes[node]["function"]]
        )
    #nx.write_gml(computation_dfg, get_latest_log_dir()+"/computation_dfg_test.gml")

    if plot:
        topological_layout_plot(computation_dfg, reverse=True)
    return computation_dfg, mallocs


def topological_layout_plot(graph, reverse=False, extra_edges=None):
    # Compute the topological order of the nodes
    if nx.is_directed_acyclic_graph(graph):
        topological_order = list(nx.topological_sort(graph))
    else:
        cycle = nx.find_cycle(graph)
        raise ValueError(f"Graph is not a Directed Acyclic Graph (DAG), topological sorting is not possible. Cycle is {cycle}")
    
    # Group nodes by level in topological order
    levels = defaultdict(int)
    in_degrees = {node: graph.in_degree(node) for node in graph.nodes()}
    
    for node in topological_order:
        level = 0 if in_degrees[node] == 0 else max(levels[parent] + 1 for parent in graph.predecessors(node))
        levels[node] = level
    
    # Arrange nodes in horizontal groups based on level
    level_nodes = defaultdict(list)
    for node, level in levels.items():
        level_nodes[level].append(node)
    
    # Assign positions: group nodes by levels from top to bottom
    pos = {}
    for level, nodes in level_nodes.items():
        x_positions = np.linspace(-len(nodes)/2, len(nodes)/2, num=len(nodes))
        for x, node in zip(x_positions, nodes):
            pos[node] = (x, -level)

    if extra_edges:
        edge_colors = ['red' if (u, v) in extra_edges else 'gray' for (u, v) in graph.edges()]
    else:
        edge_colors = ['gray' for (u, v) in graph.edges()]
    
    # Draw the graph with curved edges to avoid overlap
    plt.figure(figsize=(10, 6))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color=edge_colors, node_size=700, font_size=10, connectionstyle="arc3,rad=0.2")
    
    # Draw dashed lines between topological levels
    max_level = max(level_nodes.keys())
    for level in range(max_level):
        plt.axhline(y=-(level + 0.5), color='gray', linestyle='dashed', linewidth=0.5)

    # Show the graph
    plt.show()


def topological_layout_plot_side_by_side(
    graph1,
    graph2,
    reverse=False,
    edge_labels=False,
):
    graph1_copy = graph1.copy()
    generations = (
        reversed(list(nx.topological_generations(nx.reverse(graph1_copy))))
        if reverse
        else nx.topological_generations(graph1_copy)
    )

    for layer, nodes in enumerate(generations):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            graph1_copy.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos1 = nx.multipartite_layout(graph1_copy, subset_key="layer")

    graph2_copy = graph2.copy()
    generations = (
        reversed(list(nx.topological_generations(nx.reverse(graph2_copy))))
        if reverse
        else nx.topological_generations(graph2_copy)
    )

    for layer, nodes in enumerate(generations):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            graph2_copy.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos2 = nx.multipartite_layout(graph2_copy, subset_key="layer")

    fig, ax = plt.subplots(1, 2, figsize=(9, 9))
    nx.draw_networkx(graph1_copy, pos=pos1, ax=ax[0])
    nx.draw_networkx(graph2_copy, pos=pos2, ax=ax[1])

    if edge_labels:
        edges_1 = {}
        edges_2 = {}
        for u, v, data in graph1_copy.edges(data=True):
            edges_1[(u, v)] = data["weight"]
        for u, v, data in graph2_copy.edges(data=True):
            edges_2[(u, v)] = data["weight"]
        nx.draw_networkx_edge_labels(graph1_copy, pos1, edge_labels=edges_1, ax=ax[0])
        nx.draw_networkx_edge_labels(graph2_copy, pos2, edge_labels=edges_2, ax=ax[1])

    plt.show()

def plot_schedule_gantt(scheduled_dfg : nx.DiGraph):
    """
    Create a Gantt Plot of all operations in the scheduled DFG.
    """
    sorted_nodes = sorted(scheduled_dfg.nodes(data=True), key=lambda x: x[1]["start_time"])[:-1]

    operations = [
        (node[0], node[1]["start_time"], node[1]["end_time"], node[1]["function"]) for node in sorted_nodes
    ]

    # Generate unique colors for each resource
    unique_resources = list(set(op[3] for op in operations))
    color_map = {resource: plt.cm.tab10(i / len(unique_resources)) for i, resource in enumerate(unique_resources)}

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each operation with corresponding color
    for i, (operation, start, end, resource) in enumerate(operations):
        ax.barh(operation, (end - start), left=start, align='center', color=color_map.get(resource, "gray"))

    # Create legend
    handles = [plt.Rectangle((0,0),1,1, color=color_map[resource]) for resource in unique_resources]
    ax.legend(handles, unique_resources, title="Resources")

    # Format x-axis
    plt.xticks(rotation=45)

    # Labels and title
    plt.xlabel("Time (seconds)")
    plt.ylabel("Operations")
    plt.title("Scheduled Operations Timeline")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    return fig, ax


def convert_tech_params_to_si(latency, active_power, passive_power, frequency):
    """
    Convert the tech params from the input units to the SI units.
    latency in cycles -> seconds
    active_power in nW -> W
    passive_power in nW -> W
    """
    return latency / frequency, active_power / 1e9, passive_power / 1e9


def get_latest_log_dir():
    log_dirs = glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), "../logs/*-*-*_*-*-*")))
    log_dirs = sorted(
        log_dirs,
        key=lambda x: datetime.datetime.strptime(x.split("/")[-1], "%Y-%m-%d_%H-%M-%S"),
    )
    return log_dirs[-1]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~ SYMBOLIC UTILS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def generate_init_params_from_rcs_as_symbols(rcs):
    """
    Just some format conversion
    keys are strings
    """
    initial_params = {}
    for elem in rcs["Reff"]:
        initial_params[hw_symbols.symbol_table["Reff_" + elem]] = rcs["Reff"][elem]
        initial_params[hw_symbols.symbol_table["Ceff_" + elem]] = rcs["Ceff"][elem]
    initial_params[hw_symbols.V_dd] = rcs["other"]["V_dd"]
    initial_params[hw_symbols.MemReadEact] = rcs["other"]["MemReadEact"]
    initial_params[hw_symbols.MemWriteEact] = rcs["other"]["MemWriteEact"]
    initial_params[hw_symbols.MemPpass] = rcs["other"]["MemPpass"]
    initial_params[hw_symbols.BufReadEact] = rcs["other"]["BufReadEact"]
    initial_params[hw_symbols.BufWriteEact] = rcs["other"]["BufWriteEact"]
    initial_params[hw_symbols.BufPpass] = rcs["other"]["BufPpass"]
    initial_params[hw_symbols.OffChipIOL] = rcs["other"]["OffChipIOL"]
    initial_params[hw_symbols.OffChipIOPact] = rcs["other"]["OffChipIOPact"]

    # CACTI
    initial_params[hw_symbols.C_g_ideal] = rcs["Cacti"]["C_g_ideal"]
    initial_params[hw_symbols.C_fringe] = rcs["Cacti"]["C_fringe"]
    initial_params[hw_symbols.C_junc] = rcs["Cacti"]["C_junc"]
    initial_params[hw_symbols.C_junc_sw] = rcs["Cacti"]["C_junc_sw"]
    initial_params[hw_symbols.l_phy] = rcs["Cacti"]["l_phy"]
    initial_params[hw_symbols.l_elec] = rcs["Cacti"]["l_elec"]
    initial_params[hw_symbols.nmos_effective_resistance_multiplier] = rcs["Cacti"]["nmos_effective_resistance_multiplier"]
    initial_params[hw_symbols.Vdd] = rcs["Cacti"]["Vdd"]
    initial_params[hw_symbols.Vth] = rcs["Cacti"]["Vth"]
    initial_params[hw_symbols.Vdsat] = rcs["Cacti"]["Vdsat"]
    initial_params[hw_symbols.I_on_n] = rcs["Cacti"]["I_on_n"]
    initial_params[hw_symbols.I_on_p] = rcs["Cacti"]["I_on_p"]
    initial_params[hw_symbols.I_off_n] = rcs["Cacti"]["I_off_n"]
    initial_params[hw_symbols.I_g_on_n] = rcs["Cacti"]["I_g_on_n"]
    initial_params[hw_symbols.C_ox] = rcs["Cacti"]["C_ox"]
    initial_params[hw_symbols.t_ox] = rcs["Cacti"]["t_ox"]
    initial_params[hw_symbols.n2p_drv_rt] = rcs["Cacti"]["n2p_drv_rt"]
    initial_params[hw_symbols.lch_lk_rdc] = rcs["Cacti"]["lch_lk_rdc"]
    initial_params[hw_symbols.Mobility_n] = rcs["Cacti"]["Mobility_n"]
    initial_params[hw_symbols.gmp_to_gmn_multiplier] = rcs["Cacti"]["gmp_to_gmn_multiplier"]
    initial_params[hw_symbols.vpp] = rcs["Cacti"]["vpp"]
    initial_params[hw_symbols.Wmemcella] = rcs["Cacti"]["Wmemcella"]
    initial_params[hw_symbols.Wmemcellpmos] = rcs["Cacti"]["Wmemcellpmos"]
    initial_params[hw_symbols.Wmemcellnmos] = rcs["Cacti"]["Wmemcellnmos"]
    initial_params[hw_symbols.area_cell] = rcs["Cacti"]["area_cell"]
    initial_params[hw_symbols.asp_ratio_cell] = rcs["Cacti"]["asp_ratio_cell"]
    # initial_params[hw_symbols.vdd_cell] = rcs["Cacti"]["vdd_cell"]    # TODO check use of vdd_cell
    initial_params[hw_symbols.dram_cell_I_on] = rcs["Cacti"]["dram_cell_I_on"]
    initial_params[hw_symbols.dram_cell_Vdd] = rcs["Cacti"]["dram_cell_Vdd"]
    initial_params[hw_symbols.dram_cell_C] = rcs["Cacti"]["dram_cell_C"]
    initial_params[hw_symbols.dram_cell_I_off_worst_case_len_temp] = rcs["Cacti"]["dram_cell_I_off_worst_case_len_temp"]
    initial_params[hw_symbols.logic_scaling_co_eff] = rcs["Cacti"]["logic_scaling_co_eff"]
    initial_params[hw_symbols.core_tx_density] = rcs["Cacti"]["core_tx_density"]
    initial_params[hw_symbols.sckt_co_eff] = rcs["Cacti"]["sckt_co_eff"]
    initial_params[hw_symbols.chip_layout_overhead] = rcs["Cacti"]["chip_layout_overhead"]
    initial_params[hw_symbols.macro_layout_overhead] = rcs["Cacti"]["macro_layout_overhead"]
    initial_params[hw_symbols.sense_delay] = rcs["Cacti"]["sense_delay"]
    initial_params[hw_symbols.sense_dy_power] = rcs["Cacti"]["sense_dy_power"]
    initial_params[hw_symbols.wire_pitch] = rcs["Cacti"]["wire_pitch"]
    initial_params[hw_symbols.barrier_thickness] = rcs["Cacti"]["barrier_thickness"]
    initial_params[hw_symbols.dishing_thickness] = rcs["Cacti"]["dishing_thickness"]
    initial_params[hw_symbols.alpha_scatter] = rcs["Cacti"]["alpha_scatter"]
    initial_params[hw_symbols.aspect_ratio] = rcs["Cacti"]["aspect_ratio"]
    initial_params[hw_symbols.miller_value] = rcs["Cacti"]["miller_value"]
    initial_params[hw_symbols.horiz_dielectric_constant] = rcs["Cacti"]["horiz_dielectric_constant"]
    initial_params[hw_symbols.vert_dielectric_constant] = rcs["Cacti"]["vert_dielectric_constant"]
    initial_params[hw_symbols.ild_thickness] = rcs["Cacti"]["ild_thickness"]
    initial_params[hw_symbols.fringe_cap] = rcs["Cacti"]["fringe_cap"]
    initial_params[hw_symbols.resistivity] = rcs["Cacti"]["resistivity"]
    initial_params[hw_symbols.wire_r_per_micron] = rcs["Cacti"]["wire_r_per_micron"]
    initial_params[hw_symbols.wire_c_per_micron] = rcs["Cacti"]["wire_c_per_micron"]
    initial_params[hw_symbols.tsv_pitch] = rcs["Cacti"]["tsv_pitch"]
    initial_params[hw_symbols.tsv_diameter] = rcs["Cacti"]["tsv_diameter"]
    initial_params[hw_symbols.tsv_length] = rcs["Cacti"]["tsv_length"]
    initial_params[hw_symbols.tsv_dielec_thickness] = rcs["Cacti"]["tsv_dielec_thickness"]
    initial_params[hw_symbols.tsv_contact_resistance] = rcs["Cacti"]["tsv_contact_resistance"]
    initial_params[hw_symbols.tsv_depletion_width] = rcs["Cacti"]["tsv_depletion_width"]
    initial_params[hw_symbols.tsv_liner_dielectric_cons] = rcs["Cacti"]["tsv_liner_dielectric_cons"]

    # CACTI IO
    initial_params[hw_symbols.vdd_io] = rcs["Cacti_IO"]["vdd_io"]
    initial_params[hw_symbols.v_sw_clk] = rcs["Cacti_IO"]["v_sw_clk"]
    initial_params[hw_symbols.c_int] = rcs["Cacti_IO"]["c_int"]
    initial_params[hw_symbols.c_tx] = rcs["Cacti_IO"]["c_tx"]
    initial_params[hw_symbols.c_data] = rcs["Cacti_IO"]["c_data"]
    initial_params[hw_symbols.c_addr] = rcs["Cacti_IO"]["c_addr"]
    initial_params[hw_symbols.i_bias] = rcs["Cacti_IO"]["i_bias"]
    initial_params[hw_symbols.i_leak] = rcs["Cacti_IO"]["i_leak"]
    initial_params[hw_symbols.ioarea_c] = rcs["Cacti_IO"]["ioarea_c"]
    initial_params[hw_symbols.ioarea_k0] = rcs["Cacti_IO"]["ioarea_k0"]
    initial_params[hw_symbols.ioarea_k1] = rcs["Cacti_IO"]["ioarea_k1"]
    initial_params[hw_symbols.ioarea_k2] = rcs["Cacti_IO"]["ioarea_k2"]
    initial_params[hw_symbols.ioarea_k3] = rcs["Cacti_IO"]["ioarea_k3"]
    initial_params[hw_symbols.t_ds] = rcs["Cacti_IO"]["t_ds"]
    initial_params[hw_symbols.t_is] = rcs["Cacti_IO"]["t_is"]
    initial_params[hw_symbols.t_dh] = rcs["Cacti_IO"]["t_dh"]
    initial_params[hw_symbols.t_ih] = rcs["Cacti_IO"]["t_ih"]
    initial_params[hw_symbols.t_dcd_soc] = rcs["Cacti_IO"]["t_dcd_soc"]
    initial_params[hw_symbols.t_dcd_dram] = rcs["Cacti_IO"]["t_dcd_dram"]
    initial_params[hw_symbols.t_error_soc] = rcs["Cacti_IO"]["t_error_soc"]
    initial_params[hw_symbols.t_skew_setup] = rcs["Cacti_IO"]["t_skew_setup"]
    initial_params[hw_symbols.t_skew_hold] = rcs["Cacti_IO"]["t_skew_hold"]
    initial_params[hw_symbols.t_dqsq] = rcs["Cacti_IO"]["t_dqsq"]
    initial_params[hw_symbols.t_soc_setup] = rcs["Cacti_IO"]["t_soc_setup"]
    initial_params[hw_symbols.t_soc_hold] = rcs["Cacti_IO"]["t_soc_hold"]
    initial_params[hw_symbols.t_jitter_setup] = rcs["Cacti_IO"]["t_jitter_setup"]
    initial_params[hw_symbols.t_jitter_hold] = rcs["Cacti_IO"]["t_jitter_hold"]
    initial_params[hw_symbols.t_jitter_addr_setup] = rcs["Cacti_IO"]["t_jitter_addr_setup"]
    initial_params[hw_symbols.t_jitter_addr_hold] = rcs["Cacti_IO"]["t_jitter_addr_hold"]
    initial_params[hw_symbols.t_cor_margin] = rcs["Cacti_IO"]["t_cor_margin"]
    initial_params[hw_symbols.r_diff_term] = rcs["Cacti_IO"]["r_diff_term"]
    initial_params[hw_symbols.rtt1_dq_read] = rcs["Cacti_IO"]["rtt1_dq_read"]
    initial_params[hw_symbols.rtt2_dq_read] = rcs["Cacti_IO"]["rtt2_dq_read"]
    initial_params[hw_symbols.rtt1_dq_write] = rcs["Cacti_IO"]["rtt1_dq_write"]
    initial_params[hw_symbols.rtt2_dq_write] = rcs["Cacti_IO"]["rtt2_dq_write"]
    initial_params[hw_symbols.rtt_ca] = rcs["Cacti_IO"]["rtt_ca"]
    initial_params[hw_symbols.rs1_dq] = rcs["Cacti_IO"]["rs1_dq"]
    initial_params[hw_symbols.rs2_dq] = rcs["Cacti_IO"]["rs2_dq"]
    initial_params[hw_symbols.r_stub_ca] = rcs["Cacti_IO"]["r_stub_ca"]
    initial_params[hw_symbols.r_on] = rcs["Cacti_IO"]["r_on"]
    initial_params[hw_symbols.r_on_ca] = rcs["Cacti_IO"]["r_on_ca"]
    initial_params[hw_symbols.z0] = rcs["Cacti_IO"]["z0"]
    initial_params[hw_symbols.t_flight] = rcs["Cacti_IO"]["t_flight"]
    initial_params[hw_symbols.t_flight_ca] = rcs["Cacti_IO"]["t_flight_ca"]
    initial_params[hw_symbols.k_noise_write] = rcs["Cacti_IO"]["k_noise_write"]
    initial_params[hw_symbols.k_noise_read] = rcs["Cacti_IO"]["k_noise_read"]
    initial_params[hw_symbols.k_noise_addr] = rcs["Cacti_IO"]["k_noise_addr"]
    initial_params[hw_symbols.v_noise_independent_write] = rcs["Cacti_IO"]["v_noise_independent_write"]
    initial_params[hw_symbols.v_noise_independent_read] = rcs["Cacti_IO"]["v_noise_independent_read"]
    initial_params[hw_symbols.v_noise_independent_addr] = rcs["Cacti_IO"]["v_noise_independent_addr"]
    initial_params[hw_symbols.phy_datapath_s] = rcs["Cacti_IO"]["phy_datapath_s"]
    initial_params[hw_symbols.phy_phase_rotator_s] = rcs["Cacti_IO"]["phy_phase_rotator_s"]
    initial_params[hw_symbols.phy_clock_tree_s] = rcs["Cacti_IO"]["phy_clock_tree_s"]
    initial_params[hw_symbols.phy_rx_s] = rcs["Cacti_IO"]["phy_rx_s"]
    initial_params[hw_symbols.phy_dcc_s] = rcs["Cacti_IO"]["phy_dcc_s"]
    initial_params[hw_symbols.phy_deskew_s] = rcs["Cacti_IO"]["phy_deskew_s"]
    initial_params[hw_symbols.phy_leveling_s] = rcs["Cacti_IO"]["phy_leveling_s"]
    initial_params[hw_symbols.phy_pll_s] = rcs["Cacti_IO"]["phy_pll_s"]
    initial_params[hw_symbols.phy_datapath_d] = rcs["Cacti_IO"]["phy_datapath_d"]
    initial_params[hw_symbols.phy_phase_rotator_d] = rcs["Cacti_IO"]["phy_phase_rotator_d"]
    initial_params[hw_symbols.phy_clock_tree_d] = rcs["Cacti_IO"]["phy_clock_tree_d"]
    initial_params[hw_symbols.phy_rx_d] = rcs["Cacti_IO"]["phy_rx_d"]
    initial_params[hw_symbols.phy_dcc_d] = rcs["Cacti_IO"]["phy_dcc_d"]
    initial_params[hw_symbols.phy_deskew_d] = rcs["Cacti_IO"]["phy_deskew_d"]
    initial_params[hw_symbols.phy_leveling_d] = rcs["Cacti_IO"]["phy_leveling_d"]
    initial_params[hw_symbols.phy_pll_d] = rcs["Cacti_IO"]["phy_pll_d"]
    initial_params[hw_symbols.phy_pll_wtime] = rcs["Cacti_IO"]["phy_pll_wtime"]
    initial_params[hw_symbols.phy_phase_rotator_wtime] = rcs["Cacti_IO"]["phy_phase_rotator_wtime"]
    initial_params[hw_symbols.phy_rx_wtime] = rcs["Cacti_IO"]["phy_rx_wtime"]
    initial_params[hw_symbols.phy_bandgap_wtime] = rcs["Cacti_IO"]["phy_bandgap_wtime"]
    initial_params[hw_symbols.phy_deskew_wtime] = rcs["Cacti_IO"]["phy_deskew_wtime"]
    initial_params[hw_symbols.phy_vrefgen_wtime] = rcs["Cacti_IO"]["phy_vrefgen_wtime"]

    return initial_params


def generate_logic_init_params_from_rcs_as_symbols(rcs):
    """
    Same as generate_init_params_from_rcs_as_symbols,
    but only for logic parameters
    """
    logic_params = {}
    for elem in rcs["Reff"]:
        logic_params[hw_symbols.symbol_table["Reff_" + elem]] = rcs["Reff"][elem]
        logic_params[hw_symbols.symbol_table["Ceff_" + elem]] = rcs["Ceff"][elem]
    logic_params[hw_symbols.V_dd] = rcs["other"]["V_dd"]

    return logic_params

def generate_cacti_init_params_from_rcs_as_symbols(rcs):
    """
    Same as generate_init_params_from_rcs_as_symbols,
    but only for cacti parameters
    """
    cacti_params = {}
    cacti_params[hw_symbols.V_dd] = rcs["other"]["V_dd"]
    # CACTI
    cacti_params[hw_symbols.C_g_ideal] = rcs["Cacti"]["C_g_ideal"]
    cacti_params[hw_symbols.C_fringe] = rcs["Cacti"]["C_fringe"]
    cacti_params[hw_symbols.C_junc] = rcs["Cacti"]["C_junc"]
    cacti_params[hw_symbols.C_junc_sw] = rcs["Cacti"]["C_junc_sw"]
    cacti_params[hw_symbols.l_phy] = rcs["Cacti"]["l_phy"]
    cacti_params[hw_symbols.l_elec] = rcs["Cacti"]["l_elec"]
    cacti_params[hw_symbols.nmos_effective_resistance_multiplier] = rcs["Cacti"]["nmos_effective_resistance_multiplier"]
    cacti_params[hw_symbols.Vdd] = rcs["Cacti"]["Vdd"]
    cacti_params[hw_symbols.Vth] = rcs["Cacti"]["Vth"]
    cacti_params[hw_symbols.Vdsat] = rcs["Cacti"]["Vdsat"]
    cacti_params[hw_symbols.I_on_n] = rcs["Cacti"]["I_on_n"]
    cacti_params[hw_symbols.I_on_p] = rcs["Cacti"]["I_on_p"]
    cacti_params[hw_symbols.I_off_n] = rcs["Cacti"]["I_off_n"]
    cacti_params[hw_symbols.I_g_on_n] = rcs["Cacti"]["I_g_on_n"]
    cacti_params[hw_symbols.C_ox] = rcs["Cacti"]["C_ox"]
    cacti_params[hw_symbols.t_ox] = rcs["Cacti"]["t_ox"]
    cacti_params[hw_symbols.n2p_drv_rt] = rcs["Cacti"]["n2p_drv_rt"]
    cacti_params[hw_symbols.lch_lk_rdc] = rcs["Cacti"]["lch_lk_rdc"]
    cacti_params[hw_symbols.Mobility_n] = rcs["Cacti"]["Mobility_n"]
    cacti_params[hw_symbols.gmp_to_gmn_multiplier] = rcs["Cacti"]["gmp_to_gmn_multiplier"]
    cacti_params[hw_symbols.vpp] = rcs["Cacti"]["vpp"]
    cacti_params[hw_symbols.Wmemcella] = rcs["Cacti"]["Wmemcella"]
    cacti_params[hw_symbols.Wmemcellpmos] = rcs["Cacti"]["Wmemcellpmos"]
    cacti_params[hw_symbols.Wmemcellnmos] = rcs["Cacti"]["Wmemcellnmos"]
    cacti_params[hw_symbols.area_cell] = rcs["Cacti"]["area_cell"]
    cacti_params[hw_symbols.asp_ratio_cell] = rcs["Cacti"]["asp_ratio_cell"]
    # cacti_params[hw_symbols.vdd_cell] = rcs["Cacti"]["vdd_cell"]    # TODO check use of vdd_cell
    cacti_params[hw_symbols.dram_cell_I_on] = rcs["Cacti"]["dram_cell_I_on"]
    cacti_params[hw_symbols.dram_cell_Vdd] = rcs["Cacti"]["dram_cell_Vdd"]
    cacti_params[hw_symbols.dram_cell_C] = rcs["Cacti"]["dram_cell_C"]
    cacti_params[hw_symbols.dram_cell_I_off_worst_case_len_temp] = rcs["Cacti"]["dram_cell_I_off_worst_case_len_temp"]
    cacti_params[hw_symbols.logic_scaling_co_eff] = rcs["Cacti"]["logic_scaling_co_eff"]
    cacti_params[hw_symbols.core_tx_density] = rcs["Cacti"]["core_tx_density"]
    cacti_params[hw_symbols.sckt_co_eff] = rcs["Cacti"]["sckt_co_eff"]
    cacti_params[hw_symbols.chip_layout_overhead] = rcs["Cacti"]["chip_layout_overhead"]
    cacti_params[hw_symbols.macro_layout_overhead] = rcs["Cacti"]["macro_layout_overhead"]
    cacti_params[hw_symbols.sense_delay] = rcs["Cacti"]["sense_delay"]
    cacti_params[hw_symbols.sense_dy_power] = rcs["Cacti"]["sense_dy_power"]
    cacti_params[hw_symbols.wire_pitch] = rcs["Cacti"]["wire_pitch"]
    cacti_params[hw_symbols.barrier_thickness] = rcs["Cacti"]["barrier_thickness"]
    cacti_params[hw_symbols.dishing_thickness] = rcs["Cacti"]["dishing_thickness"]
    cacti_params[hw_symbols.alpha_scatter] = rcs["Cacti"]["alpha_scatter"]
    cacti_params[hw_symbols.aspect_ratio] = rcs["Cacti"]["aspect_ratio"]
    cacti_params[hw_symbols.miller_value] = rcs["Cacti"]["miller_value"]
    cacti_params[hw_symbols.horiz_dielectric_constant] = rcs["Cacti"]["horiz_dielectric_constant"]
    cacti_params[hw_symbols.vert_dielectric_constant] = rcs["Cacti"]["vert_dielectric_constant"]
    cacti_params[hw_symbols.ild_thickness] = rcs["Cacti"]["ild_thickness"]
    cacti_params[hw_symbols.fringe_cap] = rcs["Cacti"]["fringe_cap"]
    cacti_params[hw_symbols.resistivity] = rcs["Cacti"]["resistivity"]
    cacti_params[hw_symbols.wire_r_per_micron] = rcs["Cacti"]["wire_r_per_micron"]
    cacti_params[hw_symbols.wire_c_per_micron] = rcs["Cacti"]["wire_c_per_micron"]
    cacti_params[hw_symbols.tsv_pitch] = rcs["Cacti"]["tsv_pitch"]
    cacti_params[hw_symbols.tsv_diameter] = rcs["Cacti"]["tsv_diameter"]
    cacti_params[hw_symbols.tsv_length] = rcs["Cacti"]["tsv_length"]
    cacti_params[hw_symbols.tsv_dielec_thickness] = rcs["Cacti"]["tsv_dielec_thickness"]
    cacti_params[hw_symbols.tsv_contact_resistance] = rcs["Cacti"]["tsv_contact_resistance"]
    cacti_params[hw_symbols.tsv_depletion_width] = rcs["Cacti"]["tsv_depletion_width"]
    cacti_params[hw_symbols.tsv_liner_dielectric_cons] = rcs["Cacti"]["tsv_liner_dielectric_cons"]

    # CACTI IO
    cacti_params[hw_symbols.vdd_io] = rcs["Cacti_IO"]["vdd_io"]
    cacti_params[hw_symbols.v_sw_clk] = rcs["Cacti_IO"]["v_sw_clk"]
    cacti_params[hw_symbols.c_int] = rcs["Cacti_IO"]["c_int"]
    cacti_params[hw_symbols.c_tx] = rcs["Cacti_IO"]["c_tx"]
    cacti_params[hw_symbols.c_data] = rcs["Cacti_IO"]["c_data"]
    cacti_params[hw_symbols.c_addr] = rcs["Cacti_IO"]["c_addr"]
    cacti_params[hw_symbols.i_bias] = rcs["Cacti_IO"]["i_bias"]
    cacti_params[hw_symbols.i_leak] = rcs["Cacti_IO"]["i_leak"]
    cacti_params[hw_symbols.ioarea_c] = rcs["Cacti_IO"]["ioarea_c"]
    cacti_params[hw_symbols.ioarea_k0] = rcs["Cacti_IO"]["ioarea_k0"]
    cacti_params[hw_symbols.ioarea_k1] = rcs["Cacti_IO"]["ioarea_k1"]
    cacti_params[hw_symbols.ioarea_k2] = rcs["Cacti_IO"]["ioarea_k2"]
    cacti_params[hw_symbols.ioarea_k3] = rcs["Cacti_IO"]["ioarea_k3"]
    cacti_params[hw_symbols.t_ds] = rcs["Cacti_IO"]["t_ds"]
    cacti_params[hw_symbols.t_is] = rcs["Cacti_IO"]["t_is"]
    cacti_params[hw_symbols.t_dh] = rcs["Cacti_IO"]["t_dh"]
    cacti_params[hw_symbols.t_ih] = rcs["Cacti_IO"]["t_ih"]
    cacti_params[hw_symbols.t_dcd_soc] = rcs["Cacti_IO"]["t_dcd_soc"]
    cacti_params[hw_symbols.t_dcd_dram] = rcs["Cacti_IO"]["t_dcd_dram"]
    cacti_params[hw_symbols.t_error_soc] = rcs["Cacti_IO"]["t_error_soc"]
    cacti_params[hw_symbols.t_skew_setup] = rcs["Cacti_IO"]["t_skew_setup"]
    cacti_params[hw_symbols.t_skew_hold] = rcs["Cacti_IO"]["t_skew_hold"]
    cacti_params[hw_symbols.t_dqsq] = rcs["Cacti_IO"]["t_dqsq"]
    cacti_params[hw_symbols.t_soc_setup] = rcs["Cacti_IO"]["t_soc_setup"]
    cacti_params[hw_symbols.t_soc_hold] = rcs["Cacti_IO"]["t_soc_hold"]
    cacti_params[hw_symbols.t_jitter_setup] = rcs["Cacti_IO"]["t_jitter_setup"]
    cacti_params[hw_symbols.t_jitter_hold] = rcs["Cacti_IO"]["t_jitter_hold"]
    cacti_params[hw_symbols.t_jitter_addr_setup] = rcs["Cacti_IO"]["t_jitter_addr_setup"]
    cacti_params[hw_symbols.t_jitter_addr_hold] = rcs["Cacti_IO"]["t_jitter_addr_hold"]
    cacti_params[hw_symbols.t_cor_margin] = rcs["Cacti_IO"]["t_cor_margin"]
    cacti_params[hw_symbols.r_diff_term] = rcs["Cacti_IO"]["r_diff_term"]
    cacti_params[hw_symbols.rtt1_dq_read] = rcs["Cacti_IO"]["rtt1_dq_read"]
    cacti_params[hw_symbols.rtt2_dq_read] = rcs["Cacti_IO"]["rtt2_dq_read"]
    cacti_params[hw_symbols.rtt1_dq_write] = rcs["Cacti_IO"]["rtt1_dq_write"]
    cacti_params[hw_symbols.rtt2_dq_write] = rcs["Cacti_IO"]["rtt2_dq_write"]
    cacti_params[hw_symbols.rtt_ca] = rcs["Cacti_IO"]["rtt_ca"]
    cacti_params[hw_symbols.rs1_dq] = rcs["Cacti_IO"]["rs1_dq"]
    cacti_params[hw_symbols.rs2_dq] = rcs["Cacti_IO"]["rs2_dq"]
    cacti_params[hw_symbols.r_stub_ca] = rcs["Cacti_IO"]["r_stub_ca"]
    cacti_params[hw_symbols.r_on] = rcs["Cacti_IO"]["r_on"]
    cacti_params[hw_symbols.r_on_ca] = rcs["Cacti_IO"]["r_on_ca"]
    cacti_params[hw_symbols.z0] = rcs["Cacti_IO"]["z0"]
    cacti_params[hw_symbols.t_flight] = rcs["Cacti_IO"]["t_flight"]
    cacti_params[hw_symbols.t_flight_ca] = rcs["Cacti_IO"]["t_flight_ca"]
    cacti_params[hw_symbols.k_noise_write] = rcs["Cacti_IO"]["k_noise_write"]
    cacti_params[hw_symbols.k_noise_read] = rcs["Cacti_IO"]["k_noise_read"]
    cacti_params[hw_symbols.k_noise_addr] = rcs["Cacti_IO"]["k_noise_addr"]
    cacti_params[hw_symbols.v_noise_independent_write] = rcs["Cacti_IO"]["v_noise_independent_write"]
    cacti_params[hw_symbols.v_noise_independent_read] = rcs["Cacti_IO"]["v_noise_independent_read"]
    cacti_params[hw_symbols.v_noise_independent_addr] = rcs["Cacti_IO"]["v_noise_independent_addr"]
    cacti_params[hw_symbols.phy_datapath_s] = rcs["Cacti_IO"]["phy_datapath_s"]
    cacti_params[hw_symbols.phy_phase_rotator_s] = rcs["Cacti_IO"]["phy_phase_rotator_s"]
    cacti_params[hw_symbols.phy_clock_tree_s] = rcs["Cacti_IO"]["phy_clock_tree_s"]
    cacti_params[hw_symbols.phy_rx_s] = rcs["Cacti_IO"]["phy_rx_s"]
    cacti_params[hw_symbols.phy_dcc_s] = rcs["Cacti_IO"]["phy_dcc_s"]
    cacti_params[hw_symbols.phy_deskew_s] = rcs["Cacti_IO"]["phy_deskew_s"]
    cacti_params[hw_symbols.phy_leveling_s] = rcs["Cacti_IO"]["phy_leveling_s"]
    cacti_params[hw_symbols.phy_pll_s] = rcs["Cacti_IO"]["phy_pll_s"]
    cacti_params[hw_symbols.phy_datapath_d] = rcs["Cacti_IO"]["phy_datapath_d"]
    cacti_params[hw_symbols.phy_phase_rotator_d] = rcs["Cacti_IO"]["phy_phase_rotator_d"]
    cacti_params[hw_symbols.phy_clock_tree_d] = rcs["Cacti_IO"]["phy_clock_tree_d"]
    cacti_params[hw_symbols.phy_rx_d] = rcs["Cacti_IO"]["phy_rx_d"]
    cacti_params[hw_symbols.phy_dcc_d] = rcs["Cacti_IO"]["phy_dcc_d"]
    cacti_params[hw_symbols.phy_deskew_d] = rcs["Cacti_IO"]["phy_deskew_d"]
    cacti_params[hw_symbols.phy_leveling_d] = rcs["Cacti_IO"]["phy_leveling_d"]
    cacti_params[hw_symbols.phy_pll_d] = rcs["Cacti_IO"]["phy_pll_d"]
    cacti_params[hw_symbols.phy_pll_wtime] = rcs["Cacti_IO"]["phy_pll_wtime"]
    cacti_params[hw_symbols.phy_phase_rotator_wtime] = rcs["Cacti_IO"]["phy_phase_rotator_wtime"]
    cacti_params[hw_symbols.phy_rx_wtime] = rcs["Cacti_IO"]["phy_rx_wtime"]
    cacti_params[hw_symbols.phy_bandgap_wtime] = rcs["Cacti_IO"]["phy_bandgap_wtime"]
    cacti_params[hw_symbols.phy_deskew_wtime] = rcs["Cacti_IO"]["phy_deskew_wtime"]
    cacti_params[hw_symbols.phy_vrefgen_wtime] = rcs["Cacti_IO"]["phy_vrefgen_wtime"]

    return cacti_params