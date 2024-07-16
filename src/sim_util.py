import math
import ast
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import dfg_algo
import hw_symbols
from global_constants import SEED

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


def update_schedule_with_latency(schedule, latency):
    """
    Updates the schedule with the latency of each operation.

    Parameters:
        schedule (nx.Digraph): A list of operations in the schedule.
        latency (dict): A dictionary of operation names to their latencies.

    Returns:
        None;
        The schedule is updated in place.
    """
    for edge in schedule.edges:
        node = edge[0]
        func = schedule.nodes.data()[node]["function"]
        scaling = 1
        if func == "stall":
            func = node.split("_")[3]
        if func == "Buf" or func == "MainMem":
            scaling = schedule.nodes.data()[node]["size"]
        schedule.edges[edge]["weight"] = latency[func] * scaling # multiply by 16 for buf and main mem


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
    computation_graph: nx.DiGraph, buf_latency: float, mem_latency: float
):
    """
    Add cache and memory nodes to the computation graph to indicate when explicit memory reads occur.
    Initially, add cache and memory nodes for all Regs nodes in the computation graph.
    After allocation, can prune the cache and memory nodes that are not needed.
    """
    buf_count = 0
    mem_count = 0
    for node, data in dict(
        filter(lambda x: x[1]["function"] == "Regs", computation_graph.nodes.data())
    ).items():
        # print(f"node: {node}, data: {data}")
        size = 16 #data['size'] #-hardcoded for now; TODO: come back and fix
        computation_graph.add_node(
            f"Buf{buf_count}",
            function="Buf",
            allocation="",
            cost=buf_latency * size,
            size=size,
        )
        computation_graph.add_node(
            f"Mem{mem_count}",
            function="MainMem",
            allocation="",
            cost=mem_latency * size,
            size=size,
        )
        # weight of edge is latency of parent
        computation_graph.add_edge(f"Mem{mem_count}", f"Buf{buf_count}", function="Mem", weight=mem_latency*size)
        computation_graph.add_edge(f"Buf{buf_count}", node, function="Mem", weight=buf_latency*size)
        buf_count += 1
        mem_count += 1

    return computation_graph


def find_upstream_node_in_graph(graph:nx.DiGraph, func:str, node:str):
    """
    Find the upstream node in the graph with the specified function. Assumes all nodes have only one in edge.
    And that the desired function exists somewhere upstream.
    Returns the first upstream node with the desired function.
    Parameters:
        graph (nx.DiGraph): The graph to search.
        func (str): The function to search for.
        node (str): The node from which to search.
    """
    func_in = []
    while len(func_in) == 0:
                in_edges = list(map(
                            lambda x: (x[0], graph.nodes[x[0]]),
                            graph.in_edges(node),
                        ))
                if len(in_edges) == 0:
                    raise Exception(f"Could not find upstream node with function {func}")
                node = in_edges[0][0] # only need the name
                func_in = list(
                    filter(
                        lambda x: x[1]["function"] == func,
                        in_edges,
                    )
                ) # assuming only one upstream node
    return func_in[0]


def prune_buffer_and_mem_nodes(computation_graph: nx.DiGraph, hw_netlist: nx.DiGraph):
    """
    Call after allocation to remove unnecessary buffer and memory nodes.
    Removes memory nodes when the data is already in the buffer.
    Removes buffer nodes when the data is already in the registers.
    """
    layer = 0
    gen = list(filter(lambda x: x[1]["layer"] == layer, computation_graph.nodes.data()))
    while len(gen) != 0:
        reg_nodes = list(filter(lambda x: x[1]["function"] == "Regs", gen))
        for reg_node in reg_nodes:
            buf_in = find_upstream_node_in_graph(computation_graph, "Buf", reg_node[0]) 
            mem_in = find_upstream_node_in_graph(computation_graph, "MainMem", buf_in[0])
            allocated_reg = reg_node[1]["allocation"]
            allocated_reg_data = hw_netlist.nodes[allocated_reg]
            var_name = reg_node[0].split(";")[0]
            if allocated_reg_data["var"] == var_name:
                # remove the buffer and memory nodes
                computation_graph.remove_node(buf_in[0])
                computation_graph.remove_node(mem_in[0])
            elif hw_netlist.nodes[buf_in[1]["allocation"]]["memory_module"].find(var_name):
                # remove the memory node
                computation_graph.remove_node(mem_in[0])
                size = hw_netlist.nodes[buf_in[1]["allocation"]]["memory_module"].read(var_name)
                computation_graph.nodes[buf_in[0]]["size"] = size
                computation_graph.nodes[reg_node[0]]["size"] = size
            else:
                # read from memory and add to cache
                size = -1*hw_netlist.nodes[buf_in[1]["allocation"]]["memory_module"].read(var_name)
                computation_graph.nodes[mem_in[0]]["size"] = size
                computation_graph.nodes[buf_in[0]]["size"] = size
                computation_graph.nodes[reg_node[0]]["size"] = size
            hw_netlist.nodes[allocated_reg]["var"] = var_name

        layer += 1
        gen = list(
            filter(lambda x: x[1]["layer"] == layer, computation_graph.nodes.data())
        )

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


def rename_nodes(G, H, H_generations=None, curr_last_nodes=None):
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
    Not currently used, doesn't handle register allocation very well.

    Parameters:
        cfg (CFG): The control flow graph of the program.
        cfg_node_to_hw_map (dict): A mapping of CFG nodes to hardware graphs represented by nx.DiGraphs.

    Returns:
        nx.DiGraph: The large DFG composed from the smaller DFGs.
    """
    computation_dfg = nx.DiGraph()
    curr_last_nodes = []
    i = find_next_data_path_index(data_path, 0, [], [])[0]
    while i < len(data_path):
        node_id = data_path[i][0]
        vars = data_path_vars[i]

        node = id_to_node[node_id]
        dfg = cfg_node_to_dfg_map[node]

        if nx.is_empty(dfg):
            i = find_next_data_path_index(data_path, i + 1, [], [])[0]
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
            computation_dfg, dfg, generations, curr_last_nodes
        )
        computation_dfg = nx.compose(computation_dfg, dfg)
        generations = list(nx.topological_generations(dfg))

        curr_last_nodes = generations[-1]

        i = find_next_data_path_index(data_path, i + 1, [], [])[0]

    # create end node and connect all last nodes to it
    computation_dfg.add_node("end", function="end")
    generations = list(nx.topological_generations(computation_dfg))
    for node in generations[-1]:
        computation_dfg.add_edge(
            node, "end", weight=latency[computation_dfg.nodes[node]["function"]]
        )

    if plot:
        topological_layout_plot(computation_dfg, reverse=True)
    return computation_dfg


def topological_layout_plot(graph, reverse=False):
    graph_copy = graph.copy()
    generations = (
        reversed(list(nx.topological_generations(nx.reverse(graph_copy))))
        if reverse
        else nx.topological_generations(graph_copy)
    )

    for layer, nodes in enumerate(generations):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            graph_copy.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(graph_copy, subset_key="layer")

    fig, ax = plt.subplots(figsize=(9, 9))
    nx.draw_networkx(graph_copy, pos=pos, ax=ax)
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


def convert_tech_params_to_si(latency, active_power, passive_power, frequency):
    """
    Convert the tech params from the input units to the SI units.
    latency in cycles -> seconds
    active_power in nW -> W
    passive_power in nW -> W
    """
    return latency / frequency, active_power / 1e9, passive_power / 1e9


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~ SYMBOLIC UTILS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def generate_init_params_from_rcs_as_strings(rcs):
    """
    Just some format conversion
    keys are strings
    """
    initial_params = {}
    for elem in rcs["Reff"]:
        initial_params["Reff_" + elem] = rcs["Reff"][elem]
        initial_params["Ceff_" + elem] = rcs["Ceff"][elem]
    initial_params["f"] = rcs["other"]["f"]
    initial_params["V_dd"] = rcs["other"]["V_dd"]
    initial_params["MemReadL"] = rcs["other"]["MemReadL"]
    initial_params["MemWriteL"] = rcs["other"]["MemWriteL"]
    initial_params["MemReadPact"] = rcs["other"]["MemReadPact"]
    initial_params["MemWritePact"] = rcs["other"]["MemWritePact"]
    initial_params["MemPpass"] = rcs["other"]["MemPpass"]
    return initial_params


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
    initial_params[hw_symbols.MemReadL] = rcs["other"]["MemReadL"]
    initial_params[hw_symbols.MemWriteL] = rcs["other"]["MemWriteL"]
    initial_params[hw_symbols.MemReadEact] = rcs["other"]["MemReadEact"]
    initial_params[hw_symbols.MemWriteEact] = rcs["other"]["MemWriteEact"]
    initial_params[hw_symbols.MemPpass] = rcs["other"]["MemPpass"]
    initial_params[hw_symbols.BufL] = rcs["other"]["BufL"]
    initial_params[hw_symbols.BufReadEact] = rcs["other"]["BufReadEact"]
    initial_params[hw_symbols.BufWriteEact] = rcs["other"]["BufWriteEact"]
    initial_params[hw_symbols.BufPpass] = rcs["other"]["BufPpass"]
    initial_params[hw_symbols.OffChipIOL] = rcs["other"]["OffChipIOL"]
    initial_params[hw_symbols.OffChipIOPact] = rcs["other"]["OffChipIOPact"]
    return initial_params
