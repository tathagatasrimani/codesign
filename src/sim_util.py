import math
import ast

import dfg_algo


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


def find_nearest_mem_to_scale(num):
    """
    Finds the nearest memory size to scale based on the input.

    Parameters:
        num (int): The memory size to be scaled.

    Returns:
        int: The nearest appropriate memory size.
    """
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
