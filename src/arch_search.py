import numpy as np

import hardwareModel
from hardwareModel import HardwareModel
from staticfg import CFG
import sim_util


def generate_new_fc_arch(
    cfg: CFG, hw: HardwareModel, cfg_node_to_hw_map, data_path, id_to_node
):
    """
    Dynamically generate the asap hardware for a given DFG.
    Just extracting the stuff that used to be in the concrete simulation.
    But, this doesn't support unrolling and pattern matching the way it used to in the sim.

    Currently allocates excessive PEs because it doesn't consider the topo ordering of 
    the DFG. This is a TODO.

    Parameters:
        cfg (CFG): The control flow graph of the program.
        hw (HardwareModel): The hardware model to use for simulation.
        cfg_node_to_hw_map (dict): A mapping of CFG nodes to hardware graphs represented by nx.DiGraphs.
        data_path (list): A list of lists representing the data path of the simulation.
        id_to_node (dict): A mapping of node ids to CFG nodes.
    """
    print(f"Architecture Search Running...")
    i, pattern_seek, max_iters = sim_util.find_next_data_path_index(
        data_path, i=0, mallocs=[], frees=[]
    )
    # iterate through nodes in data dependency graph
    while i < len(data_path):
        (
            next_ind,
            pattern_seek_next,
            max_iters_next,
        ) = sim_util.find_next_data_path_index(data_path, i + 1, [], [])

        if i == len(data_path):
            break

        # init vars for new node in cfg data path
        node_id = data_path[i][0]
        cur_node = id_to_node[node_id]

        hw_graph = cfg_node_to_hw_map[cur_node]
        nodes = list(map(lambda x: x[1]["function"], list(hw_graph.nodes.data())))
        unique_funcs, counts = np.unique(nodes, return_counts=True)
        computation_graph_func_counts = dict(zip(unique_funcs, counts))

        nodes = list(map(lambda x: x[1]["function"], list(hw.netlist.nodes.data())))
        unique_funcs, counts = np.unique(nodes, return_counts=True)
        netlist_func_counts = dict(zip(unique_funcs, counts))

        # add new nodes to netlist if necessary
        for func, count in computation_graph_func_counts.items():
            if func not in netlist_func_counts.keys():
                netlist_func_counts[func] = 0
            if netlist_func_counts[func] < count:
                for i in range(netlist_func_counts[func], count):
                    hw.netlist.add_node(
                        (func + str(i)),
                        type="pe" if func is not "Regs" else "memory",
                        function=func,
                        in_use=False,
                        idx=i,
                    )

        i = next_ind

    # add edges between all nodes in netlist
    for node in hw.netlist.nodes:
        if "Regs" in node:
            continue
        for node2 in hw.netlist.nodes:
            if node2 == node:
                continue
            hw.netlist.add_edge(node2, node)
            hw.netlist.add_edge(node, node2)
