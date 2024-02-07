import numpy as np

import hardwareModel
from hardwareModel import HardwareModel
from staticfg import CFG
import sim_util


def generate_new_min_arch(hw: HardwareModel, cfg_node_to_hw_map, data_path, id_to_node):
    """
    Dynamically generate the asap hardware for a given DFG.

    During the topological traversal of the DFG, we generate new hardware
    This is a tighter bound on required hardware because we don't ensure monomrphism
    with the whole DFG node, but instead we ensure monomorphism with pairwise levels
    of the topological ordering of nodes within a node of the DFG.

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

        hw.netlist = sim_util.verify_can_execute(
            hw_graph, hw.netlist, should_update_arch=True
        )

        hardwareModel.un_allocate_all_in_use_elements(hw.netlist)


        i = next_ind
    hw.netlist.add_node("Buf0", function="Buf", size=1)
    hw.netlist.add_node("Mem0", function="MainMem", size=1)
    hw.netlist.add_edge("Buf0", "Mem0")
    hw.netlist.add_edge("Mem0", "Buf0")
    for node, data in hardwareModel.get_nodes_with_func(hw.netlist, "Regs").items():
        hw.netlist.add_edge("Buf0", node)
        hw.netlist.add_edge(node, "Buf0")
        data["size"] = 1


def generate_unrolled_arch(
    hw: HardwareModel, cfg_node_to_hw_map, data_path, id_to_node
):
    pass
