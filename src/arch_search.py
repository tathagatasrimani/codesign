import numpy as np

import hardwareModel
from hardwareModel import HardwareModel
from staticfg import CFG
import sim_util


def generate_new_min_arch(
    cfg: CFG, hw: HardwareModel, cfg_node_to_hw_map, data_path, id_to_node
):
    """
    Dynamically generate the asap hardware for a given DFG.

    Literally counts number of PEs and matches them. Doesn't consider topo monomorphicity.
    Need to do inverse of what is happening in verify_can_execute.
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

        hw.netlist = sim_util.verify_can_execute(hw_graph, hw.netlist, should_update_arch=True)

        i = next_ind

