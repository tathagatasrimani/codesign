import hardwareModel
from hardwareModel import HardwareModel
from staticfg import CFG
import sim_util


def generate_new_aladdin_arch(cfg: CFG, hw: HardwareModel, node_operation_map, data_path, id_to_node):
    """
    Dynamically generate the asap hardware for a given DFG. 
    Just extracting the stuff that used to be in the concrete simulation.
    But, this doesn't support unrolling and pattern matching the way it used to in the sim.
    """

    i, pattern_seek, max_iters = sim_util.find_next_data_path_index(data_path, i=0, mallocs=[], frees=[])
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

        for operations in node_operation_map[cur_node]:
            hw_need = sim_util.get_hw_need_lite(operations, hw)
            for elem in hw_need:
                if hw.dynamic_allocation:
                    if hw.hw_allocated[elem] < hw_need[elem]:
                        # hw_inuse[elem] = [0] * hw_need[elem]
                        # This is where I increase the number of PEs as needed.
                        hw.hw_allocated[elem] = hw_need[elem]
            
        i = next_ind
