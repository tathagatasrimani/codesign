import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import hardwareModel
from hardwareModel import HardwareModel
from staticfg import CFG, Block
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
    
    for i in range(len(data_path)):
        # init vars for new node in cfg data path
        node_id = data_path[i][0]
        if node_id not in id_to_node.keys():
            continue
        cur_node = id_to_node[node_id]

        hw_graph = cfg_node_to_hw_map[cur_node]

        hw.netlist = sim_util.verify_can_execute(
            hw_graph, hw.netlist, should_update_arch=True
        )

        hardwareModel.un_allocate_all_in_use_elements(hw.netlist)

    hw.netlist.add_node("Buf0", function="Buf", size=1)
    hw.netlist.add_node("Mem0", function="MainMem", size=1)
    hw.netlist.add_edge("Buf0", "Mem0")
    hw.netlist.add_edge("Mem0", "Buf0")
    for node, data in hardwareModel.get_nodes_with_func(hw.netlist, "Regs").items():
        hw.netlist.add_edge("Buf0", node)
        hw.netlist.add_edge(node, "Buf0")
        data["size"] = 1


def generate_unrolled_arch(
    hw: HardwareModel, cfg_node_to_hw_map, data_path, id_to_node, area_constraint, bw_constraint, memory
):
    """
    
    params:
    hw: HardwareModel
    cfg_node_to_hw_map: dict
    data_path: list
    id_to_node: dict
    area_constraint: int
    bw_constraint: int
    memory: int - only needed to scale up the memory area so that the area constraint can be effectively assessed
    """
    print(f"Generating Unrolled Architecture...")
    # print(f"Data Path: {data_path}")
    # count statistics of data path
    unique, count = np.unique([str(elem) for elem in data_path], return_counts=True)
    # print(f"Unique: {unique}, Count: {count}")
    sorted_nodes = [x for _, x in sorted(zip(count, unique), key=lambda pair: pair[0], reverse=True)]
    sorted_counts = sorted(count, reverse=True)
    # print(f"After sort Unique: {sorted_nodes}, Count: {sorted_counts}")
    # print(f"most common elem: {list(sorted_nodes[0])}")
    # saved_elem = list(sorted_nodes)[0]
    # get number of continuous most common node:
    # there should be a better way to do this
    max_continuous = 1
    prev = 0
    cont = 1
    saved_elem = 0
    for elem in data_path:
        if elem == prev and str(elem) == sorted_nodes[0]:
            saved_elem = elem
            cont += 1
        else:
            if cont > max_continuous:
                max_continuous = cont
            cont = 1
        prev = elem
    print(f"Max Continuous: {max_continuous}")

    unroll_factor = max_continuous

    while True:

        new_data_path = unroll_by_specified_factor(cfg_node_to_hw_map, data_path, id_to_node, unroll_factor, saved_elem)

        print(f"Final Data Path:\n{new_data_path}")
        # make call to gen_min_arch with new cfg_node_to_hw_map
        unique_data_path = [list(x) for x in set(tuple(x) for x in new_data_path)]
        print(f"Unique Data Path: {unique_data_path}")
        generate_new_min_arch(hw, cfg_node_to_hw_map, unique_data_path, id_to_node)
        hw.init_memory(memory, 0)
        # if area exceeds threshold, decrease unroll factor by 2x and try again
        if area_constraint is not None:
            area = hw.get_total_area()
            area_ratio = area / area_constraint
            print(f"Area: {area}, Area Constraint: {area_constraint}, Area Ratio: {area_ratio}")
            if area_ratio <= 1:
                print(f"Area Constraint Met: {area_ratio}")
                break
            else:
                unroll_factor = int(unroll_factor / area_ratio)
                hw.netlist = nx.DiGraph()
                if unroll_factor == 1:
                    break
        elif bw_constraint is not None:
            pass
        else:
            break

        # if area is less than threshold, Don't do anything for now.

    return new_data_path 

def unroll_by_specified_factor(cfg_node_to_hw_map: dict, data_path: list, id_to_node: dict, unroll_factor: int, specified_node):
    """
    
    """
    print(f"Unrolling by {unroll_factor}X...")

    # add entry to cfg_node_to_hw_map with unrolled dfg with
    # unroll factor equal to max_continuous
    single_node_comp_graph = cfg_node_to_hw_map[id_to_node[specified_node[0]]].copy()
    copy = single_node_comp_graph.copy()

    for i in range(unroll_factor - 1):
        sim_util.rename_nodes(single_node_comp_graph, copy)
        single_node_comp_graph = nx.union(single_node_comp_graph, copy)

    # sim_util.topological_layout_plot(single_node_comp_graph)

    # iterate through data path and replace nodes with unrolled nodes
    blk = Block(int(specified_node[0]) * unroll_factor)
    # edit these in place
    cfg_node_to_hw_map[blk] = single_node_comp_graph
    id_to_node[blk.id] = blk

    new_data_path = data_path.copy()
    count = 0
    i = 0
    # print(f"initial length of data path: {len(data_path)}")
    while i < len(new_data_path):
        elem = new_data_path[i]
        # print(f"idx {i}, elem: {elem}, saved_elem: {saved_elem}")
        if elem == specified_node:
            count += 1
            if count == unroll_factor:
                # print(f"found {max_continuous} continuous nodes; startin popping")
                while count > 0:
                    elem_poppped = new_data_path.pop(i - unroll_factor + 1)
                    # print(f"popped elem: {elem_poppped}")
                    count -= 1
                i = i - unroll_factor + 1
                # print(f"new len of data path after poppping:{len(data_path)}\n{data_path}")
                # print(f"i after popping: {i}; inserting new node before {data_path[i]}")
                new_data_path.insert(i, [blk.id, 0])
        elif elem != specified_node:
            count = 0
        i += 1

    return new_data_path
