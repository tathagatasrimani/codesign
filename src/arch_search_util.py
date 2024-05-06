import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import hardwareModel
from hardwareModel import HardwareModel
from staticfg import CFG, Block
import sim_util


def generate_new_min_arch_on_whole_dfg(hw: HardwareModel, computation_dfg: nx.DiGraph):
    """
    Parameters:
        hw (HardwareModel): The hardware model to use for simulation.
        computation_dfg (nx.DiGraph): The data flow graph of the program.
    """
    func_counts = hardwareModel.get_func_count(computation_dfg)
    for key in func_counts.keys():
        if key in ["end", "start"]:
            continue
        name = f"{key}0"
        if key in ["Regs", "MainMem", "Buf"]:
            continue
        else:
            hw.netlist.add_node(name, function=key, type="pe", in_use=False, idx=0)
    for node in hw.netlist.nodes():
        for node_2 in hw.netlist.nodes():
            if node != node_2:
                hw.netlist.add_edge(node, node_2)

    hw.netlist.add_nodes_from(
        [
            (
                "Regs0",
                {
                    "function": "Regs",
                    "size": 1,
                    "type": "memory",
                    "in_use": False,
                    "idx": 0,
                },
            ),
            (
                "MainMem0",
                {
                    "function": "MainMem",
                    "size": 1,
                    "type": "memory",
                    "in_use": False,
                    "idx": 0,
                },
            ),
            (
                "Buf0",
                {
                    "function": "Buf",
                    "size": 1,
                    "type": "memory",
                    "in_use": False,
                    "idx": 0,
                },
            ),
            (
                "Regs1",
                {
                    "function": "Regs",
                    "size": 1,
                    "type": "memory",
                    "in_use": False,
                    "idx": 1,
                },
            ),
            (
                "MainMem1",
                {
                    "function": "MainMem",
                    "size": 1,
                    "type": "memory",
                    "in_use": False,
                    "idx": 1,
                },
            ),
            (
                "Buf1",
                {
                    "function": "Buf",
                    "size": 1,
                    "type": "memory",
                    "in_use": False,
                    "idx": 1,
                },
            ),
        ]
    )

    hw.netlist.add_edges_from(
        [
            ("Buf0", "MainMem0"),
            ("MainMem0", "Buf0"),
            ("Buf1", "MainMem1"),
            ("MainMem1", "Buf1"),
            ("Buf0", "Regs0"),
            ("Regs0", "Buf0"),
            ("Buf1", "Regs1"),
            ("Regs1", "Buf1"),
        ]
    )

    for node in hw.netlist.nodes():
        if hw.netlist.nodes[node]["function"] in ["Regs", "MainMem", "Buf"]:
            continue
        hw.netlist.add_edges_from(
            [("Regs0", node), ("Regs1", node), (node, "Regs0"), (node, "Regs1")]
        )


def generate_new_min_arch(hw: HardwareModel, cfg_node_to_hw_map, data_path, id_to_node):
    """
    Dynamically generate the hardware for a given DFG.

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
    # print(f"Architecture Search Running...")

    for i in range(len(data_path)):
        # init vars for new node in cfg data path
        node_id = data_path[i][0]
        if node_id not in id_to_node.keys():
            continue
        cur_node = id_to_node[node_id]

        hw_graph = cfg_node_to_hw_map[cur_node]

        # nx.draw(hw_graph, with_labels=True, font_weight="bold")

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
    # nx.draw(hw.netlist, with_labels=True)
    # plt.show()


def generate_unrolled_arch(
    hw: HardwareModel,
    cfg_node_to_hw_map,
    data_path,
    id_to_node,
    area_constraint,
    bw_constraint,
    memory,
):
    """

    Generate the maximally unrolled architecture that meets the area and bandwidth constraints.

    params:
    hw: HardwareModel
    cfg_node_to_hw_map: dict
    data_path: list
    id_to_node: dict
    area_constraint: float
    bw_constraint: float
    memory: int - Total memory needed by the program.
                only needed to scale up the memory area so that the area constraint can be effectively assessed
    """
    print(f"Generating Unrolled Architecture...")
    # count statistics of data path
    unique, count = np.unique([str(elem) for elem in data_path], return_counts=True)

    sorted_nodes = [
        x for _, x in sorted(zip(count, unique), key=lambda pair: pair[0], reverse=True)
    ]

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

    unroll_factor = max_continuous
    while True:
        new_data_path = unroll_by_specified_factor(
            cfg_node_to_hw_map, data_path, id_to_node, unroll_factor, saved_elem
        )
        unique_data_path = [list(x) for x in set(tuple(x) for x in new_data_path)]
        generate_new_min_arch(hw, cfg_node_to_hw_map, unique_data_path, id_to_node)
        hw.init_memory(memory, 0)

        # if area exceeds threshold, decrease unroll factor by 2x and try again
        if area_constraint is None:
            pass_area = True

        if bw_constraint is None:
            pass_bw = True

        if area_constraint is not None:
            area = hw.get_total_area()
            area_ratio = area / area_constraint
            print(
                f"Area: {area}, Area Constraint: {area_constraint}, Area Ratio: {area_ratio}"
            )
            if area_ratio <= 1:
                print(f"Area Constraint Met: {area_ratio}")
                pass_area = True
            else:
                pass_area = False
                unroll_factor = int(unroll_factor / area_ratio)
                hw.netlist = nx.DiGraph()

        if bw_constraint is not None:
            curr_bw = hw.get_mem_compute_bw()
            bw_ratio = curr_bw / bw_constraint
            print(
                f"Bandwidth: {curr_bw}, Bandwidth Constraint: {bw_constraint}, Bandwidth Ratio: {bw_ratio}"
            )
            if bw_ratio <= 1:
                print(f"Bandwidth Constraint Met: {bw_ratio}")
                pass_bw = True
            else:
                pass_bw = False
                unroll_factor = min(unroll_factor, int(unroll_factor / bw_ratio))
                hw.netlist = nx.DiGraph()

        if pass_area and pass_bw:
            break

        # if area is less than threshold, Don't do anything for now.

    return new_data_path


def unroll_by_specified_factor(
    cfg_node_to_hw_map: dict,
    data_path: list,
    id_to_node: dict,
    unroll_factor: int,
    specified_node,
    log=True,
):
    """
    Unroll the data path by a specified factor.
    Create new merged nx.Digraphs and add to cfg_node_to_hw_map.
    """
    if log:
        print(f"Unrolling by {unroll_factor}X...")
        print(f"Specified Node: {specified_node}")
    # add entry to cfg_node_to_hw_map with unrolled dfg with
    # unroll factor equal to max_continuous
    single_node_comp_graph = cfg_node_to_hw_map[id_to_node[specified_node[0]]].copy()
    copy = single_node_comp_graph.copy()

    for i in range(unroll_factor - 1):
        sim_util.rename_nodes(single_node_comp_graph, copy)
        single_node_comp_graph = nx.union(single_node_comp_graph, copy)

    blk = Block(int(specified_node[0]) * unroll_factor)
    if blk.id not in id_to_node.keys():
        cfg_node_to_hw_map[blk] = single_node_comp_graph
        id_to_node[blk.id] = blk
    # else:

    # iterate through data path and replace nodes with unrolled nodes
    new_data_path = data_path.copy()
    count = 0
    i = 0
    while i < len(new_data_path):
        elem = new_data_path[i]
        if elem == specified_node:
            count += 1
            if count == unroll_factor:
                while count > 0:
                    new_data_path.pop(i - unroll_factor + 1)
                    count -= 1
                i = i - unroll_factor + 1
                new_data_path.insert(i, [blk.id, 0])
        elif elem != specified_node:
            count = 0
        i += 1

    return new_data_path


def pareto_pruning(hw: HardwareModel, cfg_node_to_hw_map, data_path, id_to_node):
    """ """
    pass
