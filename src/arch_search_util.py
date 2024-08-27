import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from staticfg import CFG, Block

from . import hardwareModel
from .hardwareModel import HardwareModel
from . import sim_util


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
        ]
    )

    hw.netlist.add_edges_from(
        [
            ("Buf0", "MainMem0"),
            ("MainMem0", "Buf0"),
            ("Buf0", "Regs0"),
            ("Regs0", "Buf0"),
            ("Buf0", "Regs1"),
            ("Regs1", "Buf0"),
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
    DEPRECATED
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

def pareto_pruning(hw: HardwareModel, cfg_node_to_hw_map, data_path, id_to_node):
    """ """
    pass
