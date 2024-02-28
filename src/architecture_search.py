# builtin
import argparse
from copy import deepcopy

# third party
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# custom
from simulate import ConcreteSimulator
import hardwareModel
import arch_search_util 
import sim_util

rng = np.random.default_rng()

def simulate_new_arch(hw, simulator, cfg, cfg_node_to_hw_map):
    hardwareModel.un_allocate_all_in_use_elements(hw.netlist)
    # print(f"keys in cfg_node_to_hw_map:")
    # [print(str(key) + '\n\n') for key in cfg_node_to_hw_map.keys()]
    simulator.simulate(cfg, cfg_node_to_hw_map, hw)
    avg_compute_power = 1e-6 * (
        np.mean(list(simulator.active_power_use.values()))
        + simulator.passive_power_dissipation_rate
    )
    return simulator.cycles**2 * avg_compute_power


def gen_new_arch(existing_arch):
    # print(f"node data: {existing_arch.netlist.nodes.data()}")
    removable_nodes = list(filter(lambda x: x[1]["function"] != "MainMem" and x[1]["function"] != "Buf", existing_arch.netlist.nodes.data()))
    weights = list(map(lambda x: x[1]["allocation"], removable_nodes))
    weights = weights / np.sum(weights)
    # print(f"weights: {weights}")
    node = rng.choice(removable_nodes, p=weights)
    print(f"Removing node: {node}")
    existing_arch.netlist.remove_node(node[0])
    return existing_arch


def main(args):
    """
    Currently the data path is just going to get modified once
    when I do the unrolling;
    so I can do all the setup once and then just simulate
    a bunch of diff architectures.
    """

    print(f"Running Architecture Search for {args.benchmark.split('/')[-1]}")
    simulator = ConcreteSimulator()

    # TODO: move this into cli arg
    hw = hardwareModel.HardwareModel(cfg="aladdin_const_with_mem")

    cfg, cfg_node_to_hw_map = simulator.simulator_prep(args.benchmark, hw.latency)
    print(f"type of cfg: {type(cfg)}")

    hw.netlist = nx.DiGraph()
    simulator.data_path = arch_search_util.generate_unrolled_arch(
        hw,
        cfg_node_to_hw_map,
        simulator.data_path,
        simulator.id_to_node,
        args.area,
        args.bw,
        sim_util.find_nearest_power_2(simulator.memory_needed),
    )

    for elem in simulator.data_path:
        if elem[0] not in simulator.unroll_at.keys():
            simulator.unroll_at[elem[0]] = False

    hw.init_memory(
        sim_util.find_nearest_power_2(simulator.memory_needed),
        sim_util.find_nearest_power_2(simulator.nvm_memory_needed),
    )

    # this has data path and all setup done
    # without having the visited set object
    # and anhistory from the previous run
    sim_copy = deepcopy(simulator)

    hardwareModel.un_allocate_all_in_use_elements(hw.netlist)
    # orig = deepcopy(cfg_node_to_hw_map)
    # orig_data_path = deepcopy(simulator.data_path)
    # print(f"entry block before: {cfg.entryblock}")
    # print(f"sim id_to_node before: {simulator.id_to_node.keys()}")
    # print(f"keys in cfg_node_to_hw_map:")
    # [print(str(key) + "\n") for key in cfg_node_to_hw_map.keys()]
    print(f"values in cfg_node_to_hw_map:")
    [print(str(val)) for val in cfg_node_to_hw_map.values()]

    cfg_node_to_hw_map_copy = {}
    for key, val in cfg_node_to_hw_map.items():
        cfg_node_to_hw_map_copy[key] = val.copy()

    data = simulator.simulate(cfg, cfg_node_to_hw_map, hw)
    # print(f"entry block after: {cfg.entryblock}")
    # print(f"sim id_to_node after: {simulator.id_to_node.keys()}")

    # print(f"data_path diff befre after: {len(set([elem[0] for elem in orig_data_path]).symmetric_difference(set([elem[0] for elem in simulator.data_path])))}")
    # print(f"keys in cfg_node_to_hw_map after:")
    # [print(str(key) + "\n") for key in cfg_node_to_hw_map.keys()]
    print(f"values in cfg_node_to_hw_map after:")
    [print(str(val)) for val in cfg_node_to_hw_map.values()]

    # set1 = set(orig.keys())
    # set2 = set(cfg_node_to_hw_map.keys())
    # print(f"len of orig: {len(orig)}")
    # print(f"diff cfg_node_to_hw_map: {len(set1.symmetric_difference(set2))}")
    # print (f"cfgnode to hw map after: {cfg_node_to_hw_map}")

    best_edp = np.inf
    best_hw = None
    for i in range(10):
        print(f"Simulating new architecture {i}...")
        hw_copy_ = deepcopy(hw)
        gen_new_arch(hw_copy_)
        new_hw = hw_copy_
        EDP = simulate_new_arch(new_hw, simulator, cfg, cfg_node_to_hw_map_copy)
        if EDP < best_edp:
            best_edp = EDP
            best_hw = new_hw

    if args.filepath:
        nx.write_gml(best_hw.netlist,f"architecures/{args.filepath}.gml", stringizer=lambda x: str(x))

    print(f"Best EDP: {best_edp}")
    nx.draw(best_hw.netlist, with_labels=True, )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Architecture Search",
        description="Runs a hardware simulation on a given benchmark and technology spec",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("benchmark", metavar="B", type=str)
    parser.add_argument("--notrace", action="store_true")
    parser.add_argument("-a", "--area", type=float, help="Max Area of the chip in um^2")
    parser.add_argument(
        "-b", "--bw", type=float, help="Compute - Memory Bandwidth in ??GB/s??"
    )
    parser.add_argument(
        "-f", "--filepath", type=str, help="Path to the save new architecture file"
    )
    args = parser.parse_args()
    print(
        f"args: benchmark: {args.benchmark}, trace:{args.notrace}, area:{args.area}, bw:{args.bw}, file: {args.filepath}"
    )

    main(args)
