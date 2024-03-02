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
    simulator.calculate_edp(hw)
    return simulator.edp

def gen_new_arch(existing_arch, unroll_factor, cfg_node_to_hw_map, data_path, id_to_node, mem, saved_elem):
    existing_arch.netlist = nx.DiGraph()
    new_data_path = arch_search_util.unroll_by_specified_factor(
        cfg_node_to_hw_map, data_path, id_to_node, unroll_factor, saved_elem, log=False
    )
    # print(f"old data path: {data_path}")
    # print(f"new data path: {new_data_path}")
    unique_data_path = [list(x) for x in set(tuple(x) for x in new_data_path)]
    # print(f"unqiue data path: {unique_data_path}")
    arch_search_util.generate_new_min_arch(existing_arch, cfg_node_to_hw_map, unique_data_path, id_to_node)
    existing_arch.init_memory(
        sim_util.find_nearest_power_2(mem),
        sim_util.find_nearest_power_2(0),
    )

    return new_data_path

def setup_arch_search(benchmark):
    simulator = ConcreteSimulator()

    # TODO: move this into cli arg
    hw = hardwareModel.HardwareModel(cfg="aladdin_const_with_mem")

    cfg, cfg_node_to_hw_map = simulator.simulator_prep(benchmark, hw.latency)

    hw.netlist = nx.DiGraph()

    arch_search_util.generate_new_min_arch(hw, cfg_node_to_hw_map, simulator.data_path, simulator.id_to_node)

    hw.init_memory(
        sim_util.find_nearest_power_2(simulator.memory_needed),
        sim_util.find_nearest_power_2(simulator.nvm_memory_needed),
    )

    hardwareModel.un_allocate_all_in_use_elements(hw.netlist)
   
    cfg_node_to_hw_map_copy = {}
    for key, val in cfg_node_to_hw_map.items():
        cfg_node_to_hw_map_copy[key] = val.copy()

    simulator.simulate(cfg, cfg_node_to_hw_map, hw)

    # print(f"values in cfg_node_to_hw_map after:")
    # [print(str(val)) for val in cfg_node_to_hw_map.values()]

    unique, count = np.unique([str(elem) for elem in simulator.data_path], return_counts=True)

    sorted_nodes = [
        x for _, x in sorted(zip(count, unique), key=lambda pair: pair[0], reverse=True)
    ]

    # get number of continuous most common node:
    # there should be a better way to do this
    max_continuous = 1
    prev = 0
    cont = 1
    saved_elem = 0
    for elem in simulator.data_path:
        if elem == prev and str(elem) == sorted_nodes[0]:
            saved_elem = elem
            cont += 1
        else:
            if cont > max_continuous:
                max_continuous = cont
            cont = 1
        prev = elem

    # print(f"max continuous: {max_continuous}")
    return simulator, hw, cfg, cfg_node_to_hw_map, saved_elem, max_continuous

def run_architecture_search(simulator, hw, cfg, cfg_node_to_hw_map, saved_elem, max_continuous, area_constraint, best_edp=None):
    unroll_factor = max_continuous // 2
    possible_unroll_factors = range(1, max_continuous + 1)
    possible_unrolls_edp = [0] * max_continuous
    possible_unroll_factor_probs = [1 / max_continuous] * max_continuous

    orig_data_path = deepcopy(simulator.data_path)
    # print(f"Original Data Path: {orig_data_path}")
    simulator.calculate_edp(hw)
    if best_edp is None:
        best_edp = simulator.edp
    possible_unrolls_edp[0] = best_edp
    best_hw = hw
    best_unroll = 1

    print(
        f"Original EDP: {best_edp}; power: {simulator.avg_compute_power} mW; execution time: {simulator.execution_time} s, cycles: {simulator.cycles}"
    )

    prev_edp = best_edp
    prev_unroll = 1
    unroll_low = 1
    unroll_high = max_continuous

    epsilon = 0.01  # randomness for rounding

    for i in range(2):
        print(f"Simulating new architecture {i}...")
        hw_copy_ = deepcopy(hw)
        # sim_copy_ = deepcopy(sim_copy)
        new_data_path = gen_new_arch(
            hw_copy_,
            unroll_factor,
            cfg_node_to_hw_map,
            orig_data_path,
            simulator.id_to_node,
            simulator.memory_needed,
            saved_elem,
        )
        # print([str(key) for key in cfg_node_to_hw_map.keys()])
        simulator.update_data_path(new_data_path)

        new_hw = hw_copy_

        EDP = simulate_new_arch(new_hw, simulator, cfg, cfg_node_to_hw_map)
        # print(
        #     f"EDP: {EDP}; power: {simulator.avg_compute_power} mW; execution time: {simulator.execution_time} s; cycles: {simulator.cycles}"
        # )
        # nx.draw(new_hw.netlist, with_labels=True)
        # plt.show()

        # except:
        #     print("new arch can't execute computation")
        #     continue
        area = new_hw.get_total_area()
        # print(f"Area: {area} um^2")
        if area > area_constraint:
            EDP += np.inf
            unroll_high = unroll_factor
        area_ratio = area / area_constraint
        possible_unrolls_edp[unroll_factor - 1] = EDP
        if EDP < best_edp:
            best_edp = EDP
            best_hw = new_hw
            best_unroll = unroll_factor
        if EDP <= prev_edp:
            if unroll_factor > prev_unroll:
                unroll_low = unroll_factor
            elif unroll_factor < prev_unroll:
                unroll_high = unroll_factor
            unroll_factor = round(
                (unroll_low + unroll_high + rng.choice([+1, -1]) * epsilon)
                / (2 * area_ratio)
            )
        elif EDP > prev_edp:  # backtrack
            if unroll_factor > prev_unroll:
                unroll_high = unroll_factor
            elif unroll_factor < prev_unroll:
                unroll_low = unroll_factor
            unroll_factor = round(
                (unroll_low + unroll_high + rng.choice([+1, -1]) * epsilon)
                / (2 * area_ratio)
            )
        unroll_factor = max(1, unroll_factor)
        unroll_factor = min(max_continuous, unroll_factor)

        prev_edp = EDP
        prev_unroll = unroll_factor
    # print(f"All EDPS: {possible_unrolls_edp}")
    print(f"Best Unroll Factor: {best_unroll}")
    return best_hw, best_edp


def main():
    """
    Currently the data path is just going to get modified once
    when I do the unrolling;
    so I can do all the setup once and then just simulate
    a bunch of diff architectures.
    """

    print(f"Running Architecture Search for {args.benchmark.split('/')[-1]}")

    simulator, hw, cfg,cfg_node_to_hw_map, saved_elem, max_continuous = setup_arch_search(args.benchmark)
    best_hw, best_edp = run_architecture_search(simulator, hw, cfg, cfg_node_to_hw_map, saved_elem, max_continuous, args.area)
    
    if args.filepath:
        nx.write_gml(best_hw.netlist,f"architectures/{args.filepath}.gml", stringizer=lambda x: str(x))

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
        "-f", "--filepath", type=str, help="Path to the save new architecture file"
    )
    args = parser.parse_args()
    print(
        f"args: benchmark: {args.benchmark}, trace:{args.notrace}, area:{args.area}, file: {args.filepath}"
    )

    main(args)
