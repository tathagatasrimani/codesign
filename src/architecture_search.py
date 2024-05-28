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
from global_constants import SEED

rng = np.random.default_rng(SEED)


def simulate_new_arch(hw, simulator, cfg, cfg_node_to_hw_map):
    hardwareModel.un_allocate_all_in_use_elements(hw.netlist)
    simulator.simulate(cfg, cfg_node_to_hw_map, hw)
    simulator.calculate_edp(hw)
    return simulator.edp


def gen_new_arch(
    existing_arch,
    unroll_factor,
    cfg_node_to_hw_map,
    data_path,
    id_to_node,
    mem,
    saved_elem,
):
    """
    DEPRECATED!
    """
    existing_arch.netlist = nx.DiGraph()
    new_data_path = arch_search_util.unroll_by_specified_factor(
        cfg_node_to_hw_map, data_path, id_to_node, unroll_factor, saved_elem, log=False
    )

    unique_data_path = [list(x) for x in set(tuple(x) for x in new_data_path)]
    arch_search_util.generate_new_min_arch(
        existing_arch, cfg_node_to_hw_map, unique_data_path, id_to_node
    )
    existing_arch.init_memory(
        sim_util.find_nearest_power_2(mem),
        sim_util.find_nearest_power_2(0),
    )

    return new_data_path


def setup_arch_search(benchmark, arch_init_config):
    simulator = ConcreteSimulator()

    hw = hardwareModel.HardwareModel(cfg=arch_init_config)

    computation_dfg = simulator.simulator_prep(benchmark, hw.latency)

    hw.netlist = nx.DiGraph()

    arch_search_util.generate_new_min_arch_on_whole_dfg(hw, computation_dfg)

    hw.init_memory(
        sim_util.find_nearest_power_2(simulator.memory_needed),
        sim_util.find_nearest_power_2(simulator.nvm_memory_needed),
    )

    hardwareModel.un_allocate_all_in_use_elements(hw.netlist)

    return (
        simulator,
        hw,
        computation_dfg,
    )


def get_stalled_func_counts(scheduled_dfg):
    stalled_nodes = [
        node
        for node in scheduled_dfg.nodes(data=True)
        if node[1]["function"] == "stall"
    ]
    stalled_funcs, counts = np.unique(
        list(map(lambda x: x[0].split("_")[3], stalled_nodes)), return_counts=True
    )
    return dict(zip(stalled_funcs, counts))


def get_most_stalled_func(scheduled_dfg):
    func_counts = get_stalled_func_counts(scheduled_dfg)
    return max(func_counts, key=func_counts.get)


def sample_stalled_func(scheduled_dfg):
    func_counts = get_stalled_func_counts(scheduled_dfg)
    if len(func_counts) == 0:
        return None
    try:
        return rng.choice(
            list(func_counts.keys()),
            1,
            p=list(func_counts.values()) / sum(list(func_counts.values())),
        )[0]
    except:
        return rng.choice(list(func_counts.keys()), 1)[0]


def update_hw_with_new_node(hw_netlist, scarce_function):
    if scarce_function == None:
        return
    func_nodes = hardwareModel.get_nodes_with_func(hw_netlist, scarce_function)
    idx = len(func_nodes)
    hw_netlist.add_node(
        f"{scarce_function}{idx}",
        function=scarce_function,
        idx=idx,
        in_use=False,
        type="pe",
    )
    if scarce_function == "Regs":
        hw_netlist.nodes[f"{scarce_function}{idx}"]["type"] = "memory"
        hw_netlist.nodes[f"{scarce_function}{idx}"]["size"] = 1
        # add edges to all pes
        for node2 in list(
            map(
                lambda x: x[0],
                filter(lambda x: x[1]["type"] == "pe", hw_netlist.nodes.data()),
            )
        ):
            hw_netlist.add_edge(f"{scarce_function}{idx}", node2)
            hw_netlist.add_edge(node2, f"{scarce_function}{idx}")
        # add add edges to all Bufs
        for node2 in list(
            map(
                lambda x: x[0],
                filter(lambda x: x[1]["function"] == "Buf", hw_netlist.nodes.data()),
            )
        ):
            hw_netlist.add_edge(f"{scarce_function}{idx}", node2)
            hw_netlist.add_edge(node2, f"{scarce_function}{idx}")
    elif scarce_function == "Buf":
        hw_netlist[f"{scarce_function}{idx}"]["type"] = "memory"
        hw_netlist[f"{scarce_function}{idx}"]["size"] = 1
        # add edges to all Regs
        for node2 in list(
            map(
                lambda x: x[0],
                filter(lambda x: x[1]["function"] == "Reg", hw_netlist.nodes.data()),
            )
        ):
            hw_netlist.add_edge(f"{scarce_function}{idx}", node2)
            hw_netlist.add_edge(node2, f"{scarce_function}{idx}")
        # add edges to all MainMems
        for node2 in list(
            map(
                lambda x: x[0],
                filter(
                    lambda x: x[1]["function"] == "MainMem", hw_netlist.nodes.data()
                ),
            )
        ):
            hw_netlist.add_edge(f"{scarce_function}{idx}", node2)
            hw_netlist.add_edge(node2, f"{scarce_function}{idx}")
    elif scarce_function == "MainMem":
        hw_netlist[f"{scarce_function}{idx}"]["type"] = "memory"
        hw_netlist[f"{scarce_function}{idx}"]["size"] = 1
        # add edges to all Bufs
        for node2 in list(
            map(
                lambda x: x[0],
                filter(lambda x: x[1]["function"] == "Buf", hw_netlist.nodes.data()),
            )
        ):
            hw_netlist.add_edge(f"{scarce_function}{idx}", node2)
            hw_netlist.add_edge(node2, f"{scarce_function}{idx}")
    else:
        for node2 in hw_netlist.nodes:
            if (
                "Reg" in node2
                or "Buf" in node2
                or "MainMem" in node2
                or node2 == f"{scarce_function}{idx}"
            ):
                continue
            hw_netlist.add_edge(f"{scarce_function}{idx}", node2)
            hw_netlist.add_edge(node2, f"{scarce_function}{idx}")


def run_arch_search(simulator, hw, computation_dfg, area_constraint, best_edp=None):

    old_scheduled_dfg = simulator.schedule(
        computation_dfg, hw_counts=hardwareModel.get_func_count(hw.netlist)
    )

    simulator.simulate(old_scheduled_dfg, hw)
    simulator.calculate_edp(hw)
    area = hw.get_total_area()

    if best_edp is None:
        best_edp = simulator.edp
    best_hw_netlist = hw.netlist.copy()
    best_schedule = old_scheduled_dfg

    for i in range(0):
        hw_copy = hw.netlist.copy()

        func = sample_stalled_func(old_scheduled_dfg)

        update_hw_with_new_node(hw_copy, func)

        scheduled_dfg = simulator.schedule(
            computation_dfg, hw_counts=hardwareModel.get_func_count(hw_copy)
        )

        func_counts = get_stalled_func_counts(scheduled_dfg)

        if nx.is_isomorphic(old_scheduled_dfg, scheduled_dfg):
            print("no change in schedule")
            # continue

        hw.netlist = hw_copy

        simulator.simulate(scheduled_dfg, hw)
        simulator.calculate_edp(hw)

        area = hw.get_total_area()
        if area > area_constraint:
            # shouldn't actually break here, because you can try other nodes that are smaller but still might have a good effect
            break
        elif simulator.edp < best_edp:
            best_edp = simulator.edp
            best_hw_netlist = hw_copy
            best_schedule = scheduled_dfg

        old_scheduled_dfg = scheduled_dfg
        if len(func_counts) == 0:
            break

    hw.netlist = best_hw_netlist

    return best_edp, best_schedule


def main():
    """
    Currently the data path is just going to get modified once
    when I do the unrolling;
    so I can do all the setup once and then just simulate
    a bunch of diff architectures.
    """

    print(f"Running Architecture Search for {args.benchmark.split('/')[-1]}")

    simulator, hw, computation_dfg = setup_arch_search(args.benchmark, args.config)

    best_edp = run_arch_search(simulator, hw, computation_dfg, args.area)

    if args.filepath:
        nx.write_gml(
            hw.netlist,
            f"architectures/{args.filepath}.gml",
            stringizer=lambda x: str(x),
        )
        hw.duplicate_config_section(args.config, args.filepath)

    print(f"Best EDP: {best_edp}")
    nx.draw(
        hw.netlist,
        with_labels=True,
    )
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
        "-c",
        "--config",
        type=str,
        default="aladdin_const_with_mem",
        help="Path to the architecture config file",
    )
    parser.add_argument(
        "-f", "--filepath", type=str, help="Path to the save new architecture file"
    )
    args = parser.parse_args()
    print(
        f"args: benchmark: {args.benchmark}, trace:{args.notrace}, area:{args.area}, file: {args.filepath}"
    )

    main()
