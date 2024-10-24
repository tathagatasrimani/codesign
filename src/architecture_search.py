# builtin
import argparse
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)

# third party
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# custom
from .simulate import ConcreteSimulator
from . import hardwareModel
from . import arch_search_util
from . import sim_util
from .global_constants import SEED
from .schedule import sdc_schedule

rng = np.random.default_rng(SEED)

def setup_arch_search(benchmark, arch_init_config):
    simulator = ConcreteSimulator()

    hw = hardwareModel.HardwareModel(cfg=arch_init_config)

    computation_dfg = simulator.simulator_prep(benchmark, hw.latency)

    hw.netlist = nx.DiGraph()

    arch_search_util.generate_new_min_arch_on_whole_dfg(hw, computation_dfg)
    logger.info(f"Initial netlist: {hw.netlist.nodes}")
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


def get_most_stalled_func(scheduled_dfg) -> str:
    func_counts = get_stalled_func_counts(scheduled_dfg)
    return max(func_counts, key=func_counts.get)

def greedy_select_func_to_add(computation_dfg, scheduled_dfg):
    # take the ideal schedule and look at the end times of each node in a topological order
    # first node where end time of real schedule is greater than end time of ideal schedule
    # return that node
    topo_order = list(nx.topological_sort(computation_dfg))
    ideal_schedule = sdc_schedule(computation_dfg, hardwareModel.get_func_count(scheduled_dfg), hardwareModel.netlist, no_resource_constraints=True)
    for node in topo_order:
        if scheduled_dfg.nodes[node]["end_time"] > ideal_schedule.nodes[node]["end_time"]:
            return node
    return None



def sample_stalled_func(scheduled_dfg: nx.DiGraph) -> str:
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
    logger.info(f"Updating HW with scarce node: {scarce_function}")
    if scarce_function == None:
        return
    func_nodes = hardwareModel.get_nodes_with_func(hw_netlist, scarce_function)
    logger.info(f"existing nodes with scarce function: {func_nodes}")
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
        hw_netlist.nodes[f"{scarce_function}{idx}"]["var"] = ""
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
        hw_netlist.nodes[f"{scarce_function}{idx}"]["type"] = "memory"
        hw_netlist.nodes[f"{scarce_function}{idx}"]["size"] = 1
        hw_netlist.nodes[f"{scarce_function}{idx}"]["memory_module"] = list(func_nodes.values())[0][
            "memory_module"
        ]
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
        hw_netlist.nodes[f"{scarce_function}{idx}"]["type"] = "memory"
        hw_netlist.nodes[f"{scarce_function}{idx}"]["size"] = 1
        hw_netlist.nodes[f"{scarce_function}{idx}"]["memory_module"] = list(func_nodes.values())[0][
            "memory_module"
        ]
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


def run_arch_search(
    simulator: ConcreteSimulator,
    hw: hardwareModel,
    computation_dfg: nx.DiGraph,
    area_constraint: float,
    num_steps: int = 1,
    best_edp=None,
):
    """
    Run the architecture search algorithm

    Params:
    simulator: ConcreteSimulator - the simulator object
    hw: HardwareModel - the hardware model
    computation_dfg: nx.DiGraph - the computation graph
    area_constraint: float - max area of the chip in um^2
    num_steps: int - number of steps to run the search for
    best_edp: float
    """

    old_scheduled_dfg = simulator.schedule(computation_dfg, hw)
    
    simulator.simulate(old_scheduled_dfg, hw)
    simulator.calculate_edp()
    area = hw.get_total_area()

    logger.info(
        f"AS EDP init: {simulator.edp} E-18 Js. Active Energy: {simulator.active_energy} nJ. Passive Energy: {simulator.passive_energy} nJ. Execution time: {simulator.execution_time} ns"
    )

    if best_edp is None:
        best_edp = simulator.edp

    logger.info(f"Best EDP: {best_edp} E-18 Js")
    best_hw = deepcopy(hw)
    best_schedule = old_scheduled_dfg

    best_active_energy = simulator.active_energy
    best_passive_energy = simulator.passive_energy
    best_execution_time = simulator.execution_time

    hw_copy = deepcopy(hw)

    for i in range(num_steps):
        func = greedy_select_func_to_add(computation_dfg, old_scheduled_dfg)
        if func is None:
            print("Ideal schedule reached")
            break
        # func_to_add = greedy_select_func_to_add(computation_dfg, old_scheduled_dfg)

        update_hw_with_new_node(hw_copy.netlist, func)
        hw_copy.update_netlist()
        logger.info("updated netlist")
        logger.info(f"new func counts: {hardwareModel.get_func_count(hw_copy.netlist)}")
        hw_copy.gen_cacti_results()
        logger.info("generated cacti results")

        scheduled_dfg = simulator.schedule(computation_dfg, hw_copy)
        logger.info("scheduled dfg")

        func_counts = get_stalled_func_counts(scheduled_dfg)

        hw = hw_copy

        simulator.simulate(scheduled_dfg, hw)
        simulator.calculate_edp()
        logger.info(f"simulated; execution time: {simulator.execution_time} ns, passive energy: {simulator.passive_energy} nJ, active energy: {simulator.active_energy} nJ, edp: {simulator.edp} E-18 Js")

        area = hw.get_total_area()
        if area > area_constraint:
            logger.info("Area constraint exceeded; breaking")
            # shouldn't actually break here, because you can try other nodes that are smaller but still might have a good effect
            break
        elif simulator.edp < best_edp:
            logger.info(f"Adding {func} improved EDP from {best_edp} to {simulator.edp}")
            best_edp = simulator.edp
            best_active_energy = simulator.active_energy
            best_passive_energy = simulator.passive_energy
            best_execution_time = simulator.execution_time
            best_hw = deepcopy(hw_copy)
            best_schedule = scheduled_dfg
        else: 
            logger.info(f"Adding node ({func}) did not improve EDP; reverting; EDP {simulator.edp}, best_edp {best_edp}")

        old_scheduled_dfg = scheduled_dfg
        if len(func_counts) == 0:
            break

    hw = best_hw
    hw.update_netlist()

    simulator.active_energy = best_active_energy
    simulator.passive_energy = best_passive_energy
    simulator.execution_time = best_execution_time
    simulator.edp = best_edp

    logger.info(
        f"AS EDP     : {simulator.edp} E-18 Js. Active Energy: {simulator.active_energy} nJ. Passive Energy: {simulator.passive_energy} nJ. Execution time: {simulator.execution_time} ns"
    )

    return best_schedule, best_hw


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
            f"src/architectures/{args.filepath}.gml",
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
