# builtin modules
import os
import math
import json
from pathlib import Path
import argparse
import logging
from functools import reduce

logger = logging.getLogger(__name__)

# third party modules
import matplotlib.pyplot as plt
import numpy as np
import graphviz as gv
import networkx as nx
import sympy as sp

# custom modules
from .memory import Memory
from . import schedule
from . import dfg_algo
from . import hardwareModel
from .hardwareModel import (
    HardwareModel,
)  # have to import both or else python will say the class is undefined
from . import sim_util
from . import arch_search_util
from .config_dicts import op2sym_map
from .abstract_simulate import AbstractSimulator
from openroad_interface import place_n_route

MEMORY_SIZE = 1000000

rng = np.random.default_rng()


class ConcreteSimulator(AbstractSimulator):
    def __init__(self):
        self.data = {}
        self.cycles = 0  # counter for number of cycles
        self.main_cfg = None
        self.id_to_node = {}
        self.path = os.getcwd()
        self.data_path = []
        self.active_power_use = {}
        self.mem_power_use = []
        self.node_intervals = []
        self.unroll_at = {}
        self.vars_allocated = {}
        self.where_to_free = {}
        self.memory_needed = 0
        self.nvm_memory_needed = 0
        self.total_malloc_size = 0
        self.cur_memory_size = 0
        self.new_graph = None  # this is just for visualization
        self.mem_layers = 0
        self.tech_node = 0
        self.pitch = 0
        self.cache_size = 0
        self.reads = 0
        self.nvm_reads = 0
        self.writes = 0
        self.total_read_size = 0
        self.total_nvm_read_size = 0
        self.total_write_size = 0
        self.max_regs_inuse = 0
        self.max_mem_inuse = 0
        self.total_energy = 0
        self.active_energy = 0
        self.active_energy_no_mem = 0
        self.passive_energy = 0
        self.passive_energy_no_mem = 0
        self.resource_edge_graph = None

    def get_var_size(self, var_name, mem_module: Memory):
        """
        Only called on Regs nodes that are parents / children of an operation.
        Checks the size of the variable in the mem heap that this register points to.

        Parameters:
            neighbor: The register node to be processed

        Returns:
            int: The size of the variable this register is storing.
        """
        mem_size = 0
        bracket_count = sim_util.get_matching_bracket_count(var_name)
        var_name = sim_util.get_var_name_from_arr_access(var_name)

        if var_name in mem_module.locations:
            mem_loc = mem_module.locations[var_name]
            mem_size = mem_loc.size
            for i in range(bracket_count):
                mem_size /= mem_loc.dims[i]

        return mem_size

    def get_mem_usage_of_dfg_node(self, hw_graph, mem_module: Memory):
        """
        Determine the amount of memory in use by this node in the DFG.
        Finds variables associated with all regs in this node and adds up their sizes.
        Parameters:
            hw_graph - nx.DiGraph of a single node in the DFG.
            each node in this DiGraph is a PE or a Regs node.
        Returns:
            int: The amount of memory in use by this operation.
        """

        mem_in_use = 0

        for node, data in hw_graph.nodes.data():
            if data["function"] == "Regs":
                # print(f"{node} is a Regs node; var name: {node.split(';')[0]}")
                mem_in_use += self.get_var_size(node.split(";")[0], mem_module)

        return mem_in_use

    def localize_memory(self, hw, hw_graph):
        """
        Updates memory in buffers (cache) to ensure that the data needed for the coming DFG node
        is in the cache.

        Sets a flag to indicate whether or not there was a cache hit or miss. This affects latency
        calculations.
        """
        for node, data in dict(
            filter(lambda x: x[1]["function"] == "Regs", hw_graph.nodes.data())
        ).items():
            var_name = node.split(";")[0]
            cache_hit = False
            # print(f"data[allocation]: {data['allocation']};\nhw_graph.nodes[node][allocation]: {hw_graph.nodes[node]['allocation']}")
            cache_hit = (
                hw.netlist.nodes[data["allocation"]]["var"] == var_name
            )  # no need to refetch from mem if var already in reg.
            mapped_edges = map(
                lambda edge: (edge[0], hw.netlist.nodes[edge[0]]),
                hw.netlist.in_edges(hw_graph.nodes[node]["allocation"], data=True),
            )

            in_bufs = list(
                filter(
                    lambda node_data: node_data[1]["function"] == "Buf", mapped_edges
                )
            )
            # check if cache_hit
            for buf in in_bufs:
                cache_hit = buf[1]["memory_module"].find(var_name) or cache_hit
                if cache_hit:
                    break

            # just choose one at random. Can make this smarter later.
            buf = rng.choice(in_bufs)  # in_bufs[0] # just choose one at random.
            size = buf[1]["memory_module"].read(var_name)
            buf_idx = len(
                list(filter(lambda x: x[1]["function"] == "Buf", hw_graph.nodes.data()))
            )

            # print(f"in localize memory; var: {var_name}; size: {size}")
            # add multiple bufs and mems, not just one. add a new one for each cache miss.
            # this is required to properly count active power consumption.
            # active power added once for each node in computation_graph.
            # what about latency????

            if cache_hit:  # only add the Buf
                hw_graph.add_node(
                    f"Buf{buf_idx}", function="Buf", allocation=buf[0], size=size
                )
                hw_graph.add_edge(f"Buf{buf_idx}", node, function="Mem")

            else:  # add Buf and Mem
                size = size * -1  # size will be negative because cache miss.
                mem_idx = len(
                    list(
                        filter(
                            lambda x: x[1]["function"] == "MainMem",
                            hw_graph.nodes.data(),
                        )
                    )
                )
                mem = list(
                    filter(
                        lambda x: x[1]["function"] == "MainMem", hw.netlist.nodes.data()
                    )
                )[0][0]
                hw_graph.add_node(
                    f"Buf{buf_idx}", function="Buf", allocation=buf[0], size=size
                )
                hw_graph.add_node(
                    f"Mem{mem_idx}", function="MainMem", allocation=mem, size=size
                )
                hw_graph.add_edge(f"Mem{mem_idx}", f"Buf{buf_idx}", function="Mem")
                hw_graph.add_edge(f"Buf{buf_idx}", node, function="Mem")

    def visualize_graph(self, operations):
        """
        Deprecated!
        """
        state_graph_viz = gv.Graph()
        state_graph = dfg_algo.Graph(set(), {}, state_graph_viz)
        op_count = 0
        for op in operations:
            if not op.operation:
                continue
            # if op.operation == "Regs": continue
            op_count += 1
            compute_id = dfg_algo.set_id()
            sim_util.make_node(
                state_graph,
                compute_id,
                op2sym_map[op.operation],
                None,
                op2sym_map[op.operation],
            )
            for parent in op.parents:
                if parent.operation:  # and parent.operation != "Regs":
                    parent_id = dfg_algo.set_id()
                    sim_util.make_node(
                        state_graph,
                        parent_id,
                        op2sym_map[parent.operation],
                        None,
                        op2sym_map[parent.operation],
                    )
                    sim_util.make_edge(state_graph, parent_id, compute_id, "")
            # what are we doing here that we don't want to check duplicate?
            self.get_mem_usage_of_dfg_node(
                op,
                state_graph,
                state_graph.id_to_Node[compute_id],
                check_duplicate=False,
            )
        if op_count > 0:
            Path(self.path + "src/benchmarks/pictures/state_graphs/").mkdir(
                parents=True, exist_ok=True
            )
            state_graph_viz.render(
                self.path
                + "src/benchmarks/pictures/state_graphs/"
                + args.benchmark.split("/")[-1].split(".")[0]
                + "_"
                + str(state_graph_counter),
                view=True,
            )
        state_graph_counter += 1  # what does this do, can I get rid of it?

    def reset_internal_variables(self):
        self.cycles = 0
        self.net_delay = 0
        self.active_energy = 0
        self.passive_energy = 0
        self.net_active_energy = 0
        self.total_energy = 0
        self.active_energy_no_mem = 0
        self.passive_energy_no_mem = 0

    def get_parasitic_energy(self, node, computation_dfg, hw, active_net_cap):
        node_data = computation_dfg.nodes[node]
        out_nodes = list(computation_dfg.out_edges(node))
        node_name = node_data["allocation"]
        node_bool = (
            "allocation" in node_data
            and node_name != ""
            and "Mem" not in node_name
            and "Buf" not in node_name
        )

        def wire_energy(computation_dfg, hw, active_net_cap):
            """
            param:
                computation_dfg: to access the computation node to node
                hw: to access the graph with net and cap info
                active_net_cap: dict with nets and the respective caps.
            """
            for out_node in out_nodes:
                child_node_data = computation_dfg.nodes[out_node[1]]
                child_node_name = child_node_data["allocation"]
                comp_node_bool = (
                    "allocation" in child_node_data
                    and child_node_name != ""
                    and "Mem" not in child_node_name
                    and "Buf" not in child_node_name
                )
                if comp_node_bool:
                    # temporary solution, will be changed when computation_dfg gets fixed
                    parasitic_edge = None
                    def check_edge(node1, node2, end_num):
                        if not hw.parasitic_graph.has_edge(node1, node2):
                            parasitic_nodes_out = list(hw.parasitic_graph.out_edges(node1))
                            child_node_function = child_node_data["function"]
                            for node in parasitic_nodes_out:
                                if end_num != 0:
                                    if child_node_function in node[1] and end_num in node[1]:
                                        node2 = node[1]
                                        break
                                else: 
                                    if child_node_function in node[1]:
                                        node2 = node[1]
                                        break
                        return node2
                    
                    def consolidate_cap(node1, node2):
                        parasitic_edge = hw.parasitic_graph[node1][node2]
                        net_cap = (
                        parasitic_edge["net_cap"]
                        if isinstance(parasitic_edge["net_cap"], list)
                        else [parasitic_edge["net_cap"]]
                        )
                        net_name = parasitic_edge["net"]
                        if (net_name not in active_net_cap or len(active_net_cap[net_name]) < len(net_cap)): #if net not in the capacitances recognized or the net is a ssociated with a capacitance that is shorter
                            active_net_cap[net_name] = net_cap
                    
                    if "Regs" in node_name: #or any 16 bit function, figure out a way to not explicitly state it 
                        for x in range(16):
                            child_node_name = check_edge(node_name + "_" + str(x), child_node_name, 0)
                            consolidate_cap(node_name + "_" + str(x), child_node_name)
                    elif "Regs" in child_node_name:
                        for x in range(16):
                            child_node_name = check_edge(node_name, child_node_name + "_" + str(x), "_" + str(x))
                            consolidate_cap(node_name, child_node_name)
                    else:
                        check_edge(node_name, child_node_name)
                        consolidate_cap(node_name, child_node_name)
                
        if node_bool:
            wire_energy(computation_dfg, hw, active_net_cap)

    def simulate(self, computation_dfg, hw):
        """
        Simulates one large DFG representing the whole computation.
        can I overload function names like this?

        TODO: Doesn't yet add in all buff and mem operations
        TODO: This is a great candidate to cpp ify via cppyy
        TODO: mux exists, must account for that
        """
        self.reset_internal_variables()

        generations = list(
            reversed(list(nx.topological_generations(nx.reverse(computation_dfg))))
        )

        for gen in generations:  # main loop over the computation graphs;
            if "end" in gen:  # skip the end node (only thing in the last generation)
                break
            active_net_cap = {}
            for node in gen:
                logger.info(f"node -> {node}: {computation_dfg.nodes[node]}")

                ## ================= ADD ACTIVE ENERGY CONSUMPTION =================
                scaling = 1
                node_data = computation_dfg.nodes[node]
                if node_data["function"] == "stall" or node_data["function"] == "end":
                    continue
                if node_data["function"] in ["Buf", "MainMem"]:
                    # active power should scale by size of the object being accessed.
                    # all regs have the same size, so no need to scale.
                    if node_data["function"] == "MainMem":
                        scaling = node_data["size"] / hw.memory_bus_width
                    else:
                        scaling = node_data["size"] / hw.buffer_bus_width
                    logger.info(f"scaling: {scaling}")
                    energy = (
                        (
                            hw.dynamic_energy[node_data["function"]]["Read"]
                            + hw.dynamic_energy[node_data["function"]]["Write"]
                        )
                        / 2  # avg of read and write
                        * 1e-9 # nW to W
                        * scaling
                    )

                    if node_data["function"] == "MainMem":
                        energy += (
                            hw.dynamic_power["OffChipIO"]
                            * 1e-9
                            * scaling
                            * hw.latency["OffChipIO"]
                        )
                else:
                    energy = (
                        hw.dynamic_power[node_data["function"]]
                        * 1e-9  # nW to W
                        * scaling
                        * hw.latency[node_data["function"]]  # ns
                    )
                    self.active_energy_no_mem += energy
                #fn = node_data["function"]
                #print(f"adding {energy} to objective for {fn}")
                self.active_energy += energy
                hw.compute_operation_totals[node_data["function"]] += 1

                # wires
                if hw.parasitics != "none":
                    self.get_parasitic_energy(node, computation_dfg, hw, active_net_cap)
                
            net_energy = 0
            for net in active_net_cap:
                cap_sum = sum(active_net_cap[net])
                net_energy += (
                    float(1 / 2) * float(cap_sum) * float(hw.V_dd * hw.V_dd * 1e-3)
                )  # pico -> nano
            self.active_energy += net_energy
            self.net_active_energy += net_energy
        # get start_time of end node
        nodes_dict = {node: data for node, data in computation_dfg.nodes(data=True)}
        # for key, val in nodes_dict.items():
        #     print(key, val)
        self.cycles = nodes_dict["end"]["start_time"]
        logger.info(f"longest path length calculated as end node start time: {self.cycles}")

        for elem_name, elem_data in dict(hw.netlist.nodes.data()).items():

            self.passive_energy += (
                hw.leakage_power[elem_data["function"]] * 1e-9 * self.cycles
            )
            self.passive_energy_no_mem += (
                hw.leakage_power[elem_data["function"]] * 1e-9 * self.cycles
            ) if elem_data["function"] not in ["MainMem", "Buf"] else 0

    def calculate_edp(self):
        if isinstance(self.cycles, sp.Expr):
            self.cycles = self.cycles.evalf()
        if isinstance(self.active_energy, sp.Expr):
            self.active_energy = self.active_energy.evalf()
        if isinstance(self.passive_energy, sp.Expr):
            self.passive_energy = self.passive_energy.evalf()

        self.execution_time = self.cycles # in seconds
        self.total_energy = self.active_energy + self.passive_energy
        self.total_energy_no_mem = self.active_energy_no_mem + self.passive_energy_no_mem
        self.edp = self.total_energy * self.execution_time
        logger.info(f"execution time: {self.execution_time} ns")
        logger.info(f"total energy: {self.total_energy} nJ")
        logger.info(f"total energy no mem: {self.total_energy_no_mem} nJ")


def main(args):
    print(f"Running simulator for {args.benchmark.split('/')[-1]}")
    simulator = ConcreteSimulator()

    hw = HardwareModel(cfg=args.architecture_config)

    computation_dfg, mallocs = simulator.simulator_prep(args.benchmark, hw.latency)

    gen_cacti = not args.debug_no_cacti

    hw.init_memory(
        sim_util.find_nearest_power_2(simulator.memory_needed),
        sim_util.find_nearest_power_2(simulator.nvm_memory_needed),
        mallocs,
        gen_cacti=gen_cacti
    )

    computation_dfg = simulator.schedule(computation_dfg, hw, args.schedule)

    simulator.tech_node = hw.transistor_size  # in nm
    simulator.pitch = hw.pitch  # in um
    simulator.mem_layers = hw.mem_layers
    if simulator.memory_needed < 1000000:
        simulator.cache_size = 1
    elif simulator.memory_needed < 2000000:
        simulator.cache_size = 2
    elif simulator.memory_needed < 4000000:
        simulator.cache_size = 4
    elif simulator.memory_needed < 8000000:
        simulator.cache_size = 8
    else:
        simulator.cache_size = 16

    new_gv_graph = gv.Graph()
    simulator.new_graph = dfg_algo.Graph(set(), {}, new_gv_graph)

    hardwareModel.un_allocate_all_in_use_elements(hw.netlist)
    hw.get_wire_parasitics(args.openroad_testfile, args.parasitics)
    data = simulator.simulate(computation_dfg, hw)
    simulator.calculate_edp()

    area = hw.get_total_area(gen_cacti)

    # print stats
    print("total number of cycles: ", simulator.cycles)
    print(f"execution time: {simulator.execution_time} ns")
    print(f"total energy: {simulator.total_energy} nJ")
    print(f"total wire energy: {simulator.net_active_energy} nJ")

    print(f"on chip area: {area} um^2")
    print(f"EDP: {simulator.edp} E-18 Js")

    # TODO: FIX THIS WITH CACTI
    # print(f"off chip memory area: {hw.mem_area * 1e6} um^2")
    # print(f"total area: {(area + hw.mem_area*1e6)} um^2")

    # TODO: FIX THESE WITH CACTI
    # avg_mem_power = np.mean(simulator.mem_power_use) + hw.mem_leakage_power
    # print(f"Avg mem Power: {avg_mem_power} mW")
    # print(f"Total power: {avg_mem_power + avg_compute_power} mW")

    print("total volatile reads: ", simulator.reads)
    print("total volatile read size: ", simulator.total_read_size)
    print("total nvm reads: ", simulator.nvm_reads)
    print("total nvm read size: ", simulator.total_nvm_read_size)
    print("total writes: ", simulator.writes)
    print("total write size: ", simulator.total_write_size)

    print("max regs in use: ", simulator.max_regs_inuse)
    print(f"max memory in use: {simulator.max_mem_inuse} bytes")

    print("total operations computed: ", hw.compute_operation_totals)

    # save some dump of data to json file
    names = args.benchmark.split("/")
    if not args.notrace:
        text = json.dumps(data, indent=4)
        Path(simulator.path + "/src/benchmarks/json_data").mkdir(
            parents=True, exist_ok=True
        )
        with open(simulator.path + "/src/benchmarks/json_data/" + names[-1], "w") as fh:
            fh.write(text)

    print("done!")
    return simulator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="logs/simulate.log")

    parser = argparse.ArgumentParser(
        prog="Simulate",
        description="Runs a hardware simulation on a given benchmark and technology spec",
        epilog="Text at the bottom of help",
    )

    parser.add_argument(
        "--parasitics",
        type=str,
        choices=["detailed", "estimation", "none"],
        default="detailed",
        help="determines what type of parasitic calculations are done for wires",
    )

    parser.add_argument(
        "--openroad_testfile",
        type=str,
        default="openroad_interface/tcl/test_nangate45_bigger.tcl",
        help="what tcl file will be executed for openroad",
    )

    parser.add_argument("benchmark", metavar="B", type=str)
    parser.add_argument("--notrace", action="store_true")
    parser.add_argument(
        "--architecture_config",
        type=str,
        default="mm_test",  # aladdin_const_with_mem
        help="Path to the architecture file (.gml)",
    )
    parser.add_argument('--schedule', type=str, choices=['greedy', 'sdc'], default='greedy',
                        help='Scheduling algorithm to use')
    parser.add_argument('--debug_no_cacti', type=bool, default=False, 
                        help='disable cacti in the first iteration to decrease runtime when debugging')
    args = parser.parse_args()

    print(
        f"args: benchmark: {args.benchmark}, trace: {args.notrace}, architecture: {args.architecture_config}, parasitics: {args.parasitics}, schedule: {args.schedule}"
    )
    main(args)
