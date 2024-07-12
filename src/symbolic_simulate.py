# builtin modules
import math
import argparse
import os
import sys
import logging

logger = logging.getLogger(__name__)

# third party modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sympy import *
import sympy as sp
import pyomo.environ as pyo
from pyomo.core.expr import Expr_if

# custom modules
from memory import Memory
import schedule
import dfg_algo
import hw_symbols
import sim_util
import hardwareModel
from hardwareModel import HardwareModel
from global_constants import SEED
from abstract_simulate import AbstractSimulator

rng = np.random.default_rng(SEED)

def symbolic_convex_max(a, b):
    """
    An approximation to the max function that plays well with these numeric solvers.
    """
    return 0.5 * (a + b + abs(a - b))


class SymbolicSimulator(AbstractSimulator):

    def __init__(self):
        self.cycles = 0
        self.cycles_ceil = 0
        self.id_to_node = {}
        self.path = os.getcwd()
        self.data_path = []
        self.node_intervals = []
        self.node_sum_energy = {}
        self.node_sum_cycles = {}
        self.node_sum_cycles_ceil = {}
        self.unroll_at = {}
        self.vars_allocated = {}
        self.where_to_free = {}
        self.compute_element_to_node_id = {}
        self.compute_element_neighbors = {}
        self.memory_needed = 0
        self.cur_memory_size = 0
        self.new_graph = None
        self.mem_layers = 0
        self.transistor_size = 0
        self.pitch = 0
        self.cache_size = 0
        self.reads = 0
        self.writes = 0
        self.total_read_size = 0
        self.total_write_size = 0
        self.max_regs_inuse = 0
        self.max_mem_inuse = 0
        self.edp = None
        self.edp_ceil = None
        self.initial_params = {}

    def reset_internal_variables(self):
        self.sim_cache = {}
        self.total_active_energy = 0
        self.total_passive_energy = 0
        self.total_passive_energy_ceil = 0
        self.cycles_ceil = 0
        self.cycles = 0

    def localize_memory(self, hw, hw_graph):
        """
        Updates memory in buffers (cache) to ensure that the data needed for the coming DFG node
        is in the cache.

        Sets a flag to indicate whether or not there was a cache hit or miss. This affects latency
        calculations.

        This is kind of bypassed right now after we removed the caching.
        """
        # print(f"hw_graph.nodes.data(): {hw_graph.nodes.data()}")
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
            for buf in in_bufs:
                cache_hit = buf[1]["memory_module"].find(var_name) or cache_hit
                if cache_hit:
                    break
            if not cache_hit:
                # just choose one at random. Can make this smarter later.
                buf = rng.choice(in_bufs)  # in_bufs[0] # just choose one at random.

                size = -1 * buf[1]["memory_module"].read(
                    var_name
                )  # size will be negative because cache miss; size is var size not mem object size

                # add multiple bufs and mems, not just one. add a new one for each cache miss.
                # this is required to properly count active power consumption.
                # active power added once for each node in computation_graph.
                # what about latency????
                buf_idx = len(
                    list(
                        filter(
                            lambda x: x[1]["function"] == "Buf", hw_graph.nodes.data()
                        )
                    )
                )
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
                # hw_graph.add_node(
                #     f"Buf{buf_idx}", function="Buf", allocation=buf[0], size=size
                # )
                hw_graph.add_node(
                    f"Mem{mem_idx}", function="MainMem", allocation=mem, size=size
                )
                hw_graph.add_edge(f"Mem{mem_idx}", node, function="Mem")
                # hw_graph.add_edge(f"Buf{buf_idx}", node, function="Mem")

    def passive_energy_dissipation(self, hw, total_execution_time):
        """
        Passive power is a function of purely the allocated hardware,
        not the actual computation being computed. This is calculated separately at the end.
        """
        passive_power = 0
        for node, data in dict(
            filter(lambda x: x[1]["function"] != "Buf", hw.netlist.nodes.data())
        ).items():
            scaling = 1
            if data["function"] in ["MainMem"]:
                scaling = data["size"]
            passive_power += (
                hw_symbols.symbolic_power_passive[data["function"]] * scaling
            )  # W
        return passive_power * total_execution_time  # nJ

    def simulate(self, computation_dfg: nx.DiGraph, hw: HardwareModel):
        self.reset_internal_variables()
        hw_symbols.update_symbolic_passive_power(hw.R_off_on_ratio)

        counter = 0

        generations = list(
            reversed(list(nx.topological_generations(nx.reverse(computation_dfg))))
        )
        for gen in generations:  # main loop over the computation graph;
            if "end" in gen:  # skip the end node (only thing in the last generation)
                break
            counter += 1
            for node in gen:
                logger.info(f"node: {computation_dfg.nodes[node]}")

                child_added = False

                ## ================= ADD ACTIVE ENERGY CONSUMPTION =================
                scaling = 1
                node_data = computation_dfg.nodes[node]
                if node_data["function"] == "stall" or node_data["function"] == "end":
                    continue
                if node_data["function"] in ["Buf", "MainMem"]:
                    # active power should scale by size of the object being accessed.
                    # all regs have the same size, so no need to scale.
                    scaling = node_data["size"]
                    logger.info(f"energy scaling: {scaling}")
                    energy = hw_symbols.symbolic_energy_active[
                        node_data["function"]
                    ]  # nJ
                else:
                    energy = (
                        hw_symbols.symbolic_power_active[node_data["function"]]
                        * hw_symbols.symbolic_latency_wc[node_data["function"]]
                    )  # W * ns

                self.total_active_energy += energy * scaling  # nJ

        # TODO: NOW THIS MIGHT GET TOO EXPENSIVE. MAYBE NEED TO DO STA.
        logger.info("Starting Longest Path Calculation")
        for start_node in generations[0]:
            for end_node in generations[-1]:
                if start_node == end_node:
                    continue
                logger.info(f"start_node: {start_node}, end_node: {end_node}")
                for path in nx.all_simple_paths(computation_dfg, start_node, end_node):
                    path_latency = 0
                    for node in path:
                        scaling = 1
                        node_data = computation_dfg.nodes[node]
                        if node_data["function"] == "end":
                            continue
                        elif node_data["function"] == "stall":
                            func = node.split("_")[3]  # stall names have std formats
                        else:
                            func = node_data["function"]
                        if func in ["Buf", "MainMem"]:
                            scaling = node_data["size"]
                            logger.info(f"latency scaling: {scaling}")

                        path_latency += hw_symbols.symbolic_latency_wc[func] * scaling
                    self.cycles = symbolic_convex_max(self.cycles, path_latency)
        logger.info(f"execution time: {str(self.cycles)}")

    def calculate_edp(self, hw):

        with open("MemL.txt", "r") as file:
            meml_text = file.read()

        with open("BufL.txt", "r") as file:
            bufl_text = file.read()

        meml_expr = sp.sympify(meml_text, locals=hw_symbols.symbol_table)
        bufl_expr = sp.sympify(bufl_text, locals=hw_symbols.symbol_table)

        cacti_subs = {
            # TODO
            hw_symbols.MemReadL: hw_symbols.MemWriteL,
            hw_symbols.MemWriteL: meml_expr,
            hw_symbols.BufL: bufl_expr,
        }

        self.cycles = self.cycles.subs(cacti_subs)

        self.total_passive_energy = self.passive_energy_dissipation(hw, self.cycles)
        # self.total_passive_energy_ceil = self.passive_energy_dissipation(
        #     hw, self.cycles_ceil
        # )
        self.edp = self.cycles * (self.total_active_energy + self.total_passive_energy)
        assert hw_symbols.MemReadL not in self.edp.free_symbols and hw_symbols.MemWriteL not in self.edp.free_symbols and hw_symbols.BufL not in self.edp.free_symbols

        # self.edp = self.edp.subs(subs)

    def save_edp_to_file(self):
        st = str(self.edp)
        with open("symbolic_edp.txt", "w") as f:
            f.write(st)

def main():
    print(f"Running symbolic simulator for {args.benchmark.split('/')[-1]}")

    simulator = SymbolicSimulator()

    hw = HardwareModel(cfg=args.architecture_config)

    hw.get_optimization_params_from_tech_params()
    print ("Checkpoint 1")

    computation_dfg = simulator.simulator_prep(args.benchmark, hw.latency)

    hw.init_memory(
        sim_util.find_nearest_power_2(simulator.memory_needed),
        sim_util.find_nearest_power_2(0),
    )

    print ("Checkpoint 2")

    computation_dfg = simulator.schedule(computation_dfg, hw)

    print ("Checkpoint 3")

    simulator.transistor_size = hw.transistor_size  # in nm
    simulator.pitch = hw.pitch
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
    print ("Checkpoint 4")

    hardwareModel.un_allocate_all_in_use_elements(hw.netlist)
    simulator.simulate(computation_dfg, hw)
    print ("Checkpoint 5")
    simulator.calculate_edp(hw)
    print ("Checkpoint 6")

    # simulator.edp = simulator.edp.simplify()
    simulator.save_edp_to_file()
    print ("Checkpoint 7")

    return simulator.edp


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, filename="codesign_log_dir/symbolic_simulate.log"
    )
    parser = argparse.ArgumentParser(
        prog="Simulate",
        description="Runs a hardware simulation on a given benchmark and technology spec",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("benchmark", metavar="B", type=str)
    parser.add_argument("--notrace", action="store_true")
    parser.add_argument(
        "-c",
        "--architecture_config",
        type=str,
        default="aladdin_const_with_mem",
        help="Path to the architecture file (.gml)",
    )

    args = parser.parse_args()
    print(
        f"args: benchmark: {args.benchmark}, trace: {args.notrace}, architecture: {args.architecture_config}"
    )

    main()
