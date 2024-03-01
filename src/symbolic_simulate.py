# builtin modules
import math
import argparse
import os
import sys

# third party modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sympy import *
import sympy
import pyomo.environ as pyo
from pyomo.core.expr import Expr_if

# custom modules
import schedule
import dfg_algo
import hw_symbols
import sim_util
from hardwareModel import HardwareModel


class SymbolicHardwareSimulator:

    def __init__(self):
        self.cycles = 0
        self.id_to_node = {}
        self.path = os.getcwd()
        self.data_path = []
        self.node_intervals = []
        self.node_sum_energy = {}
        self.node_sum_cycles = {}
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
        self.initial_params = {}
        self.sim_cache = {}

    def symbolic_cycle_sim(self, computation_graph):
        """
        Don't need to do isomorphism check. Just need to generate topo generations.
        Since we'll get architectures from the forward pass. We'll assume that the architecture
        can execute the computation graph.
        """
        max_cycles = 0
        energy_sum = 0
        passive_power = 0
        # print(hw_need)

        if nx.is_empty(computation_graph):
            return max_cycles, energy_sum

        generations = list(nx.topological_generations(computation_graph))

        for node, node_data in computation_graph.nodes(data=True):
            active_energy = (
                hw_symbols.symbolic_power_active[node_data["function"]]
                * (hw_symbols.symbolic_latency_wc[node_data["function"]])
            )
            print("added active energy of", hw_symbols.symbolic_power_active[node_data["function"]]* (hw_symbols.symbolic_latency_wc[node_data["function"]] / hw_symbols.f), "for", node_data["function"])
            energy_sum += active_energy

        passive_power += hw_symbols.symbolic_power_passive[node_data["function"]]

        for start_node in generations[0]: 
            for end_node in generations[-1]:
                if start_node == end_node:
                    continue
                for path in nx.all_simple_paths(computation_graph, start_node, end_node):
                    path_latency = 0
                    print(f"path: {path}")
                    for node in path:
                        print(f"node: {node}")
                        print(f"computaiton_graph[{node}]: {computation_graph.nodes()[node]}")
                        path_latency += hw_symbols.symbolic_latency_wc[computation_graph.nodes()[node]["function"]]
                    max_cycles = 0.5 * (
                        max_cycles + path_latency + abs(max_cycles - path_latency)
                    )

        energy_sum += passive_power * max_cycles
        self.cycles += max_cycles
        return max_cycles, energy_sum

    def symbolic_simulate(self, cfg, cfg_node_to_hw_map, hw: HardwareModel):
        cur_node = cfg.entryblock
        
        cur_node = cur_node.exits[
            0
        ].target  # skip over the first node in the main cfg
        i = 0

        # focus on symbolizing the node_operations
        print("data path length:", len(self.data_path))
        while i < len(self.data_path):
            print(i)
            node_id = self.data_path[i][0]
            cur_node = self.id_to_node[node_id]
            self.node_intervals.append([node_id, [self.cycles, 0]])

            if not node_id in self.node_sum_energy:
                self.node_sum_energy[node_id] = (
                    0  # just reset because we will end up overwriting it
                )

            if not node_id in self.node_sum_cycles:
                self.node_sum_cycles[node_id] = 0
            iters = 0

            if self.unroll_at[cur_node.id]:
                j = i
                while True:
                    j += 1
                    if len(self.data_path) <= j:
                        break
                    next_node_id = self.data_path[j][0]
                    if next_node_id != node_id:
                        break
                    iters += 1
                i = (
                    j - 1
                )  # skip over loop iterations because we execute them all at once

            cache_index = (iters, node_id)

            node_energy, node_cycles = 0, 0

            # if I've seen this node before, no need to recalculate
            if cache_index in self.sim_cache:
                self.node_sum_cycles[node_id] += self.sim_cache[cache_index][0]
                self.node_sum_energy[node_id] += self.sim_cache[cache_index][1]
            else:
                computation_graph = cfg_node_to_hw_map[cur_node]
                print(f"computation graph: {computation_graph.nodes(data=True)}")
                # deprecated unrolling with graph representation. 
                # graph gets modified directly when we want to unroll.
                if self.unroll_at[cur_node.id]:
                    new_state = state.copy()
                    for op in state:
                        for j in range(iters):
                            new_state.append(op)
                    state = new_state

                max_cycles, energy_sum = self.symbolic_cycle_sim(
                    computation_graph
                )
                self.node_sum_energy[node_id] += energy_sum
                node_energy += energy_sum
                self.node_sum_cycles[node_id] += max_cycles
                node_cycles += max_cycles
                self.sim_cache[cache_index] = [node_cycles, node_energy]

            self.node_intervals[-1][1][1] = self.cycles
            i += 1
        print("done with simulation")

    def set_data_path(self):
        """
        This version of set data path doesn't count memory.
        Area of memory will not be accurate.
        Just use the same function from the concrete simulator?
        """

        with open(self.path + "/instrumented_files/output.txt", "r") as f:
            src = f.read()
            l = src.split("\n")
            for i in range(len(l)):
                l[i] = l[i].split()

            last_line = "-1"
            last_node = "-1"
            for item in l:
                if len(item) == 2 and (item[0] != last_node or item[1] == last_line):
                    last_node = item[0]
                    last_line = item[1]
                    self.data_path.append(item)

        print(f"memory needed: {self.memory_needed} bytes")

    def simulator_prep(self, benchmark, latency):
        """
        For this also, can we just use the same function from the concrete simulator?
        """
        cfg, graphs, self.unroll_at = dfg_algo.main_fn(self.path, benchmark)
        cfg_node_to_hw_map = schedule.schedule(cfg, graphs, latency)
        self.set_data_path()
        for node in cfg:
            self.id_to_node[str(node.id)] = node
        # print(self.id_to_node)
        return cfg, cfg_node_to_hw_map


def main(args_in):
    global args
    args = args_in
    print(f"Running symbolic simulator for {args.benchmark.split('/')[-1]}")

    simulator = SymbolicHardwareSimulator()

    # TODO: move this to a cli param
    hw = HardwareModel(cfg="aladdin_const_with_mem")

    cfg, cfg_node_to_hw_map = simulator.simulator_prep(args.benchmark, hw.latency)

    hw.init_memory(
        sim_util.find_nearest_power_2(simulator.memory_needed),
        sim_util.find_nearest_power_2(0),
    )

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


    simulator.symbolic_simulate(cfg, cfg_node_to_hw_map, hw)

    total_cycles = 0
    for node_id in simulator.node_sum_cycles:
        total_cycles += simulator.node_sum_cycles[node_id]

    total_power = 0
    for node_id in simulator.node_sum_energy:
        total_power += simulator.node_sum_energy[node_id]

    simulator.edp = total_cycles * total_power
    #simulator.edp = simulator.edp.simplify()
    st = str(simulator.edp)
    with open("sympy.txt", "w") as f:
        f.write(st)
    return simulator.edp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Simulate",
        description="Runs a hardware simulation on a given benchmark and technology spec",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("benchmark", metavar="B", type=str)
    parser.add_argument("--notrace", action="store_true")
    parser.add_argument("-a", "--area", type=float, help="Max Area of the chip in um^2")

    args = parser.parse_args()
    print(f"args: benchmark: {args.benchmark}, trace:{args.notrace}, area:{args.area}")

    main(args)
