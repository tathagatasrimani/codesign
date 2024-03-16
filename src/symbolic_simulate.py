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
import hardwareModel
from hardwareModel import HardwareModel

rng = np.random.default_rng()


class SymbolicSimulator:

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
        self.node_sum_energy = {}
        self.node_sum_cycles = {}
        self.node_sum_cycles_ceil = {}
        self.cycles_ceil = 0
        self.cycles = 0

    def cycle_sim(self, computation_graph):
        """
        Don't need to do isomorphism check. Just need to generate topo generations.
        Since we'll get architectures from the forward pass. We'll assume that the architecture
        can execute the computation graph.
        """
        max_cycles = 0
        max_cycles_ceil = 0
        energy_sum = 0

        if nx.is_empty(computation_graph):
            return max_cycles, max_cycles_ceil, energy_sum

        generations = list(nx.topological_generations(computation_graph))

        for node, node_data in computation_graph.nodes(data=True):
            active_energy = hw_symbols.symbolic_power_active[node_data["function"]] * (
                hw_symbols.symbolic_latency_wc[node_data["function"]]
            )
            # print("added active energy of", hw_symbols.symbolic_power_active[node_data["function"]]* (hw_symbols.symbolic_latency_wc[node_data["function"]] / hw_symbols.f), "for", node_data["function"])
            energy_sum += active_energy

        for start_node in generations[0]:
            for end_node in generations[-1]:
                if start_node == end_node:
                    continue
                for path in nx.all_simple_paths(
                    computation_graph, start_node, end_node
                ):
                    path_latency = 0
                    path_latency_ceil = 0
                    # print(f"path: {path}")
                    for node in path:
                        # print(f"node: {node}")
                        # print(f"computation_graph[{node}]: {computation_graph.nodes()[node]}")
                        # THIS PATH LATENCY MAY OR MAY NOT USE CYCLE TIME OR WALL CLOCK TIME DUE TO SOLVER INSTABILITY
                        path_latency += hw_symbols.symbolic_latency_cyc[
                            computation_graph.nodes()[node]["function"]
                        ]
                        # THIS PATH LATENCY USES CYCLE TIME AS A REFERENCE FOR WHAT THE TRUE EDP IS
                        path_latency_ceil += hw_symbols.symbolic_latency_cyc[
                            computation_graph.nodes()[node]["function"]
                        ]
                    max_cycles = 0.5 * (
                        max_cycles + path_latency + abs(max_cycles - path_latency)
                    )
                    max_cycles_ceil = 0.5 * (
                        max_cycles_ceil + path_latency_ceil + abs(max_cycles_ceil - path_latency_ceil)
                    )

        # remove Mem and Buf from computation graph, so it doesn't get duplicated
        # if added again in next iteration.
        for node, data in dict(
            filter(
                lambda x: x[1]["function"] == "MainMem" or x[1]["function"] == "Buf",
                computation_graph.nodes.data(),
            )
        ).items():
            computation_graph.remove_node(node)

        self.cycles += max_cycles
        self.cycles_ceil += max_cycles_ceil
        return max_cycles, max_cycles_ceil, energy_sum

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
                )  # size will be negative because cache miss.
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
            if data["function"] in ["Regs", "Buf", "MainMem"]:
                scaling = data["size"]
            passive_power += (
                hw_symbols.symbolic_power_passive[data["function"]] * scaling
            )
        return passive_power * total_execution_time

    def simulate(self, cfg, cfg_node_to_hw_map, hw: HardwareModel):
        self.reset_internal_variables()
        hw_symbols.update_symbolic_passive_power(hw.R_off_on_ratio)
        cur_node = cfg.entryblock

        cur_node = cur_node.exits[0].target  # skip over the first node in the main cfg
        i = 0

        sim_cache = {}


        while i < len(self.data_path):
            # print(f"i: {i}")
            (
                next_ind,
                _,
                _,
            ) = sim_util.find_next_data_path_index(self.data_path, i + 1, [], [])

            node_id = self.data_path[i][0]
            cur_node = self.id_to_node[node_id]
            self.node_intervals.append([node_id, [self.cycles, 0]])

            if not node_id in self.node_sum_energy:
                self.node_sum_energy[node_id] = (
                    0  # just reset because we will end up overwriting it
                )

            if not node_id in self.node_sum_cycles:
                self.node_sum_cycles[node_id] = 0
                self.node_sum_cycles_ceil[node_id] = 0
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

            # if I've seen this node before, no need to recalculate
            if cache_index in sim_cache:
                self.node_sum_cycles[node_id] += sim_cache[cache_index][0]
                self.node_sum_cycles_ceil[node_id] += sim_cache[cache_index][1]
                self.node_sum_energy[node_id] += sim_cache[cache_index][2]
            else:
                computation_graph = cfg_node_to_hw_map[cur_node]
                # print(f"computation graph: {computation_graph.nodes(data=True)}")
                sim_util.verify_can_execute(computation_graph, hw.netlist)
                # deprecated unrolling with graph representation.
                # graph gets modified directly when we want to unroll.
                if self.unroll_at[cur_node.id]:
                    new_state = state.copy()
                    for op in state:
                        for j in range(iters):
                            new_state.append(op)
                    state = new_state
                self.localize_memory(hw, computation_graph)

                max_cycles, max_cycles_ceil, energy_sum = self.cycle_sim(computation_graph)
                self.node_sum_energy[node_id] += energy_sum
                self.node_sum_cycles[node_id] += max_cycles
                self.node_sum_cycles_ceil[node_id] += max_cycles_ceil
                sim_cache[cache_index] = [max_cycles, max_cycles_ceil, energy_sum]

            self.node_intervals[-1][1][1] = self.cycles
            i = next_ind
            if i == len(self.data_path):
                break
        # print("done with simulation")

    def set_data_path(self):
        with open(self.path + "/instrumented_files/output.txt", "r") as f:
            src = f.read()
            l = src.split("\n")
            split_lines = [l_.split() for l_ in l]  # idk what's happening here.

            last_line = "-1"
            last_node = "-1"
            valid_names = set()
            nvm_vars = {}

            # count reads and writes on first pass.
            for i in range(len(split_lines)):
                item = split_lines[i]
                if len(item) < 2:
                    continue
                if item[0] == "malloc":
                    valid_names.add(item[2])
                if item[-2] != "Read" and item[-2] != "Write":
                    continue
                var_name = item[0]
                if var_name not in valid_names and "NVM" not in var_name:
                    continue
                if item[-2] == "Read":
                    if "NVM" in item[0]:
                        self.nvm_reads += 1
                        self.total_nvm_read_size += int(item[-1])
                        if item[0] not in nvm_vars.keys():
                            nvm_vars[item[0]] = int(item[-1])
                        else:
                            nvm_vars[item[0]] = max(nvm_vars[item[0]], int(item[-1]))
                    else:
                        self.reads += 1
                        self.total_read_size += int(item[-1])
                        self.where_to_free[var_name] = i + 1
                else:
                    self.writes += 1
                    self.total_write_size += int(item[-1])
                    self.where_to_free[var_name] = i

            # second pass, construct trace that simulator follows.
            for i in range(len(split_lines)):
                item = split_lines[i]
                vars_to_pop = []
                for var_name in self.where_to_free:
                    if self.where_to_free[var_name] == i:
                        var_size = self.vars_allocated[var_name]
                        # f_new.write("free " + str(var_size) + " " + var_name + "\n")
                        self.data_path.append(["free", str(var_size), var_name])
                        vars_to_pop.append(var_name)
                        self.cur_memory_size -= var_size
                        self.vars_allocated.pop(var_name)
                for var_name in vars_to_pop:
                    self.where_to_free.pop(var_name)
                if len(item) == 2 and (item[0] != last_node or item[1] == last_line):
                    last_node = item[0]
                    last_line = item[1]
                    self.data_path.append(item)
                elif len(item) == 1 and item[0].startswith("pattern_seek"):
                    self.data_path.append(item)
                elif len(item) >= 3 and item[0] == "malloc":
                    if item[2] in self.vars_allocated:
                        if int(item[1]) == self.vars_allocated[item[2]]:
                            continue
                        # else:
                        # f_new.write("free " + str(self.vars_allocated[item[2]]) + " " + var_name + "\n")
                        # f_new.write(l[i] + '\n')
                        self.cur_memory_size -= int(self.vars_allocated[item[2]])
                    self.data_path.append(item)
                    self.vars_allocated[item[2]] = int(item[1])
                    # print(self.vars_allocated)
                    self.cur_memory_size += int(item[1])
                    self.memory_needed = max(self.memory_needed, self.cur_memory_size)
        # print(f"data_path: {self.data_path}")
        self.nvm_memory_needed = sum(nvm_vars.values())
        print(f"memory needed: {self.memory_needed} bytes")
        print(f"nvm memory needed: {self.nvm_memory_needed} bytes")

    def update_data_path(self, new_data_path):
        self.data_path = new_data_path
        for elem in self.data_path:
            if elem[0] not in self.unroll_at.keys():
                self.unroll_at[elem[0]] = False

    def simulator_prep(self, benchmark, latency):
        """
        Creates CFG, and id_to_node
        params:
            benchmark: str with path to benchmark file
            latency: dict with latency of each compute element
        """
        cfg, graphs, self.unroll_at = dfg_algo.main_fn(self.path, benchmark)
        cfg_node_to_hw_map = schedule.cfg_to_dfg(cfg, graphs, latency)
        self.set_data_path()
        for node in cfg:
            self.id_to_node[str(node.id)] = node
        # print(self.id_to_node)
        return cfg, cfg_node_to_hw_map

    def calculate_edp(self, hw):
        total_cycles = sum(self.node_sum_cycles.values())
        total_cycles_ceil = sum(self.node_sum_cycles_ceil.values())
        total_execution_time = total_cycles
        total_execution_time_ceil = total_cycles_ceil
        total_active_energy = sum(self.node_sum_energy.values())
        total_passive_energy = self.passive_energy_dissipation(hw, total_execution_time)
        total_passive_energy_ceil = self.passive_energy_dissipation(hw, total_execution_time_ceil)
        self.edp = total_execution_time * (total_active_energy + total_passive_energy)
        self.edp_ceil = total_execution_time_ceil * (total_active_energy + total_passive_energy_ceil)

    def save_edp_to_file(self):
        st = str(self.edp)
        with open("sympy.txt", "w") as f:
            f.write(st)


def main():
    print(f"Running symbolic simulator for {args.benchmark.split('/')[-1]}")

    simulator = SymbolicSimulator()

    # TODO: move this to a cli param
    hw = HardwareModel(cfg="aladdin_const_with_mem")
    hw.get_optimization_params_from_tech_params()


    cfg, cfg_node_to_hw_map = simulator.simulator_prep(args.benchmark, hw.latency)

    hw.init_memory(
        sim_util.find_nearest_power_2(simulator.memory_needed),
        sim_util.find_nearest_power_2(0),
    )
    hardwareModel.un_allocate_all_in_use_elements(hw.netlist)
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

    simulator.simulate(cfg, cfg_node_to_hw_map, hw)
    simulator.calculate_edp(hw)

    # simulator.edp = simulator.edp.simplify()
    simulator.save_edp_to_file()

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

    main()
