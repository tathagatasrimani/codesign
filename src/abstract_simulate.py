# builtin modules
import math
import argparse
import os
import sys

# third party modules
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sympy

# custom modules
from .memory import Memory
from . import schedule
from . import dfg_algo
from . import hw_symbols
from . import sim_util
from . import hardwareModel
from .hardwareModel import HardwareModel
from .global_constants import SEED

class AbstractSimulator:
    def set_data_path(self):
        """
        Keep track of variable values as they are updated.
        This is used primarily for index values in order to properly account for data dependencies.
        returns:
            data_path_vars: a list of dictionaries where each dictionary represents the variable values at a given node in the data path.
        """
        with open(self.path + "/src/instrumented_files/output.txt", "r") as f:
            src = f.read()
            l = src.split("\n")
            split_lines = [l_.split() for l_ in l]  # separate by whitespace

            last_line = "-1"
            last_node = "-1"
            valid_names = set()
            nvm_vars = {}
            data_path_vars = []

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

            vars = {}
            self.data_path.append([""])
            # second pass, construct trace that simulator follows.
            for i in range(len(split_lines)):
                item = split_lines[i]
                vars_to_pop = []
                for var_name in self.where_to_free:
                    if self.where_to_free[var_name] == i:
                        var_size = self.vars_allocated[var_name]
                        # f_new.write("free " + str(var_size) + " " + var_name + "\n")
                        self.data_path.append(["free", str(var_size), var_name])
                        data_path_vars.append(vars.copy())
                        vars = {}
                        vars_to_pop.append(var_name)
                        self.cur_memory_size -= var_size
                        self.vars_allocated.pop(var_name)
                for var_name in vars_to_pop:
                    self.where_to_free.pop(var_name)
                if len(item) == 2 and (item[0] != last_node or item[1] == last_line):
                    last_node = item[0]
                    last_line = item[1]
                    self.data_path.append(item)
                    data_path_vars.append(vars.copy())
                    vars = {}
                elif len(item) == 1 and item[0].startswith("pattern_seek"):
                    self.data_path.append(item)
                    data_path_vars.append(vars.copy())
                    vars = {}
                elif len(item) >= 3 and item[0] == "malloc":
                    if item[2] in self.vars_allocated:
                        if int(item[1]) == self.vars_allocated[item[2]]:
                            continue
                        # else:
                        # f_new.write("free " + str(self.vars_allocated[item[2]]) + " " + var_name + "\n")
                        # f_new.write(l[i] + '\n')
                        self.cur_memory_size -= int(self.vars_allocated[item[2]])
                    self.data_path.append(item)
                    data_path_vars.append(vars.copy())
                    vars = {}
                    self.vars_allocated[item[2]] = int(item[1])
                    # print(self.vars_allocated)
                    self.cur_memory_size += int(item[1])
                    self.total_malloc_size += int(item[1])
                    self.memory_needed = max(self.memory_needed, self.cur_memory_size)
                elif len(item) == 4:
                    if item[1].isnumeric():
                        vars[item[0]] = int(item[1])
            data_path_vars.append(vars)
        # print(f"data_path: {self.data_path}")
        self.nvm_memory_needed = sum(nvm_vars.values())
        print(f"memory needed: {self.memory_needed} bytes")
        print(f"nvm memory needed: {self.nvm_memory_needed} bytes")
        return data_path_vars

    def simulator_prep(self, benchmark, latency):
        """
        Creates CFG, and id_to_node
        params:
            benchmark: str with path to benchmark file
            latency: dict with latency of each compute element
        """
        cfg, graphs, self.unroll_at = dfg_algo.main_fn(self.path, benchmark)
        cfg_node_to_dfg_map = schedule.cfg_to_dfg(cfg, graphs, latency)
        """i = 0
        for cfg_node in cfg_node_to_dfg_map:
            nx.write_gml(cfg_node_to_dfg_map[cfg_node], sim_util.get_latest_log_dir()+f"/computation_dfg_{i}.gml")
            i += 1"""
        data_path_vars = self.set_data_path()
        for node in cfg:
            self.id_to_node[str(node.id)] = node
        computation_dfg, mallocs = sim_util.compose_entire_computation_graph(
            cfg_node_to_dfg_map,
            self.id_to_node,
            self.data_path,
            data_path_vars,
            latency,
            plot=False,
        )

        return computation_dfg, mallocs
    
    def update_schedule_with_parasitics(self, scheduled_dfg):
        """
        After adding wire parasitics to the scheduled dfg, update start and end times for each node,
        and edge weights for longest path calculation. To do this, we use the resource edge graph to
        ensure that we take resource dependencies into account while updating these latencies
        params:
            scheduled_dfg: nx.DiGraph representing the scheduled graph
        """
        for gen in list(nx.topological_generations(self.resource_edge_graph)):
            for node in gen:
                for parent in self.resource_edge_graph.predecessors(node):
                    edge = (parent, node)
                    if edge in scheduled_dfg.edges:
                        scheduled_dfg.edges[edge]["weight"] = (scheduled_dfg.nodes[parent]["cost"] 
                                                               + scheduled_dfg.edges[edge]["cost"]) # update edge weight with parasitic
                        self.resource_edge_graph.edges[edge]["weight"] = (scheduled_dfg.nodes[parent]["cost"] 
                                                               + scheduled_dfg.edges[edge]["cost"])
                    scheduled_dfg.nodes[node]["start_time"] = max(scheduled_dfg.nodes[node]["start_time"], 
                                                                  scheduled_dfg.nodes[parent]["start_time"] + 
                                                                  self.resource_edge_graph.edges[edge]["weight"]) # start time may be later now
                if scheduled_dfg.nodes[node]["function"] != "end":
                    scheduled_dfg.nodes[node]["end_time"] = scheduled_dfg.nodes[node]["start_time"] + scheduled_dfg.nodes[node]["cost"] # update end time
                    self.resource_edge_graph.nodes[node]["start_time"] = scheduled_dfg.nodes[node]["start_time"] # keep consistent with resource edge graph
                    self.resource_edge_graph.nodes[node]["end_time"] = scheduled_dfg.nodes[node]["end_time"]
    
    def add_parasitics_to_scheduled_dfg(self, scheduled_dfg, parasitic_graph):
        """
        Add wire parasitics from OpenROAD to computation dfg
        params:
            scheduled_dfg: nx.DiGraph representing the scheduled graph
            parasitic_graph: nx.DiGraph representing wire parasitics from OpenROAD
        """

        def check_in_parasitic_graph(node_data):
            return ("allocation" in node_data
                and node_data["allocation"] != ""
                and "Mem" not in node_data["allocation"]
                and "Buf" not in node_data["allocation"])
        def accumulate_delay_over_segments(parasitic_edge):
            net_delay = 0
            res_instance = 0
            for y in range(len(parasitic_edge["net_cap"])): # doing second order RC
                res_instance += parasitic_edge["net_res"][y]
                cap_instance = parasitic_edge["net_cap"][y]
                net_delay += res_instance * cap_instance * 1e-3
            return net_delay
        def update_net_delay(node_name_prev, node_name):
            parasitic_edge = parasitic_graph[node_name_prev][node_name]
            net_delay = 0
            if isinstance(parasitic_edge["net_cap"], list):
                net_delay = accumulate_delay_over_segments(parasitic_edge)
            else:
                net_delay = (
                    parasitic_edge["net_cap"]
                    * parasitic_edge["net_res"]
                    * 1e-3
                )  # pico -> nano
            return net_delay
        # wire latency
        for edge in scheduled_dfg.edges:
            prev_node, node = edge
            node_data = scheduled_dfg.nodes[node]
            node_data_prev = scheduled_dfg.nodes[prev_node]
            node_name = node_data["allocation"]
            node_name_prev = node_data_prev["allocation"]
            net_delay = 0
            if (check_in_parasitic_graph(node_data) and 
                    check_in_parasitic_graph(node_data_prev)):  
                if "Regs" in node_name_prev or "Regs" in node_name: # again, 16 bit
                    max_delay = 0
                    #finding the longest time and adding that
                    for x in range(16):
                        if "Regs" in node_name_prev:
                            node_name_prev = node_name_prev + "_" + str(x)
                        elif "Regs" in node_name:
                            node_name = node_name + "_" + str(x)
                        if parasitic_graph.has_edge(node_name_prev, node_name):
                            net_delay = update_net_delay(node_name_prev, node_name)
                            if max_delay < net_delay:
                                max_delay = net_delay
                    net_delay = max_delay
                else: 
                    if parasitic_graph.has_edge(node_name_prev, node_name):
                        net_delay = update_net_delay(node_name_prev, node_name)
            scheduled_dfg.edges[edge]["cost"] = net_delay
            self.resource_edge_graph.edges[edge]["cost"] = net_delay
        self.update_schedule_with_parasitics(scheduled_dfg)

    def schedule(self, computation_dfg, hw, schedule_type="greedy", prune_func=sim_util.prune_buffer_and_mem_nodes):
        """
        Schedule the computation graph.
        params:
            computation_dfg: nx.DiGraph representing the computation graph; does not have buffer
                and memory nodes explicit.
            hw: HardwareModel object
        """
        hw_counts = hardwareModel.get_func_count(hw.netlist)
        schedule.pre_schedule(computation_dfg, hw.netlist, hw.latency)
        copy = computation_dfg.copy()
        #print("HW Netlist: ", hw.netlist.nodes.data())
        if schedule_type == "greedy":
            schedule.greedy_schedule(copy, hw_counts, hw.netlist)
        elif schedule_type == "sdc":
            schedule.sdc_schedule(copy, hw_counts, hw.netlist)

        for layer, nodes in enumerate(
            reversed(list(nx.topological_generations(nx.reverse(computation_dfg))))
        ):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                copy.nodes[node]["layer"] = layer
        copy = sim_util.add_cache_mem_access_to_dfg(
            copy, hw.latency["Buf"], hw.latency["MainMem"]
        )
        if schedule_type == "greedy":
            schedule.greedy_schedule(copy, hw_counts, hw.netlist)
            copy = prune_func(copy, hw.netlist)
        elif schedule_type == "sdc":
            schedule.sdc_schedule(copy, hw_counts, hw.netlist)
            copy = prune_func(copy, hw.netlist, sdc_schedule=True)
            # Once we have pruned memory/buffer nodes, critical path may have changed. So we need to redo the scheduling
            self.resource_edge_graph = schedule.sdc_schedule(copy, hw_counts, hw.netlist, add_resource_edges=True)
            self.add_parasitics_to_scheduled_dfg(copy, hw.parasitic_graph)
            #print("longest path:", nx.dag_longest_path(self.resource_edge_graph))
            #print("longest path length:", nx.dag_longest_path_length(self.resource_edge_graph))
        

        return copy
