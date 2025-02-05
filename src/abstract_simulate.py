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
        After adding wire parasitics to the scheduled dfg, update edge weights for longest path calculation.
            scheduled_dfg: nx.DiGraph representing the scheduled graph
        """
        for gen in list(nx.topological_generations(scheduled_dfg)):
            for node in gen:
                for parent in scheduled_dfg.predecessors(node):
                    edge = (parent, node)
                    scheduled_dfg.edges[edge]["weight"] = (scheduled_dfg.nodes[parent]["cost"] 
                                                            + scheduled_dfg.edges[edge]["cost"]) # update edge weight with parasitic
    
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
        self.update_schedule_with_parasitics(scheduled_dfg)


    def verify_register_chain_order(self, graph, reg_ops_sorted, func_instances):
        reg_chains_test = [[] for instance in func_instances["Regs"]]
        for name, op in reg_ops_sorted:
            for i in range(len(func_instances["Regs"])):
                if graph.nodes[op]["allocation"] == func_instances["Regs"][i]:
                    reg_chains_test[i].append(op)
        for chain in reg_chains_test:
            for i in range(len(chain) - 1):
                first_op, next_op = graph.nodes[chain[i]], graph.nodes[chain[i+1]]
                assert np.round(first_op["end_time"],3) <= np.round(next_op["start_time"], 3), f"register allocator assigned overlapping ops to the same register. \
                                                                                                first op is {chain[i]}:{first_op}, next is {chain[i+1]}:{next_op}. This may be due to nonsensical write after write dependencies, or maybe not ;)."

    def schedule(self, computation_dfg, hw, schedule_type="greedy"):
        """
        Schedule the computation graph.
        params:
            computation_dfg: nx.DiGraph representing the computation graph; does not have buffer
                and memory nodes explicit.
            hw: HardwareModel object
        """
        hw.init_buffers_and_registers()

        hw_counts = hardwareModel.get_func_count(hw.netlist)
        schedule.pre_schedule(computation_dfg, hw.netlist, hw.latency)
        copy = computation_dfg.copy()


        topo_order_by_elem, extra_constraints = schedule.get_topological_order(copy, "Regs", hw_counts, hw.netlist)
        schedule.sdc_schedule(copy, topo_order_by_elem, extra_constraints)
        logger.info("completed initial schedule")

        # after first scheduling, perform register allocation using linear scan algorithm
        op_allocation, reg_ops_sorted = schedule.register_allocate(copy, hw_counts, hw.netlist)

        logger.info(f"reg ops sorted: {[op[1] for op in reg_ops_sorted]}")

        topologocial_order_arith = []
        seen = set()
        for _, op in reg_ops_sorted:
            children = copy.successors(op)
            for child in children:
                if child not in seen and copy.nodes[child]["function"] not in ["Regs", "Buf", "MainMem"]:
                    topologocial_order_arith.append(child)
                    seen.add(child)
        logger.info(f"processing element topological order: {topologocial_order_arith}")

        """hw.latency["Buf"] = 2
        hw.latency["MainMem"] = 4"""
        func_instances = {func: list(filter(lambda x: hw.netlist.nodes[x]["function"] == func, hw.netlist.nodes())) for func in hw_counts}
        
        self.verify_register_chain_order(copy, reg_ops_sorted, func_instances)
        #buf_chain = []
        reg_chains, buf_chain = schedule.add_higher_memory_accesses_to_scheduled_graph(copy, hw_counts, hw.netlist, op_allocation, "Regs", "Buf", reg_ops_sorted, hw.latency["Regs"], hw.latency["Buf"])
        buf_allocation = {buf_op[1]: 0 for buf_op in buf_chain}
        logger.info(f"reg chains: {reg_chains}")
        #print([buf_op[1] for buf_op in buf_chain])
        topo_order_by_elem, _ = schedule.get_topological_order(copy, "Buf", hw_counts, hw.netlist, reg_chains, topologocial_order_arith, [buf_op[1] for buf_op in buf_chain])
        #print(topo_order_by_elem["Regs0"])
        #print(topo_order_by_elem["Regs1"])
        schedule.sdc_schedule(copy, topo_order_by_elem)

        buf_chains, mem_chain = schedule.add_higher_memory_accesses_to_scheduled_graph(copy, hw_counts, hw.netlist, buf_allocation, "Buf", "MainMem", buf_chain, hw.latency["Buf"], hw.latency["MainMem"])
        #print(buf_chains[0])
        #print([mem_op[1] for mem_op in mem_chain])
        topo_order_by_elem, _ = schedule.get_topological_order(copy, "MainMem", hw_counts, hw.netlist, reg_chains, topologocial_order_arith, buf_chains[0], [mem_op[1] for mem_op in mem_chain])
        self.add_parasitics_to_scheduled_dfg(copy, hw.parasitic_graph)
        self.resource_edge_graph = schedule.sdc_schedule(copy, topo_order_by_elem, add_resource_edges=True)
        logger.info(f"longest path: {nx.dag_longest_path(self.resource_edge_graph)}")
        logger.info(f"longest path length: {nx.dag_longest_path_length(self.resource_edge_graph)}")
        

        return copy
