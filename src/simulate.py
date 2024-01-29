# builtin modules
import sys
import os
from collections import deque
import math
import json
from pathlib import Path
import argparse
import ast

# third party modules
import matplotlib.pyplot as plt
import numpy as np
import graphviz as gv
import networkx as nx

# custom modules
from memory import Memory
import schedule
import dfg_algo
import hardwareModel
from hardwareModel import HardwareModel
import hardwareModel
import sim_util
import arch_search
from arch_search import generate_new_fc_arch

MEMORY_SIZE = 1000000
state_graph_counter = 0


class HardwareSimulator:
    def __init__(self):
        self.memory_module = Memory(MEMORY_SIZE)
        self.data = {}
        self.cycles = 0  # counter for number of cycles
        self.main_cfg = None
        self.id_to_node = {}
        self.path = os.getcwd()
        self.data_path = []
        self.active_power_use = {}
        self.mem_power_use = []
        self.node_intervals = []
        self.node_avg_power = {}
        self.unroll_at = {}
        self.vars_allocated = {}
        self.where_to_free = {}
        self.memory_needed = 0
        self.nvm_memory_needed = 0
        self.cur_memory_size = 0
        self.new_graph = None  # this is just for visualization
        self.mem_layers = 0
        self.transistor_size = 0
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

    def verify_can_execute(self, computation_graph, hw_spec):
        """
        Determines whether or not the computation graph can be executed on the netlist
        specified in hw_spec.

        The way this works, if I have 10 addition operations to do in this node of the DFG, and 20 adders
        it should always allocate 10 adders to this node. <- TODO: VERIFY EXPERIMENTALLY

        Counts total size of all variables accessed in this node of the dfg

        See TODO below about a fix to generalize this.

        Parameters:
            computation_graph - nx.DiGraph of operations to be executed.
            hw_spec (HardwareModel): An object representing the current hardware allocation and specifications.

        Returns:
            dict: A dictionary representing the hardware requirements for the operations.
        """

        ## Duplicate all regs in hw_spec, so that there is a separate node for input reg A and output reg A.
        ## This doesn't scale for arbitrarily complex hw_graphs representing arbitrarily complex operations.
        ## This needs to be changed to some topological ordering based technique, but this was a quicker hack.
        ## TODO: Use some topo ordering based technique to identify whether the netlist
        ## can do the computation prescribed by the hw_graph.
        netlist_copy = hw_spec.netlist.copy()
        for reg_node in hardwareModel.get_nodes_with_func(netlist_copy, "Regs"):
            netlist_copy.add_node(
                reg_node + "_copy",
                type="pe",
                function="Regs",
                in_use=False,
                idx=hw_spec.netlist.nodes[reg_node]["idx"],
            )
            edges = list(netlist_copy.in_edges(reg_node))
            for edge in edges:
                # print(f"edge: {edge}")
                netlist_copy.add_edge(edge[0], reg_node + "_copy")
                netlist_copy.remove_edge(*edge)
            # netlist_copy.add_edge(*, reg_node + "_copy")
            # netlist_copy.remove_edge(*, reg_node)

        dgm = nx.isomorphism.DiGraphMatcher(
            netlist_copy,
            computation_graph,
            node_match=lambda n1, n2: n1["function"] == n2["function"]
            or n2["function"]
            == None,  # hw_graph can have no ops, but netlist should not
        )
        if not dgm.subgraph_is_monomorphic():
            raise Exception("hardware specification insufficient to run program")
        # this needs to be used for resource allocation
        monomorphism = dgm.subgraph_monomorphisms_iter()

    def simulate_cycles(self, hw_spec, computation_graph, total_cycles):
        """
        Simulates the operation of hardware over a number of cycles.

        Parameters:
            hw_spec (HardwareModel): The hardware model providing the specific architecture being simulated.
            computation_graph: Nx.DiGraph of operations to be executed.
            total_cycles (int): The maximum number of cycles to simulate.

        Returns:
            int: The sum of power used by nodes in all cycles.
        """
        if total_cycles == 0:
            return 0

        node_power_sum = 0

        self.active_power_use[self.cycles] = 0
        self.mem_power_use.append(0)

        print(f"\nat cycle {self.cycles}, summing total acticve power use.")
        for n, node_data in computation_graph.nodes.data():
            print(
                f"adding power {1e-6 * hw_spec.dynamic_power[node_data['function']]} for node: {n}"
            )
            self.active_power_use[self.cycles] += (
                hw_spec.dynamic_power[node_data["function"]]
                * hw_spec.latency[node_data["function"]]
            )
            hw_spec.compute_operation_totals[node_data["function"]] += 1
        self.active_power_use[self.cycles] /= total_cycles

        print(
            f"at cycle {self.cycles}, total active power use: {1e-6 * self.active_power_use[self.cycles]}"
        )

        return self.active_power_use[self.cycles]

    def get_var_size(self, var_name):
        """
        Only called on Regs nodes that are parents / children of an operation.
        Checks the size of the variable in the mem heap that this register points to.

        Parameters:
            neighbor: The register node to be processed

        Returns:
            int: The size of the variable this register is storing.
        """
        mem_size = 0
        bracket_ind = var_name.find("[")
        bracket_count = sim_util.get_matching_bracket_count(var_name)
        if bracket_ind != -1:
            var_name = var_name[:bracket_ind]

        if var_name in self.memory_module.locations:
            mem_loc = self.memory_module.locations[var_name]
            mem_size = mem_loc.size
            for i in range(bracket_count):
                mem_size /= mem_loc.dims[i]

        return mem_size

    def get_mem_usage_of_dfg_node(self, hw_graph):
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
                mem_in_use += self.get_var_size(node.split(";")[0])

        return mem_in_use

    def process_memory_operation(self, mem_op):
        """
        Processes a memory operation, handling memory allocation or deallocation based on the operation type.

        This function interprets and acts on memory operations (like allocating or freeing memory)
        within the hardware simulation. It modifies the state of the memory module according to the
        specified operation, updating the memory allocation status as necessary.

        Parameters:
        - mem_op (list): A list representing a memory operation. The first element is the operation type
        ('malloc' or 'free'), the second element is the size of the memory block, and subsequent elements
         provide additional context or dimensions for the operation.

        Usage:
        - If `mem_op` is a 'malloc' operation, the function allocates memory of the specified size and
            potentially with specified dimensions.
        - If `mem_op` is a 'free' operation, the function deallocates the memory associated with the
             given variable name.

        This function is typically called during the simulation process to dynamically manage
        memory as the simulated program executes different operations that require memory allocation
        and deallocation.
        """
        var_name = mem_op[2]
        size = int(mem_op[1])
        status = mem_op[0]
        if status == "malloc":
            dims = []
            if len(mem_op) > 3:
                dims = sim_util.get_dims(mem_op[3:])
            self.memory_module.malloc(var_name, size, dims=dims)
        elif status == "free":
            self.memory_module.free(var_name)

    def visualize_graph(self, operations):
        """
        Deprecated?
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
                hardwareModel.op2sym_map[op.operation],
                None,
                hardwareModel.op2sym_map[op.operation],
            )
            for parent in op.parents:
                if parent.operation:  # and parent.operation != "Regs":
                    parent_id = dfg_algo.set_id()
                    sim_util.make_node(
                        state_graph,
                        parent_id,
                        hardwareModel.op2sym_map[parent.operation],
                        None,
                        hardwareModel.op2sym_map[parent.operation],
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
            Path(self.path + "/benchmarks/pictures/state_graphs/").mkdir(
                parents=True, exist_ok=True
            )
            state_graph_viz.render(
                self.path
                + "/benchmarks/pictures/state_graphs/"
                + args.benchmark.split("/")[-1].split(".")[0]
                + "_"
                + str(state_graph_counter),
                view=True,
            )
        state_graph_counter += 1  # what does this do, can I get rid of it?

    def simulate(self, cfg, cfg_node_to_hw_map, hw, first):
        global state_graph_counter
        cur_node = cfg.entryblock
        if first:
            # skip over the first node in the main cfg
            cur_node = cur_node.exits[0].target
            self.main_cfg = cfg

        i = 0
        frees = []
        mallocs = []

        print(f"data path: {self.data_path}")

        i, pattern_seek, max_iters = sim_util.find_next_data_path_index(
            self.data_path, i, mallocs, frees
        )
        # iterate through nodes in data dependency graph
        while i < len(self.data_path):
            (
                next_ind,
                pattern_seek_next,
                max_iters_next,
            ) = sim_util.find_next_data_path_index(
                self.data_path, i + 1, mallocs, frees
            )

            if i == len(self.data_path):
                break

            # init vars for new node in cfg data path
            node_id = self.data_path[i][0]
            cur_node = self.id_to_node[node_id]
            self.node_intervals.append([node_id, [self.cycles, 0]])
            self.node_avg_power[
                node_id
            ] = 0  # just reset because we will end up overwriting it
            start_cycles = self.cycles  # for calculating average power
            num_unroll_iterations = 0
            pattern_nodes = [node_id]
            pattern_mallocs, pattern_frees = [mallocs], [frees]
            other_mallocs, other_frees = [], []

            # modify DFG to enable parallelization:
            if pattern_seek:
                # print("entering pattern seek")
                pattern_seek = False
                j = next_ind
                while (not pattern_seek_next) and (j + 1 < len(self.data_path)):
                    # print("hello")
                    next_node_id = self.data_path[j][0]
                    pattern_nodes.append(next_node_id)
                    pattern_seek = pattern_seek_next
                    j, pattern_seek_next, discard = sim_util.find_next_data_path_index(
                        self.data_path, j + 1, other_mallocs, other_frees
                    )
                    pattern_frees.append(other_frees)
                    pattern_mallocs.append(other_mallocs)
                    other_frees = []
                    other_mallocs = []
                if pattern_seek_next:
                    next_ind = j
                else:
                    pattern_nodes = [node_id]
                # print("found pattern")
                pattern_ind = 0
                while j + 1 < len(self.data_path):
                    next_node_id = self.data_path[j][0]
                    if next_node_id != pattern_nodes[pattern_ind]:
                        break
                    pattern_ind += 1
                    pattern_seek = pattern_seek_next
                    j, pattern_seek_next, discard = sim_util.find_next_data_path_index(
                        self.data_path, j + 1, [], []
                    )
                    if pattern_ind == len(pattern_nodes):
                        num_unroll_iterations += 1
                        pattern_ind = 0
                        next_ind = j
                        if max_iters > 1 and num_unroll_iterations + 1 == max_iters:
                            break
                if num_unroll_iterations > 0:
                    pattern_seek = True
            elif self.unroll_at[cur_node.id]:
                print(f"found unroll at node: {cur_node.id}")
                print(
                    f"ops:{[str(op) for op in operations for operations in cfg_node_to_hw_map[cur_node]]}"
                )
                # unroll the node: find the next node that is not the same as the current node
                # and count consecutive number of times this node appears in the data path
                j = next_ind
                while j + 1 < len(self.data_path):
                    next_node_id = self.data_path[j][0]
                    if next_node_id != node_id:
                        break
                    num_unroll_iterations += 1
                    pattern_seek = pattern_seek_next
                    j, pattern_seek_next, discard = sim_util.find_next_data_path_index(
                        self.data_path, j + 1, [], []
                    )
                next_ind = j

            idx = 0
            # this happens once if no pattern seeking; else happens number of times = number of unique nodes in pattern
            while idx < len(pattern_nodes):
                cur_node = self.id_to_node[pattern_nodes[idx]]
                mallocs = pattern_mallocs[idx]
                frees = pattern_frees[idx]
                for malloc in mallocs:
                    self.process_memory_operation(malloc)

                # try to unroll after pattern_seeking
                node_iters = num_unroll_iterations
                if self.unroll_at[cur_node.id]:
                    while (
                        idx + 1 != len(pattern_nodes)
                        and pattern_nodes[idx + 1] == pattern_nodes[idx]
                    ):
                        idx += 1
                        node_iters += num_unroll_iterations
                if self.unroll_at[cur_node.id] or pattern_seek:
                    for _ in range(node_iters):
                        graph = dfg_algo.dfg_per_node(cur_node)
                        node_ops = schedule.schedule_one_node(graph)
                        for k in range(len(node_ops)):
                            cfg_node_to_hw_map[cur_node][k] = (
                                cfg_node_to_hw_map[cur_node][k] + node_ops[k]
                            )

                hw_graph = cfg_node_to_hw_map[cur_node]

                self.verify_can_execute(hw_graph, hw)

                # === Count mem usage in this node ===
                mem_in_use = self.get_mem_usage_of_dfg_node(hw_graph)

                self.max_regs_inuse = min(
                    hardwareModel.num_nodes_with_func(hw.netlist, "Regs"),
                    self.max_regs_inuse,
                )
                self.max_mem_inuse = max(self.max_mem_inuse, mem_in_use)
                # === End count mem usage ===

                total_cycles = 0

                # TODO: this doesn't account for latency of each element.
                # just assumes all have equal latency.
                longest_path = nx.dag_longest_path(hw_graph)
                print(longest_path)
                try:
                    total_cycles = sum(
                        [
                            math.ceil(hw.latency[hw_graph.nodes[n]["function"]])
                            for n in longest_path
                        ]
                    )  # + math.ceil(hw.latency["Regs"]) # add latency of Reg because initial read reg cost is not included in longest path.
                except:
                    total_cycles = 0

                self.cycles += total_cycles

                self.node_avg_power[node_id] += self.simulate_cycles(
                    hw, hw_graph, total_cycles
                )
                hardwareModel.un_allocate_all_in_use_elements(hw.netlist)
                # ============================================
                # === END UPDATE =============================
                # ============================================

                idx += 1

            # should we be dividing by num cycles here. I think this is why P is going down as N increases
            # we don't actually use node_avg_power. what we care about is power_use
            if self.cycles - start_cycles > 0:
                self.node_avg_power[node_id] /= self.cycles - start_cycles
            self.node_intervals[-1][1][1] = self.cycles
            for free in frees:
                # print(free)
                if free[2] in self.memory_module.locations:
                    self.process_memory_operation(free)
            mallocs = []
            frees = []
            i = next_ind
            pattern_seek = pattern_seek_next
            max_iters = max_iters_next

        # add all passive power at the end.
        # This is done here for the dynamic allocation case where we don't know how many
        # compute elements we need until we run the program.
        self.passive_power_dissipation_rate = 0
        for elem_name, elem_data in dict(hw.netlist.nodes.data()).items():
            self.passive_power_dissipation_rate += hw.leakage_power[
                elem_data["function"]
            ]

        # for c in range(self.cycles):
        #     self.power_use[c] += passive_power
        #     self.mem_power_use[c] += hw.mem_leakage_power

        print("done with simulation")
        # Path(simulator.path + '/benchmarks/pictures/memory_graphs').mkdir(parents=True, exist_ok=True)
        # self.new_graph.gv_graph.render(self.path + '/benchmarks/pictures/memory_graphs/' + sys.argv[1][sys.argv[1].rfind('/')+1:], view = True)
        return self.data

    def set_data_path(self):
        with open(self.path + "/instrumented_files/output.txt", "r") as f:
            # f_new = open(self.path + '/instrumented_files/output_free.txt', 'w+')
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
                # if not (len(item) >= 3 and item[0] == "malloc" and item[2] not in self.vars_allocated):
                #     f_new.write(l[i] + '\n')
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

    def simulator_prep(self, benchmark, latency):
        """
        Creates CFG, and id_to_node
        params:
            benchmark: str with path to benchmark file
            latency: dict with latency of each compute element
        """
        cfg, graphs, self.unroll_at = dfg_algo.main_fn(self.path, benchmark)
        cfg_node_to_hw_map = schedule.schedule(cfg, graphs, latency)
        self.set_data_path()
        for node in cfg:
            self.id_to_node[str(node.id)] = node
        # print(self.id_to_node)
        return cfg, graphs, cfg_node_to_hw_map

    def init_new_compute_element(self, compute_unit):
        """
        Adds to some graph shit and adds ids and stuff
        params:
            compute_unit: str with std cell name, eg 'Add', 'Eq', 'LShift', etc.
        """
        compute_id = dfg_algo.set_id()
        sim_util.make_node(
            self.new_graph,
            compute_id,
            hardwareModel.op2sym_map[compute_unit],
            None,
            hardwareModel.op2sym_map[compute_unit],
        )
        # self.compute_element_neighbors[compute_id] = set()
        # self.compute_element_to_node_id[compute_unit].append(compute_id)


def main():
    print(f"Running simulator for {args.benchmark.split('/')[-1]}")
    simulator = HardwareSimulator()
    hw = HardwareModel(cfg="aladdin_const")

    cfg, graphs, cfg_node_to_hw_map = simulator.simulator_prep(
        args.benchmark, hw.latency
    )
    # print(f"cfg_node_to_hw_map:\n{cfg_node_to_hw_map}")

    # these two lines are hardcoded for now, remove and put them in HardwareModel properly
    hw.netlist = nx.DiGraph()
    hw.dynamic_allocation = True
    ## END 2 lines
    if hw.dynamic_allocation:
        arch_search.generate_new_fc_arch(
            cfg, hw, cfg_node_to_hw_map, simulator.data_path, simulator.id_to_node
        )

    nx.draw(hw.netlist, with_labels=True)
    plt.show()

    simulator.transistor_size = hw.transistor_size  # in nm
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
    for elem_name, elem_data in dict(hw.netlist.nodes.data()).items():
        # if elem == "Regs": continue
        # simulator.compute_element_to_node_id[elem] = []
        # looks like there's a lot of setup stuff that depends on the amount of hw allocated.
        simulator.init_new_compute_element(elem_data["function"])

    data = simulator.simulate(cfg, cfg_node_to_hw_map, hw, True)  # graphs

    area = 0
    for elem_name, elem_data in dict(hw.netlist.nodes.data()).items():
        area += hw.area[elem_data["function"]]
    print(f"compute area: {area * 1e-6} um^2")
    print(f"memory area: {hw.mem_area * 1e6} um^2")
    print(f"total area: {(area*1e-6 + hw.mem_area*1e6)} um^2")

    # print stats
    print("total number of cycles: ", simulator.cycles)
    avg_compute_power = 1e-6 * (
        np.mean(list(simulator.active_power_use.values()))
        + simulator.passive_power_dissipation_rate
    )
    print(f"Avg compute Power: {avg_compute_power} mW")
    # print(f"total energy {sum(simulator.power_use)} nJ")
    avg_mem_power = np.mean(simulator.mem_power_use) + hw.mem_leakage_power
    print(f"Avg mem Power: {avg_mem_power} mW")
    print(f"Total power: {avg_mem_power + avg_compute_power} mW")
    print("total volatile reads: ", simulator.reads)
    print("total volatile read size: ", simulator.total_read_size)
    print("total nvm reads: ", simulator.nvm_reads)
    print("total nvm read size: ", simulator.total_nvm_read_size)
    print("total writes: ", simulator.writes)
    print("total write size: ", simulator.total_write_size)
    print("total operations computed: ", hw.compute_operation_totals)
    print(f"hw allocated: {dict(hw.netlist.nodes.data())}")
    print("max regs in use: ", simulator.max_regs_inuse)
    print(f"max memory in use: {simulator.max_mem_inuse} bytes")

    # save some dump of data to json file
    names = args.benchmark.split("/")
    if not args.notrace:
        text = json.dumps(data, indent=4)
        Path(simulator.path + "/benchmarks/json_data").mkdir(
            parents=True, exist_ok=True
        )
        with open(simulator.path + "/benchmarks/json_data/" + names[-1], "w") as fh:
            fh.write(text)

    plt.plot(
        list(simulator.active_power_use.keys()),
        list(simulator.active_power_use.values()),
        label="active power",
    )
    plt.title("power use for " + names[-1])
    plt.xlabel("Cycle")
    plt.ylabel("Power")
    Path(simulator.path + "/benchmarks/power_plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("benchmarks/power_plots/power_use_" + names[-1] + ".pdf")
    plt.clf()
    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Simulate",
        description="Runs a hardware simulation on a given benchmark and technology spec",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("benchmark", metavar="B", type=str)
    parser.add_argument("--notrace", action="store_true")

    args = parser.parse_args()
    # print(f"args: {args.benchmark}, {args.notrace}")

    main()
