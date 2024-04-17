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
import arch_search_util
from arch_search_util import generate_new_min_arch
from config_dicts import op2sym_map

MEMORY_SIZE = 1000000
state_graph_counter = 0

rng = np.random.default_rng()


class ConcreteSimulator:
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

        self.active_power_use[self.cycles] = 0
        self.mem_power_use.append(0)

        for n, node_data in computation_graph.nodes.data():

            scaling = 1
            if node_data["function"] in ["Buf", "MainMem"]:
                # active power should scale by size of the object being accessed.
                # all regs have the saem size, so no need to scale.
                scaling = node_data["size"]
            # OLD ACTIVE POWER CALCULATION
            # self.active_power_use[self.cycles] += (
            #     hw_spec.dynamic_power[node_data["function"]]
            #     * scaling
            #     * hw_spec.latency[node_data["function"]]
            # )
            # NEW ACTIVE ENERGY CALCULATION
            self.total_energy += (
                hw_spec.dynamic_power[node_data["function"]] * 1e-9
                * scaling
                * (hw_spec.latency[node_data["function"]] / hw_spec.frequency)
            )
            hw_spec.compute_operation_totals[node_data["function"]] += 1
        self.active_power_use[self.cycles] /= total_cycles

        # buf and mem nodes added in 'localize_memory'
        # remove them here
        for node, data in dict(
            filter(lambda x: x[1]["function"] == "MainMem" or x[1]["function"] == "Buf", computation_graph.nodes.data())
        ).items():
            computation_graph.remove_node(node)

        return self.active_power_use[self.cycles]

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

    def process_memory_operation(self, mem_op, mem_module: Memory):
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
            num_elem = 1
            if len(mem_op) > 3:
                dims = sim_util.get_dims(mem_op[3:])
                # print(f"dims: {dims}")
                num_elem = np.prod(dims)
            mem_module.malloc(var_name, size, dims=dims, elem_size=size // num_elem)
        elif status == "free":
            mem_module.free(var_name)

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

    def reset_internal_variables(self):
        self.cycles = 0
        # self.active_power_use = {}
        # self.passive_power_dissipation_rate = 0
        self.total_energy = 0

    def construct_fake_double_hw(self, hw):
        func_counts = hardwareModel.get_func_count(hw.netlist)
        fake_hw = nx.DiGraph()
        for node, num in func_counts.items():
            name = node if node != "MainMem" else "Mem"
            for i in range(0, 2*num):
                fake_hw.add_node(
                    f"{name}{i}",
                    function=node,
                    allocation="",
                )
        for node in hw.netlist.nodes:
            for child in hw.netlist.successors(node):
                child_func = hw.netlist.nodes[child]["function"]
                child_idx = hw.netlist.nodes[child]["idx"] + func_counts[child_func]
                new_child = f"{child_func}{child_idx}" if child_func != "MainMem" else f"Mem{child_idx}"
                fake_hw.add_edge(node, new_child)
        print(f"fake_hw: {fake_hw.nodes.data()}")
        return fake_hw

    def simulate(self, computation_dfg, hw):
        """
        Simulates one large DFG representing the whole computation.
        can I overload function names like this?

        TODO: This is a great candidate to cpp ify via cppyy
        """
        self.reset_internal_variables()

        frees = []
        mallocs = []

        mapping = (
            {}
        )  # map from cfg node to digraph matcher w/ hw -> dfg node allocations

        print(f"hw: {hw.netlist.nodes.data()}")
        fake_hw = self.construct_fake_double_hw(hw)
        for layer, nodes in enumerate(nx.topological_generations(fake_hw)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                fake_hw.nodes[node]["layer"] = layer

        # # Compute the multipartite_layout using the "layer" node attribute
        # pos = nx.multipartite_layout(fake_hw, subset_key="layer")

        # fig, ax = plt.subplots(figsize=(9,9))
        # nx.draw_networkx(fake_hw, pos=pos, ax=ax)
        # plt.show()
        counter = 0

        generations = reversed(
            list(nx.topological_generations(nx.reverse(computation_dfg)))
        )
        for gen in generations: # main loop over the computation graphs; 
            print(f"counter: {counter}, gen: {gen}")
            if "end" in gen:  # skip the end node (only thing in the last generation)
                break
            counter += 1
            temp_C = nx.DiGraph()
            for node in gen:
                temp_C.add_nodes_from([(node, computation_dfg.nodes[node])])
                for child in computation_dfg.successors(node):
                    if child not in temp_C.nodes:
                        temp_C.add_nodes_from([(child, computation_dfg.nodes[child])])
                    temp_C.add_edge(node, child)
                # for parent in computation_dfg.predecessors(node):
                #     if parent not in temp_C.nodes:
                #         temp_C.add_nodes_from([(parent, computation_dfg.nodes[parent])])
                #     temp_C.add_edge(parent, node)
                ## ================= ADD ACTIVE ENERGY CONSUMPTION =================
                scaling = 1
                node_data = computation_dfg.nodes[node]
                if node_data["function"] in ["Buf", "MainMem"]:
                    # active power should scale by size of the object being accessed.
                    # all regs have the same size, so no need to scale.
                    scaling = node_data["size"]
                if node_data["function"] == "stall" or node_data["function"] == "end":
                    continue
                self.total_energy += (
                    hw.dynamic_power[node_data["function"]] * 1e-9
                    * scaling
                    * (hw.latency[node_data["function"]] / hw.frequency)
                )
                hw.compute_operation_totals[node_data["function"]] += 1

            # print(f"temp_c: {temp_C.nodes.data()}")

            def matcher_func(n1, n2):
                res = n1["function"] == n2["function"] or n2["function"] == None or n2["function"] == "stall" or n2["function"] == "end"
                # if n2["function"] == None or n2["function"] == "stall":
                #     print(f"n1[function]: {n1['function']}, n2[function]: {n2['function']}, res: {res}")
                return res

            dgm = nx.isomorphism.DiGraphMatcher(
                fake_hw,
                temp_C,
                node_match=matcher_func,
            )

            is_monomorphic = dgm.subgraph_is_monomorphic()
            # print(f"is_monomorphic: {is_monomorphic}")
            if not is_monomorphic:
                print(f"fake_hw: {fake_hw.nodes.data()}")
                fig, ax = plt.subplots(1,2, figsize=(9, 9))
                pos = nx.multipartite_layout(temp_C, subset_key="layer")
                nx.draw_networkx(temp_C, pos=pos, ax=ax[0])
                ax[0].set_title("temp_C")
                pos = nx.multipartite_layout(fake_hw, subset_key="layer")
                nx.draw_networkx(fake_hw, pos=pos, ax=ax[1])
                ax[1].set_title("fake_hw")
                plt.show()

            mapping = dgm.subgraph_monomorphisms_iter().__next__()
            for hw_node, op in mapping.items(): # this binding stuff has to change now with the stalls. Implement as graph coloring outside of sim?
                if op in gen: # only bind ops in the current generation
                    hw.netlist.nodes[hw_node]["allocation"].append(op.split(";")[0])
                    computation_dfg.nodes[op]["allocation"] = hw_node
                    temp_C.nodes[op]["allocation"] = hw_node

        self.cycles = nx.dag_longest_path_length(computation_dfg)
        print(f"cycles: {self.cycles}")
        '''
        # iterate through nodes in data flow graph
        while i < len(self.data_path):
            
            # init vars for new node in cfg data path
            node_id = self.data_path[i][0]
            cur_node = self.id_to_node[node_id]
            self.node_intervals.append([node_id, [self.cycles, 0]])

            num_unroll_iterations = 0
            pattern_nodes = [node_id]
            pattern_mallocs, pattern_frees = [mallocs], [frees]
            other_mallocs, other_frees = [], []

            cur_node = self.id_to_node[pattern_nodes[idx]]

            mallocs = pattern_mallocs[idx]
            frees = pattern_frees[idx]
            for malloc in mallocs:
                self.process_memory_operation(
                    malloc,
                    list(hardwareModel.get_memory_node(hw.netlist).values())[0][
                        "memory_module"
                    ],
                )
            
            # print(f"hw_graph before verify_can_execute: {str(hw_graph)}")
            
            # === Count mem usage in this node === Do this shit some better way
            mem_in_use = self.get_mem_usage_of_dfg_node(
                hw_graph,
                list(hardwareModel.get_memory_node(hw.netlist).values())[0][
                    "memory_module"
                ],
            )

            self.max_regs_inuse = min(
                hardwareModel.num_nodes_with_func(hw.netlist, "Regs"),
                self.max_regs_inuse,
            )
            self.max_mem_inuse = max(self.max_mem_inuse, mem_in_use)
            # === End count mem usage ===

            # print(f"hw_graph before localize_memory: {str(hw_graph)}")
            self.localize_memory(hw, hw_graph)
            # print(f"hw_graph after localize_memory: {str(hw_graph)}")

            total_cycles = 0


            longest_path = nx.dag_longest_path(hw_graph)
            # print(f"longest_path: {longest_path}")
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

            # print(f"hw_graph before sim_cycle: {str(hw_graph)}")
            self.simulate_cycles(hw, hw_graph, total_cycles)
            # print(f"hw_graph after sim_cycle: {str(hw_graph)}")

            # hardwareModel.un_allocate_all_in_use_elements(hw.netlist)

            self.node_intervals[-1][1][1] = self.cycles
            for free in frees:
                self.process_memory_operation(
                    free,
                    list(hardwareModel.get_memory_node(hw.netlist).values())[0][
                        "memory_module"
                    ],
                )
            mallocs = []
            frees = []
        '''
        # add all passive power at the end.
        # This is done here for the dynamic allocation case where we don't know how many
        # compute elements we need until we run the program.
        print(f"total active energy: {self.total_energy} J")
        print(f"adding passive energy;")

        for elem_name, elem_data in dict(hw.netlist.nodes.data()).items():
            scaling = 1
            if elem_data["function"] in ["Regs", "Buf", "MainMem"]:
                scaling = elem_data["size"]

            self.total_energy += (
                hw.leakage_power[elem_data["function"]]
                * 1e-9
                * (self.cycles / hw.frequency)
                * scaling
            )

        for node in hw.netlist.nodes:
            hw.netlist.nodes[node]["allocation"] = len(
                hw.netlist.nodes[node]["allocation"]
            )

    def _simulate(self, cfg, cfg_node_to_hw_map, hw):
        self.reset_internal_variables()
        cur_node = cfg.entryblock

        # skip over the first node in the main cfg
        cur_node = cur_node.exits[0].target
        self.main_cfg = cfg

        i = 0
        frees = []
        mallocs = []

        mapping = {} # map from cfg node to digraph matcher w/ hw -> dfg node allocations 

        visited_node_ids = set()

        i, pattern_seek, max_iters = sim_util.find_next_data_path_index(
            self.data_path, i, mallocs, frees
        )
        # iterate through nodes in data flow graph
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

            num_unroll_iterations = 0
            pattern_nodes = [node_id]
            pattern_mallocs, pattern_frees = [mallocs], [frees]
            other_mallocs, other_frees = [], []

            # modify DFG to enable parallelization;
            # move this to a diff step: modify DFG before sim.
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
                    self.process_memory_operation(
                        malloc,
                        list(hardwareModel.get_memory_node(hw.netlist).values())[0][
                            "memory_module"
                        ],
                    )

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

                hw_graph = cfg_node_to_hw_map[cur_node]  # .copy()
                if node_id not in visited_node_ids:
                    # print(f"hw_graph before verify_can_execute: {str(hw_graph)}")
                    if not sim_util.verify_can_execute(hw_graph, hw.netlist, should_update_arch=False):
                        print(f"nodes: {hw_graph.nodes.data()}")
                        print(f"edges: {hw_graph.edges}")
                        nx.draw(hw_graph, with_labels=True)
                        plt.show()
                        raise Exception(
                            "hardware specification insufficient to run program"
                        )
                    # print(f"hw_graph after verify_can_execute: {str(hw_graph)}")
                else:
                    for node, data in hw_graph.nodes.data():
                        # print(f"appending node: {node.split(';')[0]} to {data['allocation']}")
                        hw.netlist.nodes[data["allocation"]]["allocation"].append(node.split(";")[0])
                        # print(f"allocation: {hw.netlist.nodes[data['allocation']]}")

                # === Count mem usage in this node ===
                mem_in_use = self.get_mem_usage_of_dfg_node(
                    hw_graph,
                    list(hardwareModel.get_memory_node(hw.netlist).values())[0][
                        "memory_module"
                    ],
                )

                self.max_regs_inuse = min(
                    hardwareModel.num_nodes_with_func(hw.netlist, "Regs"),
                    self.max_regs_inuse,
                )
                self.max_mem_inuse = max(self.max_mem_inuse, mem_in_use)
                # === End count mem usage ===

                # print(f"hw_graph before localize_memory: {str(hw_graph)}")
                sim_util.localize_memory(hw, hw_graph)
                # print(f"hw_graph after localize_memory: {str(hw_graph)}")

                total_cycles = 0

                # TODO: this doesn't account for latency of each element.
                # just assumes all have equal latency.
                longest_path = nx.dag_longest_path(hw_graph)
                # print(f"longest_path: {longest_path}")
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

                # print(f"hw_graph before sim_cycle: {str(hw_graph)}")
                self.simulate_cycles(hw, hw_graph, total_cycles)
                # print(f"hw_graph after sim_cycle: {str(hw_graph)}")

                # hardwareModel.un_allocate_all_in_use_elements(hw.netlist)

                idx += 1

            self.node_intervals[-1][1][1] = self.cycles
            for free in frees:
                self.process_memory_operation(
                    free,
                    list(hardwareModel.get_memory_node(hw.netlist).values())[0][
                        "memory_module"
                    ],
                )
            mallocs = []
            frees = []
            visited_node_ids.add(node_id)
            i = next_ind
            pattern_seek = pattern_seek_next
            max_iters = max_iters_next

        # add all passive power at the end.
        # This is done here for the dynamic allocation case where we don't know how many
        # compute elements we need until we run the program.
        for elem_name, elem_data in dict(hw.netlist.nodes.data()).items():
            scaling = 1
            if elem_data["function"] in ["Regs", "Buf", "MainMem"]:
                scaling = elem_data["size"]
            # OLD PASSIVE POWER CALCULATION
            self.passive_power_dissipation_rate += (
                hw.leakage_power[elem_data["function"]] * scaling
            )
            # NEW PASSIVE ENERGY CALCULATION
            self.total_energy += (
                hw.leakage_power[elem_data["function"]]*1e-9
                * (self.cycles / hw.frequency) * scaling
            )

        for node in hw.netlist.nodes:
            hw.netlist.nodes[node]['allocation'] = len(hw.netlist.nodes[node]['allocation'])

        # print("done with simulation")
        # Path(simulator.path + '/benchmarks/pictures/memory_graphs').mkdir(parents=True, exist_ok=True)
        # self.new_graph.gv_graph.render(self.path + '/benchmarks/pictures/memory_graphs/' + sys.argv[1][sys.argv[1].rfind('/')+1:], view = True)

    def set_data_path(self):
        """
        Keep track of variable values as they are updated.
        This is used primarily for index values in order to properly account for data dependencies.
        returns:
            data_path_vars: a list of dictionaries where each dictionary represents the variable values at a given node in the data path.
        """
        with open(self.path + "/instrumented_files/output.txt", "r") as f:
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

    def update_data_path(self, new_data_path):
        self.data_path = new_data_path
        for elem in self.data_path:
            if elem[0] not in self.unroll_at.keys():
                self.unroll_at[elem[0]] = False

    def calculate_average_power(self):
        self.avg_compute_power = 1e-6 * (
            np.mean(list(self.active_power_use.values()))
            + self.passive_power_dissipation_rate
        )

    def calculate_edp(self, hw):
        self.execution_time = self.cycles / hw.frequency # in seconds
        self.edp = self.total_energy * self.execution_time

    def compose_entire_computation_graph(self, cfg_node_to_dfg_map, data_path_vars, latency, plot=False):
        """
        Composes a large DFG from the smaller DFGs.
        Not currently used, doesn't handle register allocation very well.

        Parameters:
            cfg (CFG): The control flow graph of the program.
            cfg_node_to_hw_map (dict): A mapping of CFG nodes to hardware graphs represented by nx.DiGraphs.

        Returns:
            nx.DiGraph: The large DFG composed from the smaller DFGs.
        """
        computation_dfg = nx.DiGraph()
        curr_last_nodes = []
        i = sim_util.find_next_data_path_index(self.data_path, 0, [], [])[0]
        while i < len(self.data_path):
            # print(f"idx in compose: {i}")
            node_id = self.data_path[i][0]
            vars = data_path_vars[i]

            node = self.id_to_node[node_id]
            dfg = cfg_node_to_dfg_map[node]

            if nx.is_empty(dfg):
                i = sim_util.find_next_data_path_index(self.data_path, i + 1, [], [])[0]
                continue

            # plug in index values for array accesses
            for node in dfg.nodes:
                var, id = node.split(";")
                if len(var) == 1:
                    continue
                array = var.split("[")
                if len(array) == 1: # not an array
                    continue
                var_name = array[0]
                indices = [arr.split("]")[0] for arr in array[1:]]
                for idx in indices:
                    if idx in vars.keys():
                        var_name += f"[{vars[idx]}]"
                    else:
                        var_name += f"[{idx}]"
                var_name += f";{id}"
                dfg = nx.relabel_nodes(dfg, {node: var_name})

            generations = list(nx.topological_generations(dfg))
            found_alignment = sim_util.rename_nodes(computation_dfg, dfg, generations, curr_last_nodes)
            # print(f"hw_graph.nodes after rename: {hw_graph.nodes}")
            computation_dfg = nx.compose(computation_dfg, dfg)
            # computation_dfg.add_nodes_from(hw_graph.nodes(data=True))
            generations = list(nx.topological_generations(dfg))

            # if there's no alignment then independent, allow them to occur in parallel
            # MIGHT BE A BUG HERE

            # if not found_alignment:
            #     rand_first_node = rng.choice(generations[0])
            #     if len(curr_last_nodes) != 0:
            #         curr_last_node = rng.choice(curr_last_nodes)
            #         # print(f"adding edge from {curr_last_node} to {rand_first_node}")
            #         computation_dfg.add_edge(curr_last_node, rand_first_node)
            curr_last_nodes = generations[-1]

            i = sim_util.find_next_data_path_index(self.data_path, i + 1, [], [])[0]

        # create end node and connect all last nodes to it
        computation_dfg.add_node("end", function="end")
        generations = list(nx.topological_generations(computation_dfg))
        for node in generations[-1]:
            computation_dfg.add_edge(node, "end", weight=latency[computation_dfg.nodes[node]["function"]])

        print(f"done composing computation graph")

        if plot:
            sim_util.topological_layout_plot(computation_dfg, reverse=True)
        return computation_dfg

    def simulator_prep(self, benchmark, latency, hw_counts):
        """
        Creates CFG, and id_to_node
        params:
            benchmark: str with path to benchmark file
            latency: dict with latency of each compute element
        """
        cfg, graphs, self.unroll_at = dfg_algo.main_fn(self.path, benchmark)
        cfg_node_to_dfg_map = schedule.cfg_to_dfg(cfg, graphs, latency)
        data_path_vars = self.set_data_path()
        # print(f"data_path_vars:")
        # [print(f"{vars}, {path}") for vars, path in zip(data_path_vars, self.data_path)]
        for node in cfg:
            self.id_to_node[str(node.id)] = node
        # print(self.id_to_node)
        computation_dfg = self.compose_entire_computation_graph(cfg_node_to_dfg_map, data_path_vars, latency, plot=True)
        print(f"computation_dfg: {computation_dfg.nodes.data()}")
        print(f"edges: {computation_dfg.edges.data()}")
        schedule.schedule(computation_dfg, hw_counts)
        nx.draw(computation_dfg, with_labels=True)
        plt.show()
        sim_util.topological_layout_plot(computation_dfg, reverse=True)

        for layer, nodes in enumerate(reversed(list(nx.topological_generations(nx.reverse(computation_dfg))))):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                computation_dfg.nodes[node]["layer"] = layer

        # sim_util.topological_layout_plot(computation_dfg, reverse=True)

        return computation_dfg

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
            op2sym_map[compute_unit],
            None,
            op2sym_map[compute_unit],
        )


def main(args):
    print(f"Running simulator for {args.benchmark.split('/')[-1]}")
    simulator = ConcreteSimulator()

    # TODO: move this into cli arg
    hw = HardwareModel(cfg="aladdin_const_with_mem")

    computation_dfg = simulator.simulator_prep(args.benchmark, hw.latency, hw_counts=hardwareModel.get_func_count(hw.netlist))

    hw.init_memory(
        sim_util.find_nearest_power_2(simulator.memory_needed),
        sim_util.find_nearest_power_2(simulator.nvm_memory_needed),
    )

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
    for elem_name, elem_data in dict(hw.netlist.nodes.data()).items():
        # if elem == "Regs": continue
        # simulator.compute_element_to_node_id[elem] = []
        # looks like there's a lot of setup stuff that depends on the amount of hw allocated.
        simulator.init_new_compute_element(elem_data["function"])

    hardwareModel.un_allocate_all_in_use_elements(hw.netlist)
    data = simulator.simulate(computation_dfg, hw)

    area = hw.get_total_area()

    # print stats
    print("total number of cycles: ", simulator.cycles)
    avg_compute_power = simulator.total_energy / (simulator.cycles / hw.frequency) * 1e-3
    print(f"Avg compute Power: {avg_compute_power} mW")
    print(f"total energy {simulator.total_energy} nJ")

    print(f"on chip area: {area} um^2")
    print(f"off chip memory area: {hw.mem_area * 1e6} um^2")
    print(f"total area: {(area + hw.mem_area*1e6)} um^2")

    avg_mem_power = np.mean(simulator.mem_power_use) + hw.mem_leakage_power
    print(f"Avg mem Power: {avg_mem_power} mW")
    print(f"Total power: {avg_mem_power + avg_compute_power} mW")

    print("total volatile reads: ", simulator.reads)
    print("total volatile read size: ", simulator.total_read_size)
    print("total nvm reads: ", simulator.nvm_reads)
    print("total nvm read size: ", simulator.total_nvm_read_size)
    print("total writes: ", simulator.writes)
    print("total write size: ", simulator.total_write_size)

    print("max regs in use: ", simulator.max_regs_inuse)
    print(f"max memory in use: {simulator.max_mem_inuse} bytes")

    print("total operations computed: ", hw.compute_operation_totals)
    print(f"hw allocations: {[(n, hw.netlist.nodes[n]['allocation']) for n in hw.netlist.nodes]}")
    print(f"hw allocated: {dict(hw.netlist.nodes.data())}")

    # save some dump of data to json file
    names = args.benchmark.split("/")
    if not args.notrace:
        text = json.dumps(data, indent=4)
        Path(simulator.path + "/benchmarks/json_data").mkdir(
            parents=True, exist_ok=True
        )
        with open(simulator.path + "/benchmarks/json_data/" + names[-1], "w") as fh:
            fh.write(text)

    # plt.plot(
    #     list(simulator.active_power_use.keys()),
    #     list(simulator.active_power_use.values()),
    #     label="active power",
    # )
    # plt.title("power use for " + names[-1])
    # plt.xlabel("Cycle")
    # plt.ylabel("Power")
    # Path(simulator.path + "/benchmarks/power_plots").mkdir(parents=True, exist_ok=True)
    # plt.savefig("benchmarks/power_plots/power_use_" + names[-1] + ".pdf")
    # plt.clf()
    print("done!")
    return simulator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Simulate",
        description="Runs a hardware simulation on a given benchmark and technology spec",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("benchmark", metavar="B", type=str)
    parser.add_argument("--notrace", action="store_true")
    parser.add_argument("--architecture", type=str, help="Path to the architecture file (.gml)")
    # parser.add_argument("-s", "--archsearch", action=argparse.BooleanOptionalAction)
    # parser.add_argument("-a", "--area", type=float, help="Max Area of the chip in um^2")
    # parser.add_argument(
    #     "-b", "--bw", type=float, help="Compute - Memory Bandwidth in ??GB/s??"
    # )
    # parser.add_argument("-f", "--filepath", type=str, help="Path to the save new architecture file")
    args = parser.parse_args()
    print(
        f"args: benchmark: {args.benchmark}, trace:{args.notrace}, architecture:{args.architecture}"
    )

    main(args)
