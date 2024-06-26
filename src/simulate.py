# builtin modules
import os
import math
import json
from pathlib import Path
import argparse
import logging
logger = logging.getLogger(__name__)

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
        self.active_energy = 0
        self.passive_energy = 0

    def simulate_cycles(self, hw_spec, computation_graph, total_cycles):
        """
        DEPRECATED
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
            # if node_data["function"] in ["Buf", "MainMem"]:
            #     # active power should scale by size of the object being accessed.
            #     # all regs have the saem size, so no need to scale.
            #     scaling = node_data["size"]
            # if node_data["function"] == "MainMem":
            #     print(
            #         f" found a mem object in active energy calc, adding scaling factor: {scaling}"
            #     )
            self.total_energy += (
                hw_spec.dynamic_power[node_data["function"]]
                * 1e-9
                * scaling
                * (hw_spec.latency[node_data["function"]] / hw_spec.frequency)
            )

            if node_data["function"] in ["Buf", "MainMem"]:
                # active power should scale by size of the object being accessed.
                # all regs have the saem size, so no need to scale.
                scaling = node_data["size"]

                self.total_energy += (
                    (hw_spec.dynamic_energy[node_data["function"]]["Read"] 
                    + hw_spec.dynamic_energy[node_data["function"]]["Write"]) / 2 # avg of read and write
                    * 1e-9
                    * scaling
                )
                if node_data["function"] == "MainMem":
                    self.total_energy += (
                    hw_spec.dynamic_power["OffChipIO"] * 1e-3   # CACTI IO uses mW
                    * scaling
                    * hw_spec.latency["OffChipIO"]
                    )
            else:
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
            filter(
                lambda x: x[1]["function"] == "MainMem" or x[1]["function"] == "Buf",
                computation_graph.nodes.data(),
            )
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
            cache_hit = hw.netlist.nodes[data["allocation"]]["var"] == var_name # no need to refetch from mem if var already in reg.
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
            size = buf[1]["memory_module"].read(
                var_name
            )  
            buf_idx = len(
                list(
                    filter(
                        lambda x: x[1]["function"] == "Buf", hw_graph.nodes.data()
                    )
                )
            )

            # print(f"in localize memory; var: {var_name}; size: {size}")
            # add multiple bufs and mems, not just one. add a new one for each cache miss.
            # this is required to properly count active power consumption.
            # active power added once for each node in computation_graph.
            # what about latency????

            if cache_hit:   # only add the Buf
                hw_graph.add_node(
                    f"Buf{buf_idx}", function="Buf", allocation=buf[0], size=size
                )
                hw_graph.add_edge(f"Buf{buf_idx}", node, function="Mem")

            else:           # add Buf and Mem
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
        self.active_energy = 0
        self.passive_energy = 0
        self.total_energy = 0

    def construct_fake_double_hw(self, hw):
        func_counts = hardwareModel.get_func_count(hw.netlist)
        fake_hw = nx.DiGraph()
        for node, num in func_counts.items():
            name = node if node != "MainMem" else "Mem"
            for i in range(0, 2 * num):
                fake_hw.add_node(
                    f"{name}{i}",
                    function=node,
                    allocation="",
                )
        for node in hw.netlist.nodes:
            for child in hw.netlist.successors(node):
                child_func = hw.netlist.nodes[child]["function"]
                child_idx = hw.netlist.nodes[child]["idx"] + func_counts[child_func]
                new_child = (
                    f"{child_func}{child_idx}"
                    if child_func != "MainMem"
                    else f"Mem{child_idx}"
                )
                fake_hw.add_edge(node, new_child)
        return fake_hw

    def simulate(self, computation_dfg, hw):
        """
        Simulates one large DFG representing the whole computation.
        can I overload function names like this?

        TODO: Doesn't yet add in all buff and mem operations
        TODO: This is a great candidate to cpp ify via cppyy
        """
        self.reset_internal_variables()

        fake_hw = self.construct_fake_double_hw(hw)
        for layer, nodes in enumerate(nx.topological_generations(fake_hw)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                fake_hw.nodes[node]["layer"] = layer

        counter = 0

        generations = reversed(
            list(nx.topological_generations(nx.reverse(computation_dfg)))
        )
        for gen in generations:  # main loop over the computation graphs;
            if "end" in gen:  # skip the end node (only thing in the last generation)
                break
            counter += 1
            temp_C = nx.DiGraph()
            for node in gen:
                logger.info(f"node -> {node}: {computation_dfg.nodes[node]}")
                child_added = False
                temp_C.add_nodes_from([(node, computation_dfg.nodes[node])])
                for child in computation_dfg.successors(node):
                    if computation_dfg.nodes[node]["function"] == "stall" and (
                        computation_dfg.nodes[child]["function"] == "stall"
                        or computation_dfg.nodes[child]["function"] == "end"
                    ):
                        continue

                    if child not in temp_C.nodes:
                        temp_C.add_nodes_from([(child, computation_dfg.nodes[child])])
                        child_added = True
                    temp_C.add_edge(node, child)

                ## ================= ADD ACTIVE ENERGY CONSUMPTION =================
                scaling = 1
                node_data = computation_dfg.nodes[node]
                if node_data["function"] == "stall" or node_data["function"] == "end":
                    continue
                if node_data["function"] in ["Buf", "MainMem"]:
                    # active power should scale by size of the object being accessed.
                    # all regs have the same size, so no need to scale.
                    scaling = node_data["size"]
                    logger.info(f"scaling: {scaling}")
                    energy = (
                        (hw.dynamic_energy[node_data["function"]]["Read"] 
                        + hw.dynamic_energy[node_data["function"]]["Write"]) / 2 # avg of read and write
                        * 1e-9
                        * scaling
                    )
                
                    if (node_data["function"] == "MainMem"):
                        energy += (
                        hw.dynamic_power["OffChipIO"] * 1e-9
                        * scaling
                        * hw.latency["OffChipIO"]
                        )
                else:
                    energy = (
                        hw.dynamic_power[node_data["function"]] * 1e-9 # W
                        * scaling
                        * hw.latency[node_data["function"]] # ns
                    )

                self.active_energy += energy
                hw.compute_operation_totals[node_data["function"]] += 1

            def matcher_func(n1, n2):
                res = (
                    n1["function"] == n2["function"]
                    or n2["function"] == None
                    or n2["function"] == "stall"
                    or n2["function"] == "end"
                )

                return res

        self.cycles = nx.dag_longest_path_length(computation_dfg)
        longest_path = nx.dag_longest_path(computation_dfg)
        logger.info(f"longest path: {longest_path}")
        logger.info(f"longest path length: {self.cycles}")

        for elem_name, elem_data in dict(hw.netlist.nodes.data()).items():
            scaling = 1
            if elem_data["function"] in ["MainMem"]:
                scaling = elem_data["size"]

            self.passive_energy += (
                hw.leakage_power[elem_data["function"]] * 1e-9 * self.cycles * scaling
            )

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

    def calculate_edp(self):
        self.execution_time = self.cycles # in seconds
        self.total_energy = self.active_energy + self.passive_energy
        self.edp = self.total_energy * self.execution_time

    def simulator_prep(self, benchmark, latency):
        """
        Creates CFG, and id_to_node
        params:
            benchmark: str with path to benchmark file
            latency: dict with latency of each compute element
        """
        cfg, graphs, self.unroll_at = dfg_algo.main_fn(self.path, benchmark)
        cfg_node_to_dfg_map = schedule.cfg_to_dfg(cfg, graphs, latency)
        data_path_vars = self.set_data_path()

        for node in cfg:
            self.id_to_node[str(node.id)] = node
        computation_dfg = sim_util.compose_entire_computation_graph(
            cfg_node_to_dfg_map,
            self.id_to_node,
            self.data_path,
            data_path_vars,
            latency,
            plot=False,
        )

        return computation_dfg

    def schedule(self, computation_dfg, hw):
        hw_counts = hardwareModel.get_func_count(hw.netlist)
        copy = computation_dfg.copy()
        schedule.greedy_schedule(copy, hw_counts, hw.netlist)

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
        schedule.greedy_schedule(copy, hw_counts, hw.netlist)
        copy = sim_util.prune_buffer_and_mem_nodes(copy, hw.netlist)

        return copy

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

    hw = HardwareModel(cfg=args.architecture_config)

    computation_dfg = simulator.simulator_prep(args.benchmark, hw.latency)

    hw.init_memory(
        sim_util.find_nearest_power_2(simulator.memory_needed),
        sim_util.find_nearest_power_2(simulator.nvm_memory_needed),
    )
    computation_dfg = simulator.schedule(computation_dfg, hw)

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
    data = simulator.simulate(computation_dfg, hw)
    simulator.calculate_edp()

    area = hw.get_total_area()

    # print stats
    print("total number of cycles: ", simulator.cycles)
    print(f"execution time: {simulator.execution_time} ns")
    print(f"total energy {simulator.total_energy} nJ")

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
        Path(simulator.path + "/benchmarks/json_data").mkdir(
            parents=True, exist_ok=True
        )
        with open(simulator.path + "/benchmarks/json_data/" + names[-1], "w") as fh:
            fh.write(text)

    print("done!")
    return simulator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="codesign_log_dir/simulate.log")

    parser = argparse.ArgumentParser(
        prog="Simulate",
        description="Runs a hardware simulation on a given benchmark and technology spec",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("benchmark", metavar="B", type=str)
    parser.add_argument("--notrace", action="store_true")
    parser.add_argument(
        "--architecture_config",
        type=str,
        default="aladdin_const_with_mem",
        help="Path to the architecture file (.gml)",
    )
    args = parser.parse_args()

    print(
        f"args: benchmark: {args.benchmark}, trace: {args.notrace}, architecture: {args.architecture_config}"
    )

    main(args)
