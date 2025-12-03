import logging
import os
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)

import networkx as nx
import re
import networkx as nx
import xml.etree.ElementTree as ET

from src.forward_pass import llvm_ir_parse
from src.forward_pass import vitis_create_netlist
from src import sim_util

DEBUG = False

def log_info(msg):
    if DEBUG:
        logger.info(msg)

module_map = sim_util.get_module_map()

class VariableTracker:
    def __init__(self):
        # Maps variable name to (current_node_id, timestamp)
        self.variable_states = {}
        self.timestamp = 0
    
    def get_next_timestamp(self):
        self.timestamp += 1
        return self.timestamp
    
    def get_read_node_name(self, var_name):
        """Get the current node name for reading a variable"""
        if var_name not in self.variable_states:
            return self.update_write_node(var_name)
        return self.variable_states[var_name]
    
    def update_write_node(self, var_name):
        """Update the variable state after a write"""
        ts = self.get_next_timestamp()
        self.variable_states[var_name] = f"{var_name};{ts}"
        return self.variable_states[var_name]

class VariableDB:
    def __init__(self):
        self.VariableTrackers = {}
    
    def get_read_node_name(self, var_name):
        if var_name not in self.VariableTrackers:
            self.VariableTrackers[var_name] = VariableTracker()
        return self.VariableTrackers[var_name].get_read_node_name(var_name)
    
    def update_write_node(self, var_name):
        if var_name not in self.VariableTrackers:
            self.VariableTrackers[var_name] = VariableTracker()
        return self.VariableTrackers[var_name].update_write_node(var_name)

class ResourceTracker:
    def __init__(self, is_core, core_info, resource_name):
        self.resource_name = resource_name
        self.num_ports = core_info["Ports"] if is_core else 1
        if self.num_ports == 0:
            self.num_ports = 1 # Observed that a FIFO operation has 0 ports, ignore this
        self.core_info = core_info
        self.reset_resource()
    
    def log_resource_usage(self, var_name):
        self.resource_usage[self.resource_idx].append(var_name)
        #log_info(f"Logging resource usage for {var_name} on port {self.resource_idx} of {self.resource_name}: {self.resource_usage[self.resource_idx]}")
        self.resource_idx = (self.resource_idx + 1) % self.num_ports

    def get_latest_resource_usage(self):
        assert self.resource_idx < self.num_ports, f"Resource index {self.resource_idx} is out of bounds for {self.resource_name}: {self.core_info}"
        return self.resource_usage[self.resource_idx][-1] if self.resource_usage[self.resource_idx] else None

    def reset_resource(self):
        self.resource_usage = {idx: [] for idx in range(self.num_ports)}
        self.resource_idx = 0

class ResourceDB:
    def __init__(self):
        self.resources = {}

    def check_resource_added(self, resource_name):
        return resource_name in self.resources
    
    def add_resource(self, resource_name, is_core=False, core_info=None):
        self.resources[resource_name] = ResourceTracker(is_core, core_info, resource_name)
    
    def log_resource_usage(self, resource_name, var_name):
        assert resource_name in self.resources, f"Resource {resource_name} not found"
        self.resources[resource_name].log_resource_usage(var_name)

    def get_latest_resource_usage(self, resource_name):
        assert resource_name in self.resources, f"Resource {resource_name} not found"
        return self.resources[resource_name].get_latest_resource_usage()

    def reset_resources(self):
        for resource_name in self.resources:
            self.resources[resource_name].reset_resource()

def get_core_info(line):
    core_info = {}
    pattern = r'<II = (\d+)>'
    match = re.search(pattern, line)
    core_info["II"] = int(match.group(1))

    pattern = r'<Ports = (\d+)>'
    match = re.search(pattern, line)
    core_info["Ports"] = int(match.group(1)) if match else 1
    return core_info

def match_pattern_in_line(line, pattern):
    match = re.search(pattern, line)
    return match

def get_rsc_mapping(netlist_file):
    netlist = nx.read_gml(netlist_file)
    netlist_op_dest_to_node = {}
    for n, d in netlist.nodes(data=True):
        ## extract name and bind->opset fields
        name = d.get('name')
        bind = d.get('bind', {})
        opset = bind.get('opset')
        ## remove the slash and everything after it from the opset
        if not opset:
            #log_info(f"opset is None for {n},{d}")
            continue
        if '/' in opset:
            opsets = opset.split()
            for opset in opsets:
                netlist_op_dest_to_node[opset.split('/')[0]] = name
        else:
            netlist_op_dest_to_node[opset.strip()] = name
    #log_info(f"logging netlist_op_dest_to_node")
    #for op, node in netlist_op_dest_to_node.items():
        #log_info(f"op: {op}, node: {node}")
    return netlist_op_dest_to_node

class LoopInfo:
    def __init__(self, loop_info):
        self.loop_name = loop_info["Loop Name"]
        self.min = loop_info["min"]
        self.max = loop_info["max"]
        self.latency = loop_info["Latency"]
        self.achieved = loop_info["achieved"]
        self.target = loop_info["target"]
        self.count = loop_info["Count"]
        self.pipelined = loop_info["Pipelined"]
        self.pipeline_states = loop_info["pipeline_states"] if "pipeline_states" in loop_info else []
        self.loop_states = [] # to be assigned in StateObject
        self.is_top_loop = loop_info["is_top_loop"]
        self.sub_loops = []

    def __str__(self):
        return f"LoopInfo(loop_name={self.loop_name}, min={self.min}, max={self.max}, latency={self.latency}, achieved={self.achieved}, target={self.target}, count={self.count}, pipelined={self.pipelined}, pipeline_states={self.pipeline_states}, loop_states={self.loop_states}, is_top_loop={self.is_top_loop}, sub_loops={self.sub_loops})"
    def __repr__(self):
        return self.__str__()

class DataFlowGraph:
    def __init__(self, clk_period, no_rsc_allowed_ops, allowed_functions, build_dir, basic_block_name, name, resource_mapping, states_structure, G, G_standard, G_standard_with_wire_ops, resource_db, variable_db, resource_delays_only=False, num_iters=1):
        self.clk_period = clk_period
        self.allowed_functions = allowed_functions
        self.build_dir = build_dir
        self.basic_block_name = basic_block_name
        self.name = name
        self.G = G
        self.G_standard = G_standard
        self.G_standard_with_wire_ops = G_standard_with_wire_ops
        self.resource_db = resource_db
        self.variable_db = variable_db
        self.no_rsc_allowed_ops = no_rsc_allowed_ops
        self.resource_mapping = resource_mapping
        self.cycle_nodes = []
        self.non_cycle_nodes = []
        self.states_structure = states_structure
        self.loop_dfgs = {name: {} for name in [loop for loop in self.states_structure.loops]}
        self.state_ops = {}
        self.state_transitions = {}
        self.num_iters = num_iters
        self.resource_delays_only = resource_delays_only

    def track_resource_usage(self, node):
        if self.G.nodes[node]["node_type"] == "op" and self.G.nodes[node]["core_inst"] != "N/A":
            if self.G.nodes[node]["rsc_name_unique"].startswith("N/A"):
                return
            #log_info(f"Tracking resource usage for {node}: {self.G.nodes[node]['rsc']}")
            # check for previous resource usage, add resource edge
            if self.resource_db.check_resource_added(self.G.nodes[node]["rsc_name_unique"]) and self.resource_db.get_latest_resource_usage(self.G.nodes[node]["rsc_name_unique"]) is not None:
                self.G.add_edge(self.resource_db.get_latest_resource_usage(self.G.nodes[node]["rsc_name_unique"]), node, weight=self.clk_period, resource_edge=1)
            else:
                self.resource_db.add_resource(self.G.nodes[node]["rsc_name_unique"], False, None)
            self.resource_db.log_resource_usage(self.G.nodes[node]["rsc_name_unique"], node)
        elif self.G.nodes[node]["node_type"] == "serial":
            # add resource edges for all latest used resources to the serial node
            for rsc_name_unique in self.resource_db.resources:
                latest_resource_usage = self.resource_db.get_latest_resource_usage(rsc_name_unique)
                if latest_resource_usage is not None and (latest_resource_usage, node) not in self.G.edges():
                    self.G.add_edge(latest_resource_usage, node, weight=self.clk_period, resource_edge=1)
                    #log_info(f"Adding resource edge from {latest_resource_usage} to {node}")
                self.resource_db.log_resource_usage(rsc_name_unique, node)
            leaves = self.get_graph_leaves()
            log_info(f"leaves: {leaves} for {node}")
            for leaf in leaves:
                if self.G.nodes[leaf]["node_type"] != "serial":
                    self.G.add_edge(leaf, node, weight=0.0, resource_edge=0)
    
    def get_graph_leaves(self):
        return [node for node, out_degree in self.G.out_degree() if out_degree == 0]

    def convert_to_standard_dfg_with_wire_ops(self):
        self.G_standard_with_wire_ops = self.add_wire_ops(self.G_standard)
        for loop in self.loop_dfgs:
            for iter_num in self.loop_dfgs[loop]:
                for rsc_delay_only_status in self.loop_dfgs[loop][iter_num]:
                    self.loop_dfgs[loop][iter_num][rsc_delay_only_status].convert_to_standard_dfg_with_wire_ops()
                    extra_text = "_rsc_delay_only" if rsc_delay_only_status else ""
                    nx.write_gml(self.loop_dfgs[loop][iter_num][rsc_delay_only_status].G_standard_with_wire_ops, f"{self.build_dir}/parse_results/{self.basic_block_name}/{loop}{extra_text}_graph_standard_with_wire_ops_{iter_num}.gml")

    def add_wire_ops(self, G):
        G_new = G.copy()
        for edge in G.edges():
            src = edge[0]
            dst = edge[1]
            log_info(f"edge: {edge}, src: {src}, dst: {dst}")
            log_info(f"src: {G.nodes[src]}, dst: {G.nodes[dst]}")
            if (G.edges[edge]["resource_edge"] != 1 
                and "core_inst" in G.nodes[src]
                and "core_inst" in G.nodes[dst]
                and G.nodes[src]["core_inst"] != "N/A" 
                and G.nodes[dst]["core_inst"] != "N/A"
            ):
                G_new.add_node(f"wire_{src}_{dst}", node_type="wire", src_node=src, dst_node=dst, function="Wire")
                G_new.add_edge(src, f"wire_{src}_{dst}", resource_edge=0)
                G_new.add_edge(f"wire_{src}_{dst}", dst, resource_edge=0)
                G_new.remove_edge(src, dst)
                log_info(f"Added wire op between {src} and {dst}")
        return G_new

    def remove_node_and_rewire(self, G, node):
        for src in G.predecessors(node):
            for dst in G.successors(node):
                G.add_edge(src, dst, weight=G.edges[src, node]["weight"], resource_edge=0)
        G.remove_node(node)

    def add_one_state_to_graph(self, state, start_node=None, use_start_node=False, resource_delays_only=False):
        #log_info(f"Adding one state to graph")
        for idx in range(len(state)):
            instruction = state[idx]
            op_name = self.variable_db.update_write_node(instruction["op"])
            #log_info(f"instruction: {instruction}")
            assert instruction['dst'].find("%") != -1, f"instruction {instruction} has no %"
            if instruction['dst'].split("%")[1] not in self.resource_mapping:
                assert instruction['op'] in self.no_rsc_allowed_ops, f"instruction {instruction} has no resource mapping."
                logger.warning(f"instruction {instruction} has no resource mapping.")
                rsc_name = "N/A"
            else:
                rsc_name = self.resource_mapping[instruction['dst'].split("%")[1]]
            core_id = instruction["core_id"]
            rsc_name_unique = f"{rsc_name}_{core_id}"
            if not self.resource_db.check_resource_added(rsc_name_unique):
                self.resource_db.add_resource(rsc_name_unique, instruction["core_inst"] != "N/A", instruction["core_info"])
            fn_out = module_map[instruction["op"]] if instruction["op"] in module_map else instruction["op"]
            call_fn = instruction["call_function"] 
            assert call_fn is not None, f"call_function is None for instruction {instruction}"
            if call_fn[-1] == "_":
                call_fn += "s"
                log_info(f"added s to call_function for {instruction}")
            self.G.add_node(op_name, node_type=instruction["type"], function=fn_out, function_out=fn_out, rsc=rsc_name, core_inst=instruction["core_inst"], core_id=core_id, rsc_name_unique=rsc_name_unique, call_function=call_fn)
            self.track_resource_usage(op_name)
            for src in instruction["src"]:
                src_name = self.variable_db.get_read_node_name(src)
                if src_name in self.G:
                    self.G.add_edge(src_name, op_name, weight=0.0, resource_edge=0)
                else:
                    self.G.add_node(src_name, node_type="var", function="N/A")
                    if use_start_node:
                        assert start_node is not None, "Start node is not provided"
                        self.G.add_edge(start_node, src_name, weight=0.0, resource_edge=0)
                    self.G.add_edge(src_name, op_name, weight=0.0, resource_edge=0)
            dst = self.variable_db.update_write_node(instruction["dst"])
            self.G.add_node(dst, node_type="var", function="N/A")
            if not self.resource_delays_only:
                self.G.add_edge(op_name, dst, weight=instruction['delay'], resource_edge=0)
            else:
                self.G.add_edge(op_name, dst, weight=0.0, resource_edge=0)
        assert nx.is_directed_acyclic_graph(self.G), f"Graph is not a DAG, cycle found: {nx.find_cycle(self.G)}"
        #log_info(f"longest path after adding one state: {nx.dag_longest_path_length(self.G)} ({nx.dag_longest_path(self.G)})")

    def add_loop_nodes(self, loop_name, num_iters=1, resource_delays_only=False, append_to_graph=True):
        if append_to_graph:
            G, G_standard, G_standard_with_wire_ops, resource_db, variable_db = self.G, self.G_standard, self.G_standard_with_wire_ops, self.resource_db, self.variable_db
            in_nodes_loop_start = self.get_graph_leaves()
        else:
            G, G_standard, G_standard_with_wire_ops, resource_db, variable_db = nx.DiGraph(), nx.DiGraph(), nx.DiGraph(), ResourceDB(), VariableDB()
            in_nodes_loop_start = []
        G.add_node(f"loop_start_{loop_name}", node_type="serial", function="N/A", loop_name=loop_name)
        for node in in_nodes_loop_start:
            G.add_edge(node, f"loop_start_{loop_name}", weight=0.0, resource_edge=0)
        # create the graph
        loop_dfg = DataFlowGraph(self.clk_period, self.no_rsc_allowed_ops, self.allowed_functions, self.build_dir, self.basic_block_name, loop_name, self.resource_mapping, self.states_structure.get_pruned_states_structure(self.states_structure.loops[loop_name]), G, G_standard, G_standard_with_wire_ops, resource_db, variable_db, resource_delays_only, num_iters)
        
        loop_dfg.create_graph()

        if append_to_graph:
            assert not resource_delays_only, f"dont do that"
            G.add_node(f"II_delay_{loop_name}", node_type="serial", function="II", loop_name=loop_name, pipelined=self.states_structure.loops[loop_name].pipelined, count=self.states_structure.loops[loop_name].count)
            loop_dfg.track_resource_usage(f"II_delay_{loop_name}")
            G.add_node(f"loop_end_{loop_name}", node_type="serial", function="N/A", loop_name=loop_name)
            loop_dfg.track_resource_usage(f"loop_end_{loop_name}")
            log_info(f"loop info: {self.states_structure.loops[loop_name]}")
            if self.states_structure.loops[loop_name].pipelined == "yes":
                II_delay = self.states_structure.loops[loop_name].achieved * self.clk_period * (self.states_structure.loops[loop_name].count-1)
            else:
                II_delay = int(self.states_structure.loops[loop_name].latency) * self.clk_period * (self.states_structure.loops[loop_name].count-1)
            log_info(f"II_delay: {II_delay}")
            G.add_edge(f"II_delay_{loop_name}", f"loop_end_{loop_name}", weight=II_delay, resource_edge=1)
            for sub_loop_dfg in loop_dfg.loop_dfgs:
                for iter_num in loop_dfg.loop_dfgs[sub_loop_dfg]:
                    if iter_num not in self.loop_dfgs[sub_loop_dfg]:
                        self.loop_dfgs[sub_loop_dfg][iter_num] = {}
                    for rsc_delay_only_status in loop_dfg.loop_dfgs[sub_loop_dfg][iter_num]:
                        self.loop_dfgs[sub_loop_dfg][iter_num][rsc_delay_only_status] = loop_dfg.loop_dfgs[sub_loop_dfg][iter_num][rsc_delay_only_status]
                        log_info(f"tracking sub loop graph for sub loop {sub_loop_dfg}, iter {iter_num}, rsc delay only status {rsc_delay_only_status}")
        else:
            G.add_node(f"loop_end_1x", node_type="serial", function="N/A", loop_name=loop_name)
            loop_dfg.track_resource_usage(f"loop_end_1x")
            if num_iters not in self.loop_dfgs[loop_name]:
                self.loop_dfgs[loop_name][num_iters] = {}
            self.loop_dfgs[loop_name][num_iters][resource_delays_only] = loop_dfg
            log_info(f"tracking loop graph for loop {loop_name}, iter {num_iters}, rsc delay only status {True}")

    def create_graph(self, reset_resources=False):
        if reset_resources:
            self.resource_db.reset_resources()
        self.create_graph_one_iter()

    # used only in convert function. Takes one sched report from vitis (for one basic block) and converts it to graph form
    def create_graph_one_iter(self):
        processed_states = set()
        for state in self.states_structure.state_ops:
            if state in processed_states:
                continue
            if state in self.states_structure.state_to_loop:
                loop_to_create = self.states_structure.state_to_loop[state][0]
                log_info(f"adding loop graph for {loop_to_create}")
                self.add_loop_nodes(loop_to_create)
                self.add_loop_nodes(loop_to_create, num_iters=1, resource_delays_only=True, append_to_graph=False)
                self.add_loop_nodes(loop_to_create, num_iters=1, resource_delays_only=False, append_to_graph=False)
                for state in self.states_structure.processed_loop_states:
                    processed_states.add(state)
            else:
                self.add_one_state_to_graph(self.states_structure.state_ops[state])
                processed_states.add(state)
        self.G.add_node(f"graph_end_{self.name}", node_type="serial", function="N/A")
        self.track_resource_usage(f"graph_end_{self.name}")

        assert nx.is_directed_acyclic_graph(self.G), f"Graph is not a DAG, cycle found: {nx.find_cycle(self.G)}"

    def standard_dfg_basic_block(self):
        self.G_standard = copy.deepcopy(self.G)
        for node in self.G.nodes():
            assert "node_type" in self.G.nodes[node], f"Node {node} has no node_type. {self.G.nodes[node]}"
            if self.G.nodes[node]["node_type"] == "var":
                self.remove_node_and_rewire(self.G_standard, node)
        self.G_standard = sim_util.filter_graph_by_function(self.G_standard, self.allowed_functions, exception_node_types=["serial"])
        assert nx.is_directed_acyclic_graph(self.G_standard), f"Graph is not a DAG, cycle found: {nx.find_cycle(self.G_standard)}"
        #log_info(f"longest path after removing var nodes: {nx.dag_longest_path_length(self.G_standard)} ({nx.dag_longest_path(self.G_standard)})")
        #nx.write_gml(self.G_standard, f"{self.build_dir}/{basic_block_name}_graph_{G_standard_name}.gml")
        for loop in self.loop_dfgs:
            for iter_num in self.loop_dfgs[loop]:
                for rsc_delay_only_status in self.loop_dfgs[loop][iter_num]:
                    self.loop_dfgs[loop][iter_num][rsc_delay_only_status].standard_dfg_basic_block()
                    extra_text = "_rsc_delay_only" if rsc_delay_only_status else ""
                    nx.write_gml(self.loop_dfgs[loop][iter_num][rsc_delay_only_status].G_standard, f"{self.build_dir}/parse_results/{self.basic_block_name}/{loop}{extra_text}_graph_standard_{iter_num}.gml")

class StatesStructure:
    def __init__(self, state_ops, state_transitions, loops, loop_list, basic_block_name):
        self.basic_block_name = basic_block_name
        self.state_ops = state_ops
        self.state_transitions = state_transitions
        self.loops = loops
        self.loop_list = loop_list
        log_info(f"loop list: {self.loop_list}")
        self.state_to_loop = {}
        # track incoming transitions for each state that come from downstream states
        self.backward_state_transitions = {}
        for state, transitions in state_transitions.items():
            for transition in transitions:
                if transition < state:
                    assert transition not in self.backward_state_transitions, f"dst node {transition} already in backward state transitions"
                    self.backward_state_transitions[transition] = state
        log_info(f"backward state transitions: {self.backward_state_transitions}")
        # assign loop states
        assert len(self.backward_state_transitions) == len(loops), f"Number of backward state transitions: {len(self.backward_state_transitions)} does not match number of loops: {len(loops)} for basic block: {self.basic_block_name}"
        
        idx=0
        for key in sorted(list(self.backward_state_transitions.keys())):
            start_node, end_node = key, self.backward_state_transitions[key]
            for i in range(start_node, end_node+1):
                self.loop_list[idx].loop_states.append(i)
                if i not in self.state_to_loop:
                    self.state_to_loop[i] = []
                self.state_to_loop[i].append(self.loop_list[idx].loop_name)  
            log_info(f"loop states for {self.loop_list[idx].loop_name}: {self.loop_list[idx].loop_states}")  
            idx += 1
        log_info(f"state to loop: {self.state_to_loop}")
        self.processed_loop_states = set() # updated as we process loops in get_pruned_states_structure
    
    def get_pruned_states_structure(self, loop):
        loop_states = loop.loop_states.copy()
        state_ops = {state: self.state_ops[state] for state in loop_states}
        for sub_loop in loop.sub_loops:
            for state in self.loops[sub_loop].loop_states:
                if state not in state_ops:
                    state_ops[state] = self.state_ops[state]
                    log_info(f"added state {state} to state_ops for sub loop {sub_loop}")
                    loop_states.append(state)
        state_transitions = {state: self.state_transitions[state] for state in loop_states}
        for state in state_transitions:
            new_state_transitions = []
            for transition in state_transitions[state]:
                if transition not in state_ops:
                    log_info(f"removed transition {transition} from state {state} because it is not in state_ops")
                else:
                    new_state_transitions.append(transition)
            state_transitions[state] = new_state_transitions
        # remove the backward transition for this loop (use loop.loop_states here)
        state_transitions[loop.loop_states[-1]].remove(loop.loop_states[0])
        log_info(f"removed backward transition between {loop.loop_states[-1]} and {loop.loop_states[0]} for loop {loop.loop_name}")

        remaining_loops = {other_loop: self.loops[other_loop] for other_loop in loop.sub_loops}
        remaining_loop_list = [self.loops[other_loop] for other_loop in loop.sub_loops]
        log_info(f"remaining loops: {remaining_loops}")
        self.processed_loop_states.update(loop_states)
        return StatesStructure(state_ops, state_transitions, remaining_loops, remaining_loop_list, self.basic_block_name)

class BasicBlockInfo:
    # call parse, then convert
    def __init__(self, build_dir, allowed_functions, basic_block_name, file_path, clk_period, ignore_ops, no_rsc_allowed_ops, resource_mapping):
        self.basic_block_name = basic_block_name
        self.build_dir = build_dir
        self.clk_period = clk_period
        self.state_ops = {}
        self.state_transitions = {}
        self.loops = None
        self.has_loop = False
        self.loop_dfgs = {}
        self.ignore_ops = ignore_ops
        self.no_rsc_allowed_ops = no_rsc_allowed_ops
        self.resource_mapping = resource_mapping
        self.file_path = file_path
        self.allowed_functions = allowed_functions
    
    def parse(self):
        self.parse_file(self.file_path)

    def convert(self):
        self.dfg = DataFlowGraph(self.clk_period, self.no_rsc_allowed_ops, self.allowed_functions, self.build_dir, self.basic_block_name, self.basic_block_name, self.resource_mapping, self.states_structure, nx.DiGraph(), nx.DiGraph(), nx.DiGraph(), ResourceDB(), VariableDB(), 1)
        log_info(f"creating dfg for {self.basic_block_name}")
        self.dfg.create_graph()
        nx.write_gml(self.dfg.G, f"{self.build_dir}/parse_results/{self.basic_block_name}/{self.basic_block_name}_graph.gml")

    def convert_to_standard_dfg(self):
        log_info(f"Converting to standard DFG for {self.basic_block_name}")
        self.dfg.standard_dfg_basic_block()
        nx.write_gml(self.dfg.G_standard, f"{self.build_dir}/parse_results/{self.basic_block_name}/{self.basic_block_name}_graph_standard.gml")

    def convert_to_standard_dfg_with_wire_ops(self):
        log_info(f"Converting to standard DFG with wire ops for {self.basic_block_name}")
        self.dfg.convert_to_standard_dfg_with_wire_ops()
        nx.write_gml(self.dfg.G_standard_with_wire_ops, f"{self.build_dir}/parse_results/{self.basic_block_name}/{self.basic_block_name}_graph_standard_with_wire_ops.gml")
    # label looks like "State _ <SV = _> <Delay = _>"
    def find_next_state(self, lines, idx):
        while lines[idx].find("State ") == -1 and not lines[idx].startswith("=========="): idx += 1
        if lines[idx].startswith("=========="):
            return idx, -1
        return idx+1, lines[idx].split()[1]

    def create_states_structure(self):
        self.states_structure = StatesStructure(self.state_ops, self.state_transitions, self.loops, list(self.loops.values()), self.basic_block_name)

    def parse_file(self, file_path):
        with open(file_path, "r") as file:
            lines = file.readlines()
            idx = 0

            self.loops = self.extract_loop_info_xml(file_path + ".xml")
            for loop_name in self.loops:
                self.loops[loop_name].pipeline_states = {}

            # parse FSM state transitions and pipeline info
            while lines[idx].find("Number of FSM states") == -1:
                idx += 1

            while lines[idx].find("FSM state transitions:") == -1:
                if lines[idx].find("States = {") != -1:
                    self.loops[loop_name].pipeline_states = lines[idx].strip()[lines[idx].find("States = {") + len("States = {")-1:-1].split()
                    for i in range(len(self.loops[loop_name].pipeline_states)):
                        self.loops[loop_name].pipeline_states[i] = int(self.loops[loop_name].pipeline_states[i])
                    log_info(f"pipeline states: {self.loops[loop_name].pipeline_states}")
                idx += 1
            idx += 1
            self.state_transitions = {}
            num_backward_transitions = 0
            while lines[idx].strip():
                assert lines[idx].split()[0] not in self.state_transitions
                start_state = int(lines[idx].split()[0])
                if len(lines[idx].split()) > 2:
                    dst_states = [int(dst_state) for dst_state in lines[idx].split()[2:]]
                    self.state_transitions[start_state] = []
                    for dst_state in dst_states:
                        if dst_state < start_state:
                            num_backward_transitions += 1
                        self.state_transitions[start_state].append(dst_state)
                else:
                    assert len(lines[idx].split()) == 2, f"Number of states: {len(lines[idx].split())} is not 2 for file: {file_path}"
                    if len(self.loops) != num_backward_transitions:
                        assert len(self.loops) == num_backward_transitions + 1, f"Number of backward transitions: {num_backward_transitions} not within one of number of loops: {len(self.loops)} for file: {file_path}"
                        # back to start state
                        self.state_transitions[start_state] = [1]
                        num_backward_transitions += 1
                    else:
                        self.state_transitions[start_state] = []
                idx += 1
            assert num_backward_transitions == len(self.loops), f"Number of backward state transitions: {num_backward_transitions} does not match number of loops: {len(self.loops)} for file: {file_path}"

            # parse operations in each state
            idx, next_state = self.find_next_state(lines, idx)
            while next_state != -1:
                self.state_ops[int(next_state)] = []
                while lines[idx].strip():
                    instruction = lines[idx].split("--->")[1].strip().split("\"")[1]

                    # parse operation name
                    pattern = r"--->   Operation \d+ '(\w+)'"
                    operation_name = match_pattern_in_line(lines[idx], pattern).group(1)

                    # parse operation latency
                    pattern = r'<Latency = (\d+)>'
                    operation_latency_match = match_pattern_in_line(lines[idx], pattern)
                    if operation_latency_match:
                        operation_latency = int(operation_latency_match.group(1))
                    else:
                        operation_latency = 0

                    # parse operation delay
                    pattern = r'<Delay = ([\d.]+)>'
                    operation_delay = float(match_pattern_in_line(lines[idx], pattern).group(1)) * (operation_latency+1)

                    # parse operation progress, only include first operation (1/1, 12/12, etc). We model delay by Latency * clk_period + Delay
                    pattern = r'(\d+)/(\d+)'
                    operation_progress = match_pattern_in_line(lines[idx], pattern).group().split("/")
                    first_op = int(operation_progress[0]) == int(operation_progress[1])

                    pattern = r'--->   Core (\d+) \'(\w+)\''
                    core_inst = match_pattern_in_line(lines[idx], pattern)

                    if operation_name not in self.ignore_ops and first_op:
                        parsed_op = llvm_ir_parse.parse_op(instruction, operation_name)
                        parsed_op["delay"] = operation_delay
                        if core_inst:
                            parsed_op["core_inst"] = core_inst.group(2)
                            parsed_op["core_id"] = core_inst.group(1)
                            parsed_op["core_info"] = get_core_info(lines[idx])
                        else:
                            parsed_op["core_inst"] = "N/A"
                            parsed_op["core_id"] = "N/A"
                            parsed_op["core_info"] = None
                        #log_info(parsed_op)
                        self.state_ops[int(next_state)].append(parsed_op)
                    idx += 1
                idx, next_state = self.find_next_state(lines, idx)
        self.create_states_structure()

    def extract_loop_info_xml(self, xml_file_path):
        log_info(f"Extracting loop info from {xml_file_path}")
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Find the Loop table
        loop_table = None
        for item in root.findall(".//item[@name='Loop']"):
            loop_table = item.find("table")
            break
        
        if loop_table is None:
            return None
        
        # Get the column names (keys)
        keys_elem = loop_table.find(".//keys")
        if keys_elem is not None:
            keys = keys_elem.text.split(', ')[1:] # skip loop name field here
        else:
            keys = ['min', 'max', 'Latency', 'achieved', 'target', 'Count', 'Pipelined']
        
        # Extract loop data
        loops = []
        loop_objects = {}
        latest_top_loop_name_at_level = {0: None}
        for column in loop_table.findall(".//column"):
            parts = re.split(r'([+-]+)', column.get('name', ''))
            log_info(f"parts: {parts}")
            delimiter = parts[1] if len(parts) > 1 else None
            is_top_loop = delimiter.strip().startswith("-")
            loop_name = parts[2].strip()
            values = column.text.split(', ')
            loop_data = {"Loop Name": loop_name, "has_loop": True, "is_top_loop": is_top_loop}
            if not is_top_loop:
                num_pluses = delimiter.strip().count("+")
                log_info(f"num_pluses: {num_pluses}")
                assert num_pluses > 0, f"Number of pluses: {num_pluses} is not greater than 0 for loop: {loop_name}"
                for level_idx in range(num_pluses):
                    assert latest_top_loop_name_at_level[level_idx] is not None, f"Latest top loop name is not set for level: {level_idx} for loop: {loop_name}"
                    loop_objects[latest_top_loop_name_at_level[level_idx]].sub_loops.append(loop_name)
                latest_top_loop_name_at_level[num_pluses] = loop_name
            else:
                latest_top_loop_name_at_level[0] = loop_name
            for i, value in enumerate(values):
                if i < len(keys):
                    key = keys[i]
                    log_info(f"key: {key}, value: {value}")
                    # Convert numeric values
                    if key in keys:
                        try:
                            loop_data[key] = int(value)
                        except ValueError:
                            loop_data[key] = value
                    else:
                        loop_data[key] = value
            if type(loop_data["Latency"]) == str:
                assert loop_data["Latency"].find("~") != -1, f"Latency: {loop_data['Latency']} is not a range of latencies for loop: {loop_name}"
                loop_data["Latency"] = int(loop_data["Latency"].split("~")[1]) # conservative estimate
            
            loop_objects[loop_name] = LoopInfo(loop_data)
        log_info(f"loop objects for {xml_file_path}: {loop_objects}")
        return loop_objects

class vitis_schedule_parser:
    def __init__(self, build_dir, benchmark_name, top_level_module_name, clk_period, allowed_functions):
        self.build_dir = build_dir
        self.solution_dir = os.path.join(build_dir, benchmark_name, "solution1/.autopilot/db")
        self.clk_period = clk_period
        self.top_level_module_name = top_level_module_name
        self.basic_blocks = {}
        self.ignore_ops = set([
            "spectopmodule",
            "specinterface",
            "specbitsmap",
            "specstablecontent",
            "specmemcore",
            "br",
            "alloca",
            "ret",
            "specpipeline",
            "specloopname",
            "specdataflowpipeline",
            "speclooptripcount",
            "specchannel"
        ])
        self.no_rsc_allowed_ops = set([
            "switch"
        ])
        self.benchmark_name = benchmark_name
        self.allowed_functions = allowed_functions

    def create_dfgs(self):
        log_info(f"Resource mapping from: {self.build_dir}/parse_results/{self.top_level_module_name}_full_netlist_unfiltered.gml")
        self.resource_mapping = get_rsc_mapping(f"{self.build_dir}/parse_results/{self.top_level_module_name}_full_netlist_unfiltered.gml")
        log_info(self.resource_mapping)
        for file in os.listdir(self.solution_dir):
            if file.endswith(".verbose.sched.rpt"):
                file_path = os.path.join(self.solution_dir, file)
                basic_block_name = file_path.split("/")[-1].split(".")[0]
                self.basic_blocks[basic_block_name] = BasicBlockInfo(self.build_dir, self.allowed_functions, basic_block_name, file_path, self.clk_period, self.ignore_ops, self.no_rsc_allowed_ops, self.resource_mapping)
                self.basic_blocks[basic_block_name].parse()
                self.basic_blocks[basic_block_name].convert()
                self.basic_blocks[basic_block_name].convert_to_standard_dfg()
                self.basic_blocks[basic_block_name].convert_to_standard_dfg_with_wire_ops()