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

def debug_print(msg):
    if DEBUG:
        print(msg)

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
        #debug_print(f"Logging resource usage for {var_name} on port {self.resource_idx} of {self.resource_name}: {self.resource_usage[self.resource_idx]}")
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



def get_rsc_mapping(netlist_file):
    netlist = nx.read_gml(netlist_file)
    netlist_op_dest_to_node = {}
    for n, d in netlist.nodes(data=True):
        ## extract name and bind->opset fields
        name = d.get('name')
        bind = d.get('bind', {})
        opset = bind.get('opset')
        ## remove the slash and everything after it from the opset
        if opset and '/' in opset:
            opsets = opset.split()
            for opset in opsets:
                netlist_op_dest_to_node[opset.split('/')[0]] = name
        else:
            netlist_op_dest_to_node[opset] = name
    return netlist_op_dest_to_node

class vitis_schedule_parser:
    def __init__(self, build_dir, benchmark_name, top_level_module_name, clk_period, allowed_functions):
        self.build_dir = build_dir
        self.solution_dir = os.path.join(build_dir, benchmark_name, "solution1/.autopilot/db")
        self.clk_period = clk_period
        self.top_level_module_name = top_level_module_name
        self.G = nx.DiGraph()
        self.loop_graphs = {}
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
            "speclooptripcount"
        ])
        self.benchmark_name = benchmark_name
        self.allowed_functions = allowed_functions

    def create_dfgs(self):
        self.parse()
        self.convert()
        self.convert_to_standard_dfg()

    def parse(self):
        self.resource_mapping = get_rsc_mapping(f"{self.build_dir}/parse_results/{self.top_level_module_name}_full_netlist_unfiltered.gml")
        debug_print(self.resource_mapping)
        for file in os.listdir(self.solution_dir):
            if file.endswith(".verbose.sched.rpt"):
                intf_file = file.replace(".verbose.sched.rpt", ".tbgen.tcl")
                assert os.path.exists(os.path.join(self.solution_dir, intf_file))
                self.parse_one_file(os.path.join(self.solution_dir, file), os.path.join(self.solution_dir, intf_file))
        #print(f"basic_blocks loop info before convert: {list([block_name, block['loop_info']] for block_name, block in self.basic_blocks.items())}")

    def convert(self):
        for basic_block_name in self.basic_blocks:
            self.dfg_basic_block(basic_block_name)
            nx.write_gml(self.basic_blocks[basic_block_name]["G"], f"{self.build_dir}/parse_results/{basic_block_name}/{basic_block_name}_graph.gml")
            if self.basic_blocks[basic_block_name]["loop_info"] != "N/A":
                nx.write_gml(self.basic_blocks[basic_block_name]["G_loop_2x"], f"{self.build_dir}/parse_results/{basic_block_name}/{basic_block_name}_graph_loop_2x.gml")
                nx.write_gml(self.basic_blocks[basic_block_name]["G_loop_1x"], f"{self.build_dir}/parse_results/{basic_block_name}/{basic_block_name}_graph_loop_1x.gml")

    def convert_to_standard_dfg(self):
        for basic_block_name in self.basic_blocks:
            #print(f"Converting to standard DFG for {basic_block_name}")
            self.standard_dfg_basic_block(basic_block_name, "G", "G_standard")
            nx.write_gml(self.basic_blocks[basic_block_name]["G_standard"], f"{self.build_dir}/parse_results/{basic_block_name}/{basic_block_name}_graph_standard.gml")

            if self.basic_blocks[basic_block_name]["loop_info"] != "N/A":
                self.standard_dfg_basic_block(basic_block_name, "G_loop_1x", "G_loop_1x_standard")
                nx.write_gml(self.basic_blocks[basic_block_name]["G_loop_1x_standard"], f"{self.build_dir}/parse_results/{basic_block_name}/{basic_block_name}_graph_loop_1x_standard.gml")
                #extra_edges = [edge for edge in self.basic_blocks[basic_block_name]["G_loop_1x"].edges() if self.basic_blocks[basic_block_name]["G_loop_1x"].edges[edge]["resource_edge"] == 1]
                #sim_util.svg_plot(self.basic_blocks[basic_block_name]["G_loop_1x"], f"{self.build_dir}/parse_results/{basic_block_name}/{basic_block_name}_graph_loop_1x.svg", extra_edges)

                self.standard_dfg_basic_block(basic_block_name, "G_loop_2x", "G_loop_2x_standard")
                nx.write_gml(self.basic_blocks[basic_block_name]["G_loop_2x_standard"], f"{self.build_dir}/parse_results/{basic_block_name}/{basic_block_name}_graph_loop_2x_standard.gml")
                #extra_edges = [edge for edge in self.basic_blocks[basic_block_name]["G_loop_2x"].edges() if self.basic_blocks[basic_block_name]["G_loop_2x"].edges[edge]["resource_edge"] == 1]
                #sim_util.svg_plot(self.basic_blocks[basic_block_name]["G_loop_2x"], f"{self.build_dir}/parse_results/{basic_block_name}/{basic_block_name}_graph_loop_2x.svg", extra_edges)


    # label looks like "State _ <SV = _> <Delay = _>"
    def find_next_state(self, lines, idx, basic_block_name):
        while lines[idx].find("State ") == -1 and not lines[idx].startswith("=========="): idx += 1
        if lines[idx].startswith("=========="):
            return idx, -1
        return idx+1, lines[idx].split()[1]

    def extract_c_model_arg_map_list(self, tcl_content):
        """
        Extract C_modelArgMapList from Vitis HLS TCL file into dictionary format.
        
        Args:
            tcl_content (str): Content of the .tbgen.tcl file
            
        Returns:
            list: List of dictionaries containing argument information
        """
        # Find the C_modelArgMapList section
        pattern = r'set C_modelArgMapList \{\[([^\]]+)\]\}'
        match = re.search(pattern, tcl_content, re.DOTALL)
        
        if not match:
            return []
        
        arg_map_content = match.group(1)
        
        # Parse each argument entry
        # Pattern to match: { "Name" : "v362", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"}
        arg_pattern = r'\{\s*"Name"\s*:\s*"([^"]+)"\s*,\s*"interface"\s*:\s*"([^"]+)"\s*,\s*"bitwidth"\s*:\s*(\d+)\s*,\s*"direction"\s*:\s*"([^"]+)"\s*\}'
        
        args = []
        for match in re.finditer(arg_pattern, arg_map_content):
            name, interface, bitwidth, direction = match.groups()
            args.append({
                "Name": name,
                "interface": interface,
                "bitwidth": int(bitwidth),
                "direction": direction
            })
        
        return args

    def match_pattern_in_line(self, line, pattern):
        match = re.search(pattern, line)
        return match

    def extract_loop_info_xml(self, xml_file_path):
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
        for column in loop_table.findall(".//column"):
            loop_name = column.get('name', '').split('-')[1].strip()
            values = column.text.split(', ')
            loop_data = {"Loop Name": loop_name}
            for i, value in enumerate(values):
                if i < len(keys):
                    key = keys[i]
                    debug_print(f"key: {key}, value: {value}")
                    # Convert numeric values
                    if key in keys:
                        try:
                            loop_data[key] = int(value)
                        except ValueError:
                            loop_data[key] = value
                    else:
                        loop_data[key] = value
            
            loops.append(loop_data)
        #print(f"loops for {xml_file_path}: {loops}")
        if len(loops) == 0:
            return "N/A"
        else:
            assert len(loops) == 1, f"Multiple loops found for {xml_file_path}"
            return loops[0] # only one loop per basic block for now

    def parse_one_file(self, file_path, intf_file_path):
        basic_block_name = file_path.split("/")[-1].split(".")[0]
        self.basic_blocks[basic_block_name] = {}
        self.basic_blocks[basic_block_name]["variable_db"] = VariableDB()
        self.basic_blocks[basic_block_name]["variable_db_2x"] = VariableDB()
        self.basic_blocks[basic_block_name]["variable_db_1x"] = VariableDB()
        self.basic_blocks[basic_block_name]["resource_db_map"] = {
            "2x": ResourceDB(),
            "1x": ResourceDB(),
            "full": ResourceDB()
        }
        #print(file_path)
        with open(intf_file_path, "r") as file:
            tcl_content = file.read()
            self.basic_blocks[basic_block_name]["C_modelArgMapList"] = self.extract_c_model_arg_map_list(tcl_content)

        with open(file_path, "r") as file:
            lines = file.readlines()
            idx = 0

            self.basic_blocks[basic_block_name]["loop_info"] = self.extract_loop_info_xml(file_path + ".xml")

            # parse FSM state transitions
            while lines[idx].find("FSM state transitions:") == -1:
                idx += 1
            assert lines[idx].find("FSM state transitions:") != -1
            idx += 1
            self.basic_blocks[basic_block_name]["FSM state transitions"] = {}
            while lines[idx].strip():
                assert lines[idx].split()[0] not in self.basic_blocks[basic_block_name]["FSM state transitions"]
                if len(lines[idx].split()) == 3:
                    self.basic_blocks[basic_block_name]["FSM state transitions"][int(lines[idx].split()[0])] = int(lines[idx].split()[2])
                elif len(lines[idx].split()) == 2:
                    # back to start
                    self.basic_blocks[basic_block_name]["FSM state transitions"][int(lines[idx].split()[0])] = 1
                idx += 1

            # parse operations in each state
            idx, next_state = self.find_next_state(lines, idx, basic_block_name)
            while next_state != -1:
                self.basic_blocks[basic_block_name][int(next_state)] = []
                while lines[idx].strip():
                    instruction = lines[idx].split("--->")[1].strip().split("\"")[1]

                    # parse operation name
                    pattern = r"--->   Operation \d+ '(\w+)'"
                    operation_name = self.match_pattern_in_line(lines[idx], pattern).group(1)

                    # parse operation latency
                    pattern = r'<Latency = (\d+)>'
                    operation_latency_match = self.match_pattern_in_line(lines[idx], pattern)
                    if operation_latency_match:
                        operation_latency = int(operation_latency_match.group(1))
                    else:
                        operation_latency = 0

                    # parse operation delay
                    pattern = r'<Delay = ([\d.]+)>'
                    operation_delay = float(self.match_pattern_in_line(lines[idx], pattern).group(1)) * (operation_latency+1)

                    # parse operation progress, only include first operation (1/1, 12/12, etc). We model delay by Latency * clk_period + Delay
                    pattern = r'(\d+)/(\d+)'
                    operation_progress = self.match_pattern_in_line(lines[idx], pattern).group().split("/")
                    first_op = int(operation_progress[0]) == int(operation_progress[1])

                    pattern = r'--->   Core (\d+) \'(\w+)\''
                    core_inst = self.match_pattern_in_line(lines[idx], pattern)

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
                        debug_print(parsed_op)
                        self.basic_blocks[basic_block_name][int(next_state)].append(parsed_op)
                    idx += 1
                idx, next_state = self.find_next_state(lines, idx, basic_block_name)
        debug_print(self.basic_blocks[basic_block_name])

    def loop_2x_graph(self, basic_block_name):
        self.basic_blocks[basic_block_name]["G_loop_2x"] = nx.DiGraph()
        for _ in range(2):
            for state in self.basic_blocks[basic_block_name]["cycle_nodes"]:
                self.basic_blocks[basic_block_name]["G_loop_2x"], self.basic_blocks[basic_block_name][state] = \
                    self.add_one_state_to_graph(self.basic_blocks[basic_block_name]["G_loop_2x"], basic_block_name, self.basic_blocks[basic_block_name][state], variable_db="variable_db_2x", graph_type="2x")
        self.basic_blocks[basic_block_name]["G_loop_2x"].add_node(f"loop_end_2x", node_type="serial", function="N/A")
        self.track_resource_usage(self.basic_blocks[basic_block_name]["G_loop_2x"], f"loop_end_2x", basic_block_name, "2x")
        assert nx.is_directed_acyclic_graph(self.basic_blocks[basic_block_name]["G_loop_2x"]), f"Graph is not a DAG, cycle found: {nx.find_cycle(self.basic_blocks[basic_block_name]['G_loop_2x'])}"
        #nx.write_gml(self.basic_blocks[basic_block_name]["G_loop_2x"], f"{self.build_dir}/{basic_block_name}_graph_loop_2x.gml")

    def loop_1x_graph(self, basic_block_name):
        self.basic_blocks[basic_block_name]["G_loop_1x"] = nx.DiGraph()
        for state in self.basic_blocks[basic_block_name]["cycle_nodes"]:
            self.basic_blocks[basic_block_name]["G_loop_1x"], self.basic_blocks[basic_block_name][state] = \
                self.add_one_state_to_graph(self.basic_blocks[basic_block_name]["G_loop_1x"], basic_block_name, self.basic_blocks[basic_block_name][state], variable_db="variable_db_1x", graph_type="1x")
        self.basic_blocks[basic_block_name]["G_loop_1x"].add_node(f"loop_end_1x", node_type="serial", function="N/A")
        self.track_resource_usage(self.basic_blocks[basic_block_name]["G_loop_1x"], f"loop_end_1x", basic_block_name, "1x")
        assert nx.is_directed_acyclic_graph(self.basic_blocks[basic_block_name]["G_loop_1x"]), f"Graph is not a DAG, cycle found: {nx.find_cycle(self.basic_blocks[basic_block_name]['G_loop_1x'])}"
        #nx.write_gml(self.basic_blocks[basic_block_name]["G_loop_1x"], f"{self.build_dir}/{basic_block_name}_graph_loop_1x.gml")

    def add_one_state_to_graph(self, G, basic_block_name, state, start_node=None, use_start_node=False, variable_db="variable_db", graph_type="full"):
        debug_print(f"Adding one state to graph: {state}")
        for idx in range(len(state)):
            instruction = state[idx]
            op_name = self.basic_blocks[basic_block_name][variable_db].update_write_node(instruction["op"])
            rsc_name = self.resource_mapping[instruction['dst'].split("%")[1]]
            core_id = instruction["core_id"]
            rsc_name_unique = f"{rsc_name}_{core_id}"
            if not self.basic_blocks[basic_block_name]['resource_db_map'][graph_type].check_resource_added(rsc_name_unique):
                self.basic_blocks[basic_block_name]['resource_db_map'][graph_type].add_resource(rsc_name_unique, instruction["core_inst"] != "N/A", instruction["core_info"])
            fn_out = module_map[instruction["op"]] if instruction["op"] in module_map else instruction["op"]
            G.add_node(op_name, node_type=instruction["type"], function=fn_out, function_out=fn_out, rsc=rsc_name, core_inst=instruction["core_inst"], core_id=core_id, rsc_name_unique=rsc_name_unique, call_function=instruction["call_function"])
            self.track_resource_usage(G, op_name, basic_block_name, graph_type)
            debug_print(f"Instruction: {instruction}")
            for src in instruction["src"]:
                src_name = self.basic_blocks[basic_block_name][variable_db].get_read_node_name(src)
                if src_name in G:
                    G.add_edge(src_name, op_name, weight=0.0, resource_edge=0)
                else:
                    G.add_node(src_name, node_type="var", function="N/A")
                    if use_start_node:
                        assert start_node is not None, "Start node is not provided"
                        G.add_edge(start_node, src_name, weight=0.0, resource_edge=0)
                    G.add_edge(src_name, op_name, weight=0.0, resource_edge=0)
            dst = self.basic_blocks[basic_block_name][variable_db].update_write_node(instruction["dst"])
            G.add_node(dst, node_type="var", function="N/A")
            G.add_edge(op_name, dst, weight=instruction['delay'], resource_edge=0)
        assert nx.is_directed_acyclic_graph(G), f"Graph is not a DAG, cycle found: {nx.find_cycle(G)}"
        debug_print(f"longest path after adding one state: {nx.dag_longest_path_length(G)} ({nx.dag_longest_path(G)})")
        return G, state
    
    def get_graph_leaves(self, G):
        return [node for node, out_degree in G.out_degree() if out_degree == 0]

    def add_loop_nodes(self, basic_block_name, graph_type="full"):
        in_nodes_loop_start = self.get_graph_leaves(self.basic_blocks[basic_block_name]["G"])
        self.basic_blocks[basic_block_name]["G"].add_node("loop_start", node_type="serial", function="N/A")
        for node in in_nodes_loop_start:
            self.basic_blocks[basic_block_name]["G"].add_edge(node, "loop_start", weight=0.0, resource_edge=0)
        for state in self.basic_blocks[basic_block_name]["cycle_nodes"]:
            self.basic_blocks[basic_block_name]["G"], self.basic_blocks[basic_block_name][state] = \
                self.add_one_state_to_graph(
                    self.basic_blocks[basic_block_name]["G"], 
                    basic_block_name, 
                    self.basic_blocks[basic_block_name][state], 
                    "loop_start", 
                    True
                )
        self.basic_blocks[basic_block_name]["G"].add_node(f"II_delay_{basic_block_name}", node_type="serial", function="II", pipelined=self.basic_blocks[basic_block_name]["loop_info"]["Pipelined"], count=self.basic_blocks[basic_block_name]["loop_info"]["Count"])
        self.track_resource_usage(self.basic_blocks[basic_block_name]["G"], f"II_delay_{basic_block_name}", basic_block_name, graph_type)
        self.basic_blocks[basic_block_name]["G"].add_node(f"loop_end_{basic_block_name}", node_type="serial", function="N/A")
        self.track_resource_usage(self.basic_blocks[basic_block_name]["G"], f"loop_end_{basic_block_name}", basic_block_name, graph_type)
        logger.info(f"loop info: {self.basic_blocks[basic_block_name]['loop_info']}, basic block name: {basic_block_name}")
        if self.basic_blocks[basic_block_name]["loop_info"]["Pipelined"] == "yes":
            II_delay = self.basic_blocks[basic_block_name]["loop_info"]["achieved"] * self.clk_period * (self.basic_blocks[basic_block_name]["loop_info"]["Count"]-1)
        else:
            II_delay = int(self.basic_blocks[basic_block_name]["loop_info"]["Latency"]) * self.clk_period
        logger.info(f"II_delay: {II_delay}")
        self.basic_blocks[basic_block_name]["G"].add_edge(f"II_delay_{basic_block_name}", f"loop_end_{basic_block_name}", weight=II_delay, resource_edge=1)

    # used only in convert function. Takes one sched report from vitis (for one basic block) and converts it to graph form
    def dfg_basic_block(self, basic_block_name, graph_type="full"):
        self.basic_blocks[basic_block_name]['resource_db_map'][graph_type].reset_resources()

        state_machine_graph = nx.DiGraph()
        for state in self.basic_blocks[basic_block_name]["FSM state transitions"]:
            state_machine_graph.add_node(state)
        for state in self.basic_blocks[basic_block_name]["FSM state transitions"]:
            state_machine_graph.add_edge(state, self.basic_blocks[basic_block_name]["FSM state transitions"][state])
        cycle = nx.find_cycle(state_machine_graph)
        # often times the final state transition is "<n> ---> ", with no destination, so I'm not exactly sure how to interpret. Is it going back to the beginning in the case of a pipelined loop?
        # for now, I check if there is any loop info found for this file before making this assumption. Otherwise, I assume it just exits.
        if self.basic_blocks[basic_block_name]["loop_info"] != "N/A":
            self.basic_blocks[basic_block_name]["cycle_nodes"] = [node for (node, _) in cycle]
            self.basic_blocks[basic_block_name]["non_cycle_nodes"] = [node for node in state_machine_graph.nodes() if node not in self.basic_blocks[basic_block_name]["cycle_nodes"]]
            assert len(self.basic_blocks[basic_block_name]["cycle_nodes"]) == self.basic_blocks[basic_block_name]["cycle_nodes"][-1] - self.basic_blocks[basic_block_name]["cycle_nodes"][0] + 1, f"Cycle nodes are not consecutive: {self.basic_blocks[basic_block_name]['cycle_nodes']}"
        else:
            self.basic_blocks[basic_block_name]["cycle_nodes"] = []
            self.basic_blocks[basic_block_name]["non_cycle_nodes"] = [node for node in state_machine_graph.nodes()]

        self.basic_blocks[basic_block_name]["G"] = nx.DiGraph()
        self.basic_blocks[basic_block_name]["G"].add_node("graph_start", node_type="serial", function="N/A")
        for state in self.basic_blocks[basic_block_name]["non_cycle_nodes"]:
            self.basic_blocks[basic_block_name]["G"], self.basic_blocks[basic_block_name][state] = \
                self.add_one_state_to_graph(
                    self.basic_blocks[basic_block_name]["G"], 
                    basic_block_name, 
                    self.basic_blocks[basic_block_name][state],
                    "graph_start",
                    True
                )
            # cover loop nodes at the same time, because we add serialization nodes at the start and end
            if len(self.basic_blocks[basic_block_name]["cycle_nodes"]) > 0 and state+1 == self.basic_blocks[basic_block_name]["cycle_nodes"][0]:
                #print(f"Adding loop nodes for {basic_block_name}")
                cycle_nodes = self.basic_blocks[basic_block_name]["cycle_nodes"]
                #print(f"cycle_nodes: {cycle_nodes}")
                self.add_loop_nodes(basic_block_name)
                self.loop_2x_graph(basic_block_name)
                self.loop_1x_graph(basic_block_name)
        if len(self.basic_blocks[basic_block_name]["non_cycle_nodes"]) == 0:
            self.add_loop_nodes(basic_block_name)
            self.loop_2x_graph(basic_block_name)
            self.loop_1x_graph(basic_block_name)
        self.basic_blocks[basic_block_name]["G"].add_node("graph_end", node_type="serial", function="N/A")
        self.track_resource_usage(self.basic_blocks[basic_block_name]["G"], "graph_end", basic_block_name)
        #nx.write_gml(self.basic_blocks[basic_block_name]["G"], f"{self.build_dir}/{basic_block_name}_graph.gml")

        assert nx.is_directed_acyclic_graph(self.basic_blocks[basic_block_name]["G"]), f"Graph is not a DAG, cycle found: {nx.find_cycle(self.basic_blocks[basic_block_name]['G'])}"

    def remove_node(self, G, node):
        for src in G.predecessors(node):
            for dst in G.successors(node):
                G.add_edge(src, dst, weight=G.edges[src, node]["weight"], resource_edge=0)
        G.remove_node(node)

    def track_resource_usage(self, G, node, basic_block_name, graph_type="full"):
        if G.nodes[node]["node_type"] == "op" and G.nodes[node]["core_inst"] != "N/A":
            #debug_print(f"Tracking resource usage for {node}: {G.nodes[node]['rsc']}")
            # check for previous resource usage, add resource edge
            if self.basic_blocks[basic_block_name]['resource_db_map'][graph_type].check_resource_added(G.nodes[node]["rsc_name_unique"]) and self.basic_blocks[basic_block_name]['resource_db_map'][graph_type].get_latest_resource_usage(G.nodes[node]["rsc_name_unique"]) is not None:
                G.add_edge(self.basic_blocks[basic_block_name]['resource_db_map'][graph_type].get_latest_resource_usage(G.nodes[node]["rsc_name_unique"]), node, weight=self.clk_period, resource_edge=1)
            else:
                self.basic_blocks[basic_block_name]['resource_db_map'][graph_type].add_resource(G.nodes[node]["rsc_name_unique"], False, None)
            self.basic_blocks[basic_block_name]['resource_db_map'][graph_type].log_resource_usage(G.nodes[node]["rsc_name_unique"], node)
        elif G.nodes[node]["node_type"] == "serial":
            # add resource edges for all latest used resources to the serial node
            for rsc_name_unique in self.basic_blocks[basic_block_name]['resource_db_map'][graph_type].resources:
                latest_resource_usage = self.basic_blocks[basic_block_name]['resource_db_map'][graph_type].get_latest_resource_usage(rsc_name_unique)
                if latest_resource_usage is not None and (latest_resource_usage, node) not in G.edges():
                    G.add_edge(latest_resource_usage, node, weight=self.clk_period, resource_edge=1)
                    #debug_print(f"Adding resource edge from {latest_resource_usage} to {node}")
                self.basic_blocks[basic_block_name]['resource_db_map'][graph_type].log_resource_usage(rsc_name_unique, node)
            leaves = self.get_graph_leaves(G)
            debug_print(f"leaves: {leaves} for {node}")
            for leaf in leaves:
                if G.nodes[leaf]["node_type"] != "serial":
                    G.add_edge(leaf, node, weight=0.0, resource_edge=0)


    def standard_dfg_basic_block(self, basic_block_name, G_name, G_standard_name):
        self.basic_blocks[basic_block_name][G_standard_name] = copy.deepcopy(self.basic_blocks[basic_block_name][G_name])
        #debug_print("nodes before removing var nodes: ", self.basic_blocks[basic_block_name][G_name].nodes())
        for node in self.basic_blocks[basic_block_name][G_name].nodes():
            assert "node_type" in self.basic_blocks[basic_block_name][G_name].nodes[node], f"Node {node} has no node_type. {self.basic_blocks[basic_block_name][G_name].nodes[node]}"
            if self.basic_blocks[basic_block_name][G_name].nodes[node]["node_type"] == "var":
                self.remove_node(self.basic_blocks[basic_block_name][G_standard_name], node)
        #debug_print("nodes left: ", self.basic_blocks[basic_block_name][G_standard_name].nodes())
        self.basic_blocks[basic_block_name][G_standard_name] = sim_util.filter_graph_by_function(self.basic_blocks[basic_block_name][G_standard_name], self.allowed_functions, exception_node_types=["serial"])
        assert nx.is_directed_acyclic_graph(self.basic_blocks[basic_block_name][G_standard_name]), f"Graph is not a DAG, cycle found: {nx.find_cycle(self.basic_blocks[basic_block_name][G_standard_name])}"
        logger.info(f"longest path after removing var nodes: {nx.dag_longest_path_length(self.basic_blocks[basic_block_name][G_standard_name])} ({nx.dag_longest_path(self.basic_blocks[basic_block_name][G_standard_name])})")
        #nx.write_gml(self.basic_blocks[basic_block_name][G_standard_name], f"{self.build_dir}/{basic_block_name}_graph_{G_standard_name}.gml")



if __name__ == "__main__":
    parser = vitis_schedule_parser("src/tmp_for_test/benchmark", "jacobi_2d", "jacobi_2d", 250.0)
    parser.parse()
    parser.convert()
    parser.convert_to_standard_dfg()