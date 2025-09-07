import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

import networkx as nx
import re
import networkx as nx

from src.forward_pass import llvm_ir_parse
from src.forward_pass import vitis_create_netlist
from src import sim_util

DEBUG = True

def debug_print(msg):
    if DEBUG:
        print(msg)

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
            opset = opset.split('/')[0]
        if name and opset:
            netlist_op_dest_to_node[opset] = name
    return netlist_op_dest_to_node


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

class vitis_schedule_parser:
    def __init__(self, build_dir, benchmark_name, clk_period):
        self.build_dir = build_dir
        self.solution_dir = os.path.join(build_dir, benchmark_name, "solution1/.autopilot/db")
        self.clk_period = clk_period
        self.G = nx.DiGraph()
        self.loop_graphs = {}
        if not os.path.exists(os.path.join(self.build_dir, f"{benchmark_name}_json")):
            os.makedirs(os.path.join(self.build_dir, f"{benchmark_name}_json"))
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
            "specdataflowpipeline"
        ])


    def parse(self):
        for file in os.listdir(self.solution_dir):
            if file.endswith(".verbose.sched.rpt"):
                intf_file = file.replace(".verbose.sched.rpt", ".tbgen.tcl")
                assert os.path.exists(os.path.join(self.solution_dir, intf_file))
                self.parse_one_file(os.path.join(self.solution_dir, file), os.path.join(self.solution_dir, intf_file))

    def convert(self):
        for basic_block_name in self.basic_blocks:
            self.dfg_basic_block(basic_block_name)

    def convert_to_standard_dfg(self):
        for basic_block_name in self.basic_blocks:
            self.standard_dfg_basic_block(basic_block_name)

    # label looks like "State _ <SV = _> <Delay = _>"
    def find_next_state(self, lines, idx, basic_block_name):
        while lines[idx].find("State ") == -1 and not lines[idx].startswith("=========="): idx += 1
        if lines[idx].startswith("=========="):
            return idx, -1
        return idx+1, lines[idx].split()[1]
    
    def parse_op(self, line):
        op = line.split("--->")[1].strip().split("\"")[1].split()
        if len(op) == 5:
            dst, _, op, src1, src2 = op
        elif len(op) == 7:
            dst, _, op, src1 = op
        else:
            raise ValueError(f"Unexpected opcode: {op}")
        return [dst, op, src1, src2]

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

    def parse_one_file(self, file_path, intf_file_path):
        basic_block_name = file_path.split("/")[-1].split(".")[0]
        self.basic_blocks[basic_block_name] = {}
        self.basic_blocks[basic_block_name]["variable_db"] = VariableDB()
        self.basic_blocks[basic_block_name]["variable_db_2x"] = VariableDB()
        print(file_path)
        with open(intf_file_path, "r") as file:
            tcl_content = file.read()
            self.basic_blocks[basic_block_name]["C_modelArgMapList"] = self.extract_c_model_arg_map_list(tcl_content)

        with open(file_path, "r") as file:
            lines = file.readlines()
            idx = 0
            while lines[idx].find("* Loop:") == -1:
                idx += 1
            assert lines[idx].find("* Loop:") != -1
            if lines[idx+1].strip() == "N/A":
                self.basic_blocks[basic_block_name]["loopname"] = "N/A"
            else:
                self.basic_blocks[basic_block_name]["loopname"] = lines[idx+5].split("|")[1].strip()[2:]

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

                    if operation_name == "speclooptripcount":
                        self.basic_blocks[basic_block_name]["tripcount"] = int(instruction.split()[-1])
                    elif operation_name not in self.ignore_ops and first_op:
                        parsed_op = llvm_ir_parse.parse_op(instruction, operation_name)
                        parsed_op["delay"] = operation_delay
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
                    self.add_one_state_to_graph(self.basic_blocks[basic_block_name]["G_loop_2x"], basic_block_name, self.basic_blocks[basic_block_name][state], variable_db="variable_db_2x")
        
        nx.write_gml(self.basic_blocks[basic_block_name]["G_loop_2x"], f"src/tmp/benchmark/{basic_block_name}_graph_loop_2x.gml")


    def add_one_state_to_graph(self, G, basic_block_name, state, start_node=None, use_start_node=False, variable_db="variable_db"):
        debug_print(f"Adding one state to graph: {state}")
        for idx in range(len(state)):
            instruction = state[idx]
            op_name = self.basic_blocks[basic_block_name][variable_db].update_write_node(instruction["op"])
            G.add_node(op_name, node_type="op")
            debug_print(f"Instruction: {instruction}")
            for src in instruction["src"]:
                src_name = self.basic_blocks[basic_block_name][variable_db].get_read_node_name(src)
                if src_name in G:
                    G.add_edge(src_name, op_name, weight=0.0, resource_edge=0)
                else:
                    G.add_node(src_name, node_type="var")
                    if use_start_node:
                        assert start_node is not None, "Start node is not provided"
                        G.add_edge(start_node, src_name, weight=0.0, resource_edge=0)
                    G.add_edge(src_name, op_name, weight=0.0, resource_edge=0)
            dst = self.basic_blocks[basic_block_name][variable_db].update_write_node(instruction["dst"])
            G.add_node(dst, node_type="var")
            G.add_edge(op_name, dst, weight=instruction['delay'])
        assert nx.is_directed_acyclic_graph(G), f"Graph is not a DAG, cycle found: {nx.find_cycle(G)}"
        debug_print(f"longest path after adding one state: {nx.dag_longest_path_length(G)} ({nx.dag_longest_path(G)})")
        return G, state
    
    def get_graph_leaves(self, G):
        return [node for node, out_degree in G.out_degree() if out_degree == 0]

    def add_loop_nodes(self, basic_block_name):
        in_nodes_loop_start = self.get_graph_leaves(self.basic_blocks[basic_block_name]["G"])
        self.basic_blocks[basic_block_name]["G"].add_node("loop_start", node_type="serial")
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
        in_nodes_loop_end = self.get_graph_leaves(self.basic_blocks[basic_block_name]["G"])
        self.basic_blocks[basic_block_name]["G"].add_node("loop_end", node_type="serial")
        for node in in_nodes_loop_end:
            self.basic_blocks[basic_block_name]["G"].add_edge(node, "loop_end", weight=0.0, resource_edge=0)
    
    def dfg_basic_block(self, basic_block_name):
        state_machine_graph = nx.DiGraph()
        for state in self.basic_blocks[basic_block_name]["FSM state transitions"]:
            state_machine_graph.add_node(state)
        for state in self.basic_blocks[basic_block_name]["FSM state transitions"]:
            state_machine_graph.add_edge(state, self.basic_blocks[basic_block_name]["FSM state transitions"][state])
        cycle = nx.find_cycle(state_machine_graph)
        self.basic_blocks[basic_block_name]["cycle_nodes"] = [node for (node, _) in cycle]
        self.basic_blocks[basic_block_name]["non_cycle_nodes"] = [node for node in state_machine_graph.nodes() if node not in self.basic_blocks[basic_block_name]["cycle_nodes"]]
        debug_print(self.basic_blocks[basic_block_name]["cycle_nodes"])
        debug_print(self.basic_blocks[basic_block_name]["non_cycle_nodes"])
        assert len(self.basic_blocks[basic_block_name]["cycle_nodes"]) == self.basic_blocks[basic_block_name]["cycle_nodes"][-1] - self.basic_blocks[basic_block_name]["cycle_nodes"][0] + 1, f"Cycle nodes are not consecutive: {self.basic_blocks[basic_block_name]['cycle_nodes']}"
        self.basic_blocks[basic_block_name]["G"] = nx.DiGraph()
        for state in self.basic_blocks[basic_block_name]["non_cycle_nodes"]:
            self.basic_blocks[basic_block_name]["G"], self.basic_blocks[basic_block_name][state] = \
                self.add_one_state_to_graph(
                    self.basic_blocks[basic_block_name]["G"], 
                    basic_block_name, 
                    self.basic_blocks[basic_block_name][state]
                )
            # cover loop nodes at the same time, because we add serialization nodes at the start and end
            if state+1 == self.basic_blocks[basic_block_name]["cycle_nodes"][0]:
                self.add_loop_nodes(basic_block_name)
                self.loop_2x_graph(basic_block_name)
        if len(self.basic_blocks[basic_block_name]["non_cycle_nodes"]) == 0:
            self.add_loop_nodes(basic_block_name)
            self.loop_2x_graph(basic_block_name)
        in_nodes_graph_end = self.get_graph_leaves(self.basic_blocks[basic_block_name]["G"])
        self.basic_blocks[basic_block_name]["G"].add_node("graph_end", node_type="serial")
        for node in in_nodes_graph_end:
            self.basic_blocks[basic_block_name]["G"].add_edge(node, "graph_end", weight=0.0, resource_edge=0)
        nx.write_gml(self.basic_blocks[basic_block_name]["G"], f"src/tmp/benchmark/{basic_block_name}_graph.gml")

    def remove_node(self, G, node):
        for src in G.predecessors(node):
            for dst in G.successors(node):
                G.add_edge(src, dst, weight=G.edges[src, node]["weight"], resource_edge=0)
        G.remove_node(node)

    def standard_dfg_basic_block(self, basic_block_name):
        self.basic_blocks[basic_block_name]["G_standard"] = self.basic_blocks[basic_block_name]["G"].copy()
        for node in self.basic_blocks[basic_block_name]["G"].nodes():
            if self.basic_blocks[basic_block_name]["G"].nodes[node]["node_type"] == "var":
                self.remove_node(self.basic_blocks[basic_block_name]["G_standard"], node)
        debug_print(f"longest path after removing var nodes: {nx.dag_longest_path_length(self.basic_blocks[basic_block_name]['G_standard'])}")
        nx.write_gml(self.basic_blocks[basic_block_name]["G_standard"], f"src/tmp/benchmark/{basic_block_name}_graph_standard.gml")



if __name__ == "__main__":
    parser = vitis_schedule_parser("src/tmp/benchmark", "resnet", 250.0)
    parser.parse()
    parser.convert()
    parser.convert_to_standard_dfg()