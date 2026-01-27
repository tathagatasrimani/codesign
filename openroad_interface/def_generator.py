import copy
import os
import math
import pprint
import re 
import networkx as nx
import logging
import yaml

from .openroad_functions import find_val_two, find_val_xy, find_val, value, format, clean
from openroad_interface import openroad_run

design = "codesign"

logger = logging.getLogger(__name__)

## List of supported macros.

# human readable name is just for us to read.
# the macro name in DEF is the name of the macro in the standard cell LEF file.
# search terms are used to find the macro in the output netlist from HLS. 
SUPPORTED_MACROS = [
    {"humnan_readable_name": "and", "macro_name_in_def": "BitAnd16", "search_terms": ["AND"]},
    {"humnan_readable_name": "xor", "macro_name_in_def": "BitXor16", "search_terms": ["XOR"]},
    {"humnan_readable_name": "mux", "macro_name_in_def": "Mux16", "search_terms": ["MUX"]},
    #{"humnan_readable_name": "reg", "macro_name_in_def": "DFF_X1", "search_terms": ["REG"]},
    {"humnan_readable_name": "add", "macro_name_in_def": "Add16", "search_terms": ["ADD"]},
    {"humnan_readable_name": "mult", "macro_name_in_def": "Mult16", "search_terms": ["MUL"]},
    {"humnan_readable_name": "floordiv", "macro_name_in_def": "FloorDiv16", "search_terms": ["FLOORDIV"]},
    {"humnan_readable_name": "sub", "macro_name_in_def": "Sub16", "search_terms": ["SUB"]},
    {"humnan_readable_name": "eq", "macro_name_in_def": "Eq16", "search_terms": ["EQ"]},
    {"humnan_readable_name": "shl", "macro_name_in_def": "LShift16", "search_terms": ["SHL", "LSHIFT16"]},
    {"humnan_readable_name": "lshr", "macro_name_in_def": "RShift16", "search_terms": ["LSHR"]}
]

DEBUG = False
def log_info(msg):
    if DEBUG:
        logger.info(msg)
def log_warning(msg):
    if DEBUG:
        logger.warning(msg)

MAX_STD_CELL_ROWS = 50000  # adjust as needed for memory/runtime

DISABLE_ROW_GENERATION = False  # Set to True to skip std cell row generation entirely


class DefGenerator:
    def __init__(self, cfg, codesign_root_dir, tmp_dir, NEW_database_units_per_micron, subdirectory=None):
        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.tmp_dir = tmp_dir
        self.subdirectory = subdirectory
        self.directory = os.path.join(self.codesign_root_dir, f"{self.tmp_dir}/pd")

        if subdirectory:
            self.directory = os.path.join(self.directory, subdirectory)
            
        self.macro_halo_x = 0.0
        self.macro_halo_y = 0.0
        self.max_dim_macro = 0.0
        self.NEW_database_units_per_micron = NEW_database_units_per_micron

        self.var_file = None

        self.lef_std_file = None
        self.lef_tech_file =  None

        self.core_coord_x2 =  None 
        self.core_coord_x1 =  None 
        self.core_coord_y2 =  None 
        self.core_coord_y1 =  None

        self.die_coord_x2 = None
        self.die_coord_x1 = None
        self.die_coord_y2 = None
        self.die_coord_y1 = None

        self.site_name = None
        self.units = None
        self.site_x = None
        self.site_y = None
        self.layer_min_width = None
        self.layer_x_offset = None
        self.layer_pitch_x = None
        self.layer_pitch_y = None

        # Load pin_count data from tech_params.yaml like macro_maker does
        tech_params_file_path = os.path.join(self.codesign_root_dir, "src/yaml/tech_params.yaml")
        if os.path.exists(tech_params_file_path):
            tech_params = yaml.load(open(tech_params_file_path, "r"), Loader=yaml.Loader)
            self.pin_list = tech_params.get("pin_count", {})
            log_info(f"Loaded pin_count data from tech_params.yaml: {len(self.pin_list)} macros")
        else:
            logger.warning(f"tech_params.yaml not found at {tech_params_file_path}; using default pin counts")
            self.pin_list = {}

    def get_macro_bitwidths(self, macro_name: str) -> tuple:
        """
        Get input and output bitwidths for a macro based on pin_count data.
        Returns (input_bitwidth, output_bitwidth, num_inputs).
        
        Logic:
        - Output bitwidth = output pin count
        - Input bitwidth per input = input pin count / number of inputs
        - Number of inputs is derived from the ratio of input to output pins
        """
        if macro_name not in self.pin_list:
            # Default to 16-bit if not found
            log_warning(f"Macro {macro_name} not found in pin_list, defaulting to 16-bit")
            return (16, 16, 2)  # default: 2 inputs of 16 bits, 1 output of 16 bits
        
        pin_data = self.pin_list[macro_name]
        input_pin_count = pin_data.get("input", 16)
        output_pin_count = pin_data.get("output", 16)
        
        # Output bitwidth is straightforward
        output_bitwidth = output_pin_count
        
        # Determine number of inputs and input bitwidth
        # Common patterns:
        # - input == output: 1 input (e.g., Not16: 16 in, 16 out)
        # - input == 2 * output: 2 inputs (e.g., Add16: 32 in, 16 out)
        # - input == 3 * output or special: handle muxes and special cases
        if input_pin_count == output_pin_count:
            num_inputs = 1
            input_bitwidth = input_pin_count
        elif input_pin_count == 2 * output_pin_count:
            num_inputs = 2
            input_bitwidth = output_pin_count  # each input is output_bitwidth
        elif "Mux" in macro_name:
            # Mux16 has 36 input pins: 2 data inputs (16 each) + 1 select (4 bits) = 32 + 4 = 36
            # For muxes, we'll use output_bitwidth for data inputs
            num_inputs = 2  # data inputs
            input_bitwidth = output_pin_count  # data inputs are same width as output
            # Note: select bit is handled separately in port mapping
        else:
            # Try to infer: assume 2 inputs if ratio is close to 2
            ratio = input_pin_count / output_pin_count if output_pin_count > 0 else 2
            if abs(ratio - 2.0) < 0.5:
                num_inputs = 2
                input_bitwidth = output_pin_count
            else:
                # Fallback: assume 1 input
                num_inputs = 1
                input_bitwidth = input_pin_count
        
        log_info(f"Macro {macro_name}: input_bitwidth={input_bitwidth}, output_bitwidth={output_bitwidth}, num_inputs={num_inputs}")
        return (input_bitwidth, output_bitwidth, num_inputs)


    def get_macro_halo_values(self):
        flow_path = os.path.join(self.directory, "tcl", "codesign_flow.tcl")
        if not os.path.exists(flow_path):
            logger.warning(f"codesign_flow.tcl not found at {flow_path}; skipping halo extraction.")
            return

        with open(flow_path, "r") as f:
            for line in f:
                # strip trailing backslash and whitespace (lines like: "-halo_width 10 \")
                line_clean = line.rstrip().rstrip("\\").strip()
                # look for numeric after -halo_width or -halo_height
                m_w = re.search(r"-?halo_width\s+([-+]?\d*\.?\d+)", line_clean, re.IGNORECASE)
                if m_w:
                    try:
                        self.macro_halo_x = float(m_w.group(1))
                    except Exception:
                        logger.debug(f"Failed to parse halo_width from line: {line.strip()}")
                    continue

                m_h = re.search(r"-?halo_height\s+([-+]?\d*\.?\d+)", line_clean, re.IGNORECASE)
                if m_h:
                    try:
                        self.macro_halo_y = float(m_h.group(1))
                    except Exception:
                        logger.debug(f"Failed to parse halo_height from line: {line.strip()}")

        log_info(f"Using macro halo values: x = {self.macro_halo_x}, y = {self.macro_halo_y}")

    def component_finder(self, name: str) -> str:
        '''
        returns a blank string if the component name is not a component we need

        redo this whole function 
        '''
        ## This is for hierarchically P&R'ed modules. The macro name is the same as the module name except that it will have this prefix "HIERMODULE_"
        if "HIERMODULE_" in name.upper():
            return name

        for macro in SUPPORTED_MACROS:
            if macro["macro_name_in_def"].upper() in name.upper():
                return macro["macro_name_in_def"]
        return ""

    def find_macro(self, node: dict) -> str:
        '''
        find the corresponding macro for the given node

        redo this function 
        '''
        name = node["function"]
        if "CALL" in name.upper():
            ## This is for hierarchically P&R'ed modules. The macro name is the same as the module name. It will have this prefix "HIERMODULE_"
            macro_name = node.get("call_submodule_instance_name", None)
            if macro_name is None:
                raise ValueError(f"CALL node missing 'call_submodule_instance_name' attribute: {node}")

            return f"HIERMODULE_{macro_name}"

        for macro in SUPPORTED_MACROS:
            for term in macro["search_terms"]:
                if term in name.upper():
                    return macro["macro_name_in_def"]
                
        raise ValueError(f"Macro not found for {name}")

    def edge_gen(self, in_or_out, nodes, graph) -> dict:
        '''
        generates a dict that contains either input or outputs of a node
        '''
        result = {}
        for node in nodes:
            result[node] = []
            if in_or_out == "in":
                edges = graph.in_edges(node)
                for edge in edges:
                    result[node].append(edge[0])
            else:
                edges = graph.out_edges(node)
                for edge in edges:
                    result[node].append(edge[1])

        return result
    

    def read_tcl_file(self, test_file: str):
        test_file_path = os.path.join(self.directory, "tcl", test_file)

        test_file_data = open(test_file_path)
        test_file_lines = test_file_data.readlines()

        log_info(f"Reading tcl file: {test_file}")

        # extracting vars file, die area, and core area from tcl
        for line in test_file_lines: 
            if ".vars" in line:
                var = re.findall(r'"(.*?)"', line)
                self.var_file = var[0]
            if "die_area" in line:
                die = re.findall(r'{(.*?)}', line)
                die = die[0].split()
                self.die_coord_x1 = float(die[0])
                self.die_coord_y1 = float(die[1])
                self.die_coord_x2 = float(die[2])
                self.die_coord_y2 = float(die[3])
            if "core_area" in line:
                core = re.findall(r'{(.*?)}', line)
                core = core[0].split()
                self.core_coord_x1 = float(core[0])
                self.core_coord_y1 = float(core[1])
                self.core_coord_x2 = float(core[2])
                self.core_coord_y2 = float(core[3])

    def read_lef_file(self):
        self.var_file_data = open(self.directory  +"/tcl/"+ self.var_file) 

        # extracting lef file directories and site name
        for line in self.var_file_data.readlines():
            if "tech_lef" in line:
                self.lef_tech_file = self.directory + "/tcl/" + re.findall(r'"(.*?)"', line)[0]
            if "std_cell_lef" in line:
                self.lef_std_file = self.directory + "/tcl/" + re.findall(r'"(.*?)"', line)[0]
            if "site" in line:
                site = re.findall(r'"(.*?)"', line)
                self.site_name = site[0]

        log_info(f"LEF tech file: {self.lef_tech_file}")
        log_info(f"LEF std file: {self.lef_std_file}")
        log_info(f"Site name: {self.site_name}")

    def create_macro_dict(self):
        # extracting needed macros and their respective pins from lef and puts it into a dict
        lef_std_data = open(self.lef_std_file)
        macro_name = None
        self.macro_dict = {}
        # store macro physical sizes (SIZE x BY y) from stdcell LEF
        self.macro_size_dict = {}
        for line in lef_std_data.readlines():
            if "MACRO" in line:
                macro_name = clean(value(line, "MACRO"))
                log_info(f"Found MACRO: {macro_name}")
                if self.component_finder(macro_name) != "":
                    io = {}
                    self.macro_dict[macro_name] = io
                    io["input"] = []
                    io["output"] = []
                else:
                    macro_name = ""

            elif "PIN" in line and macro_name != "": 
                pin_name = clean(value(line, "PIN"))
                if pin_name.startswith("A") or pin_name.startswith("B") or pin_name.startswith("D"):
                    self.macro_dict[macro_name]["input"].append(pin_name)
                elif pin_name.startswith("Z") or pin_name.startswith("Q") or pin_name.startswith("X"):
                    self.macro_dict[macro_name]["output"].append(pin_name)
            # capture SIZE lines inside MACRO blocks, e.g. "SIZE 22.585 BY 16.8 ;"
            elif "SIZE" in line and macro_name:
                m_size = re.search(r"SIZE\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+BY\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if m_size:
                    try:
                        sx = float(m_size.group(1))
                        sy = float(m_size.group(2))
                        self.max_dim_macro = max(self.max_dim_macro, sx, sy)
                        self.macro_size_dict[macro_name] = (sx, sy)
                        self.macro_dict[macro_name]["area"] = sx * sy
                    except Exception:
                        logger.debug(f"Couldn't parse SIZE for macro {macro_name}: {line.strip()}")
                else:
                    logger.debug(f"SIZE line did not match regex for macro {macro_name}: {line.strip()}")


    def read_tech_file(self):
        # extracting self.units and site size from tech file
        self.lef_data = open(self.lef_tech_file)
        self.lef_tech_lines = self.lef_data.readlines()
        for line in self.lef_tech_lines:
            if "DATABASE MICRONS" in line:
                self.units = float(clean(value(line, "DATABASE MICRONS")))
            if "SITE " + self.site_name in line:
                site_size = find_val_two("SIZE", self.lef_tech_lines, self.lef_tech_lines.index(line))
                self.site_x = float(site_size[0])
                self.site_y = float(site_size[1])
                break

    def run_def_generator(self, test_file: str, graph: nx.DiGraph): 
        '''
        -> nx.DiGraph, dict, dict, dict (it's not working when actaully written)
        Generates required .def file for OpenROAD. It uses the lef files specified in the tcl file to
        determine the components that will be used.
        params: 
            test_file: tcl file
            graph: nx.DiGraph, untouched
        
        returns: 
            graph: networkx graph that has been modified (pruned and new components)
            net_out_dict: dict that lists nodes and their respective output nets
            node_output: dict that lists nodes and their respective output nodes
            lef_data_dict: dict containing data from lef file that will be used for estimating parasitics
        '''

        self.get_macro_halo_values()

        
        ### 0. reading top level tcl file, lef file, and tech file ###
        self.read_tcl_file(test_file)
        self.read_lef_file()
        
        self.create_macro_dict()

        log_info(f"Macro dict: {pprint.pformat(self.macro_dict)}")

        self.read_tech_file()

        # graph reading
        nodes = list(graph)
        control_nodes = list(graph)

        log_info(f"Full Graph: {graph}")

        log_info(f"Control nodes: {control_nodes}")

        ### 1. pruning - set bitwidths based on pin_count data ###
        # First, map all nodes to their macros to get bitwidths
        node_to_macro = {}
        for node1 in control_nodes:
            macro = self.find_macro(graph.nodes[node1])
            _, output_bitwidth, _ = self.get_macro_bitwidths(macro)
            graph.nodes[node1]["count"] = output_bitwidth
            node_to_macro[node1] = [macro, copy.deepcopy(self.macro_dict.get(macro, {"input": [], "output": []}))]

        ### 2. mapping components to nodes ###
        nodes = list(graph)
        old_nodes = list(graph)
        old_graph = copy.deepcopy(graph)
        out_edge = self.edge_gen("out", old_nodes, graph)
        for node in old_nodes:
            if node not in node_to_macro:
                macro = self.find_macro(graph.nodes[node])
                self.macro_dict[macro]["function"] = graph.nodes[node]["function"]
                node_to_macro[node] = [macro, copy.deepcopy(self.macro_dict[macro])]
            macro = node_to_macro[node][0]
            _, output_bitwidth, _ = self.get_macro_bitwidths(macro)
            log_info(f"node to macro [{node}]: {node_to_macro[node]}, output_bitwidth={output_bitwidth}")
            out_edge = self.edge_gen("out", old_nodes, graph)
            # Update count if it was set incorrectly
            if graph.nodes[node].get("count") != output_bitwidth:
                graph.nodes[node]["count"] = output_bitwidth
            node_attribute = graph.nodes[node]
            # node has output_bitwidth output ports
            for x in range(output_bitwidth):
                name = str(node) + "_" + str(x)
                # this port node may have been added in a previous iteration
                if not graph.has_node(name):
                    graph.add_node(name, function=node_attribute["function"], port_idx=x, name=node, count=output_bitwidth)
                if name not in node_to_macro:
                    node_to_macro[name] = [macro, copy.deepcopy(self.macro_dict[macro])]
                # node may have multiple fanouts, take care of port x for each fanout
                for output in out_edge[node]:
                    if output not in node_to_macro:
                        output_macro = self.find_macro(graph.nodes[output])
                        self.macro_dict[output_macro]["function"] = graph.nodes[output]["function"]
                        node_to_macro[output] = [output_macro, copy.deepcopy(self.macro_dict[output_macro])]
                    output_macro = node_to_macro[output][0]
                    _, output_output_bitwidth, _ = self.get_macro_bitwidths(output_macro)
                    if graph.nodes[output].get("count") != output_output_bitwidth:
                        graph.nodes[output]["count"] = output_output_bitwidth
                    # output node has output_output_bitwidth input ports
                    node_attribute = graph.nodes[output]
                    output_name = str(output) + "_" + str(x)
                    # this port node may have been added in a previous iteration
                    if not graph.has_node(output_name):
                        graph.add_node(output_name, function=node_attribute["function"], port_idx=x, name=output, count=output_output_bitwidth)
                        graph.add_edge(output_name, output)
                    if output_name not in node_to_macro:
                        node_to_macro[output_name] = [output_macro, copy.deepcopy(self.macro_dict[output_macro])]
                    graph.add_edge(name, output_name)

        openroad_run.OpenRoadRun.export_graph(graph, "result", self.directory)

        nodes = list(graph)  

        counter = 0
        log_info("generating muxes")

        # note: old graph has edges for functional self.units
        # new graph has edges for each port of the functional self.units (based on actual bitwidths), more fine grained

        # Track which nodes connect to which mux input ports (A0 or A1)
        # Key: mux_node_name (e.g., "Mux0_0"), Value: dict mapping source node to input port index (0 for A0, 1 for A1)
        mux_input_port_map = {}

        input_dict = self.edge_gen("in", old_nodes, old_graph)
        input_dict_new = self.edge_gen("in", nodes, graph)
        for node in old_nodes:
            # Get bitwidth for this node
            if node not in node_to_macro:
                macro = self.find_macro(graph.nodes[node])
                node_to_macro[node] = [macro, copy.deepcopy(self.macro_dict[macro])]
            macro = node_to_macro[node][0]
            _, output_bitwidth, num_inputs = self.get_macro_bitwidths(macro)
            
            if "Regs" in node:
                max_inputs = 1
            else:
                max_inputs = num_inputs
            while len(input_dict[node]) > max_inputs:
                input_dict = self.edge_gen("in", old_nodes, old_graph)
                target_node1 = input_dict[node][0]
                target_node2 = input_dict[node][1]

                # Get bitwidths for target nodes
                if target_node1 not in node_to_macro:
                    t1_macro = self.find_macro(old_graph.nodes[target_node1])
                    node_to_macro[target_node1] = [t1_macro, copy.deepcopy(self.macro_dict[t1_macro])]
                if target_node2 not in node_to_macro:
                    t2_macro = self.find_macro(old_graph.nodes[target_node2])
                    node_to_macro[target_node2] = [t2_macro, copy.deepcopy(self.macro_dict[t2_macro])]
                
                t1_macro = node_to_macro[target_node1][0]
                t2_macro = node_to_macro[target_node2][0]
                _, t1_output_bitwidth, _ = self.get_macro_bitwidths(t1_macro)
                _, t2_output_bitwidth, _ = self.get_macro_bitwidths(t2_macro)
                
                # Align mux bitwidth with available source/dest widths to avoid implicit nodes
                if output_bitwidth != t1_output_bitwidth or output_bitwidth != t2_output_bitwidth:
                    logger.warning(
                        "Mux bitwidth mismatch for node %s: dest=%s, src1=%s, src2=%s. "
                        "Truncating mux to min width.",
                        node,
                        output_bitwidth,
                        t1_output_bitwidth,
                        t2_output_bitwidth,
                    )
                mux_bitwidth = min(output_bitwidth, t1_output_bitwidth, t2_output_bitwidth)
                
                for x in range(mux_bitwidth):
                    name1= target_node1 + "_" + str(x)
                    name2= target_node2 + "_" + str(x)
                    node_name = node + "_" + str(x)
                    
                    new_node = "Mux" + str(counter) + "_" + str(x)
                    # port mux
                    graph.add_node(new_node, count = mux_bitwidth, function = "Mux", name=new_node, port_idx = x)
                    if graph.has_edge(name1, node_name):
                        graph.remove_edge(name1, node_name)
                    if graph.has_edge(name2, node_name):
                        graph.remove_edge(name2, node_name)
                    graph.add_edge(name1, new_node)
                    graph.add_edge(name2, new_node)
                    graph.add_edge(new_node, node_name)
                    # Track which input port each source node should use
                    mux_input_port_map[new_node] = {target_node1: 0, target_node2: 1}  # 0 = A0, 1 = A1
                    macro_output = self.find_macro(graph.nodes[new_node])
                    node_to_macro[new_node] = [macro_output, copy.deepcopy(self.macro_dict[macro_output])]
                    log_info(f"node to macro [{new_node}]: {node_to_macro[new_node]}")
                # functional unit mux
                old_graph.add_node("Mux" + str(counter), count = mux_bitwidth, function = "Mux", name="Mux" + str(counter))
                old_graph.remove_edge(target_node1, node)
                old_graph.remove_edge(target_node2, node)
                old_graph.add_edge(target_node1, "Mux" + str(counter))
                old_graph.add_edge(target_node2, "Mux" + str(counter))
                old_graph.add_edge("Mux" + str(counter), node)
                #macro_output = self.find_macro("Mux" + str(counter))
                #node_to_macro["Mux" + str(counter)] = [macro_output, copy.deepcopy(self.macro_dict[macro_output])]
                # update list of input edges to reflect the new mux, which consumes two of the previous input edges
                input_dict[node].append("Mux" + str(counter))
                input_dict[node].remove(target_node2)
                input_dict[node].remove(target_node1)
                counter += 1 
            
        ## export the final graph with muxes before DEF generation
        openroad_run.OpenRoadRun.export_graph(graph, "per_bit_result_with_muxes", self.directory)
        openroad_run.OpenRoadRun.export_graph(old_graph, "functional_unit_result_with_muxes", self.directory)
        
        ### 3.generate header ###
        header_text = []
        header_text.append("VERSION 5.8 ;")
        for line in self.lef_tech_lines:
            if "DIVIDERCHAR" in line:
                header_text.append(line.replace("\n", ""))
                break
            elif "BUSBITCHARS" in line:
                header_text.append(line.replace("\n", ""))

        header_text.append("DESIGN {} ;".format(design))
        header_text.append("UNITS DISTANCE MICRONS {} ;".format(int(self.units)))
        header_text.append("DIEAREA ( {} {} ) ( {} {} ) ;".format(self.die_coord_x1 * self.units, self.die_coord_y1 * self.units , self.die_coord_x2 * self.units, self.die_coord_y2 * self.units))


        ### 4.generate components ###
        # generating components list and a dict that translates node names to component nums in def file
        component_text = []
        number = 1
        node_to_num = {}
        nodes = list(graph)
        node_to_component_num = {}
        # basic area estimate (square microns) by summing macro footprints for each instantiated node
        area_estimate_sq_microns = 0.0
        # Track which base mux names we've already processed to avoid duplicates
        processed_mux_bases = set()
        for node in nodes:
            if "port_idx" in graph.nodes[node] and graph.nodes[node]["function"] != "Mux":
                log_info(f"Skipping component for node: {node} because it is a functional unit port")
                continue # dont generate components for functional unit ports
            # if "port_idx" not in graph.nodes[node] and graph.nodes[node]["function"] == "Mux":
            #     log_info(f"Skipping component for node: {node} because it is a functional unit mux")
            #     continue # dont generate components for functional unit muxes
            if "port_idx" in graph.nodes[node] and graph.nodes[node]["function"] == "Mux":
                # Extract base mux name (e.g., "Mux0" from "Mux0_0")
                base_mux_name = node.rsplit("_", 1)[0]
                if base_mux_name not in processed_mux_bases:
                    # Process this mux once and add it to node_to_num for all its ports
                    processed_mux_bases.add(base_mux_name)
                    component_num = format(number)
                    log_info(f"Generating component for base mux: {base_mux_name} with number: {component_num}")
                    node_to_component_num[base_mux_name] = component_num
                    macro = node_to_macro[node][0]
                    # add macro area if available
                    msize = self.macro_size_dict.get(macro)
                    if msize and self.units:
                        eff_x = msize[0] 
                        eff_y = msize[1]
                        log_info(f"Adding area for macro {macro}: {eff_x} * {eff_y} = {eff_x * eff_y}")
                        area_estimate_sq_microns += eff_x * eff_y
                    else:
                        if macro not in self.macro_size_dict:
                            logger.debug(f"No SIZE found for macro {macro}; skipping in area estimate.")
                    component_text.append("- {} {} ;".format(component_num, macro))
                    node_to_num[base_mux_name] = format(number)
                    # Also add all port nodes to node_to_num mapping to the same component
                    for port_node in nodes:
                        if port_node.startswith(base_mux_name + "_") and graph.nodes[port_node].get("function") == "Mux":
                            node_to_num[port_node] = format(number)
                            log_info(f"Mapping mux port node {port_node} to component number: {node_to_num[port_node]}")
                    log_info(f"node to num for base mux [{base_mux_name}]: {node_to_num[base_mux_name]}")
                    number += 1
                continue 
            component_num = format(number)
            log_info(f"Generating component for node: {node} with number: {component_num}")
            node_to_component_num[node] = component_num
            ## log the whole node to macro dict in a human readable way
            #log_info(f"Node to macro mapping: {node_to_macro}")
            macro = node_to_macro[node][0]
            # add macro area if available
            msize = self.macro_size_dict.get(macro)
            if msize and self.units:
                # include halo on both sides (assume macro_halo values are per-side in microns)
                eff_x = msize[0] + 2.0 * self.macro_halo_x
                eff_y = msize[1] + 2.0 * self.macro_halo_y
                log_info(f"Adding area for macro {macro}: {eff_x} * {eff_y} = {eff_x * eff_y}")
                area_estimate_sq_microns += eff_x * eff_y
            else:
                 if macro not in self.macro_size_dict:
                     logger.debug(f"No SIZE found for macro {macro}; skipping in area estimate.")
            component_text.append("- {} {} ;".format(component_num, macro))
            node_to_num[node] = format(number)
            log_info(f"node to num [{node}]: {node_to_num[node]}")
            number += 1

        component_text.insert(0, "COMPONENTS {} ;".format(len(component_text)))
        component_text.insert(len(component_text), "END COMPONENTS")

        ## 5.generate nets ###
        # generates list of nets in def file and makes two dicts that are returned
        net_text = []
        nodes_old = list(old_graph)
        node_output =  self.edge_gen("out", nodes_old, old_graph)
        node_to_macro_copy = copy.deepcopy(node_to_macro)
        net_out_dict = {}
        for node in nodes_old:
            net_list = []
            log_info(f"Generating nets for node: {node}")
            # Get output bitwidth for this node
            if node not in node_to_macro:
                macro = self.find_macro(old_graph.nodes[node])
                node_to_macro[node] = [macro, copy.deepcopy(self.macro_dict[macro])]
            macro = node_to_macro[node][0]
            _, output_bitwidth, _ = self.get_macro_bitwidths(macro)
            
            # src node
            for x in range(output_bitwidth):
                net_name = format(str(number))
                # muxes have different instances for each port because they are 2->1 bit
                # non-mux nodes have the same instance for all ports
                if old_graph.nodes[node]["function"] == "Mux":
                    name = node+"_" +str(x)
                    pin_idx = x  # Use port index x to select Z0, Z1, Z2, etc.
                else:
                    name = node
                    pin_idx = x

                component_num = node_to_num[name]
                pin_output = node_to_macro[name][1]["output"]
                if pin_idx >= len(pin_output):
                    raise ValueError(f"Pin index {pin_idx} out of range for node {name} with output pins {pin_output} (bitwidth={output_bitwidth})")
                log_info(f"Pin output for node {name}: {pin_output}, using pin_idx={pin_idx}")
                net = "- {} ( {} {} )".format(net_name, component_num, pin_output[pin_idx])

                original_net = net
                
                # used later for wire length calculation
                if node not in net_out_dict:
                    net_out_dict[node] = []
                net_out_dict[node].append(net_name)

                if not node_output[node]:
                    log_info(f"Skipping net for node {name} because it has no outputs")
                    continue

                # dst nodes
                for output in node_output[node]:
                    # Get bitwidth for output node
                    if output not in node_to_macro:
                        output_macro = self.find_macro(old_graph.nodes[output])
                        node_to_macro[output] = [output_macro, copy.deepcopy(self.macro_dict[output_macro])]
                    output_macro = node_to_macro[output][0]
                    _, output_output_bitwidth, _ = self.get_macro_bitwidths(output_macro)
                    
                    if old_graph.nodes[output]["function"] == "Mux":
                        outgoing_name = output+"_" +str(x)
                    else:
                        outgoing_name = output
                    
                    # Only connect if the port index is valid for the output node
                    if x >= output_output_bitwidth:
                        log_warning(f"Skipping connection from {node}[{x}] to {output} (bitwidth={output_output_bitwidth}) - port index out of range")
                        continue

                    #if name == outgoing_name:
                    #    log_info(f"Skipping net for node {name} and output {outgoing_name} because it is a self loop")
                    #    continue

                    if len(node_to_macro[outgoing_name][1]["input"]) == 0:
                        node_to_macro[outgoing_name][1]["input"] = copy.deepcopy(node_to_macro_copy[outgoing_name][1]["input"])
                    pin_input = node_to_macro[outgoing_name][1]["input"]
                    log_info(f"Pin input for node {outgoing_name}: {pin_input}")
                    # For mux port nodes, use the base mux name if the port node is not in node_to_num
                    lookup_name = outgoing_name
                    if outgoing_name not in node_to_num and old_graph.nodes[output]["function"] == "Mux":
                        # Extract base mux name (e.g., "Mux0" from "Mux0_0")
                        base_mux_name = outgoing_name.rsplit("_", 1)[0]
                        if base_mux_name in node_to_num:
                            lookup_name = base_mux_name
                            log_info(f"Using base mux name {base_mux_name} for lookup instead of {outgoing_name}")
                        else:
                            raise KeyError(f"Neither {outgoing_name} nor base mux {base_mux_name} found in node_to_num")
                    elif outgoing_name not in node_to_num:
                        raise KeyError(f"Node {outgoing_name} not found in node_to_num")
                    
                    # Determine which input port to use intelligently based on bitwidths and graph structure
                    selected_pin = None
                    if old_graph.nodes[output]["function"] == "Mux":
                        # For muxes, determine which input port group (A or B) based on the source node
                        port_group_idx = 0  # 0 = A, 1 = B
                        if outgoing_name in mux_input_port_map and node in mux_input_port_map[outgoing_name]:
                            port_group_idx = mux_input_port_map[outgoing_name][node]
                            log_info(f"Mux {outgoing_name} input from {node} uses port group {port_group_idx}")
                        else:
                            # Fallback: check the graph to find which input this connection corresponds to
                            # Get all incoming edges to this mux port node and find the position
                            incoming_edges = list(graph.in_edges(outgoing_name))
                            source_nodes = [edge[0] for edge in incoming_edges]
                            # Check if the source node (or its port node) is in the list
                            source_port_node = node + "_" + str(x)
                            if source_port_node in source_nodes:
                                port_group_idx = source_nodes.index(source_port_node)
                                log_info(f"Using graph-based lookup: mux {outgoing_name} input from {source_port_node} uses port group {port_group_idx}")
                            elif node in mux_input_port_map.get(outgoing_name, {}):
                                # Try direct lookup with node name
                                port_group_idx = mux_input_port_map[outgoing_name][node]
                                log_info(f"Using direct lookup: mux {outgoing_name} input from {node} uses port group {port_group_idx}")
                            else:
                                log_warning(f"Could not determine input port group for mux {outgoing_name} from source {node}, defaulting to group 0")
                                port_group_idx = 0
                        
                        # Now find the pin by name: A{x} or B{x} based on port_group_idx
                        port_letters = ['A', 'B', 'C', 'D']
                        if port_group_idx < len(port_letters):
                            expected_pin_name = f"{port_letters[port_group_idx]}{x}"
                            try:
                                input_port_idx = pin_input.index(expected_pin_name)
                                selected_pin = pin_input[input_port_idx]
                                log_info(f"Mux {outgoing_name} input from {node} (port group {port_group_idx}, bit {x}) found pin {expected_pin_name}")
                            except ValueError:
                                # Pin not found by name, try to find any pin with matching bit index from same port group
                                matching_pins = [pin for pin in pin_input if pin.endswith(str(x)) and pin.startswith(port_letters[port_group_idx])]
                                if matching_pins:
                                    selected_pin = matching_pins[0]
                                    input_port_idx = pin_input.index(selected_pin)
                                    log_warning(f"Pin {expected_pin_name} not found, using {selected_pin}")
                                else:
                                    # Last resort: use first available pin
                                    selected_pin = pin_input[0]
                                    input_port_idx = 0
                                    log_warning(f"Could not find pin {expected_pin_name} for mux {outgoing_name}, using first available pin {selected_pin}")
                        else:
                            # Port group index exceeds available letters, use bit index
                            if x < len(pin_input):
                                selected_pin = pin_input[x]
                                input_port_idx = x
                                log_warning(f"Port group {port_group_idx} exceeds port letters, using pin at bit index {x}: {selected_pin}")
                            else:
                                selected_pin = pin_input[0]
                                input_port_idx = 0
                                log_warning(f"Bit index {x} out of range for mux {outgoing_name}, using first available pin {selected_pin}")
                    else:
                        # For non-mux nodes, intelligently select input port based on available pins and bitwidths
                        # Get all input connections to this node to determine port assignment
                        input_connections = list(old_graph.in_edges(output))
                        # Find the position of this source node in the input list
                        source_positions = [i for i, (src, _) in enumerate(input_connections) if src == node]
                        
                        # Get input bitwidth for the destination node
                        output_macro = node_to_macro[output][0]
                        input_bitwidth, _, num_inputs = self.get_macro_bitwidths(output_macro)
                        
                        selected_pin = None
                        if source_positions:
                            source_position = source_positions[0]  # Which input port group (0=A, 1=B, etc.)
                            # Construct expected pin name based on port group and bit index
                            # Pin names are typically A0, A1, ..., B0, B1, etc.
                            port_letters = ['A', 'B', 'C', 'D']
                            if source_position < len(port_letters):
                                expected_pin_name = f"{port_letters[source_position]}{x}"
                                # Find the pin by name (since pins are removed as used, we can't use index)
                                try:
                                    input_port_idx = pin_input.index(expected_pin_name)
                                    selected_pin = pin_input[input_port_idx]
                                    log_info(f"Non-mux node {outgoing_name} input from {node} (position {source_position}, bit {x}) found pin {expected_pin_name} at index {input_port_idx}")
                                except ValueError:
                                    # Pin not found by name, try to find any pin with matching bit index
                                    # Look for pins ending with the bit number
                                    matching_pins = [pin for pin in pin_input if pin.endswith(str(x))]
                                    if matching_pins:
                                        # Prefer pins from the expected port group
                                        port_group_pins = [pin for pin in matching_pins if pin.startswith(port_letters[source_position])]
                                        if port_group_pins:
                                            selected_pin = port_group_pins[0]
                                            input_port_idx = pin_input.index(selected_pin)
                                            log_warning(f"Pin {expected_pin_name} not found, using {selected_pin} from same port group")
                                        else:
                                            selected_pin = matching_pins[0]
                                            input_port_idx = pin_input.index(selected_pin)
                                            log_warning(f"Pin {expected_pin_name} not found, using {selected_pin} with matching bit index")
                                    else:
                                        # Last resort: use first available pin
                                        selected_pin = pin_input[0]
                                        input_port_idx = 0
                                        log_warning(f"Could not find pin {expected_pin_name} or matching bit {x}, using first available pin {selected_pin}")
                            else:
                                # Source position exceeds available port letters, use bit index
                                if x < len(pin_input):
                                    selected_pin = pin_input[x]
                                    input_port_idx = x
                                    log_warning(f"Source position {source_position} exceeds port letters, using pin at bit index {x}: {selected_pin}")
                                else:
                                    selected_pin = pin_input[0]
                                    input_port_idx = 0
                                    log_warning(f"Bit index {x} out of range, using first available pin {selected_pin}")
                        else:
                            # Fallback: use bit index directly (assumes single input or first input)
                            if x < len(pin_input):
                                selected_pin = pin_input[x]
                                input_port_idx = x
                                log_info(f"Non-mux node {outgoing_name} input from {node} uses pin at bit index {x}: {selected_pin}")
                            else:
                                selected_pin = pin_input[0]
                                input_port_idx = 0
                                log_warning(f"Bit index {x} out of range for {outgoing_name}, using first available pin {selected_pin}")
                        
                    if selected_pin is None:
                        raise ValueError(f"Could not determine pin for {outgoing_name} input from {node} (bit {x})")
                    
                    net = net + " ( {} {} )".format(node_to_num[lookup_name], selected_pin)

                    node_to_macro[outgoing_name][1]["input"].remove(selected_pin)

                
                number += 1
                net = net + " + USE SIGNAL ;"
                log_info(f"Net: {net}")
                net_text.append(net)

        log_info(f"Generated {len(net_text)} nets.")

        net_text.insert(0, "NETS {} ;".format(len(net_text)))
        net_text.insert(len(net_text), "END NETS")
        net_text.insert(len(net_text), "END DESIGN")

        # generating pins because system will freak out without this
        pin_text = []
        pin_text.append("PINS 1 ;")
        pin_text.append("- clk + NET clk + DIRECTION INPUT + USE SIGNAL ;")
        pin_text.append("END  PINS")


        ### 6.generate rows ###
        # using calculations sourced from OpenROAD
        
        # if all nodes are Call functions, skip row generation.
        all_call_functions = True
        for node in graph.nodes():
            if graph.nodes[node].get("function", "") != "Call":
                all_call_functions = False
                break
        
        row_text = []

        if not all_call_functions:

            # core_y = self.core_coord_y2 - self.core_coord_y1
            # counter = 0

            # core_dy = core_y * self.units
            # site_dy = self.site_y * self.units
            # site_dx = self.site_x * self.units

            # row_x = math.ceil(self.core_coord_x1 * self.units / site_dx) * site_dx
            # row_y = math.ceil(self.core_coord_y1 * self.units / site_dy) * site_dy

            # while site_dy <= core_dy - counter * site_dy:
            #     text = "ROW ROW_{} {} {} {}".format(str(counter), self.site_name, str(int(row_x)), str(int(row_y + counter * site_dy)))
                
            #     if (counter + 1)%2 == 0:
            #         text += " FS "
            #     elif (counter + 1)%2 == 1:
            #         text += " N "
                
            #     num_row = 0
            #     while (self.core_coord_x2 - self.core_coord_x1) * self.units - num_row * site_dx >= site_dx:
            #         num_row = num_row + 1 

            #     text += "DO {} BY 1 ".format(str(num_row))

            #     text += "STEP {} 0 ;".format(str(int(site_dx)))

            #     counter += 1
            #     row_text.append(text)
            #     log_info(f"Generated row: {text}")

            #     log_info(f"Generated {len(row_text)} rows.")


            core_y = self.core_coord_y2 - self.core_coord_y1
            core_dy = core_y * self.units
            site_dy = self.site_y * self.units
            site_dx = self.site_x * self.units

            core_x1_dbu = int(self.core_coord_x1 * self.units)
            core_x2_dbu = int(self.core_coord_x2 * self.units)
            core_y1_dbu = int(self.core_coord_y1 * self.units)

            # Total number of potential rows (without limit)
            num_rows_y = int(math.ceil(core_dy / site_dy))
            num_sites_x = int(math.ceil((core_x2_dbu - core_x1_dbu) / site_dx))

            if DISABLE_ROW_GENERATION:
                log_info("Row generation disabled via DISABLE_ROW_GENERATION flag.")
                # Generate at least one minimal row for OpenROAD compatibility
                # (global_placement requires rows to be defined even if no std cells exist)
                num_rows_y = 1
                log_info("Generating minimal row (1 row) for OpenROAD compatibility.")

            # --- Row cap and stride logic ---
            if num_rows_y > MAX_STD_CELL_ROWS:
                stride = math.ceil(num_rows_y / MAX_STD_CELL_ROWS)
                log_info(
                    f"Row count {num_rows_y} exceeds cap ({MAX_STD_CELL_ROWS}); "
                    f"using stride {stride} to subsample evenly."
                )
            else:
                stride = 1

            # Generate every 'stride'-th row so they cover the full core height
            for row_idx in range(0, num_rows_y, stride):
                orient = "FS" if row_idx % 2 else "N"
                y = core_y1_dbu + row_idx * site_dy
                text = (
                    f"ROW ROW_{row_idx} {self.site_name} {core_x1_dbu} {y} {orient} "
                    f"DO {num_sites_x} BY 1 STEP {int(site_dx)} 0 ;"
                )
                row_text.append(text)
                if row_idx % 1000 == 0:
                    log_info(f"Generated {len(row_text)} rows so far...")

            log_info(
                f"Generated {len(row_text)} total rows "
                f"(spanning full core height, stride={stride})."
            )

        else:
            log_info("All nodes are 'Call' functions; skipping row generation.")

        #$# 7.generate track ###
        # using calculations sourced from OpenROAD
        die_coord_x2 = self.die_coord_x2 * self.units
        die_coord_y2 = self.die_coord_y2 * self.units
        lef_width = 0;
        layer_res = 0;
        layer_cap = 0;

        # tracks aren't made from the lef file; for some reason they have their own track file that sets the numbers
        track_text = []
        for line in range(len(self.lef_tech_lines)):
            if "LAYER " in self.lef_tech_lines[line] and "ROUTING" in self.lef_tech_lines[line + 1]:
                layer_name = clean(value(self.lef_tech_lines[line], "LAYER"))

                if lef_width == 0 :
                    lef_width = float(find_val("WIDTH", self.lef_tech_lines, line))
                    layer_res = float(find_val("RESISTANCE RPERSQ", self.lef_tech_lines, line))
                    layer_cap = float(find_val("CAPACITANCE CPERSQDIST", self.lef_tech_lines, line))
                
                self.layer_min_width = lef_width * self.units

                self.layer_pitch_x = float(find_val_xy("PITCH", self.lef_tech_lines, line, "x")) * self.units
                self.layer_pitch_y = float(find_val_xy("PITCH", self.lef_tech_lines, line, "y")) * self.units

                self.layer_x_offset = float(find_val_xy("OFFSET", self.lef_tech_lines, line, "x")) * self.units
                layer_y_offset = float(find_val_xy("OFFSET", self.lef_tech_lines, line, "y")) * self.units

                x_track_count = int((die_coord_x2 - self.layer_x_offset)/ self.layer_pitch_x) + 1
                origin_x = self.layer_x_offset + self.die_coord_x1

                if origin_x - self.layer_min_width / 2 < self.die_coord_x1:
                    origin_x += self.layer_pitch_x
                    x_track_count -= 1

                last_x = origin_x + (x_track_count - 1) * self.layer_pitch_x
                if last_x + self.layer_min_width / 2 > die_coord_x2:
                    x_track_count -= 1

                y_track_count = int((die_coord_y2 - layer_y_offset)/ self.layer_pitch_y) + 1
                origin_y = layer_y_offset + self.die_coord_y1

                if origin_y - self.layer_min_width / 2 < self.die_coord_y1:
                    origin_y += self.layer_pitch_y
                    y_track_count -= 1

                last_y = origin_y + (y_track_count - 1) * self.layer_pitch_y
                if last_y + self.layer_min_width / 2 > die_coord_y2:
                    y_track_count -= 1
                
                text = "TRACKS X {} DO {} STEP {} LAYER {} ;".format(int(origin_x), int(x_track_count), int(self.layer_pitch_x), layer_name)
                track_text.append(text)
                text = "TRACKS Y {} DO {} STEP {} LAYER {} ;".format(int(origin_y), int(y_track_count), int(self.layer_pitch_y), layer_name)
                track_text.append(text)
        
        log_info(f"Generated {len(track_text)} track lines.")
                    
        if not os.path.exists( self.directory + "/results/"):
            os.makedirs(self.directory + "/results/")
        if not os.path.exists( "openroad_interface/results/"):
            os.makedirs("openroad_interface/results/")
            
        with open('openroad_interface/results/first_generated.def', 'w') as f:
            for line in header_text:
                f.write(f"{line}\n")
            for line in row_text:
                f.write(f"{line}\n")
            for line in track_text:
                f.write(f"{line}\n")
            for line in component_text:
                f.write(f"{line}\n")
            for line in pin_text:
                f.write(f"{line}\n")
            for line in net_text:
                f.write(f"{line}\n")

        lef_data_dict = {"width" : lef_width, "res" : layer_res, "cap" : layer_cap, "units" : self.units}
        os.system("cp openroad_interface/results/first_generated.def " + self.directory + "/results/first_generated.def") 

        log_info(f"DEF file generation complete.")
        log_info(f"Estimated total macro area: {area_estimate_sq_microns:.2f} square microns")

        # TODO add mux function in circuit model
        # adding this so that the lib cell generator can generate the correct cell for the mux
        if "MUX2_X1" in self.macro_dict:
            self.macro_dict["MUX2_X1"]["function"] = "Not16"

        return graph, net_out_dict, node_output, lef_data_dict, node_to_num, area_estimate_sq_microns, self.macro_dict, node_to_component_num