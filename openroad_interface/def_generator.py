import copy
import os
import math

import pprint
import re 
import networkx as nx
import logging

from .openroad_functions import find_val_two, find_val_xy, find_val, value, format, clean
from openroad_interface import openroad_run

design = "codesign"

logger = logging.getLogger(__name__)


# make it a dict
and_gate = "AND2_X1"
xor_gate = "XOR2_X1"
mux = "MUX2_X1"
reg = "DFF_X1"
add = "Add16_16"
mult = "Mult16_16"
#add = "Add50_40"
#mult = "Mult64_40"
bitxor = "BitXor50_40"
floordiv = "FloorDiv50_40" 
sub = "Sub50_40"
eq= "Eq50_40"

DEBUG = True
def log_info(msg):
    if DEBUG:
        logger.info(msg)
def log_warning(msg):
    if DEBUG:
        logger.warning(msg)


class DefGenerator:
    def __init__(self, cfg, codesign_root_dir):
        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.directory = os.path.join(self.codesign_root_dir, "src/tmp/pd")
        self.macro_halo_x = 0.0
        self.macro_halo_y = 0.0
        self.max_dim_macro = 0.0


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
                        logger.info(f"Parsed halo_width: {self.macro_halo_x}")
                    except Exception:
                        logger.debug(f"Failed to parse halo_width from line: {line.strip()}")
                    continue

                m_h = re.search(r"-?halo_height\s+([-+]?\d*\.?\d+)", line_clean, re.IGNORECASE)
                if m_h:
                    try:
                        self.macro_halo_y = float(m_h.group(1))
                        logger.info(f"Parsed halo_height: {self.macro_halo_y}")
                    except Exception:
                        logger.debug(f"Failed to parse halo_height from line: {line.strip()}")

        log_info(f"Using macro halo values: x = {self.macro_halo_x}, y = {self.macro_halo_y}")

    def component_finder(self, name: str) -> str:
        '''
        returns a blank string if the component name is not a component we need

        redo this whole function 
        '''
        if and_gate.upper() in name.upper():
            return  name
        elif bitxor.upper() in name.upper():
            return  name
        elif xor_gate.upper() in name.upper():
            return  name
        elif reg.upper() in name.upper():
            return  name
        elif mux.upper() in name.upper():
            return  name
        elif add.upper() in name.upper():
            return  name
        elif mult.upper() in name.upper():
            return  name
        elif floordiv.upper() in name.upper():
            return name
        elif sub.upper() in name.upper():
            return name
        elif eq.upper() in name.upper():
            return name
        else:
            return ""

    def find_macro(self, name: str) -> str:
        '''
        find the corresponding macro for the given node

        redo this function 
        '''
        if "AND" in name.upper():
            return  and_gate
        if "BITXOR" in name.upper():
            return  bitxor
        if "XOR" in name.upper():
            return  xor_gate
        if name.startswith("Reg"):
            return  reg
        if "ADD" in name.upper():
            return  add
        if "MUL" in name.upper():
            return  mult
        if "FLOORDIV" in name.upper():
            return  floordiv
        if "SUB" in name.upper():
            return  sub
        if "EQ" in name.upper():
            return  sub
        if "EQ" in name.upper():
            return  sub
        if "MUX" in name.upper():
            return  mux 
        else:
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

        var_file = None

        lef_std_file = None
        lef_tech_file =  None

        core_coord_x2 =  None 
        core_coord_x1 =  None 
        core_coord_y2 =  None 
        core_coord_y1 =  None

        die_coord_x2 = None
        die_coord_x1 = None
        die_coord_y2 = None
        die_coord_y1 = None

        site_name = None
        units = None
        site_x = None
        site_y = None
        layer_min_width = None
        layer_x_offset = None
        layer_pitch_x = None
        layer_pitch_y = None

        self.get_macro_halo_values()

        
        ### 0. reading tcl file and lef file ###
        test_file_data = open(test_file)
        test_file_lines = test_file_data.readlines()

        log_info(f"Reading tcl file: {test_file}")

        # extracting vars file, die area, and core area from tcl
        for line in test_file_lines: 
            if ".vars" in line:
                var = re.findall(r'"(.*?)"', line)
                var_file = var[0]
            if "die_area" in line:
                die = re.findall(r'{(.*?)}', line)
                die = die[0].split()
                die_coord_x1 = float(die[0])
                die_coord_y1 = float(die[1])
                die_coord_x2 = float(die[2])
                die_coord_y2 = float(die[3])
            if "core_area" in line:
                core = re.findall(r'{(.*?)}', line)
                core = core[0].split()
                core_coord_x1 = float(core[0])
                core_coord_y1 = float(core[1])
                core_coord_x2 = float(core[2])
                core_coord_y2 = float(core[3])
                max_dim_macro = max(core_coord_x2 - core_coord_x1, core_coord_y2 - core_coord_y1)
                self.max_dim_macro = max(self.max_dim_macro, max_dim_macro)

        var_file_data = open(self.directory  +"/tcl/"+ var_file) 

        # extracting lef file directories and site name
        for line in var_file_data.readlines():
            if "tech_lef" in line:
                lef_tech_file = self.directory + "/tcl/" + re.findall(r'"(.*?)"', line)[0]
            if "std_cell_lef" in line:
                lef_std_file = self.directory + "/tcl/" + re.findall(r'"(.*?)"', line)[0]
            if "site" in line:
                site = re.findall(r'"(.*?)"', line)
                site_name = site[0]

        # extracting needed macros and their respective pins from lef and puts it into a dict
        lef_std_data = open(lef_std_file)
        macro_name = None
        macro_names = []
        macro_dict = {}
        # store macro physical sizes (SIZE x BY y) from stdcell LEF
        macro_size_dict = {}
        for line in lef_std_data.readlines():
            if "MACRO" in line:
                macro_name = clean(value(line, "MACRO"))
                if self.component_finder(macro_name) != "":
                    macro_names.append(macro_name)
                    io = {}
                    macro_dict[macro_name] = io
                    io["input"] = []
                    io["output"] = []
                else:
                    macro_name = ""

            elif "PIN" in line and macro_name != "": 
                pin_name = clean(value(line, "PIN"))
                if pin_name.startswith("A") or pin_name.startswith("B") or pin_name.startswith("D"):
                    macro_dict[macro_name]["input"].append(pin_name)
                elif pin_name.startswith("Z") or pin_name.startswith("Q") or pin_name.startswith("X"):
                    macro_dict[macro_name]["output"].append(pin_name)
            # capture SIZE lines inside MACRO blocks, e.g. "SIZE 22.585 BY 16.8 ;"
            elif "SIZE" in line and macro_name:
                m_size = re.search(r"SIZE\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+BY\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if m_size:
                    try:
                        sx = float(m_size.group(1))
                        sy = float(m_size.group(2))
                        macro_size_dict[macro_name] = (sx, sy)
                    except Exception:
                        logger.debug(f"Couldn't parse SIZE for macro {macro_name}: {line.strip()}")
                else:
                    logger.debug(f"SIZE line did not match regex for macro {macro_name}: {line.strip()}")

        # extracting units and sit size from tech file
        lef_data = open(lef_tech_file)
        lef_tech_lines = lef_data.readlines()
        for line in lef_tech_lines:
            if "DATABASE MICRONS" in line:
                units = float(clean(value(line, "DATABASE MICRONS")))
            if "SITE " + site_name in line:
                site_size = find_val_two("SIZE", lef_tech_lines, lef_tech_lines.index(line))
                site_x = float(site_size[0])
                site_y = float(site_size[1])
                break

        # graph reading
        nodes = list(graph)
        control_nodes = list(graph)

        log_info(f"Full Graph: {graph}")

        log_info(f"Control nodes: {control_nodes}")

        ### 1. pruning ###
        for node1 in control_nodes:
            if "MainMem" in graph.nodes[node1]["function"] or "Buf" in graph.nodes[node1]["function"] or \
                "store" in graph.nodes[node1]["function"] or "load" in graph.nodes[node1]["function"]:
                graph.remove_node(node1)
                nodes.remove(node1)
            elif graph.nodes[node1]["function"] == "Regs": ##or graph.nodes[node1]["function"] == "And" or graph.nodes[node1]["function"] == "BitXor":
                graph.nodes[node1]["count"] = 16
            else:
                graph.nodes[node1]["count"] = 32

        ### 2. mapping components to nodes ###
        nodes = list(graph)
        old_nodes = list(graph)
        old_graph = copy.deepcopy(graph)
        node_to_macro = {}
        out_edge = self.edge_gen("out", old_nodes, graph)
        for node in old_nodes:
            macro = self.find_macro(graph.nodes[node]["function"])
            out_edge = self.edge_gen("out", old_nodes, graph)
            if graph.nodes[node]["count"] == 16:
                node_attribute = graph.nodes[node]
                for x in range(16):
                    name = str(node) + "_" + str(x)
                    if not graph.has_node(name):
                        graph.add_node(name, function=node_attribute["function"], count=16)
                    for output in out_edge[node]:
                        if graph.nodes[output]["count"] == 16:
                            node_attribute = graph.nodes[output]
                            output_name = str(output) + "_" + str(x)
                            if not graph.has_node(output_name):
                                graph.add_node(output_name, function=node_attribute["function"], count=16)
                            graph.add_edge(name, output_name)
                        else:
                            graph.add_edge(name, output)
                    macro_output = self.find_macro(graph.nodes[output]["function"])
                    node_to_macro[output] = [macro_output, copy.deepcopy(macro_dict[macro_output])]
                node_to_macro[name] = [macro, copy.deepcopy(macro_dict[macro])]
            else:
                for output in out_edge[node]:
                    if graph.nodes[output]["count"] == 16:
                        for x in range(16):
                            node_attribute = graph.nodes[output]
                            output_name = str(output) + "_" + str(x)
                            if not graph.has_node(output_name):
                                graph.add_node(output_name, function=node_attribute["function"], count=16)
                            graph.add_edge(node, output_name)
                            macro_output = self.find_macro(graph.nodes[output]["function"])
                            node_to_macro[output_name] = [macro_output, copy.deepcopy(macro_dict[macro_output])]
                    else:
                        graph.add_edge(node, output)    
                        macro_output = self.find_macro(graph.nodes[output]["function"])
                        node_to_macro[output] = [macro_output, copy.deepcopy(macro_dict[macro_output])]
                node_to_macro[node] = [macro, copy.deepcopy(macro_dict[macro])]
                
        for node in old_nodes:
            if graph.nodes[node]["count"] == 16:
                graph.remove_node(node)

        openroad_run.OpenRoadRun.export_graph(graph, "result")

        nodes = list(graph)  
        
        ### 3.mux stuff ###
        counter = 0 

        input_dict = self.edge_gen("in", old_nodes, old_graph)
        input_dict_new = self.edge_gen("in", nodes, graph)
        for node in old_nodes:
            max_inputs = 0
            if "Regs" in node:
                max_inputs = 1
            else:
                max_inputs = 2
            while len(input_dict[node]) > max_inputs:
                target_node1 = input_dict[node][0]
                target_node2 = input_dict[node][1]
                for x in range(16):
                    name1= None
                    name2= None
                    if old_graph.nodes[node]["count"] == 16:
                        node_name = node + "_" + str(x)
                    else:
                        node_name = node
                    if old_graph.nodes[target_node1]["count"] == 16 and old_graph.nodes[target_node2]["count"] == 16:
                        name1= target_node1 + "_" + str(x)
                        name2= target_node2 + "_" + str(x)
                    elif old_graph.nodes[target_node1]["count"] == 16:
                        name1= target_node1 + "_" + str(x)
                        name2= target_node2
                    elif old_graph.nodes[target_node2]["count"] == 16:
                        name1= target_node1
                        name2= target_node2 + "_" + str(x)
                    else:
                        name1= target_node1
                        name2= target_node2
                        
                    if graph.has_edge(name1, node_name):
                        graph.remove_edge(name1, node_name)
                    if graph.has_edge(name2, node_name):
                        graph.remove_edge(name2, node_name)

                    new_node = "Mux" + str(counter) + "_" + str(x)

                    graph.add_node(new_node, count = 16)
                    graph.add_edge(name1, new_node)
                    graph.add_edge(name2, new_node)
                    
                    graph.add_edge(new_node, node_name)

                    macro_output = self.find_macro(new_node)
                    node_to_macro[new_node] = [macro_output, copy.deepcopy(macro_dict[macro_output])]
                
                old_graph.remove_edge(target_node1, node)
                old_graph.remove_edge(target_node2, node)
                old_graph.add_node("Mux" + str(counter), count = 16)
                old_graph.add_edge(target_node1, "Mux" + str(counter))
                old_graph.add_edge(target_node2, "Mux" + str(counter))
                old_graph.add_edge("Mux" + str(counter), node)
                input_dict[node].append("Mux" + str(counter))
                input_dict[node].remove(target_node2)
                input_dict[node].remove(target_node1)
                counter += 1 
            
        
        ### 4.generate header ###
        header_text = []
        header_text.append("VERSION 5.8 ;")
        for line in lef_tech_lines:
            if "DIVIDERCHAR" in line:
                header_text.append(line.replace("\n", ""))
                break
            elif "BUSBITCHARS" in line:
                header_text.append(line.replace("\n", ""))

        header_text.append("DESIGN {} ;".format(design))
        header_text.append("UNITS DISTANCE MICRONS {} ;".format(int(units)))
        header_text.append("DIEAREA ( {} {} ) ( {} {} ) ;".format(die_coord_x1 * units, die_coord_y1 * units , die_coord_x2 * units, die_coord_y2 * units))


        ### 5.generate components ###
        # generating components list and a dict that translates node names to component nums in def file
        component_text = []
        number = 1
        node_to_num = {}
        nodes = list(graph)
        # basic area estimate (square microns) by summing macro footprints for each instantiated node
        area_estimate_sq_microns = 0.0
        for node in nodes:
            component_num = format(number)
            log_info(f"Generating component for node: {node} with number: {component_num}")
            ## log the whole node to macro dict in a human readable way
            log_info(f"Node to macro mapping: {node_to_macro}")
            macro = node_to_macro[node][0]
            # add macro area if available
            msize = macro_size_dict.get(macro)
            if msize and units:
                # include halo on both sides (assume macro_halo values are per-side in microns)
                eff_x = msize[0] + 2.0 * self.macro_halo_x
                eff_y = msize[1] + 2.0 * self.macro_halo_y
                area_estimate_sq_microns += eff_x * eff_y
            else:
                 if macro not in macro_size_dict:
                     logger.debug(f"No SIZE found for macro {macro}; skipping in area estimate.")
            component_text.append("- {} {} ;".format(component_num, macro))
            node_to_num[node] = format(number)
            number += 1

        component_text.insert(0, "COMPONENTS {} ;".format(len(component_text)))
        component_text.insert(len(component_text), "END COMPONENTS")

        ## 6.generate nets ###
        # generates list of nets in def file and makes two dicts that are returned
        net_text = []
        nodes_old = list(old_graph)
        node_output =  self.edge_gen("out", nodes_old, old_graph)
        node_to_macro_copy = copy.deepcopy(node_to_macro)
        net_out_dict = {}
        for node in nodes_old:
            net_list = []
            for x in range(16):
                net_name = format(str(number))
                if old_graph.nodes[node]["count"] == 16:
                    name = node+"_" +str(x)
                    component_num = node_to_num[name]
                    pin_output = node_to_macro[name][1]["output"]
                    net = "- {} ( {} {} )".format(net_name, component_num, pin_output[0])
                else:
                    name = node
                    component_num = node_to_num[node]
                    pin_output = node_to_macro[node][1]["output"]
                    net = "- {} ( {} {} )".format(net_name, component_num, pin_output[0])
                pin_output.remove(pin_output[0])
                
                if name not in net_out_dict:
                    net_out_dict[name] = []
                net_out_dict[name].append(net_name)
                for output in node_output[node]:
                    if old_graph.nodes[output]["count"] == 16:
                        outgoing_name = output+"_" +str(x)
                        pin_input = node_to_macro[outgoing_name][1]["input"]
                        if len(pin_input) == 0:
                            pin_input = copy.deepcopy(node_to_macro_copy[outgoing_name][1]["input"])
                            node_to_macro[outgoing_name][1]["input"] = copy.deepcopy(node_to_macro_copy[outgoing_name][1]["input"])
                        net = net + " ( {} {} )".format(node_to_num[outgoing_name], pin_input[0])

                        node_to_macro[outgoing_name][1]["input"].remove(pin_input[0])
                        
                    else:
                        pin_input = node_to_macro[output][1]["input"]
                        if len(pin_input) == 0:
                            pin_input = copy.deepcopy(node_to_macro_copy[output][1]["input"])
                            node_to_macro[output][1]["input"] = copy.deepcopy(node_to_macro_copy[output][1]["input"])
        
                        net = net + " ( {} {} )".format(node_to_num[output], pin_input[0])

                        node_to_macro[output][1]["input"].remove(pin_input[0])
                
                number += 1
                net = net + " + USE SIGNAL ;"
                net_text.append(net)

        log_info(f"Generated {len(net_text)} nets.")

        node_output = self.edge_gen("out", nodes, graph)

        net_text.insert(0, "NETS {} ;".format(len(net_text)))
        net_text.insert(len(net_text), "END NETS")
        net_text.insert(len(net_text), "END DESIGN")

        # generating pins because system will freak out without this
        pin_text = []
        pin_text.append("PINS 1 ;")
        pin_text.append("- clk + NET clk + DIRECTION INPUT + USE SIGNAL ;")
        pin_text.append("END  PINS")


        ### 7.generate rows ###
        # using calculations sourced from OpenROAD
        row_text = []
        core_y = core_coord_y2 - core_coord_y1
        counter = 0

        core_dy = core_y * units
        site_dy = site_y * units
        site_dx = site_x * units

        row_x = math.ceil(core_coord_x1 * units / site_dx) * site_dx
        row_y = math.ceil(core_coord_y1 * units / site_dy) * site_dy

        while site_dy <= core_dy - counter * site_dy:
            text = "ROW ROW_{} {} {} {}".format(str(counter), site_name, str(int(row_x)), str(int(row_y + counter * site_dy)))
            
            if (counter + 1)%2 == 0:
                text += " FS "
            elif (counter + 1)%2 == 1:
                text += " N "
            
            num_row = 0
            while (core_coord_x2 - core_coord_x1) * units - num_row * site_dx >= site_dx:
                num_row = num_row + 1 

            text += "DO {} BY 1 ".format(str(num_row))

            text += "STEP {} 0 ;".format(str(int(site_dx)))

            counter += 1
            row_text.append(text)
            log_info(f"Generated row: {text}")

        log_info(f"Generated {len(row_text)} rows.")

        #$# 8.generate track ###
        # using calculations sourced from OpenROAD
        die_coord_x2 *= units
        die_coord_y2 *= units
        lef_width = 0;
        layer_res = 0;
        layer_cap = 0;

        # tracks aren't made from the lef file; for some reason they have their own track file that sets the numbers
        track_text = []
        for line in range(len(lef_tech_lines)):
            if "LAYER " in lef_tech_lines[line] and "ROUTING" in lef_tech_lines[line + 1]:
                layer_name = clean(value(lef_tech_lines[line], "LAYER"))

                if lef_width == 0 :
                    lef_width = float(find_val("WIDTH", lef_tech_lines, line))
                    layer_res = float(find_val("RESISTANCE RPERSQ", lef_tech_lines, line))
                    layer_cap = float(find_val("CAPACITANCE CPERSQDIST", lef_tech_lines, line))
                
                layer_min_width = lef_width * units

                layer_pitch_x = float(find_val_xy("PITCH", lef_tech_lines, line, "x")) * units
                layer_pitch_y = float(find_val_xy("PITCH", lef_tech_lines, line, "y")) * units

                layer_x_offset = float(find_val_xy("OFFSET", lef_tech_lines, line, "x")) * units
                layer_y_offset = float(find_val_xy("OFFSET", lef_tech_lines, line, "y")) * units

                x_track_count = int((die_coord_x2 - layer_x_offset)/ layer_pitch_x) + 1
                origin_x = layer_x_offset + die_coord_x1

                if origin_x - layer_min_width / 2 < die_coord_x1:
                    origin_x += layer_pitch_x
                    x_track_count -= 1

                last_x = origin_x + (x_track_count - 1) * layer_pitch_x
                if last_x + layer_min_width / 2 > die_coord_x2:
                    x_track_count -= 1

                y_track_count = int((die_coord_y2 - layer_y_offset)/ layer_pitch_y) + 1
                origin_y = layer_y_offset + die_coord_y1

                if origin_y - layer_min_width / 2 < die_coord_y1:
                    origin_y += layer_pitch_y
                    y_track_count -= 1

                last_y = origin_y + (y_track_count - 1) * layer_pitch_y
                if last_y + layer_min_width / 2 > die_coord_y2:
                    y_track_count -= 1
                
                text = "TRACKS X {} DO {} STEP {} LAYER {} ;".format(int(origin_x), int(x_track_count), int(layer_pitch_x), layer_name)
                track_text.append(text)
                text = "TRACKS Y {} DO {} STEP {} LAYER {} ;".format(int(origin_y), int(y_track_count), int(layer_pitch_y), layer_name)
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

        lef_data_dict = {"width" : lef_width, "res" : layer_res, "cap" : layer_cap, "units" : units}
        os.system("cp openroad_interface/results/first_generated.def " + self.directory + "/results/first_generated.def") 

        log_info(f"DEF file generation complete.")
        log_info(f"Estimated total macro area: {area_estimate_sq_microns:.2f} square microns")

        return graph, net_out_dict, node_output, lef_data_dict, node_to_num, area_estimate_sq_microns


    def new_def_generator(self, test_file: str, graph: nx.DiGraph):
        """
        Clean and improved version of run_def_generator.
        
        Generates a DEF file for OpenROAD from a computational graph.
        This version is more modular, readable, and maintainable.
        
        Args:
            test_file: Path to TCL configuration file
            graph: NetworkX directed graph representing the circuit
            
        Returns:
            tuple: (modified_graph, net_output_dict, node_output_dict, 
                   lef_data_dict, node_to_num_mapping, area_estimate)
        """
        log_info("Starting new DEF generator...")

        self.get_macro_halo_values()
        
        # Step 1: Parse configuration files
        config = self._parse_configuration_files(test_file)
        
        # Step 2: Parse LEF files
        lef_data = self._parse_lef_files(config)
        
        # Step 3: Process the graph
        processed_graph, node_mappings = self._process_graph(graph, lef_data)
        
        # Step 4: Generate DEF file sections
        def_sections = self._generate_def_sections(processed_graph, node_mappings, config, lef_data)
        
        # Step 5: Write DEF file
        output_file = self._write_def_file(def_sections)
        
        log_info(f"DEF file generation complete: {output_file}")
        log_info(f"Estimated total macro area: {def_sections['area_estimate']:.2f} square microns")
        
        return (
            processed_graph,
            def_sections['net_output_dict'],
            def_sections['node_output_dict'],
            def_sections['lef_data_dict'],
            def_sections['node_to_num'],
            def_sections['area_estimate']
        )

    def _parse_configuration_files(self, test_file: str) -> dict:
        """Parse TCL and variable files to extract configuration."""
        log_info(f"Parsing configuration from: {test_file}")
        
        config = {
            'var_file': None,
            'lef_std_file': None,
            'lef_tech_file': None,
            'die_coords': {'x1': None, 'y1': None, 'x2': None, 'y2': None},
            'core_coords': {'x1': None, 'y1': None, 'x2': None, 'y2': None},
            'site_name': None
        }
        
        # Parse TCL file
        with open(test_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                if '.vars' in line:
                    var_match = re.findall(r'"(.*?)"', line)
                    if var_match:
                        config['var_file'] = var_match[0]
                        
                elif 'die_area' in line:
                    die_match = re.findall(r'{(.*?)}', line)
                    if die_match:
                        coords = die_match[0].split()
                        config['die_coords'] = {
                            'x1': float(coords[0]), 'y1': float(coords[1]),
                            'x2': float(coords[2]), 'y2': float(coords[3])
                        }
                        
                elif 'core_area' in line:
                    core_match = re.findall(r'{(.*?)}', line)
                    if core_match:
                        coords = core_match[0].split()
                        config['core_coords'] = {
                            'x1': float(coords[0]), 'y1': float(coords[1]),
                            'x2': float(coords[2]), 'y2': float(coords[3])
                        }
        
        # Parse variable file
        if config['var_file']:
            var_file_path = os.path.join(self.directory, "tcl", config['var_file'])
            with open(var_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    if 'tech_lef' in line:
                        lef_match = re.findall(r'"(.*?)"', line)
                        if lef_match:
                            config['lef_tech_file'] = os.path.join(self.directory, "tcl", lef_match[0])
                            
                    elif 'std_cell_lef' in line:
                        lef_match = re.findall(r'"(.*?)"', line)
                        if lef_match:
                            config['lef_std_file'] = os.path.join(self.directory, "tcl", lef_match[0])
                            
                    elif 'site' in line:
                        site_match = re.findall(r'"(.*?)"', line)
                        if site_match:
                            config['site_name'] = site_match[0]
        
        log_info(f"Configuration parsed: {len(config['die_coords'])} die coords, {len(config['core_coords'])} core coords")
        return config

    def _parse_lef_files(self, config: dict) -> dict:
        """Parse LEF files to extract macro and technology information."""
        log_info("Parsing LEF files...")
        
        lef_data = {
            'macro_dict': {},
            'macro_sizes': {},
            'units': None,
            'site_dimensions': {'x': None, 'y': None},
            'tech_info': {}
        }
        
        # Parse standard cell LEF
        if config['lef_std_file'] and os.path.exists(config['lef_std_file']):
            lef_data.update(self._parse_std_lef(config['lef_std_file']))
        
        # Parse technology LEF
        if config['lef_tech_file'] and os.path.exists(config['lef_tech_file']):
            lef_data.update(self._parse_tech_lef(config['lef_tech_file'], config['site_name']))
        
        log_info(f"LEF parsing complete: {len(lef_data['macro_dict'])} macros found")
        
        # Debug: Print macro pin information
        for macro_name, pin_info in lef_data['macro_dict'].items():
            log_info(f"Macro {macro_name}: {len(pin_info['input'])} inputs, {len(pin_info['output'])} outputs")
            if len(pin_info['input']) > 0:
                log_info(f"  Input pins: {pin_info['input'][:5]}{'...' if len(pin_info['input']) > 5 else ''}")
            if len(pin_info['output']) > 0:
                log_info(f"  Output pins: {pin_info['output'][:5]}{'...' if len(pin_info['output']) > 5 else ''}")
        
        return lef_data

    def _parse_std_lef(self, lef_file: str) -> dict:
        """Parse standard cell LEF file for macro definitions."""
        macro_dict = {}
        macro_sizes = {}
        current_macro = None
        
        with open(lef_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                if 'MACRO' in line:
                    macro_name = clean(value(line, "MACRO"))
                    logger.info(f"Found macro: {macro_name}")
                    if self.component_finder(macro_name):
                        logger.info(f"Component found: {macro_name}")
                        current_macro = macro_name
                        macro_dict[macro_name] = {'input': [], 'output': []}
                    else:
                        logger.info(f"Component not found: {macro_name}")
                        current_macro = None
                        
                elif 'PIN' in line and current_macro:
                    pin_name = clean(value(line, "PIN"))
                    logger.info(f"Found pin: {pin_name}")
                    if pin_name.startswith(('A', 'B', 'D')):
                        logger.info(f"Found input pin: {pin_name}")
                        macro_dict[current_macro]['input'].append(pin_name)
                        logger.info(f"Added input pin {pin_name} to macro {current_macro}")
                    elif pin_name.startswith(('Z', 'Q', 'X')):
                        logger.info(f"Found output pin: {pin_name}")
                        macro_dict[current_macro]['output'].append(pin_name)
                        logger.info(f"Added output pin {pin_name} to macro {current_macro}")
                        
                elif 'SIZE' in line and current_macro:
                    size_match = re.search(r"SIZE\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+BY\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                    if size_match:
                        logger.info(f"Found size: {size_match}")
                        try:
                            sx, sy = float(size_match.group(1)), float(size_match.group(2))
                            macro_sizes[current_macro] = (sx, sy)
                        except Exception as e:
                            logger.debug(f"Couldn't parse SIZE for macro {current_macro}: {line}")
        
        return {'macro_dict': macro_dict, 'macro_sizes': macro_sizes}

    def _parse_tech_lef(self, lef_file: str, site_name: str) -> dict:
        """Parse technology LEF file for units and site information."""
        units = None
        site_dimensions = {'x': None, 'y': None}
        tech_info = {}
        
        with open(lef_file, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            line = line.strip()
            
            if 'DATABASE MICRONS' in line:
                units = float(clean(value(line, "DATABASE MICRONS")))
                
            elif f'SITE {site_name}' in line:
                site_size = find_val_two("SIZE", lines, i)
                site_dimensions = {'x': float(site_size[0]), 'y': float(site_size[1])}
                break
        
        return {
            'units': units,
            'site_dimensions': site_dimensions,
            'tech_info': tech_info
        }

    def _process_graph(self, graph: nx.DiGraph, lef_data: dict) -> tuple:
        """Process the graph: map to macros and handle port connections."""
        log_info("Processing graph...")
        
        # Create a copy for processing
        processed_graph = copy.deepcopy(graph)
        node_mappings = {
            'node_to_macro': {},
            'node_to_num': {},
            'net_output_dict': {},
            'node_output_dict': {}
        }
        
        # Map each node to its corresponding macro
        for node in list(processed_graph.nodes()):
            macro = self.find_macro(processed_graph.nodes[node]["function"])
            node_mappings['node_to_macro'][node] = [macro, copy.deepcopy(lef_data['macro_dict'].get(macro, {}))]
        
        # Insert multiplexers for high fan-in nodes if needed
        self._insert_multiplexers(processed_graph, node_mappings['node_to_macro'], lef_data)
        
        log_info(f"Graph processing complete: {len(processed_graph.nodes())} nodes")
        return processed_graph, node_mappings


    def _insert_multiplexers(self, graph: nx.DiGraph, node_to_macro: dict, lef_data: dict):
        """Insert multiplexers for nodes with high fan-in - one mux per input port."""
        mux_counter = 0
        nodes_to_process = list(graph.nodes())
        
        for node in nodes_to_process:
            if not graph.has_node(node):
                continue
            
            # Get the number of input pins available for this node's macro
            macro_info = node_to_macro.get(node, [None, {}])[1]
            available_input_pins = macro_info.get('input', [])
            
            if not available_input_pins:
                continue
            
            in_edges = list(graph.in_edges(node))
            
            # For each input pin, we can handle 2 fan-ins (1 mux can take 2 inputs)
            # So we need (num_fan_ins - 1) muxes per input pin
            num_input_pins_needed = len(in_edges)
            num_available_input_pins = len(available_input_pins)
            
            log_info(f"Node {node}: {num_input_pins_needed} fan-ins, {num_available_input_pins} available input pins")
            
            # If we have more fan-ins than available input pins, we need to reduce fan-in
            if num_input_pins_needed > num_available_input_pins:
                # Create muxes to reduce fan-in to match available input pins
                inputs_to_connect = in_edges.copy()
                
                # Process each input pin slot
                for pin_idx in range(num_available_input_pins):
                    if len(inputs_to_connect) <= 1:
                        break  # No more fan-in reduction needed
                    
                    # Take the first input for this pin
                    if inputs_to_connect:
                        input1_source = inputs_to_connect.pop(0)[0]
                        graph.add_edge(input1_source, node)
                    
                    # If there's still a second input, create a mux
                    if inputs_to_connect:
                        input2_source = inputs_to_connect.pop(0)[0]
                        
                        # Create new multiplexer
                        mux_name = f"Mux{mux_counter}"
                        graph.add_node(mux_name, function="Mux")
                        
                        # Connect the second input to mux, then mux to node
                        graph.add_edge(input2_source, mux_name)
                        graph.add_edge(mux_name, node)
                        
                        # Update mappings
                        macro = self.find_macro("Mux")
                        node_to_macro[mux_name] = [macro, copy.deepcopy(lef_data['macro_dict'].get(macro, {}))]
                        
                        mux_counter += 1
                        log_info(f"Created mux {mux_name} for node {node} input pin {pin_idx}")
                
                # If there are still inputs left, we need more muxes
                while len(inputs_to_connect) > 0:
                    if len(inputs_to_connect) >= 2:
                        # Create a mux for two inputs
                        input1_source = inputs_to_connect.pop(0)[0]
                        input2_source = inputs_to_connect.pop(0)[0]
                        
                        mux_name = f"Mux{mux_counter}"
                        graph.add_node(mux_name, function="Mux")
                        
                        graph.add_edge(input1_source, mux_name)
                        graph.add_edge(input2_source, mux_name)
                        graph.add_edge(mux_name, node)
                        
                        macro = self.find_macro("Mux")
                        node_to_macro[mux_name] = [macro, copy.deepcopy(lef_data['macro_dict'].get(macro, {}))]
                        
                        mux_counter += 1
                        log_info(f"Created mux {mux_name} for additional fan-in to node {node}")
                    else:
                        # Single remaining input - connect directly
                        input_source = inputs_to_connect.pop(0)[0]
                        graph.add_edge(input_source, node)
            
            # Only remove original edges if we actually did multiplexer insertion
            if num_input_pins_needed > num_available_input_pins:
                for source, target in in_edges:
                    if graph.has_edge(source, target):
                        graph.remove_edge(source, target)

    def _generate_def_sections(self, graph: nx.DiGraph, node_mappings: dict, 
                              config: dict, lef_data: dict) -> dict:
        """Generate all sections of the DEF file."""
        log_info("Generating DEF file sections...")
        
        # Generate components and get mappings
        components, node_to_num, area_estimate = self._generate_components(graph, node_mappings, lef_data)
        
        # Update node_mappings with node_to_num for nets generation
        node_mappings['node_to_num'] = node_to_num
        
        # Generate nets and get output dictionaries
        nets, net_output_dict, node_output_dict = self._generate_nets(graph, node_mappings, lef_data)
        
        sections = {
            'header': self._generate_header(config, lef_data),
            'rows': self._generate_rows(config, lef_data),
            'tracks': self._generate_tracks(config, lef_data),
            'components': components,
            'pins': self._generate_pins(),
            'nets': nets,
            'area_estimate': area_estimate,
            'net_output_dict': net_output_dict,
            'node_output_dict': node_output_dict,
            'node_to_num': node_to_num,
            'lef_data_dict': {
                'width': lef_data.get('tech_info', {}).get('width', 0),
                'res': lef_data.get('tech_info', {}).get('res', 0),
                'cap': lef_data.get('tech_info', {}).get('cap', 0),
                'units': lef_data['units']
            }
        }
        
        return sections

    def _generate_header(self, config: dict, lef_data: dict) -> list:
        """Generate DEF file header section."""
        header = [
            "VERSION 5.8 ;",
            "BUSBITCHARS \"[]\" ;",
            "DIVIDERCHAR \"/\" ;",
            f"DESIGN {design} ;",
            f"UNITS DISTANCE MICRONS {int(lef_data['units'])} ;"
        ]
        
        # Die area
        die = config['die_coords']
        units = lef_data['units']
        header.append(f"DIEAREA ( {die['x1'] * units} {die['y1'] * units} ) ( {die['x2'] * units} {die['y2'] * units} ) ;")
        
        return header

    def _generate_rows(self, config: dict, lef_data: dict) -> list:
        """Generate DEF file rows section."""
        rows = []
        core = config['core_coords']
        site = lef_data['site_dimensions']
        units = lef_data['units']
        site_name = config['site_name']
        
        core_dy = (core['y2'] - core['y1']) * units
        site_dy = site['y'] * units
        site_dx = site['x'] * units
        
        row_x = math.ceil(core['x1'] * units / site_dx) * site_dx
        row_y = math.ceil(core['y1'] * units / site_dy) * site_dy
        
        counter = 0
        while site_dy <= core_dy - counter * site_dy:
            orientation = "FS" if (counter + 1) % 2 == 0 else "N"
            
            # Calculate number of sites per row
            core_width = (core['x2'] - core['x1']) * units
            num_sites = int(core_width / site_dx)
            
            row_text = f"ROW ROW_{counter} {site_name} {int(row_x)} {int(row_y + counter * site_dy)} {orientation} DO {num_sites} BY 1 STEP {int(site_dx)} 0 ;"
            rows.append(row_text)
            counter += 1
        
        return rows

    def _generate_tracks(self, config: dict, lef_data: dict) -> list:
        """Generate DEF file tracks section by parsing the tracks file."""
        tracks = []
        die = config['die_coords']
        units = lef_data['units']
        
        # Parse tracks file
        tracks_file = os.path.join(self.directory, "tcl", "codesign_files", "codesign.tracks")
        if not os.path.exists(tracks_file):
            log_info(f"Tracks file not found at {tracks_file}, using default tracks")
            return self._generate_default_tracks(config, lef_data)
        
        log_info(f"Parsing tracks from: {tracks_file}")
        
        with open(tracks_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('make_tracks') and not line.startswith('#'):
                    track_info = self._parse_track_line(line, die, units)
                    if track_info:
                        tracks.extend(track_info)
        
        log_info(f"Generated {len(tracks)} track lines")
        return tracks

    def _parse_track_line(self, line: str, die: dict, units: float) -> list:
        """Parse a make_tracks line and generate corresponding DEF track lines."""
        # Example: make_tracks metal1 -x_offset 0.095 -x_pitch 0.19 -y_offset 0.07 -y_pitch 0.14
        parts = line.split()
        if len(parts) < 9:
            logger.warning(f"Invalid track line format: {line}")
            return []
        
        layer = parts[1]  # metal1, metal2, etc.
        
        # Extract parameters
        x_offset = None
        x_pitch = None
        y_offset = None
        y_pitch = None
        
        for i in range(2, len(parts), 2):
            if i + 1 < len(parts):
                param = parts[i]
                value = float(parts[i + 1])
                
                if param == '-x_offset':
                    x_offset = value
                elif param == '-x_pitch':
                    x_pitch = value
                elif param == '-y_offset':
                    y_offset = value
                elif param == '-y_pitch':
                    y_pitch = value
        
        if x_offset is None or x_pitch is None or y_offset is None or y_pitch is None:
            logger.warning(f"Missing track parameters in line: {line}")
            return []
        
        # Convert to DEF units (microns)
        x_offset_microns = x_offset * units
        x_pitch_microns = x_pitch * units
        y_offset_microns = y_offset * units
        y_pitch_microns = y_pitch * units
        
        # Calculate die dimensions
        die_x = (die['x2'] - die['x1']) * units
        die_y = (die['y2'] - die['y1']) * units
        
        # Calculate number of tracks
        x_tracks = int((die_x - x_offset_microns) / x_pitch_microns) + 1
        y_tracks = int((die_y - y_offset_microns) / y_pitch_microns) + 1
        
        # Generate track lines
        track_lines = [
            f"TRACKS X {int(x_offset_microns)} DO {x_tracks} STEP {int(x_pitch_microns)} LAYER {layer} ;",
            f"TRACKS Y {int(y_offset_microns)} DO {y_tracks} STEP {int(y_pitch_microns)} LAYER {layer} ;"
        ]
        
        log_info(f"Generated tracks for {layer}: X={x_tracks} tracks, Y={y_tracks} tracks")
        
        return track_lines

    def _generate_default_tracks(self, config: dict, lef_data: dict) -> list:
        """Generate default tracks if tracks file is not available."""
        tracks = []
        die = config['die_coords']
        units = lef_data['units']
        
        # Basic track generation - simplified version
        die_x = (die['x2'] - die['x1']) * units
        die_y = (die['y2'] - die['y1']) * units
        
        # Simple track grid
        track_pitch = 280  # Default pitch
        origin_x = 190
        origin_y = 140
        
        x_tracks = int(die_x / track_pitch) + 1
        y_tracks = int(die_y / track_pitch) + 1
        
        tracks.extend([
            f"TRACKS X {origin_x} DO {x_tracks} STEP {track_pitch} LAYER metal1 ;",
            f"TRACKS Y {origin_y} DO {y_tracks} STEP {track_pitch} LAYER metal1 ;"
        ])
        
        return tracks

    def _generate_components(self, graph: nx.DiGraph, node_mappings: dict, lef_data: dict) -> tuple:
        """Generate DEF file components section and return mappings."""
        components = []
        node_to_num = {}
        area_estimate = 0.0
        
        for i, node in enumerate(graph.nodes(), 1):
            component_num = str(i)
            
            # Get macro info with fallback
            if node in node_mappings['node_to_macro']:
                macro = node_mappings['node_to_macro'][node][0]
            else:
                # Fallback: try to determine macro from node function
                if 'function' in graph.nodes[node]:
                    try:
                        macro = self.find_macro(graph.nodes[node]["function"])
                    except ValueError:
                        macro = "UNKNOWN"
                        logger.warning(f"No macro found for node {node}, using UNKNOWN")
                else:
                    macro = "UNKNOWN"
                    logger.warning(f"No function attribute for node {node}, using UNKNOWN")
            
            components.append(f"- {component_num} {macro} ;")
            node_to_num[node] = component_num
            
            # Calculate area estimate
            macro_size = lef_data['macro_sizes'].get(macro)
            if macro_size and lef_data['units']:
                eff_x = macro_size[0] + 2.0 * self.macro_halo_x
                eff_y = macro_size[1] + 2.0 * self.macro_halo_y
                area_estimate += eff_x * eff_y
                logger.info(f"Calculated area estimate for {macro}: {eff_x} * {eff_y} = {eff_x * eff_y}")
                max_dim_macro = max(eff_x, eff_y)
                self.max_dim_macro = max(self.max_dim_macro, max_dim_macro)
                logger.info(f"Updated max dimension macro: {self.max_dim_macro}")
        
        components.insert(0, f"COMPONENTS {len(components)} ;")
        components.append("END COMPONENTS")
        
        return components, node_to_num, area_estimate

    def _generate_pins(self) -> list:
        """Generate DEF file pins section."""
        return [
            "PINS 1 ;",
            "- clk + NET clk + DIRECTION INPUT + USE SIGNAL ;",
            "END  PINS"
        ]

    def _generate_nets(self, graph: nx.DiGraph, node_mappings: dict, lef_data: dict) -> tuple:
        """Generate DEF file nets section with proper port connections."""
        nets = []
        net_counter = len(graph.nodes()) + 1  # Start after component numbers
        net_output_dict = {}
        node_output_dict = {}
        
        # Track which output pins have been used for each node
        used_output_pins = {node: set() for node in graph.nodes()}
        used_input_pins = {node: set() for node in graph.nodes()}
        
        # Generate nets for each edge in the graph
        for source, target in graph.edges():
            # Get macro info for source and target nodes
            source_macro_info = node_mappings['node_to_macro'].get(source, [None, {}])[1]
            target_macro_info = node_mappings['node_to_macro'].get(target, [None, {}])[1]
            
            source_outputs = source_macro_info.get('output', [])
            target_inputs = target_macro_info.get('input', [])
            
            log_info(f"Creating nets from {source} to {target}")
            log_info(f"Source outputs: {source_outputs}")
            log_info(f"Target inputs: {target_inputs}")
            
            # Create multiple nets for each available output pin that connects to an input pin
            # Try to connect as many pins as possible between the two units
            nets_created = 0
            
            # Get available output and input pins
            # For fan-out: don't exclude used output pins, only exclude used input pins
            available_outputs = source_outputs  # Allow fan-out from all output pins
            available_inputs = [pin for pin in target_inputs if pin not in used_input_pins[target]]
            
            log_info(f"Available outputs from {source}: {available_outputs}")
            log_info(f"Available inputs to {target}: {available_inputs}")
            
            # Create connections for all available input pins
            # Each input pin can be driven by an output pin (with fan-out)
            for i, input_pin in enumerate(available_inputs):
                # Cycle through output pins for fan-out
                output_pin = available_outputs[i % len(available_outputs)]
                
                net_name = str(net_counter)
                net_counter += 1
                
                # Get component numbers
                source_component_num = node_mappings.get('node_to_num', {}).get(source, "1")
                target_component_num = node_mappings.get('node_to_num', {}).get(target, "1")
                
                # Create net definition
                net = f"- {net_name} ( {source_component_num} {output_pin} ) ( {target_component_num} {input_pin} ) + USE SIGNAL ;"
                nets.append(net)
                nets_created += 1
                
                # Mark input pins as used (prevent multiple connections to same input)
                # Don't mark output pins as used (allow fan-out)
                used_input_pins[target].add(input_pin)
                
                # Update output dictionaries
                if source not in net_output_dict:
                    net_output_dict[source] = []
                net_output_dict[source].append(net_name)
                
                log_info(f"Created net {net_name}: {source_component_num} {output_pin} -> {target_component_num} {input_pin}")
            
            if len(available_inputs) > 0 and len(available_outputs) == 0:
                log_info(f"Warning: No output pins available from {source} to drive {len(available_inputs)} input pins in {target}")
            elif len(available_inputs) > len(available_outputs):
                log_info(f"Info: {len(available_inputs)} input pins will be driven by {len(available_outputs)} output pins (fan-out ratio: {len(available_inputs)/len(available_outputs):.1f})")
            
            # Reset used input pins for next connection if all input pins have been used
            available_inputs = [pin for pin in target_inputs if pin not in used_input_pins[target]]
            if len(available_inputs) == 0:
                used_input_pins[target] = set()

            if nets_created == 0:
                logger.warning(f"No nets created for connection from {source} to {target}")
        
        # Generate node output dictionary
        for node in graph.nodes():
            node_output_dict[node] = list(graph.successors(node))
        
        nets.insert(0, f"NETS {len(nets)} ;")
        nets.append("END NETS")
        nets.append("END DESIGN")
        
        return nets, net_output_dict, node_output_dict

    def _write_def_file(self, sections: dict) -> str:
        """Write the complete DEF file."""
        output_dir = "openroad_interface/results"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.directory, "results"), exist_ok=True)
        
        output_file = os.path.join(output_dir, "first_generated.def")
        
        with open(output_file, 'w') as f:
            # Write all sections
            for line in sections['header']:
                f.write(f"{line}\n")
            for line in sections['rows']:
                f.write(f"{line}\n")
            for line in sections['tracks']:
                f.write(f"{line}\n")
            for line in sections['components']:
                f.write(f"{line}\n")
            for line in sections['pins']:
                f.write(f"{line}\n")
            for line in sections['nets']:
                f.write(f"{line}\n")
        
        # Copy to main results directory
        main_output = os.path.join(self.directory, "results", "first_generated.def")
        os.system(f"cp {output_file} {main_output}")
        
        return output_file