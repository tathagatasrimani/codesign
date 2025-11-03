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
add = "Add16"
mult = "Mult16"
#mult = "Mult64_40"
#add = "ADD16_X1"
#mult = "MUL16_X1"
bitxor = "BitXor16"
floordiv = "FloorDiv16" 
sub = "Sub16"
eq= "Eq16"

DEBUG = True
def log_info(msg):
    if DEBUG:
        logger.info(msg)
def log_warning(msg):
    if DEBUG:
        logger.warning(msg)

MAX_STD_CELL_ROWS = 50000  # adjust as needed for memory/runtime


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
        elif and_gate.upper() in name.upper():
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
            return  eq
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
        
        test_file_path = os.path.join(self.directory, "tcl", test_file)

        test_file_data = open(test_file_path)
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

        log_info(f"LEF tech file: {lef_tech_file}")
        log_info(f"LEF std file: {lef_std_file}")
        log_info(f"Site name: {site_name}")

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
                        self.max_dim_macro = max(self.max_dim_macro, sx, sy)
                        macro_size_dict[macro_name] = (sx, sy)
                        macro_dict[macro_name]["area"] = sx * sy
                    except Exception:
                        logger.debug(f"Couldn't parse SIZE for macro {macro_name}: {line.strip()}")
                else:
                    logger.debug(f"SIZE line did not match regex for macro {macro_name}: {line.strip()}")


        log_info(f"Macro dict: {pprint.pformat(macro_dict)}")

        # extracting units and site size from tech file
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
            graph.nodes[node1]["count"] = 16
            """if "MainMem" in graph.nodes[node1]["function"] or "Buf" in graph.nodes[node1]["function"] or \
                "store" in graph.nodes[node1]["function"] or "load" in graph.nodes[node1]["function"]:
                graph.remove_node(node1)
                nodes.remove(node1)
            elif graph.nodes[node1]["function"] == "Regs": ##or graph.nodes[node1]["function"] == "And" or graph.nodes[node1]["function"] == "BitXor":
                graph.nodes[node1]["count"] = 16
            else:
                graph.nodes[node1]["count"] = 32"""

        ### 2. mapping components to nodes ###
        nodes = list(graph)
        old_nodes = list(graph)
        old_graph = copy.deepcopy(graph)
        node_to_macro = {}
        out_edge = self.edge_gen("out", old_nodes, graph)
        for node in old_nodes:
            macro = self.find_macro(graph.nodes[node])
            macro_dict[macro]["function"] = graph.nodes[node]
            node_to_macro[node] = [macro, copy.deepcopy(macro_dict[macro])]
            log_info(f"node to macro [{node}]: {node_to_macro[node]}")
            out_edge = self.edge_gen("out", old_nodes, graph)
            assert graph.nodes[node]["count"] == 16, f"Node {node} has {graph.nodes[node]['count']} ports, expected 16"
            node_attribute = graph.nodes[node]
            # node has 16 output ports
            for x in range(16):
                name = str(node) + "_" + str(x)
                # this port node may have been added in a previous iteration
                if not graph.has_node(name):
                    graph.add_node(name, function=node_attribute["function"], port_idx=x, name=node, count=16)
                # node may have multiple fanouts, take care of port x for each fanout
                for output in out_edge[node]:
                    assert graph.nodes[output]["count"] == 16, f"Node {output} has {graph.nodes[output]['count']} ports, expected 16"
                    # output node has 16 input ports
                    node_attribute = graph.nodes[output]
                    output_name = str(output) + "_" + str(x)
                    # this port node may have been added in a previous iteration
                    if not graph.has_node(output_name):
                        graph.add_node(output_name, function=node_attribute["function"], port_idx=x, name=output, count=16)
                        graph.add_edge(output_name, output)
                    graph.add_edge(name, output_name)
                
        # not sure why this was here
        """for node in old_nodes:
            if graph.nodes[node]["count"] == 16:
                graph.remove_node(node)"""

        openroad_run.OpenRoadRun.export_graph(graph, "result", self.directory)

        nodes = list(graph)  

        counter = 0
        log_info("generating muxes")

        # note: old graph has edges for functional units
        # new graph has edges for each of the 16 ports of the functional units, more fine grained

        input_dict = self.edge_gen("in", old_nodes, old_graph)
        input_dict_new = self.edge_gen("in", nodes, graph)
        for node in old_nodes:
            if "Regs" in node:
                max_inputs = 1
            else:
                max_inputs = 2
            while len(input_dict[node]) > max_inputs:
                input_dict = self.edge_gen("in", old_nodes, old_graph)
                target_node1 = input_dict[node][0]
                target_node2 = input_dict[node][1]
                for x in range(16):
                    assert old_graph.nodes[node]["count"] == 16, f"Node {node} has {old_graph.nodes[node]['count']} ports, expected 16"
                    assert old_graph.nodes[target_node1]["count"] == 16, f"Node {target_node1} has {old_graph.nodes[target_node1]['count']} ports, expected 16"
                    assert old_graph.nodes[target_node2]["count"] == 16, f"Node {target_node2} has {old_graph.nodes[target_node2]['count']} ports, expected 16"
                    node_name = node + "_" + str(x)
                    name1= target_node1 + "_" + str(x)
                    name2= target_node2 + "_" + str(x)
                    new_node = "Mux" + str(counter) + "_" + str(x)
                    # port mux
                    graph.add_node(new_node, count = 16, function = "Mux", name=new_node, port_idx = x)
                    if graph.has_edge(name1, node_name):
                        graph.remove_edge(name1, node_name)
                    if graph.has_edge(name2, node_name):
                        graph.remove_edge(name2, node_name)
                    graph.add_edge(name1, new_node)
                    graph.add_edge(name2, new_node)
                    graph.add_edge(new_node, node_name)
                    macro_output = self.find_macro(new_node)
                    node_to_macro[new_node] = [macro_output, copy.deepcopy(macro_dict[macro_output])]
                    log_info(f"node to macro [{new_node}]: {node_to_macro[new_node]}")
                # functional unit mux
                old_graph.add_node("Mux" + str(counter), count = 16, function = "Mux", name="Mux" + str(counter))
                old_graph.remove_edge(target_node1, node)
                old_graph.remove_edge(target_node2, node)
                old_graph.add_edge(target_node1, "Mux" + str(counter))
                old_graph.add_edge(target_node2, "Mux" + str(counter))
                old_graph.add_edge("Mux" + str(counter), node)
                #macro_output = self.find_macro("Mux" + str(counter))
                #node_to_macro["Mux" + str(counter)] = [macro_output, copy.deepcopy(macro_dict[macro_output])]
                # update list of input edges to reflect the new mux, which consumes two of the previous input edges
                input_dict[node].append("Mux" + str(counter))
                input_dict[node].remove(target_node2)
                input_dict[node].remove(target_node1)
                counter += 1 
            
        
        ### 3.generate header ###
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


        ### 4.generate components ###
        # generating components list and a dict that translates node names to component nums in def file
        component_text = []
        number = 1
        node_to_num = {}
        nodes = list(graph)
        node_to_component_num = {}
        # basic area estimate (square microns) by summing macro footprints for each instantiated node
        area_estimate_sq_microns = 0.0
        for node in nodes:
            if "port_idx" in graph.nodes[node] and graph.nodes[node]["function"] != "Mux":
                log_info(f"Skipping component for node: {node} because it is a functional unit port")
                continue # dont generate components for functional unit ports
            if "port_idx" not in graph.nodes[node] and graph.nodes[node]["function"] == "Mux":
                log_info(f"Skipping component for node: {node} because it is a functional unit mux")
                continue # dont generate components for functional unit muxes
            component_num = format(number)
            log_info(f"Generating component for node: {node} with number: {component_num}")
            node_to_component_num[node] = component_num
            ## log the whole node to macro dict in a human readable way
            #log_info(f"Node to macro mapping: {node_to_macro}")
            macro = node_to_macro[node][0]
            # add macro area if available
            msize = macro_size_dict.get(macro)
            if msize and units:
                # include halo on both sides (assume macro_halo values are per-side in microns)
                if graph.nodes[node]["function"] == "Mux":
                    eff_x = msize[0] 
                    eff_y = msize[1]
                else:
                    eff_x = msize[0] + 2.0 * self.macro_halo_x
                    eff_y = msize[1] + 2.0 * self.macro_halo_y
                log_info(f"Adding area for macro {macro}: {eff_x} * {eff_y} = {eff_x * eff_y}")
                area_estimate_sq_microns += eff_x * eff_y
            else:
                 if macro not in macro_size_dict:
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
            # src node
            for x in range(16):
                net_name = format(str(number))
                assert old_graph.nodes[node]["count"] == 16, f"Node {node} has {old_graph.nodes[node]['count']} ports, expected 16"
                # muxes have different instances for each port because they are 2->1 bit
                # non-mux nodes have the same instance for all ports because each is 32->16 bits (2 input)
                if old_graph.nodes[node]["function"] == "Mux":
                    name = node+"_" +str(x)
                    pin_idx = 0
                else:
                    name = node
                    pin_idx = x

                component_num = node_to_num[name]
                pin_output = node_to_macro[name][1]["output"]
                log_info(f"Pin output for node {name}: {pin_output}")
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
                    assert old_graph.nodes[output]["count"] == 16, f"Node {output} has {old_graph.nodes[output]['count']} ports, expected 16"
                    if old_graph.nodes[output]["function"] == "Mux":
                        outgoing_name = output+"_" +str(x)
                    else:
                        outgoing_name = output

                    #if name == outgoing_name:
                    #    log_info(f"Skipping net for node {name} and output {outgoing_name} because it is a self loop")
                    #    continue

                    if len(node_to_macro[outgoing_name][1]["input"]) == 0:
                        node_to_macro[outgoing_name][1]["input"] = copy.deepcopy(node_to_macro_copy[outgoing_name][1]["input"])
                    pin_input = node_to_macro[outgoing_name][1]["input"]
                    log_info(f"Pin input for node {outgoing_name}: {pin_input}")
                    net = net + " ( {} {} )".format(node_to_num[outgoing_name], pin_input[0])

                    node_to_macro[outgoing_name][1]["input"].remove(pin_input[0])

                
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

            # core_y = core_coord_y2 - core_coord_y1
            # counter = 0

            # core_dy = core_y * units
            # site_dy = site_y * units
            # site_dx = site_x * units

            # row_x = math.ceil(core_coord_x1 * units / site_dx) * site_dx
            # row_y = math.ceil(core_coord_y1 * units / site_dy) * site_dy

            # while site_dy <= core_dy - counter * site_dy:
            #     text = "ROW ROW_{} {} {} {}".format(str(counter), site_name, str(int(row_x)), str(int(row_y + counter * site_dy)))
                
            #     if (counter + 1)%2 == 0:
            #         text += " FS "
            #     elif (counter + 1)%2 == 1:
            #         text += " N "
                
            #     num_row = 0
            #     while (core_coord_x2 - core_coord_x1) * units - num_row * site_dx >= site_dx:
            #         num_row = num_row + 1 

            #     text += "DO {} BY 1 ".format(str(num_row))

            #     text += "STEP {} 0 ;".format(str(int(site_dx)))

            #     counter += 1
            #     row_text.append(text)
            #     log_info(f"Generated row: {text}")

            #     log_info(f"Generated {len(row_text)} rows.")


            core_y = core_coord_y2 - core_coord_y1
            core_dy = core_y * units
            site_dy = site_y * units
            site_dx = site_x * units

            core_x1_dbu = int(core_coord_x1 * units)
            core_x2_dbu = int(core_coord_x2 * units)
            core_y1_dbu = int(core_coord_y1 * units)

            # Total number of potential rows (without limit)
            num_rows_y = int(math.ceil(core_dy / site_dy))
            num_sites_x = int(math.ceil((core_x2_dbu - core_x1_dbu) / site_dx))

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
                    f"ROW ROW_{row_idx} {site_name} {core_x1_dbu} {y} {orient} "
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

        # TODO add mux function in circuit model
        # adding this so that the lib cell generator can generate the correct cell for the mux
        if "MUX2_X1" in macro_dict:
            macro_dict["MUX2_X1"]["function"] = "Invert"

        return graph, net_out_dict, node_output, lef_data_dict, node_to_num, area_estimate_sq_microns, macro_dict, node_to_component_num
