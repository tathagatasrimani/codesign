import copy
import os
import math
from typing import Any

import re 
import networkx as nx

from .var import directory
from .functions import find_val_two, find_val_xy, find_val, value, format, clean

design = "gcd"

and_gate = "AND2_X1"
xor_gate = "XOR2_X1"
reg = "DFF_X1"
mux = "MUX2_X1"
add = "Add50_40"
mult = "Mult64_40"


def component_finder(name: str) -> str:
    '''
    returns a blank string if the component name is not a component we need
    '''
    if and_gate.upper() in name.upper():
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
    else:
        return ""
    
def find_macro(name: str) -> str:
    '''
    find the corresponding macro for the given node
    '''
    if "AND" in name.upper():
        return  and_gate
    if "XOR" in name.upper():
        return  xor_gate
    if name.startswith("Reg"):
        return  reg
    if "MUX" in name.upper():
        return  mux 
    if "ADD" in name.upper():
        return  add
    if "MULT" in name.upper():
        return  mult
    else:
        return ""

def edge_gen(in_or_out, nodes, graph) -> dict:
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


def def_generator(tcl_file_directory: str, graph_file_directory: str): 
    '''
     -> nx.DiGraph, dict, dict, dict (it's not working when actaully written)
    Generates required .def file for OpenROAD.

    params: 
        tcl_file_imported: tcl file directory
        graph_file_directory: graph file directory
    
    returns: 
        graph: networkx graph that has been modified (pruned and new components)
        net_out_dict: dict that lists nodes and thier respective edges (all nodes have one output)
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

    
    ### 0. reading tcl file and lef file ###
    test_file_data = open(tcl_file_directory)
    test_file_lines = test_file_data.readlines()

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


    var_file_data = open(directory  + var_file) 

    # extracting lef file directories and site name
    for line in var_file_data.readlines():
        if "tech_lef" in line:
            lef_tech_file = directory + re.findall(r'"(.*?)"', line)[0]
        if "std_cell_lef" in line:
            lef_std_file = directory + re.findall(r'"(.*?)"', line)[0]
        if "site" in line:
            site = re.findall(r'"(.*?)"', line)
            site_name = site[0]

    os.system("cp std_cell_lef/Nangate45_stdcell.lef" + " ./" + lef_std_file) 

    # extracting needed macros and their respective pins from lef and puts it into a dict
    lef_std_data = open(lef_std_file)
    macro_name = None
    macro_names = []
    macro_dict = {}
    for line in lef_std_data.readlines():
        if "MACRO" in line:
            macro_name = clean(value(line, "MACRO"))
            if component_finder(macro_name) != "":
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
    graph = nx.read_gml(graph_file_directory)
    nodes = list(graph)
    control_nodes = list(graph)

    ### 1. pruning ###
    for node1 in control_nodes:
        if "Mem" in node1 or "Buf" in node1:
                graph.remove_node(node1)
                nodes.remove(node1)

    # generating dict of nodes and their respective input nodes
    input_dict = edge_gen("in", nodes, graph)

    ### 2. mux tree ###
    # due to each gate taking only 2 inputs, breaking down inputs into mux trees
    # this will be eventually replaced to accomadate 16 bit 

    counter = 0 
    for node in nodes:
        num = 0
        if node.startswith("Reg"):
            num = 1
        else:
            num = 2
            
        if "Add" not in node and "Mult" not in node:
            while len(input_dict[node]) > num:
                target_node1 = input_dict[node][0]
                target_node2 = input_dict[node][1]

                graph.remove_edge(target_node2, node)
                graph.remove_edge(target_node1, node)

                input_dict[node].remove(target_node2)
                input_dict[node].remove(target_node1)

                new_node = "Mux" + str(counter)
                counter += 1 

                graph.add_edge(target_node1, new_node)
                graph.add_edge(target_node2, new_node)
                
                graph.add_edge(new_node, node)
                input_dict[node].append(new_node)


    ### 3. mapping components to nodes ###
    nodes = list(graph)
    node_to_macro = {}
    for node in nodes:
        macro = find_macro(node)
        node_to_macro[node] = [macro, copy.deepcopy(macro_dict[macro])]


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
    for node in nodes:
        component_num = format(number)
        macro = node_to_macro[node][0]
        component_text.append("- {} {} ;".format(component_num, macro))
        node_to_num[node] = format(number)
        number += 1

    component_text.insert(0, "COMPONENTS {} ;".format(len(component_text)))
    component_text.insert(len(component_text), "END COMPONENTS")

    ## 6.generate nets ###
    # generates list of nets in def file and makes two dicts that are returned
    net_text = []
    node_output = edge_gen("out", nodes, graph)
    net_out_dict = {}
    for node in nodes:
        net_name = format(str(number))
        component_name = node_to_num[node]
        pin_out = node_to_macro[node][1]["output"]
        net = "- {} ( {} {} )".format(net_name, component_name, pin_out[0])
        pin_out.remove(pin_out[0])
        net_out_dict[node] = net_name

        for output in node_output[node]:
            pin_in = node_to_macro[output][1]["input"]
            net = net + " ( {} {} )".format(node_to_num[output], pin_in[0])
            pin_in.remove(pin_in[0])

        number += 1
        net = net + " + USE SIGNAL ;"
        net_text.append(net)

    net_text.insert(0, "NETS {} ;".format(len(net_text)))
    net_text.insert(len(net_text), "END NETS")
    net_text.insert(len(net_text), "END DESIGN")

    # generating pins because system will freak out without this
    pin_text = []
    pin_text.append("PINS 1 ;")
    pin_text.append("- clk + NET clk + DIRECTION INPUT + USE SIGNAL ;")
    pin_text.append("END  PINS")


    #$# 7.generate rows ###
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
                
    if not os.path.exists( directory + "results/"):
        os.makedirs(directory + "results/")
    if not os.path.exists( "results/"):
        os.makedirs("results/")
        
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
    os.system("cp openroad_interface/results/first_generated.def " + directory + "results/first_generated.def") 

    return graph, net_out_dict, node_output, lef_data_dict, node_to_num
