import copy
import os
import math

import re 
import networkx as nx

from .var import directory
from .functions import find_val_two, find_val_xy, find_val, value, format, clean
from . import place_n_route as pnr

design = "gcd"

# make it a dict
and_gate = "AND2_X1"
xor_gate = "XOR2_X1"
mux = "MUX2_X1"
reg = "DFF_X1"
add = "Add50_40"
mult = "Mult64_40"
floordiv = "FloorDiv50_40" 
sub = "Sub50_40"
eq= "Eq50_40"

def component_finder(name: str) -> str:
    '''
    returns a blank string if the component name is not a component we need

    redo this whole function 
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
    elif floordiv.upper() in name.upper():
        return name
    elif sub.upper() in name.upper():
        return name
    elif eq.upper() in name.upper():
        return name
    else:
        return ""
    
def find_macro(name: str) -> str:
    '''
    find the corresponding macro for the given node

    redo this function 
    '''
    if "AND" in name.upper():
        return  and_gate
    if "XOR" in name.upper():
        return  xor_gate
    if name.startswith("Reg"):
        return  reg
    if "ADD" in name.upper():
        return  add
    if "MULT" in name.upper():
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


def def_generator(tcl_file_directory: str, graph: nx.DiGraph): 
    '''
     -> nx.DiGraph, dict, dict, dict (it's not working when actaully written)
    Generates required .def file for OpenROAD.

    params: 
        tcl_file_imported: tcl file directory
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

    os.system("cp openroad_interface/std_cell_lef/Nangate45_stdcell.lef" + " ./" + lef_std_file) 

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
    nodes = list(graph)
    control_nodes = list(graph)

    ### 1. pruning ###
    for node1 in control_nodes:
        if "MainMem" in graph.nodes[node1]["function"] or "Buf" in graph.nodes[node1]["function"]:
            graph.remove_node(node1)
            nodes.remove(node1)
        elif graph.nodes[node1]["function"] == "Regs" or graph.nodes[node1]["function"] == "And" or graph.nodes[node1]["function"] == "BitXor":
            graph.nodes[node1]["count"] = 16
        else:
            graph.nodes[node1]["count"] = 32

    ### 2. mapping components to nodes ###
    nodes = list(graph)
    old_nodes = list(graph)
    old_graph = copy.deepcopy(graph)
    node_to_macro = {}
    out_edge = edge_gen("out", old_nodes, graph)
    for node in old_nodes:
        macro = find_macro(node)
        out_edge = edge_gen("out", old_nodes, graph)
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
                macro_output = find_macro(output)
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
                        macro_output = find_macro(output_name)
                        node_to_macro[output_name] = [macro_output, copy.deepcopy(macro_dict[macro_output])]
                else:
                    graph.add_edge(node, output)    
                    macro_output = find_macro(output)
                    node_to_macro[output] = [macro_output, copy.deepcopy(macro_dict[macro_output])]
            node_to_macro[node] = [macro, copy.deepcopy(macro_dict[macro])]
            
    for node in old_nodes:
        if graph.nodes[node]["count"] == 16:
            graph.remove_node(node)
    pnr.export_graph(graph, "result")
    nodes = list(graph)  
    
    ### 3.mux stuff ###
    counter = 0 

    input_dict = edge_gen("in", old_nodes, old_graph)
    input_dict_new = edge_gen("in", nodes, graph)
    for node in old_nodes:
        max = 0
        if "Regs" in node:
            max = 1
        else:
            max = 2
        while len(input_dict[node]) > max:
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
                
                macro_output = find_macro(new_node)
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
    nodes_old = list(old_graph)
    node_output =  edge_gen("out", nodes_old, old_graph)
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

    node_output = edge_gen("out", nodes, graph)

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
    os.system("cp openroad_interface/results/first_generated.def " + directory + "results/first_generated.def") 

    return graph, net_out_dict, node_output, lef_data_dict, node_to_num
