import re
import math
import os

import numpy as np
import networkx as nx

from var import directory

def total_euclidean_distance(net_list: list, coord_data: dict, unit: float):
    '''
    takes in the net list and finds the corresponding coordinate and calculates the euclidean distance from the 
    first component of the net list. it also stores these into a list edge_list for resistance calculation purposes. 
    '''
    the_component = net_list[0]
    result = 0.0
    edge_list = []
    for x in range(1, len(net_list)):
        edge = math.sqrt(pow(coord_data[the_component]["x"]/unit - coord_data[net_list[x]]["x"]/unit, 2) + pow(coord_data[the_component]["y"]/unit - coord_data[net_list[x]]["y"]/unit, 2))
        edge_list.append(edge)
        result += edge
    return result, edge_list

def global_estimation():
    wire_global_file = directory + "results/wire_length_global.txt"
    global_pattern = r"grt:\s_[0-9]+_\s[0-9]*\.[0-9]+\s[0-9]+"

    global_length_data = []
    wire_global_data = open(wire_global_file)
    wire_global_lines = wire_global_data.readlines()
    for line in wire_global_lines:
        if re.search(global_pattern, line) != None:
            match = re.search(global_pattern, line)
            globa_length = match.group(0).split()[2]
            global_length_data.append(float(globa_length))
    return global_length_data

def parasitic_estimation(graph, design_name: str, net_out_dict: dict, node_output: dict, lef_data: dict) -> dict:
    '''
    calculates the resistance and capacitance using the euclidean distance and generates a .gml file with the edge attributes

    param:
        graph: modified networkx graph 
        design_name: design name for naming the file 
        net_out_dict: dict that lists nodes and thier respective edges (all nodes have one output)
        node_output: dict that lists nodes and their respective output nodes
        lef_data: dict with lef file information (res, cap, width, units)

    return:
        dict: all res cap length data
    '''
    final_def_file = directory + "results/final_generated-tcl.def"
    pattern = r"_\w+_\s+\w+\s+\+\s+PLACED\s+\(\s*\d+\s+\d+\s*\)\s+\w+\s*;"
    net_pattern =  r'-\s(_\d+_)\s((?:\(\s_\d+_\s\w+\s\)\s*)+).*'
    component_pattern = r'(_\w+_)'

    units = lef_data["units"]
    layer_res = lef_data["res"]
    layer_cap = lef_data["cap"]
    lef_width = lef_data["width"]

    # going through lef file and getting macro placements and nets 
    final_def_data = open(final_def_file)
    final_def_lines = final_def_data.readlines()
    macro_coords= {}
    component_nets= {}
    for line in final_def_lines:
        if re.search(pattern, line) != None:
            coord = re.findall(r'\((.*?)\)', line)[0].split()
            match = re.search(component_pattern, line)
            macro_coords[match.group(0)] = {"x" : float(coord[0]), "y" : float(coord[1])}
        if re.search(net_pattern, line) != None:
            pins = re.findall(r'\(\s(.*?)\s\w+\s\)', line)
            match = re.search(component_pattern, line)
            component_nets[match.group(0)] = pins

    # calculating length and res 
    estimated_length_data = []
    estimated_length = {}
    estimated_res_data = []
    estimated_res = {}
    for key in component_nets:
        node_key = list(net_out_dict.keys())[list(net_out_dict.values()).index(key)]
        length, edge_list = total_euclidean_distance(component_nets[key], macro_coords, units)
        estimated_length_data.append(length)
        estimated_length[node_key] =length
        edge_list = [edge/lef_width * layer_res for edge in edge_list]
        # adding res in parallel where net has more than one edge
        if len(edge_list) > 1:
            resistance = edge_list[0]
            for x in range(1, len(edge_list)):
                resistance =  resistance * edge_list[x] / (resistance + edge_list[x])
            estimated_res_data.append(resistance)
            estimated_res[node_key] = resistance
        else:
            estimated_res_data.append(edge_list[0])
            estimated_res[node_key] = edge_list[0]

    # calculating cap
    estimated_cap_data = []
    estimated_cap = {}
    for key in estimated_length:
        cap = estimated_length[key]/lef_width * layer_cap * pow(10,4)
        estimated_cap_data.append(cap)
        estimated_cap[key] = cap

    # edge attribution 
    for output_pin in net_out_dict:
        for pin in node_output[output_pin]:
            graph[output_pin][pin]['net'] = net_out_dict[output_pin]
            graph[output_pin][pin]['net_length'] = estimated_length[output_pin]
            graph[output_pin][pin]['net_res'] = estimated_res[output_pin]
            graph[output_pin][pin]['net_cap'] = estimated_cap[output_pin]
    
    if not os.path.exists("results/"):
        os.makedirs("results/")
    nx.write_gml(graph, "results/estimated_" + design_name + ".gml")

    return {"length":estimated_length_data, "res": estimated_res_data, "cap" : estimated_cap_data}
    
