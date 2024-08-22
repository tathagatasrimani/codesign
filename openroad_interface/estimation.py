import re
import math
import os

import numpy as np
import networkx as nx

from var import directory

def total_euclidean_distance(net_list: list, graph: nx.DiGraph, unit: float, node_to_num: dict):
    '''
    takes in the net list and finds the corresponding coordinate and calculates the euclidean distance from the 
    first component of the net list. it also stores these into a list edge_list for resistance calculation purposes. 
    '''
    the_component = net_list[0]
    the_node_key = list(node_to_num.keys())[list(node_to_num.values()).index(the_component)]
    result = 0.0
    edge_list = []
    for x in range(1, len(net_list)):
        node_key = list(node_to_num.keys())[list(node_to_num.values()).index(net_list[x])]
        edge = math.sqrt(pow(graph.nodes[the_node_key]["x"]/unit - graph.nodes[node_key]["x"]/unit, 2) + pow(graph.nodes[the_node_key]["y"]/unit - graph.nodes[node_key]["y"]/unit, 2))
        edge_list.append(edge)
        result += edge
    return result, edge_list

def global_estimation(wire_global_file : str = directory + "results/wire_length_global.txt"):
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

def parasitic_estimation(graph: nx.DiGraph, component_nets: dict, net_out_dict: dict, lef_data: dict, node_to_num: dict):
    '''
    calculates the resistance and capacitance using the euclidean distance and generates a .gml file with the edge attributes

    param:
        graph: digraph to pass on as argument for calculating length
        component_nets: dict that list components for the respective net id 
        net_out_dict:  dict that lists nodes and their respective net (all components utilize one output, therefore this is a same assumption to use)
        lef_data: dict with layer information (units, res, cap, width)
        node_to_num: dict that gives component id equivalent for node
    return:
        estimated_res: dict of res for each net
        estimated_res_data: list of res for each net
        estimated_cap: dict of cap for each net
        estimated_cap_data: list of cap for each net
        estimated_length: dict of length for each net
        estimated_length_data: list of length for each net
    '''
    units = lef_data["units"]
    layer_res = lef_data["res"]
    layer_cap = lef_data["cap"]
    lef_width = lef_data["width"]

    # calculating length and res 
    estimated_length_data = []
    estimated_length = {}
    estimated_res_data = []
    estimated_res = {}
    for key in component_nets:
        node_key = list(net_out_dict.keys())[list(net_out_dict.values()).index(key)]
        length, edge_list = total_euclidean_distance(component_nets[key], graph, units, node_to_num)
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
    return estimated_res, estimated_res_data, estimated_cap, estimated_cap_data, estimated_length, estimated_length_data
