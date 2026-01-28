import re
import math

import numpy as np
import networkx as nx
import pandas as pd

class Net:
    def __init__(self, net_id, segments, src : str, dsts : list[str]):
        self.net_id = net_id
        self.segments = segments
        self.src = src
        self.dsts = dsts

class Segment:
    def __init__(self, layer, length):
        self.layer = layer
        self.length = length

def parse_route_guide_with_layer_breakdown(
    filepath: str, units_per_micron: float = 2000.0, updated_graph: nx.DiGraph = None, net_id_to_src_dsts: dict = None
) -> pd.DataFrame:
    """
    Parses a route guide file and computes estimated wire lengths for each net,
    broken down by metal layer.

    Parameters:
        filepath (str): Path to the route guide file.
        units_per_micron (float): Conversion factor from internal units to microns.
        updated_graph (nx.DiGraph): Updated graph with buffers added.
        net_id_to_src_dsts (dict): Dictionary of net ids to source and destination nodes in the updated graph.
    Returns:
        pd.DataFrame: DataFrame with columns ["net", "total_wl", "metal1", "metal2", "metal3"]
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Match net names and their route block (names can be _123_, net1, or any [A-Za-z_][A-Za-z0-9_]*)
    # The format is:
    # <net_name>\n(\n<coords/layers...>\n) with the parens on separate lines
    # Use DOTALL to span multiple lines and anchor name at line start
    net_blocks = re.findall(r'(?m)^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)', content, re.DOTALL)

    results = {}
    for net_id, block in net_blocks:
        # Match each routing box and its metal layer
        boxes = re.findall(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(metal\d+)', block)
        layer_lengths = {
            "metal1": 0.0, 
            "metal2": 0.0, 
            "metal3": 0.0, 
            "metal4": 0.0,
            "metal5": 0.0,
            "metal6": 0.0,
            "metal7": 0.0,
            "metal8": 0.0,
            "metal9": 0.0,
            "metal10": 0.0
        }
        segments = []
        for x1, y1, x2, y2, layer in boxes:
            x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
            length = max(x2 - x1, y2 - y1)
            if layer in layer_lengths:
                layer_lengths[layer] += length
                segments.append(Segment(layer, length / units_per_micron))
        total_length = sum(layer_lengths.values()) / units_per_micron
        layer_lengths_microns = {k: round(v / units_per_micron, 1) for k, v in layer_lengths.items()}

        assert net_id in net_id_to_src_dsts, f"Net {net_id} not found in net_id_to_src_dsts, block: {block}"

        src, dsts = net_id_to_src_dsts[net_id]
        net = Net(net_id, segments, src, dsts)
        results[net_id] = net

    return results

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

def global_estimation(wire_global_file : str = "/results/wire_length_global.txt"):
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
        node_key = None
        for k, v in net_out_dict.items():
            if key in v:
                node_key = k
                break
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
        cap = estimated_length[key]/lef_width * layer_cap 
        estimated_cap_data.append(cap)
        estimated_cap[key] = cap
    return estimated_res, estimated_res_data, estimated_cap, estimated_cap_data, estimated_length, estimated_length_data
