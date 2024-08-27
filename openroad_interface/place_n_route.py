import re
import os
import copy

import networkx as nx

from .var import directory
from . import estimation as est  
from . import detailed as det

def openroad_run():
    os.chdir(directory)
    print("running openroad")
    os.system("openroad test.tcl > /dev/null 2>&1")
    print("done")
    os.chdir("../../..")

def export_graph(graph, design_name, est_or_det: str):
    if not os.path.exists("openroad_interface/results/"):
        os.makedirs("openroad_interface/results/")
    nx.write_gml(graph, "openroad_interface/results/" + est_or_det + "_" + design_name + ".gml")

def mux_removal(graph: nx.DiGraph, design_name):
    graph_copy = copy.deepcopy(graph)
    for node in graph.nodes:
        if "MUX" in node.upper():
            input_nodes = list(graph_copy.in_edges(node))
            output_node = list(graph_copy.out_edges(node))[0][1]

            old_edge_1 = graph_copy[input_nodes[0][0]][node]
            old_edge_2 = graph_copy[input_nodes[1][0]][node]

            old_output = graph_copy[node][str(output_node)]
            for attribute in ["net_length", "net_cap", "net_res"]:
                if type(old_edge_1[attribute]) != list:
                    old_edge_1[attribute] = [old_edge_1[attribute]]
                if type(old_edge_2[attribute]) != list:
                    old_edge_2[attribute] = [old_edge_2[attribute]]
                old_edge_1[attribute].append(float(old_output[attribute]))
                old_edge_2[attribute].append(float(old_output[attribute]))
            graph_copy.add_edge(input_nodes[0][0], output_node, **old_edge_1)
            graph_copy.add_edge(input_nodes[1][0], output_node, **old_edge_2)
            graph_copy.remove_node(node)
    export_graph(graph_copy, design_name, "nomux")
    return graph_copy

def coord_scraping(graph: nx.DiGraph, 
                   node_to_num: dict, 
                   final_def_directory : str = directory + "results/final_generated-tcl.def"):
    '''
    going through the .def file and getting macro placements and nets 
    param:
        graph: digraph to add coordinate attribute to nodes
        node_to_num: dict that gives component id equivalent for node
        final_def_directory: final def directory, defaults to def directory in openroad
    return:
        graph: digraph with the new coordinate attributes
        component_nets: dict that list components for the respective net id 
    '''
    pattern = r"_\w+_\s+\w+\s+\+\s+PLACED\s+\(\s*\d+\s+\d+\s*\)\s+\w+\s*;"
    net_pattern =  r'-\s(_\d+_)\s((?:\(\s_\d+_\s\w+\s\)\s*)+).*'
    component_pattern = r'(_\w+_)'
    final_def_data = open(final_def_directory)
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

    for node in node_to_num:
        coord = macro_coords[node_to_num[node]]
        graph.nodes[node]['x'] = coord['x']
        graph.nodes[node]['y'] = coord['y'] 
    return graph, component_nets


def estimated_place_n_route(graph: nx.DiGraph, 
                            design_name: str, 
                            net_out_dict: dict, 
                            node_output: dict, 
                            lef_data: dict, 
                            node_to_num: dict) -> dict:
    '''
    runs openroad, calculates rcl, and then adds attributes to the graph

    params: 
        graph: networkx graph
        design_name: design name
        net_out_dict: dict that lists nodes and thier respective edges (all nodes have one output)
        node_output: dict that lists nodes and their respective output nodes
        lef_data: dict with layer information (units, res, cap, width)
        node_to_num: dict that gives component id equivalent for node
    returns: 
        dict: contains list of resistance, capacitance, length, and net data
        graph: newly modified digraph with rcl attributes
    '''
    design_name = design_name.replace(".gml", "")
    
    # run openroad
    openroad_run()
    
    graph, component_nets = coord_scraping(graph, node_to_num)
    estimated_res, estimated_res_data, estimated_cap, estimated_cap_data, estimated_length, estimated_length_data = est.parasitic_estimation(graph, component_nets, net_out_dict, lef_data, node_to_num)

    # edge attribution 
    net_graph_data = []
    for output_pin in net_out_dict:
        for pin in node_output[output_pin]:
            graph[output_pin][pin]['net'] = net_out_dict[output_pin]
            graph[output_pin][pin]['net_length'] = estimated_length[output_pin] # microns
            graph[output_pin][pin]['net_res'] = float(estimated_res[output_pin]) # ohms
            graph[output_pin][pin]['net_cap'] = float(estimated_cap[output_pin]) # picofarads
        net_graph_data.append(net_out_dict[output_pin])
    
    new_graph = mux_removal(graph, design_name)
    export_graph(new_graph, design_name, "estimate")

    return {"length":estimated_length_data, "res": estimated_res_data, "cap" : estimated_cap_data, "net": net_graph_data}, new_graph


def detailed_place_n_route(graph: nx.DiGraph, 
                           design_name: str, 
                           net_out_dict: dict, 
                           node_output: dict, 
                           lef_data: dict, 
                           node_to_num: dict) -> dict:
    '''
    runs openroad, calculates rcl, and then adds attributes to the graph

    params: 
        graph: networkx graph
        design_name: design name
        net_out_dict:  dict that lists nodes and their respective net (all components utilize one output, therefore this is a same assumption to use)
        node_output: dict that lists nodes and their respective output nodes
        lef_data: dict with layer information (units, res, cap, width)
        node_to_num: dict that gives component id equivalent for node
    returns: 
        dict: contains list of resistance, capacitance, length, and net data
        graph: newly modified digraph with rcl attributes
    '''
    design_name = design_name.replace(".gml", "")
    
    # run openroad
    openroad_run()
    
    # run parasitic_calc and length_calculations
    graph, _ = coord_scraping(graph, node_to_num)
    net_cap, net_res = det.parasitic_calc()
    length_dict = det.length_calculations(lef_data["units"])

    # add edge attributions
    net_graph_data = []
    res_graph_data = []
    cap_graph_data = []
    len_graph_data = []
    for output_pin in net_out_dict:
        for node in node_output[output_pin]:
            graph[output_pin][node]['net'] = net_out_dict[output_pin]
            graph[output_pin][node]['net_length'] = length_dict[net_out_dict[output_pin]]
            graph[output_pin][node]['net_res'] = net_res[net_out_dict[output_pin]]
            graph[output_pin][node]['net_cap'] = net_cap[net_out_dict[output_pin]]
        net_graph_data.append(net_out_dict[output_pin])
        len_graph_data.append(float(length_dict[net_out_dict[output_pin]])) # ohms
        res_graph_data.append(float(net_res[net_out_dict[output_pin]])) # res
        cap_graph_data.append(float(net_cap[net_out_dict[output_pin]])) # picofarads

    new_graph = mux_removal(graph, design_name)
    export_graph(new_graph, design_name, "detailed")

    return {"res": res_graph_data, "cap": cap_graph_data, "length": len_graph_data, "net": net_graph_data}, new_graph


