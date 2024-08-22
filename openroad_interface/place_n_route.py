import re
import os

import networkx as nx

from var import directory
import estimation as est  
import detailed as det

def coord_scraping(graph: nx.DiGraph, node_to_num: dict, final_def_directory : str = directory + "results/final_generated-tcl.def"):
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


def estimated_place_n_route(graph: nx.DiGraph, design_name: str, net_out_dict: dict, node_output: dict, lef_data: dict, node_to_num: dict) -> dict:
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

    # run openroad
    os.chdir(directory)
    os.system("openroad test.tcl")
    os.chdir("../..")
    
    graph, component_nets = coord_scraping(graph, node_to_num)
    estimated_res, estimated_res_data, estimated_cap, estimated_cap_data, estimated_length, estimated_length_data = est.parasitic_estimation(graph, component_nets, net_out_dict, lef_data, node_to_num)

    # edge attribution 
    net_graph_data = []
    for output_pin in net_out_dict:
        for pin in node_output[output_pin]:
            graph[output_pin][pin]['net'] = net_out_dict[output_pin]
            graph[output_pin][pin]['net_length'] = estimated_length[output_pin]
            graph[output_pin][pin]['net_res'] = estimated_res[output_pin]
            graph[output_pin][pin]['net_cap'] = estimated_cap[output_pin]
        net_graph_data.append(net_out_dict[output_pin])
    
    if not os.path.exists("results/"):
        os.makedirs("results/")
    nx.write_gml(graph, "results/estimated_" + design_name + ".gml")

    return {"length":estimated_length_data, "res": estimated_res_data, "cap" : estimated_cap_data, "net": net_graph_data}, graph


def detailed_place_n_route(graph: nx.DiGraph, design_name: str, net_out_dict: dict, node_output: dict, lef_data: dict, node_to_num: dict) -> dict:
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
    # run openroad
    os.chdir(directory)
    os.system("openroad test.tcl")
    os.chdir("../..")
    
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
        res_graph_data.append(float(net_res[net_out_dict[output_pin]]))
        cap_graph_data.append(float(net_cap[net_out_dict[output_pin]]) * pow(10,4))
        len_graph_data.append(float(length_dict[net_out_dict[output_pin]]))

    if not os.path.exists("results/"):
        os.makedirs("results/")
    nx.write_gml(graph, "results/openroad_" + design_name + ".gml")

    return {"res": res_graph_data, "cap": cap_graph_data, "length": len_graph_data, "net": net_graph_data}, graph


