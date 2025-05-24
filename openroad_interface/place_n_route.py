import re
import os
import copy
import shutil

import networkx as nx

from . import def_generator as df
from . import estimation as est
from . import detailed as det
from .working_directory import directory

def openroad_run():
    old_dir = os.getcwd()
    os.chdir(directory + "/tcl")
    print("running openroad")
    os.system(os.path.dirname(os.path.abspath(__file__)) + "/OpenROAD/build/src/openroad codesign_top.tcl > " + directory + "/codesign_pd.log")#> /dev/null 2>&1     
    print("done")
    os.chdir(old_dir)


def export_graph(graph, est_or_det: str):
    if not os.path.exists("openroad_interface/results/"):
        os.makedirs("openroad_interface/results/")
    nx.write_gml(
        graph, "openroad_interface/results/" + est_or_det + ".gml"
    )


def mux_listing(graph, node_output):
    """
    goes through the graph and finds nodes that are not Muxs. If it encounters one, it will go through
    the graph to find the path of Muxs until the another non-Mux node is found. All rcl are put into a
    list and added as an edge attribute for the non-mux node to non-mux node connection

    param:
        graph: graph with the net attributes already attached
        node_output: dict of nodes and their respective outputs
    """
    for node in graph.nodes():
        if "Mux" not in node:
            for output in node_output[node]:
                net_name = graph[node][output]["net"]
                path = []
                if "Mux" in output:
                    while "Mux" in output:
                        path.append(output)
                        output = node_output[output][0]
                graph.add_edge(node, output)
                if len(path) != 0:
                    for attribute in ["net_res", "net_cap", "net_length"]:
                        val_list = []
                        val_list.append(graph[node][path[0]][attribute])
                        node_index = 0
                        for node_index in range(1, len(path)):
                            val_list.append(
                                graph[path[node_index - 1]][path[node_index]][attribute]
                            )
                        val_list.append(graph[path[node_index]][output][attribute])
                        graph[node][output][attribute] = val_list
                    graph[node][output]["net"] = net_name


def mux_removal(graph: nx.DiGraph):
    """
    Removes the mux nodes from the graph. Does not do the connecting
    param:
        graph: graph with the new edge connections, after mux listing
    """
    reference = copy.deepcopy(graph.nodes())
    for node in reference:
        if "Mux" in node:
            graph.remove_node(node)


def coord_scraping(
    graph: nx.DiGraph,
    node_to_num: dict,
    final_def_directory: str = directory + "/results/final_generated-tcl.def",
):
    """
    going through the .def file and getting macro placements and nets
    param:
        graph: digraph to add coordinate attribute to nodes
        node_to_num: dict that gives component id equivalent for node
        final_def_directory: final def directory, defaults to def directory in openroad
    return:
        graph: digraph with the new coordinate attributes
        component_nets: dict that list components for the respective net id
    """
    pattern = r"_\w+_\s+\w+\s+\+\s+PLACED\s+\(\s*\d+\s+\d+\s*\)\s+\w+\s*;"
    net_pattern = r"-\s(_\d+_)\s((?:\(\s_\d+_\s\w+\s\)\s*)+).*"
    component_pattern = r"(_\w+_)"
    final_def_data = open(final_def_directory)
    final_def_lines = final_def_data.readlines()
    macro_coords = {}
    component_nets = {}
    for line in final_def_lines:
        if re.search(pattern, line) is not None:
            coord = re.findall(r"\((.*?)\)", line)[0].split()
            match = re.search(component_pattern, line)
            macro_coords[match.group(0)] = {"x": float(coord[0]), "y": float(coord[1])}
        if re.search(net_pattern, line) is not None:
            pins = re.findall(r"\(\s(.*?)\s\w+\s\)", line)
            match = re.search(component_pattern, line)
            component_nets[match.group(0)] = pins

    for node in node_to_num:
        coord = macro_coords[node_to_num[node]]
        graph.nodes[node]["x"] = coord["x"]
        graph.nodes[node]["y"] = coord["y"]
    return graph, component_nets

def place_n_route(
    graph: nx.DiGraph,
    test_file: str, 
    arg_parasitics: str
):
    dict = None
    if "none" not in arg_parasitics:
        graph, net_out_dict, node_output, lef_data, node_to_num = setup(graph, test_file)
        dict, graph = extraction(graph, arg_parasitics, net_out_dict, node_output, lef_data, node_to_num)
    else: 
        graph = none_place_n_route(graph)
    return dict, graph
    
def setup(
    graph: nx.DiGraph,
    test_file: str
):
    """
    the main function. generates def file, runs openroad, does all openroad and estimated calculations.
    param:
        graph: hardware netlist graph
        test_file: tcl file
        arg_parasitics: detailed, estimation, or none. determines which parasitic calculation is executed.

    return:
        pandas dataframe: contains all parasitic information
    """

    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    shutil.copytree(os.path.dirname(os.path.abspath(__file__)) + "/tcl", directory + "/tcl")
    shutil.copytree(os.path.dirname(os.path.abspath(__file__)) + "/results", directory + "/results")

    graph, net_out_dict, node_output, lef_data, node_to_num = df.def_generator(
        test_file, graph
    )

    return graph, net_out_dict, node_output, lef_data, node_to_num

def extraction(graph, arg_parasitics, net_out_dict, node_output, lef_data, node_to_num): 
    # 3. extract parasitics
    print("running extractions")
    dict = {}
    if arg_parasitics == "detailed":
        dict, graph = detailed_place_n_route(
            graph, net_out_dict, node_output, lef_data, node_to_num
        )
    elif arg_parasitics == "estimation":
        dict, graph = estimated_place_n_route(
            graph, net_out_dict, node_output, lef_data, node_to_num
        )

    return dict, graph

def estimated_place_n_route(
    graph: nx.DiGraph,
    net_out_dict: dict,
    node_output: dict,
    lef_data: dict,
    node_to_num: dict,
) -> dict:
    """
    runs openroad, calculates rcl, and then adds attributes to the graph

    params:
        graph: networkx graph
        net_out_dict: dict that lists nodes and thier respective edges (all nodes have one output)
        node_output: dict that lists nodes and their respective output nodes
        lef_data: dict with layer information (units, res, cap, width)
        node_to_num: dict that gives component id equivalent for node
    returns:
        dict: contains list of resistance, capacitance, length, and net data
        graph: newly modified digraph with rcl attributes
    """

    # run openroad
    openroad_run()

    wire_length_df = est.parse_route_guide_with_layer_breakdown(directory + "/results/codesign_codesign-tcl.route_guide")
    wire_length_by_edge = {}
    for node in net_out_dict:
        for output in node_output[node]:
            for net in net_out_dict[node]:
                if (node, output) not in wire_length_by_edge:
                    wire_length_by_edge[(node, output)] = wire_length_df.loc[net]
                else:
                    wire_length_by_edge[(node, output)] += wire_length_df.loc[net]

    #mux_listing(graph, node_output)
    #mux_removal(graph)

    export_graph(graph, "estimated_nomux")

    return wire_length_by_edge, graph


def detailed_place_n_route(
    graph: nx.DiGraph,
    net_out_dict: dict,
    node_output: dict,
    lef_data: dict,
    node_to_num: dict,
) -> dict:
    """
    runs openroad, calculates rcl, and then adds attributes to the graph

    params:
        graph: networkx graph
        net_out_dict:  dict that lists nodes and their respective net (all components utilize one output, therefore this is a same assumption to use)
        node_output: dict that lists nodes and their respective output nodes
        lef_data: dict with layer information (units, res, cap, width)
        node_to_num: dict that gives component id equivalent for node
    returns:
        dict: contains list of resistance, capacitance, length, and net data
        graph: newly modified digraph with rcl attributes
    """

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
    
    for output_net in net_out_dict:
        for net in net_out_dict[output_net]:
            for node in node_output[output_net]:
                graph[output_net][node]["net"] = net
                graph[output_net][node]["net_length"] = length_dict[net]
                graph[output_net][node]["net_res"] = float(net_res[net])
                graph[output_net][node]["net_cap"] = float(net_cap[net])
            net_graph_data.append(net)
            len_graph_data.append(float(length_dict[net]))  # length
            res_graph_data.append(float(net_res[net]) if net in net_res else 0)  # ohms
            cap_graph_data.append(float(net_cap[net]) if net in net_cap else 0)  # picofarads
        
        

    export_graph(graph, "detailed")

    mux_listing(graph, node_output)
    mux_removal(graph)

    export_graph(graph, "detailed_nomux")

    return {
        "res": res_graph_data,
        "cap": cap_graph_data,
        "length": len_graph_data,
        "net": net_graph_data,
    }, graph


def none_place_n_route(
    graph: nx.DiGraph,
) -> dict:
    """
    runs openroad, calculates rcl, and then adds attributes to the graph
    params:
        graph: networkx graph
    returns:
        graph: newly modified digraph with rcl attributes
    """

    # edge attribution
    for u, v in graph.edges():
        graph[u][v]["net"] = 0
        graph[u][v]["net_length"] = 0
        graph[u][v]["net_res"] = 0
        graph[u][v]["net_cap"] = 0


    return graph
