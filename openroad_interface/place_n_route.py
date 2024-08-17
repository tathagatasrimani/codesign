import os 

import networkx as nx

from var import directory
import length_calculations as lc
import parasitic_calc as pc

def place_n_route(graph, design_name: str, net_out_dict: dict, node_output: dict, lef_data: dict) -> dict:
    '''
    Generates required .def file for OpenROAD and generates a .gml file with the edge attributes

    params: 
        graph: modified networkx graph
        design_name: design name
        net_out_dict: dict that lists nodes and thier respective edges (all nodes have one output)
        node_output: dict that lists nodes and their respective output nodes
    returns: 
        dict: contains list of resistance, capacitance, and length information
    '''

    # run parasitic_calc and length_calculations
    net_cap, net_res = pc.parasitic_calc()
    length_dict = lc.length_calculations(lef_data["units"])

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

    return {"res": res_graph_data, "cap": cap_graph_data, "length": len_graph_data, "net": net_graph_data}


