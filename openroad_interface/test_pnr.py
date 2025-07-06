

import networkx as nx
import copy
from openroad_interface import place_n_route
from openroad_interface import estimation as est
import pandas as pd
from openroad_interface import def_generator as df
directory = "openroad_interface"
def main():
    test_file = "openroad_interface/tcl/codesign_top.tcl"
    pnr = "estimation"
    arch = nx.read_gml("src/tmp/benchmark/netlist-from-catapult.gml")
    print(arch)
    graph, net_out_dict, node_output, lef_data, node_to_num = df.def_generator(test_file, arch)
    graph = nx.read_gml("openroad_interface/results/estimated_with_mux.gml")
    wire_length_df = est.parse_route_guide_with_layer_breakdown("src/tmp/pd/results/codesign_codesign-tcl.route_guide")
    wire_length_by_edge = {}
    print(f"net_out_dict: {net_out_dict}")
    print(f"node_output: {node_output}")
    for node in net_out_dict:
        for output in node_output[node]:
            for net in net_out_dict[node]:
                print(f"for node {node} and output {output}, net is {net}")
                print(f"wire_length_df.loc[net]: {wire_length_df.loc[net]}")
                if (node, output) not in wire_length_by_edge:
                    wire_length_by_edge[(node, output)] = copy.deepcopy(wire_length_df.loc[net])
                else:
                    wire_length_by_edge[(node, output)]["total_wl"] += wire_length_df.loc[net]["total_wl"]
                    wire_length_by_edge[(node, output)]["metal1"] += wire_length_df.loc[net]["metal1"]
                    wire_length_by_edge[(node, output)]["metal2"] += wire_length_df.loc[net]["metal2"]
                    wire_length_by_edge[(node, output)]["metal3"] += wire_length_df.loc[net]["metal3"]

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(wire_length_df)
    print(wire_length_by_edge)
    wire_length_by_edge = place_n_route.mux_listing(graph, node_output, wire_length_by_edge)
    place_n_route.mux_removal(graph)
    print("after modification, wire_length_by_edge:")
    print(wire_length_by_edge)
    #dict, graph = place_n_route.place_n_route(arch, "openroad_interface/tcl/codesign_top.tcl", pnr)
    #print(dict)
    #nx.write_gml(graph, "src/tmp/benchmark/netlist-from-catapult-pnr.gml")

if __name__ == "__main__":
    main()