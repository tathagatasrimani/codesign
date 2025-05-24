

import networkx as nx

from openroad_interface import place_n_route

def main():
    pnr = "estimation"
    arch = nx.read_gml("openroad_interface/test_architectures/netlist_0.gml")
    print(arch)
    dict, graph = place_n_route.place_n_route(arch, "openroad_interface/tcl/codesign_top.tcl", pnr)

if __name__ == "__main__":
    main()