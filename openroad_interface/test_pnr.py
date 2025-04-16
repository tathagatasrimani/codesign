

import networkx as nx

from openroad_interface import place_n_route

def main():
    pnr = "estimation"
    arch = nx.read_gml("openroad_interface/test_architectures/tiny.gml")
    print(arch)
    place_n_route.place_n_route(arch, "openroad_interface/tcl/test_nangate45_bigger.tcl", pnr)

if __name__ == "__main__":
    main()