### TODO: This file needs to be updated to work with the refactored OpenRoad interface code. 

import networkx as nx

from openroad_interface import estimation

def main():
    df = estimation.parse_route_guide_with_layer_breakdown("/results/codesign_codesign-tcl.route_guide")

    print(df)
    print(df.loc["_017_"]["metal1"])

if __name__ == "__main__":
    main()