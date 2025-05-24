
import networkx as nx

from openroad_interface import estimation
from .working_directory import directory

def main():
    df = estimation.parse_route_guide_with_layer_breakdown(directory + "/results/codesign_codesign-tcl.route_guide")

    print(df)
    print(df.loc["_017_"]["metal1"])

if __name__ == "__main__":
    main()