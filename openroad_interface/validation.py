import shutil
import os

import networkx as nx
import pandas as pd

from . import place_n_route as pnr
from .var import directory
from . import def_generator as df
from . import graph_plotter as gp


def pandas_organize(design_name: str, estimation_dict: dict, detailed_dict: dict):
    """
    takes data from estimations and detailed calculations and putting them in a pandas dataframe.
    param:
        design_name: design name
        estimation_dict: estimated parasitics of each net
        detailed_dict: detailed parasitics of each net

    return:
        result: pandas dataframe
    """
    design_name = design_name.replace(".gml", "")
    d1 = {
        "design": [design_name] * len(detailed_dict["res"]),
        "net_id": detailed_dict["net"],
        "method": ["openroad"] * len(detailed_dict["res"]),
        "res": detailed_dict["res"],
        "cap": detailed_dict["cap"],
        "length": detailed_dict["length"],
    }
    openroad_dataframe = pd.DataFrame(data=d1)

    d2 = {
        "design": [design_name] * len(estimation_dict["res"]),
        "net_id": estimation_dict["net"],
        "method": ["estimate"] * len(estimation_dict["res"]),
        "res": estimation_dict["res"],
        "cap": estimation_dict["cap"],
        "length": estimation_dict["length"],
    }
    estimate_dataframe = pd.DataFrame(data=d2)

    result = pd.concat([openroad_dataframe, estimate_dataframe])

    return result

def validation(
    design_name: str,
    test_directory: str, 
    arg_parasitics: str
):
    graph, net_out_dict, node_output, lef_data, node_to_num = setup(design_name, test_directory)
    dict, graph = extraction(graph, arg_parasitics, design_name, net_out_dict, node_output, lef_data, node_to_num)
    return dict, graph
    
def setup(
    design_name: str,
    test_directory: str
):
    """
    the main function. generates def file, runs openroad, does all openroad and estimated calculations.
    param:
        design_name: design name
        test_directory: tcl file directory
        arg_parasitics: detailed, estimation, or none. determines which parasitic calculation is executed.

    return:
        pandas dataframe: contains all parasitic information
    """
    print("reading graph")
    graph = nx.read_gml("src/architectures/" + design_name)

    # 1. generate def
    # export PATH=./../build/src:$PATH ./../src/hardwareModel.py
    os.system("cp " + test_directory + " ./" + directory)
    os.system("cp openroad_interface/tcl/codesign_flow.tcl ./" + directory)
    # os.system("cp tcl/codesign_flow_short.tcl ./" + directory) once you figure out how to run this
    shutil.copyfile(test_directory, directory + "test.tcl")
    graph, net_out_dict, node_output, lef_data, node_to_num = df.def_generator(
        test_directory, graph
    )

    return graph, net_out_dict, node_output, lef_data, node_to_num

def extraction(graph, arg_parasitics, design_name, net_out_dict, node_output, lef_data, node_to_num): 
    # 3. extract parasitics
    print("running extractions")
    dict = {}
    if arg_parasitics == "detailed":
        dict, graph = pnr.detailed_place_n_route(
            graph, design_name, net_out_dict, node_output, lef_data, node_to_num
        )
    elif arg_parasitics == "estimation":
        dict, graph = pnr.estimated_place_n_route(
            graph, design_name, net_out_dict, node_output, lef_data, node_to_num
        )
    elif arg_parasitics == "none":
        graph = pnr.none_place_n_route(
            graph, design_name, net_out_dict, node_output, lef_data, node_to_num
        )

    return dict, graph


if __name__ == "__main__":
    aes_arch_detailed, _ = validation(
        "aes_arch.gml", "openroad_interface/tcl/test_nangate45.tcl", "detailed"
    )
    aes_arch_estimated, _ = validation(
        "aes_arch.gml", "openroad_interface/tcl/test_nangate45.tcl", "estimation"
    )
    mm_test_detailed = validation(
        "mm_test.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "detailed"
    )
    mm_test_estimation = validation(
        "mm_test.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "estimation"
    )

    aes_arch_copy_data = pandas_organize("aes_arch_copy.gml", aes_arch_estimated, aes_arch_detailed)
    mm_test_data = pandas_organize("aes_arch_copy.gml", aes_arch_estimated, aes_arch_detailed)

    combined_data = pd.concat([aes_arch_copy_data, mm_test_data])
    combined_data.to_csv("openroad_interface/results/result_rcl.csv")

    designs = ["aes_arch_copy", "mm_test"]
    title = {"res": "Resistance", "cap": "Capacitance", "length": "Length"}
    units = {"res": "ohms log", "cap": "picofarad log", "length": "microns log"}

    gp.box_whiskers_plot(designs, units, title, show_flier=True)
    gp.box_whiskers_plot(designs, units, title, show_flier=False)
    print("graphs generated")
