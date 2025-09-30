

### TODO: This file needs to be updated to work with the refactored OpenRoad interface code. 

import networkx as nx
import pandas as pd

from . import openroad_run as pnr
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

def validation():
    aes_arch_detailed, _ = pnr.place_n_route(
        "aes_arch.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "detailed"
    )
    aes_arch_estimated, _ = pnr.place_n_route(
        "aes_arch.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "estimation"
    )
    mm_test_detailed, _ = pnr.place_n_route(
        "mm_test.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "detailed"
    )
    mm_test_estimation, _ = pnr.place_n_route(
        "mm_test.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "estimation"
    )

    aes_arch_copy_data = pandas_organize("aes_arch_copy.gml", aes_arch_estimated, aes_arch_detailed)
    mm_test_data = pandas_organize("mm_test.gml", mm_test_estimation, mm_test_detailed)

    combined_data = pd.concat([aes_arch_copy_data, mm_test_data])
    combined_data.to_csv("openroad_interface/results/result_rcl.csv")

    designs = ["aes_arch_copy", "mm_test"]
    title = {"res": "Resistance", "cap": "Capacitance", "length": "Length"}
    units = {"res": "ohms log", "cap": "picofarad log", "length": "microns log"}

    gp.box_whiskers_plot(designs, units, title, show_flier=True)
    gp.box_whiskers_plot(designs, units, title, show_flier=False)
    print("graphs generated")


if __name__ == "__main__":
    validation()