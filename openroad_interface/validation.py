import shutil
import os
import math
import sys

import networkx as nx
import pandas as pd 

from src.hardwareModel import *
from . import estimation as est
from . import place_n_route as pnr
from .var import directory
from . import def_generator as df
from . import graph_plotter as gp

def pandas_organize(design_name: str, estimation_dict: dict, detailed_dict: dict ):
    '''
    takes data from estimations and detailed calculations and putting them in a pandas dataframe. 
    param:
        design_name: design name
        estimation_dict: estimated parasitics of each net
        detailed_dict: detailed parasitics of each net 
    
    return:
        result: pandas dataframe
    '''
    d1 = {"design": [design_name] * len(detailed_dict["res"]), "net_id" : detailed_dict["net"], "method": ["openroad"] * len(detailed_dict["res"]), "res": detailed_dict["res"], "cap": detailed_dict["cap"], "length": detailed_dict["length"]}
    openroad_dataframe = pd.DataFrame(data=d1)

    d2 = {"design": [design_name] * len(estimation_dict["res"]), "net_id" : estimation_dict["net"], "method": ["estimate"] * len(estimation_dict["res"]), "res": estimation_dict["res"], "cap": estimation_dict["cap"], "length": estimation_dict["length"]}
    estimate_dataframe = pd.DataFrame(data=d2)

    result = pd.concat([openroad_dataframe, estimate_dataframe])

    return result


def validation(design_name: str, test_directory: str):
    '''
    the main function. generates def file, runs openroad, does all openroad and estimated calculations. 
    param:
        design_name: design name
        test_directory: tcl file directory
    
    return:
        pandas dataframe: contains all parasitic information
    '''
    graph = nx.read_gml("src/architectures/" + design_name + ".gml")
    # hardware = HardwareModel(path_to_graphml = graph_directory + design_name + ".gml")
    # hardware.get_total_area(self)
    # call the require function 

    # 1. generate def 
    # export PATH=./../build/src:$PATH ./../src/hardwareModel.py
    os.system("cp " + test_directory + " ./" + directory) 
    os.system("cp openroad_interface/tcl/codesign_flow.tcl ./" + directory) 
    # os.system("cp tcl/codesign_flow_short.tcl ./" + directory) once you figure out how to run this
    shutil.copyfile(test_directory, directory + "test.tcl")
    graph, net_out_dict, node_output, lef_data, node_to_num= df.def_generator(test_directory, graph)

    # 3. extract parasitics
    detailed_dict, detailed_graph = pnr.detailed_place_n_route(graph, design_name, net_out_dict, node_output, lef_data, node_to_num)
    estimation_dict, estimated_graph = pnr.estimated_place_n_route(graph, design_name, net_out_dict, node_output, lef_data, node_to_num)
    global_length = est.global_estimation()

    return pandas_organize(design_name, estimation_dict, detailed_dict)

if __name__ == "__main__":

    aes_arch_copy_data= validation("aes_arch_copy", "openroad_interface/tcl/test_nangate45.tcl")
    mm_test_data = validation("mm_test", "openroad_interface/tcl/test_nangate45_bigger.tcl")

    combined_data = pd.concat([aes_arch_copy_data, mm_test_data])
    combined_data.to_csv("openroad_interface/results/result_rcl.csv")  

    designs = ["aes_arch_copy", "mm_test"]
    title = {"res":"Resistance over different designs using OpenROAD and estimation", "cap" : "Capacitance over different designs using OpenROAD and estimation", "length" : "Length over different designs using OpenROAD and estimation"}
    units = {"res" : "ohms log", "cap" : "Farad * 10^-5 log", "length" : "microns log"}

    gp.box_whiskers_plot(designs, units, title, show_flier = True)
    gp.box_whiskers_plot(designs, units, title, show_flier = False)
      
