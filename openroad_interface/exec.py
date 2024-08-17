import shutil
import os
import math

import networkx as nx
import pandas as pd 

import estimation as est
from var import directory
import def_generator as df
import place_n_route as pnr
import graph_plotter as gp

def exec(design_name: str, test_file: str):
    '''
    the main function. generates def file, runs openroad, does all calculations. 
    param:
        design_name: design name
        test_file: tcl file directory
    
    return:
        pandas dataframe: contains all parasitic information
    '''

    # 1. generate def 
    # export PATH=./../build/src:$PATH
    os.system("cp " + test_file + " ./" + directory) 
    os.system("cp tcl/codesign_flow.tcl ./" + directory) 
    shutil.copyfile(test_file, directory + "test.tcl")

    graph_directory = "../src/architectures/"
    graph, net_out_dict, node_output, lef_data = df.def_generator(test_file, graph_directory + design_name + ".gml")

    # 2. run openroad
    os.chdir(directory)
    os.system("openroad test.tcl")
    os.chdir("../..")

    # 3. extract parasitics
    openroad_dict = pnr.place_n_route(graph, design_name, net_out_dict, node_output, lef_data)
    estimation_dict = est.parasitic_estimation(graph, design_name, net_out_dict, node_output, lef_data)
    global_length = est.global_estimation()

    # 4. pandas
    d1 = {"design": [design_name] * len(openroad_dict["res"]), "net_id" : openroad_dict["net"], "method": ["openroad"] * len(estimation_dict["res"]), "res": openroad_dict["res"], "cap": openroad_dict["cap"], "length": openroad_dict["length"]}
    openroad_dataframe = pd.DataFrame(data=d1)

    d2 = {"design": [design_name] * len(estimation_dict["res"]), "net_id" : openroad_dict["net"], "method": ["estimate"] * len(estimation_dict["res"]), "res": estimation_dict["res"], "cap": estimation_dict["cap"], "length": estimation_dict["length"]}
    estimate_dataframe = pd.DataFrame(data=d2)

    result = pd.concat([openroad_dataframe, estimate_dataframe])

    return result

if __name__ == "__main__":

    aes_arch_copy_data = exec("aes_arch_copy", "tcl/test_nangate45.tcl")
    mm_test_data = exec("mm_test", "tcl/test_nangate45_bigger.tcl")

    combined_data = pd.concat([aes_arch_copy_data, mm_test_data])
    combined_data.to_csv("results/result_rcl.csv")  

    designs = ["aes_arch_copy", "mm_test"]
    title = {"res":"Resistance over different designs using OpenROAD and estimation", "cap" : "Capacitance over different designs using OpenROAD and estimation", "length" : "Length over different designs using OpenROAD and estimation"}
    units = {"res" : "ohms log base 2", "cap" : "Farad * 10^-5 log base 2", "length" : "microns log base 2"}

    for element_directory in ["res", "cap", "length"]:
        gp.box_whiskers_plot_design(element_directory, designs, units=units[element_directory], title= title[element_directory], show_flier = True)
        gp.box_whiskers_plot_design(element_directory, designs, units=units[element_directory], title= title[element_directory], show_flier = False)
      