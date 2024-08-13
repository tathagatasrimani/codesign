import networkx as nx
import os
import shutil
from var import *
from graph_plotter import *
from parasitic_estimation import *
# from parasitic_estimation import *



# 1. generate def 
os.system("export PATH=./../../OpenROAD/build/src:$PATH")
os.system("cp test_nangate45.tcl ./../../OpenROAD/test") 
shutil.copyfile("test_nangate45.tcl", "../../OpenROAD/test/test.tcl")

from def_generator import graph, net_out_dict, output_dict, lef_pitch, layer_res, layer_cap, units

# 2. run openroad
os.chdir("../../OpenROAD/test")
os.system("openroad test.tcl")
os.chdir("../../codesign/openroad_interface")

# 3. run parasitic_calc and length_calculations
from parasitic_calc import net_cap, net_res
from length_calculations import length_list


# 4. add edge attributions

res_graph_data = []
cap_graph_data = []
len_graph_data = []
for output_pin in net_out_dict:
    for pin in output_dict[output_pin]:
        graph[output_pin][pin]['net'] = net_out_dict[output_pin]
        graph[output_pin][pin]['net_length'] = length_list[net_out_dict[output_pin]]
        graph[output_pin][pin]['net_res'] = net_res[net_out_dict[output_pin]][3]
        graph[output_pin][pin]['net_cap'] = net_cap[net_out_dict[output_pin]]
    res_graph_data.append(float(net_res[net_out_dict[output_pin]][3]))
    cap_graph_data.append(float(net_cap[net_out_dict[output_pin]])* pow(10,4))
    len_graph_data.append(float(length_list[net_out_dict[output_pin]]))

if not os.path.exists("results/"):
    os.makedirs("results/")
nx.write_gml(graph, "results/final.gml")

# 5. generating estimations
estimation_dict = parasitic_estimation(lef_pitch, layer_res, layer_cap, units)

# 6. generating graphs
box_whiskers_plot("results/" + graph_file, True, [res_graph_data, estimation_dict["res"]], [cap_graph_data, estimation_dict["cap"]] , [len_graph_data, estimation_dict["length"]])

# nx.draw(graph, with_labels=True)
# plt.show()
