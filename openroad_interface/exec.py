# this file will be located in the OpenROAD root

import networkx as nx
import matplotlib.pyplot as plt
import os
# 0. import macro_maker macros and add them into lef file (skip for now)

# # 1. generate def 
from def_generator import graph, net_out_dict, output_dict

# # 2. run openroad
os.system("openroad ../test/test1.tcl")
# os.system("openroad test.tcl")

# 3. run parasiti_calc and length_calculations
from parasitic_calc import net_cap, net_res
from length_calculations import length_list

# print(net_cap)
# print(net_res)
# print(length_list)
# print(net_out_dict)


# 4. add edge attributions
for output_pin in net_out_dict:
    for pin in output_dict[output_pin]:
        graph[output_pin][pin]['net'] = net_out_dict[output_pin]
        graph[output_pin][pin]['net_length'] = length_list[net_out_dict[output_pin]]
        graph[output_pin][pin]['net_res'] = net_res[net_out_dict[output_pin]][3]
        graph[output_pin][pin]['net_cap'] = net_cap[net_out_dict[output_pin]]

nx.write_gml(graph, "final.gml")

# nx.draw(graph, with_labels=True)
# plt.show()