import networkx as nx
import os
import shutil

graph_file = "test_files/mm_test.gml"
# 0. import macro_maker macros and add them into lef file (skip for now)

def main():
    # # 1. generate def 
    os.system("export PATH=./../../OpenROAD/build/src:$PATH")
    os.system("cp test_nangate45.tcl ./../../OpenROAD/test") 
    shutil.copyfile("test_nangate45.tcl", "../../OpenROAD/test/test.tcl")

    from def_generator import graph, net_out_dict, output_dict

    # # 2. run openroad
    os.chdir("../../OpenROAD/test")
    os.system("openroad test.tcl")
    os.chdir("../../codesign/openroad_interface")

    # 3. run parasitic_calc and length_calculations
    from parasitic_calc import net_cap, net_res
    from length_calculations import length_list

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

if __name__ == '__main__':
    main()