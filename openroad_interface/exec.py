
def exec(design_file, test_file):
    from var import directory
    from def_generator import def_generator
    import networkx as nx
    import os
    import shutil
    import parasitic_estimation as pe
    from length_calculations import length_calculations
    import parasitic_calc as pc

    # 1. generate def 
    os.system("export PATH=./OpenROAD/build/src:$PATH") ## please figure this out 
    os.system("cp " + test_file + " ./" + directory) 
    shutil.copyfile(test_file, directory + "test.tcl")

    graph, net_out_dict, output_dict, lef_pitch, layer_res, layer_cap, units = def_generator(test_file, "test_files/" + design_file + ".gml")

    # 2. run openroad
    os.chdir(directory)
    os.system("openroad test.tcl")
    os.chdir("../..")

    # 3. run parasitic_calc and length_calculations
    net_cap, net_res = pc.parasitic_calc()
    length_list = length_calculations()

    # 4. add edge attributions
    res_graph_data = []
    cap_graph_data = []
    len_graph_data = []
    print (length_list)
    print(net_out_dict)
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
    estimation_dict = pe.parasitic_estimation(lef_pitch, layer_res, layer_cap, units)

    return [res_graph_data, estimation_dict["res"]], [cap_graph_data, estimation_dict["cap"]] , [len_graph_data, estimation_dict["length"]]


    # nx.draw(graph, with_labels=True)
    # plt.show()



if __name__ == "__main__":
    import graph_plotter as gp
    aes_arch_copy_data = exec("aes_arch_copy", "test/test_nangate45.tcl")
    mm_test_data = exec("mm_test", "test/test_nangate45.tcl")

    designs = ["aes_arch_copy", "mm_test"]
    design_res_data = [aes_arch_copy_data[0], mm_test_data[0]]
    design_cap_data = [aes_arch_copy_data[1], mm_test_data[1]]
    design_length_data = [aes_arch_copy_data[2], mm_test_data[2]]

    title = {"res":"Resistance over different designs using OpenROAD and estimation", "cap" : "Capacitance over different designs using OpenROAD and estimation", "length" : "Length over different designs using OpenROAD and estimation"}
    units = {"res" : "ohms", "cap" : "Farad * 10^-5", "length" : "microns"}

    designs_elements = {"res" : design_res_data, "cap" : design_cap_data, "length" : design_length_data}

    for element_directory in ["res", "cap", "length"]:
        gp.box_whiskers_plot("results/" + element_directory, designs, designs_elements[element_directory], openroad_color = 'red', estimated_color = 'blue', units=units[element_directory], title= title[element_directory], show_flier = True)
        gp.box_whiskers_plot("results/" + element_directory, designs, designs_elements[element_directory], openroad_color = 'red', estimated_color = 'blue', units=units[element_directory], title= title[element_directory], show_flier = False)