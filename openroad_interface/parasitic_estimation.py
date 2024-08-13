from var import directory
import re
import math


def total_euclidean_distance(net_list, coord_data, unit):
    the_component = net_list[0]
    result = 0.0
    for x in range(1, len(net_list)):
        result += math.sqrt(pow(coord_data[the_component]["x"]/unit - coord_data[net_list[x]]["x"]/unit, 2) + pow(coord_data[the_component]["y"]/unit - coord_data[net_list[x]]["y"]/unit, 2))
    return result

def parasitic_estimation(lef_pitch, layer_res, layer_cap, units):
    final_def_file = directory + "results/final_generated-tcl.def"

    final_def_data = open(final_def_file)
    final_def_lines = final_def_data.readlines()
    pattern = r"_\w+_\s+\w+\s+\+\s+PLACED\s+\(\s*\d+\s+\d+\s*\)\s+\w+\s*;"
    net_pattern = r'-\s+(_\d+_)\s+\(\s+(_\d+_)\s+\w\s+\)\s+\(\s+(_\d+_)\s+\w\s+\)'
    component_pattern = r'(_\w+_)'

    macro_coords= {}
    component_nets= {}
    for line in final_def_lines:
        if re.search(pattern, line) != None:
            coord = re.findall(r'\((.*?)\)', line)[0].split()
            match = re.search(component_pattern, line)
            macro_coords[match.group(0)] = {"x" : float(coord[0]), "y" : float(coord[1])}
        if re.search(net_pattern, line) != None:
            pins = re.findall(r'\(\s(.*?)\s\w\s\)', line)
            match = re.search(component_pattern, line)
            component_nets[match.group(0)] = pins

    estimated_length_data = []
    for key in component_nets:
        estimated_length_data.append(total_euclidean_distance(component_nets[key], macro_coords, units))

    estimated_res_data = []
    estimated_cap_data = []
    for length in estimated_length_data:
        estimated_res_data.append(length/lef_pitch * layer_res)
        estimated_cap_data.append(length/lef_pitch * layer_cap * pow(10,4))
    return {"length":estimated_length_data, "res": estimated_res_data, "cap" : estimated_cap_data}
    
