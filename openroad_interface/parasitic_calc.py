import copy
import re

from functions import find_val_two, find_val_xy, find_val, value, format, clean
from var import directory

ID_index = 0 
pin_index_1 = 1
pin_index_2 = 2
res_index = 3

output_pin_pattern = r'\*[0-9]+:[A-Za-z]'


def all_pins(input_net : list) -> set:
    '''
    returns a set of nodes (like nodes on a circuit, but they will be referred to as pins) in the net. set so no duplicates.
    '''
    result = set()
    for segment in input_net:
        result.add(segment[pin_index_1])
    return result

def pins_count(input_net : list, pin_set : set) -> dict:
    '''
    counts the number of times a pin is outputting (as in it is the first column of pins)
    '''
    result = {}
    for pin in pin_set:
        for segment in input_net:
            if segment[pin_index_1] == pin and pin in result: 
                result[pin] += 1
            elif segment[pin_index_1] == pin :
                result[pin] = 1
    return result

def res_parallel(net_list : list, pin_count : dict):
    '''
    recursive function that takes in the net list and pin count and adds the resistance of available branches in parallel.
    function runs until it cannot find viable parallel adding.
    '''
    for item1 in reversed(net_list):
        ref_pin_1_index = net_list.index(item1)
        for item2 in reversed(net_list):
            if net_list.index(item2) == ref_pin_1_index:
                continue
            elif (re.search(output_pin_pattern, item1[pin_index_2]) != None or item1[pin_index_2] == "") and (re.search(output_pin_pattern, item2[pin_index_2]) != None or item2[pin_index_2] == "") : 
                ref_pin_2 = copy.deepcopy(item2)
                ref_pin_1 = copy.deepcopy(item1)
                new_res = 0
                if item2[pin_index_1] == item1[pin_index_1] :
                    new_res = float(ref_pin_2[res_index]) * float(ref_pin_1[res_index]) / (float(ref_pin_2[res_index]) + float(ref_pin_1[res_index]))    
                    pin_count[item1[pin_index_1]] -= 1
                else:
                    continue
                new_list = copy.deepcopy(net_list)
                new_list.remove(ref_pin_1) 
                ref_pin_2_index = new_list.index(ref_pin_2)
                new_list[ref_pin_2_index][pin_index_2] = ""
                new_list[ref_pin_2_index][res_index] = new_res
                return res_parallel(new_list, pin_count)
            else :
                continue
    return net_list, pin_count


def res_series(net_list : list, pin_count : dict):
    '''
    recursive function that takes in the net list and pin count and add resistance in series.
    function ends when there is no viable series adding.
    '''
    for item1 in reversed(net_list):
        ref_pin_1_index = net_list.index(item1)
        for item2 in reversed(net_list):
            if net_list.index(item2) == ref_pin_1_index:
                continue
            elif (re.search(output_pin_pattern, item1[pin_index_2]) != None or item1[pin_index_2] == "") and item1[pin_index_1] in pin_count:
                ref_pin_2 = copy.deepcopy(item2)
                ref_pin_1 = copy.deepcopy(item1)
                new_res = 0
                if item2[pin_index_2] == item1[pin_index_1] and pin_count[item1[pin_index_1]] == 1:
                    new_res = float(ref_pin_2[res_index]) + float(ref_pin_1[res_index])
                    pin_count[item1[pin_index_2]] = 0
                else:
                    continue
                new_list = copy.deepcopy(net_list)
                new_list.remove(ref_pin_1) 
                ref_pin_2_index = new_list.index(item2)
                new_list[ref_pin_2_index][pin_index_2] = ""
                new_list[ref_pin_2_index][res_index] = new_res
                return res_series(new_list, pin_count)
            else :
                continue
    return net_list, pin_count
        
def parasitic_calc():
    '''
    reads spef file and finds capacitance and does resistance calculations for each net. 
    
    returns:
        net_cap: dict of capacitances
        net_res: dict of resistances 
    '''

    spef_file = "results/generated-tcl.spef"
    map_pattern = fr'\*[0-9]+\s_[0-9]+_\n'

    # collecting net capacitances and lines of resistances from spef file
    net_res = {}
    net_cap = {}
    mapping = {} # this is dict between componnent id in the spef file vs the def file
    spef_data = open(directory + spef_file)
    spef_lines = spef_data.readlines()
    for line in range(len(spef_lines)):
        segment_res = []
        if re.search(map_pattern, spef_lines[line]) != None:
            mapping[spef_lines[line].split(" ")[0]] = clean(spef_lines[line].split(" ")[1])
        if "*D_NET" in spef_lines[line]:
            net_name = spef_lines[line].split(" ")[1]
            net_cap[net_name] = clean(spef_lines[line].split(" ")[2])

            num = 0
            while "*RES" not in spef_lines[line + num]:
                num += 1
            num += 1
            while "*END" not in spef_lines[line + num]:
                segment_list = clean(spef_lines[line + num]).split(" ")
                segment_res.append(segment_list)
                num += 1
            net_res[net_name] = segment_res

    result = 0

    # calculate every net resistance and connect to a macro ID alongside cap
    net_res_final = {}
    for net in mapping:
        if net in net_res:
            pin_set = all_pins(net_res[net])
            pin_dict = pins_count(net_res[net], pin_set)
            while len(net_res[net]) > 1:
                net_res[net], pin_dict = res_parallel(net_res[net], pin_dict)
                net_res[net], pin_dict = res_series(net_res[net], pin_dict)
            net_res_final[mapping[net]] = net_res[net][0][3]
            net_cap[mapping[net]] = net_cap[net]

    net_res = net_res_final
    return net_cap, net_res