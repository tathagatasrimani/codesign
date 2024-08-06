### read def spef file assign net with capacitance and resistance (total resistance)
from functions import *
import copy
### adding up the resistnace
stef_file = "../test/results/generated.spef"

# may library

map_pattern = fr'\*[0-9]+\s_[0-9]+_\n'

net_res = {}
net_cap = {}
map = {}
stef_data = open(stef_file)
stef_lines = stef_data.readlines()
for line in range(len(stef_lines)):
    segment_res = []
    if re.search(map_pattern, stef_lines[line]) != None:
        map[stef_lines[line].split(" ")[0]] = clean(stef_lines[line].split(" ")[1])
    if "*D_NET" in stef_lines[line]:
        net_name = stef_lines[line].split(" ")[1]
        net_cap[net_name] = clean(stef_lines[line].split(" ")[2])
        num = 0
        while "*RES" not in stef_lines[line + num]:
            num += 1
        num += 1
        while "*END" not in stef_lines[line + num]:
            segment_list = clean(stef_lines[line + num]).split(" ")
            segment_res.append(segment_list)
            num += 1
        net_res[net_name] = segment_res
    
# print (map)

ID_index = 0 
pin_index_1 = 1
pin_index_2 = 2
res_index = 3

result = 0

output_pin_pattern = r'\*[0-9]+:[A-Za-z]'

def all_pins(input_net):
    result = set()
    for segment in input_net:
        result.add(segment[pin_index_1])
    return result

def pins_count(input_net, pin_set):
    result = {}
    for pin in pin_set:
        for segment in input_net:
            if segment[pin_index_1] == pin and pin in result: 
                result[pin] += 1
            elif segment[pin_index_1] == pin :
                result[pin] = 1
    return result


# forgive me for this function
def res_parallel(net_list, pin_count):
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
                # getting rid of the first one instead of the second one because we have set the logic
                # up so that second item will always be the one with the correct pin 
                ref_pin_2_index = new_list.index(ref_pin_2)
                new_list[ref_pin_2_index][pin_index_2] = ""
                new_list[ref_pin_2_index][res_index] = new_res
                return res_parallel(new_list, pin_count)
            else :
                continue
    return [net_list, pin_dict]


def res_series(net_list, pin_count):
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
    return [net_list, pin_count]


# connect every res value to a net 
for net in map:
    if net in net_res:
        pin_set = all_pins(net_res[net])
        pin_dict = pins_count(net_res[net], pin_set)
        while len(net_res[net]) > 1:
            result = res_parallel(net_res[net], pin_dict)
            net_res[net] = result[0]
            pin_dict = result[1]
            result = res_series(net_res[net], pin_dict)
            net_res[net] = result[0] #no passing by reference, but need to return two differnet things from the function
            pin_dict = result[1]
        net_res[map[net]] = net_res[net][0]
        net_cap[map[net]] = net_cap[net]

# print (net_res)
# print (net_cap)