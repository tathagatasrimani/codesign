import networkx as nx
import matplotlib.pyplot as plt
import copy
from collections import Counter
import re 
from functions import *
import os

design = "gcd"


graph_file = "aes_arch_copy.gml"
lef_std_file = "../test/Nangate45/Nangate45_stdcell.lef"
lef_tech_file = "../test/Nangate45/Nangate45_tech.lef"
core_coord_x2 = 90.25
core_coord_x1 = 10.07
core_coord_y2 = 91
core_coord_y1 = 11.2

die_coord_x2 =100.13
die_coord_x1 = 0
die_coord_y2 = 100.8
die_coord_y1 = 0


and_gate = "AND2_X1"
xor_gate = "XOR2_X1"
reg = "DFF_X1"
mux = "MUX2_X1"

site_name = None
units = None
site_x = None
site_y = None
layer_min_width = None
layer_x_offset = None
layer_pitch = None


        
def find_val_two(string, data, start):
    pattern = fr'{string}\s+\S+\s+\S+\s;'
    match = None
    result = []
    for num in range(len(data)):
        match = re.search(pattern, data[start + num])  
        if match != None:  
            break
    result.append(clean(value(match.group(0), string)).split(" ")[0])
    result.append(clean(value(match.group(0), string)).split(" ")[1])
    return result

def component_finder(name):
    if and_gate in name.upper():
            return  name
    if xor_gate in name.upper():
            return  name
    if reg in name.upper():
            return  name
    if mux in name.upper():
            return  name
    else:
            return ""
    
def macro_find(name):
    if "AND" in name.upper():
        return  and_gate
    if "XOR" in name.upper():
        return  xor_gate
    if name.startswith("Reg"):
        return  reg
    if "MUX" in name.upper():
        return  mux 
    else:
        return ""

def is_op(name):
    if "AND" in name.upper():
        return  True
    if "XOR" in name.upper():
        return  True
    else:
        return False

def edge_gen(dict ,in_or_out):
    for node in nodes:
        dict[node] = []
        if in_or_out == "in":
            edges = graph.in_edges(node)
            for edge in edges:
                dict[node].append(edge[0])
        else:
            edges = graph.out_edges(node)
            for edge in edges:
                dict[node].append(edge[1])



### 0. reading lef file ###
lef_std_data = open(lef_std_file)
lef_std_lines = lef_std_data.readlines()
macro_name = None
macro_names = []
macro_dict = {}
for line in lef_std_lines:
    if "MACRO" in line:
        macro_name = clean(value(line, "MACRO"))
        if component_finder(macro_name) != "":
            macro_names.append(macro_name)
            io = {}
            macro_dict[macro_name] = io
            io["input"] = []
            io["output"] = []
        else:
            macro_name = ""

    elif "PIN" in line and macro_name != "": 
        pin_name = clean(value(line, "PIN"))
        if pin_name.startswith("A") or pin_name.startswith("B") or pin_name.startswith("D"):
            macro_dict[macro_name]["input"].append(pin_name)
        elif pin_name.startswith("Z") or pin_name.startswith("Q"):
            macro_dict[macro_name]["output"].append(pin_name)

lef_data = open(lef_tech_file)
lef_tech_lines = lef_data.readlines()
for line in lef_tech_lines:
    if "SITE" in line:
        site_name = clean(value(line, "SITE"))
    if "DATABASE MICRONS" in line:
        units = float(clean(value(line, "DATABASE MICRONS")))
    if "SIZE" in line:
        site_size = clean(value(line, "SIZE"))
        site_x = float(clean(site_size.split("BY",1)[0]))
        site_y = float(clean(site_size.split("BY",1)[1]))


graph = nx.read_gml(graph_file)
nodes = list(graph)
control_nodes = list(graph)

### 1. pruning ###
for node1 in control_nodes:
    if "Mem" in node1 or "Buf" in node1:
            graph.remove_node(node1)
            nodes.remove(node1)

# nx.draw(graph, with_labels=True)
# plt.show()

input_dict = {}
edge_gen(input_dict ,"in")
# print (input_dict)



### 2. mux tree ###
counter = 0 
for node in nodes:
    num = 2
    if node.startswith("Reg"):
        num = 1
    else:
        num = 2
    while len(input_dict[node]) > num:
        target_node1 = input_dict[node][0]
        target_node2 = input_dict[node][1]

        graph.remove_edge(target_node2, node)
        graph.remove_edge(target_node1, node)

        input_dict[node].remove(target_node2)
        input_dict[node].remove(target_node1)

        new_node = "Mux" + str(counter)
        counter += 1 

        graph.add_edge(target_node1, new_node)
        graph.add_edge(target_node2, new_node)
        
        graph.add_edge(new_node, node)
        input_dict[node].append(new_node)





### 3. mapping components to nodes ###
nodes = list(graph)
node_to_macro = {}
for node in nodes:
    macro = macro_find(node)
    node_to_macro[node] = [macro, copy.deepcopy(macro_dict[macro])]
# print(node_to_macro)


# nx.draw(graph, with_labels=True)
# plt.show()
        



### 4.generate header ###
header_text = []
header_text.append("VERSION 5.8 ;")
for line in lef_tech_lines:
    if "DIVIDERCHAR" in line:
        header_text.append(line.replace("\n", ""))
        break
    elif "BUSBITCHARS" in line:
        header_text.append(line.replace("\n", ""))

header_text.append("DESIGN {} ;".format(design))
header_text.append("UNITS DISTANCE MICRONS {} ;".format(int(units)))
header_text.append("DIEAREA ( {} {} ) ( {} {} ) ;".format(int(die_coord_x1 * units), int(die_coord_y1 *units ), int(die_coord_x2 * units), int(die_coord_y2 * units)))
# print(header_text)

with open('generated/header.txt', 'w') as f:
    for line in header_text:
        f.write(f"{line}\n")


### 5.generate components ###
component_text = []
number = 1
node_to_num = {}

component_dict = {}
for node in nodes:
    component_num = format(number)
    macro = node_to_macro[node][0]
    component_text.append("- {} {} ;".format(component_num, macro))
    node_to_num[node] = format(number)
    number += 1
# print(component_text)

component_text.insert(0, "COMPONENTS {} ;".format(len(component_text)))
component_text.insert(len(component_text), "END COMPONENTS")
with open('generated/component.txt', 'w') as f:
    for line in component_text:
        f.write(f"{line}\n")




## 6.generate nets ###
net_text = []
output_dict = {}
net_out_dict = {}
edge_gen(output_dict ,"out")
for node in nodes:
    net_name = format(str(number))
    component_name = node_to_num[node]
    pin_out = node_to_macro[node][1]["output"]
    net = "- {} ( {} {} )".format(net_name, component_name, pin_out[0])
    pin_out.remove(pin_out[0])
    net_out_dict[node] = net_name

    for output in output_dict[node]:
        pin_in = node_to_macro[output][1]["input"]
        net = net + " ( {} {} )".format(node_to_num[output], pin_in[0])
        pin_in.remove(pin_in[0])

    number += 1
    net = net + " + USE SIGNAL ;"
    net_text.append(net)

# print(net_out_dict)
# print(output_dict)
net_text.insert(0, "NETS {} ;".format(len(net_text)))
net_text.insert(len(net_text), "END NETS")
net_text.insert(len(net_text), "END DESIGN")

with open('generated/net.txt', 'w') as f:
    for line in net_text:
        f.write(f"{line}\n")

# generating pins
pin_text = []
pin_text.append("PINS 1 ;")
pin_text.append("- clk + NET clk + DIRECTION INPUT + USE SIGNAL ;")
pin_text.append("END  PINS")



#$# 7.generate rows ###
row_text = []
core_y = core_coord_y2 - core_coord_y1
counter = 0

core_dy = core_y * units
site_dy = site_y * units
site_dx = site_x * units

while site_dy <= core_dy - counter * site_dy:
    text = "ROW ROW_{} {} {} {}".format(str(counter), site_name, str(int(core_coord_x1 * units)), str(int(core_coord_y1 * units + counter * site_dy)))
    
    if (counter + 1)%2 == 0:
        text += " FS "
    elif (counter + 1)%2 == 1:
        text += " N "
    
    num_row = 0
    while (core_coord_x2 - core_coord_x1) * units - num_row * site_dx >= site_dx:
        num_row = num_row + 1 

    text += "DO {} BY 1 ".format(str(num_row))

    text += "STEP {} 0 ;".format(str(int(site_dx)))

    counter += 1
    row_text.append(text)


with open('generated/ROW.txt', 'w') as f:
    for line in row_text:
        f.write(f"{line}\n")


'''void InitFloorplan::initFloorplan(
    double utilization,
    double aspect_ratio,
    int core_space_bottom,
    int core_space_top,
    int core_space_left,
    int core_space_right,
    odb::dbSite* base_site,
    const std::vector<odb::dbSite*>& additional_sites)
{

  utilization /= 100;
  const double design_area = designArea();

  double InitFloorplan::designArea()
{
  double design_area = 0.0;
  for (dbInst* inst : block_->getInsts()) {
    dbMaster* master = inst->getMaster();
    const double area
        = master->getHeight() * static_cast<double>(master->getWidth());
    design_area += area;
  }
  return design_area;
}

  const double core_area = design_area / utilization;
  const int core_width = std::sqrt(core_area / aspect_ratio);
  const int core_height = round(core_width * aspect_ratio);

  const int core_lx = core_space_left;
  const int core_ly = core_space_bottom;
  const int core_ux = core_lx + core_width;
  const int core_uy = core_ly + core_height;
  const int die_lx = 0;
  const int die_ly = 0;
  const int die_ux = core_ux + core_space_right;
  const int die_uy = core_uy + core_space_top;
  initFloorplan({die_lx, die_ly, die_ux, die_uy},
                {core_lx, core_ly, core_ux, core_uy},
                base_site,
                additional_sites);'''


#$# 8.generate track ###

die_coord_x2 *= units
die_coord_y2 *= units

## tracks aren't made from the lef file; for some reason they have their own track file that sets the numbers
track_text = []
for line in range(len(lef_tech_lines)):
    if "LAYER " in lef_tech_lines[line] and "metal" in lef_tech_lines[line] and "TYPE" in lef_tech_lines[line+1]:
        layer_name = clean(value(lef_tech_lines[line], "LAYER"))

        layer_min_width = float(find_val("WIDTH", lef_tech_lines, line)) * units
        # print(layer_min_width)
        layer_pitch = float(find_val("PITCH", lef_tech_lines, line)) * units
        # print(layer_x_pitch)
        layer_x_offset = float(find_val_two("OFFSET", lef_tech_lines, line)[0]) * units
        # print(layer_x_offset)

        layer_y_offset = float(find_val_two("OFFSET", lef_tech_lines, line)[1]) * units
        # print(layer_x_offset)

        x_track_count = int((die_coord_x2 - layer_x_offset)/ layer_pitch) + 1
        origin_x = layer_x_offset + die_coord_x1

        if origin_x - layer_min_width / 2 < die_coord_x1:
            origin_x += layer_pitch
            x_track_count -= 1

        last_x = origin_x + (x_track_count - 1) * layer_pitch
        if last_x + layer_min_width / 2 > die_coord_x2:
            x_track_count -= 1

        y_track_count = int((die_coord_y2 - layer_y_offset)/ layer_pitch) + 1
        origin_y = layer_y_offset + die_coord_y1

        if origin_y - layer_min_width / 2 < die_coord_y1:
            origin_y += layer_pitch
            y_track_count -= 1

        last_y = origin_y + (y_track_count - 1) * layer_pitch
        if last_y + layer_min_width / 2 > die_coord_y2:
            y_track_count -= 1
        
        text = "TRACKS X {} DO {} STEP {} LAYER {} ;".format(int(origin_x), int(x_track_count), int(layer_pitch), layer_name)
        track_text.append(text)
        text = "TRACKS Y {} DO {} STEP {} LAYER {} ;".format(int(origin_y), int(y_track_count), int(layer_pitch), layer_name)
        track_text.append(text)

with open('generated/TRACK.txt', 'w') as f:
    for line in track_text:
        f.write(f"{line}\n")

            
if not os.path.exists("../test/results/"):
    os.makedirs("../test/results/")

with open('../test/results/first_generated.def', 'w') as f:
    for line in header_text:
        f.write(f"{line}\n")
    for line in row_text:
        f.write(f"{line}\n")
    for line in track_text:
        f.write(f"{line}\n")
    for line in component_text:
        f.write(f"{line}\n")
    for line in pin_text:
        f.write(f"{line}\n")
    for line in net_text:
        f.write(f"{line}\n")

