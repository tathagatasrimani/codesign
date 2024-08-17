import math
import copy
import yaml

from functions import *

lef_tech_file = "OpenROAD/test/Nangate45/Nangate45_tech.lef" 
spacing = None
manufacturing_grid = None
pin_size = 1000
VDD = 1000
VSS = 1000
row_height = None
spacing_constant = 3 
die_coord_x2 = 100.13 * 1000
die_coord_y2 = 100.8 * 1000

######################################################################
#######                                                        #######
#######          all units are converted to nanometer          #######
#######                x 1000 is the conversion                #######
#######                                                        #######
######################################################################

# this script generates txt files that contain lef documentation for different generated macros

def best(list):
    constant = copy.deepcopy(list)
    best = abs(constant[0]["x"] - constant[0]["y"])
    best_design = constant[0]
    for num in range(1, len(constant) - 1):
        cur = abs(constant[num]["x"] - constant[num]["y"])
        if cur < best:
            best_design = constant[num]
            best = cur
    return best_design

def evaluate(list):
    constant = copy.deepcopy(list)
    new_list = []
    for solution in constant:
        if die_coord_x2 >= solution["x"] and die_coord_y2 >= solution["y"]:
            new_list.append(solution)

    return new_list

def generate_macro_lef(design_list):
    intro = "MACRO {}\n  CLASS CORE ;\n  ORIGIN 0 0 ;\n  FOREIGN {} 0 0 ;\n  SIZE {} BY {} ;\n  SITE FreePDK45_38x28_10R_NP_162NW_34O ;\n".format(design_list["name"], design_list["name"], design_list["x"]/1000, design_list["y"]/1000)
    pins = [intro]
    pin_number = 0
    input_pin_number = 32
    input_reference = 32
    for pin_x in range(design_list["pin_x"]):
        for pin_y in range(design_list["pin_y"]):
            x1 = 0 + pin_size * pin_x + design_list["spacing_x"] * (pin_x+1)
            x2 = 0 + pin_size * (pin_x+1) + design_list["spacing_x"] * (pin_x+1)
            y1 = 0 + pin_size * pin_y + design_list["spacing_y"] * (pin_y+1) + VDD
            y2 = 0 + pin_size * (pin_y+1) + design_list["spacing_y"] * (pin_y+1) + VDD
            direction = "INPUT"

            if input_pin_number == 0:
                direction = 'OUTPUT'

            pin_name = ""
            if direction == 'OUTPUT':
                pin_name = "Z" + str(pin_number - input_reference) 
                print(str(pin_number - input_reference) )
            else:
                pin_name = "A" + str(pin_number)
                input_pin_number -= 1

            pin ="   PIN {}\n    DIRECTION {} ;\n    USE SIGNAL ;\n    PORT\n      LAYER metal1 ;\n        RECT {} {} {} {} ;\n    END\n   END {}\n".format(pin_name, direction, x1/1000, y1/1000, x2/1000, y2/1000, pin_name, design_list["name"])
            pin_number += 1
            pins.append(pin)

    pin ="   PIN VDD\n    DIRECTION INOUT ;\n    USE POWER ;\n    PORT\n      LAYER metal1 ;\n        RECT {} {} {} {} ;\n    END\n   END VDD\n".format( 0, 0, design_list["x"]/1000, VDD/1000, pin_name)
    pins.append(pin)

    pin ="   PIN VSS\n    DIRECTION INOUT ;\n    USE POWER ;\n    PORT\n      LAYER metal1 ;\n        RECT {} {} {} {} ;\n    END\n   END VSS\n".format( 0, (design_list["y"]-VDD)/1000, design_list["x"]/1000, design_list["y"]/1000, pin_name)
    pins.append(pin)

    return (pins)

# specify how many rows and columns of pins you want ,and generates the design 
def macro_maker(row_height, macro_list, op, heigh_num, pin_num_x, pin_num_y):
    row_height *= heigh_num
    constant_row_height = copy.deepcopy(row_height)
    row_height = row_height - pin_size * pin_num_y - VDD - VSS
    spacing_y = 0

    if (spacing <= row_height/(pin_num_y + 1)):
        spacing_y = math.floor(row_height/(pin_num_y + 1)/manufacturing_grid) * manufacturing_grid
    else:
        return
    row_height = constant_row_height

    for size in area_list:
        macro_area = area_list[size][op]
        constant_macro_x = math.ceil(macro_area / row_height / manufacturing_grid) * manufacturing_grid
        macro_x  = constant_macro_x - pin_size * pin_num_x
        spacing_x = 0
        if (spacing <= macro_x/(pin_num_x + 1)):
            spacing_x = math.floor(macro_x/(pin_num_x + 1) / manufacturing_grid) * manufacturing_grid
            macro_design = {"name": op + str(pin_num_x * pin_num_y) + "_" + str(size), "pin_x" : pin_num_x, "pin_y" : pin_num_y, "spacing_x" : spacing_x, "spacing_y" :spacing_y, "x" :constant_macro_x, "y" :row_height}
            macro_list.append(macro_design)
            # print ("invalid design")

tech_params = yaml.load(open("../src/params/tech_params.yaml", "r"), Loader=yaml.Loader)
area_list = tech_params["area"]

# getting the spacing from the lef file 
lef_data = open(lef_tech_file)
lef_tech_lines = lef_data.readlines()
for line in range(len(lef_tech_lines)):
    if "LAYER metal1\n" == lef_tech_lines[line]:
      spacing = float(find_val("SPACING", lef_tech_lines, line)) * 1000 
    elif "SIZE" in lef_tech_lines[line]:
        site_size = clean(value(lef_tech_lines[line], "SIZE"))
        row_height = float(clean(site_size.split("BY",1)[1])) * 1000 
    elif "MANUFACTURINGGRID" in lef_tech_lines[line]:
        manufacturing_grid = float(clean(value(lef_tech_lines[line], "MANUFACTURINGGRID"))) * 1000

print (manufacturing_grid)

add_macro_design_list = []
mult_macro_design_list = []

# iterates through different row heights
for x in range(10,35):
    macro_maker(row_height, mult_macro_design_list, "Mult", x, 16, 4)
    macro_maker(row_height, add_macro_design_list, "Add", x, 10, 5)


# picked that best design manually
add_lef= generate_macro_lef(add_macro_design_list[2])

with open('macro_pin_add_50.txt', 'w') as f:
    for pin in add_lef:
        f.write(f"{pin}")
    f.write("END " + add_macro_design_list[2]["name"])

mult_lef= generate_macro_lef(mult_macro_design_list[len(mult_macro_design_list)-1])

with open('macro_pin_mult_64.txt', 'w') as f:
    for pin in mult_lef:
        f.write(f"{pin}")
    f.write("END " + mult_macro_design_list[len(mult_macro_design_list)-1]["name"])
                                                        