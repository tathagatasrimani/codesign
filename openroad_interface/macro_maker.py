import copy
import yaml
from functions import *

lef_tech_file = "Nangate45_tech.lef"
spacing = None
pin_size = 125
pins_needed = 64
row_height = None
spacing_constant = 3 
die_coord_x2 = 100.13 * 1000
die_coord_y2 = 100.8 * 1000

######################################################################
#######                                                        #######
#######          all units are converted to nanometer          #######
#######                                                        #######
######################################################################


#specifically so i dont have to do math to make an add and mult macro 
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

   
def macro_maker(row_height, macro_list, op, heigh_num, pin_num_x, pin_num_y):
    row_height *= heigh_num
    constant_row_height = copy.deepcopy(row_height)
    row_height = row_height - pin_size * pin_num_y
    spacing_y = 0
    print(row_height/(pin_num_y - 1))

    if (spacing <= row_height/(pin_num_y - 1)):
        spacing_y = round(row_height/(pin_num_y - 1), 2)
    else:
        return
    row_height = constant_row_height

    # iterating through every area size ADD
    for size in area_list:
        macro_area = area_list[size][op]
        constant_macro_x = macro_area / row_height
        macro_x  = constant_macro_x - pin_size * pin_num_x
        spacing_x = 0
        if (spacing <= macro_x/(pin_num_x - 1)):
            spacing_x = macro_x/(pin_num_x - 1)
            macro_design = {"name": op + str(pin_num_x * pin_num_y) + "_X" + str(size), "pin_x" : pin_num_x, "pin_y" : pin_num_y, "spacing_x" : spacing_x, "spacing_y" :spacing_y, "x" :constant_macro_x, "y" :row_height}
            macro_list.append(macro_design)
        else:
            print ("invalid design")

            
def macro_maker_pin(row_height, macro_list, op, heigh_num):
    row_height *= heigh_num
    constant_row_height = copy.deepcopy(row_height)
    pin_num_y = 0 
    spacing_y = 1

    row_height -= pin_size
    while row_height > pin_size + spacing:
        row_height -= spacing * spacing_constant
        row_height -= pin_size
        pin_num_y += 1
    extra_row_height = row_height - pin_size
    spacing_y = round(spacing * spacing_constant + extra_row_height/pin_num_y, 2)
    pin_num_y += 1
    row_height = constant_row_height

    # iterating through every area size ADD
    for size in area_list:
        macro_area = area_list[size][op]
        macro_x = int(macro_area / row_height)
        constant_x = copy.deepcopy(macro_x)
        pin_num_x = 1
        macro_x -= pin_size
        while macro_x > pin_size + spacing:
            macro_x -= spacing
            macro_x -= pin_size
            pin_num_x += 1
            if pin_num_x * pin_num_y >= pins_needed - pin_num_y:
                break
        if macro_x < 0 :
            macro_x += spacing
            macro_x += pin_size
        extra_row_height = macro_x - pin_size
        spacing_x = round(spacing + extra_row_height/pin_num_x, 2)
        pin_num_x = pin_num_x + 1
        macro_x = constant_x
        if pin_num_x * pin_num_y >= pins_needed:
            macro_design = {"name": op + str(pin_num_x * pin_num_y) + "_X" + str(size), "pin_x" : pin_num_x, "pin_y" : pin_num_y, "spacing_x" : spacing_x, "spacing_y" :spacing_y, "x" :macro_x, "y" :row_height}
            macro_list.append(macro_design)


tech_params = yaml.load(open("tech_params.yaml", "r"), Loader=yaml.Loader)
area_list = tech_params["area"]
# print(area_list)


# getting the spacing from the lef file 
lef_data = open(lef_tech_file)
lef_tech_lines = lef_data.readlines()
for line in range(len(lef_tech_lines)):
    if "LAYER metal1\n" == lef_tech_lines[line]:
      spacing = float(find_val("SPACING", lef_tech_lines, line)) * 1000 
      print(spacing)
    elif "SIZE" in lef_tech_lines[line]:
        site_size = clean(value(lef_tech_lines[line], "SIZE"))
        row_height = float(clean(site_size.split("BY",1)[1])) * 1000 

# print(spacing)
add_macro_design_list_pins = []
mult_macro_design_list_pins = []
add_macro_design_list_8x8 = []
mult_macro_design_list_8x8 = []
# determining how many pins vertically
for x in range(1,11):
    macro_maker(row_height, mult_macro_design_list_8x8, "Mult", x, 8, 9)
    macro_maker(row_height, add_macro_design_list_8x8, "Add", x, 8, 9)
    macro_maker_pin(row_height, mult_macro_design_list_pins, "Mult", x)
    macro_maker_pin(row_height, add_macro_design_list_pins, "Add", x)
# print(evaluate(add_macro_design_list_pins)
# mult_macro_design_list_pins =evaluate(mult_macro_design_list_pins)
# print (mult_macro_design_list_pins)
# print (evaluate(add_macro_design_list_pins))
print (mult_macro_design_list_8x8)
# print (best(evaluate(add_macro_design_list_8x8)))

# print (add_macro_design_list)


def generate_macro_lef(design_list):
    intro = "MACRO {}\n  CLASS CORE ;\n  ORIGIN 0 0 ;\n  FOREIGN {} 0 0 ;\n  SIZE {} BY {} ;\n  SITE FreePDK45_38x28_10R_NP_162NW_34O ;\n".format(design_list["name"], design_list["name"], design_list["x"]/1000, design_list["y"]/1000)
    pins = [intro]
    pin_number = 0
    input_pin_number = 32
    output_pin_number = 32
    for pin_x in range(design_list["pin_x"]):
        for pin_y in range(design_list["pin_y"]):
            x1 = 0 + pin_size * pin_x + design_list["spacing_x"] * pin_x
            x2 = 0 + pin_size * (pin_x+1) + design_list["spacing_x"] * pin_x
            y1 = 0 + pin_size * pin_y + design_list["spacing_y"] * pin_y
            y2 = 0 + pin_size * (pin_y+1) + design_list["spacing_y"] * pin_y
            direction = "INPUT"

            if input_pin_number == 0:
                direction = 'OUTPUT'

            pin_name = ""
            if pin_number == 0 :
                pin_name = "VDD"
            elif pin_number == design_list["pin_y"] :
                pin_name = "VSS"
            elif direction == 'OUTPUT':
                pin_name = "Z" + str(pin_number - 32) 
                output_pin_number -= 1
            else:
                pin_name = "A" + str(pin_number)
                input_pin_number -= 1

            pin ="   PIN {}\n    DIRECTION {} ;\n    USE SIGNAL ;\n    PORT\n      LAYER metal1 ;\n        RECT {} {} {} {} ;\n    END\n   END {}\n".format(pin_name, direction, x1/1000, y1/1000, x2/1000, y2/1000, pin_name, design_list["name"])
            pin_number += 1
            pins.append(pin)
    return (pins)


pinsss= generate_macro_lef(best(evaluate(add_macro_design_list_8x8)))
# print (pinsss)

with open('macro_pin.txt', 'w') as f:

    for pin in pinsss:
        f.write(f"{pin}")
    f.write("\nEND " + best(evaluate(add_macro_design_list_8x8))["name"])
                                                        

# while ()
# x1 = chip_x1 + pin_size * (num-1) + spacing * (num-1)
# x2 = chip_x1 + pin_size * num + spacing * (num-1)
# y1 = chip_y1 + spacing * (num-1)
# y2 = chip_y1 + pin_size * num + spacing * (num-1)
