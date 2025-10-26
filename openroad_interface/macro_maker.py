import math
import copy
import yaml
import os

from openroad_functions import find_val, clean, value

lef_tech_file = "./OpenROAD/test/Nangate45/Nangate45_tech.lef" 

## Specifies which PDK was used to generate the reference area values.
REFERENCE_AREA_TECH_NODE = 7

## specifies which PDK's LEF files are used. 
REFERENCE_LEF_TECH_NODE = 45


## the factor by which the area must be scaled up to match the LEF data
AREA_SCALE_FACTOR = (REFERENCE_LEF_TECH_NODE / REFERENCE_AREA_TECH_NODE) ** 2

## The target aspect ratio for the macros (Height / Width)
TARGET_ASPECT_RATIO = 1.0

######################################################################
#######                                                        #######
#######          all units are converted to nanometer          #######
#######                x 1000 is the conversion                #######
#######                                                        #######
######################################################################

class MacroMaker:
    def __init__(self):
        tech_params = yaml.load(open("../src/yaml/tech_params.yaml", "r"), Loader=yaml.Loader)
        self.area_list = tech_params["area"][REFERENCE_AREA_TECH_NODE]
        self.pin_list = tech_params["pin_count"]

        self.get_data_from_lef()

        self.pin_size = 1000
        self.VDD_height = 1000
        self.VSS_height = 1000

        self.output_lef_file = "macro_lef.txt"

        ## remove the output lef file if it already exists
        if os.path.exists(self.output_lef_file):
            os.remove(self.output_lef_file)


    def get_data_from_lef(self):
        # getting the spacing from the lef file
        lef_data = open(lef_tech_file)
        lef_tech_lines = lef_data.readlines()
        for line in range(len(lef_tech_lines)):
            if "LAYER metal1" == lef_tech_lines[line].strip():
                width = float(find_val("WIDTH", lef_tech_lines, line)) * 1000
                self.spacing = float(find_val("SPACING", lef_tech_lines, line)) * 1000
                self.metal_pitch = width + self.spacing
                self.pin_size = 2 * self.metal_pitch  # two routing tracks wide
            elif "SIZE" in lef_tech_lines[line]:
                site_size = clean(value(lef_tech_lines[line], "SIZE"))
                self.row_height = float(clean(site_size.split("BY", 1)[1])) * 1000
            elif "MANUFACTURINGGRID" in lef_tech_lines[line]:
                self.manufacturing_grid = float(clean(value(lef_tech_lines[line], "MANUFACTURINGGRID"))) * 1000

        print(f"spacing = {self.spacing}, metal_pitch = {self.metal_pitch}, row_height= {self.row_height}, manufacturing_grid= {self.manufacturing_grid}")

    def create_all_macros(self):

        # iterates through all needed macros
        ref_tech_param = copy.deepcopy(self.area_list)

        print("ref_tech_param: ")
        for key, value in ref_tech_param.items():
            print(f"  {key}: {value}")

        for macro in list(ref_tech_param):
            print("Generating designs for macro: " + macro)
            if macro == "Buf" or macro == "Register":
                continue
            macro_design_list_1 = []
            # iterates through different row heights
            for x in range(1,50):
                self.macro_maker(macro_design_list_1, macro, x)

            if len(macro_design_list_1) == 0:
                print("no valid design for " + macro)
                exit(1)
            else:
                # successfully generated designs
                # print("Successfully generated designs for " + macro)
                best1 = self.find_best_aspect_ratio(macro_design_list_1)
                # print(best1)
                seventyfive_lef = self.generate_macro_lef(best1)
                # print("Generated LEF for " + macro + " with 25% aspect ratio")
                # append generated LEF text to the configured output file
                with open(self.output_lef_file, 'a') as f:
                    for pin in seventyfive_lef:
                        f.write(f"{pin}")
                    f.write("END " + best1["name"] + "\n\n")

    def generate_macro_lef(self, design_list):
        intro = "MACRO {}\n  CLASS CORE ;\n  ORIGIN 0 0 ;\n  FOREIGN {} 0 0 ;\n  SIZE {} BY {} ;\n  SITE FreePDK45_38x28_10R_NP_162NW_34O ;\n".format(design_list["name"], design_list["name"], design_list["x"]/1000, design_list["y"]/1000)
        pins = [intro]
        pin_number = 0
        input_pin_number = design_list["input_pin_count"]
        input_reference = input_pin_number
        total_pins = design_list["input_pin_count"] + design_list["output_pin_count"]
        for pin_x in range(design_list["pin_x"]):
            for pin_y in range(design_list["pin_y"]):
                if pin_number >= total_pins:
                    break
                
                x1 = 0 + self.pin_size * pin_x + design_list["spacing_x"] * (pin_x+1)
                x2 = 0 + self.pin_size * (pin_x+1) + design_list["spacing_x"] * (pin_x+1)
                y1 = 0 + self.pin_size * pin_y + design_list["spacing_y"] * (pin_y+1) + self.VDD_height
                y2 = 0 + self.pin_size * (pin_y+1) + design_list["spacing_y"] * (pin_y+1) + self.VDD_height
                direction = "INPUT"

                if input_pin_number == 0:
                    direction = 'OUTPUT'

                pin_name = ""
                if direction == 'OUTPUT':
                    pin_name = "Z" + str(pin_number - input_reference) 
                    # print(str(pin_number - input_reference) )
                else:
                    pin_name = "A" + str(pin_number)
                    input_pin_number -= 1

                pin = "   PIN {}\n    DIRECTION {} ;\n    USE SIGNAL ;\n    PORT\n      LAYER metal1 ;\n        RECT {} {} {} {} ;\n    END\n   END {}\n".format(pin_name, direction, x1/1000, y1/1000, x2/1000, y2/1000, pin_name, design_list["name"])
                pin_number += 1
                pins.append(pin)

        pin = "   PIN VDD\n    DIRECTION INOUT ;\n    USE POWER ;\n    PORT\n      LAYER metal1 ;\n        RECT {} {} {} {} ;\n    END\n   END VDD\n".format( 0, 0, design_list["x"]/1000, self.VDD_height/1000, pin_name)
        pins.append(pin)

        pin = "   PIN VSS\n    DIRECTION INOUT ;\n    USE POWER ;\n    PORT\n      LAYER metal1 ;\n        RECT {} {} {} {} ;\n    END\n   END VSS\n".format( 0, (design_list["y"]-self.VDD_height)/1000, design_list["x"]/1000, design_list["y"]/1000, pin_name)
        pins.append(pin)

        return (pins)

    # specify how many rows and columns of pins you want and generates the design.
    def macro_maker(self, macro_list, op, height_num):
        """Generates macro designs and appends them to macro_list.

        Parameters:
        row_height (int): The height of the rows.
        macro_list (list): The list to append the generated macros to.
        op (str): The operation to perform. Used for naming the macro.
        height_num (int): The number of height units. Total height = row_height * height_num.
        """

        ## print out the parameters for debugging
        print(f"Generating macro with row_height: {self.row_height}, op: {op}, height_num: {height_num}")

        cell_height = self.row_height * height_num

        if cell_height < (self.VDD_height + self.VSS_height + 2 * self.pin_size + 2 * self.spacing):
            print("Cell height too small to fit pins and power rails.")
            return

        ## find the total number of pins needed for this operation
        num_input_pins = self.pin_list[op]["input"]
        num_output_pins = self.pin_list[op]["output"]

        total_pins = num_input_pins + num_output_pins

        ## The maximum number of pins that can fit in a single column given the cell height and pin size, assumming minimum spacing. 
        max_pins_per_column = math.floor((cell_height - self.VDD_height - self.VSS_height - self.spacing) / (self.spacing + self.pin_size))

        pin_num_y = min(max_pins_per_column, total_pins)
        
        pin_num_x = math.ceil(total_pins / pin_num_y)

        ## The leftover height after placing the pins and power rails, used to calculate spacing in the y-direction.
        leftover_height_for_spacing = cell_height - self.pin_size * pin_num_y - self.VDD_height - self.VSS_height

        if (self.spacing <= leftover_height_for_spacing/(pin_num_y + 1)):
            ## If this condition is true, that means that there is enough space to place the pins with the required spacing in the y-direction.
            print("Pins fit in y-direction")
            spacing_y = math.floor(leftover_height_for_spacing/(pin_num_y + 1)/self.manufacturing_grid) * self.manufacturing_grid
        else:
            raise ValueError("We should never end up in a situation where the pins do not fit in the y-direction.")


        print(f"op= {op}")
        macro_area = self.area_list[op] * AREA_SCALE_FACTOR
        macro_area *= 1000000  # convert from um^2 to nm^2
        constant_macro_x = math.ceil(macro_area / cell_height / self.manufacturing_grid) * self.manufacturing_grid
        macro_x  = constant_macro_x - self.pin_size * pin_num_x
        spacing_x = 0
        if (self.spacing <= macro_x/(pin_num_x + 1)):
            spacing_x = math.floor(macro_x/(pin_num_x + 1) / self.manufacturing_grid) * self.manufacturing_grid
            macro_design = {"name": op, "pin_x" : pin_num_x, "pin_y" : pin_num_y, "input_pin_count" : num_input_pins, "output_pin_count" : num_output_pins, "spacing_x" : spacing_x, "spacing_y" :spacing_y, "x" :constant_macro_x, "y" :cell_height}
            macro_list.append(macro_design)
            print("Valid design: " + str(macro_design))

        else:
            print("Pins do not fit in x-direction")
            pass

    def find_best_aspect_ratio(self, design_list):
        best = design_list[0]["y"]/design_list[0]["x"]
        best_design = design_list[0]
        for item in design_list[1:]:
            y = item["y"]
            x = item["x"]
            ratio = y/x
            if abs(best - TARGET_ASPECT_RATIO) > abs(ratio - TARGET_ASPECT_RATIO):
                best = ratio
                best_design = item 
        return best_design

if __name__ == "__main__":
    macro_maker_instance = MacroMaker()
    macro_maker_instance.create_all_macros()
