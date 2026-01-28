import logging
import math
import copy
import yaml
import os

from .openroad_functions import find_val, clean, value

lef_tech_file = "./OpenROAD/test/Nangate45/Nangate45_tech.lef" 

## Specifies which PDK was used to generate the reference area values.
REFERENCE_AREA_TECH_NODE = 7

## specifies which PDK's LEF files are used. 
REFERENCE_LEF_TECH_NODE = 45


## the factor by which the area must be scaled up to match the LEF data
AREA_SCALE_FACTOR = REFERENCE_LEF_TECH_NODE / REFERENCE_AREA_TECH_NODE

## The target aspect ratio for the macros (Height / Width)
TARGET_ASPECT_RATIO = 1.0

logger = logging.getLogger(__name__)

DEBUG = True

def debug_print(message):
    if DEBUG:
        logger.info(message)

BASE_TECH_LEF_FILE = "openroad_interface/tcl/codesign_files/codesign_tech.lef"

## Specifies which PDK was used to generate the reference area values.
REFERENCE_AREA_TECH_NODE = 7

## specifies which PDK's LEF files are used. 
REFERENCE_LEF_TECH_NODE = 45


## the factor by which the area must be scaled up to match the LEF data
AREA_SCALE_FACTOR = (REFERENCE_LEF_TECH_NODE / REFERENCE_AREA_TECH_NODE) ** 2

## The target aspect ratio for the macros (Height / Width)
TARGET_ASPECT_RATIO = 1.0

## The maximum possible row count deviation allowed from the target aspect ratio (in number of rows in either direction)
MAX_ROW_COUNT_DEVIATION = 20

## The size of the pin in tracks
PIN_SIZE_TRACKS = 2  # in tracks

## The height of the power rails in tracks (Metal1)
POWER_RAIL_HEIGHT = 2  # in tracks

## The cutoff in rows for standard cell vs macro designation. If the height of the macro is less than this value, it is considered a standard cell.
STD_CELL_VS_MACRO_CUTOFF = 0 ## in rows

######################################################################
#######                                                        #######
#######          all units are converted to nanometer          #######
#######                x 1000 is the conversion                #######
#######                                                        #######
######################################################################

class MacroMaker:
    def __init__(self, cfg, codesign_root_dir, tmp_dir, run_openroad, subdirectory=None, output_lef_file="generated_macros.lef", area_list = None, pin_list = None, add_ending_text = True, custom_lef_files_to_include=None):
        """Initializes the MacroMaker with optional area and pin lists.

        NOTE: MacroMaker assumes that the input LEF file is for 45nm technology node and that the area values are for 7nm technology node.
        The resulting macros generated are for the 45nm technology node. Further scaling may be done downstream by the scale LEF function.

        The resulting LEF file will have all the macros specified in area_list with the corresponding areas and pin counts.

        Parameters:
            output_lef_file (str): The name of the output LEF file.
            area_list (dict): A dictionary mapping macro names to their area values.
            pin_list (dict): A dictionary mapping macro names to their pin counts.
            add_ending_text (bool): Whether to add ending text to the output LEF file.

        
        """
        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.tmp_dir = tmp_dir
        self.run_openroad = run_openroad
        self.directory = os.path.join(self.codesign_root_dir, f"{self.tmp_dir}/pd")
        self.subdirectory = subdirectory

        ## results will be placed here. This is necessary for running the flow hierarchically. 
        if subdirectory is not None:
            self.directory = os.path.join(self.directory, subdirectory)

        tech_params_file_path = os.path.join(self.codesign_root_dir, "src/yaml/tech_params.yaml")

        tech_params = yaml.load(open(tech_params_file_path, "r"), Loader=yaml.Loader)

        if area_list is None:
            self.area_list = tech_params["area"][REFERENCE_AREA_TECH_NODE]
        else:
            self.area_list = area_list
        if pin_list is None:
            self.pin_list = tech_params["pin_count"]
        else:
            self.pin_list = pin_list

        self.output_lef_file_path = os.path.join(self.directory, output_lef_file)

        # This is the lef file before any modifications are made. 
        self.base_lef_tech_file = os.path.join(self.codesign_root_dir, BASE_TECH_LEF_FILE)
        self.get_data_from_lef()

        self.add_ending_text = add_ending_text
        self.custom_lef_files_to_include = custom_lef_files_to_include

        # Track grid definitions (nm) extracted from your make_tracks commands.
        # You may later read this from tech.lef automatically if desired.
        self.track_x_offset = 95      # 0.095 um  = 95 nm
        self.track_x_pitch  = 190     # 0.19  um  = 190 nm
        self.track_y_offset = 70      # 0.07  um  = 70 nm
        self.track_y_pitch  = 140     # 0.14  um  = 140 nm

    def snap_to_track(self, value, offset, pitch):
        """Snap a coordinate (in nm) to the nearest track center."""
        return offset + round((value - offset) / pitch) * pitch

    def get_data_from_lef(self):
        # getting the spacing from the lef file
        lef_data = open(self.base_lef_tech_file)
        lef_tech_lines = lef_data.readlines()
        for line in range(len(lef_tech_lines)):
            if "LAYER metal1" == lef_tech_lines[line].strip():
                self.metal1_width = float(find_val("WIDTH", lef_tech_lines, line)) * 1000
                self.spacing = float(find_val("SPACING", lef_tech_lines, line)) * 1000
                self.metal_pitch = self.metal1_width + self.spacing
                
            elif "SIZE" in lef_tech_lines[line]:
                site_size = clean(value(lef_tech_lines[line], "SIZE"))
                self.row_height = float(clean(site_size.split("BY", 1)[1])) * 1000
            elif "MANUFACTURINGGRID" in lef_tech_lines[line]:
                self.manufacturing_grid = float(clean(value(lef_tech_lines[line], "MANUFACTURINGGRID"))) * 1000

        self.pin_size = math.ceil(
                PIN_SIZE_TRACKS * self.metal_pitch / self.manufacturing_grid
            ) * self.manufacturing_grid
        self.VDD_height = math.ceil(
                POWER_RAIL_HEIGHT * self.metal1_width / self.manufacturing_grid
            ) * self.manufacturing_grid
        self.VSS_height = math.ceil(
                POWER_RAIL_HEIGHT * self.metal1_width / self.manufacturing_grid
            ) * self.manufacturing_grid

        debug_print(f"spacing = {self.spacing}, metal_pitch = {self.metal_pitch}, row_height= {self.row_height}, manufacturing_grid= {self.manufacturing_grid}, pin_size= {self.pin_size}, VDD_height= {self.VDD_height}, VSS_height= {self.VSS_height}")

    def include_custom_lef_files(self):
        """Includes custom LEF files into the output LEF file.

        Parameters:
            lef_file_list (list): A list of file paths to LEF files to include. These paths must have one or more macros defined without any LIBRARY or END LIBRARY statements.
        """
        if self.custom_lef_files_to_include is None:
            return
        with open(self.output_lef_file_path, 'a') as f:
            for module, lef_file in self.custom_lef_files_to_include.items():
                debug_print(f"Including custom LEF file: {lef_file} for module: {module}")
                with open(lef_file, 'r') as custom_lef:
                    for line in custom_lef:
                        f.write(line)
            f.write("\n")  # Add a newline after including all custom LEF files
    
    def create_all_macros(self):

        # iterates through all needed macros
        ref_tech_param = copy.deepcopy(self.area_list)

        debug_print("ref_tech_param: ")
        for key, value in ref_tech_param.items():
            debug_print(f"  {key}: {value}")

        for macro in list(ref_tech_param):
            debug_print("Generating designs for macro: " + macro)
            if macro == "Buf" or macro == "Register":
                continue
            macro_design_list_1 = []

            # get area of this macro
            macro_area = self.area_list[macro] * AREA_SCALE_FACTOR
            macro_area_nm2 = macro_area * 1000000  # convert from um^2 to nm^2
            
            debug_print(f"macro_area (nm^2) for {macro}: {macro_area_nm2}")
            debug_print(f"self.row_height: {self.row_height}")
            # we want our macros to be as close to square as possible
            ideal_macro_height_rows = math.floor(math.sqrt(macro_area_nm2)/self.row_height)

            debug_print(f"ideal_macro_height_rows for {macro}: {ideal_macro_height_rows}")

            # iterates through different row heights
            for x in range(ideal_macro_height_rows - MAX_ROW_COUNT_DEVIATION, ideal_macro_height_rows + MAX_ROW_COUNT_DEVIATION + 1):
                new_design = self.macro_maker(macro, x)
                if new_design is not None:
                    macro_design_list_1.append(new_design)

            if len(macro_design_list_1) == 0:
                debug_print("no valid design for " + macro)
                exit(1)
            else:
                # successfully generated designs
                # debug_print("Successfully generated designs for " + macro)
                best1 = self.find_best_aspect_ratio(macro_design_list_1)
                # debug_print(best1)
                seventyfive_lef = self.generate_macro_lef(best1)
                # debug_print("Generated LEF for " + macro + " with 25% aspect ratio")
                # append generated LEF text to the configured output file
                with open(self.output_lef_file_path, 'a') as f:
                    for pin in seventyfive_lef:
                        f.write(f"{pin}")
                    f.write("END " + best1["name"] + "\n\n")
        
        self.include_custom_lef_files()
        
        ## write this message to the end of the lef file if needed.
            #END LIBRARY
            #
            # End of file
            #
        if self.add_ending_text:
            with open(self.output_lef_file_path, 'a') as f: 
                f.write("END LIBRARY\n\n")
                f.write("#\n# End of file\n#\n")



    def generate_macro_lef(self, design_list):
        block_or_core = "BLOCK"
        if design_list["y"] <= self.row_height * STD_CELL_VS_MACRO_CUTOFF:
            block_or_core = "CORE"
        
        intro = f"MACRO {design_list['name']}\n  CLASS {block_or_core} ;\n  ORIGIN 0 0 ;\n  FOREIGN {design_list['name']} 0 0 ;\n  SIZE {design_list['x']/1000} BY {design_list['y']/1000} ;\n  SITE codesign_site ;\n"
        pins = [intro]
        pin_number = 0
        input_pin_number = design_list["input_pin_count"]
        input_reference = input_pin_number
        total_pins = design_list["input_pin_count"] + design_list["output_pin_count"]

        def snap_x(v_nm): return self.snap_to_track(v_nm, self.track_x_offset, self.track_x_pitch)
        def snap_y(v_nm): return self.snap_to_track(v_nm, self.track_y_offset, self.track_y_pitch)


        for pin_x in range(design_list["pin_x"]):
            for pin_y in range(design_list["pin_y"]):
                if pin_number >= total_pins:
                    break
                
                
                direction = "INPUT"

                if input_pin_number == 0:
                    direction = 'OUTPUT'

                pin_name = ""
                if direction == 'OUTPUT':
                    pin_name = "Z" + str(pin_number - input_reference) 
                    # debug_print(str(pin_number - input_reference) )
                else:
                    pin_name = "A" + str(pin_number)
                    input_pin_number -= 1


                # --- Raw lower-left coordinates (in nm) ---
                x1_raw = self.pin_size * pin_x + design_list["spacing_x"] * (pin_x + 1)
                y1_raw = self.pin_size * pin_y + design_list["spacing_y"] * (pin_y + 1) + self.VDD_height

                # --- Snap lower-left corner to track grid ---
                x1 = snap_x(x1_raw)
                y1 = snap_y(y1_raw)

                # --- Upper-right coordinates preserve pin size ---
                x2 = x1 + self.pin_size
                y2 = y1 + self.pin_size

                # --- Create LEF pin entry ---
                pin = (
                    "   PIN {}\n"
                    "    DIRECTION {} ;\n"
                    "    USE SIGNAL ;\n"
                    "    PORT\n"
                    "      LAYER metal1 ;\n"
                    "        RECT {} {} {} {} ;\n"
                    "    END\n"
                    "   END {}\n"
                ).format(
                    pin_name,
                    direction,
                    x1/1000, y1/1000, x2/1000, y2/1000,
                    pin_name
                )

                pin_number += 1
                pins.append(pin)

        pin = "   PIN VDD\n    DIRECTION INOUT ;\n    USE POWER ;\n    PORT\n      LAYER metal1 ;\n        RECT {} {} {} {} ;\n    END\n   END VDD\n".format( 0, 0, design_list["x"]/1000, self.VDD_height/1000)
        pins.append(pin)

        pin = "   PIN VSS\n    DIRECTION INOUT ;\n    USE POWER ;\n    PORT\n      LAYER metal1 ;\n        RECT {} {} {} {} ;\n    END\n   END VSS\n".format( 0, (design_list["y"]-self.VDD_height)/1000, design_list["x"]/1000, design_list["y"]/1000)
        pins.append(pin)

        return (pins)

    # specify how many rows and columns of pins you want and generates the design.
    def macro_maker(self, op, height_num):
        """Generates macro designs and appends them to macro_list.

        Parameters:
        row_height (int): The height of a single row. 
        macro_list (list): The list to append the generated macros to.
        op (str): The operation to perform. Used for naming the macro.
        height_num (int): The number of height units. Total height = row_height * height_num.
        """

        ## debug_print out the parameters for debugging
        debug_print(f"Generating macro with row_height: {self.row_height}, op: {op}, height_num: {height_num}")

        cell_height = self.row_height * height_num

        if cell_height < (self.VDD_height + self.VSS_height + 2 * self.pin_size + 2 * self.spacing):
            debug_print("Cell height too small to fit pins and power rails.")
            return

        ## find the total number of pins needed for this operation
        num_input_pins = self.pin_list[op]["input"]
        num_output_pins = self.pin_list[op]["output"]

        total_pins = num_input_pins + num_output_pins

        ## The maximum number of pins that can fit in a single column given the cell height and pin size, assumming minimum spacing. 
        max_pins_per_column = math.floor((cell_height - self.VDD_height - self.VSS_height - self.spacing) / (self.spacing + self.pin_size))

        pin_num_y = min(max_pins_per_column, total_pins)
        
        pin_num_x = 0
        if total_pins > 0:
            pin_num_x = math.ceil(total_pins / pin_num_y)

        ## The leftover height after placing the pins and power rails, used to calculate spacing in the y-direction.
        leftover_height_for_spacing = cell_height - self.pin_size * pin_num_y - self.VDD_height - self.VSS_height

        if (self.spacing <= leftover_height_for_spacing/(pin_num_y + 1)):
            ## If this condition is true, that means that there is enough space to place the pins with the required spacing in the y-direction.
            debug_print("Pins fit in y-direction")
            spacing_y = math.floor(leftover_height_for_spacing/(pin_num_y + 1)/self.manufacturing_grid) * self.manufacturing_grid
        else:
            raise ValueError("We should never end up in a situation where the pins do not fit in the y-direction.")


        debug_print(f"op= {op}")
        macro_area = self.area_list[op] * AREA_SCALE_FACTOR
        macro_area *= 1000000  # convert from um^2 to nm^2
        debug_print(f"MACRO_MAKER: macro_area (nm^2) = {macro_area}")
        constant_macro_x = math.ceil((macro_area / cell_height) / self.manufacturing_grid) * self.manufacturing_grid
        debug_print(f"MACRO_MAKER: constant_macro_x = {constant_macro_x}")
        macro_x  = constant_macro_x - self.pin_size * pin_num_x
        debug_print(f"MACRO_MAKER: macro_x = {macro_x}")
        spacing_x = 0

        debug_print(f"self.spacing = {self.spacing}, pin_num_x = {pin_num_x}, macro_x/(pin_num_x + 1) = {macro_x/(pin_num_x + 1)}")
        if (self.spacing <= macro_x/(pin_num_x + 1)):
            spacing_x = math.floor(macro_x/(pin_num_x + 1) / self.manufacturing_grid) * self.manufacturing_grid
            macro_design = {"name": op, "pin_x" : pin_num_x, "pin_y" : pin_num_y, "input_pin_count" : num_input_pins, "output_pin_count" : num_output_pins, "spacing_x" : spacing_x, "spacing_y" :spacing_y, "x" :constant_macro_x, "y" :cell_height}
            debug_print("Valid design: " + str(macro_design))
            return macro_design

        else:
            debug_print("Pins do not fit in x-direction")

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
