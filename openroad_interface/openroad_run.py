import logging
import re
import os
import copy
import shutil
from math import sqrt

import logging

logger = logging.getLogger(__name__)

import networkx as nx

from openroad_interface import def_generator
from . import estimation as est
from . import detailed as det

class OpenRoadRun:
    def __init__(self, cfg, codesign_root_dir):
        """
        Initialize the OpenRoadRun with configuration and root directory.

        :param cfg: top level codesign config file
        :param codesign_root_dir: root directory of codesign (where src and test are)
        """
        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.directory = os.path.join(self.codesign_root_dir, "src/tmp/pd")

    def run(
    self,
    graph: nx.DiGraph,
    test_file: str, 
    arg_parasitics: str,
    area_constraint: int,
    alpha: float = 1.0  # Add alpha here
    ):
        """
        Runs the OpenROAD flow.
        params:
            arg_parasitics: detailed, estimation, or none. determines which parasitic calculation is executed.

        """
        alpha = 2.0
        logger.info(f"Starting place and route with parasitics: {arg_parasitics}")
        dict = {edge: {} for edge in graph.edges()}
        if "none" not in arg_parasitics:
            logger.info("Running setup for place and route.")
            graph, net_out_dict, node_output, lef_data, node_to_num = self.setup(graph, test_file, area_constraint, alpha)
            logger.info("Setup complete. Running extraction.")
            dict, graph = self.extraction(graph, arg_parasitics, net_out_dict, node_output, lef_data, node_to_num)
            logger.info("Extraction complete.")
        else: 
            logger.info("No parasitics selected. Running none_place_n_route.")
            graph = self.none_place_n_route(graph)
        logger.info("Place and route finished.")
        return dict, graph

    def setup(
        self,
        graph: nx.DiGraph,
        test_file: str,
        area_constraint: int,
        alpha: float 
    ):
        """
        Sets up the OpenROAD environment. This method creates the working directory, copies tcl files, and generates the def file
        param:
            graph: hardware netlist graph
            test_file: tcl file
            
            area_constraint: area constraint for the placement

        """
        alpha = 1.125
        logger.info("Setting up environment for place and route.")
        if os.path.exists(self.directory):
            logger.info(f"Removing existing directory: {self.directory}")
            shutil.rmtree(self.directory)
        os.makedirs(self.directory)
        logger.info(f"Created directory: {self.directory}")
        shutil.copytree(os.path.dirname(os.path.abspath(__file__)) + "/tcl", self.directory + "/tcl")
        logger.info(f"Copied tcl files to {self.directory}/tcl")
        os.makedirs(self.directory + "/results")
        logger.info(f"Created results directory: {self.directory}/results")

        self.update_area_constraint(area_constraint)
        self._scale_track_pitches(alpha)
        self._scale_tech_lef(alpha)
        self._scale_stdcell_lef(alpha)
        self._scale_pdn_config(alpha)
        self._scale_die_area(alpha)
        self._scale_vars_file(alpha)
        #write script here for editing files
        df = def_generator.DefGenerator(self.cfg, self.codesign_root_dir)
        
        graph, net_out_dict, node_output, lef_data, node_to_num = df.run_def_generator(
            test_file, graph
        )
        logger.info("DEF generation complete.")

        return graph, net_out_dict, node_output, lef_data, node_to_num

    def update_area_constraint(self, area_constraint: int):
        """
        Updates the area constraint in the tcl file based on the input area constraint.
        param:
            area_constraint: area constraint for the placement
        """
        ## edit the tcl file to have the correct area constraint
        with open(self.directory + "/tcl/codesign_top.tcl", "r") as file:
            tcl_data = file.readlines()

        ## compute the new area constraint
        new_sidelength = int(sqrt(area_constraint))  

        ## round to the nearest multiple of 100
        new_sidelength = round(new_sidelength / 100) * 100

        ## find a line that contains "set die_area" and replace it with the new area constraint
        for i, line in enumerate(tcl_data):
            if "set die_area" in line:
                tcl_data[i] = f"set die_area {{0 0 {new_sidelength} {new_sidelength}}}\n"
                logger.info(f"Updated die_area to {new_sidelength}x{new_sidelength}")
            if "set core_area" in line:
                tcl_data[i] = f"set core_area {{50 50 {new_sidelength - 50} {new_sidelength - 50}}}\n"
                logger.info(f"Updated core_area to {new_sidelength - 50}x{new_sidelength - 50}")

        ## write the new tcl file
        with open(self.directory + "/tcl/codesign_top.tcl", "w") as file:
            file.writelines(tcl_data)
        
        logger.info(f"Wrote updated tcl file with the area constraints: {new_sidelength}x{new_sidelength}")

    def _scale_track_pitches(self, alpha: float):
        """
        Updates the track pitches in the temporary codesign.tracks file.
        This method reads the file, scales the x and y pitch values by dividing
        them by the alpha factor, and writes the file back.

        :param alpha: The scaling factor to divide the pitches by.
        """
        # If alpha is 1, no changes are needed.
        if alpha == 1.0:
            logger.info("Alpha is 1.0, no scaling of track pitches required.")
            return

        if alpha <= 0:
            logger.error("Alpha must be positive. Skipping track pitch scaling.")
            return

        logger.info(f"Scaling track pitches by a division factor of {alpha}.")
        
        # Construct the full path to the file inside the temporary directory
        tracks_file_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign.tracks")

        try:
            # Read all lines from the file
            with open(tracks_file_path, "r") as file:
                lines = file.readlines()

            modified_lines = []

            # Define a helper function to perform the replacement using regex
            def replace_pitch(match_obj):
                # match_obj.group(1) will be "-x_pitch " or "-y_pitch "
                # match_obj.group(2) will be the numeric value like "0.19"
                keyword_and_space = match_obj.group(1)
                value_str = match_obj.group(2)
                
                # Get decimal places from original value
                if '.' in value_str:
                    decimal_places = len(value_str.split('.')[1])
                else:
                    decimal_places = 0
                
                new_value = float(value_str) / alpha
                return f"{keyword_and_space}{new_value:.{decimal_places}f}"

            # Process each line
            for line in lines:
                if line.strip().startswith("make_tracks"):
                    # Find and replace x_pitch and y_pitch values
                    line = re.sub(r"(-x_pitch\s+)([\d\.]+)", replace_pitch, line)
                    line = re.sub(r"(-y_pitch\s+)([\d\.]+)", replace_pitch, line)
                modified_lines.append(line)

            # Write the modified lines back to the file
            with open(tracks_file_path, "w") as file:
                file.writelines(modified_lines)
            
            logger.info(f"Successfully updated pitches in {tracks_file_path}.")

        except FileNotFoundError:
            logger.error(f"Could not find tracks file to scale at {tracks_file_path}. Please check the path.")
        except Exception as e:
            logger.error(f"An error occurred while scaling track pitches: {e}")

    def _scale_tech_lef(self, alpha: float):
        """
        Scale technology LEF dimensions by dividing linear values by alpha and
        area values (e.g., MINAREA) by alpha^2.

        Applies to: SITE SIZE, LAYER WIDTH/MINWIDTH/SPACING/PITCH/OFFSET/MINAREA,
        VIA/CUT parameters (SIZE, SPACING, ENCLOSURE/OVERHANG), MANUFACTURINGGRID,
        and any RECT coordinates.
        """
        if alpha == 1.0:
            logger.info("Alpha is 1.0, no scaling of tech LEF required.")
            return
        if alpha <= 0:
            logger.error("Alpha must be positive. Skipping tech LEF scaling.")
            return

        lef_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign_tech.lef")
        try:
            with open(lef_path, "r") as f:
                lines = f.readlines()

            def get_decimal_places(val_str: str) -> int:
                """Get the number of decimal places from the original value string"""
                if '.' in val_str:
                    return len(val_str.split('.')[1])
                return 0

            def scale_len(val: str) -> str:
                decimal_places = get_decimal_places(val)
                scaled_val = float(val) / alpha
                return f"{scaled_val:.{decimal_places}f}"

            def scale_area(val: str) -> str:
                decimal_places = get_decimal_places(val)
                scaled_val = float(val) / (alpha * alpha)
                return f"{scaled_val:.{decimal_places}f}"

            # Process line by line to avoid regex conflicts
            modified_lines = []
            in_spacing_table = False
            
            for line in lines:
                original_line = line
                
                # Track if we're in a SPACINGTABLE block
                if "SPACINGTABLE" in line:
                    in_spacing_table = True
                elif line.strip().endswith(";") and in_spacing_table:
                    in_spacing_table = False
                
                # Scale SITE SIZE
                if re.match(r"\s*SIZE\s+[\d\.]+\s+BY\s+[\d\.]+\s*;", line):
                    line = re.sub(r"(\s*SIZE\s+)([\d\.]+)\s+BY\s+([\d\.]+)(\s*;)",
                                lambda m: f"{m.group(1)}{scale_len(m.group(2))} BY {scale_len(m.group(3))}{m.group(4)}",
                                line)
                
                # Scale MANUFACTURINGGRID - always set to 0.0045 for OpenROAD compatibility
                elif re.match(r"\s*MANUFACTURINGGRID\s+[\d\.]+\s*;", line):
                    line = re.sub(r"(\s*MANUFACTURINGGRID\s+)([\d\.]+)(\s*;)",
                                lambda m: f"{m.group(1)}0.0045{m.group(3)}",
                                line)
                
                # Scale standalone LAYER properties (not in SPACINGTABLE)
                elif not in_spacing_table and re.match(r"\s*(WIDTH|MINWIDTH|SPACING|PITCH|OFFSET|MINAREA)\s+[\d\.]+\s*;", line):
                    if "MINAREA" in line:
                        line = re.sub(r"(\s*MINAREA\s+)([\d\.]+)(\s*;)",
                                    lambda m: f"{m.group(1)}{scale_area(m.group(2))}{m.group(3)}",
                                    line)
                    else:
                        line = re.sub(r"(\s*(?:WIDTH|MINWIDTH|SPACING|PITCH|OFFSET)\s+)([\d\.]+)(\s*;)",
                                    lambda m: f"{m.group(1)}{scale_len(m.group(2))}{m.group(3)}",
                                    line)
                
                # Scale CUT SIZE
                elif re.match(r"\s*CUT\s+SIZE\s+[\d\.]+\s+BY\s+[\d\.]+\s*;", line):
                    line = re.sub(r"(\s*CUT\s+SIZE\s+)([\d\.]+)\s+BY\s+([\d\.]+)(\s*;)",
                                lambda m: f"{m.group(1)}{scale_len(m.group(2))} BY {scale_len(m.group(3))}{m.group(4)}",
                                line)
                
                # Scale ENCLOSURE and OVERHANG
                elif re.match(r"\s*(ENCLOSURE|OVERHANG)\s+[\d\.]+\s*;", line):
                    line = re.sub(r"(\s*(?:ENCLOSURE|OVERHANG)\s+)([\d\.]+)(\s*;)",
                                lambda m: f"{m.group(1)}{scale_len(m.group(2))}{m.group(3)}",
                                line)
                
                # Scale RECT coordinates
                elif "RECT" in line:
                    line = re.sub(r"(RECT\s+)([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)",
                                lambda m: f"{m.group(1)}{scale_len(m.group(2))} {scale_len(m.group(3))} {scale_len(m.group(4))} {scale_len(m.group(5))}",
                                line)
                
                # Scale SAMENET spacing values
                elif "SAMENET" in line:
                    line = re.sub(r"(SAMENET\s+\w+\s+\w+\s+)([\d\.]+)(\s*;)",
                                lambda m: f"{m.group(1)}{scale_len(m.group(2))}{m.group(3)}",
                                line)
                
                # Scale VIARULE SPACING
                elif "SPACING" in line and "BY" in line and "VIARULE" not in line:
                    line = re.sub(r"(SPACING\s+)([\d\.]+)\s+BY\s+([\d\.]+)(\s*;)",
                                lambda m: f"{m.group(1)}{scale_len(m.group(2))} BY {scale_len(m.group(3))}{m.group(4)}",
                                line)
                
                # Scale SPACINGTABLE values
                elif in_spacing_table:
                    # Scale PARALLELRUNLENGTH values
                    if "PARALLELRUNLENGTH" in line:
                        line = re.sub(r"([\d\.]+)",
                                    lambda m: scale_len(m.group(0)),
                                    line)
                    # Scale WIDTH table entries
                    elif re.match(r"\s*WIDTH\s+[\d\.]+", line):
                        line = re.sub(r"([\d\.]+)",
                                    lambda m: scale_len(m.group(0)),
                                    line)
                
                modified_lines.append(line)

            with open(lef_path, "w") as f:
                f.writelines(modified_lines)
            logger.info(f"Scaled technology LEF at {lef_path} with alpha={alpha}.")
        except FileNotFoundError:
            logger.error(f"Tech LEF not found at {lef_path}.")
        except Exception as e:
            logger.error(f"Error scaling tech LEF: {e}")

    def _scale_stdcell_lef(self, alpha: float):
        """
        Scale standard-cell LEF dimensions by dividing linear values by alpha.
        Applies to: MACRO SIZE, ORIGIN, PIN/OBS RECTs, and any RECT coordinates.
        """
        if alpha == 1.0:
            logger.info("Alpha is 1.0, no scaling of stdcell LEF required.")
            return
        if alpha <= 0:
            logger.error("Alpha must be positive. Skipping stdcell LEF scaling.")
            return

        lef_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign_stdcell.lef")
        try:
            with open(lef_path, "r") as f:
                data = f.read()

            def get_decimal_places(val_str: str) -> int:
                """Get the number of decimal places from the original value string"""
                if '.' in val_str:
                    return len(val_str.split('.')[1])
                return 0

            def scale_len(val: str) -> str:
                decimal_places = get_decimal_places(val)
                scaled_val = float(val) / alpha
                return f"{scaled_val:.{decimal_places}f}"

            # Scale MACRO SIZE
            data = re.sub(r"(^\s*SIZE\s+)([\d\.]+)\s+BY\s+([\d\.]+)(\s*;)",
                          lambda m: f"{m.group(1)}{scale_len(m.group(2))} BY {scale_len(m.group(3))}{m.group(4)}",
                          data, flags=re.MULTILINE)

            # Scale ORIGIN coordinates
            data = re.sub(r"(^\s*ORIGIN\s+)([\d\.]+)\s+([\d\.]+)(\s*;)",
                          lambda m: f"{m.group(1)}{scale_len(m.group(2))} {scale_len(m.group(3))}{m.group(4)}",
                          data, flags=re.MULTILINE)

            # Scale all RECT coordinates (for PINs, OBS, etc.)
            data = re.sub(r"(RECT\s+)([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)",
                          lambda m: f"{m.group(1)}{scale_len(m.group(2))} {scale_len(m.group(3))} {scale_len(m.group(4))} {scale_len(m.group(5))}",
                          data)

            # Scale FOREIGN offsets if present
            data = re.sub(r"(FOREIGN\s+\w+\s+)([\d\.]+)\s+([\d\.]+)(\s*;)",
                          lambda m: f"{m.group(1)}{scale_len(m.group(2))} {scale_len(m.group(3))}{m.group(4)}",
                          data)

            with open(lef_path, "w") as f:
                f.write(data)
            logger.info(f"Scaled stdcell LEF at {lef_path} with alpha={alpha}.")
        except FileNotFoundError:
            logger.error(f"Stdcell LEF not found at {lef_path}.")
        except Exception as e:
            logger.error(f"Error scaling stdcell LEF: {e}")

    def _scale_pdn_config(self, alpha: float):
        """
        Scale PDN configuration to match the scaled manufacturing grid.
        Ensures PDN widths are multiples of the manufacturing grid.
        """
        if alpha == 1.0:
            logger.info("Alpha is 1.0, no scaling of PDN config required.")
            return
        if alpha <= 0:
            logger.error("Alpha must be positive. Skipping PDN config scaling.")
            return

        pdn_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign.pdn.tcl")
        lef_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign_tech.lef")
        
        try:
            # Read the actual manufacturing grid from the scaled tech LEF
            manufacturing_grid = 0.0050  # default fallback
            with open(lef_path, "r") as f:
                for line in f:
                    if "MANUFACTURINGGRID" in line:
                        match = re.search(r"MANUFACTURINGGRID\s+([\d\.]+)", line)
                        if match:
                            manufacturing_grid = float(match.group(1))
                            break
            
            with open(pdn_path, "r") as f:
                lines = f.readlines()

            def get_decimal_places(val_str: str) -> int:
                """Get the number of decimal places from the original value string"""
                if '.' in val_str:
                    return len(val_str.split('.')[1])
                return 0

            def scale_len(val: str) -> str:
                decimal_places = get_decimal_places(val)
                scaled_val = float(val) / alpha
                return f"{scaled_val:.{decimal_places}f}"

            def round_to_manufacturing_grid(val: float) -> float:
                """Round value to nearest multiple of 0.0045 manufacturing grid"""
                grid = 0.0045
                rounded_val = round(val / grid) * grid
                return rounded_val
            
            def round_to_grid_width(val: float, original_val_str: str) -> float:
                """Round value to nearest multiple of PDN width grid"""
                # OpenROAD enforces 0.0090 grid for PDN widths
                grid = 0.0090
                rounded_val = round(val / grid) * grid
                return rounded_val
            
            def round_to_grid_offset(val: float, original_val_str: str) -> float:
                """Round value to nearest multiple of PDN offset grid"""
                # Use manufacturing grid for all PDN values
                return round_to_manufacturing_grid(val)

            modified_lines = []
            for line in lines:
                # Scale PDN stripe widths and pitches
                if "-width" in line:
                    # Extract and scale width values using width grid (0.0090)
                    line = re.sub(r"(-width\s+\{)([\d\.]+)(\})",
                                lambda m: f"{m.group(1)}{round_to_grid_width(float(m.group(2)) / alpha, m.group(2)):.4f}{m.group(3)}",
                                line)
                
                if "-pitch" in line:
                    # Extract and scale pitch values using manufacturing grid
                    line = re.sub(r"(-pitch\s+\{)([\d\.]+)(\})",
                                lambda m: f"{m.group(1)}{round_to_manufacturing_grid(float(m.group(2)) / alpha):.4f}{m.group(3)}",
                                line)
                
                if "-offset" in line:
                    # Extract and scale offset values using manufacturing grid
                    line = re.sub(r"(-offset\s+\{)([\d\.]+)(\})",
                                lambda m: f"{m.group(1)}{round_to_manufacturing_grid(float(m.group(2)) / alpha):.4f}{m.group(3)}",
                                line)
                
                if "-halo" in line:
                    # Extract and scale halo values using manufacturing grid
                    line = re.sub(r"(-halo\s+\{)([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)(\})",
                                lambda m: f"{m.group(1)}{round_to_manufacturing_grid(float(m.group(2)) / alpha):.4f} {round_to_manufacturing_grid(float(m.group(3)) / alpha):.4f} {round_to_manufacturing_grid(float(m.group(4)) / alpha):.4f} {round_to_manufacturing_grid(float(m.group(5)) / alpha):.4f}{m.group(6)}",
                                line)
                
                modified_lines.append(line)

            with open(pdn_path, "w") as f:
                f.writelines(modified_lines)
            logger.info(f"Scaled PDN config at {pdn_path} with alpha={alpha}, grid={manufacturing_grid:.6f}.")
        except FileNotFoundError:
            logger.error(f"PDN config not found at {pdn_path}.")
        except Exception as e:
            logger.error(f"Error scaling PDN config: {e}")

    def _scale_die_area(self, alpha: float):
        """
        Scale die area and core area in codesign_top.tcl to match the scaled technology.
        """
        if alpha == 1.0:
            logger.info("Alpha is 1.0, no scaling of die area required.")
            return
        if alpha <= 0:
            logger.error("Alpha must be positive. Skipping die area scaling.")
            return

        tcl_path = os.path.join(self.directory, "tcl", "codesign_top.tcl")
        
        try:
            with open(tcl_path, "r") as f:
                lines = f.readlines()

            def get_decimal_places(val_str: str) -> int:
                """Get the number of decimal places from the original value string"""
                if '.' in val_str:
                    return len(val_str.split('.')[1])
                return 0

            def scale_val(val_str: str) -> str:
                decimal_places = get_decimal_places(val_str)
                scaled_val = float(val_str) / alpha
                # Round to manufacturing grid for die/core area
                grid = 0.0045
                grid_aligned = round(scaled_val / grid) * grid
                return f"{grid_aligned:.{decimal_places}f}"

            modified_lines = []
            for line in lines:
                # Scale die_area values
                if "set die_area" in line:
                    line = re.sub(r"(\{)([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)(\})",
                                lambda m: f"{m.group(1)}{scale_val(m.group(2))} {scale_val(m.group(3))} {scale_val(m.group(4))} {scale_val(m.group(5))}{m.group(6)}",
                                line)
                
                # Scale core_area values
                if "set core_area" in line:
                    line = re.sub(r"(\{)([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)(\})",
                                lambda m: f"{m.group(1)}{scale_val(m.group(2))} {scale_val(m.group(3))} {scale_val(m.group(4))} {scale_val(m.group(5))}{m.group(6)}",
                                line)
                
                modified_lines.append(line)

            with open(tcl_path, "w") as f:
                f.writelines(modified_lines)
            logger.info(f"Scaled die area in {tcl_path} with alpha={alpha}.")
        except FileNotFoundError:
            logger.error(f"TCL file not found at {tcl_path}.")
        except Exception as e:
            logger.error(f"Error scaling die area: {e}")

    def _scale_vars_file(self, alpha: float):
        """
        Scale various parameters in codesign.vars to match the scaled technology.
        """
        if alpha == 1.0:
            logger.info("Alpha is 1.0, no scaling of vars file required.")
            return
        if alpha <= 0:
            logger.error("Alpha must be positive. Skipping vars file scaling.")
            return

        vars_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign.vars")
        
        try:
            with open(vars_path, "r") as f:
                lines = f.readlines()

            def get_decimal_places(val_str: str) -> int:
                """Get the number of decimal places from the original value string"""
                if '.' in val_str:
                    return len(val_str.split('.')[1])
                return 0

            def scale_val(val_str: str) -> str:
                decimal_places = get_decimal_places(val_str)
                scaled_val = float(val_str) / alpha
                return f"{scaled_val:.{decimal_places}f}"

            modified_lines = []
            for line in lines:
                # Scale tapcell distance
                if "-distance" in line:
                    line = re.sub(r"(-distance\s+)([\d\.]+)",
                                lambda m: f"{m.group(1)}{scale_val(m.group(2))}",
                                line)
                
                # Scale macro place halo
                if "macro_place_halo" in line:
                    line = re.sub(r"(\{)([\d\.]+)\s+([\d\.]+)(\})",
                                lambda m: f"{m.group(1)}{scale_val(m.group(2))} {scale_val(m.group(3))}{m.group(4)}",
                                line)
                
                # Scale macro place channel
                if "macro_place_channel" in line:
                    line = re.sub(r"(\{)([\d\.]+)\s+([\d\.]+)(\})",
                                lambda m: f"{m.group(1)}{scale_val(m.group(2))} {scale_val(m.group(3))}{m.group(4)}",
                                line)
                
                # Scale tie separation
                if "tie_separation" in line:
                    line = re.sub(r"(\s+)([\d\.]+)$",
                                lambda m: f"{m.group(1)}{scale_val(m.group(2))}",
                                line)
                
                # Scale cts cluster diameter
                if "cts_cluster_diameter" in line:
                    line = re.sub(r"(\s+)([\d\.]+)$",
                                lambda m: f"{m.group(1)}{scale_val(m.group(2))}",
                                line)
                
                modified_lines.append(line)

            with open(vars_path, "w") as f:
                f.writelines(modified_lines)
            logger.info(f"Scaled vars file at {vars_path} with alpha={alpha}.")
        except FileNotFoundError:
            logger.error(f"Vars file not found at {vars_path}.")
        except Exception as e:
            logger.error(f"Error scaling vars file: {e}")

    def run_openroad_executable(self):
        """
        Runs the OpenROAD executable. Run this after setup.
        """
        logger.info("Starting OpenROAD run.")
        old_dir = os.getcwd()
        os.chdir(self.directory + "/tcl")
        logger.info(f"Changed directory to {self.directory + '/tcl'}")
        print("running openroad")
        logger.info("Running OpenROAD command.")
        os.system(os.path.dirname(os.path.abspath(__file__)) + "/OpenROAD/build/src/openroad codesign_top.tcl > " + self.directory + "/codesign_pd.log")#> /dev/null 2>&1
        print("done")
        logger.info("OpenROAD run completed.")
        os.chdir(old_dir)
        logger.info(f"Returned to original directory {old_dir}")


    def mux_listing(self, graph, node_output, wire_length_by_edge):
        """
        goes through the graph and finds nodes that are not Muxs. If it encounters one, it will go through
        the graph to find the path of Muxs until the another non-Mux node is found. All rcl are put into a
        list and added as an edge attribute for the non-mux node to non-mux node connection

        param:
            graph: graph with the net attributes already attached
            node_output: dict of nodes and their respective outputs
        """
        #print(f"wire_length_by_edge before modification: {wire_length_by_edge}")
        logger.info("Starting mux listing.")
        edges_to_remove = set()
        for node in graph.nodes():
            #print(f"considering node {node}")
            if "Mux" not in node:
                #print(f"outputs of {node}: {node_output[node]}")
                for output in node_output[node]:
                    path = []
                    if "Mux" in output:
                        while "Mux" in output:
                            path.append(output)
                            output = node_output[output][0]
                        graph.add_edge(node, output)
                        #print(f"path from {node} to {output}: {path}")
                        if len(path) != 0 and (node, output) not in wire_length_by_edge:
                            #print(f"adding wire length by edge")
                            wire_length_by_edge[(node, output)] = wire_length_by_edge[(node, path[0])]
                            edges_to_remove.add((node, path[0]))
                            for i in range(1, len(path)):
                                wire_length_by_edge[(node, output)]["total_wl"] += wire_length_by_edge[(path[i-1], path[i])]["total_wl"]
                                wire_length_by_edge[(node, output)]["metal1"] += wire_length_by_edge[(path[i-1], path[i])]["metal1"]
                                wire_length_by_edge[(node, output)]["metal2"] += wire_length_by_edge[(path[i-1], path[i])]["metal2"]
                                wire_length_by_edge[(node, output)]["metal3"] += wire_length_by_edge[(path[i-1], path[i])]["metal3"]
                                edges_to_remove.add((path[i-1], path[i]))
                            wire_length_by_edge[(node, output)]["total_wl"] += wire_length_by_edge[(path[-1], output)]["total_wl"]
                            wire_length_by_edge[(node, output)]["metal1"] += wire_length_by_edge[(path[-1], output)]["metal1"]
                            wire_length_by_edge[(node, output)]["metal2"] += wire_length_by_edge[(path[-1], output)]["metal2"]
                            wire_length_by_edge[(node, output)]["metal3"] += wire_length_by_edge[(path[-1], output)]["metal3"]
                            edges_to_remove.add((path[-1], output))
                            #print(f"wire length by edge after modification: {wire_length_by_edge[(node, output)]}")
        for edge in edges_to_remove:
            #print(f"removing edge {edge}")
            wire_length_by_edge.pop(edge)
        return wire_length_by_edge


    def mux_removal(self, graph: nx.DiGraph):
        """
        Removes the mux nodes from the graph. Does not do the connecting
        param:
            graph: graph with the new edge connections, after mux listing
        """
        logger.info("Removing mux nodes from graph.")
        reference = copy.deepcopy(graph.nodes())
        for node in reference:
            if "Mux" in node:
                graph.remove_node(node)
                logger.info(f"Removed mux node: {node}")


    def coord_scraping(
        self,
        graph: nx.DiGraph,
        node_to_num: dict,
        final_def_directory: str = None,
    ):
        """
        going through the .def file and getting macro placements and nets
        param:
            graph: digraph to add coordinate attribute to nodes
            node_to_num: dict that gives component id equivalent for node
            final_def_directory: final def directory, defaults to def directory in openroad
        return:
            graph: digraph with the new coordinate attributes
            component_nets: dict that list components for the respective net id
        """
        logger.info("Scraping coordinates and nets from DEF file.")
        pattern = r"_\w+_\s+\w+\s+\+\s+PLACED\s+\(\s*\d+\s+\d+\s*\)\s+\w+\s*;"
        net_pattern = r"-\s(_\d+_)\s((?:\(\s_\d+_\s\w+\s\)\s*)+).*"
        component_pattern = r"(_\w+_)"
        if final_def_directory is None:
            final_def_directory = self.directory + "/results/final_generated-tcl.def"
        final_def_data = open(final_def_directory)
        final_def_lines = final_def_data.readlines()
        macro_coords = {}
        component_nets = {}
        for line in final_def_lines:
            if re.search(pattern, line) is not None:
                coord = re.findall(r"\((.*?)\)", line)[0].split()
                match = re.search(component_pattern, line)
                macro_coords[match.group(0)] = {"x": float(coord[0]), "y": float(coord[1])}
                logger.info(f"Found macro {match.group(0)} at ({coord[0]}, {coord[1]})")
            if re.search(net_pattern, line) is not None:
                pins = re.findall(r"\(\s(.*?)\s\w+\s\)", line)
                match = re.search(component_pattern, line)
                component_nets[match.group(0)] = pins
                logger.info(f"Found net {match.group(0)} with pins {pins}")

        for node in node_to_num:
            coord = macro_coords[node_to_num[node]]
            graph.nodes[node]["x"] = coord["x"]
            graph.nodes[node]["y"] = coord["y"]
            logger.info(f"Assigned coordinates to node {node}: {coord}")
        logger.info("Coordinate scraping complete.")
        return graph, component_nets


        
    

    def extraction(self, graph, arg_parasitics, net_out_dict, node_output, lef_data, node_to_num): 
        # 3. extract parasitics
        logger.info(f"Starting extraction with parasitics option: {arg_parasitics}")
        dict = {}
        if arg_parasitics == "detailed":
            logger.info("Running detailed place and route.")
            dict, graph = self.detailed_place_n_route(
                graph, net_out_dict, node_output, lef_data, node_to_num
            )
            logger.info("Detailed extraction complete.")
        elif arg_parasitics == "estimation":
            logger.info("Running estimated place and route.")
            dict, graph = self.estimated_place_n_route(
                graph, net_out_dict, node_output, lef_data, node_to_num
            )
            logger.info("Estimated extraction complete.")

        return dict, graph

    def estimated_place_n_route(
        self,
        graph: nx.DiGraph,
        net_out_dict: dict,
        node_output: dict,
        lef_data: dict,
        node_to_num: dict,
    ) -> dict:
        """
        runs openroad, calculates rcl, and then adds attributes to the graph

        params:
            graph: networkx graph
            net_out_dict: dict that lists nodes and thier respective edges (all nodes have one output)
            node_output: dict that lists nodes and their respective output nodes
            lef_data: dict with layer information (units, res, cap, width)
            node_to_num: dict that gives component id equivalent for node
        returns:
            dict: contains list of resistance, capacitance, length, and net data
            graph: newly modified digraph with rcl attributes
        """

        # run openroad
        logger.info("Starting estimated place and route.")
        self.run_openroad_executable()

        wire_length_df = est.parse_route_guide_with_layer_breakdown(self.directory + "/results/codesign_codesign-tcl.route_guide")
        wire_length_by_edge = {}
        for node in net_out_dict:
            for output in node_output[node]:
                for net in net_out_dict[node]:
                    if (node, output) not in wire_length_by_edge:
                        wire_length_by_edge[(node, output)] = wire_length_df.loc[net]
                    else:
                        wire_length_by_edge[(node, output)] += wire_length_df.loc[net]
        self.export_graph(graph, "estimated_with_mux")

        wire_length_by_edge = self.mux_listing(graph, node_output, wire_length_by_edge)
        self.mux_removal(graph)

        self.export_graph(graph, "estimated_nomux")

        return wire_length_by_edge, graph


    def detailed_place_n_route(
        self,
        graph: nx.DiGraph,
        net_out_dict: dict,
        node_output: dict,
        lef_data: dict,
        node_to_num: dict,
    ) -> dict:
        """
        runs openroad, calculates rcl, and then adds attributes to the graph

        params:
            graph: networkx graph
            net_out_dict:  dict that lists nodes and their respective net (all components utilize one output, therefore this is a same assumption to use)
            node_output: dict that lists nodes and their respective output nodes
            lef_data: dict with layer information (units, res, cap, width)
            node_to_num: dict that gives component id equivalent for node
        returns:
            dict: contains list of resistance, capacitance, length, and net data
            graph: newly modified digraph with rcl attributes
        """

        # run openroad
        logger.info("Starting detailed place and route.")
        self.run_openroad_executable()

        # run parasitic_calc and length_calculations
        graph, _ = self.coord_scraping(graph, node_to_num)
        net_cap, net_res = self.det.parasitic_calc(self.directory + "/results/generated-tcl.spef")

        length_dict = det.length_calculations(lef_data["units"], self.directory + "/results/final_generated-tcl.def")

        # add edge attributions
        net_graph_data = []
        res_graph_data = []
        cap_graph_data = []
        len_graph_data = []
        
        for output_net in net_out_dict:
            for net in net_out_dict[output_net]:
                for node in node_output[output_net]:
                    graph[output_net][node]["net"] = net
                    graph[output_net][node]["net_length"] = length_dict[net]
                    graph[output_net][node]["net_res"] = float(net_res[net])
                    graph[output_net][node]["net_cap"] = float(net_cap[net])
                net_graph_data.append(net)
                len_graph_data.append(float(length_dict[net]))  # length
                res_graph_data.append(float(net_res[net]) if net in net_res else 0)  # ohms
                cap_graph_data.append(float(net_cap[net]) if net in net_cap else 0)  # picofarads
            
            

        self.export_graph(graph, "detailed")

        self.mux_listing(graph, node_output)
        self.mux_removal(graph)

        self.export_graph(graph, "detailed_nomux")

        return {
            "res": res_graph_data,
            "cap": cap_graph_data,
            "length": len_graph_data,
            "net": net_graph_data,
        }, graph


    def none_place_n_route(
        self,
        graph: nx.DiGraph,
    ) -> dict:
        """
        runs openroad, calculates rcl, and then adds attributes to the graph
        params:
            graph: networkx graph
        returns:
            graph: newly modified digraph with rcl attributes
        """

        # edge attribution
        logger.info("Running none_place_n_route: setting default edge attributes.")
        for u, v in graph.edges():
            graph[u][v]["net"] = 0
            graph[u][v]["net_length"] = 0
            graph[u][v]["net_res"] = 0
            graph[u][v]["net_cap"] = 0
            logger.info(f"Set default attributes for edge ({u}, {v})")

        logger.info("none_place_n_route finished.")
        return graph
    

    @staticmethod
    def export_graph(graph, est_or_det: str):
        logger.info(f"Exporting graph to GML for {est_or_det}.")
        if not os.path.exists("openroad_interface/results/"):
            os.makedirs("openroad_interface/results/")
            logger.info("Created results directory.")
        nx.write_gml(
            graph, "openroad_interface/results/" + est_or_det + ".gml"
        )
        logger.info(f"Graph exported to openroad_interface/results/{est_or_det}.gml")