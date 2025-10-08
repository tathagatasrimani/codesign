import logging
import os
import re

logger = logging.getLogger(__name__)




L_EFF_FREEPDK45 = 0.025E-6  # effective channel length for FreePDK45 (microns)
## NOTE: this is an approximation.

class ScaleLefFiles:
    def __init__(self, cfg, codesign_root_dir):
        """
        Scales LEF files by a given factor.

        :param cfg: top level codesign config file
        :param codesign_root_dir: root directory of codesign (where src and test are)
        """
        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.directory = os.path.join(self.codesign_root_dir, "src/tmp/pd")
        
        self.original_manufacturing_grid = 0.005  # default 
        self.new_manufacturing_grid = 0.005  # default
        self.database_units_per_micron = 2000  # default
        self.database_units_scale = 1  ## scale the number of database units per micron by this factor
        self.min_manufacturing_grid = 0.0005  # manufacturing grid must be a multiple of this

    def scale_lef_files(self, L_eff_current: float):
        """
        Scales the technology and standard cell LEF files by the given factor.

        :param L_eff_current: The current effective channel length (L_eff) in microns.
        """

        ## NOTE: Alpha is the factor that we are scaling the technology down by vs FreePDK45.
        ## if alpha > 1, we are scaling DOWN (making features smaller)

        alpha = L_EFF_FREEPDK45 / L_eff_current

        logger.info(f"Scaling LEF files with alpha = {alpha:.4f} (L_eff_current = {L_eff_current:.4f} microns)")

        if alpha > 5.0:
            ## scale the database units scale by 5x if needed. Otherwise, leave it at 1x to avoid integer overflows.
            self.database_units_scale = 5
            self.database_units_per_micron = 10000  # database units per micron
            self.min_manufacturing_grid = 0.0001  # manufacturing grid must be a multiple of this
            self.write_DBU(self.database_units_per_micron)


        self.get_original_manufacturing_grid()
        self.new_manufacturing_grid = self.find_new_manufacturing_grid(alpha)

        logger.info(f"New manufacturing grid after scaling: {self.new_manufacturing_grid:.6f}")

        ## ensure that we are scaling by the proper factor.
        quantized_alpha = self.original_manufacturing_grid / self.new_manufacturing_grid
        
        self.scale_track_pitches(quantized_alpha)
        self.scale_tech_lef(quantized_alpha)
        self.scale_stdcell_lef(quantized_alpha)
        self.scale_pdn_config(quantized_alpha)
        self.scale_vars_file(quantized_alpha)

        verify = self.verify_on_grid
        
        verify(os.path.join(self.directory, "tcl", "codesign_files", "codesign.tracks"), tag="codesign.tracks")
        verify(os.path.join(self.directory, "tcl", "codesign_files", "codesign_tech.lef"), tag="codesign_tech.lef")
        verify(os.path.join(self.directory, "tcl", "codesign_files", "codesign_stdcell.lef"), tag="codesign_stdcell.lef")
        verify(os.path.join(self.directory, "tcl", "codesign_top.tcl"), tag="codesign_top.tcl")
        verify(os.path.join(self.directory, "tcl", "codesign_files", "codesign.vars"), tag="codesign.vars")
        verify(os.path.join(self.directory, "tcl", "codesign_files", "codesign.pdn.tcl"), tag="codesign.pdn.tcl")


    ###################################################################################################################################################################
    ## Helper methods

    def find_new_manufacturing_grid(self, alpha: float) -> float:
        """
        Computes the new manufacturing grid after scaling by alpha.
        Reads the original manufacturing grid from the original tech LEF file.

        :param alpha: The scaling factor.
        :return: The new manufacturing grid value, rounded to an integer multiple of 0.0001.
        """
        new_grid = self.original_manufacturing_grid / alpha

        # Round new_grid to the nearest multiple of self.min_manufacturing_grid
        new_grid_rounded = round(new_grid / self.min_manufacturing_grid) * self.min_manufacturing_grid

        logger.info(
            f"Original manufacturing grid: {self.original_manufacturing_grid:.6f}, "
            f"New manufacturing grid (raw): {new_grid:.6f}, "
            f"Rounded to {self.min_manufacturing_grid:.4f} multiple: {new_grid_rounded:.6f}"
        )
        return new_grid_rounded
    
    
    def get_original_manufacturing_grid(self) -> float:
        """
        Read the MANUFACTURINGGRID value from the *original* (unscaled) technology LEF.
        Default to 0.005 if not found.
        """
        tech_lef = os.path.join(self.codesign_root_dir, "openroad_interface/tcl/codesign_files", "codesign_tech.lef")
        grid = 0.005
        if os.path.exists(tech_lef):
            with open(tech_lef) as f:
                for line in f:
                    if "MANUFACTURINGGRID" in line:
                        m = re.search(r"MANUFACTURINGGRID\s+([\d\.]+)", line)
                        if m:
                            grid = float(m.group(1))
                            break
        return grid

    @staticmethod
    def get_decimal_places(val_str: str, min_dec=4) -> int:
            if '.' in val_str:
                return max(min_dec, len(val_str.split('.')[1]))
            return min_dec

    def scale_length(self, alpha: float, val: str) -> str:
        decimals = self.get_decimal_places(val, 4)
        scaled_val = float(val) / alpha
        ##scaled_val *= self.database_units_scale
        scaled_val = self.round_to_manufacturing_grid(scaled_val)
        return f"{scaled_val:.{decimals}f}"
    
    def scale_area(self, alpha: float, val: str) -> str:
        decimals = self.get_decimal_places(val, 6)
        scaled_val = float(val) / (alpha * alpha)
        ##scaled_val *= self.database_units_scale**2
        scaled_val = self.round_to_manufacturing_grid_area(scaled_val)
        return f"{scaled_val:.{decimals}f}"

    def round_to_manufacturing_grid(self, val: float) -> float:
        """
        Round a linear value to the nearest multiple of the new manufacturing grid.
        """
        return round(val / self.new_manufacturing_grid) * self.new_manufacturing_grid

    def round_to_manufacturing_grid_area(self, val: float) -> float:
        """
        Round an area value to the nearest multiple of new manufacturing grid^2.
        """
        return round(val / (self.new_manufacturing_grid * self.new_manufacturing_grid)) * (self.new_manufacturing_grid * self.new_manufacturing_grid)
    
    ###################################################################################################################################################################

    def write_DBU(self, new_dbu):
        """ Sets the database units in the tech LEF file.
        """
        ## open the tech lef file
        lef_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign_tech.lef")
        with open(lef_path, "r") as f:
            lines = f.readlines()
        
        modified_lines = []
        for line in lines:
            if "DATABASE MICRONS" in line:
                line = re.sub(r"(DATABASE MICRONS\s+)(\d+)(\s*;)", lambda m: f"{m.group(1)}{new_dbu}{m.group(3)}", line)
            modified_lines.append(line)

        with open(lef_path, "w") as f:
            f.writelines(modified_lines)

    def scale_track_pitches(self, alpha: float):
        """
        Updates the track pitches in the temporary codesign.tracks file.
        This method reads the file, scales the x and y pitch values by dividing
        them by the alpha factor, and writes the file back.

        :param alpha: The scaling factor to divide the pitches by.
        """

        logger.info(f"Scaling track pitches by a division factor of {alpha}.")
        
        # Construct the full path to the file inside the temporary directory
        tracks_file_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign.tracks")

        # Read all lines from the file
        with open(tracks_file_path, "r") as file:
            lines = file.readlines()

        modified_lines = []

        # Define a helper function to perform the replacement using regex
        def replace_grid_param(match_obj):
            keyword_and_space = match_obj.group(1)
            value_str = match_obj.group(2)

            new_value = self.scale_length(alpha, value_str)

            return f"{keyword_and_space}{new_value}"

        modified_lines = []
        for line in lines:
            if line.strip().startswith("make_tracks"):
                # Round both offsets and pitches
                line = re.sub(r"(-x_offset\s+)([\d\.]+)", replace_grid_param, line)
                line = re.sub(r"(-x_pitch\s+)([\d\.]+)", replace_grid_param, line)
                line = re.sub(r"(-y_offset\s+)([\d\.]+)", replace_grid_param, line)
                line = re.sub(r"(-y_pitch\s+)([\d\.]+)", replace_grid_param, line)
                line = re.sub(r"(-x_pitch\s+)([\d\.]+)", replace_grid_param, line)
                line = re.sub(r"(-y_pitch\s+)([\d\.]+)", replace_grid_param, line)
            modified_lines.append(line)

        # Write the modified lines back to the file
        with open(tracks_file_path, "w") as file:
            file.writelines(modified_lines)
        
        logger.info(f"Successfully updated pitches in {tracks_file_path}.")

    def scale_tech_lef(self, alpha: float):
        """
        Scale technology LEF dimensions by dividing linear values by alpha and
        area values (e.g., MINAREA) by alpha^2. Also scales the manufacturing grid
        itself (new_grid = old_grid / alpha) and snaps all values to the new grid.

        Applies to: SITE SIZE, LAYER WIDTH/MINWIDTH/SPACING/PITCH/OFFSET/MINAREA,
        VIA/CUT parameters (SIZE, SPACING, ENCLOSURE/OVERHANG), MANUFACTURINGGRID,
        and any RECT coordinates.
        """

        lef_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign_tech.lef")

        logger.info(
            f"Scaling tech LEF with alpha={alpha}. "
            f"old_grid={self.original_manufacturing_grid:.6f} → new_grid={self.new_manufacturing_grid:.6f}"
        )

        with open(lef_path, "r") as f:
            lines = f.readlines()

        # ---- Helpers ----
        def scale_len(val: str) -> str:
            return self.scale_length(alpha, val)

        def scale_area(val: str) -> str:
            return self.scale_area(alpha, val)

        modified_lines = []
        in_spacing_table = False

        for line in lines:
            # Handle SPACINGTABLE blocks first and ensure the terminating line is processed.
            if "SPACINGTABLE" in line or in_spacing_table:
                # enter spacing table if we see the keyword
                if "SPACINGTABLE" in line:
                    in_spacing_table = True

                # Scale any numeric tokens inside spacing table lines (parallel runlengths, WIDTH rows, etc.)
                # Use a float-aware regex to catch signed/unsigned decimals.
                line = re.sub(r"([-+]?\d*\.?\d+)", lambda m: scale_len(m.group(1)), line)

                # Always append the processed spacing-table line.
                modified_lines.append(line)

                # If this line ends the table (terminator with semicolon), exit spacing table after processing.
                if line.strip().endswith(";"):
                    in_spacing_table = False
                continue

            # SITE SIZE
            if re.match(r"\s*SIZE\s+[-+]?\d*\.?\d+\s+BY\s+[-+]?\d*\.?\d+\s*;", line):
                line = re.sub(
                    r"(\s*SIZE\s+)([-+]?\d*\.?\d+)\s+BY\s+([-+]?\d*\.?\d+)(\s*;)",
                    lambda m: f"{m.group(1)}{scale_len(m.group(2))} BY {scale_len(m.group(3))}{m.group(4)}",
                    line,
                )

            # MANUFACTURINGGRID
            elif re.match(r"\s*MANUFACTURINGGRID\s+[-+]?\d*\.?\d+\s*;", line):
                line = f"  MANUFACTURINGGRID {self.new_manufacturing_grid:.6f} ;\n"

            # OFFSET (two values)
            elif re.match(r"\s*OFFSET\s+[-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+\s*;", line):
                line = re.sub(
                    r"(\s*OFFSET\s+)([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)(\s*;)",
                    lambda m: f"{m.group(1)}{scale_len(m.group(2))} {scale_len(m.group(3))}{m.group(4)}",
                    line,
                )

            # WIDTH/MINWIDTH/SPACING/PITCH/MINAREA
            elif re.match(r"\s*(WIDTH|MINWIDTH|SPACING|PITCH|MINAREA)\s+[-+]?\d*\.?\d+\s*;", line):
                if "MINAREA" in line:
                    line = re.sub(
                        r"(\s*MINAREA\s+)([-+]?\d*\.?\d+)(\s*;)",
                        lambda m: f"{m.group(1)}{scale_area(m.group(2))}{m.group(3)}",
                        line,
                    )
                else:
                    line = re.sub(
                        r"(\s*(?:WIDTH|MINWIDTH|SPACING|PITCH)\s+)([-+]?\d*\.?\d+)(\s*;)",
                        lambda m: f"{m.group(1)}{scale_len(m.group(2))}{m.group(3)}",
                        line,
                    )

            # CUT SIZE
            elif re.match(r"\s*CUT\s+SIZE\s+[-+]?\d*\.?\d+\s+BY\s+[-+]?\d*\.?\d+\s*;", line):
                line = re.sub(
                    r"(\s*CUT\s+SIZE\s+)([-+]?\d*\.?\d+)\s+BY\s+([-+]?\d*\.?\d+)(\s*;)",
                    lambda m: f"{m.group(1)}{scale_len(m.group(2))} BY {scale_len(m.group(3))}{m.group(4)}",
                    line,
                )

            # ENCLOSURE / OVERHANG
            elif re.match(r"\s*(ENCLOSURE|OVERHANG)\s+[-+]?\d*\.?\d+\s*;", line):
                line = re.sub(
                    r"(\s*(?:ENCLOSURE|OVERHANG)\s+)([-+]?\d*\.?\d+)(\s*;)",
                    lambda m: f"{m.group(1)}{scale_len(m.group(2))}{m.group(3)}",
                    line,
                )

            # RECT coordinates (including those inside VIA/LAYER blocks)
            elif re.search(r"\bRECT\b", line):
                # match RECT followed by four signed/unsigned float coordinates
                # preserve any trailing semicolon/spacing
                line = re.sub(
                    r"(\bRECT\s+)([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)(\s*;?)",
                    lambda m: (
                        f"{m.group(1)}"
                        f"{scale_len(m.group(2))} "
                        f"{scale_len(m.group(3))} "
                        f"{scale_len(m.group(4))} "
                        f"{scale_len(m.group(5))}"
                        f"{m.group(6)}"
                    ),
                    line,
                )

            # SAMENET spacing
            elif "SAMENET" in line:
                line = re.sub(
                    r"(SAMENET\s+\w+\s+\w+\s+)([-+]?\d*\.?\d+)(\s*;)",
                    lambda m: f"{m.group(1)}{scale_len(m.group(2))}{m.group(3)}",
                    line,
                )

            # VIARULE SPACING
            elif "SPACING" in line and "BY" in line and "VIARULE" not in line:
                line = re.sub(
                    r"(SPACING\s+)([-+]?\d*\.?\d+)\s+BY\s+([-+]?\d*\.?\d+)(\s*;)",
                    lambda m: f"{m.group(1)}{scale_len(m.group(2))} BY {scale_len(m.group(3))}{m.group(4)}",
                    line,
                )

            modified_lines.append(line)

        with open(lef_path, "w") as f:
            f.writelines(modified_lines)

        logger.info(
            f"Scaled technology LEF at {lef_path} with alpha={alpha}, "
            f"new manufacturing grid={self.new_manufacturing_grid:.6f}."
        )

    def scale_stdcell_lef(self, alpha: float):
        """
        Scale standard-cell LEF dimensions by dividing linear values by alpha.
        Applies to: MACRO SIZE, ORIGIN, FOREIGN, PIN/OBS RECTs, and any RECT coordinates.
        """

        lef_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign_stdcell.lef")
        logger.info(
            f"Scaling stdcell LEF with alpha={alpha}. "
            f"old_grid={self.original_manufacturing_grid:.6f} → new_grid={self.new_manufacturing_grid:.6f}. "
            f"path={lef_path}"
        )

        if not os.path.exists(lef_path):
            logger.warning(f"Stdcell LEF not found at {lef_path}; nothing to edit.")
            return
        
        try:
            with open(lef_path, "r") as f:
                lines = f.readlines()
            logger.info(f"SCALE STD CELL: Read in {len(lines)} lines from {lef_path}.")
        except Exception as e:
            logger.error(f"Failed to read stdcell LEF {lef_path}: {e}")
            return

        # ----------------- Helper functions -----------------
        def scale_len(val: str) -> str:
            return self.scale_length(alpha, val)

        modified_lines = []

        for line in lines:
            # MACRO SIZE line: e.g., "SIZE 0.95 BY 2.72 ;"
            if re.match(r"^\s*SIZE\s+[\d\.Ee\+\-]+\s+BY\s+[\d\.Ee\+\-]+\s*;", line):
                line = re.sub(
                    r"(^\s*SIZE\s+)([\d\.Ee\+\-]+)\s+BY\s+([\d\.Ee\+\-]+)(\s*;)",
                    lambda m: f"{m.group(1)}{scale_len(m.group(2))} BY {scale_len(m.group(3))}{m.group(4)}",
                    line,
                )

            # ORIGIN line: e.g., "ORIGIN 0.0 0.0 ;"
            elif re.match(r"^\s*ORIGIN\s+[\d\.Ee\+\-]+\s+[\d\.Ee\+\-]+\s*;", line):
                line = re.sub(
                    r"(^\s*ORIGIN\s+)([\d\.Ee\+\-]+)\s+([\d\.Ee\+\-]+)(\s*;)",
                    lambda m: f"{m.group(1)}{scale_len(m.group(2))} {scale_len(m.group(3))}{m.group(4)}",
                    line,
                )

            # FOREIGN line: e.g., "FOREIGN NAND2_X1 0.0 0.0 ;"
            elif re.match(r"^\s*FOREIGN\s+\S+\s+[\d\.Ee\+\-]+\s+[\d\.Ee\+\-]+\s*;", line):
                line = re.sub(
                    r"(^\s*FOREIGN\s+\S+\s+)([\d\.Ee\+\-]+)\s+([\d\.Ee\+\-]+)(\s*;)",
                    lambda m: f"{m.group(1)}{scale_len(m.group(2))} {scale_len(m.group(3))}{m.group(4)}",
                    line,
                )

            # RECT coordinates: e.g., "RECT 0.5 1.2 1.0 1.8"
            elif re.search(r"\bRECT\b", line):
                line = re.sub(
                    r"(RECT\s+)([\d\.Ee\+\-]+)\s+([\d\.Ee\+\-]+)\s+([\d\.Ee\+\-]+)\s+([\d\.Ee\+\-]+)",
                    lambda m: f"{m.group(1)}{scale_len(m.group(2))} {scale_len(m.group(3))} "
                            f"{scale_len(m.group(4))} {scale_len(m.group(5))}",
                    line,
                )
 
            modified_lines.append(line)

        # ----------------- Write updated LEF -----------------
        logger.info(f"SCALE_STD: writing {len(modified_lines)} lines to {lef_path}")
        with open(lef_path, "w") as f:
            f.writelines(modified_lines)
        logger.info("SCALE_STD: write complete")

        logger.info(
            f"Scaled stdcell LEF at {lef_path} with alpha={alpha}, "
            f"new manufacturing grid={self.new_manufacturing_grid:.6f}."
        )

    def get_core_xmin(self):
        """Read the leftmost x-coordinate from the core_area line in codesign_top.tcl."""
        tcl_path = os.path.join(self.directory, "tcl", "codesign_top.tcl")
        with open(tcl_path) as f:
            for line in f:
                if "set core_area" in line:
                    nums = re.findall(r"[\d.]+", line)
                    if len(nums) == 4:
                        return float(nums[0])
        return 0.0

    def scale_pdn_config(self, alpha: float):
        """
        Scale PDN configuration to match the scaled manufacturing grid.
        Ensures PDN widths are multiples of the manufacturing grid.
        """

        pdn_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign.pdn.tcl")
        lef_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign_tech.lef")
        
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

        core_xmin = self.get_core_xmin()
        core_offset = 0.0
        logger.info(f"Recomputing PDN offset from core_xmin={core_xmin:.4f} -> {core_offset:.4f}")

        modified_lines = []
        for line in lines:
            # Scale PDN stripe widths and pitches
            if "-width" in line:
                # Extract and scale width values using width grid (0.0090)
                line = re.sub(r"(-width\s+\{)([\d\.]+)(\})",
                            lambda m: f"{m.group(1)}{self.scale_length(alpha, m.group(2))}{m.group(3)}",
                            line)
            
            if "-pitch" in line:
                # Extract and scale pitch values using manufacturing grid
                line = re.sub(r"(-pitch\s+\{)([\d\.]+)(\})",
                            lambda m: f"{m.group(1)}{self.scale_length(alpha, m.group(2))}{m.group(3)}",
                            line)
            
            if "add_pdn_stripe" in line:
                # Replace existing offset (with or without braces)
                if "-offset" in line:
                    line = re.sub(
                        r"(-offset\s+)(?:\{?[\d\.]+\}?)",
                        lambda m: f"{m.group(1)}{{{core_offset:.4f}}}",
                        line
                    )
                else:
                    # Only insert offset for stripe commandss
                    if "-followpins" in line:
                        line = line.replace("-followpins", f"-offset {{{core_offset:.4f}}} -followpins")
                    else:
                        line = line.strip() + f" -offset {{{core_offset:.4f}}}\n"
            
            if "-halo" in line:
                # Extract and scale halo values using manufacturing grid
                line = re.sub(r"(-halo\s+\{)([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)(\})",
                            lambda m: f"{m.group(1)}{self.scale_length(alpha, m.group(2))} {self.scale_length(alpha, m.group(3))} {self.scale_length(alpha, m.group(4))} {self.scale_length(alpha, m.group(5))}{m.group(6)}",
                            line)
            
            modified_lines.append(line)

        with open(pdn_path, "w") as f:
            f.writelines(modified_lines)
        logger.info(f"Scaled PDN config at {pdn_path} with alpha={alpha}, grid={manufacturing_grid:.6f}.")

    def scale_vars_file(self, alpha: float):
        """
        Scale various parameters in codesign.vars to match the scaled technology.
        """
        # Even when alpha == 1.0 we still want to ensure certain values
        # (e.g. macro_place_halo) are aligned to the manufacturing grid.
        if alpha == 1.0:
            logger.info("Alpha is 1.0, running vars file pass (only grid-aligning macro_place_halo).")
            grid_only_pass = True
        else:
            grid_only_pass = False
        if alpha <= 0:
            logger.error("Alpha must be positive. Skipping vars file scaling.")
            return

        vars_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign.vars")
        
        with open(vars_path, "r") as f:
            lines = f.readlines()

        def scale_val(val_str: str) -> str:
            return self.scale_length(alpha, val_str)

        modified_lines = []
        for line in lines:
            # Scale tapcell distance
            if "-distance" in line:
                if not grid_only_pass:
                    line = re.sub(r"(-distance\s+)([\d\.]+)",
                                lambda m: f"{m.group(1)}{scale_val(m.group(2))}",
                                line)
            
            # Scale macro place halo
            if "macro_place_halo" in line:
                # always grid-align these values regardless of grid_only_pass
                # support optional whitespace and preserve formatting
                line = re.sub(r"(\{)\s*([\d\.]+)\s+([\d\.]+)\s*(\})",
                            lambda m: f"{m.group(1)}{scale_val(m.group(2))} {scale_val(m.group(3))}{m.group(4)}",
                            line)
            
            # Scale macro place channel
            if "macro_place_channel" in line and not grid_only_pass:
                line = re.sub(r"(\{)\s*([\d\.]+)\s+([\d\.]+)\s*(\})",
                            lambda m: f"{m.group(1)}{scale_val(m.group(2))} {scale_val(m.group(3))}{m.group(4)}",
                            line)
                
            # Scale tie separation
            if "tie_separation" in line and not grid_only_pass:
                line = re.sub(r"(\s+)([\d\.]+)$",
                            lambda m: f"{m.group(1)}{scale_val(m.group(2))}",
                            line)
                
            # Scale cts cluster diameter
            if "cts_cluster_diameter" in line and not grid_only_pass:
                line = re.sub(r"(\s+)([\d\.]+)$",
                            lambda m: f"{m.group(1)}{scale_val(m.group(2))}",
                            line)
                
            modified_lines.append(line)

        with open(vars_path, "w") as f:
                f.writelines(modified_lines)
        logger.info(f"Scaled vars file at {vars_path} with alpha={alpha}.")

    def verify_on_grid(self, path: str, tag: str = ""):
        bad = []
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                for s in re.findall(r"[-+]?\d+\.\d+", line):
                    v = float(s)
                    r = v / self.new_manufacturing_grid
                    if abs(r - round(r)) > 1e-6:
                        bad.append((lineno, line.strip(), v))
        if bad:
            logger.warning(f"[GRID] Off-grid values found in {tag or path}:")
            for lineno, text, v in bad[:20]:
                logger.warning(f"  line {lineno}: {v}  →  {v/self.new_manufacturing_grid}")
        else:
            logger.info(f"[GRID] All values are aligned in {tag or path}.")


    
