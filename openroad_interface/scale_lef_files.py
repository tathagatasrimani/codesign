from decimal import ROUND_HALF_UP, Decimal, getcontext
import logging
import os
import re

logger = logging.getLogger(__name__)

from fractions import Fraction


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

        ## original values from FreePDK45
        self.OLD_database_units_per_micron = None  # currently is 2000 in FreePDK45
        self.OLD_manufacturing_grid_dbu = None  # currently is 10 in FreePDK45 (0.005 microns)
        self.OLD_site_size_x_dbu = None
        self.OLD_site_size_y_dbu = None

        ## scaling factors
        self.alpha = None  # scaling factor (L_eff_FreePDK45 / L_eff_current)
        self.database_units_scale = None  ## scale the number of database units per micron by this factor

        ## new values after scaling
        self.NEW_database_units_per_micron = None  # default
        self.NEW_manufacturing_grid_dbu = None  # default
        self.NEW_site_size_x_dbu = None
        self.NEW_site_size_y_dbu = None

    def log_config_vars(self):
        # Log initial state (convert Fraction to float when appropriate)
        def _fmt(v):
            if v is None:
                return "None"
            try:
                return f"{float(v)}" if isinstance(v, Fraction) else str(v)
            except Exception:
                return str(v)

        logger.info(
            "Initial scale state: OLD_database_units_per_micron=%s, OLD_manufacturing_grid_dbu=%s, "
            "OLD_site_size_x_dbu=%s, OLD_site_size_y_dbu=%s, alpha=%s, database_units_scale=%s, "
            "NEW_database_units_per_micron=%s, NEW_manufacturing_grid_dbu=%s, "
            "NEW_site_size_x_dbu=%s, NEW_site_size_y_dbu=%s",
            _fmt(self.OLD_database_units_per_micron),
            _fmt(self.OLD_manufacturing_grid_dbu),
            _fmt(self.OLD_site_size_x_dbu),
            _fmt(self.OLD_site_size_y_dbu),
            _fmt(self.alpha),
            _fmt(self.database_units_scale),
            _fmt(self.NEW_database_units_per_micron),
            _fmt(self.NEW_manufacturing_grid_dbu),
            _fmt(self.NEW_site_size_x_dbu),
            _fmt(self.NEW_site_size_y_dbu),
        )

    def scale_lef_files(self, L_eff_current: float):
        """
        Scales the technology and standard cell LEF files by the given factor.

        :param L_eff_current: The current effective channel length (L_eff) in microns.
        """

        ## read in the original values from the original tech lef file
        self.get_original_dbu_per_micron()
        self.get_original_manufacturing_grid_dbu()
        self.get_original_site_sizes_dbu()
        
        ## NOTE: Alpha is the factor that we are scaling the technology down by vs FreePDK45.
        ## if alpha > 1, we are scaling DOWN (making features smaller)

        L_eff_current_frac = Fraction(L_eff_current)
        L_eff_free_pdk45_frac = Fraction(L_EFF_FREEPDK45)

        self.alpha = L_eff_free_pdk45_frac / L_eff_current_frac
        self.alpha = Fraction(1.0)

        logger.info(
            "Scaling LEF files with alpha = %.4f (L_eff_current = %.4f microns)"
            % (float(self.alpha), float(L_eff_current))
        )

        ## Scale the DBU per micron appropriately to avoid integer overflows in OpenROAD.
        # possible_dbu_per_micron = set([100, 200, 400, 800, 1000, 2000, 4000, 8000, 10000, 20000])

        # ideal_dbu_per_micron = self.OLD_database_units_per_micron * self.alpha

        # logger.info(f"Ideal new database units per micron: {float(ideal_dbu_per_micron)} DBU/micron")

        # ## find the closest possible dbu_per_micron that is >= ideal_dbu_per_micron
        # candidates = [dbu for dbu in possible_dbu_per_micron if dbu >= ideal_dbu_per_micron]

        # self.NEW_database_units_per_micron = Fraction(min(candidates) if candidates else max(possible_dbu_per_micron))
        self.NEW_database_units_per_micron = Fraction(2000) # FORCING 2000 DBU/MICRON FOR NOW TO AVOID ISSUES
        self.database_units_scale = self.NEW_database_units_per_micron / self.OLD_database_units_per_micron

        self.write_DBU(self.NEW_database_units_per_micron)
        self.find_new_manufacturing_grid_in_dbu()
        self.scale_site_sizes()
        self.write_out_site_sizes()

        ## log instance variables for debugging
        self.log_config_vars()

        self.snap_area_to_site_grid()
        self.scale_halo()

        self.scale_tech_lef()
        self.scale_track_pitches()
        self.scale_stdcell_lef()
        self.scale_pdn_config()
        self.scale_vars_file()

        verify = self.verify_on_grid
        
        verify(os.path.join(self.directory, "tcl", "codesign_files", "codesign.tracks"), tag="codesign.tracks")
        verify(os.path.join(self.directory, "tcl", "codesign_files", "codesign_tech.lef"), tag="codesign_tech.lef")
        verify(os.path.join(self.directory, "tcl", "codesign_files", "codesign_stdcell.lef"), tag="codesign_stdcell.lef")
        verify(os.path.join(self.directory, "tcl", "codesign_top.tcl"), tag="codesign_top.tcl")
        verify(os.path.join(self.directory, "tcl", "codesign_files", "codesign.vars"), tag="codesign.vars")
        verify(os.path.join(self.directory, "tcl", "codesign_files", "codesign.pdn.tcl"), tag="codesign.pdn.tcl")


    ###################################################################################################################################################################
    ## Helper methods

    def get_original_dbu_per_micron(self):
        """
        Read the DATABASE MICRONS value from the *original* (unscaled) technology LEF and returns it.
        """
        tech_lef = os.path.join(self.codesign_root_dir, "openroad_interface/tcl/codesign_files", "codesign_tech.lef")
        dbu = ""
        if os.path.exists(tech_lef):
            with open(tech_lef) as f:
                for line in f:
                    if "DATABASE MICRONS" in line:
                        m = re.search(r"DATABASE MICRONS\s+(\d+)", line)
                        if m:
                            dbu = m.group(1)
                            break

        dbu = int(dbu)

        logger.info(f"Original database units per micron from {tech_lef} read in as: {dbu} DBU/micron")
        self.OLD_database_units_per_micron = Fraction(dbu)

    def get_original_manufacturing_grid_dbu(self):
        """
        Read the MANUFACTURINGGRID value from the *original* (unscaled) technology LEF and returns it in DBU.
        """
        tech_lef = os.path.join(self.codesign_root_dir, "openroad_interface/tcl/codesign_files", "codesign_tech.lef")
        grid = ""
        if os.path.exists(tech_lef):
            with open(tech_lef) as f:
                for line in f:
                    if "MANUFACTURINGGRID" in line:
                        m = re.search(r"MANUFACTURINGGRID\s+([\d\.]+)", line)
                        if m:
                            grid = m.group(1)
                            break

        manufacturing_grid_dbu = round(Fraction(grid)*self.OLD_database_units_per_micron)
        logger.info(f"Original manufacturing grid from {tech_lef} read in as: {grid} microns ({manufacturing_grid_dbu} DBU)")
        self.OLD_manufacturing_grid_dbu = Fraction(manufacturing_grid_dbu)

    def get_original_site_sizes_dbu(self):
        """
        Reads the SITE SIZE values from the *original* (unscaled) technology LEF
        and returns them in DBU (integer, no float math).
        Example LEF block:
            SITE codesign_site
            SYMMETRY y ;
            CLASS core ;
            SIZE 1.216 BY 8.96 ;
            END codesign_site
        """
        tech_lef = os.path.join(
            self.codesign_root_dir,
            "openroad_interface/tcl/codesign_files",
            "codesign_tech.lef"
        )

        site_x = ""
        site_y = ""

        if os.path.exists(tech_lef):
            with open(tech_lef) as f:
                inside_site = False
                for line in f:
                    # Detect start of SITE block
                    if re.match(r"^\s*SITE\s+\S+", line):
                        inside_site = True
                        continue

                    # Look for SIZE line once inside SITE block
                    if inside_site and "SIZE" in line:
                        m = re.search(r"SIZE\s+([\d\.]+)\s+BY\s+([\d\.]+)", line)
                        if m:
                            site_x = m.group(1)
                            site_y = m.group(2)
                            break  # done, we found our site size

                    # Detect end of SITE block
                    if inside_site and line.strip().startswith("END"):
                        inside_site = False

        # Convert safely to integer DBU (no floats)
        if site_x and site_y:
            site_x_dbu = int(Fraction(site_x) * self.OLD_database_units_per_micron + Fraction(1, 2))
            site_y_dbu = int(Fraction(site_y) * self.OLD_database_units_per_micron + Fraction(1, 2))
        else:
            site_x_dbu = site_y_dbu = 0

        logger.info(
            f"Original site size from {tech_lef} read as: "
            f"{site_x} x {site_y} microns "
            f"({site_x_dbu} x {site_y_dbu} DBU)"
        )

        # Store as Fractions for downstream integer-safe math
        self.OLD_site_size_x_dbu = Fraction(site_x_dbu)
        self.OLD_site_size_y_dbu = Fraction(site_y_dbu)

    def find_new_manufacturing_grid_in_dbu(self):
        """
        Computes the new manufacturing grid after scaling by alpha and returns it in DBU.
        """
        
        new_grid_dbus = round((self.OLD_manufacturing_grid_dbu / self.alpha) * self.database_units_scale)
        self.NEW_manufacturing_grid_dbu = Fraction(new_grid_dbus)
        logger.info(f"New manufacturing grid after scaling: {float(self.NEW_manufacturing_grid_dbu)} DBU.")


    def scale_site_sizes(self):
        """
        Computes the new site sizes after scaling by alpha. The input sizes are in DBU, and the output sizes are in DBU.
        """
        new_site_x_dbus = round((self.OLD_site_size_x_dbu / self.alpha) * self.database_units_scale)
        
        self.NEW_site_size_x_dbu = self.round_to_manufacturing_grid(Fraction(new_site_x_dbus))
        logger.info(f"New site size X after scaling: {float(self.NEW_site_size_x_dbu)} DBU.")

        new_site_y_dbus = round((self.OLD_site_size_y_dbu / self.alpha) * self.database_units_scale)
        self.NEW_site_size_y_dbu = self.round_to_manufacturing_grid(Fraction(new_site_y_dbus))
        logger.info(f"New site size Y after scaling: {float(self.NEW_site_size_y_dbu)} DBU.")

    def write_out_site_sizes(self):
        """ Writes out the new site sizes to the tech LEF file.
        """
        ## open the tech lef file
        lef_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign_tech.lef")
        with open(lef_path, "r") as f:
            lines = f.readlines()
        
        modified_lines = []
        for line in lines:
            if "SIZE" in line:
                line = re.sub(r"(SIZE\s+)([\d\.]+)(\s+BY\s+)([\d\.]+)(\s*;)", lambda m: f"{m.group(1)}{self.dbu_to_lef_text(int(self.NEW_site_size_x_dbu))}{m.group(3)}{self.dbu_to_lef_text(int(self.NEW_site_size_y_dbu))}{m.group(5)}", line)
            modified_lines.append(line)

        with open(lef_path, "w") as f:
            f.writelines(modified_lines)


    def dbu_to_lef_text(self, dbu_val: int, precision: int = 6) -> str:
        """Convert integer DBU back to decimal string safely."""
        # Defensive context
        getcontext().prec = precision + 10  # extra precision margin
        getcontext().rounding = ROUND_HALF_UP

        # Convert inputs safely
        dbu_val_decimal = Decimal(str(dbu_val))
        dbu_per_micron_decimal = Decimal(str(self.NEW_database_units_per_micron))

        # Avoid division-by-zero and invalid quantization
        if dbu_per_micron_decimal == 0:
            raise ZeroDivisionError("NEW_database_units_per_micron cannot be zero")

        val = dbu_val_decimal / dbu_per_micron_decimal

        # Clamp to requested precision, but avoid InvalidOperation by pre-rounding
        quantizer = Decimal(10) ** (-precision)
        val = val.quantize(quantizer, rounding=ROUND_HALF_UP)

        # Convert to normalized string
        text = format(val.normalize(), 'f')
        return text
    
    def scale_length(self, alpha: Fraction, val: str) -> str:
        """ Scale a length value by dividing by alpha and snapping to the new manufacturing grid. 
        :param alpha: The scaling factor. A Fraction.
        :param val: string value in microns.

        :return: string value in microns after scaling and snapping to grid.
        """
        val_in_dbu = round(Fraction(val) * self.OLD_database_units_per_micron) * self.database_units_scale
        scaled_val = Fraction(val_in_dbu) / alpha
        
        rounded_val_in_dbu = self.round_to_manufacturing_grid(scaled_val)

        final_string = self.dbu_to_lef_text(int(rounded_val_in_dbu))

        ## express the fraction as a precise decimal string
        return f"{final_string}"
    
    def scale_length_site_snap(self, alpha: Fraction, val: str, dir: str) -> str:
        """ Scale a length value by dividing by alpha and snapping to the new site size grid. 
        :param alpha: The scaling factor. A Fraction.
        :param val: string value in microns.
        :param dir: 'x' or 'y' for which site dimension to snap to.

        :return: string value in microns after scaling and snapping to grid.
        """
        val_in_dbu = round(Fraction(val) * self.OLD_database_units_per_micron) * self.database_units_scale
        scaled_val = Fraction(val_in_dbu) / alpha
        
        if dir == 'x':
            snapped_val_in_dbu = max(1, round(scaled_val / self.NEW_site_size_x_dbu)) * self.NEW_site_size_x_dbu
        elif dir == 'y':
            snapped_val_in_dbu = max(1, round(scaled_val / self.NEW_site_size_y_dbu)) * self.NEW_site_size_y_dbu
        else:
            raise ValueError("dir must be 'x' or 'y'")

        final_string = self.dbu_to_lef_text(int(snapped_val_in_dbu))

        ## express the fraction as a precise decimal string
        return f"{final_string}"
    
    def scale_area(self, alpha: float, val: str) -> str:
        """ Scale an area value by dividing by alpha^2 and snapping to the new manufacturing grid^2.
        """
        val_in_dbu = round(Fraction(val) * self.OLD_database_units_per_micron) * self.database_units_scale
        scaled_val = Fraction(val_in_dbu) / (alpha * alpha)

        rounded_val_in_dbu = self.round_to_manufacturing_grid_area(scaled_val)

        final_string = self.dbu_to_lef_text(int(rounded_val_in_dbu))

        ## express the fraction as a precise decimal string
        return f"{final_string}"

    def round_to_manufacturing_grid(self, val_in_dbu: Fraction) -> Fraction:
        """
        Round a value in DBU to the nearest multiple of new manufacturing grid.
        """
        return round(val_in_dbu / self.NEW_manufacturing_grid_dbu) * self.NEW_manufacturing_grid_dbu

    def round_to_manufacturing_grid_area(self, val_in_dbu: Fraction) -> Fraction:
        """
        Round an area value to the nearest multiple of new manufacturing grid^2.
        """
        return round(val_in_dbu / (self.NEW_manufacturing_grid_dbu * self.NEW_manufacturing_grid_dbu)) * (self.NEW_manufacturing_grid_dbu * self.NEW_manufacturing_grid_dbu)
    
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

    def snap_area_to_site_grid(self):
        """ Snap the overall area constraint to the nearest multiple of the site grid.
        """
        ## read in the top tcl file
        tcl_path = os.path.join(self.directory, "tcl", "codesign_top.tcl")
        with open(tcl_path, "r") as f:
            lines = f.readlines()   

        ## find the area. Snap it to the nearest multiple of the site grid.
        modified_lines = []
        
        grid_x = self.NEW_site_size_x_dbu
        grid_y = self.NEW_site_size_y_dbu

        def snap_value(val: Fraction, grid: Fraction) -> Fraction:
            new_DBU_snapped_to_grid = round((val*self.NEW_database_units_per_micron) / grid) * grid
            return new_DBU_snapped_to_grid

        def fmt_num(v: Fraction) -> str:
            return self.dbu_to_lef_text(int(v))
        for line in lines:
            # match lines like: set die_area {0 0 2800 2800}
            if line.strip().startswith("set die_area"):
                nums = re.findall(r"-?\d+\.?\d*", line)
                if len(nums) == 4:
                    x0, y0, x1, y1 = map(Fraction, nums)
                    new_x0 = snap_value(x0, grid_x)
                    new_y0 = snap_value(y0, grid_y)
                    new_x1 = snap_value(x1, grid_x)
                    new_y1 = snap_value(y1, grid_y)
                    new_block = "{" + f"{fmt_num(new_x0)} {fmt_num(new_y0)} {fmt_num(new_x1)} {fmt_num(new_y1)}" + "}"
                    line = re.sub(r"\{\s*-?\d+\.?\d*\s+-?\d+\.?\d*\s+-?\d+\.?\d*\s+-?\d+\.?\d*\s*\}", new_block, line)

            # match lines like: set core_area {50 50 2750 2750}
            if line.strip().startswith("set core_area"):
                nums = re.findall(r"-?\d+\.?\d*", line)
                if len(nums) == 4:
                    lx, ly, x1, y1 = map(Fraction, nums)
                    new_lx = max(Fraction(grid_x), snap_value(lx, grid_x))
                    new_ly = max(Fraction(grid_y), snap_value(ly, grid_y))
                    new_x1 = snap_value(x1, grid_x)
                    new_y1 = snap_value(y1, grid_y)
                    new_block = "{" + f"{fmt_num(new_lx)} {fmt_num(new_ly)} {fmt_num(new_x1)} {fmt_num(new_y1)}" + "}"
                    line = re.sub(r"\{\s*-?\d+\.?\d*\s+-?\d+\.?\d*\s+-?\d+\.?\d*\s+-?\d+\.?\d*\s*\}", new_block, line)

            modified_lines.append(line)

        # write back the modified top tcl
        with open(tcl_path, "w") as f:
            f.writelines(modified_lines)

    def scale_tech_lef(self):
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
            f"Scaling tech LEF with alpha={float(self.alpha)}. "
            f"old_grid={float(self.OLD_manufacturing_grid_dbu):.6f} -> new_grid={float(self.NEW_manufacturing_grid_dbu):.6f}"
        )

        with open(lef_path, "r") as f:
            lines = f.readlines()

        # ---- Helpers ----
        def scale_len(val: str) -> str:
            return self.scale_length(self.alpha, val)

        def scale_area(val: str) -> str:
            return self.scale_area(self.alpha, val)

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

            # MANUFACTURINGGRID
            elif re.match(r"\s*MANUFACTURINGGRID\s+[-+]?\d*\.?\d+\s*;", line):
                line = f"  MANUFACTURINGGRID {self.dbu_to_lef_text(self.NEW_manufacturing_grid_dbu)} ;\n"

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
                # match RECT followed by four signed/unsigned decimal coordinates
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
            f"Scaled technology LEF at {lef_path} with alpha={float(self.alpha)}, "
            f"new manufacturing grid={float(self.NEW_manufacturing_grid_dbu):.6f}."
        )

    def scale_track_pitches(self):
        """
        Updates the track pitches in the temporary codesign.tracks file.
        This method reads the file, scales the x and y pitch values by dividing
        them by the alpha factor, and writes the file back.
        """

        
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

            new_value = self.scale_length(self.alpha, value_str)

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

    def scale_stdcell_lef(self):
        """
        Scale standard-cell LEF dimensions by dividing linear values by alpha.
        Applies to: MACRO SIZE, ORIGIN, FOREIGN, PIN/OBS RECTs, and any RECT coordinates.
        """

        lef_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign_stdcell.lef")
        logger.info(
            f"Scaling stdcell LEF with alpha={self.alpha}. "
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
            return self.scale_length(self.alpha, val)

        def scale_len_site_snap(val: str, dir: str) -> str:
            return self.scale_length_site_snap(self.alpha, val, dir)

        modified_lines = []

        for line in lines:
            # MACRO SIZE line: e.g., "SIZE 0.95 BY 2.72 ;"
            if re.match(r"^\s*SIZE\s+[\d\.Ee\+\-]+\s+BY\s+[\d\.Ee\+\-]+\s*;", line):
                line = re.sub(
                    r"(^\s*SIZE\s+)([\d\.Ee\+\-]+)\s+BY\s+([\d\.Ee\+\-]+)(\s*;)",
                    lambda m: f"{m.group(1)}{scale_len_site_snap(m.group(2), 'x')} BY {scale_len_site_snap(m.group(3), 'y')}{m.group(4)}",
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
            f"Scaled stdcell LEF at {lef_path} with alpha={self.alpha}"
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
        raise ValueError("core_area not found or malformed in codesign_top.tcl")

    def scale_pdn_config(self):
        """
        Scale PDN configuration to match the scaled manufacturing grid.
        Ensures PDN widths are multiples of the manufacturing grid.
        """

        pdn_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign.pdn.tcl")
        
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
                            lambda m: f"{m.group(1)}{self.scale_length(self.alpha, m.group(2))}{m.group(3)}",
                            line)
            
            if "-pitch" in line:
                # Extract and scale pitch values using manufacturing grid
                line = re.sub(r"(-pitch\s+\{)([\d\.]+)(\})",
                            lambda m: f"{m.group(1)}{self.scale_length(self.alpha, m.group(2))}{m.group(3)}",
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
                            lambda m: f"{m.group(1)}{self.scale_length(self.alpha, m.group(2))} {self.scale_length(self.alpha, m.group(3))} {self.scale_length(self.alpha, m.group(4))} {self.scale_length(self.alpha, m.group(5))}{m.group(6)}",
                            line)
            
            modified_lines.append(line)

        with open(pdn_path, "w") as f:
            f.writelines(modified_lines)

    def scale_vars_file(self):
        """
        Scale various parameters in codesign.vars to match the scaled technology.
        """

        vars_path = os.path.join(self.directory, "tcl", "codesign_files", "codesign.vars")
        
        with open(vars_path, "r") as f:
            lines = f.readlines()

        def scale_val(val_str: str) -> str:
            return self.scale_length(self.alpha, val_str)
        
        modified_lines = []
        for line in lines:
            # Scale tapcell distance
            if "-distance" in line:
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
            if "macro_place_channel" in line:
                line = re.sub(r"(\{)\s*([\d\.]+)\s+([\d\.]+)\s*(\})",
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

    def scale_halo(self):
        """
        Scale the halo value in codesign_flow.tcl by dividing it by alpha and snapping to grid.
        Targets lines containing:
            -halo_width  <value> \
            -halo_height <value> \
        """
        flow_path = os.path.join(self.directory, "tcl", "codesign_flow.tcl")
        if not os.path.exists(flow_path):
            logger.warning(f"codesign_flow.tcl not found at {flow_path}; skipping halo scaling.")
            return

        with open(flow_path, "r") as f:
            lines = f.readlines()

        def repl_width(m):
            return f"{m.group(1)}{self.scale_length(self.alpha, m.group(2))}"

        def repl_height(m):
            return f"{m.group(1)}{self.scale_length(self.alpha, m.group(2))}"

        modified_lines = []
        for line in lines:
            if "-halo_width" in line:
                line = re.sub(r"(-halo_width\s+)([-+]?\d*\.?\d+)", repl_width, line)
            if "-halo_height" in line:
                line = re.sub(r"(-halo_height\s+)([-+]?\d*\.?\d+)", repl_height, line)
            modified_lines.append(line)

        with open(flow_path, "w") as f:
            f.writelines(modified_lines)

    def verify_on_grid(self, path: str, tag: str = ""):
        """
        Verify numeric values in file are exact multiples of the new manufacturing grid.
        Uses Fraction/Decimal arithmetic exclusively (no float fallback).
        Requires self.NEW_manufacturing_grid_dbu and self.NEW_database_units_per_micron to be set (Fractions).
        """
        from decimal import Decimal, InvalidOperation

        if not hasattr(self, "NEW_manufacturing_grid_dbu") or not hasattr(self, "NEW_database_units_per_micron"):
            logger.error("verify_on_grid requires NEW_manufacturing_grid_dbu and NEW_database_units_per_micron (Fractions) to be set.")
            return

        grid_frac = self.NEW_manufacturing_grid_dbu / self.NEW_database_units_per_micron  # Fraction (microns)

        bad = []
        with open(path, "r") as f:
            for lineno, line in enumerate(f, 1):
                tokens = re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", line)
                for s in tokens:
                    try:
                        # Decimal -> Fraction to handle scientific notation and exact decimals
                        v_frac = Fraction(Decimal(s))
                    except (InvalidOperation, ValueError):
                        # skip tokens that cannot be parsed precisely
                        continue

                    try:
                        r = v_frac / grid_frac  # Fraction
                    except Exception:
                        bad.append((lineno, line.strip(), v_frac, None))
                        continue

                    if r.denominator != 1:
                        bad.append((lineno, line.strip(), v_frac, r))

        if bad:
            logger.warning(f"[GRID] Off-grid values found in {tag or path}:")
            for lineno, text, v_frac, ratio in bad[:200]:
                ratio_str = str(ratio) if isinstance(ratio, Fraction) else "N/A"
                logger.warning(f"  line {lineno}: value={v_frac}  →  {ratio_str} grid units; text: {text}")
        else:
            logger.info(f"[GRID] All values are aligned in {tag or path}.")



