import os
import subprocess
import yaml
import traceback
import argparse
import re
import logging

logger = logging.getLogger(__name__)

import pandas as pd

from src.cacti.cacti_python.parameter import g_ip
from src.cacti.cacti_python.parameter import g_tp
from src.cacti.cacti_python.cacti_interface import uca_org_t
from src.cacti.cacti_python.Ucache import *
from src.cacti.cacti_python.parameter import sympy_var

from hw_symbols import *
import sympy as sp

valid_tech_nodes = [0.022, 0.032, 0.045, 0.065, 0.090, 0.180]


def gen_symbolic(name, cache_cfg, opt_vals, use_piecewise=False):
    """
    Generates SymPy expressions for access time and energy, outputting results to text files.

    Inputs:
    name : str
        Base name for the output files.
    cache_cfg : str
        Path to the cache configuration file. Relative to the Cacti directory.
    opt_vals : dict
        Optimization values with keys: "ndwl", "ndbl", "nspd", "ndcm", "ndsam1", "ndsam2",
        "repeater_spacing", "repeater_size".
    use_piecewise : bool, optional
        Flag for using piecewise functions (default is True).

    Returns:
    IO_info : dict
        Dictionary containing IO-related data (e.g., area, power, margin).

    Outputs:
    Sympy expression files for access time, dynamic and leakage power, IO details in 'src/cacti/sympy' directory.
    """

    g_ip.parse_cfg(os.path.join(CACTI_DIR, cache_cfg))
    g_ip.error_checking()

    g_ip.ndwl = opt_vals["ndwl"]
    g_ip.ndbl = opt_vals["ndbl"]
    g_ip.nspd = opt_vals["nspd"]
    g_ip.ndcm = opt_vals["ndcm"]

    g_ip.ndsam1 = opt_vals["ndsam1"]
    g_ip.ndsam2 = opt_vals["ndsam2"]

    g_ip.repeater_spacing = opt_vals["repeater_spacing"]
    g_ip.repeater_size = opt_vals["repeater_size"]

    g_ip.use_piecewise = use_piecewise
    g_ip.print_detail_debug = False

    fin_res = uca_org_t()
    fin_res = solve_single()

    # Create the directory path
    output_dir = os.path.join(CACTI_DIR, "symbolic_expressions")
    os.makedirs(output_dir, exist_ok=True)

    # Write files to the cacti/sympy directory
    with open(os.path.join(output_dir, f'{name + "_access_time"}.txt'), "w") as file:
        file.write(str(fin_res.access_time))

    with open(os.path.join(output_dir, f'{name + "_read_dynamic"}.txt'), "w") as file:
        file.write(str(fin_res.power.readOp.dynamic))

    with open(os.path.join(output_dir, f'{name + "_write_dynamic"}.txt'), "w") as file:
        file.write(str(fin_res.power.writeOp.dynamic))

    # TODO: why is this read leakage? Should be standby leakage. What does read leakage even mean?
    with open(os.path.join(output_dir, f'{name + "_read_leakage"}.txt'), "w") as file:
        file.write(str(fin_res.power.readOp.leakage))

    IO_info = {
        "io_area": fin_res.io_area,
        "io_timing_margin": fin_res.io_timing_margin,
        "io_dynamic_power": fin_res.io_dynamic_power,
        "io_phy_power": fin_res.io_phy_power,
        "io_termination_power": fin_res.io_termination_power,
    }

    for key, value in IO_info.items():
        file_name = f'{name + "_" + key}.txt'
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "w") as file:
            file.write(str(value))

    return IO_info


def gen_vals(
    filename="base_cache",
    cache_size=None,
    blockSize=None,
    cache_type=None,
    bus_width=None,
    transistor_size=None,
    addr_timing=None,
    force_cache_config=None,
    technology=None,
    debug=False,
) -> pd.DataFrame:
    """
    Generates a Cacti .cfg file based on input and cacti_input, runs Cacti,
    and retrieves timing and power values.

    TODO: Change to just call `run_existing_cacti_cfg`

    Inputs:
    filename : str, optional
        Base name for the generated .cfg file (default is "base_cache").
    cache_size : int, optional
        Size of the cache in bytes.
    blockSize : int, optional
        Size of each cache block in bytes.
    cache_type : str, optional
        Type of cache (e.g., "cache" or "main memory").
    bus_width : int, optional
        Width of the input/output bus in bits.
    transistor_size : float, optional (in um)
        Size of the transistor technology node (e.g., 0.045).
    addr_timing : float, optional
        Address timing value for memory access.
    force_cache_config : str, optional
        Force specific cache configuration settings.
    technology : str, optional
        Technology parameter to override defaults.
    debug : bool, optional
        Enables debug logging (default is False).

    Returns:
    pd.DataFrame
        DataFrame containing timing, power, and IO-related values from the Cacti run.
            # CACTI: access time (ns), search energy (nJ), read energy (nJ), write energy (nJ), leakage bank power (mW)
            # CACTI IO: area (sq.mm), timing (ps), dynamic power (mW), PHY power (mW), termination and bias power (mW)
            # latency (ns)

    Outputs:
    A .cfg file is generated for Cacti, and Cacti run results are
    returned in a DataFrame.
    """

    logger.info(
        f"Running Cacti with the following parameters: filename: {filename}, cache_size: {cache_size}, blockSize: {blockSize}, cache_type: {cache_type}, bus_width: {bus_width}, transistor_size: {transistor_size}, addr_timing: {addr_timing}, force_cache_config: {force_cache_config}, technology: {technology}"
    )

    # load in default values
    config_values = read_config_file(os.path.join(CACTI_DIR, f"cfg/{filename}.cfg"))

    # If user doesn't give input, default to cacti_input vals
    if cache_size == None:
        cache_size = config_values["cache_size"]

    if blockSize == None:
        blockSize = config_values["block_size"]

    if cache_type == None:
        cache_type = config_values["cache_type"]

    if cache_type == "cache":
        associativity = 0
        num_search_ports = 1
    else:
        associativity = config_values["assoc"]
        num_search_ports = config_values["num_search_ports"]

    if bus_width == None:
        bus_width = config_values["bus_width"]

    if cache_type == "main memory":
        mem_bus_width = bus_width
    else:
        mem_bus_width = config_values["mem_data_width"]

    if transistor_size == None:
        transistor_size = config_values["F_sz_um"]  # TODO check whether nm or um
    else:
        transistor_size = min(valid_tech_nodes, key=lambda x: abs(transistor_size - x))

    if addr_timing == None:
        addr_timing = config_values["addr_timing"]

    if force_cache_config == None:
        force_cache_config = config_values["force_cache_config"]

    # lines written to [filename].cfg file
    cfg_lines = [
        "# Cache size",
        "-size (bytes) {}".format(cache_size),
        "",
        "# power gating",
        '-Array Power Gating - "{}"'.format(config_values["array_power_gated"]),
        '-WL Power Gating - "{}"'.format(config_values["wl_power_gated"]),
        '-CL Power Gating - "{}"'.format(config_values["cl_power_gated"]),
        '-Bitline floating - "{}"'.format(config_values["bitline_floating"]),
        '-Interconnect Power Gating - "{}"'.format(
            config_values["interconnect_power_gated"]
        ),
        '-Power Gating Performance Loss "{}"'.format(config_values["perfloss"]),
        "",
        "# Line size",
        "-block size (bytes) {}".format(blockSize),
        "",
        "# To model Fully Associative cache, set associativity to zero",
        "-associativity {}".format(associativity),
        "",
        "-read-write port {}".format(config_values["num_rw_ports"]),
        "-exclusive read port {}".format(config_values["num_rd_ports"]),
        "-exclusive write port {}".format(config_values["num_wr_ports"]),
        "-single ended read ports {}".format(config_values["num_se_rd_ports"]),
        "-search port {}".format(num_search_ports),
        "",
        "# Multiple banks connected using a bus",
        "-UCA bank count {}".format(config_values["nbanks"]),
        "-technology (u) {}".format(transistor_size),
        "",
        "# following three parameters are meaningful only for main memories",
        "-page size (bits) {}".format(config_values["page_sz_bits"]),
        "-burst length {}".format(config_values["burst_len"]),
        "-internal prefetch width {}".format(config_values["int_prefetch_w"]),
        "",
        "# following parameter can have one of five values",
        '-Data array cell type - "{}"'.format(config_values["data_array_cell_type"]),
        "",
        "# following parameter can have one of three values",
        '-Data array peripheral type - "{}"'.format(
            config_values["data_array_peri_type"]
        ),
        "",
        "# following parameter can have one of five values",
        '-Tag array cell type - "{}"'.format(config_values["tag_array_cell_type"]),
        "",
        "# following parameter can have one of three values",
        '-Tag array peripheral type - "{}"'.format(
            config_values["tag_array_peri_type"]
        ),
        "",
        "# Bus width include data bits and address bits required by the decoder",
        "-output/input bus width {}".format(bus_width),
        "",
        "# 300-400 in steps of 10",
        "-operating temperature (K) {}".format(config_values["temp"]),
        "",
        "# Type of memory",
        '-cache type "{}"'.format(cache_type),
        "",
        "# to model special structure like branch target buffers, directory, etc.",
        "# change the tag size parameter",
        '# if you want cacti to calculate the tagbits, set the tag size to "default"',
        '-tag size (b) "{}"'.format(config_values["tag_size"]),
        "",
        "# fast - data and tag access happen in parallel",
        "# sequential - data array is accessed after accessing the tag array",
        "# normal - data array lookup and tag access happen in parallel",
        "#          final data block is broadcasted in data array h-tree",
        "#          after getting the signal from the tag array",
        '-access mode (normal, sequential, fast) - "{}"'.format(
            "fast"
            if config_values["access_mode"] == 2
            else "sequential" if config_values["access_mode"] == 1 else "normal"
        ),
        "",
        "# DESIGN OBJECTIVE for UCA (or banks in NUCA)",
        f"-design objective (weight delay, dynamic power, leakage power, cycle time, area) {config_values['delay_wt']}:{config_values['dynamic_power_wt']}:{config_values['leakage_power_wt']}:{config_values['cycle_time_wt']}:{config_values['area_wt']}",
        "",
        "# Percentage deviation from the minimum value",
        f"-deviate (delay, dynamic power, leakage power, cycle time, area) {config_values['delay_dev']}:{config_values['dynamic_power_dev']}:{config_values['leakage_power_dev']}:{config_values['cycle_time_dev']}:{config_values['area_dev']}"
        "",
        "# Objective for NUCA",
        f"-NUCAdesign objective (weight delay, dynamic power, leakage power, cycle time, area) {config_values['delay_wt_nuca']}:{config_values['dynamic_power_wt_nuca']}:{config_values['leakage_power_wt_nuca']}:{config_values['cycle_time_wt_nuca']}:{config_values['area_wt_nuca']}",
        f"-NUCAdeviate (delay, dynamic power, leakage power, cycle time, area) {config_values['delay_dev_nuca']}:{config_values['dynamic_power_dev_nuca']}:{config_values['leakage_power_dev_nuca']}:{config_values['cycle_time_dev_nuca']}:{config_values['area_dev_nuca']}",
        "",
        "# Set optimize tag to ED or ED^2 to obtain a cache configuration optimized for",
        "# energy-delay or energy-delay sq. product",
        "# Note: Optimize tag will disable weight or deviate values mentioned above",
        "# Set it to NONE to let weight and deviate values determine the",
        "# appropriate cache configuration",
        "-Optimize ED or ED^2 (ED, ED^2, NONE): {}".format(
            '"ED^2"'
            if config_values["ed"] == 2
            else '"ED"' if config_values["ed"] == 1 else '"NONE"'
        ),
        "-Cache model (NUCA, UCA)  - {}".format(
            '"UCA"' if config_values["nuca"] == 0 else '"NUCA"'
        ),
        "",
        "# In order for CACTI to find the optimal NUCA bank value the following",
        "# variable should be assigned 0.",
        "-NUCA bank count {}".format(config_values["nuca_bank_count"]),
        "",
        "# Wire signaling",
        '-Wire signaling (fullswing, lowswing, default) - "{}"'.format(
            "default" if config_values["force_wiretype"] == 0 else config_values["wt"]
        ),
        '-Wire inside mat - "{}"'.format(
            "global"
            if config_values["wire_is_mat_type"] == 2
            else "semi-global" if config_values["wire_is_mat_type"] == 1 else "local"
        ),
        '-Wire outside mat - "{}"'.format(
            "global" if config_values["wire_os_mat_type"] == 2 else "semi-global"
        ),
        '-Interconnect projection - "{}"'.format(
            "aggressive" if config_values["ic_proj_type"] == 0 else "conservative"
        ),
        "",
        "# Contention in network",
        "-Core count {}".format(config_values["cores"]),
        '-Cache level (L2/L3) - "{}"'.format(
            "L2" if config_values["cache_level"] == 0 else "L3"
        ),
        '-Add ECC - "{}"'.format(str(config_values["add_ecc_b_"]).lower()),
        '-Print level (DETAILED, CONCISE) - "{}"'.format(
            "DETAILED" if config_values["print_detail"] == 1 else "CONCISE"
        ),
        "",
        "# for debugging",
        '-Print input parameters - "{}"'.format(
            str(config_values["print_input_args"]).lower()
        ),
        "# force CACTI to model the cache with the",
        "# following Ndbl, Ndwl, Nspd, Ndsam,",
        "# and Ndcm values",
        '-Force cache config - "{}"'.format(str(force_cache_config).lower()),
        "-Ndwl {}".format(config_values["ndwl"]),
        "-Ndbl {}".format(config_values["ndbl"]),
        "-Nspd {}".format(config_values["nspd"]),
        "-Ndcm {}".format(config_values["ndcm"]),
        "-Ndsam1 {}".format(config_values["ndsam1"]),
        "-Ndsam2 {}".format(config_values["ndsam2"]),
        "",
        "#### Default CONFIGURATION values for baseline external IO parameters to DRAM. More details can be found in the CACTI-IO technical report (), especially Chapters 2 and 3.",
        "# Memory Type",
        '-dram_type "{}"'.format(config_values["io_type"]),
        "# Memory State",
        '-io state "{}"'.format(config_values["io_state"]),
        "# Address bus timing",
        "-addr_timing {}".format(addr_timing),
        "# Memory Density",
        "-mem_density {}".format(config_values["mem_density"]),
        "# IO frequency",
        "-bus_freq {} MHz".format(int(config_values["bus_freq"])),
        "# Duty Cycle",
        "-duty_cycle {}".format(config_values["duty_cycle"]),
        "# Activity factor for Data",
        "-activity_dq {}".format(config_values["activity_dq"]),
        "# Activity factor for Control/Address",
        "-activity_ca {}".format(config_values["activity_ca"]),
        "# Number of DQ pins",
        "-num_dq {}".format(config_values["num_dq"]),
        "# Number of DQS pins",
        "-num_dqs {}".format(config_values["num_dqs"]),
        "# Number of CA pins",
        "-num_ca {}".format(config_values["num_ca"]),
        "# Number of CLK pins",
        "-num_clk {}".format(config_values["num_clk"]),
        "# Number of Physical Ranks",
        "-num_mem_dq {}".format(config_values["num_mem_dq"]),
        "# Width of the Memory Data Bus",
        "-mem_data_width {}".format(mem_bus_width),
        "# RTT Termination Resistance",
        "-rtt_value {}".format(config_values["rtt_value"]),
        "# RON Termination Resistance",
        "-ron_value {}".format(config_values["ron_value"]),
        "# Time of flight for DQ",
        "# tflight_value",  # Fill
        "# Parameter related to MemCAD",
        "# Number of BoBs",
        "-num_bobs {}".format(config_values["num_bobs"]),
        "# Memory System Capacity in GB",
        "-capacity {}".format(config_values["capacity"]),
        "# Number of Channel per BoB",
        "-num_channels_per_bob {}".format(config_values["num_channels_per_bob"]),
        "# First Metric for ordering different design points",
        '-first metric "{}"'.format(config_values["first_metric"]),
        "# Second Metric for ordering different design points",
        '-second metric "{}"'.format(config_values["second_metric"]),
        "# Third Metric for ordering different design points",
        '-third metric "{}"'.format(config_values["third_metric"]),
        "# Possible DIMM option to consider",
        '-DIMM model "{}"'.format(config_values["dimm_model"]),
        "# If channels of each bob have the same configurations",
        '-mirror_in_bob "{}"'.format(config_values["mirror_in_bob"]),
        "# if we want to see all channels/bobs/memory configurations explored",
        '# -verbose "T"',
        '# -verbose "F"',
    ]

    # write file
    input_filename = f"cfg/{filename}.cfg"
    cactiInput = os.path.join(CACTI_DIR, input_filename)
    with open(cactiInput, "w") as file:
        for line in cfg_lines:
            file.write(line + "\n")

    stdout_filename = "cacti_stdout.log"
    stdout_file_path = os.path.join(CACTI_DIR, stdout_filename)

    cmd = ["./cacti", "-infile", input_filename]

    with open(stdout_file_path, "w") as f:
        p = subprocess.Popen(cmd, cwd=CACTI_DIR, stdout=f, stderr=subprocess.PIPE)

    p.wait()
    if p.returncode != 0:
        with open(stdout_file_path, "r") as f:
            raise Exception(
                f"Cacti Error in {filename}",
                {p.stderr.read().decode()},
                {f.read().split("\n")[-2] if f.read() else "No output"},
            )

    output_filename = f"cfg/{filename}.cfg.out"
    cactiOutput = os.path.normpath(os.path.join(CACTI_DIR, output_filename))

    # SHORTCUT FOR NOW
    if not os.path.exists(cactiOutput):
        from collections import defaultdict

        default_dict = defaultdict(int)
        print("ISSUE HERE!")
        print()
        return default_dict

    output_data = pd.read_csv(cactiOutput, sep=", ", engine="python")
    output_data = output_data.iloc[
        -1
    ]  # get just the last row which is the most recent run

    IO_freq = config_values["bus_freq"] * 1e6
    IO_latency = addr_timing / IO_freq

    output_data["IO latency (s)"] = IO_latency

    return output_data


def run_existing_cacti_cfg(filename):
    """
    Retrieves timing and power values from a Cacti run of an existing .cfg file.

    Inputs:
    filename : str
        Name of the existing .cfg file to run in Cacti. path relative to CACTI_DIR

    Returns:
    pd.DataFrame
        DataFrame containing timing, power, and IO-related values from the Cacti run.
            # CACTI: access time (ns), search energy (nJ), read energy (nJ), write energy (nJ), leakage bank power (mW)
            # CACTI IO: area (sq.mm), timing (ps), dynamic power (mW), PHY power (mW), termination and bias power (mW)
            # latency (ns)

    Outputs:
    Cacti run results are returned in a DataFrame after executing the existing .cfg file.
    """

    stdout_filename = "cacti_stdout.log"
    stdout_file_path = os.path.join(CACTI_DIR, stdout_filename)

    cmd = ["./cacti", "-infile", filename]

    with open(stdout_file_path, "w") as f:
        p = subprocess.Popen(cmd, cwd=CACTI_DIR, stdout=f, stderr=subprocess.PIPE)

    p.wait()
    if p.returncode != 0:
        with open(stdout_file_path, "r") as f:
            raise Exception(
                f"Cacti Error in {filename}",
                {p.stderr.read().decode()},
                {f.read().split("\n")[-2]},
            )

    output_filename = filename + ".out"
    cactiOutput = os.path.join(CACTI_DIR, output_filename)
    output_data = pd.read_csv(cactiOutput, sep=", ", engine="python")
    output_data = output_data.iloc[
        -1
    ]  # get just the last row which is the most recent run

    # get IO params
    bus_freq = None
    addr_timing = None
    cacti_input_filename = os.path.join(CACTI_DIR, filename)

    with open(cacti_input_filename, "r") as file:
        for line in file:
            if "-bus_freq" in line:
                bus_freq = line.split()[1] + " " + line.split()[2]
            elif "-addr_timing" in line:
                addr_timing = line.split()[1]

    IO_freq = convert_frequency(bus_freq)
    IO_latency = float(addr_timing) / IO_freq

    output_data["IO latency (s)"] = IO_latency

    return output_data


def convert_frequency(string):
    """
    Helper for converting frequency string to Hz for 'run_existing_cacti_cfg' and 'gen_vals'.

    Inputs:
    string : str
        Frequency string with a value and unit (e.g., "2 GHz", "500 MHz").

    Returns:
    int
        Frequency in Hz, or prints an error message if the input format is invalid.

    Outputs:
    Frequency value in Hz based on the input string.
    """

    parts = string.split()

    if len(parts) == 2 and parts[1].lower() in ("ghz", "mhz"):
        try:
            value = int(parts[0])  # Extract the integer part
            unit = parts[1].lower()  # Extract the unit and convert to lowercase

            if unit == "ghz":
                return value * 10**9
            elif unit == "mhz":
                return value * 10**6
        except ValueError:
            print("Invalid input format")
    else:
        print("Invalid input format")


def update_dat(rcs, dat_file):
    """
    Replaces contents of the optimization .dat file with new Cacti parameter values.

    Inputs:
    rcs : dict
        Dictionary containing updated Cacti parameters.
    dat_file : str
        Path to the .dat file to be updated.

    Outputs:
    Updates the specified .dat file by replacing values with those from the `rcs["Cacti"]` dictionary.
    """

    cacti_params = rcs["Cacti"]
    cacti_params.pop("I_off_n", None)
    cacti_params.pop("I_g_on_n", None)
    cacti_params.pop("Wmemcella", None)
    cacti_params.pop("Wmemcellpmos", None)
    cacti_params.pop("Wmemcellnmos", None)
    cacti_params.pop("area_cell", None)
    cacti_params.pop("asp_ratio_cell", None)

    for key, value in cacti_params.items():
        replace_values_in_dat_file(dat_file, key, value)


def replace_values_in_dat_file(dat_file_path, key, new_value):
    """
    Helper to replace the original value in the .dat file with a new value.

    Inputs:
    dat_file_path : str
        Path to the .dat file to be modified.
    key : str
        The parameter key whose value needs to be replaced.
    new_value : float
        The new value to replace the original, calculated as (original value - cacti_python_delta).

    Returns:
    dict
        A dictionary containing the original values of the replaced parameters.

    Outputs:
    Updates the specified .dat file by replacing values for the given key with the new value.
    """

    original_values = {}

    with open(dat_file_path, "r") as file:
        lines = file.readlines()

    pattern = re.compile(rf"^-{key}\s+\((.*?)\)\s+(.*)$")
    for i, line in enumerate(lines):
        match = pattern.match(line.strip())
        if match:
            # Extract original values and store them
            original_values[i] = match.group(2).split()
            # Keep the unit label (e.g., (F/um), (V), etc.)
            unit_label = match.group(1)
            # Replace the numeric values with the new value
            lines[i] = (
                f"-{key} ({unit_label}) "
                + " ".join([str(new_value)] * len(original_values[i]))
                + "\n"
            )

    with open(dat_file_path, "w") as file:
        file.writelines(lines)

    return original_values


def restore_original_values_in_dat_file(dat_file_path, original_values):
    """
    Helper to restore the original values in the .dat file.

    Inputs:
    dat_file_path : str
        Path to the .dat file to be modified.
    original_values : dict
        Dictionary containing the original values to restore, with line numbers as keys.

    Outputs:
    Updates the specified .dat file by restoring the original values for the parameters.
    """

    with open(dat_file_path, "r") as file:
        lines = file.readlines()

    for i, values in original_values.items():
        parts = lines[i].split()
        # Preserve the key and unit label
        key_and_unit = " ".join(parts[:2])
        # Replace the rest with the original values
        lines[i] = f"{key_and_unit} " + " ".join(values) + "\n"

    with open(dat_file_path, "w") as file:
        file.writelines(lines)


# Function to read the file and extract values
def read_config_file(in_file: str):
    # Initialize dictionary to store extracted values
    config_dict = {}
    with open(in_file, "r") as fp:
        lines = fp.readlines()
    
    # raise Exception()

    for line in lines:
        line = line.strip()
        if line.startswith("-size"):
            config_dict["cache_size"] = int(line.split()[-1])
        elif line.startswith("-page size"):
            config_dict["page_sz_bits"] = int(line.split()[-1])
        elif line.startswith("-burst length"):
            config_dict["burst_len"] = int(line.split()[-1])
        elif line.startswith("-internal prefetch width"):
            config_dict["int_prefetch_w"] = int(line.split()[-1])
        elif line.startswith("-block size (bytes)"):
            config_dict["line_sz"] = int(line.split()[-1])
        elif line.startswith("-associativity"):
            config_dict["assoc"] = int(line.split()[-1])
        elif line.startswith("-read-write port"):
            config_dict["num_rw_ports"] = int(line.split()[-1])
        elif line.startswith("-exclusive read port"):
            config_dict["num_rd_ports"] = int(line.split()[-1])
        elif line.startswith("-exclusive write port"):
            config_dict["num_wr_ports"] = int(line.split()[-1])
        elif line.startswith("-single"):
            config_dict["num_se_rd_ports"] = int(line.split()[-1])
        elif line.startswith("-search port"):
            config_dict["num_search_ports"] = int(line.split()[-1])
        elif line.startswith("-UCA bank"):
            config_dict["nbanks"] = int(line.split()[-1])
        elif line.startswith("-technology"):
            config_dict["F_sz_um"] = float(line.split()[-1])
            config_dict["F_sz_nm"] = config_dict["F_sz_um"] * 1000
        elif line.startswith("-output/input bus"):
            config_dict["bus_width"] = int(float(line.split()[-1]))
        elif line.startswith("-operating temperature"):
            config_dict["temp"] = int(line.split()[-1])
        elif line.startswith("-cache type"):
            config_dict["cache_type"] = line.split('"')[1]
            config_dict["is_cache"] = "cache" in config_dict["cache_type"]
            config_dict["is_main_mem"] = "main memory" in config_dict["cache_type"]
            config_dict["is_3d_mem"] = (
                "3D memory or 2D main memory" in config_dict["cache_type"]
            )
            config_dict["pure_cam"] = "cam" in config_dict["cache_type"]
            config_dict["pure_ram"] = (
                "ram" in config_dict["cache_type"] or config_dict["is_main_mem"]
            )
        elif line.startswith("-print option"):
            print_option = line.split('"')[1]
            config_dict["print_detail_debug"] = "debug detail" in print_option
        elif line.startswith("-burst depth"):
            config_dict["burst_depth"] = int(line.split()[-1])
        elif line.startswith("-IO width"):
            config_dict["io_width"] = int(line.split()[-1])
        elif line.startswith("-system frequency"):
            config_dict["sys_freq_MHz"] = int(line.split()[-1])
        elif line.startswith("-stacked die"):
            config_dict["num_die_3d"] = int(line.split()[-1])
        elif line.startswith("-partitioning granularity"):
            config_dict["partition_gran"] = int(line.split()[-1])
        elif line.startswith("-TSV projection"):
            config_dict["TSV_proj_type"] = int(line.split()[-1])
        elif line.startswith("-tag size"):
            config_dict["tag_size"] = line.split('"')[1]
            if "default" in config_dict["tag_size"]:
                config_dict["specific_tag"] = False
                config_dict["tag_w"] = 42
            else:
                config_dict["specific_tag"] = True
                config_dict["tag_w"] = int(line.split()[-1])
        elif line.startswith("-access mode"):
            access_mode = line.split('"')[1]
            if "fast" in access_mode:
                config_dict["access_mode"] = 2
            elif "sequential" in access_mode:
                config_dict["access_mode"] = 1
            elif "normal" in access_mode:
                config_dict["access_mode"] = 0
            else:
                raise ValueError("Invalid access mode")
        elif line.startswith("-Data array cell type"):
            config_dict["data_array_cell_type"] = line.split('"')[1]
            cell_type = config_dict["data_array_cell_type"]
            if "itrs-hp" in cell_type:
                config_dict["data_arr_ram_cell_tech_type"] = 0
            elif "itrs-lstp" in cell_type:
                config_dict["data_arr_ram_cell_tech_type"] = 1
            elif "itrs-lop" in cell_type:
                config_dict["data_arr_ram_cell_tech_type"] = 2
            elif "lp-dram" in cell_type:
                config_dict["data_arr_ram_cell_tech_type"] = 3
            elif "comm-dram" in cell_type:
                config_dict["data_arr_ram_cell_tech_type"] = 4
            else:
                raise ValueError("Invalid data array cell type")
        elif line.startswith("-Data array peripheral type"):
            config_dict["data_array_peri_type"] = line.split('"')[1]
            peri_type = config_dict["data_array_peri_type"]
            if "itrs-hp" in peri_type:
                config_dict["data_arr_peri_global_tech_type"] = 0
            elif "itrs-lstp" in peri_type:
                config_dict["data_arr_peri_global_tech_type"] = 1
            elif "itrs-lop" in peri_type:
                config_dict["data_arr_peri_global_tech_type"] = 2
            else:
                raise ValueError("Invalid data array peripheral type")
        elif line.startswith("-Tag array cell type"):
            config_dict["tag_array_cell_type"] = line.split('"')[1]
            cell_type = config_dict["tag_array_cell_type"]
            if "itrs-hp" in cell_type:
                config_dict["tag_arr_ram_cell_tech_type"] = 0
            elif "itrs-lstp" in cell_type:
                config_dict["tag_arr_ram_cell_tech_type"] = 1
            elif "itrs-lop" in cell_type:
                config_dict["tag_arr_ram_cell_tech_type"] = 2
            elif "lp-dram" in cell_type:
                config_dict["tag_arr_ram_cell_tech_type"] = 3
            elif "comm-dram" in cell_type:
                config_dict["tag_arr_ram_cell_tech_type"] = 4
            else:
                raise ValueError("Invalid tag array cell type")
        elif line.startswith("-Tag array peripheral type"):
            config_dict["tag_array_peri_type"] = line.split('"')[1]
            peri_type = config_dict["tag_array_peri_type"]
            if "itrs-hp" in peri_type:
                config_dict["tag_arr_peri_global_tech_type"] = 0
            elif "itrs-lstp" in peri_type:
                config_dict["tag_arr_peri_global_tech_type"] = 1
            elif "itrs-lop" in peri_type:
                config_dict["tag_arr_peri_global_tech_type"] = 2
            else:
                raise ValueError("Invalid tag array peripheral type")
        elif line.startswith("-design"):
            match = re.search(
                r"-design objective \(weight delay, dynamic power, leakage power, cycle time, area\) (\d+):(\d+):(\d+):(\d+):(\d+)",
                line,
            )
            if match:
                config_dict["delay_wt"] = int(match.group(1))
                config_dict["dynamic_power_wt"] = int(match.group(2))
                config_dict["leakage_power_wt"] = int(match.group(3))
                config_dict["cycle_time_wt"] = int(match.group(4))
                config_dict["area_wt"] = int(match.group(5))
        # Repeat the same pattern for the rest of the `self.` attributes

        elif line.startswith("-deviate"):
            match = re.search(
                r"-deviate \(delay, dynamic power, leakage power, cycle time, area\) (\d+):(\d+):(\d+):(\d+):(\d+)",
                line,
            )
            if match:
                config_dict["delay_dev"] = int(match.group(1))
                config_dict["dynamic_power_dev"] = int(match.group(2))
                config_dict["leakage_power_dev"] = int(match.group(3))
                config_dict["cycle_time_dev"] = int(match.group(4))
                config_dict["area_dev"] = int(match.group(5))
        elif line.startswith("-Optimize"):
            optimize = line.split('"')[1]
            if "ED^2" in optimize:
                config_dict["ed"] = 2
            elif "ED" in optimize:
                config_dict["ed"] = 1
            else:
                config_dict["ed"] = 0
        elif line.startswith("-NUCAdesign"):
            match = re.search(
                r"-NUCAdesign objective \(weight delay, dynamic power, leakage power, cycle time, area\) (\d+):(\d+):(\d+):(\d+):(\d+)",
                line,
            )
            if match:
                config_dict["delay_wt_nuca"] = int(match.group(1))
                config_dict["dynamic_power_wt_nuca"] = int(match.group(2))
                config_dict["leakage_power_wt_nuca"] = int(match.group(3))
                config_dict["cycle_time_wt_nuca"] = int(match.group(4))
                config_dict["area_wt_nuca"] = int(match.group(5))
        elif line.startswith("-NUCAdeviate"):
            match = re.search(
                r"-NUCAdeviate \(delay, dynamic power, leakage power, cycle time, area\) (\d+):(\d+):(\d+):(\d+):(\d+)",
                line,
            )
            if match:
                config_dict["delay_dev_nuca"] = int(match.group(1))
                config_dict["dynamic_power_dev_nuca"] = int(match.group(2))
                config_dict["leakage_power_dev_nuca"] = int(match.group(3))
                config_dict["cycle_time_dev_nuca"] = int(match.group(4))
                config_dict["area_dev_nuca"] = int(match.group(5))
        elif line.startswith("-Cache model"):
            cache_model = line.split('"')[1]
            config_dict["nuca"] = 0 if "UCA" in cache_model else 1
        elif line.startswith("-NUCA bank count"):
            config_dict["nuca_bank_count"] = int(line.split()[-1])
            if config_dict["nuca_bank_count"] != 0:
                config_dict["force_nuca_bank"] = 1
        elif line.startswith("-Wire inside mat"):
            wire_type = line.split('"')[1]
            if "global" in wire_type:
                config_dict["wire_is_mat_type"] = 2
            elif "local" in wire_type:
                config_dict["wire_is_mat_type"] = 0
            else:
                config_dict["wire_is_mat_type"] = 1
        elif line.startswith("-Wire outside mat"):
            wire_type = line.split('"')[1]
            if "global" in wire_type:
                config_dict["wire_os_mat_type"] = 2
            else:
                config_dict["wire_os_mat_type"] = 1
        elif line.startswith("-Interconnect projection"):
            ic_proj_type = line.split('"')[1]
            config_dict["ic_proj_type"] = 0 if "aggressive" in ic_proj_type else 1
        elif line.startswith("-Wire signaling"):
            wire_signaling = line.split('"')[1]
            if "default" in wire_signaling:
                config_dict["force_wiretype"] = 0
                config_dict["wt"] = "Global"
            elif "Global_10" in wire_signaling:
                config_dict["force_wiretype"] = 1
                config_dict["wt"] = "Global_10"
            elif "Global_20" in wire_signaling:
                config_dict["force_wiretype"] = 1
                config_dict["wt"] = "Global_20"
            elif "Global_30" in wire_signaling:
                config_dict["force_wiretype"] = 1
                config_dict["wt"] = "Global_30"
            elif "Global_5" in wire_signaling:
                config_dict["force_wiretype"] = 1
                config_dict["wt"] = "Global_5"
            elif "Global" in wire_signaling:
                config_dict["force_wiretype"] = 1
                config_dict["wt"] = "Global"
            elif "fullswing" in wire_signaling:
                config_dict["force_wiretype"] = 1
                config_dict["wt"] = "Full_swing"
            elif "lowswing" in wire_signaling:
                config_dict["force_wiretype"] = 1
                config_dict["wt"] = "Low_swing"
            else:
                raise ValueError("Unknown wire type")
        elif line.startswith("-Core count"):
            config_dict["cores"] = int(line.split()[-1])
            if config_dict["cores"] > 16:
                raise ValueError("No. of cores should be less than 16!")
        elif line.startswith("-Cache level"):
            cache_level = line.split('"')[1]
            config_dict["cache_level"] = 0 if "L2" in cache_level else 1
        elif line.startswith("-Print level"):
            print_level = line.split('"')[1]
            config_dict["print_detail"] = 1 if "DETAILED" in print_level else 0
        elif line.startswith("-Add ECC"):
            add_ecc = line.split('"')[1]
            config_dict["add_ecc_b_"] = True if "true" in add_ecc else False
        elif line.startswith("-CLDriver vertical"):
            cl_driver = line.split('"')[1]
            config_dict["cl_vertical"] = True if "true" in cl_driver else False
        elif line.startswith("-Array Power Gating"):
            array_power = line.split('"')[1]
            config_dict["array_power_gated"] = True if "true" in array_power else False
        elif line.startswith("-Bitline floating"):
            bitline_float = line.split('"')[1]
            config_dict["bitline_floating"] = True if "true" in bitline_float else False
        elif line.startswith("-WL Power Gating"):
            wl_power = line.split('"')[1]
            config_dict["wl_power_gated"] = True if "true" in wl_power else False
        elif line.startswith("-CL Power Gating"):
            cl_power = line.split('"')[1]
            config_dict["cl_power_gated"] = True if "true" in cl_power else False
        elif line.startswith("-Interconnect Power Gating"):
            interconnect_power = line.split('"')[1]
            config_dict["interconnect_power_gated"] = (
                True if "true" in interconnect_power else False
            )
        elif line.startswith("-Power Gating Performance Loss"):
            val = line.split()[-1]
            cleaned_value = val.strip('"').strip()
            config_dict["perfloss"] = float(cleaned_value)
        elif line.startswith("-Print input parameters"):
            print_input = line.split('"')[1]
            config_dict["print_input_args"] = True if "true" in print_input else False
        elif line.startswith("-Force cache config"):
            force_cache = line.split('"')[1]
            config_dict["force_cache_config"] = True if "true" in force_cache else False
        elif line.startswith("-Ndbl"):
            config_dict["ndbl"] = int(line.split()[-1])
        elif line.startswith("-Ndwl"):
            config_dict["ndwl"] = int(line.split()[-1])
        elif line.startswith("-Nspd"):
            config_dict["nspd"] = int(line.split()[-1])
        elif line.startswith("-Ndsam1"):
            config_dict["ndsam1"] = int(line.split()[-1])
        elif line.startswith("-Ndsam2"):
            config_dict["ndsam2"] = int(line.split()[-1])
        elif line.startswith("-Ndcm"):
            config_dict["ndcm"] = int(line.split()[-1])
        elif line.startswith("-dram_type"):
            dram_type = line.split('"')[1]
            if "DDR3" in dram_type:
                config_dict["io_type"] = "DDR3"
            elif "DDR4" in dram_type:
                config_dict["io_type"] = "DDR4"
            elif "LPDDR2" in dram_type:
                config_dict["io_type"] = "LPDDR2"
            elif "WideIO" in dram_type:
                config_dict["io_type"] = "WideIO"
            elif "Low_Swing_Diff" in dram_type:
                config_dict["io_type"] = "Low_Swing_Diff"
            elif "Serial" in dram_type:
                config_dict["io_type"] = "Serial"
            else:
                raise ValueError("Invalid Input for dram type")
        elif line.startswith("-io state"):
            io_state = line.split('"')[1]
            config_dict["io_state"] = io_state
            if "READ" in io_state:
                config_dict["iostate"] = "READ"
            elif "WRITE" in io_state:
                config_dict["iostate"] = "WRITE"
            elif "IDLE" in io_state:
                config_dict["iostate"] = "IDLE"
            elif "SLEEP" in io_state:
                config_dict["iostate"] = "SLEEP"
            else:
                raise ValueError("Invalid Input for io state")
        elif line.startswith("-addr_timing"):
            config_dict["addr_timing"] = float(
                re.search(r"-addr_timing (\d+(\.\d+)?)", line).group(1)
            )
        elif line.startswith("-dram ecc"):
            dram_ecc = line.split('"')[1]
            if "NO_ECC" in dram_ecc:
                config_dict["dram_ecc"] = "NO_ECC"
            elif "SECDED" in dram_ecc:
                config_dict["dram_ecc"] = "SECDED"
            elif "CHIP_KILL" in dram_ecc:
                config_dict["dram_ecc"] = "CHIP_KILL"
            else:
                raise ValueError("Invalid Input for dram ecc")
        elif line.startswith("-dram dimm"):
            dram_dimm = line.split('"')[1]
            if "UDIMM" in dram_dimm:
                config_dict["dram_dimm"] = "UDIMM"
            elif "RDIMM" in dram_dimm:
                config_dict["dram_dimm"] = "RDIMM"
            elif "LRDIMM" in dram_dimm:
                config_dict["dram_dimm"] = "LRDIMM"
            else:
                raise ValueError("Invalid Input for dram dimm")
        elif line.startswith("-bus_bw"):
            config_dict["bus_bw"] = float(line.split()[-1])
        elif line.startswith("-duty_cycle"):
            config_dict["duty_cycle"] = float(line.split()[-1])
        elif line.startswith("-mem_density"):
            config_dict["mem_density"] = float(
                re.search(r"-mem_density (\d+)", line).group(1)
            )
        elif line.startswith("-activity_dq"):
            config_dict["activity_dq"] = float(
                re.search(r"-activity_dq (\d+\.\d+)", line).group(1)
            )
        elif line.startswith("-activity_ca"):
            config_dict["activity_ca"] = float(
                re.search(r"-activity_ca (\d+\.\d+)", line).group(1)
            )
        elif line.startswith("-bus_freq"):
            config_dict["bus_freq"] = float(
                re.search(r"-bus_freq (\d+)", line).group(1)
            )
        elif line.startswith("-num_dqs"):
            # dqs check has to be before dq check else won't pass
            match = re.search(r"-num_dqs (\d+)", line)
            if match:
                config_dict["num_dqs"] = int(match.group(1))
            else:
                config_dict["num_dqs"] = 0
        elif line.startswith("-num_dq"):
            match = re.search(r"-num_dq (\d+)", line)
            if match:
                config_dict["num_dq"] = int(match.group(1))
        elif line.startswith("-num_ca"):
            config_dict["num_ca"] = int(re.search(r"-num_ca (\d+)", line).group(1))
        elif line.startswith("-num_clk"):
            config_dict["num_clk"] = int(re.search(r"-num_clk (\d+)", line).group(1))
            if config_dict["num_clk"] <= 0:
                raise ValueError("num_clk should be greater than zero!")
        elif line.startswith("-num_mem_dq"):
            config_dict["num_mem_dq"] = int(
                re.search(r"-num_mem_dq (\d+)", line).group(1)
            )
        elif line.startswith("-mem_data_width"):
            config_dict["mem_data_width"] = int(
                re.search(r"-mem_data_width (\d+)", line).group(1)
            )
        elif line.startswith("-num_bobs"):
            config_dict["num_bobs"] = int(line.split()[-1])
        elif line.startswith("-capacity"):
            value = line.split()[-1]
            if "." in value:
                config_dict["capacity"] = float(value)
            else:
                config_dict["capacity"] = int(value)
        elif line.startswith("-num_channels_per_bob"):
            config_dict["num_channels_per_bob"] = int(line.split()[-1])
        elif line.startswith("-first metric"):
            first_metric = line.split('"')[1]
            if "Cost" in first_metric:
                config_dict["first_metric"] = "Cost"
            elif "Energy" in first_metric:
                config_dict["first_metric"] = "Energy"
            elif "Bandwidth" in first_metric:
                config_dict["first_metric"] = "Bandwidth"
            else:
                raise ValueError("Invalid Input for first metric")
        elif line.startswith("-second metric"):
            second_metric = line.split('"')[1]
            if "Cost" in second_metric:
                config_dict["second_metric"] = "Cost"
            elif "Energy" in second_metric:
                config_dict["second_metric"] = "Energy"
            elif "Bandwidth" in second_metric:
                config_dict["second_metric"] = "Bandwidth"
            else:
                raise ValueError("Invalid Input for second metric")
        elif line.startswith("-third metric"):
            third_metric = line.split('"')[1]
            if "Cost" in third_metric:
                config_dict["third_metric"] = "Cost"
            elif "Energy" in third_metric:
                config_dict["third_metric"] = "Energy"
            elif "Bandwidth" in third_metric:
                config_dict["third_metric"] = "Bandwidth"
            else:
                raise ValueError("Invalid Input for third metric")
        elif line.startswith("-DIMM model"):
            dimm_model = line.split('"')[1]
            if "JUST_UDIMM" in dimm_model:
                config_dict["dimm_model"] = "JUST_UDIMM"
            elif "JUST_RDIMM" in dimm_model:
                config_dict["dimm_model"] = "JUST_RDIMM"
            elif "JUST_LRDIMM" in dimm_model:
                config_dict["dimm_model"] = "JUST_LRDIMM"
            elif "ALL" in dimm_model:
                config_dict["dimm_model"] = "ALL"
            else:
                raise ValueError("Invalid Input for DIMM model")
        elif line.startswith("-Low Power Permitted"):
            low_power = line.split('"')[1]
            config_dict["low_power_permitted"] = True if "T" in low_power else False
        elif line.startswith("-load"):
            config_dict["load"] = float(line.split()[-1])
        elif line.startswith("-row_buffer_hit_rate"):
            config_dict["row_buffer_hit_rate"] = float(line.split()[-1])
        elif line.startswith("-rd_2_wr_ratio"):
            config_dict["rd_2_wr_ratio"] = float(line.split()[-1])
        elif line.startswith("-same_bw_in_bob"):
            same_bw = line.split('"')[1]
            config_dict["same_bw_in_bob"] = True if "T" in same_bw else False
        elif line.startswith("-mirror_in_bob"):
            mirror = line.split('"')[1]
            config_dict["mirror_in_bob"] = True if "T" in mirror else False
        elif line.startswith("-total_power"):
            total_power = line.split('"')[1]
            config_dict["total_power"] = True if "T" in total_power else False
        elif line.startswith("-verbose"):
            verbose = line.split('"')[1]
            config_dict["verbose"] = True if "T" in verbose else False
        elif line.startswith("-rtt_value"):
            config_dict["rtt_value"] = float(line.split()[-1])
        elif line.startswith("-ron_value"):
            config_dict["ron_value"] = float(line.split()[-1])
        elif line.startswith("-tflight_value"):
            config_dict["tflight_value"] = line.split()[-1]  # or convert it as needed

    A = 0
    seq_access = False
    fast_access = True

    if config_dict["access_mode"] == 0:
        seq_access = False
        fast_access = False
    elif config_dict["access_mode"] == 1:
        seq_access = True
        fast_access = False
    elif config_dict["access_mode"] == 2:
        seq_access = False
        fast_access = True

    if config_dict["is_main_mem"]:
        if config_dict["ic_proj_type"] == 0 and not g_ip.is_3d_mem:
            print("DRAM model supports only conservative interconnect projection!\n\n")
            raise ValueError(
                "DRAM model supports only conservative interconnect projection"
            )
            # return False

    B = config_dict["line_sz"]

    if B < 1:
        print("Block size must >= 1")
        return False
    elif B * 8 < config_dict["bus_width"]:
        print(f"Block size must be at least {config_dict['bus_width'] / 8}")
        raise ValueError(f"Block size must be at least {config_dict['bus width'] / 8}")
        # return False

    if config_dict["F_sz_um"] <= 0:
        print("Feature size must be > 0")
        raise ValueError("Feature size must be > 0")
        # return False
    elif config_dict["F_sz_um"] > 0.091:
        print("Feature size must be <= 90 nm")
        raise ValueError(f"Feature size must be <= 90 nm; feauture size is {config_dict['F_sz_um']}")
        # return False

    RWP = config_dict["num_rw_ports"]
    ERP = config_dict["num_rd_ports"]
    EWP = config_dict["num_wr_ports"]
    NSER = config_dict["num_se_rd_ports"]
    SCHP = config_dict["num_search_ports"]

    if (RWP + ERP + EWP) < 1:
        print("Must have at least one port")
        raise ValueError("Must have at least one port")
        # return False

    if not is_pow2(config_dict["nbanks"]):
        print(
            "Number of subbanks should be greater than or equal to 1 and should be a power of 2"
        )
        raise ValueError(
            "Number of subbanks should be greater than or equal to 1 and should be a power of 2"
        )
        # return False

    C = config_dict["cache_size"] / config_dict["nbanks"]
    if C < 64 and not g_ip.is_3d_mem:
        print("Cache size must >=64")
        raise ValueError("Cache size must >=64")
        # return False

    if config_dict["is_cache"] and config_dict["assoc"] == 0:
        config_dict["fully_assoc"] = True
    else:
        config_dict["fully_assoc"] = False

    if config_dict["pure_cam"] and config_dict["assoc"] != 0:
        print("Pure CAM must have associativity as 0")
        raise ValueError("Pure CAM must have associativity as 0")
        # return False

    if (
        config_dict["assoc"] == 0
        and not config_dict["pure_cam"]
        and not config_dict["is_cache"]
    ):
        print("Only CAM or Fully associative cache can have associativity as 0")
        raise ValueError(
            "Only CAM or Fully associative cache can have associativity as 0"
        )
        # return False

    if (config_dict["fully_assoc"] or config_dict["pure_cam"]) and (
        config_dict["data_arr_ram_cell_tech_type"]
        != config_dict["tag_arr_ram_cell_tech_type"]
        or config_dict["data_arr_peri_global_tech_type"]
        != config_dict["tag_arr_peri_global_tech_type"]
    ):
        print(
            "CAM and fully associative cache must have same device type for both data and tag array"
        )
        raise ValueError(
            "CAM and fully associative cache must have same device type for both data and tag array"
        )
        # return False

    if (config_dict["fully_assoc"] or config_dict["pure_cam"]) and (
        config_dict["data_arr_ram_cell_tech_type"] == lp_dram
        or config_dict["data_arr_ram_cell_tech_type"] == comm_dram
    ):
        print("DRAM based CAM and fully associative cache are not supported")
        raise ValueError("DRAM based CAM and fully associative cache are not supported")

    if (config_dict["fully_assoc"] or config_dict["pure_cam"]) and config_dict[
        "is_main_mem"
    ]:
        print("CAM and fully associative cache cannot be as main memory")
        raise ValueError("CAM and fully associative cache cannot be as main memory")

    if (config_dict["fully_assoc"] or config_dict["pure_cam"]) and SCHP < 1:
        print("CAM and fully associative must have at least 1 search port")
        raise ValueError("CAM and fully associative must have at least 1 search port")
    if (
        RWP == 0
        and ERP == 0
        and SCHP > 0
        and (config_dict["fully_assoc"] or config_dict["pure_cam"])
    ):
        ERP = SCHP

    if config_dict["assoc"] == 0:
        A = C / B
    else:
        if config_dict["assoc"] == 1:
            A = 1
        else:
            A = config_dict["assoc"]
            if not is_pow2(A):
                print("Associativity must be a power of 2")
                raise ValueError("Associativity must be a power of 2")

    if C / (B * A) <= 1 and config_dict["assoc"] != 0 and not g_ip.is_3d_mem:
        print(
            "Number of sets is too small: "
            + " Need to either increase cache size, or decrease associativity or block size"
            + " (or use fully associative cache)"
        )
        raise ValueError(
            "Number of sets is too small: "
            + "Need to either increase cache size, or decrease associativity or block size"
            + " (or use fully associative cache)"
        )
    config_dict["block_size"] = B

    if seq_access:
        config_dict["tag_assoc"] = A
        config_dict["data_assoc"] = 1
        config_dict["is_seq_acc"] = True
    else:
        config_dict["tag_assoc"] = A
        config_dict["data_assoc"] = A
        config_dict["is_seq_acc"] = False

    if config_dict["assoc"] == 0:
        config_dict["data_assoc"] = 1

    config_dict["num_rw_ports"] = RWP
    config_dict["num_rd_ports"] = ERP
    config_dict["num_wr_ports"] = EWP
    config_dict["num_se_rd_ports"] = NSER

    if not (config_dict["fully_assoc"] or config_dict["pure_cam"]):
        config_dict["num_search_ports"] = 0

    config_dict["nsets"] = C / (B * A)

    if (
        config_dict["temp"] < 300
        or config_dict["temp"] > 400
        or config_dict["temp"] % 10 != 0
    ):
        print(
            f"{config_dict['temp']} Temperature must be between 300 and 400 Kelvin and multiple of 10."
        )
        raise ValueError(f"{config_dict['temp']} Temperature must be between 300 and 400 Kelvin and multiple of 10.")

    if config_dict["nsets"] < 1 and not g_ip.is_3d_mem:
        print("Less than one set...")
        raise ValueError("Less than one set...")

    config_dict["power_gating"] = (
        config_dict["array_power_gated"]
        or config_dict["bitline_floating"]
        or config_dict["wl_power_gated"]
        or config_dict["cl_power_gated"]
        or config_dict["interconnect_power_gated"]
    )

    logger.info("Done reading cacti cfg file.")
    return config_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a config file and a data file."
    )
    parser.add_argument(
        "--cfg_name",
        type=str,
        default="cache",
        help="Path to the configuration file (default: mem_validate_cache)",
    )
    parser.add_argument(
        "--adjust",
        action="store_true",
        help="Boolean flag to detail adjust cfg through arguments",
    )
    parser.add_argument(
        "--dat_file",
        type=str,
        default="src/cacti/tech_params/90nm.dat",
        help="Path to the data file (default: src/cacti/tech_params/90nm.dat)",
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=131072,
        help="Path to the data file (default: 131072)",
    )
    parser.add_argument(
        "--blockSize", type=int, default=64, help="Path to the data file (default: 64)"
    )
    parser.add_argument(
        "--cacheType",
        type=str,
        default="main memory",
        help="Path to the data file (default: main memory)",
    )
    parser.add_argument(
        "--busWidth", type=int, default=64, help="Path to the data file (default: 64)"
    )

    args = parser.parse_args()
    cache_cfg = f"src/cacti/cfg/{args.cfg_name}.cfg"

    if args.adjust:
        buf_vals = gen_vals(
            args.cfg_name,
            cache_size=args.cache_size,
            blockSize=args.blockSize,
            cache_type=args.cacheType,
            bus_width=args.busWidth,
        )
    else:
        buf_vals = run_existing_cacti_cfg(cache_cfg)

    buf_opt = {
        "ndwl": buf_vals["Ndwl"],
        "ndbl": buf_vals["Ndbl"],
        "nspd": buf_vals["Nspd"],
        "ndcm": buf_vals["Ndcm"],
        "ndsam1": buf_vals["Ndsam_level_1"],
        "ndsam2": buf_vals["Ndsam_level_2"],
        "repeater_spacing": buf_vals["Repeater spacing"],
        "repeater_size": buf_vals["Repeater size"],
    }

    sympy_file = args.cfg_name
    IO_info = gen_symbolic(sympy_file, cache_cfg, buf_opt, use_piecewise=False)
