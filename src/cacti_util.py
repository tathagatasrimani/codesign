import os
import subprocess
import yaml
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
CACTI_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), 'cacti'))

def gen_symbolic(name, cache_cfg, opt_vals, use_piecewise=False):
    """
    Generates SymPy expressions for access time and energy, outputting results to text files.

    Inputs:
    name : str
        Base name for the output files.
    cache_cfg : str
        Path to the cache configuration file.
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

    print(f"gen_symbolic: cwd: {os.getcwd()}")
    g_ip.parse_cfg(os.path.join(CACTI_DIR, cache_cfg))
    g_ip.error_checking()
    # g_ip.display_ip()

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
    output_dir = os.path.join(CACTI_DIR, 'symbolic_expressions')

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write files to the cacti/sympy directory
    with open(os.path.join(output_dir, f'{name + "_access_time"}.txt'), 'w') as file:
        file.write(str(fin_res.access_time))

    with open(os.path.join(output_dir, f'{name + "_read_dynamic"}.txt'), 'w') as file:
        file.write(str(fin_res.power.readOp.dynamic))

    with open(os.path.join(output_dir, f'{name + "_write_dynamic"}.txt'), 'w') as file:
        file.write(str(fin_res.power.writeOp.dynamic))

    with open(os.path.join(output_dir, f'{name + "_read_leakage"}.txt'), 'w') as file:
        file.write(str(fin_res.power.readOp.leakage))

    IO_info = {
        "io_area": fin_res.io_area,
        "io_timing_margin": fin_res.io_timing_margin,
        "io_dynamic_power": fin_res.io_dynamic_power,
        "io_phy_power": fin_res.io_phy_power,
        "io_termination_power": fin_res.io_termination_power
    }

    for key, value in IO_info.items():
        file_name = f'{name + "_" + key}.txt'
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w') as file:
            file.write(str(value))

    return IO_info

def gen_vals(filename = "base_cache", cacheSize = None, blockSize = None,
             cache_type = None, bus_width = None, transistor_size = None,
             addr_timing = None, force_cache_config = None, technology = None,
             debug = False) -> pd.DataFrame:
    """
    Generates a Cacti .cfg file based on input and cacti_input, runs Cacti, 
    and retrieves timing and power values.

    Inputs:
    filename : str, optional
        Base name for the generated .cfg file (default is "base_cache").
    cacheSize : int, optional
        Size of the cache in bytes.
    blockSize : int, optional
        Size of each cache block in bytes.
    cache_type : str, optional
        Type of cache (e.g., "cache" or "main memory").
    bus_width : int, optional
        Width of the input/output bus in bits.
    transistor_size : float, optional
        Size of the transistor technology node (e.g., 45nm).
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

    Outputs:
    A .cfg file is generated for Cacti, and Cacti run results are 
    returned in a DataFrame.
    """
    
    # load in default values
    logger.info(f"Running Cacti with the following parameters: filename: {filename}, cacheSize: {cacheSize}, blockSize: {blockSize}, cache_type: {cache_type}, bus_width: {bus_width}, transistor_size: {transistor_size}, addr_timing: {addr_timing}, force_cache_config: {force_cache_config}, technology: {technology}")
    # with open(os.path.normpath(os.path.join(os.path.dirname(__file__), 'params/cacti_input.yaml')), "r") as yamlfile:
    #     config_values = yaml.safe_load(yamlfile)

    config_values = read_config_file(os.path.join(CACTI_DIR, f"{filename}.cfg"))
    print(config_values)

    # If user doesn't give input, default to cacti_input vals
    if cacheSize == None:
        cacheSize = config_values["cache_size"]

    if blockSize == None:
        blockSize = config_values["block_size"]

    if cache_type == None:
        cache_type = config_values["cache_type"]
    
    if cache_type == "cache":
        associativity = 0
        num_search_ports = 1
    else:
        associativity = config_values["associativity"]
        num_search_ports = config_values["num_search_ports"]
        

    if bus_width == None:
        bus_width = config_values["bus_width"]

    if cache_type == "main memory":
        mem_bus_width = bus_width
    else:
        mem_bus_width = config_values["mem_data_width"]

    if transistor_size == None:
        transistor_size = config_values["transistor_size"]
    else:
        transistor_size = min(valid_tech_nodes, key=lambda x: abs(transistor_size - x))

    if addr_timing == None:
        addr_timing = config_values["addr_timing"]

    if force_cache_config == None:
        force_cache_config = config_values["force_cache_config"]

    # lines written to [filename].cfg file
    cfg_lines = [
        "# Cache size",
        "-size (bytes) {}".format(cacheSize),
        "",
        "# power gating",
        '-Array Power Gating - "{}"'.format(config_values["Array_Power_Gating"]),
        '-WL Power Gating - "{}"'.format(config_values["WL_Power_Gating"]),
        '-CL Power Gating - "{}"'.format(config_values["CL_Power_Gating"]),
        '-Bitline floating - "{}"'.format(config_values["Bitline_floating"]),
        '-Interconnect Power Gating - "{}"'.format(
            config_values["Interconnect_Power_Gating"]
        ),
        '-Power Gating Performance Loss "{}"'.format(
            config_values["Power_Gating_Performance_Loss"]
        ),
        "",
        "# Line size",
        "-block size (bytes) {}".format(blockSize),
        "",
        "# To model Fully Associative cache, set associativity to zero",
        "-associativity {}".format(associativity),
        "",
        "-read-write port {}".format(config_values["read_write_port"]),
        "-exclusive read port {}".format(config_values["exclusive_read_port"]),
        "-exclusive write port {}".format(config_values["exclusive_write_port"]),
        "-single ended read ports {}".format(config_values["single_ended_read_ports"]),
        "-search port {}".format(num_search_ports),
        "",
        "# Multiple banks connected using a bus",
        "-UCA bank count {}".format(config_values["UCA_bank_count"]),
        "-technology (u) {}".format(transistor_size),
        "",
        "# following three parameters are meaningful only for main memories",
        "-page size (bits) {}".format(config_values["page_size"]),
        "-burst length {}".format(config_values["burst_length"]),
        "-internal prefetch width {}".format(config_values["internal_prefetch_width"]),
        "",
        "# following parameter can have one of five values",
        '-Data array cell type - "{}"'.format(config_values["Data_array_cell_type"]),
        "",
        "# following parameter can have one of three values",
        '-Data array peripheral type - "{}"'.format(
            config_values["Data_array_peripheral_type"]
        ),
        "",
        "# following parameter can have one of five values",
        '-Tag array cell type - "{}"'.format(config_values["Tag_array_cell_type"]),
        "",
        "# following parameter can have one of three values",
        '-Tag array peripheral type - "{}"'.format(
            config_values["Tag_array_peripheral_type"]
        ),
        "",
        "# Bus width include data bits and address bits required by the decoder",
        "-output/input bus width {}".format(bus_width),
        "",
        "# 300-400 in steps of 10",
        "-operating temperature (K) {}".format(config_values["operating_temperature"]),
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
            config_values["access_mode"]
        ),
        "",
        "# DESIGN OBJECTIVE for UCA (or banks in NUCA)",
        "-design objective (weight delay, dynamic power, leakage power, cycle time, area) {}".format(
            config_values[
                "design_objective"
            ]
        ),
        "",
        "# Percentage deviation from the minimum value",
        "-deviate (delay, dynamic power, leakage power, cycle time, area) {}".format(
            config_values["deviate"]
        ),
        "",
        "# Objective for NUCA",
        "-NUCAdesign objective (weight delay, dynamic power, leakage power, cycle time, area) {}".format(
            config_values[
                "NUCAdesign_objective"
            ]
        ),
        "-NUCAdeviate (delay, dynamic power, leakage power, cycle time, area) {}".format(
            config_values[
                "NUCAdeviate"
            ]
        ),
        "",
        "# Set optimize tag to ED or ED^2 to obtain a cache configuration optimized for",
        "# energy-delay or energy-delay sq. product",
        "# Note: Optimize tag will disable weight or deviate values mentioned above",
        "# Set it to NONE to let weight and deviate values determine the",
        "# appropriate cache configuration",
        '-Optimize ED or ED^2 (ED, ED^2, NONE): "{}"'.format(
            config_values["Optimize_ED_or_ED^2"]
        ),
        '-Cache model (NUCA, UCA)  - "{}"'.format(
            config_values["Cache_model_NUCA_UCA"]
        ),
        "",
        "# In order for CACTI to find the optimal NUCA bank value the following",
        "# variable should be assigned 0.",
        "-NUCA bank count {}".format(config_values["NUCA_bank_count"]),
        "",
        "# Wire signaling",
        '-Wire signaling (fullswing, lowswing, default) - "{}"'.format(
            config_values["Wire_signaling"]
        ),
        '-Wire inside mat - "{}"'.format(config_values["Wire_inside_mat"]),
        '-Wire outside mat - "{}"'.format(config_values["Wire_outside_mat"]),
        '-Interconnect projection - "{}"'.format(
            config_values["Interconnect_projection"]
        ),
        "",
        "# Contention in network",
        "-Core count {}".format(config_values["Core_count"]),
        '-Cache level (L2/L3) - "{}"'.format(config_values["Cache_level"]),
        '-Add ECC - "{}"'.format(config_values["Add_ECC"]),
        '-Print level (DETAILED, CONCISE) - "{}"'.format(config_values["Print_level"]),
        "",
        "# for debugging",
        '-Print input parameters - "{}"'.format(
            config_values["Print_input_parameters"]
        ),
        "# force CACTI to model the cache with the",
        "# following Ndbl, Ndwl, Nspd, Ndsam,",
        "# and Ndcm values",
        '-Force cache config - "{}"'.format(force_cache_config),
        "-Ndwl {}".format(config_values["Ndwl"]),
        "-Ndbl {}".format(config_values["Ndbl"]),
        "-Nspd {}".format(config_values["Nspd"]),
        "-Ndcm {}".format(config_values["Ndcm"]),
        "-Ndsam1 {}".format(config_values["Ndsam1"]),
        "-Ndsam2 {}".format(config_values["Ndsam2"]),
        "",
        "#### Default CONFIGURATION values for baseline external IO parameters to DRAM. More details can be found in the CACTI-IO technical report (), especially Chapters 2 and 3.",
        "# Memory Type",
        '-dram_type "{}"'.format(config_values["dram_type"]),
        "# Memory State",
        '-io state "{}"'.format(config_values["io_state"]),
        "# Address bus timing",
        "-addr_timing {}".format(addr_timing),
        "# Memory Density",
        "-mem_density {}".format(config_values["mem_density"]),
        "# IO frequency",
        "-bus_freq {}".format(config_values["bus_freq"]),
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
        '-DIMM model "{}"'.format(config_values["DIMM_model"]),
        "# If channels of each bob have the same configurations",
        '-mirror_in_bob "{}"'.format(config_values["mirror_in_bob"]),
        "# if we want to see all channels/bobs/memory configurations explored",
        '# -verbose "T"',
        '# -verbose "F"',
    ]

    cactiDir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'cacti'))
    # write file
    input_filename = filename + ".cfg"
    cactiInput = os.path.join(CACTI_DIR, input_filename)
    with open(cactiInput, 'w') as file:
        for line in cfg_lines:
            file.write(line + '\n')

    stdout_filename = "cacti_stdout.log"
    stdout_file_path = os.path.join(CACTI_DIR, stdout_filename)

    stderr_filename = "cacti_stderr.log"
    stderr_file_path = os.path.join(cactiDir, stderr_filename)

    cmd = ['./cacti', '-infile', input_filename]
    
    with open(stdout_file_path, "w") as f:
        p = subprocess.Popen(cmd, cwd=CACTI_DIR, stdout=f, stderr=subprocess.PIPE)
    
    p.wait()
    if p.returncode != 0:
        with open(stdout_file_path, "r") as f:
            raise Exception(f"Cacti Error in {filename}", {p.stderr.read().decode()}, {f.read().split("\n")[-2]})

    output_filename = filename + ".cfg.out"
    cactiOutput = os.path.normpath(os.path.join(CACTI_DIR, output_filename))
    output_data = pd.read_csv(cactiOutput, sep=", ", engine='python')
    output_data = output_data.iloc[-1] # get just the last row which is the most recent run

    IO_freq = convert_frequency(config_values['bus_freq'])
    IO_latency = (addr_timing / IO_freq)

    output_data["IO latency (s)"] = IO_latency

    # CACTI: access time (ns), search energy (nJ), read energy (nJ), write energy (nJ), leakage bank power (mW)
    # CACTI IO: area (sq.mm), timing (ps), dynamic power (mW), PHY power (mW), termination and bias power (mW)
    # latency (ns)
    return output_data

def run_existing_cacti_cfg(filename):
    """
    Retrieves timing and power values from a Cacti run of an existing .cfg file.

    Inputs:
    filename : str
        Name of the existing .cfg file to run in Cacti.

    Returns:
    pd.DataFrame
        DataFrame containing timing, power, and IO-related values from the Cacti run.

    Outputs:
    Cacti run results are returned in a DataFrame after executing the existing .cfg file.
    """

    cactiDir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'cacti'))
    print(f"cactiDir: {CACTI_DIR}")
    print(f"filename: {filename}")

    # write file
    input_filename = filename.replace("src/cacti/", "")
    print(f"input_filename: {input_filename}")

    stdout_filename = "cacti_stdout.log"
    stdout_file_path = os.path.join(CACTI_DIR, stdout_filename)

    cmd = ['./cacti', '-infile', input_filename]
    
    with open(stdout_file_path, "w") as f:
        p = subprocess.Popen(cmd, cwd=CACTI_DIR, stdout=f, stderr=subprocess.PIPE)
    
    p.wait()
    if p.returncode != 0:
        with open(stdout_file_path, "r") as f:
            raise Exception(f"Cacti Error in {filename}", {p.stderr.read().decode()}, {f.read().split("\n")[-2]})

    print(cmd)

    output_filename = input_filename + ".out"
    cactiOutput = os.path.join(CACTI_DIR, output_filename)
    output_data = pd.read_csv(cactiOutput, sep=", ", engine='python')
    output_data = output_data.iloc[-1] # get just the last row which is the most recent run

    # get IO params
    bus_freq = None
    addr_timing = None
    cacti_input_filename = os.path.join(CACTI_DIR, input_filename)

    print(f"cacti_input_filename: {cacti_input_filename}")

    with open(cacti_input_filename, 'r') as file:
        for line in file:
            if "-bus_freq" in line:
                bus_freq = line.split()[1] + " " + line.split()[2]
            elif "-addr_timing" in line:
                addr_timing = line.split()[1]

    IO_freq = convert_frequency(bus_freq)
    IO_latency = (float(addr_timing) / IO_freq)

    output_data["IO latency (s)"] = IO_latency

    # CACTI: access time (ns), search energy (nJ), read energy (nJ), write energy (nJ), leakage bank power (mW)
    # CACTI IO: area (sq.mm), timing (ps), dynamic power (mW), PHY power (mW), termination and bias power (mW)
    # latency (ns)
    print(f"exiting run_existing_cacti_cfg")
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
    
    if len(parts) == 2 and parts[1].lower() in ('ghz', 'mhz'):
        try:
            value = int(parts[0])  # Extract the integer part
            unit = parts[1].lower()  # Extract the unit and convert to lowercase

            if unit == 'ghz':
                return value * 10**9
            elif unit == 'mhz':
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
    cur_dir = os.getcwd()
    if os.path.basename(cur_dir) == 'codesign':
        # Change to the 'src' directory
        if not dat_file_path.startswith("src/"):
            dat_file_path = "src/" + dat_file_path

    with open(dat_file_path, 'r') as file:
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
            lines[i] = f"-{key} ({unit_label}) " + " ".join([str(new_value)] * len(original_values[i])) + "\n"

    with open(dat_file_path, 'w') as file:
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

    cur_dir = os.getcwd()
    if os.path.basename(cur_dir) == 'codesign':
        # Change to the 'src' directory
        dat_file_path = "src/" + dat_file_path
    
    with open(dat_file_path, 'r') as file:
        lines = file.readlines()
    
    for i, values in original_values.items():
        parts = lines[i].split()
        # Preserve the key and unit label
        key_and_unit = " ".join(parts[:2])
        # Replace the rest with the original values
        lines[i] = f"{key_and_unit} " + " ".join(values) + "\n"

    with open(dat_file_path, 'w') as file:
        file.writelines(lines)

# Function to read the file and extract values
def read_config_file(file_path: str):
    # Initialize dictionary to store extracted values
    config_dict = {}

    # Define regular expressions for each pattern in the file
    patterns = {
        "cache_size": re.compile(r"-size \(bytes\) (\d+)"),
        "Array_Power_Gating": re.compile(r'-Array Power Gating - "([^"]*)"'),
        "WL_Power_Gating": re.compile(r'-WL Power Gating - "([^"]*)"'),
        "CL_Power_Gating": re.compile(r'-CL Power Gating - "([^"]*)"'),
        "Bitline_floating": re.compile(r'-Bitline floating - "([^"]*)"'),
        "Interconnect_Power_Gating": re.compile(r'-Interconnect Power Gating - "([^"]*)"'),
        "Power_Gating_Performance_Loss": re.compile(r'-Power Gating Performance Loss "([^"]*)"'),
        "block_size": re.compile(r"-block size \(bytes\) (\d+)"),
        "associativity": re.compile(r"-associativity (\d+)"),
        "read_write_port": re.compile(r"-read-write port (\d+)"),
        "exclusive_read_port": re.compile(r"-exclusive read port (\d+)"),
        "exclusive_write_port": re.compile(r"-exclusive write port (\d+)"),
        "single_ended_read_ports": re.compile(r"-single ended read ports (\d+)"),
        "num_search_ports": re.compile(r"-search port (\d+)"),
        "UCA_bank_count": re.compile(r"-UCA bank count (\d+)"),
        "transistor_size": re.compile(r"-technology \(u\) (\d+\.?\d*)"),
        "page_size": re.compile(r"-page size \(bits\) (\d+)"),
        "burst_length": re.compile(r"-burst length (\d+)"),
        "internal_prefetch_width": re.compile(r"-internal prefetch width (\d+)"),
        "Data_array_cell_type": re.compile(r'-Data array cell type - "([^"]*)"'),
        "Data_array_peripheral_type": re.compile(r'-Data array peripheral type - "([^"]*)"'),
        "Tag_array_cell_type": re.compile(r'-Tag array cell type - "([^"]*)"'),
        "Tag_array_peripheral_type": re.compile(r'-Tag array peripheral type - "([^"]*)"'),
        "bus_width": re.compile(r"-output/input bus width (\d+)"),
        "operating_temperature": re.compile(r"-operating temperature \(K\) (\d+)"),
        "cache_type": re.compile(r'-cache type "([^"]*)"'),
        "tag_size": re.compile(r'-tag size \(b\) "([^"]*)"'),
        "access_mode": re.compile(r'-access mode \(normal, sequential, fast\) - "([^"]*)"'),
        "design_objective": re.compile(r"-design objective \(weight delay, dynamic power, leakage power, cycle time, area\) ([^,]+)"),
        "deviate": re.compile(r"-deviate \(delay, dynamic power, leakage power, cycle time, area\) ([^,]+)"),
        "NUCAdesign_objective": re.compile(r"-NUCAdesign objective \(weight delay, dynamic power, leakage power, cycle time, area\) ([^,]+)"),
        "NUCAdeviate": re.compile(r"-NUCAdeviate \(delay, dynamic power, leakage power, cycle time, area\) ([^,]+)"),
        "Optimize_ED_or_ED^2": re.compile(r'-Optimize ED or ED\^2 \(ED, ED\^2, NONE\): "([^"]*)"'),
        "Cache_model_NUCA_UCA": re.compile(r'-Cache model \(NUCA, UCA\)  - "([^"]*)"'),
        "NUCA_bank_count": re.compile(r"-NUCA bank count (\d+)"),
        "Wire_signaling": re.compile(r'-Wire signaling \(fullswing, lowswing, default\) - "([^"]*)"'),
        "Wire_inside_mat": re.compile(r'-Wire inside mat - "([^"]*)"'),
        "Wire_outside_mat": re.compile(r'-Wire outside mat - "([^"]*)"'),
        "Interconnect_projection": re.compile(r'-Interconnect projection - "([^"]*)"'),
        "Core_count": re.compile(r"-Core count (\d+)"),
        "Cache_level": re.compile(r'-Cache level \(L2/L3\) - "([^"]*)"'),
        "Add_ECC": re.compile(r'-Add ECC - "([^"]*)"'),
        "Print_level": re.compile(r'-Print level \(DETAILED, CONCISE\) - "([^"]*)"'),
        "Print_input_parameters": re.compile(r'-Print input parameters - "([^"]*)"'),
        "force_cache_config": re.compile(r'-Force cache config - "([^"]*)"'),
        "Ndwl": re.compile(r"-Ndwl (\d+)"),
        "Ndbl": re.compile(r"-Ndbl (\d+)"),
        "Nspd": re.compile(r"-Nspd (\d+)"),
        "Ndcm": re.compile(r"-Ndcm (\d+)"),
        "Ndsam1": re.compile(r"-Ndsam1 (\d+)"),
        "Ndsam2": re.compile(r"-Ndsam2 (\d+)"),
        "dram_type": re.compile(r'-dram_type "([^"]*)"'),
        "io_state": re.compile(r'-io state "([^"]*)"'),
        "addr_timing": re.compile(r"-addr_timing (\d+)"),
        "mem_density": re.compile(r"-mem_density (\d+)"),
        "bus_freq": re.compile(r"-bus_freq (\d+)"),
        "duty_cycle": re.compile(r"-duty_cycle (\d+\.?\d*)"),
        "activity_dq": re.compile(r"-activity_dq (\d+\.?\d*)"),
        "activity_ca": re.compile(r"-activity_ca (\d+\.?\d*)"),
        "num_dq": re.compile(r"-num_dq (\d+)"),
        "num_dqs": re.compile(r"-num_dqs (\d+)"),
        "num_ca": re.compile(r"-num_ca (\d+)"),
        "num_clk": re.compile(r"-num_clk (\d+)"),
        "num_mem_dq": re.compile(r"-num_mem_dq (\d+)"),
        "mem_data_width": re.compile(r"-mem_data_width (\d+)"),
        "rtt_value": re.compile(r"-rtt_value (\d+)"),
        "ron_value": re.compile(r"-ron_value (\d+)"),
        "num_bobs": re.compile(r"-num_bobs (\d+)"),
        "capacity": re.compile(r"-capacity (\d+\.?\d*)"),
        "num_channels_per_bob": re.compile(r"-num_channels_per_bob (\d+)"),
        "first_metric": re.compile(r'-first metric "([^"]*)"'),
        "second_metric": re.compile(r'-second metric "([^"]*)"'),
        "third_metric": re.compile(r'-third metric "([^"]*)"'),
        "DIMM_model": re.compile(r'-DIMM model "([^"]*)"'),
        "mirror_in_bob": re.compile(r'-mirror_in_bob "([^"]*)"')
    }

    # Open the file and read it line by line
    with open(file_path, "r") as file:
        for line in file:
            # Check each pattern and extract values
            for key, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    # Convert to appropriate data type
                    if key in [
                        "cache_size",
                        "block_size",
                        "associativity",
                        "read_write_port",
                        "exclusive_read_port",
                        "exclusive_write_port",
                        "single_ended_read_ports",
                        "num_search_ports",
                        "UCA_bank_count",
                        "bus_width",
                        "operating_temperature",
                        "Core_count",
                        "Ndwl",
                        "Ndbl",
                        "Nspd",
                        "Ndcm",
                        "Ndsam1",
                        "Ndsam2",
                        "addr_timing",
                        "mem_density",
                        "bus_freq",
                        "num_dq",
                        "num_dqs",
                        "num_ca",
                        "num_clk",
                        "num_mem_dq",
                        "mem_data_width",
                        "rtt_value",
                        "ron_value",
                        "num_bobs",
                        "num_channels_per_bob",
                    ]:
                        config_dict[key] = int(match.group(1))  # Convert to integer
                    elif key in [
                        "transistor_size",
                        "Power_Gating_Performance_Loss",
                        "duty_cycle",
                        "activity_dq",
                        "activity_ca",
                        "capacity",
                    ]:
                        config_dict[key] = float(match.group(1))  # Convert to float
                    else:
                        config_dict[key] = match.group(1)  # Keep as string
                    break  # Stop checking other patterns if a match is found

    return config_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a config file and a data file.")
    parser.add_argument('-cfg_name', type=str, default='cache', help="Path to the configuration file (default: mem_validate_cache)")
    parser.add_argument("-adjust", type=str, default="false", help="Boolean flag to detail adjust cfg through arguments")
    parser.add_argument('-dat_file', type=str, default='src/cacti/tech_params/90nm.dat', help="Path to the data file (default: src/cacti/tech_params/90nm.dat)")
    parser.add_argument('-cacheSize', type=int, default=131072, help="Path to the data file (default: 131072)")
    parser.add_argument('-blockSize', type=int, default=64, help="Path to the data file (default: 64)")
    parser.add_argument('-cacheType', type=str, default="main memory", help="Path to the data file (default: main memory)")
    parser.add_argument('-busWidth', type=int, default=64, help="Path to the data file (default: 64)")

    args = parser.parse_args()
    cache_cfg = f"src/cacti/cfg/{args.cfg_name}.cfg"

    adjust = args.adjust.lower() == "true"  
    if adjust:
        buf_vals = gen_vals(
            args.cfg_name,
            cacheSize=args.cacheSize, 
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
