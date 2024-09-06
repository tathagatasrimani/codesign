import os
import subprocess
import yaml
import argparse
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

def cacti_gen_sympy(name, cache_cfg, opt_vals, use_piecewise=True):
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

    g_ip.parse_cfg(cache_cfg)
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

    fin_res = uca_org_t()
    fin_res = solve_single()

    # Create the directory path
    output_dir = os.path.join('src', 'cacti', 'sympy')

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
    with open("params/cacti_input.yaml", "r") as yamlfile:
        config_values = yaml.safe_load(yamlfile)

    if cache_type == None:
        cache_type = config_values["cache_type"]

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
        bus_width = config_values["output/input_bus_width"]

    if cache_type == "main memory":
        mem_bus_width = bus_width
    else:
        mem_bus_width = config_values["mem_data_width"]

    if transistor_size == None:
        transistor_size = config_values["technology"]
    else:
        transistor_size = min(valid_tech_nodes, key=lambda x: abs(transistor_size - x))

    if addr_timing == None:
        addr_timing = config_values["addr_timing"]

    if force_cache_config == None:
        force_cache_config = config_values["Force_cache_config"]

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
                "design_objective_weight_delay_dynamic_power_leakage_power_cycle_time_area"
            ]
        ),
        "",
        "# Percentage deviation from the minimum value",
        "-deviate (delay, dynamic power, leakage power, cycle time, area) {}".format(
            config_values["deviate_delay_dynamic_power_leakage_power_cycle_time_area"]
        ),
        "",
        "# Objective for NUCA",
        "-NUCAdesign objective (weight delay, dynamic power, leakage power, cycle time, area) {}".format(
            config_values[
                "NUCAdesign_objective_weight_delay_dynamic_power_leakage_power_cycle_time_area"
            ]
        ),
        "-NUCAdeviate (delay, dynamic power, leakage power, cycle time, area) {}".format(
            config_values[
                "NUCAdeviate_delay_dynamic_power_leakage_power_cycle_time_area"
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
    cactiInput = os.path.join(cactiDir, input_filename)
    with open(cactiInput, 'w') as file:
        for line in cfg_lines:
            file.write(line + '\n')

    stdout_filename = "cacti_stdout.log"
    stdout_file_path = os.path.join(cactiDir, stdout_filename)

    cmd = ['./cacti', '-infile', input_filename]

    p = subprocess.Popen(cmd, cwd=cactiDir) #, stdout=stdout_file_path, stderr=subprocess.PIPE)
    p.wait()
    if p.returncode != 0:
        raise Exception(f"Cacti Error in {filename}", {p.stderr.read().decode()}, {p.stdout.read().decode().split("\n")[-2]})

    output_filename = filename + ".cfg.out"
    cactiOutput = os.path.normpath(os.path.join(cactiDir, output_filename))
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

    # write file
    input_filename = filename.replace("src/cacti/", "")
    print(input_filename)
    cmd = ['./cacti', '-infile', input_filename]

    print(cmd)

    p = subprocess.Popen(cmd, cwd=cactiDir) #, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    if p.returncode != 0:
        raise Exception(f"Cacti Error in {filename}", {p.stderr.read().decode()}, {p.stdout.read().decode().split("\n")[-2]})

    output_filename = input_filename + ".out"
    cactiOutput = os.path.join(cactiDir, output_filename)
    output_data = pd.read_csv(cactiOutput, sep=", ", engine='python')
    output_data = output_data.iloc[-1] # get just the last row which is the most recent run

    # get IO params
    bus_freq = None
    addr_timing = None
    cacti_input_filename = "src/cacti/" + input_filename

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
    IO_info = cacti_gen_sympy(sympy_file, cache_cfg, buf_opt, use_piecewise=False)



