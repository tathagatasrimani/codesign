import os
import subprocess
import yaml
import argparse
import logging
logger = logging.getLogger(__name__)

import pandas as pd

from cacti.cacti_python.parameter import g_ip
from cacti.cacti_python.parameter import g_tp
from cacti.cacti_python.cacti_interface import uca_org_t
from cacti.cacti_python.Ucache import *
from cacti.cacti_python.parameter import sympy_var

from cacti.cacti_python.mat import Mat
from cacti.cacti_python.bank import Bank

import cacti.cacti_python.get_dat as dat

from hw_symbols import *
import sympy as sp

valid_tech_nodes = [0.022, 0.032, 0.045, 0.065, 0.090, 0.180]

'''
Generate sympy expression for access_time (will add for energy)
Outputs results to text file
'''
def cacti_gen_sympy(name, cache_cfg, opt_vals, use_piecewise=True):
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

    with open(f'{name + "_access_time"}.txt', 'w') as file:
        file.write(str(fin_res.access_time))

    with open(f'{name + "_read_dynamic"}.txt', 'w') as file:
        file.write(str(fin_res.power.readOp.dynamic))

    with open(f'{name + "_write_dynamic"}.txt', 'w') as file:
        file.write(str(fin_res.power.writeOp.dynamic))

    with open(f'{name + "_read_leakage"}.txt', 'w') as file:
        file.write(str(fin_res.power.readOp.leakage))

    IO_info = {
        "io_area": fin_res.io_area,
        "io_timing_margin": fin_res.io_timing_margin,
        "io_dynamic_power": fin_res.io_dynamic_power,
        "io_phy_power": fin_res.io_phy_power,
        "io_termination_power": fin_res.io_termination_power
    }
    return IO_info


'''
Validates output of sympy_file with cacti run.
'''
def validate(sympy_file, cache_cfg, dat_file):
    # initalize input parameters from .cfg
    g_ip.parse_cfg(cache_cfg)
    g_ip.error_checking()

    # TODO there are seperate tech params for each Type (Device, Memory, Interconnect)
    tech_params = {}
    dat.scan_dat(tech_params, dat_file, g_ip.data_arr_ram_cell_tech_type, g_ip.data_arr_ram_cell_tech_type, g_ip.temp)
    tech_params = {k: (10**(-9) if v == 0 else v) for k, v in tech_params.items() if v is not None and not math.isnan(v)}
    print(tech_params)
    
    print(f'READING {sympy_file}')
    with open(sympy_file, 'r') as file:
        expression_str = file.read()

    expression = sp.sympify(expression_str)
    # print(expression)
    result = expression.subs(tech_params)
    
    result = result.subs(sp.I, 0)


    validate_vals = gen_vals(
        "validate_mem_cache",
        cacheSize=g_ip.cache_sz, # TODO: Add in buffer sizing
        blockSize=g_ip.block_sz,
        cache_type="cache",
        bus_width=g_ip.out_w,
        transistor_size=g_ip.F_sz_um,
        force_cache_config="false",
    )

    print(f'Transistor size: {g_ip.F_sz_um}')

    print(f"result : {result}")
    # print(f"validate_vals {validate_vals}")
    validate_access_time = float(validate_vals["Access time (ns)"])
    validate_read_dynamic = float(validate_vals["Dynamic read energy (nJ)"])
    validate_write_dynamic = float(validate_vals["Dynamic write energy (nJ)"])
    validate_leakage = float(validate_vals["Standby leakage per bank(mW)"])
    print(f"validate_access_time (ns): {validate_access_time}")
    print(f"validate_read_dynamic (nJ): {validate_read_dynamic}")
    print(f"validate_write_dynamic (nJ): {validate_write_dynamic}")
    print(f"validate_leakage (mW): {validate_leakage}")
    
    return result, validate_access_time

def validate_energy(sympy_file, cache_cfg, dat_file, IO_info):
    # initalize input parameters from .cfg
    g_ip.parse_cfg(cache_cfg)
    g_ip.error_checking()

    # TODO there are seperate tech params for each Type (Device, Memory, Interconnect)
    tech_params = {}
    dat.scan_dat(tech_params, dat_file, g_ip.data_arr_ram_cell_tech_type, g_ip.data_arr_ram_cell_tech_type, g_ip.temp)
    tech_params = {k: (10**(-9) if v == 0 else v) for k, v in tech_params.items() if v is not None and not math.isnan(v)}
    print(tech_params)
    
    sympy_filename = sympy_file.rstrip(".txt")
    print(f'READING {sympy_filename}')
    sympy_file_access_time = sympy_filename + "_access_time.txt"
    sympy_file_read_dynamic = sympy_filename + "_read_dynamic.txt"
    sympy_file_write_dynamic = sympy_filename + "_write_dynamic.txt"
    sympy_file_read_leakage = sympy_filename + "_read_leakage.txt"

    print(f'{sympy_file_read_dynamic, sympy_file_write_dynamic, sympy_file_read_leakage}')
    
    # with open(sympy_file, 'r') as file:
    #     expression_str = file.read()

    # expression = sp.sympify(expression_str)
    # # print(expression)
    # result = expression.subs(tech_params)
    
    # result = result.subs(sp.I, 0)

    with open(sympy_file_access_time, 'r') as file:
        expression_str = file.read()

    expression = sp.sympify(expression_str)
    # print(expression)
    result = expression.subs(tech_params)
    
    result_access_time = result.subs(sp.I, 0)
    result_access_time = result_access_time.evalf()

    with open(sympy_file_read_dynamic, 'r') as file:
        expression_str = file.read()

    expression = sp.sympify(expression_str)
    # print(expression)
    result = expression.subs(tech_params)
    
    result_read_dynamic = result.subs(sp.I, 0)

    with open(sympy_file_write_dynamic, 'r') as file:
        expression_str = file.read()

    expression = sp.sympify(expression_str)
    # print(expression)
    result = expression.subs(tech_params)
    
    result_write_dynamic = result.subs(sp.I, 0)

    with open(sympy_file_read_leakage, 'r') as file:
        expression_str = file.read()

    expression = sp.sympify(expression_str)
    # print(expression)
    result = expression.subs(tech_params)
    
    result_read_leakage = result.subs(sp.I, 0)

    validate_vals = gen_vals(
        "validate_mem_energy_cache",
        cacheSize=g_ip.cache_sz, # TODO: Add in buffer sizing
        blockSize=g_ip.block_sz,
        cache_type="main memory",
        bus_width=g_ip.out_w,
        transistor_size=g_ip.F_sz_um,
        force_cache_config="false",
    )

    print(f'Transistor size: {g_ip.F_sz_um}')
    print(f'is_cache: {g_ip.is_cache}')

    # print
    print(f'access_time: {result_access_time}')
    print(f"result : {result_read_dynamic, result_write_dynamic, result_read_leakage}")
    
    print(f"io_area: {IO_info['io_area']}")
    print(f"io_timing_margin: {IO_info['io_timing_margin']}")
    print(f"io_dynamic_power: {IO_info['io_dynamic_power']}")
    print(f"io_phy_power: {IO_info['io_phy_power']}")
    print(f"io_termination_power: {IO_info['io_termination_power']}")

    # print(f"validate_vals {validate_vals}")
    validate_access_time = float(validate_vals["Access time (ns)"])
    validate_read_dynamic = float(validate_vals["Dynamic read energy (nJ)"])
    validate_write_dynamic = float(validate_vals["Dynamic write energy (nJ)"])
    validate_leakage = float(validate_vals["Standby leakage per bank(mW)"])
    print(f"validate_access_time (ns): {validate_access_time}")
    print(f"validate_read_dynamic (nJ): {validate_read_dynamic}")
    print(f"validate_write_dynamic (nJ): {validate_write_dynamic}")
    print(f"validate_leakage (mW): {validate_leakage}")

    # write to CSV
    data = {
        "access_time (ns)": [result_access_time],
        "result_read_dynamic (nJ)": [result_read_dynamic],
        "result_write_dynamic (nJ)": [result_write_dynamic],
        "result_leakage (mW)": [result_read_leakage],
        "result_io_area": [IO_info['io_area']],
        "result_io_timing_margin": [IO_info['io_timing_margin']],
        "result_io_dynamic_power": [IO_info['io_dynamic_power']],
        "result_io_phy_power": [IO_info['io_phy_power']],
        "result_io_termination_power": [IO_info['io_termination_power']],
        "validate_access_time (ns)": [float(validate_vals["Access time (ns)"])],
        "validate_read_dynamic (nJ)": [float(validate_vals["Dynamic read energy (nJ)"])],
        "validate_write_dynamic (nJ)": [float(validate_vals["Dynamic write energy (nJ)"])],
        "validate_leakage (mW)": [float(validate_vals["Standby leakage per bank(mW)"])],
        "validate_io_area": [float(validate_vals["IO area"])],
        "validate_io_timing": [float(validate_vals["IO timing"])],
        "validate_io_power_dynamic": [float(validate_vals["IO power dynamic"])],
        "validate_io_power_phy": [float(validate_vals["IO power PHY"])],
        "validate_io_power_termination_and_bias": [float(validate_vals["IO power termination and bias"])],
        "transistor_size (um)": [g_ip.F_sz_um],
        "is_cache": [g_ip.is_cache]
    }

    df = pd.DataFrame(data)

    directory = "cacti_plot"
    if not os.path.exists(directory):
        os.makedirs(directory)
    csv_file = os.path.join(directory, "validate_results.csv")
    
    file_exists = os.path.isfile(csv_file)
    df.to_csv(csv_file, mode='a', header=not file_exists, index=False)

    print(f"Data successfully appended to {csv_file}")
    
    return result, validate_access_time

"""
Generates Cacti .cfg file based on input and cacti_input.
Feeds .cfg into Cacti and runs.
Retrieves timing and power values from Cacti run.
"""
def gen_vals(filename = "base_cache", cacheSize = None, blockSize = None,
             cache_type = None, bus_width = None, transistor_size = None,
             addr_timing = None, force_cache_config = None, technology = None,
             debug = False) -> pd.DataFrame:
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

    cactiDir = os.path.join(os.path.dirname(__file__), 'cacti')

    # write file
    input_filename = filename + ".cfg"
    cactiInput = os.path.join(cactiDir, input_filename)
    with open(cactiInput, 'w') as file:
        for line in cfg_lines:
            file.write(line + '\n')

    cmd = ['./cacti', '-infile', input_filename]

    p = subprocess.Popen(cmd, cwd=cactiDir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    if p.returncode != 0:
        #raise Exception(f"Cacti Error in {filename}", {p.stderr.read().decode()}, {p.stdout.read().decode().split("\n")[-2]})
        print(f"Cacti Error in {filename}", {p.stderr.read().decode()}, {p.stdout.read().decode().split("\n")[-2]})

    output_filename = filename + ".cfg.out"
    cactiOutput = os.path.join(cactiDir, output_filename)
    output_data = pd.read_csv(cactiOutput, sep=", ", engine='python')
    output_data = output_data.iloc[-1] # get just the last row which is the most recent run

    IO_freq = convert_frequency(config_values['bus_freq'])
    IO_latency = (addr_timing / IO_freq)

    output_data["IO latency (s)"] = IO_latency

    # CACTI: access time (ns), search energy (nJ), read energy (nJ), write energy (nJ), leakage bank power (mW)
    # CACTI IO: area (sq.mm), timing (ps), dynamic power (mW), PHY power (mW), termination and bias power (mW)
    # latency (ns)
    return output_data

# for debugging
# if __name__ == '__main__':
# print(gen_vals("test0"))
# print(gen_vals("test1", 131072, 64, "cache", 512, 4.0))

def convert_frequency(string):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a config file and a data file.")
    parser.add_argument('-cfg_name', type=str, default='mem_validate_cache', help="Path to the configuration file (default: mem_validate_cache)")
    parser.add_argument('-dat_file', type=str, default='cacti/tech_params/90nm.dat', help="Path to the data file (default: cacti/tech_params/90nm.dat)")
    parser.add_argument('-cacheSize', type=int, default=131072, help="Path to the data file (default: 131072)")
    parser.add_argument('-blockSize', type=int, default=64, help="Path to the data file (default: 64)")
    parser.add_argument('-cacheType', type=str, default="main memory", help="Path to the data file (default: main memory)")
    parser.add_argument('-busWidth', type=int, default=64, help="Path to the data file (default: 64)")

    args = parser.parse_args()

    buf_vals = gen_vals(
        args.cfg_name,
        cacheSize=args.cacheSize, # TODO: Add in buffer sizing
        blockSize=args.blockSize,
        cache_type=args.cacheType,
        bus_width=args.busWidth,
    )
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


    cache_cfg = f"cacti/{args.cfg_name}.cfg"
    IO_info = cacti_gen_sympy("sympy_validate", cache_cfg, buf_opt, use_piecewise=False)
    sympy_file = "sympy_validate.txt"
    dat_file = f"{args.dat_file}"

    # import time
    # print(f'cache_cfg {cache_cfg}')
    # print(f'dat_file {dat_file}')
    # print(f'args {args}')
    # time.sleep(20)

    validate_energy(sympy_file, cache_cfg, dat_file, IO_info)

