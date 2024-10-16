import os
import subprocess
import argparse
import sys
import logging
logger = logging.getLogger(__name__)

import yaml
import pandas as pd
import sympy as sp

from src.cacti.cacti_python.parameter import g_ip
import src.cacti.cacti_python.get_dat as dat
from src import cacti_util

from src.cacti import TRANSISTOR_SIZES, CACTI_DIR
from src.cacti.cacti_python.parameter import g_tp
from src.cacti.cacti_python.Ucache import *
import src.cacti.cacti_python.get_dat as dat
import src.cacti.cacti_python.get_IO as IO

from src.hw_symbols import *

valid_tech_nodes = [0.022, 0.032, 0.045, 0.065, 0.090, 0.180]

def gen_abs_results(sympy_file, cache_cfg, dat_file):
    """
    Generates absolute results from SymPy and Cacti, compares with C Cacti results, and writes to a CSV file.

    Inputs:
    sympy_file : str
        Path to the base SymPy expression file.
    cache_cfg : str
        Path to the cache configuration file.
    dat_file : str
        Path to the technology .dat file.

    Outputs:
    Logs and compares the results of SymPy expressions (access time, dynamic energy, leakage, IO properties) with C Cacti results.
    Results are appended to 'abs_validate_results.csv'.
    Returns the SymPy result and the C Cacti access time.
    """

    print(f"top of gen_abs_results; sympy_file: {sympy_file}, cache_cfg: {cache_cfg}, dat_file: {dat_file}")

    # Get CACTI C results to verify
    cfg_name = cache_cfg.replace(".cfg", "").replace("cfg/", "")
    transistor_size = float(dat_file.split("/")[-1].split(".")[0][:-2])*1e-3
    print(f"transistor size: {transistor_size}")
    validate_vals = cacti_util.gen_vals(
        cfg_name,
        transistor_size=transistor_size,
    )

    validate_access_time = float(validate_vals["Access time (ns)"])
    validate_read_dynamic = float(validate_vals["Dynamic read energy (nJ)"])
    validate_write_dynamic = float(validate_vals["Dynamic write energy (nJ)"])
    validate_leakage = float(validate_vals["Standby leakage per bank(mW)"])

    buf_opt = {
            "ndwl": validate_vals["Ndwl"],
            "ndbl": validate_vals["Ndbl"],
            "nspd": validate_vals["Nspd"],
            "ndcm": validate_vals["Ndcm"],
            "ndsam1": validate_vals["Ndsam_level_1"],
            "ndsam2": validate_vals["Ndsam_level_2"],
            "repeater_spacing": validate_vals["Repeater spacing"],
            "repeater_size": validate_vals["Repeater size"],
        }
    IO_info = cacti_util.gen_symbolic(sympy_file, cache_cfg, buf_opt, use_piecewise=False)

    dat_file = os.path.join(CACTI_DIR, dat_file)

    g_ip.parse_cfg(os.path.join(CACTI_DIR, cache_cfg))
    g_ip.error_checking()

    # get tech_param values
    tech_params = {}
    dat.scan_dat(tech_params, dat_file, g_ip.data_arr_ram_cell_tech_type, g_ip.data_arr_ram_cell_tech_type, g_ip.temp)
    tech_params = {k: (10**(-9) if v == 0 else v) for k, v in tech_params.items() if v is not None and not math.isnan(v)}

    # get IO tech_param values
    IO_tech_params = {}
    IO.scan_IO(IO_tech_params, g_ip, g_ip.io_type, g_ip.num_mem_dq, g_ip.mem_data_width, g_ip.num_dq, g_ip.dram_dimm, 1, g_ip.bus_freq)
    IO_tech_params = {k: (10**(-9) if v == 0 else v) for k, v in IO_tech_params.items() if v is not None and not math.isnan(v)}

    # every file should start with this name
    sympy_filename = os.path.join(CACTI_DIR, "symbolic_expressions/" + sympy_file.rstrip(".txt"))

    # PLUG IN CACTI
    sympy_file_access_time = sympy_filename + "_access_time.txt"
    sympy_file_read_dynamic = sympy_filename + "_read_dynamic.txt"
    sympy_file_write_dynamic = sympy_filename + "_write_dynamic.txt"
    sympy_file_read_leakage = sympy_filename + "_read_leakage.txt"

    with open(sympy_file_access_time, 'r') as file:
        expression_str = file.read()

    expression = sp.sympify(expression_str)
    result = expression.subs(tech_params)

    result_access_time = result.subs(sp.I, 0)
    result_access_time = result_access_time.evalf()

    with open(sympy_file_read_dynamic, 'r') as file:
        expression_str = file.read()

    expression = sp.sympify(expression_str)
    result = expression.subs(tech_params)

    result_read_dynamic = result.subs(sp.I, 0).evalf()

    with open(sympy_file_write_dynamic, 'r') as file:
        expression_str = file.read()

    expression = sp.sympify(expression_str)
    result = expression.subs(tech_params)

    result_write_dynamic = result.subs(sp.I, 0).evalf()

    with open(sympy_file_read_leakage, 'r') as file:
        expression_str = file.read()

    expression = sp.sympify(expression_str)
    result = expression.subs(tech_params)

    result_read_leakage = result.subs(sp.I, 0).evalf()

    # PLUG IN CACIT IO
    sympy_file_io_area = sympy_filename + "_io_area.txt"
    sympy_file_io_timing_margin = sympy_filename + "_io_timing_margin.txt"
    sympy_file_io_dynamic_power = sympy_filename + "_io_dynamic_power.txt"
    sympy_file_io_phy_power = sympy_filename + "_io_phy_power.txt"
    sympy_file_io_termination_power = sympy_filename + "_io_termination_power.txt"

    with open(sympy_file_io_area, 'r') as file:
        expression_str = file.read()

    expression = sp.sympify(expression_str)
    result = expression.subs(IO_tech_params)
    result_io_area = result.subs(sp.I, 0).evalf()

    with open(sympy_file_io_timing_margin, 'r') as file:
        expression_str = file.read()

    expression = sp.sympify(expression_str)
    result = expression.subs(IO_tech_params)
    result_io_timing_margin = result.subs(sp.I, 0).evalf()

    with open(sympy_file_io_dynamic_power, 'r') as file:
        expression_str = file.read()

    expression = sp.sympify(expression_str)
    result = expression.subs(IO_tech_params)
    result_io_dynamic_power = result.subs(sp.I, 0).evalf()

    with open(sympy_file_io_phy_power, 'r') as file:
        expression_str = file.read()

    expression = sp.sympify(expression_str)
    result = expression.subs(IO_tech_params)
    result_io_phy_power = result.subs(sp.I, 0).evalf()

    with open(sympy_file_io_termination_power, 'r') as file:
        expression_str = file.read()

    expression = sp.sympify(expression_str)
    result = expression.subs(IO_tech_params)
    result_io_termination_power = result.subs(sp.I, 0).evalf()

    # print
    # print(f'Transistor size: {g_ip.F_sz_um}')
    # print(f'is_cache: {g_ip.is_cache}')

    # print(f'access_time: {result_access_time}')
    # print(f"result : {result_read_dynamic, result_write_dynamic, result_read_leakage}")

    # print(f"io_area: {result_io_area}")
    # print(f"io_timing_margin: {result_io_timing_margin}")
    # print(f"io_dynamic_power: {result_io_dynamic_power}")
    # print(f"io_phy_power: {result_io_phy_power}")
    # print(f"io_termination_power: {result_io_termination_power}")
    # print(f"validate_access_time (ns): {validate_access_time}")
    # print(f"validate_read_dynamic (nJ): {validate_read_dynamic}")
    # print(f"validate_write_dynamic (nJ): {validate_write_dynamic}")
    # print(f"validate_leakage (mW): {validate_leakage}")

    # print(f'validate_io_area: {float(validate_vals["IO area"])}')
    # print(f'validate_io_timing: {float(validate_vals["IO timing"])}')
    # print(f'validate_io_power_dynamic: {float(validate_vals["IO power dynamic"])}')
    # print(f'validate_io_power_phy: {float(validate_vals["IO power PHY"])}')
    # print(f'validate_io_power_termination_and_bias: {float(validate_vals["IO power termination and bias"])}')

    # write to CSV
    data = {
        "access_time (ns)": [result_access_time],
        "result_read_dynamic (nJ)": [result_read_dynamic],
        "result_write_dynamic (nJ)": [result_write_dynamic],
        "result_leakage (mW)": [result_read_leakage],
        "result_io_area": [result_io_area],
        "result_io_timing_margin": [result_io_timing_margin],
        "result_io_dynamic_power": [result_io_dynamic_power],
        "result_io_phy_power": [result_io_phy_power],
        "result_io_termination_power": [result_io_termination_power],
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
        "bus width": [g_ip.out_w],
        "cache size (bytes)": [g_ip.cache_sz],
        "block size (bytes)": [g_ip.block_sz],
        "is_cache": [g_ip.is_cache]
    }

    df = pd.DataFrame(data)

    directory = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(directory, exist_ok=True)

    csv_file = os.path.join(directory, "abs_validate_results.csv")

    file_exists = os.path.isfile(csv_file)
    df.to_csv(csv_file, mode='a', header=not file_exists, index=False)

    print(f"Data successfully appended to {csv_file}")

    return result, validate_access_time


def eval_log_file(input_log, output_log, cache_cfg, dat_file):
    print(f"top of eval_log_file; cache_cfg: {cache_cfg}, dat_file: {dat_file}")

    dat_file = os.path.join(CACTI_DIR, "tech_params", dat_file)

    g_ip.parse_cfg(os.path.join(CACTI_DIR, cache_cfg))
    g_ip.error_checking()

    # Get tech_param values
    tech_params = {}
    dat.scan_dat(tech_params, dat_file, g_ip.data_arr_ram_cell_tech_type, g_ip.data_arr_ram_cell_tech_type, g_ip.temp)
    tech_params = {k: (10**(-9) if v == 0 else v) for k, v in tech_params.items() if v is not None and not math.isnan(v)}

    # Get IO tech_param values
    IO_tech_params = {}
    IO.scan_IO(IO_tech_params, g_ip, g_ip.io_type, g_ip.num_mem_dq, g_ip.mem_data_width, g_ip.num_dq, g_ip.dram_dimm, 1, g_ip.bus_freq)
    IO_tech_params = {k: (10**(-9) if v == 0 else v) for k, v in IO_tech_params.items() if v is not None and not math.isnan(v)}

    # Ensure the input file exists
    if not os.path.isfile(input_log):
        raise FileNotFoundError(f"{input_log} not found.")
    
    # Open the input and output files
    with open(input_log, 'r') as infile, open(output_log, 'w') as outfile:
        for line in infile:
            # Check if line starts with NO_SYM
            if line.startswith("NO_SYM"):
                outfile.write(line)  # Simply write the line without substitution
                continue  # Skip to the next line
            
            # Try to parse the expression after the colon and keep the prefix
            try:
                # Split the line by colon and get the part after the colon (expression) and before the colon (prefix)
                if ':' in line:
                    prefix, expression_str = line.split(':', 1)
                    expression_str = expression_str.strip()  # Remove any leading/trailing spaces
                    prefix = prefix.strip()  # Clean up the prefix
                else:
                    raise ValueError(f"No colon found in line: {line.strip()}")
                
                # Convert the expression string to a SymPy expression
                expression = sp.sympify(expression_str)
                
                # Substitute tech_params into the expression
                result = expression.subs(tech_params)
                
                # Further substitute IO_tech_params if necessary
                result = result.subs(IO_tech_params)
                
                # Evaluate the expression
                evaluated_result = result.evalf()
                
                # Write the prefix and the evaluated result to the output file
                outfile.write(f"{prefix}: {evaluated_result}\n")
            except Exception as e:
                # In case of error, write an error message and skip the line
                outfile.write(f"Error evaluating line: {line.strip()}\nReason: {str(e)}\n")
    
    print(f"Evaluation completed. Results written to {output_log}")

if __name__ == "__main__":
    # T_SZ = [0.090] #, 0.180]
    # cfg_file = f"cfg/debug_cache.cfg"
    # dat_files = [f"{int(tech*1e3)}nm.dat" for tech in T_SZ]
    # input_log_file = 'src/cacti/cacti_debug.log'
    # output_log_file = 'src/cacti/cacti_debug_eval.log'

    # # cache_cfg = f"src/cacti/cfg/debug_cache.cfg"
    # # cacti_util.run_existing_cacti_cfg(cache_cfg)

    # eval_log_file(input_log_file, output_log_file, cfg_file, dat_files[0])

    parser = argparse.ArgumentParser(description="Specify config (--config), set SymPy name (--sympy) and optionally generate SymPy (-gen)")
    parser.add_argument("-c", "--config", type=str, default="debug_cache", help="Path or Name to the configuration file; don't append src/cacti/ or .cfg")
    parser.add_argument("-d", "--dat", type=str,  help="Specify technology nm -> e.g. '90nm'; if not provdied, do 45, 90, and 180")
    parser.add_argument("-s", "--sympy", type=str, help="Optionally path to the SymPy file if not named the same as cfg")
    parser.add_argument("-g", "--gen", action="store_true", help="Boolean flag to generate Sympy from Cache CFG")

    args = parser.parse_args()

    # all paths here are relative to CACTI_DIR

    cfg_file = f"cfg/{args.config}.cfg"
    sympy_file = args.config
    if args.sympy:
        sympy_file = args.sympy

    if args.dat:
        dat_files = [f"{args.dat}.dat"]
    else:
        # dat_files = [f"{int(tech*1e3)}nm.dat" for tech in TRANSISTOR_SIZES]
        T_SZ = [0.090] #, 0.180]
        dat_files = [f"{int(tech*1e3)}nm.dat" for tech in T_SZ]

    print(f"dat files: {dat_files}")

    for dat_file in dat_files:
        print(f"Running for {dat_file}\n")
        dat_file = os.path.join("tech_params", dat_file)
        gen_abs_results(sympy_file, cfg_file, dat_file)
