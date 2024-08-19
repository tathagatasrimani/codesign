import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import sympy as sp
import math
import re
import time

# Change the working directory to /Users/dw/Documents/codesign/codesign/src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(project_root)

# Now `current_directory` should reflect the new working directory
current_directory = os.getcwd()
print(current_directory)

# Add the parent directory to sys.path
sys.path.insert(0, current_directory)

# Now you can safely import modules that rely on the correct working directory
import cacti_util
import hw_symbols
from cacti.cacti_python.parameter import g_ip
import cacti.cacti_python.get_dat as dat

def load(cache_cfg):
    # Initialize input parameters from .cfg
    g_ip.parse_cfg(cache_cfg)
    g_ip.error_checking()
    print(f'block size: {g_ip.block_sz}')

def cacti_python_diff(sympy_file, tech_params):    
    # Read the sympy expression from file
    print(f'READING {sympy_file}')
    with open(sympy_file, 'r') as file:
        expression_str = file.read()

    # Convert string to SymPy expression
    print("sympify")
    expression = sp.sympify(expression_str)

    # get initial access time
    access_time = 3 # set for 3 estimate for speed, expression.subs(tech_params)

    # get gradient
    # Apply common subexpression elimination (CSE)
    reduced_exprs, cse_expr = sp.cse(expression)
    reduced_expression = sum(cse_expr)

    # Substitute tech_params into reduced_exprs before plugging back
    substituted_exprs = [(symbol, expr.subs(tech_params)) for symbol, expr in reduced_exprs]
    
    # differentiate
    Vdd = sp.symbols('Vdd')  # Ensure Vdd is treated as a real variable
    diff_expression = sp.diff(reduced_expression, Vdd)

    # Back-substitute the substituted expressions into the differentiated expression
    for symbol, expr in reversed(substituted_exprs):
        diff_expression = diff_expression.subs(symbol, expr)

    # Substitute tech_params into the gradient
    gradient = diff_expression.subs(tech_params)

    # Force Vdd to be treated as real by removing re(Vdd) and any Subs expressions
    gradient = gradient.subs(sp.re(Vdd), Vdd)
    gradient = gradient.subs(sp.im(Vdd), 0)  # Ensure imaginary part is 0

    # Simplify the expression to eliminate unnecessary complexity
    gradient = sp.simplify(gradient)

    # Evaluate the expression to a numerical value
    gradient = gradient.evalf()

    # calcs
    order_of_magnitude = math.floor(math.log10(abs(gradient)))
    scaling_factor = 10 ** (-2 - order_of_magnitude)
    delta = scaling_factor * gradient

    new_Vdd = tech_params["Vdd"] - delta
    new_access_time = access_time - (gradient * delta)

    print(f'new_Vdd: {new_Vdd}; Vdd: {tech_params["Vdd"]}; delta: {delta}')
    print(f'Gradient Result: {gradient}')
    print(f'New access_time: {new_access_time}; access_time: {access_time}')

    result = {
        "delta": delta,
        "gradient": gradient * 10**4
    }
    return result


def cacti_c_diff(dat_file_path, new_value):
    original_val = cacti_util.gen_vals(
        "validate_mem_cache",
        cacheSize=g_ip.cache_sz, # TODO: Add in buffer sizing
        blockSize=g_ip.block_sz,
        cache_type="cache",
        bus_width=g_ip.out_w,
        transistor_size=g_ip.F_sz_um,
        force_cache_config="false",
    )

    original_vals = replace_values_in_dat_file(dat_file_path, "Vdd", new_value)
    time.sleep(5)

    next_val = cacti_util.gen_vals(
        "validate_mem_cache",
        cacheSize=g_ip.cache_sz, # TODO: Add in buffer sizing
        blockSize=g_ip.block_sz,
        cache_type="cache",
        bus_width=g_ip.out_w,
        transistor_size=g_ip.F_sz_um,
        force_cache_config="false",
    )

    restore_original_values_in_dat_file(dat_file_path, original_vals)
    # time.sleep(5)

    gradient = float(original_val["Access time (ns)"]) - float(next_val["Access time (ns)"])
    return gradient

def replace_values_in_dat_file(dat_file_path, key, new_value):
    original_values = {}
    with open(dat_file_path, 'r') as file:
        lines = file.readlines()
    
    # Pattern to match the key followed by any text, ensuring the line starts with the key
    pattern = re.compile(rf"^-{key}\s+\(\w+\)\s+(.*)$")
    
    for i, line in enumerate(lines):
        match = pattern.match(line.strip())
        if match:
            # Extract original values and store them
            original_values[i] = match.group(1).split()
            
            # Keep the unit label (e.g., (V))
            unit_label = re.search(r'\(\w+\)', line).group(0)
            
            # Replace the numeric values with the new value
            lines[i] = f"-{key} {unit_label} " + " ".join([str(new_value)] * len(original_values[i])) + "\n"

    with open(dat_file_path, 'w') as file:
        file.writelines(lines)
    
    return original_values

def restore_original_values_in_dat_file(dat_file_path, original_values):
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
    # Set default values
    default_cfg_file = "cacti/mem_validate_cache.cfg"
    default_sympy_file = "sympy_mem_validate_access_time.txt"

    # Get the cfg_file and sympy_file from command-line arguments or use defaults
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else default_cfg_file
    sympy_file = sys.argv[2] if len(sys.argv) > 2 else default_sympy_file

    print(cfg_file)
    print(sympy_file)
    time.sleep(10)

    # Since you are now in `parent_dir`, the files are referenced directly
    load(cfg_file)

    if g_ip.F_sz_nm == 90:
        dat_file = os.path.join('cacti', 'tech_params', '90nm.dat')
    elif g_ip.F_sz_nm == 65:
        dat_file = os.path.join('cacti', 'tech_params', '65nm.dat')
    elif g_ip.F_sz_nm == 45:
        dat_file = os.path.join('cacti', 'tech_params', '45nm.dat')
    elif g_ip.F_sz_nm == 32:
        dat_file = os.path.join('cacti', 'tech_params', '32nm.dat')
    elif g_ip.F_sz_nm == 22:
        dat_file = os.path.join('cacti', 'tech_params', '22nm.dat')
    else:
        dat_file = os.path.join('cacti', 'tech_params', '180nm.dat')

    print(f'DAT FILE: {dat_file}')

    # read from dat
    tech_params = {}
    dat.scan_dat(tech_params, dat_file, g_ip.data_arr_ram_cell_tech_type, g_ip.data_arr_ram_cell_tech_type, g_ip.temp)
    tech_params = {k: (10**(-9) if v == 0 else v) for k, v in tech_params.items() if v is not None and not math.isnan(v)}
    
    cacti_python_res = cacti_python_diff(sympy_file, tech_params)
    new_val = tech_params["Vdd"] - cacti_python_res['delta']
    print(f'new_val: {new_val}')
    cacti_gradient = cacti_c_diff(dat_file, new_val)

    print(f"ACCESS_TIME: python: {cacti_python_res['gradient']}; C: {cacti_gradient}")


