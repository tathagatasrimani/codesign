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

def cacti_python_diff(sympy_file, dat_file):
    # Load tech parameters (assuming this part is correct)
    tech_params = {}
    dat.scan_dat(tech_params, dat_file, g_ip.data_arr_ram_cell_tech_type, g_ip.data_arr_ram_cell_tech_type, g_ip.temp)
    tech_params = {k: (10**(-9) if v == 0 else v) for k, v in tech_params.items() if v is not None and not math.isnan(v)}
    print(f"Tech params: {tech_params}")
    
    # Read the sympy expression from file
    print(f'READING {sympy_file}')
    with open(sympy_file, 'r') as file:
        expression_str = file.read()

    # Convert string to SymPy expression
    print("sympify")
    expression = sp.sympify(expression_str)

    # Apply common subexpression elimination (CSE)
    print("cse")
    reduced_exprs, cse_expr = sp.cse(expression)
    reduced_expression = sum(cse_expr)
    diff_file_path = os.path.join(os.path.dirname(sympy_file), 'diff_reduced.txt')
    with open(diff_file_path, 'w') as diff_file:
        diff_file.write(str(reduced_expression))
    print("written reduced")
    
    # Define Vdd as a real variable
    Vdd = sp.symbols('Vdd')  # Ensure Vdd is treated as a real variable
    print("differentiate")
    diff_expression = sp.diff(reduced_expression, Vdd)

    # Substitute tech_params into reduced_exprs before plugging back
    print(f"Substituting tech_params into reduced_exprs")
    substituted_exprs = [(symbol, expr.subs(tech_params)) for symbol, expr in reduced_exprs]

    # Back-substitute the substituted expressions into the differentiated expression
    print(f"Substituting reduced expressions back into the gradient")
    for symbol, expr in reversed(substituted_exprs):
        diff_expression = diff_expression.subs(symbol, expr)

    # Substitute any remaining tech_params into the gradient
    gradient = diff_expression.subs(tech_params)
    print(f"Gradient after substitution: {gradient}")

    # Force Vdd to be treated as real by removing re(Vdd) and any Subs expressions
    gradient = gradient.subs(sp.re(Vdd), Vdd)
    gradient = gradient.subs(sp.im(Vdd), 0)  # Ensure imaginary part is 0

    # Simplify the expression to eliminate unnecessary complexity
    gradient = sp.simplify(gradient)
    print(f"Simplified gradient: {gradient}")

    # Evaluate the expression to a numerical value
    gradient = gradient.evalf()
    print(f"Gradient after evalf: {gradient}")

    new_Vdd = tech_params["Vdd"] - gradient
    print(f'new_Vdd: {new_Vdd}; Vdd: {tech_params["Vdd"]}')
    
    print(f'Gradient Result: {gradient}')
    return new_Vdd


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

    gradient = float(original_val["Access time (ns)"]) - float(next_val["Access time (ns)"])
    
    return gradient

def replace_values_in_dat_file(dat_file_path, key, new_value):
    original_values = {}
    with open(dat_file_path, 'r') as file:
        lines = file.readlines()
    
    pattern = re.compile(rf"^(-{key} .*)$")
    
    for i, line in enumerate(lines):
        if pattern.match(line.strip()):
            # Extract original values and store them
            parts = line.strip().split()
            original_values[i] = parts[1:]  # Store original values# Replace the values with the new value
            lines[i] = f"-{key} " + " ".join([str(new_value)] * (len(parts) - 1)) + "\n"

    with open(dat_file_path, 'w') as file:
        file.writelines(lines)
    
    return original_values

def restore_original_values_in_dat_file(dat_file_path, original_values):
    with open(dat_file_path, 'r') as file:
        lines = file.readlines()
    
    for i, values in original_values.items():
        lines[i] = f"-{lines[i].split()[0]} " + " ".join(values) + "\n"

    with open(dat_file_path, 'w') as file:
        file.writelines(lines)

if __name__ == "__main__":
    # `current_directory` now reflects the correct working directory
    file_path = os.path.join(current_directory, 'validate_results.csv')
    print(current_directory)

    # Since you are now in `parent_dir`, the files are referenced directly
    sympy_file = os.path.join('Buf_access_time.txt') 
    dat_file = os.path.join('cacti', 'tech_params', '90nm.dat')

    cacti_python_diff(sympy_file, dat_file)
