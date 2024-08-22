import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import sympy as sp
import math
import re
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv


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

def cacti_python_diff(sympy_file, tech_params, diff_var, metric=None): 
    sympy_file

    if metric:
        file = f'{sympy_file}_{metric}.txt'      
        metric_res = cacti_python_diff_single(file, tech_params, diff_var)
        results = {
            f"{metric}": metric_res
        }
    else:
        file = f'{sympy_file}_access_time.txt'      
        access_time_res = cacti_python_diff_single(file, tech_params, diff_var)

        file = f'{sympy_file}_read_dynamic.txt'
        read_dynamic_res = cacti_python_diff_single(file, tech_params, diff_var)

        file = f'{sympy_file}_write_dynamic.txt'
        write_dynamic_res = cacti_python_diff_single(file, tech_params, diff_var)

        file = f'{sympy_file}_read_leakage.txt'
        read_leakage_res = cacti_python_diff_single(file, tech_params, diff_var)

        results = {
            "access_time": access_time_res,
            "read_dynamic": read_dynamic_res,
            "write_dynamic": write_dynamic_res,
            "read_leakage": read_leakage_res
        }
    return results
    
def cacti_python_diff_single(sympy_file, tech_params, diff_var):
    # print(f'DIFF var: {diff_var}')
    
    # Read the sympy expression from file
    # print(f'READING {sympy_file}')
    with open(sympy_file, 'r') as file:
        expression_str = file.read()

    # Convert string to SymPy expression
    # print("Sympify")
    expression = sp.sympify(expression_str, locals=hw_symbols.__dict__)
    
    # eliminate ceilings
    expression = expression.replace(sp.ceiling, lambda x: x)

    # Initial access time estimation (placeholder)
    access_time = 3  # Placeholder value for speed estimation

    # Apply common subexpression elimination (CSE)
    reduced_exprs, cse_expr = sp.cse(expression)
    reduced_expression = sum(cse_expr)

    # print(f'reduce: {[r[0] for r in reduced_exprs]}')
    # time.sleep(40)

    # Make a copy of the tech_params dictionary and remove diff_var
    tech_params_copy = tech_params.copy()
    tech_params_copy.pop(diff_var, None)
    # print(f'tech_params_copy: {tech_params_copy}')

    # Substitute tech_params into the reduced expressions
    substituted_exprs = [(symbol, expr.subs(tech_params_copy)) for symbol, expr in reduced_exprs]
    # print(f"SUB: {substituted_exprs}")
    
    # Back-substitute the substituted expressions into the reduced expression
    for symbol, expr in reversed(substituted_exprs):
        # with open("cacti_validation/diff_expression.csv", 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow([symbol, f" substituted expression: {expr}"])
        reduced_expression = reduced_expression.subs(symbol, expr)
    
    # print(f'Reduced Expression after Substitution: {reduced_expression}')
    # time.sleep(20)

    # Differentiate the reduced expression with respect to diff_var
    print("Differentiating...")
    diff_expression = sp.diff(reduced_expression, diff_var)
    # diff_expression = diff_expression.doit()
    print("Differentiation done")

    # # Check if `_xi_1` appears in the differentiated expression
    # if '_xi_1' in str(diff_expression):
    #     print("`_xi_1` detected in the differentiation process.")
    #     for subs in diff_expression.atoms(sp.Subs):
    #         if '_xi_1' in str(subs):
    #             subs_dict = subs.as_subs_dict()
    #             print(f"`_xi_1` is substituted with: {subs_dict}")
    # time.sleep(40)

    # Substitute tech_params into the gradient expression
    gradient = diff_expression.subs(tech_params)
    gradient = gradient.doit()

    # Simplify the expression to eliminate unnecessary complexity
    gradient = sp.simplify(gradient)

    # Evaluate the expression to a numerical value
    gradient = gradient.evalf()
    # print(f'Gradient (numerical): {gradient}')

    # Scaling and delta calculations
    scaling_factor = choose_scaling_factor(gradient, tech_params[diff_var])
    delta = scaling_factor * gradient

    new_diff_var = tech_params[diff_var] - delta
    new_access_time = access_time - (gradient * delta)

    # Log the diff expression and gradient to a CSV file
    with open("cacti_validation/diff_expression.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow([diff_var, f" diff expression: {diff_expression}"])
        writer.writerow([diff_var, f" gradient: {gradient}"])
        writer.writerow([diff_var, f" delta: {delta}"])
        writer.writerow([diff_var, f" new_diff_var: {new_diff_var}"])
        writer.writerow([diff_var, f" new_access_time: {new_access_time}"])

    # print(f'new_{diff_var}: {new_diff_var}; {diff_var}: {tech_params[diff_var]}; delta: {delta}')
    # print(f'Gradient Result: {gradient}')
    # print(f'New access_time: {new_access_time}; access_time: {access_time}')

    result = {
        "delta": delta,
        "gradient": gradient 
    }
    return result


def cacti_c_diff(dat_file_path, new_value, diff_var):
    original_val = cacti_util.gen_vals(
        "validate_mem_cache",
        cacheSize=g_ip.cache_sz, # TODO: Add in buffer sizing
        blockSize=g_ip.block_sz,
        cache_type="cache",
        bus_width=g_ip.out_w,
        transistor_size=g_ip.F_sz_um,
        force_cache_config="false",
    )

    original_vals = replace_values_in_dat_file(dat_file_path, diff_var, new_value)
    # time.sleep(10)

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
    print(f'KEY is {key}')
    # time.sleep(5)
    original_values = {}
    
    with open(dat_file_path, 'r') as file:
        lines = file.readlines()
    
    # Pattern to match the key followed by any unit in parentheses and then numeric values
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

def calculate_similarity_matrix(experimental, expected):
    # Handle the case where both values are 0
    if experimental == 0 and expected == 0:
        return 100
    # Calculate similarity normally
    return 100 * (1 - np.abs(experimental - expected) / np.abs(expected))

def choose_scaling_factor(pgrad, value):
    # Calculate the order of magnitude difference between the gradient and the value
    scale_factor = value / pgrad if pgrad != 0 else 1.0

    # Optionally, adjust the scaling factor to make smaller adjustments
    # For example, you might want to use only a fraction of the calculated factor
    adjustment_factor = 0.01  # This can be tuned based on how aggressive you want the adjustments to be
    scaled_factor = scale_factor * adjustment_factor

    return scaled_factor


def append_to_csv(config_key, diff_params_similarities, csv_filename='cacti_validation/grad_results.csv'):
    # Check if the file exists by attempting to open it in append mode
    try:
        with open(csv_filename, 'r'):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    # Open the file in append mode
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(['Config Key', 'Similarities'])

        # Write the config_key in one row
        writer.writerow([config_key])

        # Prepare the row data with the format 'diff_param: similarity'
        
        similarities_str = '; '.join(
            [
                f'{param}: {similarity:.2f}' if not (similarity == sp.zoo or math.isnan(similarity)) else f'{param}: NaN'
                for param, similarity in diff_params_similarities
            ]
        )
        
        # Write the similarities on the next row
        writer.writerow([similarities_str])
    
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process configuration and SymPy files.")
    parser.add_argument("-CFG", type=str, default="cacti/mem_validate_cache.cfg", help="Path to the configuration file")
    parser.add_argument("-SYMPY", type=str, default="sympy_mem_validate", help="Path to the SymPy file")
    parser.add_argument("-v", type=str, default="Vdd", help="Tech parameter key")
    parser.add_argument("-gen", type=str, default="false", help="Boolean flag to generate output")
    parser.add_argument("-metric", type=str, default="access_time", help="Metric to be used for evaluation")

    # Parse the arguments
    args = parser.parse_args()

    # Assign values to variables
    cfg_file = args.CFG
    sympy_file = args.SYMPY
    diff_var = args.v
    gen_flag = args.gen.lower() == "true"  
    metric = args.metric

    print(cfg_file)
    print(sympy_file)
    print(diff_var)
    print(f"Generate flag is set to: {gen_flag}")
    print(f"Metric is set to: {metric}")

    if gen_flag:
        print("Generating additional output as requested.")
        # Include logic to handle the case when the -gen flag is True
    else:
        print("Skipping additional output generation.")

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
    # print(g_ip.temp)
    # time.sleep(10)
    dat.scan_dat(tech_params, dat_file, g_ip.data_arr_ram_cell_tech_type, g_ip.data_arr_ram_cell_tech_type, g_ip.temp)
    tech_params = {getattr(hw_symbols, k, None): (10**(-9) if v == 0 else v) for k, v in tech_params.items() if v is not None and not math.isnan(v)}
    # tech_params = {sp.symbols(k, positive=True): (10**(-9) if v == 0 else v) for k, v in tech_params.items() if v is not None and not math.isnan(v)}
    print(f'MAIN tech_p {tech_params}')
    tech_param_keys = list(tech_params.keys())
    tech_param_keys.remove(getattr(hw_symbols, "I_off_n", None))
    tech_param_keys.remove(getattr(hw_symbols, "I_g_on_n", None))

    tech_param_keys.remove(getattr(hw_symbols, "Wmemcella", None))
    tech_param_keys.remove(getattr(hw_symbols, "Wmemcellpmos", None))
    tech_param_keys.remove(getattr(hw_symbols, "Wmemcellnmos", None))
    tech_param_keys.remove(getattr(hw_symbols, "area_cell", None))
    tech_param_keys.remove(getattr(hw_symbols, "asp_ratio_cell", None))
    
    # print(tech_param_keys)
    # time.sleep(30)
    # get all diff results

    # tech_param_keys = ["Vdd", "Vth"]
    config_key = f'Cache={g_ip.is_cache}, {g_ip.F_sz_nm}'
    
    # Collect similarities
    diff_params_similarities = []
    for diff_param in tech_param_keys:
        python_results = cacti_python_diff(sympy_file, tech_params, diff_param, metric)

        # Access time
        new_val = tech_params[diff_param] - python_results[metric]['delta']
        # print(f"delta: {python_results[metric]['delta']}")
        # print(f'new_val: {new_val}')
        # print(f"gradient: {python_results[metric]['gradient']}")
        # print(f"diff_param {diff_param}; org_value: {tech_params[diff_param]}")
        # time.sleep(3)

        with open("cacti_validation/gradient_values.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([diff_param, f" pgrad: {python_results[metric]['gradient']}", f" value: {tech_params[diff_param]}", f" delta: {python_results[metric]['delta']}", f" new_val: {new_val}"])

        if(new_val <= 0):
            similarity = 100
        else:
            cacti_gradient = cacti_c_diff(dat_file, new_val, diff_param)
            python_change = python_results[metric]['gradient'] * python_results[metric]['delta']
            similarity = calculate_similarity_matrix(python_change, cacti_gradient)
            # print(f"{diff_param} is {metric}: python: {python_change}; C: {cacti_gradient}")
            with open("cacti_validation/gradient_values.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([diff_param, f" pchange: {python_change}", f" cchange: {cacti_gradient}", f"similairty: {similarity}"])
            # time.sleep(6)

        diff_params_similarities.append((diff_param, similarity))

    # Append all the results to the CSV in one row
    append_to_csv(config_key, diff_params_similarities)



