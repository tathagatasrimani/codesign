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

# Work in codesign/src for ease
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(project_root)
current_directory = os.getcwd()
sys.path.insert(0, current_directory)

import cacti_util
import hw_symbols
from cacti.cacti_python.parameter import g_ip
import cacti.cacti_python.get_dat as dat

### Python CACTI Gradient Generation
"""
Top function to generate the Python cacti gradient.
Can specify a metric to isolate -> only generates for that metric.
Otherwise, generates for access_time, read_dynamic, write_dynamic, and read_leakage.
"""
def cacti_python_diff(sympy_file, tech_params, diff_var, metric=None): 
    sympy_file = "cacti/sympy/" + sympy_file

    if metric:
        file = f'{sympy_file}_{metric}.txt'      
        metric_res = cacti_python_diff_single(file, tech_params, diff_var)
        results = {
            f"{metric}": metric_res
        }
    else:
        print("in top diff")
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
    
"""
Helper to diff tech_param from a given sympy expression file.
"""
def cacti_python_diff_single(sympy_file, tech_params, diff_var):
    print(f"In diff single {sympy_file}; {diff_var}")
    with open(sympy_file, 'r') as file:
        expression_str = file.read()

    # Convert string to SymPy expression
    expression = sp.sympify(expression_str, locals=hw_symbols.__dict__)
    expression = expression.replace(sp.ceiling, lambda x: x)    # sympy diff can't handle ceilings

    # Apply common subexpression elimination (CSE)
    reduced_exprs, cse_expr = sp.cse(expression)
    reduced_expression = sum(cse_expr)

    # Make a copy of the tech_params dictionary to keep the diff_var when plugging in tech_params
    tech_params_copy = tech_params.copy()
    tech_params_copy.pop(diff_var, None)

    # Back-substitute the substituted exprs into the reduced expr; use tech_params_copy to keep diff_var
    substituted_exprs = [(symbol, expr.subs(tech_params_copy)) for symbol, expr in reduced_exprs]
    for symbol, expr in reversed(substituted_exprs):
        reduced_expression = reduced_expression.subs(symbol, expr)
    
    # Differentiate the reduced expression with respect to diff_var
    diff_expression = sp.diff(reduced_expression, diff_var)
    print("differentiating")

    # Substitute tech_params into the gradient expression
    gradient = diff_expression.subs(tech_params)
    gradient = gradient.doit()

    # Simplify and evaluate to a numerical value
    gradient = sp.simplify(gradient)
    gradient = gradient.evalf()

    # Scaling and delta calculations
    scaling_factor = choose_scaling_factor(gradient, tech_params[diff_var])
    delta = scaling_factor * gradient

    new_diff_var = tech_params[diff_var] - delta
    access_time = 3  # Placeholder value for speed estimation
    new_access_time = access_time - (gradient * delta)

    # Uncomment to log the diff expression and gradient to a CSV file
    with open("cacti_validation/results/diff_expression.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow([diff_var, f" diff expression: {diff_expression}"])
        writer.writerow([diff_var, f" gradient: {gradient}"])
        writer.writerow([diff_var, f" delta: {delta}"])
        writer.writerow([diff_var, f" new_diff_var: {new_diff_var}"])
        writer.writerow([diff_var, f" new_access_time: {new_access_time}"])

    result = {
        "delta": delta,
        "gradient": gradient 
    }
    return result

"""
Helper to choose a reasonable delta from the calculated gradient.
"""
def choose_scaling_factor(pgrad, value):
    scale_factor = value / pgrad if pgrad != 0 else 1.0
    adjustment_factor = 0.01 
    scaled_factor = scale_factor * adjustment_factor

    return scaled_factor

### C CACTI Gradient Generation
"""
Top function to generate the C cacti gradient
"""
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

    original_vals = cacti_util.replace_values_in_dat_file(dat_file_path, diff_var, new_value)
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

    cacti_util.restore_original_values_in_dat_file(dat_file_path, original_vals)
    # time.sleep(5)

    gradient = float(original_val["Access time (ns)"]) - float(next_val["Access time (ns)"])
    return gradient

### MAIN
"""
Calculates the similarity of two gradients.
100 is same mag and same sign, -100 as same mag and different sign.
Values can exceed 100 and -100.
If both python_grad and c_grad are 0, return 100 similarity.
If the c_grad is 0, then "NA" is returned.
"""
def calculate_similarity_matrix(python_grad, c_grad):
    # Handle the case where both values are 0
    if python_grad == 0 and c_grad == 0:
        return 100
    
    if c_grad == 0:
        return "NA"

    # Calculate similarity
    magnitude = 100 * (1 - np.abs(python_grad - c_grad) / np.abs(c_grad))
    sign = -1 if (python_grad * c_grad < 0) else 1
    greater_than_1 = np.abs(python_grad - c_grad) / np.abs(c_grad) > 1    # essentially if they are the same sign, but doesn't fall in the -1 to 1 range

    similarity = magnitude * sign * -1 if greater_than_1 else magnitude * sign
    return similarity

"""
Generates diff result for a specific choice of
sympy expression, cache configuration, and dat technology file
"""
def gen_diff(sympy_file, cfg_file, dat_file=None):
    print(f"In gen_diff {sympy_file}; {cfg_file}; {dat_file}")
    # init input params from .cfg
    g_ip.parse_cfg(cfg_file)
    g_ip.error_checking()
    print(f'block size: {g_ip.block_sz}')

    if dat_file == None:
        # init tech params from .dat
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

    tech_params = {}
    dat.scan_dat(tech_params, dat_file, g_ip.data_arr_ram_cell_tech_type, g_ip.data_arr_ram_cell_tech_type, g_ip.temp)
    tech_params = {getattr(hw_symbols, k, None): (10**(-9) if v == 0 else v) for k, v in tech_params.items() if v is not None and not math.isnan(v)}

    # Get parameter list and tech_config
    tech_param_keys = list(tech_params.keys())
    config_key = f'Cache={g_ip.is_cache}, {g_ip.F_sz_nm}'

    # For now don't calculate these since there are multiple instances
    tech_param_keys.remove(getattr(hw_symbols, "I_off_n", None))
    tech_param_keys.remove(getattr(hw_symbols, "I_g_on_n", None))

    tech_param_keys.remove(getattr(hw_symbols, "Wmemcella", None))
    tech_param_keys.remove(getattr(hw_symbols, "Wmemcellpmos", None))
    tech_param_keys.remove(getattr(hw_symbols, "Wmemcellnmos", None))
    tech_param_keys.remove(getattr(hw_symbols, "area_cell", None))
    tech_param_keys.remove(getattr(hw_symbols, "asp_ratio_cell", None))
    
    # diff each parameter
    for diff_param in tech_param_keys:
        python_results = cacti_python_diff(sympy_file, tech_params, diff_param)  # format [metric]['gradient'/'delta']

        for metric, metric_results in python_results.items():
            new_val = tech_params[diff_param] - python_results[metric]['delta']

            # Log the CACTI Python Gradient Info (gradient, original value, delta, new value)
            python_info_csv = "cacti_validation/results/python_grad_info.csv"
            try:
                with open(python_info_csv, 'r'):
                    file_exists = True
                if not file_exists:
                    raise FileNotFoundError
            except FileNotFoundError:
                file_exists = False

            with open(python_info_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(['Python Grad', 'Org Value', 'Delta', 'New Value'])
                writer.writerow([diff_param, f"pgrad: {python_results[metric]['gradient']}", f"value: {tech_params[diff_param]}", f"delta: {python_results[metric]['delta']}", f"new_val: {new_val}"])

            # Log the Gradient Comparison between Python and C CACTI
            if new_val <= 0:  # catch in case if delta is greater than original data value
                similarity = "NAN"
                cacti_gradient = "NA"
            else:
                cacti_gradient = cacti_c_diff(dat_file, new_val, diff_param)
                python_change = python_results[metric]['gradient'] * python_results[metric]['delta']
                similarity = calculate_similarity_matrix(python_change, cacti_gradient)

            cfg_name = cfg_file.split('/')[-1]
            cfg_name = cfg_name.replace('.cfg', '')
            results_csv = f'cacti_validation/results/{cfg_name}_{metric}_grad_results.csv'
            
            try:
                with open(results_csv, 'r'):
                    file_exists = True
                if not file_exists:
                    raise FileNotFoundError
            except FileNotFoundError:
                file_exists = False

            with open(results_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(['Tech Config', 'Var Name', 'Python Gradient', 'C Gradient', 'Similarities'])
                writer.writerow([config_key, diff_param, python_change, cacti_gradient, similarity])
    
"""
Parses arguments. [Optionally generates the .cfg file and/or sympy expression file]
Loads in .cfg and .dat values.
Differentiates with respect to each parameter
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify config (-CFG), set SymPy name (-SYMPY) and optionally generate SymPy (-gen)")
    parser.add_argument("-CFG", type=str, default="cache", help="Path or Name to the configuration file; don't append cacti/ or .cfg")
    parser.add_argument("-DAT", type=str, default="", help="nm tech -> just specify '90nm'; if not provided, 45, 90, 180 will be tested")
    parser.add_argument("-SYMPY", type=str, default="", help="Optionally path to the SymPy file if not named the same as cfg")
    parser.add_argument("-gen", type=str, default="false", help="Boolean flag to generate Sympy from Cache CFG")

    args = parser.parse_args()
    dat_nm = args.DAT

    # try to keep convention where sympy expressions have same name as cfg
    if (args.SYMPY):
        sympy_file = args.SYMPY
    else:
        sympy_file = args.CFG

    gen_flag = args.gen.lower() == "true"  

    # If you haven't generated sympy expr from cache cfg yet
    # Gen Flag true and can set sympy flag to set the name of the sympy expr
    if gen_flag:
        cfg_file = "cacti/cfg/" + args.CFG + ".cfg"
        buf_vals = cacti_util.run_existing_cacti_cfg(cfg_file)

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
        
        sympy_name = args.CFG   # try to keep convention where sympy expressions have same name as cfg
        IO_info = cacti_util.cacti_gen_sympy(sympy_name, cfg_file, buf_opt, use_piecewise=False)
    else:
        cfg_file = f'cacti/cfg/{args.CFG}.cfg'

    if dat_nm:
        dat_file = f"cacti/tech_params/{dat_nm}.dat"
        gen_diff(sympy_file, cfg_file, dat_file)
    else:
        dat_file_90nm = os.path.join('cacti', 'tech_params', '90nm.dat')
        gen_diff(sympy_file, cfg_file, dat_file_90nm)

        dat_file_45nm = os.path.join('cacti', 'tech_params', '45nm.dat')
        gen_diff(sympy_file, cfg_file, dat_file_45nm)

        dat_file_180nm = os.path.join('cacti', 'tech_params', '180nm.dat')
        gen_diff(sympy_file, cfg_file, dat_file_180nm)