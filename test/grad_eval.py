import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import sympy as sp
import math
import re
import time
import argparse
import csv
import logging
import glob
import multiprocessing as mp

logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.cacti import TRANSISTOR_SIZES, CACTI_DIR
from src import CACTI_DIR
from src import cacti_util
from src import hw_symbols
from src.cacti.cacti_python.parameter import g_ip
from src.cacti.cacti_python import get_dat, get_IO

rng = np.random.default_rng()

output_label_mapping = {
    "access_time": "Access time (ns)",
    "read_dynamic": "Dynamic read energy (nJ)",
    "write_dynamic": "Dynamic write energy (nJ)",
    "read_leakage": "Standby leakage per bank(mW)",
}


def cacti_python_diff(sympy_file, tech_params, diff_var, metric=None):
    """
    DEPRECATED?
    Top function that generates the Python Cacti gradient for specified metrics.
    If no metric is provided, generates for access_time, read_dynamic,
    write_dynamic, and read_leakage.

    Calls `cacti_python_diff_single` for each of the above output variables.

    Inputs:
    sympy_file : str
        Prefix  to the base SymPy expression file to which the metric name will be appended.
        eg. `<cfg_name>_<tech>`
    tech_params : dict
        Dictionary containing the technology parameters for substitution.
    diff_var : str
        The variable to differentiate with respect to.
    metric : str, optional
        Specific metric to isolate and generate gradients for (e.g., access_time,
        read_dynamic).

    Returns:
    dict
        Dictionary containing the gradients for the specified or default metrics.
    """
    logger.info(
        f"Top of cacti_python_diff(sympy_file: {sympy_file}, tech_params: <not printed>, diff_var: {diff_var}, metric: {metric})"
    )
    sympy_file = os.path.join(CACTI_DIR, "symbolic_expressions/" + sympy_file)

    metrics = (
        [metric]
        if metric
        else ["access_time", "read_dynamic", "write_dynamic", "read_leakage"]
    )
    results = {}
    for _metric in metrics:
        file = f"{sympy_file}_{_metric}.txt"
        logger.info(f"Calling cacti_python_diff_single metric: {_metric}")
        metric_res = cacti_python_diff_single(file, tech_params, _metric, diff_var)
        logger.info("Finished cacti_python_diff_single")
        logger.info(f"results for {_metric}: {metric_res}")
        results[_metric] = metric_res

    return results


def evaluate_python_diff(
    y_expr, y_expr_file_name, tech_params, x_symbol, y_name, Q: mp.Queue
):
    """
    Will be called in a process
    """
    dydx_expr_file_name = cacti_util.differentiate(
        y_expr, x_symbol, y_expr_file_name, tech_params
    )
    res = evaluate_derivative(dydx_expr_file_name, tech_params, x_symbol)
    res["y_name"] = y_name
    res["x_name"] = x_symbol
    Q.put(res)
    # Q.close()


def evaluate_derivative(dydx_file_name, tech_params, x_symbol):
    """
    Evalualate the partial derivative expression given in the file at
    the current technology parameters.
    """
    dydx_expr = sp.sympify(open(dydx_file_name).read(), locals=hw_symbols.symbol_table)

    dydx = dydx_expr.xreplace(tech_params).evalf()

    delta_x = 0.01 * tech_params[x_symbol]
    delta_y = -1 * delta_x * dydx

    logger.info(
        f"x: {x_symbol}, x.val: {tech_params[x_symbol]}, dydx: {dydx}, delta_x: {delta_x}, delta_y: {delta_y}"
    )

    result = {
        "x": tech_params[x_symbol],
        "delta_x": delta_x,
        "dydx": dydx,
        "delta_y": delta_y,
    }
    return result


def cacti_python_diff_single(y_expr_file, tech_params, y_name, x_name):
    """
    DEPRECATED?
    Computes the gradient of a technology parameter from a SymPy expression file.

    calculate dy/dx

    Inputs:
    sympy_file : str
        Absolute path to the base SymPy expression file to which the metric name will be appended.
        eg. `~/codesign/src/cacti/symbolic_expressions/<cfg_name>_<tech>_<metric>.txt`
    tech_params : dict
        Dictionary containing the technology parameters for substitution.
    diff_var : str
        The variable to differentiate with respect to.

    Returns:
    dict
        Dictionary containing the gradient and delta for the specified variable.
    """
    with open(y_expr_file, "r") as f:
        y_expr = sp.sympify(f.read(), locals=hw_symbols.symbol_table)

    logger.info(f"num free expressions: {y_expr.free_symbols}")
    # Convert string to SymPy expression
    y_expr = y_expr.replace(sp.ceiling, lambda x: x)  # sympy diff can't handle ceilings

    # Apply common subexpression elimination (CSE)
    # reduced_exprs, cse_expr = sp.cse(y_expr)
    # reduced_expression = sum(cse_expr)

    # Substitute in all the tech params except the one we're differentiating wrt

    # Make a copy of the tech_params dictionary to keep the diff_var when plugging in tech_params
    tech_params_copy = tech_params.copy()
    tech_params_copy.pop(x_name, None)

    # Back-substitute the substituted exprs into the reduced expr; use tech_params_copy to keep diff_var
    # substituted_exprs = [
    #     (symbol, expr.subs(tech_params_copy)) for symbol, expr in reduced_exprs
    # ]
    # for symbol, expr in reversed(substituted_exprs):
    #     reduced_expression = reduced_expression.subs(symbol, expr)
    reduced_expression = y_expr.xreplace(tech_params_copy)

    logger.info(f"{y_name}: {reduced_expression}")
    # Differentiate the reduced expression with respect to diff_var
    dydx_expr = sp.diff(reduced_expression, x_name)
    logger.info(f"dydx_expr: {dydx_expr}")

    # Substitute tech_params into the gradient expression
    dydx = dydx_expr.xreplace(tech_params).evalf()
    logger.info(f"dydx: {dydx}")
    # gradient = gradient.doit()

    # Simplify and evaluate to a numerical value
    # gradient = sp.simplify(gradient)
    # partial_derivative = partial_derivative.evalf()

    # step size calculation; negate step size for gradient descent
    delta_x = 0.1 * tech_params[x_name]
    # choose_step_size(dydx, tech_params[diff_var])
    delta_y = -1 * delta_x * dydx

    # new_value = tech_params[diff_var] + delta
    logger.info(
        f"x: {x_name}, x.val: {tech_params[x_name]}, dydx: {dydx}, delta_x: {delta_x}, delta_y: {delta_y}"
    )

    # Uncomment to log the diff expression and gradient to a CSV file
    # is this whole thing unused?
    # res_dir = os.path.join(os.path.dirname(__file__), "results")
    # os.makedirs(res_dir, exist_ok=True)
    # with open(f"{res_dir}/diff_expression.csv", "a", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow([diff_var, f" partial_derivative: {partial_derivative}"])
    #     writer.writerow([diff_var, f" delta: {delta}"])
    #     writer.writerow([diff_var, f" new_value: {new_value}"])

    result = {"delta_x": delta_x, "dydx": dydx, "delta_y": delta_y}
    return result


def choose_step_size(dydx, y):
    """
    DEPRECATED?
    Helper to choose a reasonable step size based on the
    gradient and the parameter value. Get a 1% change in the
    parameter value.

    delta_x * dydx = delta_y = 1% * y
    delta_x = 1% * value / pd

    Inputs:
    pd : float
        The calculated partial derivative.
    value : float
        The current value of the parameter being differentiated.

    Returns:
    float
        A scaled factor for adjusting the parameter.
    """
    logger.info(f"Top of choose_step_size(pd: {pd}, value: {value})")
    desired_delta = 0.01 * value
    logger.info(f"desired_delta: {desired_delta}")
    step_size = desired_delta / pd if pd != 0 else 0
    logger.info(f"step_size: {step_size}")
    return step_size


def cacti_c_diff(
    cfg_file, dat_file_path, new_x_value, x_name, y_name: str = "access_time"
):
    """
    Top function to generate the C Cacti gradient by running
    Cacti twice (before and after changing the parameter).

    Inputs:
    dat_file_path : str
        Path to the .dat file containing Cacti parameters.
    new_x_value : float
        The new value to be used for the parameter during the
        second Cacti run.
    x_name : str
        The variable to differentiate with respect to.
    out_val : str
        Which of the cacti outputs to consider here. Default is access time. Needs to match
        the key format in the output of the cacti_util.gen_vals function.

    Returns:
    float
        The calculated gradient, which is the difference in access
        time between the two Cacti runs.
    """

    original_val = cacti_util.gen_vals(
        cfg_file,
    )

    original_vals = cacti_util.replace_values_in_dat_file(
        dat_file_path, x_name, new_x_value
    )

    next_val = cacti_util.gen_vals(
        cfg_file,
    )

    cacti_util.restore_original_values_in_dat_file(dat_file_path, original_vals)

    delta_y = float(original_val[output_label_mapping[y_name]]) - float(
        next_val[output_label_mapping[y_name]]
    )
    return delta_y


def calculate_similarity_matrix(python_delta_y, c_delta_y):
    """
    Calculates the similarity of two gradients.
    A score of 100 indicates same magnitude and sign, while
    -100 indicates same magnitude and opposite sign.

    delta_x is the same for both runs (python and c).

    Inputs:
    python_grad : float
        The gradient calculated using Python.
    c_grad : float
        The gradient calculated using C.

    Returns:
    float or str
        The similarity score or "NA" if the C gradient is
        zero. If both gradients are zero, returns 100.
    """

    # Handle the case where both values are 0
    if python_delta_y == 0 and c_delta_y == 0:
        return 100
    elif c_delta_y == 0:
        return "NA"

    # Calculate similarity
    magnitude = 100 * (1 - np.abs(python_delta_y - c_delta_y) / np.abs(c_delta_y))
    sign = np.sign(python_delta_y * c_delta_y)
    greater_than_1 = (
        np.abs(python_delta_y - c_delta_y) / np.abs(c_delta_y) > 1
    )  # essentially if they are the same sign, but doesn't fall in the -1 to 1 range

    similarity = magnitude * sign * -1 if greater_than_1 else magnitude * sign
    return similarity


def gen_diff(sympy_file, cfg_file, dat_file, gen_flag=True):
    """
    Generates the gradient results for a specific SymPy expression,
    cache configuration, and technology file.

    Inputs:
    sympy_file : str
        Path to the SymPy expression file. Should be relative to the CACTI_DIR
    cfg_file : str
        Path to the cache configuration file. Should be relative to the CACTI_DIR
    dat_file : str, optional
        Path to the technology .dat file (default determined by
        g_ip.F_sz_nm if not provided). Relative to the CACTI_DIR

    Outputs:
    Logs the gradient comparison between Python and C CACTI results,
    and stores them in CSV files.
    """

    print(
        f"Top of gen diff: sympy_file: {sympy_file}, cfg_file: {cfg_file}, dat_file: {dat_file}"
    )

    if gen_flag:
        logger.info("Generating symbolic expressions")
        buf_vals = cacti_util.gen_vals(
            cfg_file.split("/")[1].replace(".cfg", ""),
            transistor_size=int(args.dat[:-2]) * 1e-3,
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

        # try to keep convention where sympy expressions have same name as cfg
        cacti_util.gen_symbolic(sympy_file, cfg_file, buf_opt, use_piecewise=False)

        logger.info(f"finished gen symbolic")

    cfg_file = os.path.join(CACTI_DIR, cfg_file)

    # init input params from .cfg
    g_ip.parse_cfg(cfg_file)
    g_ip.error_checking()

    dat_file = os.path.join(CACTI_DIR, dat_file)

    tech_params = {}
    get_dat.scan_dat(
        tech_params,
        dat_file,
        g_ip.data_arr_ram_cell_tech_type,
        g_ip.data_arr_ram_cell_tech_type,
        g_ip.temp,
    )
    tech_params = {
        getattr(hw_symbols, k, None): (10 ** (-9) if v == 0 else v)
        for k, v in tech_params.items()
        if v is not None and not math.isnan(v)
    }

    get_IO.scan_IO(
        tech_params,
        g_ip,
        g_ip.io_type,
        g_ip.num_mem_dq,
        g_ip.mem_data_width,
        g_ip.num_dq,
        g_ip.dram_dimm,
        1,
        g_ip.bus_freq,
    )
    cacti_IO_params = {
        k: (1 if v is None or math.isnan(v) else (10 ** (-9) if v == 0 else v))
        for k, v in tech_params.items()
    }

    tech_param_keys = list(tech_params.keys())

    print(f"tech_param_keys: {tech_param_keys}")

    config_key = f"Cache={g_ip.is_cache}, {g_ip.F_sz_nm}"

    # For now don't calculate these since there are multiple instances?
    # What does this mean?
    tech_param_keys.remove(getattr(hw_symbols, "I_off_n", None))
    tech_param_keys.remove(getattr(hw_symbols, "I_g_on_n", None))

    tech_param_keys.remove(getattr(hw_symbols, "Wmemcella", None))
    tech_param_keys.remove(getattr(hw_symbols, "Wmemcellpmos", None))
    tech_param_keys.remove(getattr(hw_symbols, "Wmemcellnmos", None))
    tech_param_keys.remove(getattr(hw_symbols, "area_cell", None))
    tech_param_keys.remove(getattr(hw_symbols, "asp_ratio_cell", None))

    # ============ FOR TESTING =================
    # leave this in to make testing quicker.
    # tech_param_keys = [
    #     hw_symbols.symbol_table["C_ox"],
    #     hw_symbols.symbol_table["Vdsat"],
    #     hw_symbols.symbol_table["C_g_ideal"],
    # ] 
    # tech_param_keys = rng.choice(tech_param_keys, 5, replace=False)
    # ============ END TESTING =================

    results_data = {
        "x_name": [],
        "y_name": [],
        "x": [],
        "python dydx": [],
        "delta x": [],
        "python delta y": [],
        "c delta y": [],
        "similarity": [],
    }

    cfg = cfg_file.split("/")[-1].replace(".cfg", "")
    dat = dat_file.split("/")[-1].replace(".dat", "")[:-2]

    results_save_file = os.path.join(
        os.path.dirname(__file__), "results", f"{cfg}_{dat}_grad_results.csv"
    )
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_save_file)

    mp.set_start_method("spawn")

    symbolic_expression_files = glob.glob(
        os.path.join(CACTI_DIR, "symbolic_expressions", f"{cfg}_{dat}*.txt")
    )
    pd_results_dir = os.path.join(
        CACTI_DIR, "symbolic_expressions", "partial_derivatives"
    )
    os.makedirs(pd_results_dir, exist_ok=True)

    Q = mp.Queue()
    processes = []
    n = 0  # total number of processes started

    # start processes for all unique input, output pairs (tech_param, cacti_output)
    for f in symbolic_expression_files:
        print(f"Reading {f}")
        y_name = f.split("/")[-1].replace(".txt", "").replace(f"{cfg}_{dat}_", "")
        # print(f"y_name: {y_name}; cfg_dat_: '{cfg}_{dat}_'")
        expr = sp.sympify(open(f).read(), locals=hw_symbols.symbol_table)
        for free_symbol in expr.free_symbols:
            if free_symbol not in tech_param_keys:
                continue

            processes.append(
                mp.Process(
                    target=evaluate_python_diff,
                    args=(expr, f, tech_params, free_symbol, y_name, Q),
                )
            )
            processes[-1].start()
            n += 1

    # wait for all processes to finish, but process each one as it finishes
    for _ in range(n):
        data = Q.get()
        print(data)
        new_x_val = data["x"] + data["delta_x"]
        c_delta_y = cacti_c_diff(
            cfg, dat_file, new_x_val, data["x_name"], data["y_name"]
        )
        similarity = calculate_similarity_matrix(data["delta_y"], c_delta_y)
        data["c delta y"] = c_delta_y
        data["similarity"] = similarity
        tmp = pd.DataFrame([data])
        tmp.rename(
            columns={
                "delta_x": "delta x",
                "dydx": "python dydx",
                "delta_y": "python delta y",
            },
            inplace=True,
        )
        results_df = pd.concat([results_df, tmp], ignore_index=True)
        results_df.to_csv(results_save_file)


if __name__ == "__main__":
    """
    Parses arguments for configuration, SymPy generation, and technology parameters.
    Optionally generates the .cfg and/or SymPy expression files and loads .cfg and .dat values.
    Differentiates with respect to each parameter.

    Inputs:
    -CFG : str, optional
        Name or path to the configuration file (default: "cache"). Do not include "src/cacti/" or ".cfg".
    -DAT : str, optional
        Technology node (e.g., "90nm"). If not provided, defaults to running for 45nm, 90nm, and 180nm.
    -SYMPY : str, optional
        Path to the SymPy expression file (if different from the config name).
    -gen : str, optional
        Boolean flag ("true"/"false") to generate SymPy expressions from the configuration file (default: "false").

    Outputs:
    Generates and processes the specified SymPy and Cacti configuration files, performing differentiation and gradient calculations.
    Stores results in CSV files.
    """
    # format="%(levelname}s:%(name)s:%(funcName)s:%(message)s"
    logging.basicConfig(level=logging.INFO, filename="logs/cacti_grad_validation.log")
    logger.info("\n\n=====================\n\n")
    parser = argparse.ArgumentParser(
        description="Specify config (--config), set Dat file (--dat) and optionally generate SymPy (--gen)"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="debug_cache",
        help="Path or Name to the configuration file; don't append src/cacti/ or .cfg",
    )
    parser.add_argument(
        "-d",
        "--dat",
        type=str,
        help="nm tech -> just specify '90nm'; if not provided, 45, 90, 180 will be tested",
    )
    parser.add_argument(
        "-g",
        "--gen",
        action="store_false",
        help="Boolean flag to generate Sympy from Cache CFG (default is True)",
    )

    args = parser.parse_args()
    dat_nm = args.dat

    cfg_file = "cfg/" + args.config + ".cfg"
    sympy_file = f"{args.config}_{dat_nm[:-2]}"

    if args.dat:
        dat_files = [f"{args.dat}.dat"]
    else:
        dat_files = [f"{int(tech*1e3)}nm.dat" for tech in TRANSISTOR_SIZES]

    print(f"dat files: {dat_files}")

    for dat_file in dat_files:
        print(f"Running for {dat_file}\n")
        dat_file = os.path.join("tech_params", dat_file)
        gen_diff(sympy_file, cfg_file, dat_file, args.gen)
    
    print(f"nDone running for all dat files")
