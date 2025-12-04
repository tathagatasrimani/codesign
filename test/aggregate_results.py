import json
import os

import matplotlib.pyplot as plt
import numpy as np

from test.visualize_block_vectors import get_latest_log_dir


def plot_results_in_table(results_dirs, obj_list, save_path=None):
    """
    Create a table where:
      - rows = objective functions (obj_list)
      - columns = device parameters (plot_list)
      - cell values = final parameter values from each run

    Assumes:
      - len(results_dirs) == len(obj_list)
      - each results_dir contains "figs/param_data.json"
      - the last entry in param_data is a dict of parameter -> value
    """
    plot_list_map = {
        "supply voltage": "V_dd",
        "effective threshold voltage": "V_th",
        "gate width": "W",
        "gate length": "L",
        "t_ox": "tox"
    }

    assert len(results_dirs) == len(
        obj_list
    ), "results_dirs and obj_list must have the same length"

    # Collect values into a 2D list: rows=obj, cols=param
    data = np.zeros((len(obj_list), len(plot_list_map) + 1), dtype=float)

    for i, results_dir in enumerate(results_dirs):
        param_path = os.path.join(results_dir, "figs", "plot_param_data.json")
        with open(param_path, "r") as f:
            param_data = json.load(f)
        final_params = param_data[-1]

        for j, param in enumerate(plot_list_map):
            value = final_params.get(param, float("nan"))
            data[i, j] = value

        # collect activity factor        
        activity_factor_path = os.path.join(results_dir, "block_vectors", f"block_vectors_forward_pass_{len(param_data)-1}.json")
        with open(activity_factor_path, "r") as f:
            activity_factor_data = json.load(f)
        top_key = list(activity_factor_data.keys())[0]
        activity_factor = activity_factor_data[top_key]["top"]["computation_activity_factor"]
        data[i, len(plot_list_map)] = activity_factor

    # Create a nicely formatted table using matplotlib
    fig, ax = plt.subplots(figsize=(1.8 * (len(plot_list_map) + 1) + 2, 0.7 * len(obj_list) + 2))
    ax.axis("off")

    # Format values (e.g., scientific notation where appropriate)
    cell_text = []
    for i in range(len(obj_list)):
        row = []
        for j in range(len(plot_list_map) + 1):
            v = data[i, j]
            if np.isnan(v):
                row.append("—")
            elif abs(v) >= 1e3 or (abs(v) > 0 and abs(v) < 1e-2):
                row.append(f"{v:.2e}")
            else:
                row.append(f"{v:.4g}")
        cell_text.append(row)

    table = ax.table(
        cellText=cell_text,
        rowLabels=[obj.upper() for obj in obj_list],
        colLabels=list(plot_list_map.values()) + ["Activity Factor"],
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.1, 1.4)

    # Style header cells
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight="bold")
        if row == 0:
            cell.set_facecolor("#e9ecef")
        elif col == -1:
            cell.set_facecolor("#f8f9fa")

    ax.set_title("Final Device Parameters by Objective", fontweight="bold", pad=20)

    plt.tight_layout()

    if save_path is None:
        save_path = "aggregate_results_table.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_results_from_regression_results(regression_results_dir, benchmark_names):
    save_dir = os.path.join(regression_results_dir, "../tables")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for benchmark_name in benchmark_names:
        sub_dirs = [d for d in os.listdir(regression_results_dir) if d.find(benchmark_name) != -1]
        results_dirs = []
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(regression_results_dir, sub_dir)
            result_dir = get_latest_log_dir(os.path.join(sub_dir_path, "log"))
            results_dirs.append(result_dir)
        obj_list = [result_dir.split("/")[-1] for result_dir in results_dirs]
        plot_results_in_table(results_dirs, obj_list, os.path.join(save_dir, f"{benchmark_name}_table.png"))

if __name__ == "__main__":
    """results_dirs = [
        get_latest_log_dir("/scratch/patrick/codesign/test/regression_results/benchmark_results_test.list/benchmark_suite/vitis_mobilenet_delay/log"),
        get_latest_log_dir("/scratch/patrick/codesign/test/regression_results/benchmark_results_test.list/benchmark_suite/vitis_mobilenet_energy/log"),
        get_latest_log_dir("/scratch/patrick/codesign/test/regression_results/benchmark_results_test.list/benchmark_suite/vitis_mobilenet_edp/log"),
        get_latest_log_dir("/scratch/patrick/codesign/test/regression_results/benchmark_results_test.list/benchmark_suite/vitis_mobilenet_ed2/log"),
    ]
    obj_list = [
        "delay",
        "energy",
        "edp",
        "ed2"
    ]
    plot_results_in_table(results_dirs, obj_list)

    results_dirs = [
        get_latest_log_dir("/scratch/patrick/codesign/test/regression_results/benchmark_results_test.list/benchmark_suite/vitis_lenet_delay/log"),
        get_latest_log_dir("/scratch/patrick/codesign/test/regression_results/benchmark_results_test.list/benchmark_suite/vitis_lenet_energy/log"),
        get_latest_log_dir("/scratch/patrick/codesign/test/regression_results/benchmark_results_test.list/benchmark_suite/vitis_lenet_edp/log"),
        get_latest_log_dir("/scratch/patrick/codesign/test/regression_results/benchmark_results_test.list/benchmark_suite/vitis_lenet_ed2/log"),
    ]
    obj_list = [
        "delay",
        "energy",
        "edp",
        "ed2"
    ]
    plot_results_in_table(results_dirs, obj_list)"""
    """results_dirs = [
        get_latest_log_dir("/scratch/patrick/codesign/logs")
    ]
    obj_list = [
        "delay"
    ]
    plot_results_in_table(results_dirs, obj_list)"""
    benchmark_names = [
        "mobilenet",
        "lenet"
    ]
    plot_results_from_regression_results(os.path.join(os.path.dirname(__file__), "regression_results/benchmark_results_test.list/benchmark_suite"), benchmark_names)