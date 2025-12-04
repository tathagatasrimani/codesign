import json
import os

import matplotlib.pyplot as plt
import numpy as np
import shutil
from test.visualize_block_vectors import get_latest_log_dir


def plot_results_in_table(results_dirs, obj_list, save_path, top_block_name):
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
        if os.path.exists(os.path.join(results_dir, "block_vectors", f"block_vectors_forward_pass_{len(param_data)-1}.json")):
            activity_factor_path = os.path.join(results_dir, "block_vectors", f"block_vectors_forward_pass_{len(param_data)-1}.json")
        else:
            activity_factor_path = os.path.join(results_dir, "block_vectors", f"block_vectors_forward_pass_{len(param_data)-2}.json")
        with open(activity_factor_path, "r") as f:
            activity_factor_data = json.load(f)
        activity_factor = activity_factor_data[top_block_name]["top"]["computation_activity_factor"]
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

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def aggregate_wire_delay_plots(results_dirs, obj_list, save_path):
    """
    Aggregate all wire delay plots into a single figure.

    - One subplot per objective
    - Half of the plots on the top row, half on the bottom row
    - Each subplot shows the corresponding figs/wire_delays_over_iters.png
    - Titles are taken from obj_list

    The combined figure is saved to `save_path`.
    """
    num_plots = len(results_dirs)
    if num_plots == 0:
        return

    # Arrange in 2 rows: top and bottom. Use ceil(num_plots / 2) columns.
    n_cols = (num_plots + 1) // 2
    n_rows = 2 if num_plots > 1 else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Normalize axes to a flat list for easy indexing
    if n_rows == 1:
        # axes is 1D array
        axes_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes_list = axes.flatten()

    # Plot each image into its subplot
    for i, (results_dir, obj_name) in enumerate(zip(results_dirs, obj_list)):
        ax = axes_list[i]
        wire_delay_path = os.path.join(results_dir, "figs", "wire_delays_over_iters.png")
        wire_delay_img = plt.imread(wire_delay_path)
        ax.imshow(wire_delay_img)
        ax.set_title(f"{obj_name}")
        ax.axis("off")

    # Turn off any unused axes (when num_plots is odd)
    for j in range(num_plots, len(axes_list)):
        axes_list[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def aggregate_param_trend_plots(results_dirs, obj_list, save_path, plot_list_map):
    """
    Aggregate all parameter trend plots for each objective into a single figure.

    For each results_dir / objective:
      - One subplot per parameter trend (supply voltage, Vth, etc.)
      - Half of the plots on the top row, half on the bottom row
      - Each subplot shows the corresponding *_over_iters.png image

    The combined figure is saved to `save_path`.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, results_dir in enumerate(results_dirs):
        fig_names = [plot+"_over_iters.png" for plot in plot_list_map]
        fig_paths = [os.path.join(results_dir, "figs", fig_name) for fig_name in fig_names]
        num_plots = len(fig_paths)
        # Arrange in 2 rows: top and bottom. Use ceil(num_plots / 2) columns.
        n_cols = (num_plots + 1) // 2
        n_rows = 2 if num_plots > 1 else 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))

        # Normalize axes to a flat list for easy indexing
        if n_rows == 1:
            # axes is 1D array
            axes_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes_list = axes.flatten()
        # Plot each image into its own subplot
        for j, fig_path in enumerate(fig_paths):
            ax = axes_list[j]
            param_trend_img = plt.imread(fig_path)
            ax.imshow(param_trend_img)
            ax.axis("off")

        # Turn off any unused axes (when num_plots is odd)
        for j in range(num_plots, len(axes_list)):
            axes_list[j].axis("off")

        # Figure-level title for the entire grid
        plt.tight_layout()
        plt.savefig(save_path+"/"+obj_list[i]+"_param_trend.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

def plot_results_from_regression_results(regression_results_dir, benchmark_names, top_block_names):
    save_dir = os.path.join(regression_results_dir, "../aggregate_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, benchmark_name in enumerate(benchmark_names):
        save_sub_dir = os.path.join(save_dir, benchmark_name)
        if not os.path.exists(save_sub_dir):
            os.makedirs(save_sub_dir)
        sub_dirs = [d for d in os.listdir(regression_results_dir) if d.find("_" + benchmark_name) != -1]
        results_dirs = []
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(regression_results_dir, sub_dir)
            result_dir = get_latest_log_dir(os.path.join(sub_dir_path, "log"))
            results_dirs.append(result_dir)
        obj_list = [sub_dir.split("/")[-1] for sub_dir in sub_dirs]
        plot_results_in_table(results_dirs, obj_list, os.path.join(save_sub_dir, f"{benchmark_name}_table.png"), top_block_names[i])
        aggregate_wire_delay_plots(
            results_dirs,
            obj_list,
            os.path.join(save_sub_dir, f"{benchmark_name}_wire_delays.png"),
        )
        plot_list_map = ["supply voltage", "effective threshold voltage", "gate width", "gate length", "t_ox", "k_gate"]
        aggregate_param_trend_plots(
            results_dirs,
            obj_list,
            os.path.join(save_sub_dir, f"{benchmark_name}_param_trend"),
            plot_list_map,
        )
        plot_list_map = ["sensitivities/all_sensitivities", "constraint_slack/all_constraint_slack"]
        aggregate_param_trend_plots(
            results_dirs,
            obj_list,
            os.path.join(save_sub_dir, f"{benchmark_name}_sensitivity_trend"),
            plot_list_map,
        )
        for i, result_dir in enumerate(results_dirs):
            shutil.copytree(os.path.join(result_dir, "figs/block_vectors_visualization"), os.path.join(save_sub_dir, f"block_vectors_visualization/{obj_list[i]}"), dirs_exist_ok=True)
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
        "llama",
        #"lenet"
    ]
    top_block_names = [
        "llama",
        #"forward"
    ]
    plot_results_from_regression_results(os.path.join(os.path.dirname(__file__), "regression_results/benchmark_results_test.list/benchmark_suite"), benchmark_names, top_block_names)