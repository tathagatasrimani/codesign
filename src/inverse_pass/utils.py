import os
import glob
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)
# third party
import pyomo.environ as pyo
import sympy as sp
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Plot style configuration ---
PLOT_STYLE = {
    'figure.figsize': (12, 7),
    "font.size": 50,
    "axes.titlesize": 60,
    "axes.labelsize": 50,
    "xtick.labelsize": 50,
    "ytick.labelsize": 50,
    "legend.fontsize": 50,
    "figure.titlesize": 60
}

def apply_plot_style():
    """Apply consistent styling to matplotlib plots."""
    for key, value in PLOT_STYLE.items():
        plt.rcParams[key] = value

@dataclass
class DesignPointResult:
    """Stores metrics for a single design point evaluation."""
    design_point: Dict[str, Any]
    obj_value: float
    delay: float
    dynamic_energy: float
    leakage_power: float
    total_power: float
    clk_period: float
    Ieff: float
    Ioff: float
    L: float
    W: float
    V_dd: float
    V_th: float
    tox: float
    satisfies_constraints: bool
    constraint_violations: Dict[str, float] = field(default_factory=dict)
    # Block-level scalar metrics (from ObjectiveEvaluator)
    execution_time: float = 0.0       # block execution time (ns)
    total_active_energy: float = 0.0  # block active energy (nJ)
    total_passive_energy: float = 0.0 # block passive energy (nJ)
    total_area: float = 0.0           # block area (um^2)
    # Latency breakdown (critical path ns and % per category)
    latency_breakdown: Dict[str, float] = field(default_factory=lambda: {"clk": 0.0, "logic": 0.0, "memory": 0.0, "wire": 0.0})
    latency_breakdown_pct: Dict[str, float] = field(default_factory=lambda: {"clk": 0.0, "logic": 0.0, "memory": 0.0, "wire": 0.0})
    latency_memory_by_block: Dict[str, float] = field(default_factory=dict)
    latency_memory_by_block_pct: Dict[str, float] = field(default_factory=dict)
    total_logic_ops: int = 0
    total_memory_ops: int = 0
    # Active energy breakdown (nJ and % per category)
    active_energy_breakdown: Dict[str, float] = field(default_factory=lambda: {"logic": 0.0, "memory": 0.0, "wire": 0.0})
    active_energy_breakdown_pct: Dict[str, float] = field(default_factory=lambda: {"logic": 0.0, "memory": 0.0, "wire": 0.0})
    active_energy_memory_by_block: Dict[str, float] = field(default_factory=dict)
    active_energy_memory_by_block_pct: Dict[str, float] = field(default_factory=dict)
    # Passive power breakdown (W and % per category; wires excluded)
    passive_power_breakdown: Dict[str, float] = field(default_factory=lambda: {"logic": 0.0, "memory": 0.0})
    passive_power_breakdown_pct: Dict[str, float] = field(default_factory=lambda: {"logic": 0.0, "memory": 0.0})
    passive_power_memory_by_block: Dict[str, float] = field(default_factory=dict)
    passive_power_memory_by_block_pct: Dict[str, float] = field(default_factory=dict)
    # Loop II breakdown: loop_name -> {II, resource_II, recurrence_II, delay_1x_ns, bottleneck, critical_recurrence_node, critical_recurrence_ns}
    loop_ii_info: Dict[str, Any] = field(default_factory=dict)

def plot_2d_scatter(
    top_results: List[DesignPointResult],
    x_attr: str,
    y_attr: str,
    x_label: str,
    y_label: str,
    title: str,
    filename: str,
    colors: List[float],
    iteration: int,
    top_percent: float,
    n_top: int,
    n_valid: int,
    obj_type: str = "Objective",
    output_dir: str = None,
    eps: float = 1e-30,
    log_scale: bool = True
):
    """
    Create a 2D scatter plot of two variables from design results.
    
    Args:
        top_results: List of top DesignPointResult objects to plot
        x_attr: Attribute name for x-axis values (e.g., 'Ieff', 'delay', 'dynamic_energy')
        y_attr: Attribute name for y-axis values (e.g., 'Ioff', 'leakage_power')
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Plot title
        filename: Base filename for saving (without extension)
        colors: List of color values for each point (normalized objective values)
        iteration: Current optimization iteration number
        top_percent: Fraction of top designs being visualized
        n_top: Number of top designs being plotted
        n_valid: Total number of valid designs
        output_dir: Directory to save the plot (if None, displays interactively)
        eps: Small offset to add to values (for log scale handling of zeros)
        log_scale: Whether to use log scale for both axes (default: True)
    """
    # Apply consistent plot styling
    apply_plot_style()

    # Separate valid and invalid results
    valid_results = [r for r in top_results if r.satisfies_constraints]
    invalid_results = [r for r in top_results if not r.satisfies_constraints]

    # Extract x and y values for valid results
    x_vals_valid = [getattr(r, x_attr) + eps for r in valid_results]
    y_vals_valid = [getattr(r, y_attr) + eps for r in valid_results]
    colors_valid = [colors[i] for i, r in enumerate(top_results) if r.satisfies_constraints]

    # Extract x and y values for invalid results
    x_vals_invalid = [getattr(r, x_attr) + eps for r in invalid_results]
    y_vals_invalid = [getattr(r, y_attr) + eps for r in invalid_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot valid results with colors - no label, we'll create custom legend
    if valid_results:
        scatter = ax.scatter(
            x_vals_valid,
            y_vals_valid,
            c=colors_valid,
            cmap='viridis_r',
            s=120,
            alpha=0.7,
            edgecolors='white',
            linewidths=0.5
        )
        # Dummy scatter with gray color for legend (represents all colored circles)
        ax.scatter([], [], c='gray', s=120, alpha=0.7, edgecolors='white', label='Valid Designs')
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        title_txt = f'SYSTEM {obj_type.upper()}'
        title_txt = title_txt.lower().title()
        cbar.set_label(title_txt, fontsize=18, labelpad=2, fontweight='bold')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Best', 'Worst'], fontweight='bold')
        cbar.ax.invert_yaxis()
        cbar.ax.tick_params(labelsize=14)

    # Plot invalid results with black X markers
    if invalid_results:
        ax.scatter(
            x_vals_invalid,
            y_vals_invalid,
            c='dimgray',
            marker='x',
            s=80,
            alpha=0.6,
            linewidths=1.5,
            label='Power Budget Violation',
            zorder=5
        )

    ax.set_xlabel(x_label, fontsize=18, labelpad=10)
    ax.set_ylabel(y_label, fontsize=18, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=15)

    # Mark the best valid design
    if valid_results:
        best_valid_idx = 0
        ax.scatter([x_vals_valid[best_valid_idx]], [y_vals_valid[best_valid_idx]],
                  c='red', s=400, marker='*', label='Best Design', zorder=6,
                  edgecolors='darkred', linewidths=1)

    ax.legend(fontsize=14, loc='best', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f'{filename}_iteration_{iteration}.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved 2D {x_attr}/{y_attr} plot to {filepath}")
        plt.close(fig)
    else:
        plt.show()


def plot_metric_lines(
    top_results: List[DesignPointResult],
    metrics: List[str],
    labels: List[str],
    title_prefix: str,
    filename_prefix: str,
    colors: List[float],
    iteration: int,
    top_percent: float,
    n_top: int,
    n_valid: int,
    obj_type: str = "Objective",
    output_dir: str = None,
    eps: float = 1e-30,
    scale: List[str] = None
):
    """
    Create horizontal line plots for one or more metrics from design results.
    All metrics are plotted in a single figure with subplots, where each subplot
    has points plotted along a horizontal line, x-axis is the metric value and 
    colors represent design rank/objective.
    
    Args:
        top_results: List of top DesignPointResult objects to plot (already sorted by objective)
        metrics: List of attribute names to plot (e.g., ['delay', 'leakage_power'])
        labels: List of labels for each metric (e.g., ['Delay (s)', 'Passive Power (W)'])
        title_prefix: Prefix for plot titles (e.g., 'Top 10% Designs')
        filename_prefix: Prefix for filenames (e.g., 'delay_line')
        colors: List of color values for each point (normalized objective values, 0=best)
        iteration: Current optimization iteration number
        top_percent: Fraction of top designs being visualized
        n_top: Number of top designs being plotted
        n_valid: Total number of valid designs
        output_dir: Directory to save the plots (if None, displays interactively)
        eps: Small offset to add to values (for log scale handling of zeros)
        scale: List of strings indicating the scale for each axis (default: ['linear', 'linear', 'linear', 'linear', 'linear'])
    """
    if scale is None:
        scale = ['linear'] * len(metrics)
    if len(metrics) != len(labels):
        raise ValueError(f"Number of metrics ({len(metrics)}) must match number of labels ({len(labels)})")

    # Apply consistent plot styling
    apply_plot_style()

    n_metrics = len(metrics)

    # Create a single figure with subplots (one row per metric)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(16, 2.5 * n_metrics + 2))
    
    # Handle case where there's only one metric (axes won't be an array)
    if n_metrics == 1:
        axes = [axes]
    
    # Create a shared colorbar (use the first scatter plot's colormap)
    scatter_objects = []
    
    for idx, (metric, label, ax) in enumerate(zip(metrics, labels, axes)):
        # Separate valid and invalid results
        valid_results = [r for r in top_results if r.satisfies_constraints]
        invalid_results = [r for r in top_results if not r.satisfies_constraints]
        
        # Extract metric values for valid results
        metric_vals_valid = [getattr(r, metric) + eps for r in valid_results]
        colors_valid = [colors[i] for i, r in enumerate(top_results) if r.satisfies_constraints]
        
        # Extract metric values for invalid results
        metric_vals_invalid = [getattr(r, metric) + eps for r in invalid_results]
        
        # Y-axis is fixed at 0 (horizontal line)
        y_pos_valid = [0] * len(metric_vals_valid) if valid_results else []
        y_pos_invalid = [0] * len(metric_vals_invalid) if invalid_results else []
        
        # Plot valid points along horizontal line, colored by rank/objective
        if valid_results:
            scatter = ax.scatter(
                metric_vals_valid,
                y_pos_valid,
                c=colors_valid,
                cmap='viridis_r',
                s=120,
                alpha=0.8,
                zorder=5,
                edgecolors='black',
                linewidths=0.8
            )
            if idx == 0:  # Only add to scatter_objects for colorbar
                scatter_objects.append(scatter)
        
        # Plot invalid results with black X markers
        if invalid_results:
            ax.scatter(
                metric_vals_invalid,
                y_pos_invalid,
                c='black',
                marker='x',
                s=150,
                alpha=0.8,
                linewidths=2,
                zorder=6,
                label='Invalid Designs' if idx == 0 else ''
            )
        
        # Draw a horizontal line
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.3, zorder=1)
        
        ax.set_xlabel(label, fontsize=14, labelpad=8)
        ax.set_ylabel('')
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_ylim(-0.1, 0.1)  # Small range to keep line visible
        if scale[idx] == 'log':
            ax.set_xscale('log')

        # Shorter title to avoid overlap
        ax.set_title(f'{label}', fontsize=15, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.3, axis='x')
        ax.tick_params(axis='x', labelsize=12)
        
        # Mark the best valid design
        if valid_results:
            ax.scatter([metric_vals_valid[0]], [0], c='red', s=300, marker='*', 
                      label='Best Design' if idx == 0 else '', zorder=7, 
                      edgecolors='black', linewidths=1.2)
    
    # Add a shared colorbar if there are valid results
    if scatter_objects:
        cbar = fig.colorbar(scatter_objects[0], ax=axes, orientation='horizontal',
                            pad=0.12, aspect=40, location='bottom')
        
        title_txt = f'SYSTEM {obj_type.upper()}'
        title_txt = title_txt.lower().title()
        cbar.set_label(title_txt, fontsize=16, labelpad=2, fontweight='bold')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Best', 'Worst'], fontweight='bold')
        cbar.ax.invert_xaxis()  # horizontal colorbar
        cbar.ax.tick_params(labelsize=14)

    # Add overall title
    fig.suptitle(f'{title_prefix}',
                 fontsize=18, fontweight='bold', y=0.995)

    # Add legend only once (from first subplot)
    if n_metrics > 0:
        axes[0].legend(loc='upper right', fontsize=12, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # Leave more space at bottom for colorbar

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f'{filename_prefix}_line_iteration_{iteration}.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved line plots for {n_metrics} metrics to {filepath}")
        plt.close(fig)
    else:
        plt.show()


def visualize_memory_latency_scatter(
    top_results: List[DesignPointResult],
    iteration: int,
    obj_type: str,
    colors: List[float],
    output_dir: str = None,
):
    """
    For each memory block, plot hit latency vs write latency as a 2D scatter
    colored by objective value.  Blocks where either latency column is absent
    or where all top results share the same design point are skipped.
    """
    mem_entries = {}
    for r, color in zip(top_results, colors):
        for mem_name, mem_info in r.design_point.get("memory", {}).items():
            if not isinstance(mem_info, dict):
                continue
            mem_entries.setdefault(mem_name, []).append((mem_info, color))

    varied_blocks = {
        name: entries for name, entries in mem_entries.items()
        if len({e[0].get("index") for e in entries}) > 1
    }
    if not varied_blocks:
        return

    eps = 1e-30
    for mem_name, entries in varied_blocks.items():
        mem_infos  = [e[0] for e in entries]
        mem_colors = [e[1] for e in entries]

        numeric_cols = []
        for k in mem_infos[0]:
            if k in ("index", "capacity"):
                continue
            try:
                float(mem_infos[0][k])
                numeric_cols.append(k)
            except (TypeError, ValueError):
                pass

        hit_col   = next((c for c in numeric_cols if "hit"   in c.lower() and "latency" in c.lower()), None)
        write_col = next((c for c in numeric_cols if "write" in c.lower() and "latency" in c.lower()), None)
        if hit_col is None or write_col is None:
            continue

        hit_lats   = np.array([float(m.get(hit_col,   0)) + eps for m in mem_infos])
        write_lats = np.array([float(m.get(write_col, 0)) + eps for m in mem_infos])

        capacity = mem_infos[0].get("capacity") if mem_infos else None
        cap_str  = f"  [{capacity}]" if capacity else ""

        plt.rcdefaults()
        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(hit_lats[1:], write_lats[1:], c=mem_colors[1:],
                             cmap='viridis_r', s=100, alpha=0.75,
                             edgecolors='white', linewidths=0.3)
        ax.scatter([hit_lats[0]], [write_lats[0]], c='red', s=600, marker='*',
                   label='Best Design', zorder=6, edgecolors='black', linewidths=2)
        ax.scatter([], [], c='gray', s=100, alpha=0.75,
                   edgecolors='white', label='Valid Design')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(hit_col,   fontsize=14, labelpad=6, fontweight='bold')
        ax.set_ylabel(write_col, fontsize=14, labelpad=6, fontweight='bold')
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'Memory Hit vs Write Latency: {mem_name}{cap_str}',
                     fontsize=15, fontweight='bold', pad=10)
        ax.legend(fontsize=12, loc='upper left', framealpha=0.9)

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(f'System {obj_type.upper()}', fontsize=14, labelpad=2, fontweight='bold')
        obj_vals = [r.obj_value for r in top_results]
        obj_min, obj_max = min(obj_vals), max(obj_vals)
        if obj_max > obj_min:
            log_min = np.log(obj_min + eps)
            log_max = np.log(obj_max + eps)
            tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
            tick_vals = [np.exp(log_min + c * (log_max - log_min)) for c in tick_positions]
            tick_labels = [f'{v:.2e}' for v in tick_vals]
            tick_labels[0] = f'Best\n{tick_vals[0]:.2e}'
            tick_labels[-1] = f'Worst\n{tick_vals[-1]:.2e}'
        else:
            tick_positions = [0, 1]
            tick_labels = ['Best', 'Worst']
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels, fontweight='bold')
        cbar.ax.invert_yaxis()
        cbar.ax.tick_params(labelsize=10)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            safe_name = mem_name.replace('/', '_').replace(' ', '_')
            filepath = os.path.join(output_dir, f'memory_{capacity}_{safe_name}_hit_vs_write_latency_iter_{iteration}.png')
            plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.3)
            logger.info(f"Saved hit vs write latency plot for '{mem_name}' ({capacity}KB) to {filepath}")
            plt.close(fig)
        else:
            plt.show()


def visualize_top_memory_designs(
    top_results: List[DesignPointResult],
    iteration: int,
    obj_type: str,
    colors: List[float],
    output_dir: str = None,
):
    """
    Create 3D scatter plots for memory design choices among the top results.

    For each memory block where the design point index varies across the top
    results, plots area vs latency vs leakage (log10 scale) colored by objective
    value — matching the style of the logic delay/energy/power 3D plot.
    Blocks where all top results chose the same design point are skipped.
    """
    # Collect memory info per block: {mem_name: [(info_dict, color), ...]}
    mem_entries = {}
    for r, color in zip(top_results, colors):
        for mem_name, mem_info in r.design_point.get("memory", {}).items():
            if not isinstance(mem_info, dict):
                continue
            mem_entries.setdefault(mem_name, []).append((mem_info, color))

    eps = 1e-30
    for mem_name, entries in mem_entries.items():
        mem_infos = [e[0] for e in entries]
        mem_colors = [e[1] for e in entries]

        # Find numeric columns (excluding non-numeric metadata keys)
        numeric_cols = []
        for k in mem_infos[0]:
            if k in ("index", "capacity"):
                continue
            try:
                float(mem_infos[0][k])
                numeric_cols.append(k)
            except (TypeError, ValueError):
                pass

        if len(numeric_cols) < 2:
            continue

        area_col    = next((c for c in numeric_cols if "area"    in c.lower()), numeric_cols[0])
        latency_col = next((c for c in numeric_cols if "latency" in c.lower()), None)

        if latency_col is None or latency_col == area_col:
            latency_col = next((c for c in numeric_cols if c != area_col), None)
        if latency_col is None:
            continue

        leakage_col        = next((c for c in numeric_cols if "leak"  in c.lower()), None)
        hit_energy_col     = next((c for c in numeric_cols if "hit"   in c.lower() and "energy"  in c.lower()), None)
        write_energy_col   = next((c for c in numeric_cols if "write" in c.lower() and "energy"  in c.lower()), None)
        write_latency_col  = next((c for c in numeric_cols if "write" in c.lower() and "latency" in c.lower()), None)

        # Each entry: (z column name, file suffix)
        z_axes = []
        if leakage_col    is not None: z_axes.append((leakage_col,    "leakage"))
        if hit_energy_col is not None: z_axes.append((hit_energy_col, "hit_energy"))
        if not z_axes:
            continue

        areas     = np.array([float(m.get(area_col,    0)) + eps for m in mem_infos])
        latencies = np.array([float(m.get(latency_col, 0)) + eps for m in mem_infos])
        log_areas     = np.log10(areas)
        log_latencies = np.log10(latencies)

        capacity = mem_infos[0].get("capacity") if mem_infos else None
        cap_str  = f"  [{capacity}]" if capacity else ""

        best_idx  = 0
        other_idx = list(range(1, len(mem_infos)))

        for z_col, z_suffix in z_axes:
            z_vals = np.array([float(m.get(z_col, 0)) + eps for m in mem_infos])
            log_z  = np.log10(z_vals)

            plt.rcdefaults()
            fig = plt.figure(figsize=(9, 6))
            ax  = fig.add_subplot(111, projection='3d', computed_zorder=False)

            if other_idx:
                scatter = ax.scatter(
                    log_areas[other_idx], log_latencies[other_idx], log_z[other_idx],
                    c=[mem_colors[i] for i in other_idx],
                    cmap='viridis_r', s=100, alpha=0.75,
                    edgecolors='white', linewidths=0.3, zorder=1,
                )
            else:
                scatter = ax.scatter([], [], [], c=[], cmap='viridis_r')
            ax.scatter([], [], [], c='gray', s=100, alpha=0.75,
                       edgecolors='white', label='Valid Design')
            ax.scatter([log_areas[best_idx]], [log_latencies[best_idx]], [log_z[best_idx]],
                       c='red', s=600, marker='*', label='Best Design',
                       edgecolors='black', linewidths=2, zorder=100)

            ax.set_xlabel(f'{area_col} [log₁₀]',    fontsize=14, labelpad=8, fontweight='bold')
            ax.set_ylabel(f'{latency_col} [log₁₀]', fontsize=14, labelpad=8, fontweight='bold')
            ax.set_zlabel(f'{z_col} [log₁₀]',       fontsize=14, labelpad=8, fontweight='bold')
            ax.tick_params(axis='x', labelsize=11, pad=5)
            ax.tick_params(axis='y', labelsize=11, pad=5)
            ax.tick_params(axis='z', labelsize=11, pad=5)
            ax.view_init(elev=20, azim=45)

            ax.set_title(f'Memory Design Space: {mem_name}{cap_str}  (log₁₀ scale)',
                         fontsize=15, fontweight='bold', pad=10)
            ax.legend(fontsize=12, loc='upper left', framealpha=0.9)

            title_txt = f'System {obj_type.upper()}'
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02, aspect=25)
            cbar.set_label(title_txt, fontsize=14, labelpad=2, fontweight='bold')
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Best', 'Worst'], fontweight='bold')
            cbar.ax.invert_yaxis()
            cbar.ax.tick_params(labelsize=12)

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                safe_name = mem_name.replace('/', '_').replace(' ', '_')
                filepath = os.path.join(output_dir, f'memory_{capacity}_{safe_name}_{z_suffix}_iter_{iteration}.png')
                plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.5)
                logger.info(f"Saved memory design plot for '{mem_name}' ({capacity}KB) ({z_suffix}) to {filepath}")
                plt.close(fig)
            else:
                plt.show()


def visualize_top_fu_designs(
    top_results: List[DesignPointResult],
    iteration: int,
    obj_type: str,
    colors: List[float],
    output_dir: str = None,
):
    """
    Create 3D scatter plots for FU logic design choices among the top results.

    For each FU resource where the design point index varies across the top results,
    plots delay vs E_act_inv vs P_pass_inv (log10 scale) colored by objective value —
    matching the style of the aggregate logic delay/energy/power 3D plot.
    FU resources where all top results chose the same design point are skipped.
    """
    fu_entries = {}
    for r, color in zip(top_results, colors):
        for fu_name, fu_info in r.design_point.get("fu_logic", {}).items():
            if not isinstance(fu_info, dict):
                continue
            fu_entries.setdefault(fu_name, []).append((fu_info, color))

    eps = 1e-30
    for fu_name, entries in fu_entries.items():
        fu_infos  = [e[0] for e in entries]
        fu_colors = [e[1] for e in entries]

        if len({e.get("index") for e in fu_infos}) <= 1:
            continue

        delays   = np.array([float(m.get("delay",      0)) + eps for m in fu_infos])
        energies = np.array([float(m.get("E_act_inv",  0)) + eps for m in fu_infos])
        powers   = np.array([float(m.get("P_pass_inv", 0)) + eps for m in fu_infos])
        log_delays   = np.log10(delays)
        log_energies = np.log10(energies)
        log_powers   = np.log10(powers)

        function  = fu_infos[0].get("function", fu_name)
        best_idx  = 0
        other_idx = list(range(1, len(fu_infos)))

        plt.rcdefaults()
        fig = plt.figure(figsize=(9, 6))
        ax  = fig.add_subplot(111, projection='3d', computed_zorder=False)

        if other_idx:
            scatter = ax.scatter(
                log_delays[other_idx], log_energies[other_idx], log_powers[other_idx],
                c=[fu_colors[i] for i in other_idx],
                cmap='viridis_r', s=100, alpha=0.75,
                edgecolors='white', linewidths=0.3, zorder=1,
            )
        else:
            scatter = ax.scatter([], [], [], c=[], cmap='viridis_r')
        ax.scatter([], [], [], c='gray', s=100, alpha=0.75,
                   edgecolors='white', label='Valid Design')
        ax.scatter([log_delays[best_idx]], [log_energies[best_idx]], [log_powers[best_idx]],
                   c='red', s=600, marker='*', label='Best Design',
                   edgecolors='black', linewidths=2, zorder=100)

        ax.set_xlabel('Delay [log₁₀]',   fontsize=14, labelpad=8, fontweight='bold')
        ax.set_ylabel('E_act [log₁₀]',   fontsize=14, labelpad=8, fontweight='bold')
        ax.set_zlabel('P_pass [log₁₀]',  fontsize=14, labelpad=8, fontweight='bold')
        ax.tick_params(axis='x', labelsize=11, pad=5)
        ax.tick_params(axis='y', labelsize=11, pad=5)
        ax.tick_params(axis='z', labelsize=11, pad=5)
        ax.view_init(elev=20, azim=45)

        ax.set_title(f'FU Design Space: {fu_name} ({function})  (log₁₀ scale)',
                     fontsize=15, fontweight='bold', pad=10)
        ax.legend(fontsize=12, loc='upper left', framealpha=0.9)

        title_txt = f'System {obj_type.upper()}'
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02, aspect=25)
        cbar.set_label(title_txt, fontsize=14, labelpad=2, fontweight='bold')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Best', 'Worst'], fontweight='bold')
        cbar.ax.invert_yaxis()
        cbar.ax.tick_params(labelsize=12)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            safe_name = fu_name.replace('/', '_').replace(' ', '_')
            filepath = os.path.join(output_dir, f'fu_{safe_name}_delay_energy_power_iter_{iteration}.png')
            plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.5)
            logger.info(f"Saved FU design plot for '{fu_name}' ({function}) to {filepath}")
            plt.close(fig)
        else:
            plt.show()


def visualize_top_designs(all_results: List[DesignPointResult], iteration: int, obj_type: str, top_percent: float = 0.1, output_dir: str = None):
    """
    Create visualizations of the top designs by objective value.
    Generates two plots:
    1. 2D scatter plot of Ieff vs Ioff
    2. 3D scatter plot of delay vs dynamic energy vs passive power

    Args:
        all_results: List of DesignPointResult from all workers
        iteration: Current optimization iteration number
        top_percent: Fraction of top designs to visualize (default 10%)
        output_dir: Directory to save the plots (if None, displays interactively)

    Returns:
        List of top DesignPointResult objects
    """
    # Reset matplotlib state to prevent accumulation between iterations
    plt.close('all')
    plt.rcdefaults()

    # Filter to only designs that satisfy constraints
    #valid_results = [r for r in all_results if r.satisfies_constraints]
    valid_results = all_results

    if not valid_results:
        logger.warning("No valid designs to visualize")
        return None, None

    # Sort by objective value (ascending = better)
    sorted_results = sorted(valid_results, key=lambda r: r.obj_value)

    # Take top percentage
    n_top = max(1, int(len(sorted_results) * top_percent))
    top_results = sorted_results[:n_top]

    # Extract metrics for plotting
    eps = 1e-30  # tiny offset for any zero values
    obj_values = [r.obj_value for r in top_results]

    # Normalize objective values for coloring using log scale (0 = best, 1 = worst among top)
    obj_min, obj_max = min(obj_values), max(obj_values)
    if obj_max > obj_min:
        # Use log scale for normalization
        log_obj_values = [np.log(v + eps) for v in obj_values]
        log_obj_min, log_obj_max = min(log_obj_values), max(log_obj_values)
        if log_obj_max > log_obj_min:
            colors = [(log_v - log_obj_min) / (log_obj_max - log_obj_min) for log_v in log_obj_values]
        else:
            colors = [0.0] * len(obj_values)
    else:
        colors = [0.0] * len(obj_values)

    # 2D: delay vs passive power
    plot_2d_scatter(
        top_results=top_results,
        x_attr='delay',
        y_attr='leakage_power',
        x_label='Stage Delay (s)',
        y_label='Passive Power (W)',
        title=f'Design Space Map: Delay vs Passive Power',
        filename='delay_passive_power_2d',
        colors=colors,
        iteration=iteration,
        top_percent=top_percent,
        n_top=n_top,
        n_valid=len(valid_results),
        obj_type=obj_type,
        output_dir=output_dir,
        eps=eps,
        log_scale=True
    )

    # 3D: delay vs dynamic energy vs passive power
    # Note: matplotlib 3D axes don't support set_xscale('log'), so we manually transform to log10
    delays = np.array([r.delay + eps for r in top_results])
    dynamic_energies = np.array([r.dynamic_energy + eps for r in top_results])
    passive_powers = np.array([r.leakage_power + eps for r in top_results])

    # Transform to log10 for plotting
    log_delays = np.log10(delays)
    log_dynamic = np.log10(dynamic_energies)
    log_passive = np.log10(passive_powers)

    # Find the best valid design index (first valid in sorted list)
    best_valid_idx = None
    for i, r in enumerate(top_results):
        if r.satisfies_constraints:
            best_valid_idx = i
            break

    # Separate valid and invalid results (excluding best valid design)
    valid_indices = [i for i in range(len(top_results)) if top_results[i].satisfies_constraints and i != best_valid_idx]
    invalid_indices = [i for i in range(len(top_results)) if not top_results[i].satisfies_constraints]

    # Apply consistent plot styling
    apply_plot_style()

    fig2 = plt.figure(figsize=(9, 6))
    ax2 = fig2.add_subplot(111, projection='3d', computed_zorder=False)

    # Plot valid points (excluding best) - no label here, we'll create custom legend
    if valid_indices:
        scatter2 = ax2.scatter(
            log_delays[valid_indices],
            log_dynamic[valid_indices],
            log_passive[valid_indices],
            c=[colors[i] for i in valid_indices],
            cmap='viridis_r',
            s=100,
            alpha=0.75,
            edgecolors='white',
            linewidths=0.3,
            zorder=1
        )
    else:
        # Need at least one scatter for colorbar
        scatter2 = ax2.scatter([], [], [], c=[], cmap='viridis_r')

    # Create a dummy scatter with gray color for legend (represents all colored circles)
    ax2.scatter([], [], [], c='gray', s=100, alpha=0.75, edgecolors='white', label='Valid Design')

    # Plot invalid points with X markers
    if invalid_indices:
        ax2.scatter(
            log_delays[invalid_indices],
            log_dynamic[invalid_indices],
            log_passive[invalid_indices],
            c='dimgray',
            marker='x',
            s=80,
            alpha=0.6,
            linewidths=1.5,
            zorder=2,
            label='Power Budget Exceeded'
        )

    # Set axis labels - simplified
    ax2.set_xlabel('Stage Delay [s]', fontsize=14, labelpad=8, fontweight='bold')
    ax2.set_ylabel('Dynamic Energy [J]', fontsize=14, labelpad=8, fontweight='bold')
    ax2.set_zlabel('Passive Power [W]', fontsize=14, labelpad=8, fontweight='bold')

    # Title centered over the plot (shifted left to account for colorbar)
    ax2.set_title('Design Space Map  (log₁₀ scale)', fontsize=25, fontweight='bold', pad=10)

    # Style tick labels
    ax2.tick_params(axis='x', labelsize=11, pad=5)
    ax2.tick_params(axis='y', labelsize=11, pad=5)
    ax2.tick_params(axis='z', labelsize=11, pad=5)

    # Set integer-only ticks on Dynamic Energy (y) axis
    y_min, y_max = int(np.floor(log_dynamic.min())), int(np.ceil(log_dynamic.max()))
    ax2.set_yticks(range(y_min, y_max + 1))

    # Colorbar with BEST/WORST labels
    title_txt = f'SYSTEM {obj_type.upper()}'
    title_txt = title_txt.lower().title()
    cbar2 = fig2.colorbar(scatter2, ax=ax2, shrink=0.6, pad=0.02, aspect=25)
    cbar2.set_label(title_txt, fontsize=14, labelpad=2, fontweight='bold')
    cbar2.set_ticks([0, 1])
    cbar2.set_ticklabels(['Best', 'Worst'], fontweight='bold')
    cbar2.ax.invert_yaxis()
    cbar2.ax.tick_params(labelsize=12)

    # Mark the best valid design with a prominent star - plotted LAST with high zorder
    if best_valid_idx is not None:
        ax2.scatter([log_delays[best_valid_idx]], [log_dynamic[best_valid_idx]], [log_passive[best_valid_idx]],
                    c='red', s=600, marker='*', label='Best Design',
                    edgecolors='black', linewidths=2, zorder=100)
    ax2.legend(fontsize=12, loc='upper left', framealpha=0.9)

    # Adjust viewing angle for better visibility
    ax2.view_init(elev=20, azim=45)

    if output_dir:
        filepath2 = os.path.join(output_dir, f'delay_energy_power_3d_iteration_{iteration}.png')
        plt.savefig(filepath2, dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0.5)
        logger.info(f"Saved 3D delay/energy/power plot to {filepath2}")
        plt.close(fig2)
    else:
        plt.show()

    return top_results, colors


def regenerate_plots_from_log_dir(log_dir: str, obj_type: str = "edp", top_percent: float = 1):
    """
    Load pickled DesignPointResult lists from a previous log directory and
    regenerate all visualisation plots into that same directory.

    Discovers all ``all_design_point_results_iter_<N>.pkl`` files in *log_dir*
    and calls :func:`visualize_top_designs` for each iteration found.

    Args:
        log_dir:     Path to a previous run's log directory (the directory that
                     contains the .pkl files and where plots will be written).
        obj_type:    Objective type string used for axis labels (e.g. "edp").
        top_percent: Fraction of top designs to visualise (default 10 %).
    """
    pattern = os.path.join(log_dir, "all_design_point_results_iter_*.pkl")
    pkl_files = sorted(glob.glob(pattern))
    assert pkl_files, f"No design-point result pkl files found in {log_dir!r} (pattern: {pattern})"

    for pkl_path in pkl_files:
        # Extract iteration number from filename
        basename = os.path.basename(pkl_path)
        iteration = int(basename.replace("all_design_point_results_iter_", "").replace(".pkl", ""))

        with open(pkl_path, "rb") as f:
            all_results: List[DesignPointResult] = pickle.load(f)

        logger.info(f"Regenerating plots for iteration {iteration} ({len(all_results)} results) from {pkl_path}")
        top_results, colors = visualize_top_designs(all_results, iteration, obj_type,
                                                    top_percent=top_percent, output_dir=log_dir)
        if top_results is None:
            continue

        n_top = len(top_results)
        n_valid = len(all_results)
        eps = 1e-30

        plot_2d_scatter(
            top_results=top_results,
            x_attr='Ieff', y_attr='Ioff',
            x_label='Ieff (A)', y_label='Ioff (A)',
            title='Design Space Map: Ieff vs Ioff',
            filename='ieff_ioff_2d',
            colors=colors, iteration=iteration,
            top_percent=top_percent, n_top=n_top, n_valid=n_valid,
            obj_type=obj_type, output_dir=log_dir, eps=eps, log_scale=True,
        )
        plot_2d_scatter(
            top_results=top_results,
            x_attr='dynamic_energy', y_attr='leakage_power',
            x_label='Dynamic Energy (J)', y_label='Leakage Power (W)',
            title='Design Space Map: Dynamic Energy vs Leakage Power',
            filename='dynamic_energy_leakage_power_2d',
            colors=colors, iteration=iteration,
            top_percent=top_percent, n_top=n_top, n_valid=n_valid,
            obj_type=obj_type, output_dir=log_dir, eps=eps, log_scale=True,
        )
        plot_metric_lines(
            top_results=top_results,
            metrics=['L', 'W', 'V_dd', 'V_th', 'tox'],
            labels=['L (m)', 'W (m)', 'V_dd (V)', 'V_th (V)', 'tox (m)'],
            title_prefix='Design Space Map',
            filename_prefix='L_W_V_dd_V_th_tox',
            colors=colors, iteration=iteration,
            top_percent=top_percent, n_top=n_top, n_valid=n_valid,
            obj_type=obj_type, output_dir=log_dir, eps=eps,
            scale=['log', 'log', 'linear', 'linear', 'log'],
        )
        visualize_top_memory_designs(top_results, iteration, obj_type, colors, output_dir=log_dir)
        visualize_memory_latency_scatter(top_results, iteration, obj_type, colors, output_dir=log_dir)
        visualize_top_fu_designs(top_results, iteration, obj_type, colors, output_dir=log_dir)