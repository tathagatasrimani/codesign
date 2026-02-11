from dataclasses import dataclass
from typing import Dict, List, Any, Optional

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
        return

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

    # --- Plot 1: 2D scatter of Ieff vs Ioff ---
    plot_2d_scatter(
        top_results=top_results,
        x_attr='Ieff',
        y_attr='Ioff',
        x_label='Ieff (A)',
        y_label='Ioff (A)',
        title=f'Design Space Map: Ieff vs Ioff',
        filename='ieff_ioff_2d',
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

    # --- Plot 2: 3D scatter of delay vs dynamic energy vs passive power ---
    # Note: matplotlib 3D axes don't support set_xscale('log'), so we manually transform to log10
    delays = np.array([r.delay + eps for r in top_results])
    dynamic_energies = np.array([r.dynamic_energy + eps for r in top_results])
    passive_powers = np.array([r.leakage_power + eps for r in top_results])

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
    
    # Line plots for delay and passive power
    plot_metric_lines(
        top_results=top_results,
        metrics=['L', 'W', 'V_dd', 'V_th', 'tox'],
        labels=['L (m)', 'W (m)', 'V_dd (V)', 'V_th (V)', 'tox (m)'],
        title_prefix='Design Space Map',
        filename_prefix='L_W_V_dd_V_th_tox',
        colors=colors,
        iteration=iteration,
        top_percent=top_percent,
        n_top=n_top,
        n_valid=len(valid_results),
        obj_type=obj_type,
        output_dir=output_dir,
        eps=eps,
        scale=['log', 'log', 'linear', 'linear', 'log']
    )
    plot_2d_scatter(
        top_results=top_results,
        x_attr='dynamic_energy',
        y_attr='leakage_power',
        x_label='Dynamic Energy (J)',
        y_label='Leakage Power (W)',
        title=f'Design Space Map: Dynamic Energy vs Leakage Power',
        filename='dynamic_energy_leakage_power_2d',
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

    return top_results