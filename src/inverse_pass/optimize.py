# first party
import argparse
import logging
import time
import sys
import copy
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
logger = logging.getLogger(__name__)

# third party
import pyomo.environ as pyo
import sympy as sp
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# custom
from src.inverse_pass.preprocess import Preprocessor
from src.inverse_pass import curve_fit
from src.inverse_pass.constraint import Constraint

from src import sim_util
from src.hardware_model.hardwareModel import BlockVector
from src.hardware_model.objective_evaluator import ObjectiveEvaluator

multistart = False

from src.sim_util import solve_gp_with_fallback


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

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot valid results with colors
    if valid_results:
        scatter = ax.scatter(
            x_vals_valid,
            y_vals_valid,
            c=colors_valid,
            cmap='viridis_r',
            s=50,
            alpha=0.7,
            label='Valid Designs'
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Relative Objective (0=best)')
    
    # Plot invalid results with black X markers
    if invalid_results:
        ax.scatter(
            x_vals_invalid,
            y_vals_invalid,
            c='black',
            marker='x',
            s=100,
            alpha=0.8,
            linewidths=2,
            label='Invalid Designs (constraint violation)',
            zorder=5
        )
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_title(title)

    # Mark the best valid design
    if valid_results:
        best_valid_idx = 0
        ax.scatter([x_vals_valid[best_valid_idx]], [y_vals_valid[best_valid_idx]], 
                  c='red', s=200, marker='*', label='Best Design', zorder=6)
    
    ax.legend()
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f'{filename}_iteration_{iteration}.png')
        plt.savefig(filepath, dpi=150)
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
    
    n_metrics = len(metrics)
    
    # Create a single figure with subplots (one row per metric)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 2 * n_metrics + 1))
    
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
        
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('')
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_ylim(-0.1, 0.1)  # Small range to keep line visible
        if scale[idx] == 'log':
            ax.set_xscale('log')
        
        # Shorter title to avoid overlap
        ax.set_title(f'{label} ({n_top} of {n_valid} valid)', fontsize=11, pad=5)
        ax.grid(True, alpha=0.3, axis='x')
        ax.tick_params(axis='x', labelsize=9)
        
        # Mark the best valid design
        if valid_results:
            ax.scatter([metric_vals_valid[0]], [0], c='red', s=300, marker='*', 
                      label='Best Design' if idx == 0 else '', zorder=7, 
                      edgecolors='black', linewidths=1.2)
    
    # Add a shared colorbar if there are valid results
    if scatter_objects:
        cbar = fig.colorbar(scatter_objects[0], ax=axes, orientation='horizontal', 
                            pad=0.15, aspect=40, location='bottom')
        cbar.set_label('Relative Objective (0=best)', fontsize=10, labelpad=10)
    
    # Add overall title
    fig.suptitle(f'{title_prefix} ({n_top} of {n_valid} valid)', 
                 fontsize=12, y=0.995)
    
    # Add legend only once (from first subplot)
    if n_metrics > 0:
        axes[0].legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # Leave more space at bottom for colorbar
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f'{filename_prefix}_line_iteration_{iteration}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Saved line plots for {n_metrics} metrics to {filepath}")
        plt.close(fig)
    else:
        plt.show()


def visualize_top_designs(all_results: List[DesignPointResult], iteration: int, top_percent: float = 0.1, output_dir: str = None):
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
        title=f'Top {top_percent*100:.0f}% Designs: Ieff vs Ioff ({n_top} of {len(valid_results)})',
        filename='ieff_ioff_2d',
        colors=colors,
        iteration=iteration,
        top_percent=top_percent,
        n_top=n_top,
        n_valid=len(valid_results),
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
        x_label='Delay (s)',
        y_label='Passive Power (W)',
        title=f'Top {top_percent*100:.0f}% Designs: Delay vs Passive Power ({n_top} of {len(valid_results)})',
        filename='delay_passive_power_2d',
        colors=colors,
        iteration=iteration,
        top_percent=top_percent,
        n_top=n_top,
        n_valid=len(valid_results),
        output_dir=output_dir,
        eps=eps,
        log_scale=True
    )
    
    # Line plots for delay and passive power
    plot_metric_lines(
        top_results=top_results,
        metrics=['L', 'W', 'V_dd', 'V_th', 'tox'],
        labels=['L (m)', 'W (m)', 'V_dd (V)', 'V_th (V)', 'tox (m)'],
        title_prefix=f'Top {top_percent*100:.0f}% Designs',
        filename_prefix='L_W_V_dd_V_th_tox',
        colors=colors,
        iteration=iteration,
        top_percent=top_percent,
        n_top=n_top,
        n_valid=len(valid_results),
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
        title=f'Top {top_percent*100:.0f}% Designs: Dynamic Energy vs Leakage Power ({n_top} of {len(valid_results)})',
        filename='dynamic_energy_leakage_power_2d',
        colors=colors,
        iteration=iteration,
        top_percent=top_percent,
        n_top=n_top,
        n_valid=len(valid_results),
        output_dir=output_dir,
        eps=eps,
        log_scale=True
    )
    # Transform to log10 for plotting
    log_delays = np.log10(delays)
    log_dynamic = np.log10(dynamic_energies)
    log_passive = np.log10(passive_powers)

    fig2 = plt.figure(figsize=(12, 9))
    ax2 = fig2.add_subplot(111, projection='3d')

    scatter2 = ax2.scatter(
        log_delays,
        log_dynamic,
        log_passive,
        c=colors,
        cmap='viridis_r',
        s=50,
        alpha=0.7
    )

    ax2.set_xlabel('log10(Delay) [s]')
    ax2.set_ylabel('log10(Dynamic Energy) [J]')
    ax2.set_zlabel('log10(Passive Power) [W]')
    ax2.set_title(f'Top {top_percent*100:.0f}% Designs ({n_top} of {len(valid_results)} valid)')

    cbar2 = fig2.colorbar(scatter2, ax=ax2, shrink=0.6, pad=0.1)
    cbar2.set_label('Relative Objective (0=best)')

    # Mark the best design
    ax2.scatter([log_delays[0]], [log_dynamic[0]], [log_passive[0]],
                c='red', s=200, marker='*', label='Best Design')
    ax2.legend()
    plt.tight_layout()

    if output_dir:
        filepath2 = os.path.join(output_dir, f'delay_energy_power_3d_iteration_{iteration}.png')
        plt.savefig(filepath2, dpi=150)
        logger.info(f"Saved 3D delay/energy/power plot to {filepath2}")
        plt.close(fig2)
    else:
        plt.show()

    return top_results


def log_info(msg, stage):
    if stage == "before optimization":
        print(msg)
    elif stage == "after optimization":
        logger.info(msg)

def satisfies_constraints(total_power, design_point, max_system_power, tech_model):
    if total_power > max_system_power:
        logger.info(f"total power {total_power} is greater than max system power {max_system_power} for design point {design_point}")
        return False
    return True

def _worker_basic_optimization_chunk(args_tuple):
    """
    Worker function that evaluates all design points in its chunk and returns
    detailed metrics for each one.

    Args:
        args_tuple: (worker_id, chunk, evaluator, max_system_power)
            - worker_id: Worker identifier
            - chunk: List of (idx, design_point) tuples
            - evaluator: ObjectiveEvaluator instance (pickleable)
            - max_system_power: Maximum allowed system power

    Returns:
        Tuple of (worker_id, list of DesignPointResult)
    """
    worker_id, chunk, evaluator, max_system_power = args_tuple

    # Get tech_model from evaluator for convenience
    tech_model = evaluator.tech_model

    results = []
    best_obj_val = math.inf

    for idx, design_point in chunk:
        evaluator.set_params_from_design_point(design_point)
        lower_clk_period = sim_util.xreplace_safe(tech_model.delay * 150, tech_model.base_params.tech_values)
        upper_clk_period = sim_util.xreplace_safe(tech_model.delay * 5000, tech_model.base_params.tech_values)
        clk_periods = np.logspace(np.log10(lower_clk_period), np.log10(upper_clk_period), 1)
        for clk_period in clk_periods:
            evaluator.set_clk_period(clk_period)

            evaluator.calculate_objective()

            if evaluator.tech_model.base_params.tech_values[evaluator.tech_model.base_params.clk_period] > clk_period:
                logger.info(f"worker {worker_id} clk period {clk_period} is greater than minimum clk period {evaluator.tech_model.base_params.tech_values[evaluator.tech_model.base_params.clk_period]}, had to increase it")
                clk_period = evaluator.tech_model.base_params.tech_values[evaluator.tech_model.base_params.clk_period]

            # Extract metrics
            delay = sim_util.xreplace_safe(tech_model.delay * 1e-9, tech_model.base_params.tech_values)
            dynamic_energy = sim_util.xreplace_safe(tech_model.E_act_inv, tech_model.base_params.tech_values)
            leakage_power = sim_util.xreplace_safe(tech_model.P_pass_inv, tech_model.base_params.tech_values)
            total_power = evaluator.total_power
            obj_value = evaluator.obj

            # Extract Ieff and Ioff from tech model
            Ieff = sim_util.xreplace_safe(tech_model.Ieff, tech_model.base_params.tech_values)
            Ioff = sim_util.xreplace_safe(tech_model.I_sub, tech_model.base_params.tech_values)
            V_dd = sim_util.xreplace_safe(tech_model.base_params.V_dd, tech_model.base_params.tech_values)
            V_th_eff = sim_util.xreplace_safe(tech_model.V_th_eff, tech_model.base_params.tech_values)
            tox = sim_util.xreplace_safe(tech_model.base_params.tox, tech_model.base_params.tech_values)
            L = sim_util.xreplace_safe(tech_model.base_params.L, tech_model.base_params.tech_values)
            W = sim_util.xreplace_safe(tech_model.base_params.W, tech_model.base_params.tech_values)

            constraints_satisfied = satisfies_constraints(evaluator.total_power, design_point, max_system_power, tech_model)

            result = DesignPointResult(
                design_point=design_point,
                obj_value=obj_value,
                delay=delay,
                dynamic_energy=dynamic_energy,
                leakage_power=leakage_power,
                total_power=total_power,
                clk_period=clk_period,
                Ieff=Ieff,
                Ioff=Ioff,
                L=L,
                W=W,
                V_dd=V_dd,
                V_th=V_th_eff,
                tox=tox,
                satisfies_constraints=constraints_satisfied
            )
            results.append(result)

            # Log only when we find a new best
            if constraints_satisfied and obj_value < best_obj_val:
                best_obj_val = obj_value
                logger.info(f"worker {worker_id} new best objective value: {obj_value}, design point: {design_point}")

    return (worker_id, results)


def _worker_evaluate_design_points_chunk(args_tuple):
    """
    Worker function for parallel brute force optimization.
    Each worker evaluates multiple design points and returns its local best.
    Must be defined at module level to be picklable.

    Args:
        args_tuple: (worker_id, design_points_chunk, tech_model, hw_obj, constraints)
            - worker_id: identifier for this worker
            - design_points_chunk: list of (index, design_point_dict) tuples to evaluate
            - tech_model: deep copy of the tech model for this worker
            - hw_obj: the objective function
            - constraints: list of cvxpy constraints

    Returns:
        (worker_id, best_design_point, best_obj_val, best_variables_dict, num_evaluated, num_failed)
    """
    worker_id, design_points_chunk, tech_model, prob = args_tuple

    best_design_point = None
    best_obj_val = math.inf
    best_variables_dict = None
    num_evaluated = 0
    num_failed = 0

    # Build cvxpy problem
    for idx, design_point in design_points_chunk:
        try:
            # Set parameters from design point
            tech_model.set_params_from_design_point(design_point)
            prob.solve(gp=True, **sim_util.GP_SOLVER_OPTS_RELAXED)

            num_evaluated += 1

            # Update local best if this solution is better
            if prob.value is not None and prob.value < best_obj_val:
                logger.info(f"new best obj val for worker {worker_id} after evaluation {num_evaluated}, {num_failed} failed: {prob.value}")
                logger.info(f"worker {worker_id} new best design point: {design_point}")

                best_design_point = design_point
                best_obj_val = prob.value
                best_variables_dict = {var.name(): var.value for var in prob.variables()}

        except Exception as e:
            num_failed += 1

    return (worker_id, best_design_point, best_obj_val, best_variables_dict, num_evaluated, num_failed)


class Optimizer:
    def __init__(self, hw, tmp_dir, save_dir, max_power, max_power_density, test_config=False, opt_pipeline="block_vector"):
        self.hw = hw
        self.disabled_knobs = []
        self.objective_constraint_inds = []
        self.initial_alpha = None
        self.test_config = test_config
        self.tmp_dir = tmp_dir
        self.save_dir = save_dir
        self.opt_pipeline = opt_pipeline
        self.bbv_op_delay_constraints = []
        self.bbv_path_constraints = []
        self.max_system_power = max_power
        self.max_system_power_density = max_power_density

    def evaluate_constraints(self, constraints, stage):
        for constraint_obj in constraints:
            constraint = constraint_obj.constraint
            value = 0
            if isinstance(constraint, sp.Ge):
                value = sim_util.xreplace_safe(constraint.rhs - constraint.lhs, self.hw.circuit_model.tech_model.base_params.tech_values)
            elif isinstance(constraint, sp.Le):
                value = sim_util.xreplace_safe(constraint.lhs - constraint.rhs, self.hw.circuit_model.tech_model.base_params.tech_values)
            log_info(f"constraint {constraint_obj.label} value: {value}", stage)
            tol = 1e-3
            if value > tol:
                log_info(f"CONSTRAINT VIOLATED {stage}", stage)

    def create_constraints(self, improvement, lower_bound, approx_problem=False):
        # system level and objective constraints, and pull in tech model constraints

        constraints = []
        constraints.append(Constraint(self.hw.obj_scaled >= lower_bound, "obj_scaled >= lower_bound"))
        self.objective_constraint_inds = [len(constraints)-1]

        # don't want a leakage-dominated design
        #constraints.append(self.hw.total_active_energy >= 2*self.hw.total_passive_energy*self.hw.circuit_model.tech_model.capped_power_scale)
        for knob in self.disabled_knobs:
            constraints.append(Constraint(sp.Eq(knob, knob.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)), "knob = tech_values[knob]"))
        if not self.test_config:
            total_power = self.hw.total_passive_power*self.hw.circuit_model.tech_model.capped_power_scale + self.hw.total_active_energy / (self.hw.execution_time* self.hw.circuit_model.tech_model.capped_delay_scale)
        else:
            total_power = self.hw.total_passive_power*self.hw.circuit_model.tech_model.capped_power_scale_total + self.hw.total_active_energy / (self.hw.execution_time* self.hw.circuit_model.tech_model.capped_delay_scale_total)
        assert self.max_system_power is not None, "max system power is not initialized"
        constraints.append(Constraint(total_power <= self.max_system_power, "total_power <= max_system_power")) # hard limit on power
        P_tot_device_per_cm2 = (self.hw.circuit_model.tech_model.E_act_inv / self.hw.circuit_model.tech_model.delay + self.hw.circuit_model.tech_model.P_pass_inv) / (self.hw.circuit_model.tech_model.param_db["A_gate"] * 1e4) # convert from W/m^2 to W/cm^2
        constraints.append(Constraint(P_tot_device_per_cm2 <= self.max_system_power_density, "P_tot_device_per_cm2 <= max_system_power_density"))
        # ensure that forward pass can't add more than 10x parallelism in the next iteration. power scale is based on the amount we scale area down by,
        # because in the next forward pass we assume that much parallelism will be added, and therefore increase power
        #if not self.test_config:
        #constraints.append(Constraint(self.hw.circuit_model.tech_model.capped_power_scale <= improvement, "capped_power_scale <= improvement"))

        assert len(self.hw.circuit_model.tech_model.constraints) > 0, "tech model constraints are empty"
        constraints.extend(self.hw.circuit_model.tech_model.base_params.constraints)
        constraints.extend(self.hw.circuit_model.tech_model.constraints)
        constraints.extend(self.bbv_op_delay_constraints)
        constraints.extend(self.bbv_path_constraints)

        if not approx_problem:
            constraints.extend(self.hw.circuit_model.constraints)

        self.evaluate_constraints(constraints, "before optimization")

        #print(f"constraints: {constraints}")
        return constraints
    
    def create_opt_model(self, improvement, lower_bound):
        constraints = self.create_constraints(improvement, lower_bound)
        model = pyo.ConcreteModel()
        self.preprocessor = Preprocessor(self.hw.circuit_model.tech_model.base_params, out_file=f"{self.tmp_dir}/solver_out.txt")
        opt, scaled_model, model, multistart_options = (
            self.preprocessor.begin(model, self.hw.obj_scaled, improvement, multistart=multistart, constraint_objs=constraints)
        )
        return opt, scaled_model, model, multistart_options
    
    # can be used as a starting point for the optimizer
    def generate_approximate_solution(self, improvement, iteration, execution_time,multistart=False):
        print(f"execution time: {execution_time.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)}")

        # passive energy consumption is dependent on execution time, so we need to recalculate it
        self.hw.calculate_passive_energy_vitis(execution_time)
        self.hw.save_obj_vals(execution_time)
        print(f"obj: {self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)}, obj scaled: {self.hw.obj_scaled.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)}")
        lower_bound = sim_util.xreplace_safe(self.hw.obj_scaled, self.hw.circuit_model.tech_model.base_params.tech_values) / improvement
        self.constraints = self.create_constraints(improvement, lower_bound, approx_problem=(self.opt_pipeline != "block_vector"))
        if self.opt_pipeline == "block_vector":
            self.hw.calculate_sensitivity_analysis(blackbox=True, constraints=self.constraints)
        model = pyo.ConcreteModel()
        self.approx_preprocessor = Preprocessor(self.hw.circuit_model.tech_model.base_params, out_file=f"{self.tmp_dir}/solver_out_approx_{iteration}.txt", solver_name="ipopt")
        opt, scaled_model, model, multistart_options = (
            self.approx_preprocessor.begin(model, self.hw.obj_scaled, improvement, multistart=multistart, constraint_objs=self.constraints)
        )

        if self.test_config:
            self.hw.execution_time = execution_time

        return opt, scaled_model, model, multistart_options

    # sets of delay, energy, and leakage power values are provided, fit a pareto front to them
    def fit_ed_pareto_front(self, delay_vals, energy_vals, p_leakage_vals):
        c, a = curve_fit.fit_ed_curve(delay_vals, energy_vals) # c, a for y=c*x^a
        
        return c, a

    def generate_design_points(self, count, improvement, execution_time):
        tech_param_sets = []
        obj_vals = []
        scaled_obj_vals = []
        original_tech_values = copy.deepcopy(self.hw.circuit_model.tech_model.base_params.tech_values)
        for i in range(count):
            stdout = sys.stdout
            Error = False
            with open(f"{self.tmp_dir}/ipopt_out_approx_{i}.txt", "w") as f:
                sys.stdout = f
                # INITIAL SOLVE
                opt_approx, scaled_model_approx, model_approx, multistart_options_approx = self.generate_approximate_solution(improvement, i, execution_time)
                try:
                    # run solver
                    results = opt_approx.solve(scaled_model_approx, symbolic_solver_labels=True)
                except Exception as e:
                    print(f"Error: {e}")
                    Error = True
                # just let "infeasible" solutions through for now, often they are not violating any constraints
                if (Error or results.solver.termination_condition not in ["optimal", "acceptable", "infeasible", "maxIterations"]):
                    print(f"First solve attempt failed, trying again with multistart solver...")
                    #raise Exception("First solve attempt failed")
                    Error = False    
                    opt_approx, scaled_model_approx, model_approx, multistart_options_approx = self.generate_approximate_solution(improvement, i, execution_time, multistart=True)
                    # Try with more relaxed tolerances
                    #opt_approx.options["constr_viol_tol"] = 1e-4
                    #opt_approx.options["acceptable_constr_viol_tol"] = 1e-2
                    #opt_approx.options["acceptable_tol"] = 1e-4
                    try:
                        results = opt_approx.solve(scaled_model_approx, **multistart_options_approx)
                    except Exception as e:
                        print(f"Error: {e}")
                        Error = True

                # IF SOLVER FOUND AN OK SOLUTION, DISPLAY RESULT
                if results.solver.termination_condition in ["optimal", "acceptable", "infeasible", "maxIterations"]: 
                    print(f"approximate solver found {results.solver.termination_condition} solution in iteration {i}")
                    pyo.TransformationFactory("core.scale_model").propagate_solution(
                        scaled_model_approx, model_approx
                    )
                    model_approx.display()
                else:
                    print(f"approximate solver failed in iteration {i} with termination condition: {results.solver.termination_condition}")
                    Error = True
            sys.stdout = stdout
            if not Error:
                # PARSE SOLVER OUTPUT IF NO ERROR
                f = open(f"{self.tmp_dir}/ipopt_out_approx_{i}.txt", "r")
                sim_util.parse_output(f, self.hw)
                print(f"scaled objective used in approximation is now: {self.hw.obj_scaled.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)}")
                print(f"objective used in approximation is now: {self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)}")
                self.hw.display_objective("after approximate solver, before recalculating objective")
                if not self.test_config:
                    self.hw.circuit_model.update_circuit_values()
                    print(f"value of clk period: {self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.clk_period]}")
                    self.hw.calculate_objective(form_dfg=False)

                # store result of this design point
                scaled_obj_vals.append(self.hw.obj_scaled.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values))
                obj_vals.append(self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values))
                self.hw.display_objective("after approximate solver")
                tech_param_sets.append(self.hw.circuit_model.tech_model.base_params.tech_values)

                # resetting original tech parameters, will decide later which one out of tech_param_sets to use, or just keep the original
                self.hw.circuit_model.tech_model.base_params.tech_values = copy.deepcopy(original_tech_values)
        return tech_param_sets, obj_vals, scaled_obj_vals

    # in each step, we optimize the delay of the current critical path. So we use a representation of the execution time which only includes the critical path.
    def calculate_current_execution_time(self):
        cur_delay = sim_util.xreplace_safe(self.hw.execution_time, self.hw.circuit_model.tech_model.base_params.tech_values)
        print(f"cur delay calculated for scale factor: {cur_delay}")
        
        # get base delays for each op type
        logic_base_delay = self.hw.circuit_model.tech_model.delay
        logic_rsc_base_delay = self.hw.circuit_model.tech_model.base_params.clk_period
        interconnect_base_delay = self.hw.circuit_model.tech_model.m1_Rsq * self.hw.circuit_model.tech_model.m1_Csq
        interconnect_rsc_base_delay = self.hw.circuit_model.tech_model.base_params.clk_period
        # TODO: add memory

        # set delay ratios for each op type, they should all start out as 1
        logic_delay_ratio = logic_base_delay / sim_util.xreplace_safe(logic_base_delay, self.hw.circuit_model.tech_model.base_params.tech_values)
        logic_rsc_delay_ratio = logic_rsc_base_delay / sim_util.xreplace_safe(logic_rsc_base_delay, self.hw.circuit_model.tech_model.base_params.tech_values)
        interconnect_delay_ratio = interconnect_base_delay / sim_util.xreplace_safe(interconnect_base_delay, self.hw.circuit_model.tech_model.base_params.tech_values)
        interconnect_rsc_delay_ratio = interconnect_rsc_base_delay / sim_util.xreplace_safe(interconnect_rsc_base_delay, self.hw.circuit_model.tech_model.base_params.tech_values)
        # TODO: add memory

        # multiply ratios by sensitivities and add them up. TODO: add memory
        delay_ratio_from_original = (logic_delay_ratio * self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.logic_sensitivity] + 
                                    logic_rsc_delay_ratio * self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.logic_resource_sensitivity] + 
                                    interconnect_delay_ratio * self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.interconnect_sensitivity] + 
                                    interconnect_rsc_delay_ratio * self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.interconnect_resource_sensitivity])

        epsilon = 1e-6
        assert sim_util.xreplace_safe(delay_ratio_from_original, self.hw.circuit_model.tech_model.base_params.tech_values) - 1 <= epsilon, "delay ratio from original should start out as 1"

        # scale it up to the actual value of delay
        current_execution_time = cur_delay * delay_ratio_from_original
        #print(f"current execution time calculated for scale factor: {current_execution_time}")
        return current_execution_time

    def run_ipopt_optimization(self, improvement, lower_bound, execution_time):
        tech_param_sets, obj_vals, scaled_obj_vals = self.generate_design_points(1, improvement, execution_time)
        if not tech_param_sets or not obj_vals:
            raise RuntimeError("No successful design points found. All IPOPT solver runs failed. Check solver parameters and constraints.")
        optimal_design_idx = scaled_obj_vals.index(min(scaled_obj_vals))
        print(f"optimal design idx: {optimal_design_idx}")
        print(f"obj vals: {obj_vals}")
        print(f"scaled obj vals: {scaled_obj_vals}")
        assert scaled_obj_vals[optimal_design_idx] < lower_bound * improvement, "no better design point found"
        self.hw.circuit_model.tech_model.base_params.tech_values = tech_param_sets[optimal_design_idx].copy()
        if not self.test_config:
            self.hw.calculate_objective(form_dfg=False)
        return obj_vals[optimal_design_idx]

    def get_one_bbv_op_delay_constraint(self, op_type, op_delay, improvement):
        amdahl_limit = sim_util.xreplace_safe(getattr(self.hw.circuit_model.tech_model.base_params, op_type + "_amdahl_limit"), self.hw.circuit_model.tech_model.base_params.tech_values)
        if amdahl_limit == math.inf:
            print(f"amdahl limit is infinite for {op_type}, skipping constraint")
            return []
        if amdahl_limit < improvement: # skip because we will eventually need to optimize a path with this op type anyways
            print(f"amdahl limit is less than improvement for {op_type}, skipping constraint")
            return []
        op_delay_ratio = op_delay / sim_util.xreplace_safe(op_delay, self.hw.circuit_model.tech_model.base_params.tech_values)
        delay_contrib = ((1/amdahl_limit) * op_delay) * op_delay_ratio
        self.hw.save_obj_vals(self.hw.execution_time, execution_time_override=True, execution_time_override_val=delay_contrib)
        obj_scaled_op = self.hw.obj_scaled
        obj_scaled_op_init = sim_util.xreplace_safe(obj_scaled_op, self.hw.circuit_model.tech_model.base_params.tech_values)
        # this constraint should always start out as feasible because amdahl limit >= improvement
        constr = obj_scaled_op <= obj_scaled_op_init * (amdahl_limit/improvement)

        # reset hw model state
        self.hw.save_obj_vals(self.hw.execution_time)
        intial_obj_scaled = sim_util.xreplace_safe(self.hw.obj_scaled, self.hw.circuit_model.tech_model.base_params.tech_values)
        assert intial_obj_scaled >= obj_scaled_op_init, "op specific obj scaled should be less than or equal to the original obj scaled"

        return [Constraint(constr, f"bbv_op_delay_{op_type}")]

    def set_bbv_op_delay_constraints(self, improvement):
        self.bbv_op_delay_constraints = []
        op_delays = {
            "logic": self.hw.circuit_model.tech_model.delay,
            "memory": 1,
            "interconnect": self.hw.circuit_model.tech_model.m1_Rsq * self.hw.circuit_model.tech_model.m1_Csq,
            "logic_resource": self.hw.circuit_model.tech_model.base_params.clk_period,
            "memory_resource": self.hw.circuit_model.tech_model.base_params.clk_period,
            "interconnect_resource": self.hw.circuit_model.tech_model.m1_Rsq * self.hw.circuit_model.tech_model.m1_Csq,
        }
        for op_type in BlockVector.op_types:
            print(f"setting bbv op delay constraint for {op_type}")
            constr = self.get_one_bbv_op_delay_constraint(op_type, op_delays[op_type], improvement)
            self.bbv_op_delay_constraints.extend(constr)

    def block_vector_based_optimization(self, improvement, lower_bound):
        improvement_remaining = improvement
        iteration = 0
        best_tech_values = copy.deepcopy(self.hw.circuit_model.tech_model.base_params.tech_values)
        best_obj_scaled = self.hw.obj_scaled.xreplace(best_tech_values)
        self.bbv_path_constraints = []
        self.set_bbv_op_delay_constraints(improvement)

        while improvement_remaining > 1.5 and iteration < 10:
            # symbolic execution time of critical path only
            execution_time = self.calculate_current_execution_time()
            tech_param_sets, obj_vals, scaled_obj_vals = self.generate_design_points(1, improvement_remaining, execution_time)
            if not tech_param_sets or not obj_vals:
                raise RuntimeError(f"No successful design points found in iteration {iteration}.")
            optimal_design_idx = scaled_obj_vals.index(min(scaled_obj_vals))
            print(f"optimal design idx: {optimal_design_idx}")
            print(f"obj vals: {obj_vals}")
            print(f"scaled obj vals: {scaled_obj_vals}")
            assert scaled_obj_vals[optimal_design_idx] < lower_bound * improvement, "no better design point found"
            self.hw.circuit_model.tech_model.base_params.tech_values = tech_param_sets[optimal_design_idx].copy()
            self.hw.calculate_objective(form_dfg=False)
            true_scaled_obj_val = sim_util.xreplace_safe(self.hw.obj_scaled, self.hw.circuit_model.tech_model.base_params.tech_values)
            print(f"actual scaled obj val after recalculating block vectors: {true_scaled_obj_val}")
            assert true_scaled_obj_val >= scaled_obj_vals[optimal_design_idx], "actual scaled obj val should be greater than or equal to the scaled obj val from the solver (due to near-critical paths)"

            improvement_remaining = true_scaled_obj_val / lower_bound
            print(f"improvement remaining: {improvement_remaining}")
            iteration += 1
            if true_scaled_obj_val < best_obj_scaled:
                best_tech_values = copy.deepcopy(tech_param_sets[optimal_design_idx])
                best_obj_scaled = true_scaled_obj_val
            # ensure that this path does not become critical again
            self.bbv_path_constraints.append(Constraint(execution_time <= sim_util.xreplace_safe(execution_time, self.hw.circuit_model.tech_model.base_params.tech_values), "bbv_path_constraint"))
        
        assert best_obj_scaled < lower_bound * improvement, "no better design point found"
        self.hw.circuit_model.tech_model.base_params.tech_values = best_tech_values
        self.hw.calculate_objective(form_dfg=False)
        return sim_util.xreplace_safe(self.hw.obj, self.hw.circuit_model.tech_model.base_params.tech_values)
            
    def logic_device_optimization(self, improvement, lower_bound):
        execution_time = self.hw.circuit_model.tech_model.delay * (sim_util.xreplace_safe(self.hw.execution_time, self.hw.circuit_model.tech_model.base_params.tech_values)/sim_util.xreplace_safe(self.hw.circuit_model.tech_model.delay, self.hw.circuit_model.tech_model.base_params.tech_values))
        obj_val = self.run_ipopt_optimization(improvement, lower_bound, execution_time)
        return obj_val

    def ipopt(self, improvement):
        """
        Run the IPOPT optimization routine for the hardware model using Pyomo.

        Args:
            tech_params (dict): Technology parameters for optimization.
            edp (sympy.Expr): Symbolic EDP Expression.
            improvement (float): Improvement factor for optimization.
            cacti_subs (dict): Substitution dictionary for CACTI parameters.

        Returns:
            None
        """
        logger.info("Optimizing using IPOPT")

        #param_replace = {param: sp.Abs(param, evaluate=False) for param in self.hw.circuit_model.tech_model.base_params.tech_values}
        #print("param_replace: ", param_replace)
        #print("symbolic obj before abs: ", self.hw.obj)
        #self.hw.obj = self.hw.obj.xreplace(param_replace)
        #print("symbolic obj after abs: ", self.hw.obj)

        lower_bound = sim_util.xreplace_safe(self.hw.obj, self.hw.circuit_model.tech_model.base_params.tech_values) / improvement

        start_time = time.time()
        if self.opt_pipeline == "logic_device":
            obj_val = self.logic_device_optimization(improvement, lower_bound)
        elif self.opt_pipeline == "block_vector":
            obj_val = self.block_vector_based_optimization(improvement, lower_bound)
        else:
            raise ValueError(f"Invalid optimization pipeline: {self.opt_pipeline}")

        logger.info(f"time to run optimization: {time.time()-start_time}")

        lag_factor = obj_val / lower_bound
        print(f"lag factor: {lag_factor}")
        return lag_factor, False

    def fit_optimization(self, improvement):
        start_time = time.time()
        lower_bound = sim_util.xreplace_safe(self.hw.obj, self.hw.circuit_model.tech_model.base_params.tech_values) / improvement
        self.constraints = self.hw.constraints + self.hw.circuit_model.constraints + self.hw.circuit_model.tech_model.constraints_cvxpy
        #clk_period_constr = [self.hw.circuit_model.tech_model.base_params.clk_period == self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.clk_period]]
        constraints = [constraint.constraint for constraint in self.constraints] 
        for constraint in constraints:
            assert constraint.is_dgp(), f"constraint is not DGP: {constraint}"
            print(f"constraint: {constraint}")
        print(f"objective: {self.hw.obj}")

        prob = cp.Problem(cp.Minimize(self.hw.obj), constraints)
        #obj_val = solve_gp_with_fallback(prob)
        prob.solve(gp=True, **sim_util.GP_SOLVER_OPTS_RELAXED)
        for var in prob.variables():
            print(f"variable: {var.name}, value: {var.value}")
            self.hw.circuit_model.tech_model.base_params.set_symbol_value(var, var.value)
        print(f"cvxpy optimization status: {prob.status}")
        print(f"cvxpy optimization value: {prob.value}")
        print(f"cvxpy optimization constraints: {prob.constraints}")
        logger.info(f"time to run cvxpy optimization: {time.time()-start_time}")
        self.hw.circuit_model.tech_model.process_optimization_results()
        for constraint in self.constraints:
            constraint.set_slack_cvxpy()

        return prob.value / lower_bound, False

    def get_system_constraints_brute_force(self):
        system_constraints = []
        total_power = self.hw.total_passive_power + self.hw.total_active_energy / self.hw.execution_time
        system_constraints.append(Constraint(total_power <= self.max_system_power, "total_power <= max_system_power")) # hard limit on power
        return system_constraints
    
    def brute_force_optimization(self, improvement, n_processes=1):
        """
        Brute force optimization over design points in pareto_df.

        Args:
            improvement: Improvement factor for optimization.
            n_processes: Number of parallel processes to use (default: 1 for sequential).

        Returns:
            (ratio of best_obj_val to lower_bound, False)
        """
        start_time = time.time()
        cur_obj_val = sim_util.xreplace_safe(self.hw.obj, self.hw.circuit_model.tech_model.base_params.tech_values)
        lower_bound = cur_obj_val / improvement
        self.constraints = self.hw.constraints + self.hw.circuit_model.constraints + self.hw.circuit_model.tech_model.constraints_cvxpy
        system_constraints = self.get_system_constraints_brute_force()
        constraints = [constraint.constraint for constraint in self.constraints] + [constraint.constraint for constraint in system_constraints]
        for constraint in constraints:
            assert constraint.is_dgp(), f"constraint is not DGP: {constraint}"
            print(f"constraint: {constraint}")
        print(f"objective: {self.hw.obj}")

        total_design_points = len(self.hw.circuit_model.tech_model.pareto_df)
        logger.info(f"Starting brute force optimization with {total_design_points} design points using {n_processes} process(es)")
        logger.info(f"objective: {self.hw.obj}")
        prob = cp.Problem(cp.Minimize(self.hw.obj), constraints)
        # Parallel execution
        best_design_point, best_obj_val, best_design_variables = self._brute_force_parallel(
            prob, n_processes
        )

        if best_design_point is None or best_obj_val >= cur_obj_val:
            logger.warning("No better solution found in this iteration")
            return cur_obj_val, False

        # Re-solve with the best design point to get the actual cvxpy variable objects
        # This ensures we have proper variable references for set_symbol_value
        logger.info(f"Re-solving with best design point: {best_design_point}")
        self.hw.circuit_model.tech_model.set_params_from_design_point(best_design_point)
        prob.solve(gp=True, **sim_util.GP_SOLVER_OPTS_RELAXED)

        # Apply the solution to the tech model using the actual variable objects
        for var in prob.variables():
            print(f"variable: {var.name()}, value: {var.value}")
            self.hw.circuit_model.tech_model.base_params.set_symbol_value(var, var.value)

        logger.info(f"time to run cvxpy optimization: {time.time()-start_time}")
        self.hw.circuit_model.tech_model.process_optimization_results()

        return best_obj_val / lower_bound, False

    def _brute_force_parallel(self, prob, n_processes):
        """
        Parallel brute force optimization using ProcessPoolExecutor.
        Each worker evaluates a chunk of design points and maintains its own local best.
        After all workers complete, we compare their local bests to find the global best.

        Args:
            prob: cvxpy problem.
            n_processes: Number of parallel processes.

        Returns:
            (best_design_point, best_obj_val, best_design_variables_dict)
        """
        tech_model = self.hw.circuit_model.tech_model

        # Prepare design points with indices and randomize order
        design_points = [
            (i, row._asdict()) for i, row in enumerate(tech_model.pareto_df.itertuples(index=False))
        ]
        np.random.shuffle(design_points)

        # Partition design points into chunks for each worker
        chunk_size = math.ceil(total_design_points / n_processes)
        chunks = [
            design_points[i * chunk_size : (i + 1) * chunk_size]
            for i in range(n_processes)
        ]
        # Filter out empty chunks (in case n_processes > total_design_points)
        chunks = [chunk for chunk in chunks if chunk]
        actual_n_workers = len(chunks)

        logger.info(f"Partitioning {total_design_points} design points into {actual_n_workers} chunks of ~{chunk_size} each")

        # Create tasks: each worker gets a chunk of design points
        # No need to copy tech_model since ProcessPoolExecutor uses separate processes
        tasks = [
            (worker_id, chunk, tech_model, prob)
            for worker_id, chunk in enumerate(chunks)
        ]

        # Submit all tasks and wait for results
        logger.info(f"Submitting {actual_n_workers} worker tasks...")
        with ProcessPoolExecutor(max_workers=actual_n_workers) as executor:
            futures = [executor.submit(_worker_evaluate_design_points_chunk, task) for task in tasks]

            # Collect results from all workers
            worker_results = []
            for future in as_completed(futures):
                result = future.result()
                worker_results.append(result)
                worker_id, best_dp, best_val, best_vars, num_eval, num_fail = result
                logger.info(f"Worker {worker_id} completed: evaluated {num_eval}, failed {num_fail}, best value = {best_val}")

        # Find global best by comparing all workers' local bests
        best_design_point = None
        best_obj_val = math.inf
        best_design_variables = None

        for worker_id, local_best_dp, local_best_val, local_best_vars, num_eval, num_fail in worker_results:
            if local_best_dp is not None and local_best_val is not None and local_best_val < best_obj_val:
                best_design_point = local_best_dp
                best_obj_val = local_best_val
                best_design_variables = local_best_vars
                logger.info(f"New global best from worker {worker_id}: {best_obj_val}")

        total_evaluated = sum(r[4] for r in worker_results)
        total_failed = sum(r[5] for r in worker_results)
        logger.info(f"Parallel optimization complete. Total evaluated: {total_evaluated}, failed: {total_failed}")
        logger.info(f"Global best objective value: {best_obj_val}")

        return best_design_point, best_obj_val, best_design_variables
    

    def cvxpy_optimization(self, improvement, n_processes=100):
        if self.opt_pipeline == "fit":
            return self.fit_optimization(improvement)
        elif self.opt_pipeline == "brute_force":
            return self.brute_force_optimization(improvement, n_processes=n_processes)
        else:
            raise ValueError(f"Invalid optimization pipeline: {self.opt_pipeline}")

    def basic_optimization(self, improvement, iteration, n_processes=100):
        self.iteration = iteration
        self.constraints = []
        start_time = time.time()
        cur_obj_val = sim_util.xreplace_safe(self.hw.obj, self.hw.circuit_model.tech_model.base_params.tech_values)

        total_design_points = len(self.hw.circuit_model.tech_model.pareto_df)
        logger.info(f"Starting brute force optimization with {total_design_points} design points using {n_processes} process(es)")
        logger.info(f"objective: {self.hw.obj}")
        # Parallel execution

        # Prepare design points with indices and randomize order
        design_points = [
            (i, row._asdict()) for i, row in enumerate(self.hw.circuit_model.tech_model.pareto_df.itertuples(index=False))
        ]
        np.random.shuffle(design_points)

        # Partition design points into chunks for each worker
        chunk_size = math.ceil(total_design_points / n_processes)
        chunks = [
            design_points[i * chunk_size : (i + 1) * chunk_size]
            for i in range(n_processes)
        ]
        # Filter out empty chunks (in case n_processes > total_design_points)
        chunks = [chunk for chunk in chunks if chunk]
        actual_n_workers = len(chunks)

        logger.info(f"Partitioning {total_design_points} design points into {actual_n_workers} chunks of ~{chunk_size} each")

        # Create ObjectiveEvaluator - this is pickleable and avoids cvxpy objects
        logger.info(f"Creating ObjectiveEvaluator from HardwareModel...")
        evaluator = ObjectiveEvaluator.from_hardware_model(self.hw)
        logger.info(f"starting total_active_energy: {sim_util.xreplace_safe(self.hw.total_active_energy, self.hw.circuit_model.tech_model.base_params.tech_values)}")
        logger.info(f"starting total_passive_power: {sim_util.xreplace_safe(self.hw.total_passive_energy/self.hw.execution_time, self.hw.circuit_model.tech_model.base_params.tech_values)}")

        # Create tasks: each worker gets a chunk of design points and the evaluator
        tasks = [
            (worker_id, chunk, evaluator, self.max_system_power)
            for worker_id, chunk in enumerate(chunks)
        ]

        # Submit all tasks and wait for results
        logger.info(f"Submitting {actual_n_workers} worker tasks...")
        all_results: List[DesignPointResult] = []

        with ProcessPoolExecutor(max_workers=actual_n_workers) as executor:
            futures = [executor.submit(_worker_basic_optimization_chunk, task) for task in tasks]

            # Collect results from all workers
            for future in as_completed(futures):
                worker_id, worker_results = future.result()
                all_results.extend(worker_results)

                # Find this worker's best for logging
                valid_results = [r for r in worker_results if r.satisfies_constraints]
                if valid_results:
                    worker_best = min(valid_results, key=lambda r: r.obj_value)
                    logger.info(f"Worker {worker_id} completed: best value = {worker_best.obj_value}")
                else:
                    logger.info(f"Worker {worker_id} completed: no valid designs")

        # Sort all results by objective value
        valid_results = [r for r in all_results if r.satisfies_constraints]
        sorted_results = sorted(valid_results, key=lambda r: r.obj_value)

        logger.info(f"Total designs evaluated: {len(all_results)}, valid designs: {len(valid_results)}")

        # Find global best
        if sorted_results:
            best_result = sorted_results[0]
            best_design_point = best_result.design_point
            best_obj_val = best_result.obj_value
            best_value_clk_period = best_result.clk_period
            logger.info(f"Global best objective value: {best_obj_val}, design point: {best_design_point}")

            # Visualize top 10% of designs
            visualize_top_designs(all_results, self.iteration, top_percent=1, output_dir=self.save_dir)
        else:
            best_design_point = None
            best_obj_val = math.inf
            best_value_clk_period = None
            logger.warning("No valid designs found")

        if best_design_point is None or best_obj_val >= cur_obj_val:
            logger.warning("No better solution found in this iteration")
            return cur_obj_val, False

        self.hw.circuit_model.tech_model.set_params_from_design_point(best_design_point)
        self.hw.circuit_model.tech_model.base_params.set_symbol_value(self.hw.circuit_model.tech_model.base_params.clk_period, best_value_clk_period)
        self.hw.calculate_objective()
        logger.info(f"ending total_active_energy: {sim_util.xreplace_safe(self.hw.total_active_energy, self.hw.circuit_model.tech_model.base_params.tech_values)}")
        logger.info(f"ending total_passive_power: {sim_util.xreplace_safe(self.hw.total_passive_energy/self.hw.execution_time, self.hw.circuit_model.tech_model.base_params.tech_values)}")
        return 1, False
        

    # note: improvement/regularization parameter currently only for inverse pass validation, so only using it for ipopt
    # example: improvement of 1.1 = 10% improvement
    def optimize(self, opt, improvement=10, disabled_knobs=[], iteration=0):
        self.disabled_knobs = disabled_knobs
        """
        Optimize the hardware model using the specified optimization method.

        Args:

        Returns:
            None
        """
        if opt == "ipopt":
            return self.ipopt(improvement)
        elif opt == "cvxpy":
            return self.cvxpy_optimization(improvement)
        elif opt == "basic":
            return self.basic_optimization(improvement, iteration)
        else:
            raise ValueError(f"Invalid solver: {opt}")


def main():
    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="logs/optimize.log")
    parser = argparse.ArgumentParser(
        prog="Optimize",
        description="Optimization part of the Inverse Pass. This runs after an analytic equation for the cost is created.",
        epilog="Text at the bottom of help",
    )

    parser.add_argument(
        "-c",
        "--architecture_config",
        default="aladdin_const_with_mem",
        type=str,
        help="Path to the architecture config file",
    )
    parser.add_argument(
        "-o",
        "--opt",
        type=str,
        default="ipopt",
    )

    args = parser.parse_args()
    main()
