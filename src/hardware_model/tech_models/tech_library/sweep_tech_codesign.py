from src import codesign
import argparse
import csv
import itertools
import logging
import os
from datetime import datetime
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import math
from src import sim_util

logger = logging.getLogger(__name__)

debug = True

def log_info(msg):
    if debug:
        logger.info(msg)

# Helper function to format numeric values in scientific notation with 4 decimal places
def format_scientific(value):
    """Format numeric values in scientific notation with 4 decimal places."""
    if isinstance(value, (int, float)):
        return f"{float(value):.2e}"
    return value

# Global worker function for ProcessPoolExecutor (must be picklable)
def _worker_evaluate_configuration(args_tuple):
    """
    Worker function for process pool execution.
    Must be defined at module level to be picklable.
    """
    idx, param_values, params_to_sweep, tech_model = args_tuple

    try:
        # Create a dictionary of current parameter values
        current_config = dict(zip(params_to_sweep, param_values))

        # Set parameter values in tech_model.base_params.tech_values
        for param_name, param_value in current_config.items():
            if hasattr(tech_model.base_params, param_name):
                param_symbol = getattr(tech_model.base_params, param_name)
                tech_model.base_params.tech_values[param_symbol] = param_value

        # Re-initialize transistor equations with new parameter values
        tech_model.init_transistor_equations()

        # Collect results: input parameters + output metrics from param_db
        result_row = {}

        # Add input parameters
        for param_name, param_value in current_config.items():
            result_row[param_name] = format_scientific(param_value)

        # Add output metrics from param_db
        for metric_name, metric_value in tech_model.sweep_output_db.items():
            # Handle both numeric and symbolic values
            if hasattr(metric_value, 'evalf'):
                # Symbolic expression - evaluate it
                evaluated_value = sim_util.xreplace_safe(metric_value, tech_model.base_params.tech_values)
                result_row[metric_name] = format_scientific(evaluated_value)
            else:
                # Already a numeric value
                result_row[metric_name] = format_scientific(metric_value)

        # Validate all constraints
        for constraint in tech_model.constraints:
            tol = 1e-25
            slack = sim_util.xreplace_safe(constraint.slack, tech_model.base_params.tech_values)
            if (slack > tol):
                log_info(f"CONSTRAINT VIOLATED {constraint.label} for config {idx}, slack is {slack}")
                # Constraint violated
                return (idx, None)

        return (idx, result_row)

    except Exception as e:
        return (idx, None)

class SweepTechCodesign:
    def __init__(self, args):
        self.args = args
        self.codesign_module = codesign.Codesign(
            self.args
        )
        # Configure logger to use the same log file as codesign
        # Get the log file path from codesign's save_dir
        log_file = f"{self.codesign_module.save_dir}/codesign.log"
        # Add a file handler to the logger if it doesn't already have one
        # Check both by type and filename to avoid duplicates
        existing_log_files = [h.baseFilename for h in logger.handlers if isinstance(h, logging.FileHandler)]
        if log_file not in existing_log_files:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)
        # Disable propagation to avoid duplicate messages from parent loggers
        logger.propagate = False

    def sweep_tech(self, params_to_sweep, value_ranges, output_dir="sweep_results", flush_interval=10, n_processes=1):
        """
        Sweep through different tech model configurations in parallel using ProcessPoolExecutor.

        Args:
            params_to_sweep: List of parameter names to sweep (e.g., ['L', 'V_dd', 'tox'])
            value_ranges: Dictionary mapping parameter names to lists of values to sweep
                         e.g., {'L': [7e-9, 5e-9, 3e-9], 'V_dd': [0.7, 0.8, 0.9]}
            output_dir: Directory to save the CSV output file
            flush_interval: Number of iterations between CSV flushes (default: 10)
            n_processes: Number of processes to use for parallel evaluation (default: 1)

        Returns:
            Path to the generated CSV file
        """
        # Validate inputs
        if not params_to_sweep:
            logger.error("No parameters specified for sweeping")
            return None

        for param in params_to_sweep:
            if param not in value_ranges:
                logger.error(f"Parameter '{param}' not found in value_ranges")
                return None
            if not value_ranges[param]:
                logger.error(f"No values provided for parameter '{param}'")
                return None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(output_dir, f"tech_sweep_{self.codesign_module.cfg['args']['model_cfg']}_{timestamp}.csv")

        # Get the tech model reference
        tech_model = self.codesign_module.hw.circuit_model.tech_model

        # Generate all combinations of parameter values
        param_value_lists = [value_ranges[param] for param in params_to_sweep]
        all_combinations = list(itertools.product(*param_value_lists))

        total_runs = len(all_combinations)
        logger.info(f"Starting tech sweep with {total_runs} configurations")
        logger.info(f"Sweeping parameters: {params_to_sweep}")
        logger.info(f"Using {n_processes} process(es) for parallel execution")
        logger.info(f"Flushing results to CSV every {flush_interval} iterations")

        # Create a pool of tech_model copies (one per worker)
        logger.info(f"Creating {n_processes} copies of tech_model for parallel execution...")
        tech_model_pool = [copy.deepcopy(tech_model) for _ in range(n_processes)]
        logger.info(f"Tech model copies created successfully")

        # Prepare CSV file
        csv_initialized = False
        fieldnames = None
        csvfile = None
        writer = None
        results_buffer = []
        successful_count = 0
        max_num_jobs_at_once = 10000
        num_iterations = math.ceil(len(all_combinations) / max_num_jobs_at_once)
        total_completed_count = 0

        try:
            for batch_num in range(num_iterations):
                batch_start_idx = batch_num * max_num_jobs_at_once
                batch_end_idx = min((batch_num + 1) * max_num_jobs_at_once, len(all_combinations))
                current_combinations = all_combinations[batch_start_idx:batch_end_idx]

                logger.info(f"Starting batch {batch_num + 1}/{num_iterations} ({len(current_combinations)} configurations)")

                with ProcessPoolExecutor(max_workers=n_processes) as executor:
                    tasks = [
                        (global_idx, param_values, params_to_sweep, tech_model_pool[(global_idx - 1) % n_processes])
                        for local_idx, param_values in enumerate(current_combinations, 1)
                        for global_idx in [batch_start_idx + local_idx]
                    ]
                    future_to_idx = {
                        executor.submit(_worker_evaluate_configuration, task): task[0]
                        for task in tasks
                    }

                    # Process results as they complete
                    for future in as_completed(future_to_idx):
                        idx, result_row = future.result()
                        total_completed_count += 1
                        if (total_completed_count % 100 == 0):
                            logger.info(f"Progress: {total_completed_count}/{total_runs} configurations completed")

                        if result_row is not None:
                            results_buffer.append(result_row)
                            successful_count += 1

                            # Initialize CSV file and write header on first successful result
                            if not csv_initialized and results_buffer:
                                all_columns = set()
                                for result in results_buffer:
                                    all_columns.update(result.keys())

                                output_metrics = sorted([col for col in all_columns if col in tech_model.sweep_output_db.keys()])
                                input_params = sorted([col for col in all_columns if col in params_to_sweep])
                                other_cols = sorted([col for col in all_columns if col not in output_metrics and col not in input_params])
                                fieldnames = output_metrics + input_params + other_cols

                                csvfile = open(csv_filename, 'w', newline='')
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writeheader()
                                csv_initialized = True
                                logger.info(f"Initialized CSV file: {csv_filename}")
                        else:
                            logger.warning(f"Configuration {idx} failed")

                        # Flush to CSV every flush_interval successful results or on last iteration
                        if csv_initialized and len(results_buffer) > 0:
                            if len(results_buffer) >= flush_interval or total_completed_count == total_runs:
                                writer.writerows(results_buffer)
                                csvfile.flush()
                                logger.info(f"Flushed {len(results_buffer)} results to CSV (total written: {successful_count}/{total_runs})")
                                results_buffer = []

                # Flush any remaining results from this batch
                if csv_initialized and results_buffer:
                    writer.writerows(results_buffer)
                    csvfile.flush()
                    logger.info(f"End of batch {batch_num + 1}: Flushed {len(results_buffer)} remaining results")
                    results_buffer = []

                logger.info(f"Batch {batch_num + 1}/{num_iterations} complete")

            # All batches complete - close CSV and return
            if csv_initialized:
                csvfile.close()
                logger.info(f"Sweep completed. Results saved to: {csv_filename}")
                logger.info(f"Total configurations: {total_runs}, Successful: {successful_count}")
                return csv_filename
            else:
                logger.error("No results to write")
                return None

        except Exception as e:
            logger.error(f"Error during sweep: {e}")
            if csvfile:
                csvfile.close()
            raise

    def prune_design_space(self, csv_filename, objectives=None, minimize_all=True):
        """
        Create a Pareto front based on the results in the CSV file using paretoset library.

        Args:
            csv_filename: Path to the CSV file containing sweep results
            objectives: List of column names to use for Pareto optimization.
                       If None, uses all output metrics from sweep_output_db
            minimize_all: If True, minimizes all objectives. If False or dict provided,
                         can specify per-objective. Dict format: {'metric': True/False}
                         where True=minimize, False=maximize

        Returns:
            pandas DataFrame of Pareto-optimal designs
        """
        import pandas as pd
        from paretoset import paretoset

        # Read the CSV file
        df = pd.read_csv(csv_filename)

        # Determine which columns to use as objectives
        if objectives is None:
            # Get output metrics from the tech model
            objectives = sorted([col for col in df.columns
                               if col in self.codesign_module.hw.circuit_model.tech_model.pareto_metric_db])

        logger.info(f"Computing Pareto front for objectives: {objectives}")

        # Handle minimize/maximize for each objective
        if isinstance(minimize_all, bool):
            # If boolean, apply to all objectives
            minimize_dict = {obj: minimize_all for obj in objectives}
        else:
            # If dict provided, use it
            minimize_dict = minimize_all

        # Build sense list for paretoset ("min" or "max" for each objective)
        sense = ["min" if minimize_dict.get(obj, True) else "max" for obj in objectives]

        # Compute Pareto front using paretoset
        pareto_mask = paretoset(df[objectives], sense=sense)

        # Get Pareto-optimal designs
        pareto_df = df[pareto_mask]

        logger.info(f"Found {len(pareto_df)} Pareto-optimal designs out of {len(df)} total designs")

        # Create output filename for Pareto front
        output_dir = os.path.join(os.path.dirname(csv_filename), "pareto_fronts")
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(csv_filename).rsplit('.', 1)[0]
        pareto_filename = os.path.join(output_dir, f"{base_name}_pareto.csv")

        # Save Pareto-optimal designs
        pareto_df.to_csv(pareto_filename, index=False)

        logger.info(f"Saved Pareto front to: {pareto_filename}")

        # Fit parametric surface model for use in optimization
        try:
            logger.info("Fitting parametric Pareto surface model...")

            fit_results = self.just_fit_pareto_surface(pareto_df, objectives)

        except Exception as e:
            logger.error(f"Error fitting Pareto surface: {e}")
            logger.info("Continuing without surface fit...")

        return pareto_filename

    def _generate_cvxpy_constraint_example(self, metric, model, param_names):
        """Generate CVXPY constraint code example for a given metric model."""
        # Check if this is a posynomial model (new format) or monomial model (old format)
        if 'type' in model and model['type'] == 'posynomial' and 'terms' in model:
            # Posynomial format: sum of terms
            # Build sum of monomial terms
            term_strs = []
            for term in model['terms']:
                c = term['coefficient']
                if len(term['exponents']) == 0:
                    # Constant term
                    term_strs.append(f"{c:.3e}")
                else:
                    monomial_parts = [f"{c:.3e}"]
                    for param, exp in term['exponents'].items():
                        if exp == 1.0:
                            monomial_parts.append(param)
                        else:
                            monomial_parts.append(f"cp.power({param}, {exp:.3f})")
                    term_strs.append(" * ".join(monomial_parts))
            return f"    {metric} >= " + " + ".join(term_strs)
        elif 'c' in model and 'exponents' in model:
            # Old monomial format: single term
            c = model['c']
            exp_parts = [f"cp.power({p}, {model['exponents'][p]:.3f})" 
                        for p in param_names if p in model['exponents']]
            return f"    {metric} >= {c:.3e} * " + " * ".join(exp_parts)
        else:
            # Fallback: just show the formula
            return f"    # {metric}: {model.get('formula', 'N/A')}"

    def export_pareto_model_for_cvxpy(self, fit_results, filename=None):
        """
        Export the fitted Pareto model in a format suitable for CVXPY geometric programming.

        Works with results from fit_pareto_surface_parametric() or fit_pareto_output_space().

        Args:
            fit_results: Results from fit_pareto_surface_parametric() or fit_pareto_output_space()
            filename: Optional filename to save the model (JSON format)

        Returns:
            Dictionary with CVXPY-compatible format
        """
        import json

        # Determine which type of fit results we have
        if 'param_names' in fit_results:
            # Parametric fit with abstract knobs (a0, a1, ...)
            param_names = fit_results['param_names']
            is_parametric = True
        elif 'independent_vars' in fit_results:
            # Fit with explicit independent variables
            param_names = fit_results['independent_vars']
            is_parametric = False
        else:
            raise ValueError("Unrecognized fit_results format")

        cvxpy_model = {
            'type': 'parametric' if is_parametric else 'explicit',
            'param_names': param_names,
            'output_metrics': fit_results['output_metrics'],
            'n_params': len(param_names),
            'constraints': []
        }

        # Add parameter bounds for parametric models
        if is_parametric and 'param_values' in fit_results:
            param_values = fit_results['param_values']
            cvxpy_model['param_bounds'] = {
                param_names[i]: {
                    'min': float(param_values[:, i].min()),
                    'max': float(param_values[:, i].max())
                }
                for i in range(len(param_names))
            }

        # Create constraint specifications for each output metric
        for metric, model in fit_results['models'].items():
            # Check model type
            if 'type' in model and model['type'] == 'posynomial' and 'terms' in model:
                # Posynomial format: sum of terms
                constraint = {
                    'output': metric,
                    'type': 'posynomial',
                    'terms': model['terms'],
                    'formula': model['formula'],
                    'r_squared': model['r_squared']
                }
            elif 'type' in model and model['type'] == 'monomial' and 'coefficient' in model:
                # Monomial format: single term (c * prod(x_i^p_i))
                constraint = {
                    'output': metric,
                    'type': 'monomial',
                    'coefficient': model['coefficient'],
                    'exponents': model['exponents'],
                    'formula': model.get('formula', ''),
                    'r_squared': model.get('r_squared', 0.0)
                }
            elif 'c' in model and 'exponents' in model:
                # Legacy monomial format
                constraint = {
                    'output': metric,
                    'type': 'monomial',
                    'coefficient': model['c'],
                    'exponents': model['exponents'],
                    'formula': model.get('formula', ''),
                    'r_squared': model.get('r_squared', 0.0)
                }
            else:
                raise ValueError(f"Unrecognized model format for metric {metric}: {list(model.keys())}")
            cvxpy_model['constraints'].append(constraint)

        # Add usage instructions
        usage_desc = 'Parametric Pareto surface model with design parameters. Use in CVXPY with DGP.'
        cvxpy_model['usage'] = {
            'description': usage_desc,
        }

        if filename:
            with open(filename, 'w') as f:
                # Custom JSON encoder for better readability
                json.dump(cvxpy_model, f, indent=2)
            logger.info(f"Exported CVXPY model to: {filename}")

        return cvxpy_model

    def fit_pareto_surface_parametric(self, pareto_df, output_metrics, n_params=3,
                                        input_params=None):
        """
        Fit a parametric representation of the Pareto surface using abstract knobs.

        Creates abstract parameters (a0, a1, ...) that parameterize the Pareto surface.
        Each output metric is fit as a monomial function of these abstract knobs:
            metric = c * a0^p0 * a1^p1 * ... * an^pn

        The abstract parameters are derived from the output metrics themselves,
        then each metric is fit as a monomial of these parameters using linear
        regression in log-space.

        Args:
            pareto_df: DataFrame containing Pareto-optimal designs
            output_metrics: List of output metrics to fit
            n_params: Number of abstract parameters (knobs) to create
            input_params: Ignored, kept for API compatibility

        Returns:
            Dictionary with fitted models and helper functions
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.neighbors import NearestNeighbors

        # Create abstract parameter names
        param_names = [f'a{i}' for i in range(n_params)]

        logger.info(f"Fitting parametric Pareto surface with {n_params} abstract knobs for {len(output_metrics)} metrics")

        # Extract output data
        Y = pareto_df[output_metrics].values

        assert np.all(Y > 0), "Some output metrics have non-positive values"

        n_points = len(Y)
        logger.info(f"Using {n_points} valid Pareto points")

        # Create abstract parameters from output metrics
        # Use first n_params output metrics (in log space) as the abstract parameters
        # This gives us parameters that naturally span the Pareto surface
        log_Y = np.log(Y)

        # Normalize each parameter to range [1, 10] for numerical stability
        # This gives good dynamic range in log space: log(1)=0, log(10)=2.3
        param_values = np.zeros((n_points, n_params))
        for i in range(n_params):
            if i < len(output_metrics):
                vals = log_Y[:, i]
                # Normalize to [0, 1] then scale to [1, 10]
                val_min, val_max = vals.min(), vals.max()
                if val_max > val_min:
                    normalized = (vals - val_min) / (val_max - val_min)  # [0, 1]
                    param_values[:, i] = 10.0 + 2.0 * normalized  # [1, 10]
                else:
                    param_values[:, i] = 5.0 * np.ones(n_points)  # Constant metric
            else:
                # If we need more params than metrics, use constant
                param_values[:, i] = 5.0 * np.ones(n_points)

        logger.info(f"Abstract parameters (normalized to [1, 10]):")
        for i in range(n_params):
            logger.info(f"  {param_names[i]}: range [{param_values[:, i].min():.3f}, {param_values[:, i].max():.3f}]")

        # Results dictionary
        results = {
            'output_metrics': output_metrics,
            'n_params': n_params,
            'param_names': param_names,
            'n_points': n_points,
            'models': {},
            'param_values': param_values,
            'pareto_data': pareto_df
        }

        # Log-transform abstract parameters for linear regression
        log_params = np.log(param_values)

        # Fit each output metric as a monomial: y = c * prod(a_i^p_i)
        # In log-space: log(y) = log(c) + sum(p_i * log(a_i))
        for i, metric in enumerate(output_metrics):
            y = Y[:, i]
            log_y = np.log(y)

            # Linear regression in log-space
            reg = LinearRegression()
            reg.fit(log_params, log_y)

            # Extract coefficient and exponents
            log_c = reg.intercept_
            c = np.exp(log_c)
            exponents = reg.coef_

            # Build model dictionary
            model = {
                'type': 'monomial',
                'coefficient': float(c),
                'exponents': {param_names[j]: float(exponents[j]) for j in range(n_params)}
            }

            # Build formula string
            exp_str = " * ".join([f"{p}^{e:.3f}" for p, e in model['exponents'].items()])
            model['formula'] = f"{metric} = {c:.3e} * {exp_str}"

            # Evaluation function
            def make_monomial_eval(coef, exps, params):
                def eval_func(*args, **kwargs):
                    if args:
                        param_dict = {params[i]: args[i] for i in range(len(params))}
                    else:
                        param_dict = kwargs
                    result = coef
                    for param, exp in exps.items():
                        result *= np.power(param_dict[param], exp)
                    return result
                return eval_func

            model['eval'] = make_monomial_eval(c, model['exponents'], param_names)

            # Compute R²
            y_pred = model['eval'](*[param_values[:, j] for j in range(n_params)])
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            model['r_squared'] = float(1 - (ss_res / ss_tot))

            results['models'][metric] = model
            logger.info(f"  {metric}: R²={model['r_squared']:.4f}")
            logger.info(f"    Formula: {model['formula']}")

        # Create nearest neighbor index in abstract parameter space
        param_normalized = (param_values - param_values.min(axis=0)) / (param_values.max(axis=0) - param_values.min(axis=0) + 1e-10)
        results['param_nn_index'] = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        results['param_nn_index'].fit(param_normalized)
        results['param_min'] = param_values.min(axis=0)
        results['param_range'] = param_values.max(axis=0) - param_values.min(axis=0) + 1e-10

        # Helper: find nearest design given abstract parameter values
        def find_nearest_design(*args, **kwargs):
            if args:
                query_params = np.array([args])
            else:
                query_params = np.array([[kwargs[p] for p in param_names]])
            query_normalized = (query_params - results['param_min']) / results['param_range']
            distance, index = results['param_nn_index'].kneighbors(query_normalized)
            return pareto_df.iloc[index[0][0]].to_dict(), float(distance[0][0])

        results['find_nearest_design'] = find_nearest_design

        # Helper: evaluate all metrics at given parameter values
        def evaluate_surface(*args, **kwargs):
            return {metric: results['models'][metric]['eval'](*args, **kwargs)
                    for metric in output_metrics}

        results['evaluate_surface'] = evaluate_surface

        return results

    def plot_pareto_fit_cross_section(self, fit_results, metric_x, metric_y, output_file=None):
        """
        Plot a cross-section of the Pareto surface showing actual data points
        and the fitted model approximation.

        Args:
            fit_results: Results from fit_pareto_surface_parametric()
            metric_x: Output metric for x-axis (e.g., 'delay')
            metric_y: Output metric for y-axis (e.g., 'Edynamic')
            output_file: Optional path to save the plot

        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt

        pareto_df = fit_results['pareto_data']
        param_values = fit_results['param_values']
        param_names = fit_results['param_names']
        n_params = fit_results['n_params']
        output_metrics = fit_results['output_metrics']

        # Get actual data points
        actual_x = pareto_df[metric_x].values
        actual_y = pareto_df[metric_y].values

        # Get predicted values from the fitted model
        model_x = fit_results['models'][metric_x]
        model_y = fit_results['models'][metric_y]

        pred_x = model_x['eval'](*[param_values[:, j] for j in range(n_params)])
        pred_y = model_y['eval'](*[param_values[:, j] for j in range(n_params)])

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Actual vs Predicted scatter (both metrics)
        ax1 = axes[0]
        ax1.scatter(actual_x, actual_y, alpha=0.5, label='Actual', c='blue', s=20)
        ax1.scatter(pred_x, pred_y, alpha=0.5, label='Predicted', c='red', s=20, marker='x')
        ax1.set_xlabel(metric_x)
        ax1.set_ylabel(metric_y)
        ax1.set_title(f'Pareto Front: {metric_x} vs {metric_y}')
        ax1.legend()
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Actual vs Predicted for metric_x
        ax2 = axes[1]
        ax2.scatter(actual_x, pred_x, alpha=0.5, c='blue', s=20)
        # Plot y=x line
        min_val = min(actual_x.min(), pred_x.min())
        max_val = max(actual_x.max(), pred_x.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        ax2.set_xlabel(f'Actual {metric_x}')
        ax2.set_ylabel(f'Predicted {metric_x}')
        ax2.set_title(f'{metric_x}: R²={model_x["r_squared"]:.4f}')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot 3: Actual vs Predicted for metric_y
        ax3 = axes[2]
        ax3.scatter(actual_y, pred_y, alpha=0.5, c='blue', s=20)
        # Plot y=x line
        min_val = min(actual_y.min(), pred_y.min())
        max_val = max(actual_y.max(), pred_y.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        ax3.set_xlabel(f'Actual {metric_y}')
        ax3.set_ylabel(f'Predicted {metric_y}')
        ax3.set_title(f'{metric_y}: R²={model_y["r_squared"]:.4f}')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to: {output_file}")

        return fig

    def plot_all_metrics_fit(self, fit_results, output_file=None):
        """
        Plot actual vs predicted for all output metrics in a grid.

        Args:
            fit_results: Results from fit_pareto_surface_parametric()
            output_file: Optional path to save the plot

        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt

        pareto_df = fit_results['pareto_data']
        param_values = fit_results['param_values']
        param_names = fit_results['param_names']
        n_params = fit_results['n_params']
        output_metrics = fit_results['output_metrics']

        n_metrics = len(output_metrics)
        cols = min(2, n_metrics)
        rows = (n_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, metric in enumerate(output_metrics):
            ax = axes[i]
            model = fit_results['models'][metric]

            actual = pareto_df[metric].values
            pred = model['eval'](*[param_values[:, j] for j in range(n_params)])

            ax.scatter(actual, pred, alpha=0.5, c='blue', s=20)

            # Plot y=x line
            min_val = min(actual.min(), pred.min())
            max_val = max(actual.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')

            ax.set_xlabel(f'Actual {metric}')
            ax.set_ylabel(f'Predicted {metric}')
            ax.set_title(f'{metric}: R²={model["r_squared"]:.4f}')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to: {output_file}")

        return fig

    def optimize_and_lookup_design(self, model_json_file, fit_results, output_file=None,
                                   objective='edp', additional_constraints=None):
        """
        Optimize a Pareto surface model and look up the actual design parameters.

        Args:
            model_json_file: Path to JSON file with Pareto surface model
            fit_results: Results from fit_pareto_surface_parametric() containing find_nearest_design()
            output_file: Optional path to save the optimal design JSON
            objective: Optimization objective ('edp', 'delay', 'energy', 'area')
            additional_constraints: Optional dict with constraints like {'delay': 1e-3, 'area': 3e-16}

        Returns:
            Dictionary with optimization results and design parameters
        """
        import json
        from src.hardware_model.tech_models.tech_library.optimize_pareto import optimize_pareto_surface

        # Optimize
        opt_results = optimize_pareto_surface(
            model_json_file,
            objective=objective,
            additional_constraints=additional_constraints
        )

        if not opt_results:
            logger.error("Optimization failed")
            return None

        # Look up the actual design parameters
        param_values = opt_results['parameters']
        design, distance = fit_results['find_nearest_design'](**param_values)

        # Log results
        logger.info("\n" + "="*60)
        logger.info("OPTIMAL DESIGN POINT")
        logger.info("="*60)
        logger.info(f"\nOptimal abstract parameters:")
        for name, value in param_values.items():
            logger.info(f"  {name} = {value:.6f}")

        logger.info(f"\nOptimal output metrics:")
        for name, value in opt_results['output_metrics'].items():
            logger.info(f"  {name} = {value:.6e}")

        logger.info(f"\nOptimal EDP: {opt_results['derived_metrics']['EDP']:.6e}")

        logger.info(f"\nNearest design point (distance={distance:.6e}):")
        # Print params of design
        all_params = [k for k in design.keys()]
        for param in sorted(all_params):
            logger.info(f"  {param} = {design[param]:.6e}")

        # Create result dictionary
        result = {
            'abstract_parameters': param_values,
            'output_metrics': opt_results['output_metrics'],
            'derived_metrics': opt_results['derived_metrics'],
            'design_parameters': design,
            'distance_to_pareto': distance,
            'objective': objective
        }

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"\nSaved optimal design to: {output_file}")

        return result


def test_sweep_tech_codesign(args):
    sweep_tech_codesign = SweepTechCodesign(args)
    if sweep_tech_codesign.codesign_module.cfg['args']['model_cfg'] == 'mvs_general_cfg':
        value_ranges = {"L": list(np.linspace(15e-9, 200e-9, 4)),
                        "W": list(np.linspace(15e-9, 1e-8, 4)),
                        "V_dd": list(np.linspace(0.7, 1.8, 4)),
                        "V_th": list(np.linspace(1.0, 2.5, 4)),
                        "tox": list(np.linspace(1e-9, 100e-9, 4)),
                        "beta_p_n": list(np.linspace(2.0, 2.0, 1)),
                        "mD_fac": list(np.linspace(0.5, 0.5, 1)),
                        "mu_eff_n": list(np.linspace(250.0e-4, 250.0e-4, 1)),
                        "mu_eff_p": list(np.linspace(125.0e-4, 125.0e-4, 1)),
                        "k_gate": list(np.linspace(3.9, 25, 3)),
                        "eps_semi": list(np.linspace(11.7, 11.7, 1)),
                        "tsemi": list(np.linspace(10.0e-9, 10e-9, 1)),
                        "Lext": list(np.linspace(10.0e-9, 10e-9, 1)),
                        "Lc": list(np.linspace(20.0e-9, 20e-9, 1)),
                        "eps_cap": list(np.linspace(3.9, 3.9, 1)),
                        "rho_c_n": list(np.linspace(7.0e-11, 7.0e-11, 1)),
                        "rho_c_p": list(np.linspace(7.0e-11, 7.0e-11, 1)),
                        "Rsh_c_n": list(np.linspace(7.0e-11, 7.0e-11, 1)),
                        "Rsh_c_p": list(np.linspace(7.0e-11, 7.0e-11, 1)),
                        "Rsh_ext_n": list(np.linspace(9000, 9000, 1)),
                        "Rsh_ext_p": list(np.linspace(9000, 9000, 1)),
                        "FO": list(np.linspace(4, 4, 1)),
                        "M": list(np.linspace(0.0001, 0.001, 1)),
                        "a": list(np.linspace(0.0001, 0.001, 1)),
                    }
    elif sweep_tech_codesign.codesign_module.cfg['args']['model_cfg'] == 'vs_cfg_latest':
        value_ranges = {"L": list(np.logspace(np.log10(15e-9), np.log10(200e-9), 5)),
                        "W": list(np.logspace(np.log10(15e-9), np.log10(1e-8), 5)),
                        "V_dd": list(np.linspace(0.2, 1.8, 5)),
                        "V_th": list(np.linspace(0.2, 1.1, 5)),
                        "tox": list(np.logspace(np.log10(1e-9), np.log10(50e-9), 5)),
                        "k_gate": list(np.linspace(3.9, 25, 3)),
                    }


    params_to_sweep = list(value_ranges.keys())
    output_dir = os.path.join(os.path.dirname(__file__), "design_spaces")
    n_processes = getattr(args, 'n_processes', 1)
    csv_filename = sweep_tech_codesign.sweep_tech(params_to_sweep, value_ranges, output_dir, n_processes=n_processes)
    if csv_filename is not None:
        pareto_csv_filename = sweep_tech_codesign.prune_design_space(csv_filename)
        just_fit_pareto_surface(args, pareto_csv_filename, sweep_tech_codesign.codesign_module.cfg['args']['optimize'])
    else:
        logger.error("No results to prune")

    

def just_prune_design_space(args, filename):
    sweep_tech_codesign = SweepTechCodesign(args)
    sweep_tech_codesign.prune_design_space(filename)

def just_fit_pareto_surface(args, pareto_csv_filename, optimize=False, n_params=7):
    """
    Fit a parametric Pareto surface model to an existing Pareto front CSV file.

    Args:
        args: Command line arguments (for initialization)
        pareto_csv_filename: Path to Pareto front CSV file
        optimize: Whether to optimize the Pareto surface (default: False)
        n_params: Number of abstract parameters to use (default: 2)
    """
    import pandas as pd

    sweep_tech_codesign = SweepTechCodesign(args)

    # Read Pareto CSV
    logger.info(f"Reading Pareto front from: {pareto_csv_filename}")
    pareto_df = pd.read_csv(pareto_csv_filename)

    # Get output metrics from tech model
    output_metrics = sorted([col for col in pareto_df.columns
                            if col in sweep_tech_codesign.codesign_module.hw.circuit_model.tech_model.pareto_metric_db])

    logger.info(f"Found output metrics: {output_metrics}")

    # Fit parametric surface
    fit_results = sweep_tech_codesign.fit_pareto_surface_parametric(
        pareto_df,
        output_metrics=output_metrics,
        n_params=n_params
    )

    # Export model
    output_dir = os.path.dirname(pareto_csv_filename)
    base_name = os.path.basename(pareto_csv_filename).rsplit('.', 1)[0]
    model_filename = os.path.join(output_dir, f"{base_name}_model.json")

    sweep_tech_codesign.export_pareto_model_for_cvxpy(fit_results, filename=model_filename)

    logger.info(f"Fitted Pareto surface model saved to: {model_filename}")

    # Generate diagnostic plots
    plot_filename = os.path.join(output_dir, "figs", f"{base_name}_fit.png")
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    sweep_tech_codesign.plot_all_metrics_fit(fit_results, output_file=plot_filename)

    # Also generate cross-section plot for delay vs Edynamic if both exist
    things_to_plot = [['delay', 'Edynamic'], ['Edynamic', 'Pstatic'], ['Pstatic', 'area']]
    for thing in things_to_plot:
        if thing[0] in output_metrics and thing[1] in output_metrics:
            cross_section_filename = os.path.join(output_dir, "figs", f"{base_name}_{thing[0]}_vs_{thing[1]}_cross_section.png")
            sweep_tech_codesign.plot_pareto_fit_cross_section(
                fit_results, thing[0], thing[1], output_file=cross_section_filename
            )

    if optimize:
        sweep_tech_codesign.optimize_and_lookup_design(model_filename, fit_results)

    return fit_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Design space sweep and fit",
        description="Script to sweep and fit the design space",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        help="Name of benchmark to run"
    )
    parser.add_argument(
        "-f",
        "--savedir",
        type=str,
        default="logs/sweep_tech_codesign",
        help="Path to the save new architecture file",
    )
    parser.add_argument(
        "-i",
        "--inverse_pass_improvement",
        type=float,
        default=10,
        help="Improvement factor for inverse pass",
    )
    parser.add_argument(
        "--parasitics",
        type=str,
        choices=["detailed", "estimation", "none"],
        default="estimation",
        help="determines what type of parasitic calculations are done for wires",
    )

    parser.add_argument(
        "--openroad_testfile",
        type=str,
        default="src/tmp/pd/tcl/codesign_top.tcl",
        help="what tcl file will be executed for openroad",
    )
    parser.add_argument(
        "-a",
        "--area",
        type=float,
        default=1000000,
        help="Area constraint in um2",
    )
    parser.add_argument(
        "--no_memory",
        type=bool,
        default=False,
        help="disable memory modeling",
    )
    parser.add_argument(
        "-N",
        "--num_opt_iters",
        type=int,
        default=3,
        help="Number of tech optimization iterations"
    )
    parser.add_argument(
        "--figdir",
        type=str,
        default="logs/sweep_tech_codesign/figs",
        help="path to save figs"
    )
    parser.add_argument(
        "--debug_no_cacti",
        type=bool,
        default=False,
        help="disable cacti in the first iteration to decrease runtime when debugging"
    )
    parser.add_argument(
        "--tech_node",
        type=str,
        help="tech node in nm"
    )
    parser.add_argument(
        "--obj",
        type=str,
        default="edp",
        help="objective function to optimize"
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        help="symbolic model configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="codesign configuration file"
    )
    parser.add_argument(
        "--opt_pipeline",
        type=str,
        default="logic_device",
        help="optimization pipeline to use for inverse pass"
    )
    parser.add_argument(
        "--additional_cfg_file",
        type=str,
        help="path to an additional configuration file",
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=1,
        help="Number of processes to use for parallel sweep execution (default: 1)"
    )
    parser.add_argument(
        "--just_prune",
        type=bool,
        default=False,
        help="just prune the design space, don't run the sweep"
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="filename of the csv file to prune"
    )
    parser.add_argument(
        "--just_fit",
        type=bool,
        default=False,
        help="just fit the pareto surface, don't run the sweep"
    )
    parser.add_argument(
        "--optimize",
        type=bool,
        default=False,
        help="optimize the pareto surface, don't run the sweep"
    )
    args = parser.parse_args()
    if args.just_prune:
        just_prune_design_space(args, args.filename)
    elif args.just_fit:
        just_fit_pareto_surface(args, args.filename, args.optimize)
    else:
        test_sweep_tech_codesign(args)