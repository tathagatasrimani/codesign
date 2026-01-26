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
import json

logger = logging.getLogger(__name__)

debug = True

def log_info(msg):
    if debug:
        logger.info(msg)

def _check_sweep_constraints_batch(args_tuple):
    """
    Worker function for parallel constraint checking (processes a batch of combinations).
    Must be at module level to be picklable.
    """
    batch_combos, params_to_sweep, symbol_map, constraints_leq, constraints_eq = args_tuple

    valid_combos = []
    for param_values in batch_combos:
        # Build substitution dict
        subs_dict = {}
        for param_name, param_value in zip(params_to_sweep, param_values):
            if param_name in symbol_map:
                subs_dict[symbol_map[param_name]] = param_value

        is_valid = True

        # Check leq constraints (constraint_expr <= 0 required)
        for constraint_name, constraint_expr in constraints_leq.items():
            try:
                val = float(constraint_expr.subs(subs_dict))
                if val > 0:
                    is_valid = False
                    break
            except (TypeError, ValueError):
                pass

        if not is_valid:
            continue

        # Check eq constraints (constraint_expr == 0 required)
        for constraint_name, constraint_expr in constraints_eq.items():
            try:
                val = float(constraint_expr.subs(subs_dict))
                if abs(val) > 1e-10:
                    is_valid = False
                    break
            except (TypeError, ValueError):
                pass

        if is_valid:
            valid_combos.append(param_values)

    return valid_combos


def _check_sweep_constraints(param_values, params_to_sweep, tech_model):
    """
    Check if a parameter combination satisfies the sweep constraints.
    Pre-filters combinations before submitting to worker pool for efficiency.

    Returns True if all constraints are satisfied, False otherwise.
    """
    # Create parameter mapping
    param_dict = dict(zip(params_to_sweep, param_values))

    # Build substitution dict using tech_model's symbols
    subs_dict = {}
    for param_name, param_value in param_dict.items():
        if hasattr(tech_model.base_params, param_name):
            param_symbol = getattr(tech_model.base_params, param_name)
            subs_dict[param_symbol] = param_value

    # Check leq constraints (constraint_expr <= 0 required)
    if hasattr(tech_model, 'sweep_constraints_leq'):
        for constraint_name, constraint_expr in tech_model.sweep_constraints_leq.items():
            try:
                val = float(constraint_expr.subs(subs_dict))
                if val > 0:
                    log_info(f"LESS THAN OR EQUAL TO CONSTRAINT VIOLATED {constraint_name} {constraint_expr} for config {param_dict}")
                    return False
            except (TypeError, ValueError):
                # If substitution fails (missing params), skip this constraint
                pass

    # Check eq constraints (constraint_expr == 0 required)
    if hasattr(tech_model, 'sweep_constraints_eq'):
        for constraint_name, constraint_expr in tech_model.sweep_constraints_eq.items():
            try:
                val = float(constraint_expr.subs(subs_dict))
                if abs(val) > 1e-10:  # small tolerance for floating point
                    log_info(f"EQUALITY CONSTRAINT VIOLATED {constraint_name} {constraint_expr} for config {param_dict}")
                    return False
            except (TypeError, ValueError):
                pass

    return True

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
        constraints_to_eval = tech_model.constraints + tech_model.sweep_constraints
        for constraint in constraints_to_eval:
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
        from src import codesign
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

        # Calculate total combinations without materializing (memory efficient)
        param_value_lists = [value_ranges[param] for param in params_to_sweep]
        total_combinations = 1
        for lst in param_value_lists:
            total_combinations *= len(lst)
        logger.info(f"Total combinations to evaluate: {total_combinations}")

        # Don't materialize all_combinations yet - will use generator for filtering
        all_combinations = None  # Will be set after filtering or if no filtering needed

        # Pre-filter combinations using sweep constraints (more efficient than checking in workers)
        has_sweep_constraints = (hasattr(tech_model, 'sweep_constraints_leq') and tech_model.sweep_constraints_leq) or \
                                (hasattr(tech_model, 'sweep_constraints_eq') and tech_model.sweep_constraints_eq)
        if has_sweep_constraints:
            initial_count = total_combinations
            logger.info(f"Filtering {initial_count} combinations using sweep constraints (parallel with {n_processes} processes)...")

            # Extract symbol map and constraints for pickling
            symbol_map = {}
            for param_name in params_to_sweep:
                if hasattr(tech_model.base_params, param_name):
                    symbol_map[param_name] = getattr(tech_model.base_params, param_name)

            constraints_leq = getattr(tech_model, 'sweep_constraints_leq', {})
            constraints_eq = getattr(tech_model, 'sweep_constraints_eq', {})

            # Memory-efficient streaming: process in chunks without materializing all combinations
            batch_size = 10000  # Small batches to limit memory per worker
            max_pending_batches = n_processes * 2  # Limit concurrent batches in memory

            valid_combinations = []
            combo_generator = itertools.product(*param_value_lists)
            processed_count = 0
            num_batches = math.ceil(initial_count / batch_size)
            logger.info(f"Processing {num_batches} batches of {batch_size} combinations each (streaming)")

            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                futures = []
                batch_num = 0

                while True:
                    # Submit batches up to the limit
                    while len(futures) < max_pending_batches:
                        batch = list(itertools.islice(combo_generator, batch_size))
                        if not batch:
                            break
                        future = executor.submit(
                            _check_sweep_constraints_batch,
                            (batch, params_to_sweep, symbol_map, constraints_leq, constraints_eq)
                        )
                        futures.append(future)
                        batch_num += 1

                    if not futures:
                        break

                    # Wait for at least one to complete
                    done_futures = [f for f in futures if f.done()]
                    if not done_futures:
                        # Wait for the first one
                        done_futures = [futures[0]]
                        done_futures[0].result()  # Block until complete

                    for future in done_futures:
                        futures.remove(future)
                        valid_batch = future.result()
                        valid_combinations.extend(valid_batch)
                        processed_count += batch_size
                        if processed_count % (batch_size * 100) == 0:
                            logger.info(f"Processed {processed_count}/{initial_count} ({100*processed_count/initial_count:.1f}%), valid so far: {len(valid_combinations)}")

            all_combinations = valid_combinations
            filtered_count = initial_count - len(all_combinations)
            logger.info(f"Filtered out {filtered_count} invalid combinations ({filtered_count/initial_count*100:.1f}%)")
        else:
            # No constraints - materialize all combinations
            logger.info(f"No sweep constraints, materializing {total_combinations} combinations...")
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

        # Build sense list for paretoset ("min" or "max" for each objective)
        sense = [self.codesign_module.hw.circuit_model.tech_model.pareto_metric_db[obj] for obj in objectives]

        logger.info(f"Computing Pareto front for objectives: {objectives} with sense: {sense}")
        print(f"Computing Pareto front for objectives: {objectives} with sense: {sense}")

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

            fit_results = just_fit_pareto_surface(self.args, pareto_filename)

        except Exception as e:
            logger.error(f"Error fitting Pareto surface: {e}")
            logger.info("Continuing without surface fit...")

        return pareto_filename

    def plot_pareto_fit_cross_section(self, pareto_df, metric_x, metric_y, model_dict, output_file=None):
        """
        Plot a cross-section of the Pareto surface showing actual data points
        and the fitted model approximation.

        Args:
            pareto_df: pandas DataFrame of Pareto-optimal designs
            metric_x: Output metric for x-axis (e.g., 'delay')
            metric_y: Output metric for y-axis (e.g., 'Edynamic')
            model_dict: Dictionary containing fitted model with 'constraints' and 'input_metrics'
            output_file: Optional path to save the plot

        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt

        # Get actual data points
        actual_x = pareto_df[metric_x].values
        actual_y = pareto_df[metric_y].values

        # Calculate predicted values using the monomial model
        predicted_x = self._predict_from_model(pareto_df, model_dict, metric_x)
        predicted_y = self._predict_from_model(pareto_df, model_dict, metric_y)

        # Create figure
        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(actual_x, actual_y, alpha=0.5, label='Actual', c='blue', s=20)
        ax.scatter(predicted_x, predicted_y, alpha=0.5, label='Predicted', c='red', s=20, marker='x')
        ax.set_xlabel(metric_x)
        ax.set_ylabel(metric_y)
        ax.set_title(f'Pareto Front: {metric_x} vs {metric_y}')
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to: {output_file}")

        return fig

    def _predict_from_model(self, pareto_df, model_dict, output_metric):
        """
        Predict output values using the fitted monomial model.

        Args:
            pareto_df: pandas DataFrame with input data
            model_dict: Dictionary containing 'constraints' with monomial fits
            output_metric: Name of the output metric to predict

        Returns:
            numpy array of predicted values
        """
        # Find the constraint for this output metric
        constraint = None
        for c in model_dict['constraints']:
            if model_dict['constraints'][c]['output'] == output_metric:
                constraint = c
                break

        if constraint is None:
            raise ValueError(f"No constraint found for output metric: {output_metric}")

        # Calculate predicted values: coeff * prod(input_i ^ exp_i)
        coefficient = model_dict['constraints'][constraint]['coefficient']
        exponents = model_dict['constraints'][constraint]['exponents']
        input_metrics = model_dict['input_metrics']

        predicted = np.ones(len(pareto_df)) * coefficient
        for input_name in input_metrics:
            if input_name in exponents:
                exp = exponents[input_name]
                input_values = pareto_df[input_name].values
                predicted = predicted * np.power(input_values, exp)

        return predicted

    def plot_predicted_vs_actual(self, pareto_df, model_dict, output_metric, output_file=None):
        """
        Plot predicted vs actual values for a single output metric with y=x reference line.

        Args:
            pareto_df: pandas DataFrame of Pareto-optimal designs
            model_dict: Dictionary containing fitted model with 'constraints' and 'input_metrics'
            output_metric: Name of the output metric to plot
            output_file: Optional path to save the plot

        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt

        # Get actual and predicted values
        actual = pareto_df[output_metric].values
        predicted = self._predict_from_model(pareto_df, model_dict, output_metric)

        # Get R-squared from model
        r_squared = None
        for c in model_dict['constraints']:
            if c['output'] == output_metric:
                r_squared = c.get('r_squared', None)
                break

        # Create figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

        # Plot data points
        ax.scatter(predicted, actual, alpha=0.5, c='blue', s=20, label='Data points')

        # Plot y=x reference line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='y = x')

        ax.set_xlabel(f'Predicted {output_metric}')
        ax.set_ylabel(f'Actual {output_metric}')
        title = f'Predicted vs Actual: {output_metric}'
        if r_squared is not None:
            title += f' (R² = {r_squared:.4f})'
        ax.set_title(title)
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to: {output_file}")

        return fig

    def plot_all_predicted_vs_actual(self, pareto_df, model_dict, base_name, output_dir=None):
        """
        Plot predicted vs actual for all output metrics in a grid.

        Args:
            pareto_df: pandas DataFrame of Pareto-optimal designs
            model_dict: Dictionary containing fitted model
            output_dir: Optional directory to save individual plots

        Returns:
            matplotlib figure with subplots
        """
        import matplotlib.pyplot as plt

        output_metrics = model_dict['output_metrics']
        n_metrics = len(output_metrics)

        # Calculate grid dimensions
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, output_metric in enumerate(output_metrics):
            ax = axes[i]

            # Get actual and predicted values
            actual = pareto_df[output_metric].values
            predicted = self._predict_from_model(pareto_df, model_dict, output_metric)

            # Get R-squared from model
            r_squared = None
            for c in model_dict['constraints']:
                if model_dict['constraints'][c]['output'] == output_metric:
                    r_squared = model_dict['constraints'][c].get('r_squared', None)
                    break

            # Plot data points
            ax.scatter(predicted, actual, alpha=0.5, c='blue', s=20)

            # Plot y=x reference line
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)

            ax.set_xlabel(f'Predicted')
            ax.set_ylabel(f'Actual')
            title = f'{output_metric}'
            if r_squared is not None:
                title += f' (R² = {r_squared:.4f})'
            ax.set_title(title)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if output_dir:
            output_file = os.path.join(output_dir, f'{base_name}_predicted_vs_actual_all.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to: {output_file}")

        return fig

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
                        "M": list(np.linspace(2, 2, 1)),
                        "a": list(np.linspace(0.5, 0.5, 1)),
                    }
    elif sweep_tech_codesign.codesign_module.cfg['args']['model_cfg'] == 'vs_cfg_latest':
        value_ranges = {"L": list(np.logspace(np.log10(15e-9), np.log10(3e-6), 20)),
                        "W": list(np.logspace(np.log10(15e-9), np.log10(9e-6), 20)),
                        "V_dd": list(np.logspace(np.log10(0.1), np.log10(5), 20)),
                        "V_th": list(np.logspace(np.log10(0.2), np.log10(1.1), 20)),
                        "tox": list(np.logspace(np.log10(1e-9), np.log10(100e-9), 10)),
                        "k_gate": list(np.linspace(3.9, 25, 3)),
                    }
    elif sweep_tech_codesign.codesign_module.cfg['args']['model_cfg'] == 'mvs_self_consistent_cfg':
        value_ranges = {"L": list(np.logspace(np.log10(15e-9), np.log10(200e-9), 7)),
                        "W": list(np.logspace(np.log10(15e-9), np.log10(500e-9), 7)),
                        "V_dd": list(np.logspace(np.log10(0.1), np.log10(3), 5)),
                        "V_th": list(np.logspace(np.log10(0.2), np.log10(1.5), 5)),
                        "tox": list(np.logspace(np.log10(1e-9), np.log10(30e-9), 5)),
                        "beta_p_n": list(np.logspace(np.log10(2.0), np.log10(2.0), 1)),
                        "mD_fac": list(np.logspace(np.log10(0.5), np.log10(0.5), 1)),
                        "mu_eff_n": list(np.logspace(np.log10(250.0e-4), np.log10(250.0e-4), 1)),
                        "mu_eff_p": list(np.logspace(np.log10(125.0e-4), np.log10(125.0e-4), 1)),
                        "k_gate": list(np.logspace(np.log10(3.9), np.log10(25), 2)),
                        "eps_semi": list(np.logspace(np.log10(11.7), np.log10(11.7), 1)),
                        "tsemi": list(np.logspace(np.log10(5.0e-9), np.log10(100e-9), 5)),
                        "Lext": list(np.logspace(np.log10(5.0e-9), np.log10(20e-9), 3)),
                        "Lc": list(np.logspace(np.log10(10.0e-9), np.log10(40e-9), 3)),
                        "eps_cap": list(np.logspace(np.log10(3.9), np.log10(3.9), 1)),
                        "rho_c_n": list(np.logspace(np.log10(5.0e-11), np.log10(5.0e-11), 1)),
                        "rho_c_p": list(np.logspace(np.log10(5.0e-11), np.log10(5.0e-11), 1)),
                        "Rsh_c_n": [10, 50, 100],
                        "Rsh_c_p": [10, 50, 100], 
                        "Rsh_ext_n": [100, 500, 1000],
                        "Rsh_ext_p": [100, 500, 1000],
                        "FO": list(np.logspace(np.log10(4), np.log10(4), 1)),
                        "M": list(np.logspace(np.log10(2), np.log10(2), 1)),
                        "a": list(np.logspace(np.log10(0.5), np.log10(0.5), 1)),
                    }


    params_to_sweep = list(value_ranges.keys())
    output_dir = os.path.join(os.path.dirname(__file__), "design_spaces")
    n_processes = getattr(args, 'n_processes', 1)
    csv_filename = sweep_tech_codesign.sweep_tech(params_to_sweep, value_ranges, output_dir, n_processes=n_processes)
    if csv_filename is not None:
        pareto_csv_filename = sweep_tech_codesign.prune_design_space(csv_filename)
        just_fit_pareto_surface(args, pareto_csv_filename)
    else:
        logger.error("No results to prune")

def just_prune_design_space(args, filename):
    sweep_tech_codesign = SweepTechCodesign(args)
    sweep_tech_codesign.prune_design_space(filename)

def fit_outputs_to_inputs(d, pareto_df):
    """
    Fit monomial models: output_i = coeff * (input_1^exp_1) * (input_2^exp_2) * ...

    In log-space: log(output_i) = log(coeff) + exp_1*log(input_1) + exp_2*log(input_2) + ...

    Returns a list of monomial constraint dictionaries.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    outputs = pareto_df[d["output_metrics"]].values
    inputs = pareto_df[d["input_metrics"]].values

    # Filter out rows with non-positive values
    valid_mask = (outputs > 0).all(axis=1) & (inputs > 0).all(axis=1)
    logger.info(f"Valid rows for fitting: {valid_mask.sum()} / {len(valid_mask)}")

    outputs = outputs[valid_mask]
    inputs = inputs[valid_mask]

    log_outputs = np.log(outputs)
    log_inputs = np.log(inputs)

    model = LinearRegression()
    model.fit(log_inputs, log_outputs)

    # model.coef_ shape: (n_outputs, n_inputs) - exponents for each output
    # model.intercept_ shape: (n_outputs,) - log(coefficient) for each output

    logger.info(f"Model coefficients (exponents): {model.coef_}")
    logger.info(f"Model intercepts (log of coefficients): {model.intercept_}")

    # Build monomial constraints list
    constraints = {}
    input_names = d["input_metrics"]
    output_names = d["output_metrics"]

    for i, output_name in enumerate(output_names):
        # Get exponents for this output
        if model.coef_.ndim == 1:
            # Single output case
            exponents = model.coef_
            intercept = model.intercept_
        else:
            # Multiple outputs case
            exponents = model.coef_[i]
            intercept = model.intercept_[i]

        coefficient = np.exp(intercept)

        # Build exponents dict
        exponents_dict = {input_names[j]: float(exponents[j]) for j in range(len(input_names))}

        # Build formula string
        formula_parts = [f"{coefficient:.3e}"]
        for input_name, exp in exponents_dict.items():
            formula_parts.append(f"{input_name}^{exp:.3f}")
        formula = f"{output_name} = " + " * ".join(formula_parts)

        # Calculate R-squared for this output
        if model.coef_.ndim == 1:
            predicted_log = log_inputs @ model.coef_ + model.intercept_
        else:
            predicted_log = log_inputs @ model.coef_[i] + model.intercept_[i]
        r_squared = r2_score(log_outputs[:, i] if log_outputs.ndim > 1 else log_outputs, predicted_log)

        constraint = {
            "output": output_name,
            "type": "monomial",
            "coefficient": float(coefficient),
            "exponents": exponents_dict,
            "formula": formula,
            "r_squared": float(r_squared)
        }
        constraints[output_name] = constraint

        logger.info(f"Fitted {output_name}: {formula} (R²={r_squared:.4f})")

    return model, constraints

def just_fit_pareto_surface(args, pareto_csv_filename):
    """
    Fit a parametric Pareto surface model to an existing Pareto front CSV file.

    Args:
        args: Command line arguments (for initialization)
        pareto_csv_filename: Path to Pareto front CSV file
    """
    import pandas as pd

    sweep_tech_codesign = SweepTechCodesign(args)

    # Read Pareto CSV
    logger.info(f"Reading Pareto front from: {pareto_csv_filename}")
    pareto_df = pd.read_csv(pareto_csv_filename)
    #pareto_df["V_ov"] = pareto_df["V_dd"] - pareto_df["V_th_eff"]
    #pareto_df.to_csv(pareto_csv_filename, index=False)

    # Get output metrics from tech model
    output_metrics = sorted([col for col in pareto_df.columns
                            if col in sweep_tech_codesign.codesign_module.hw.circuit_model.tech_model.pareto_metric_db])
    input_metrics = sorted([col for col in pareto_df.columns
                            if col in sweep_tech_codesign.codesign_module.hw.circuit_model.tech_model.input_metric_db])

    logger.info(f"Found output metrics: {output_metrics}")
    logger.info(f"Found input metrics: {input_metrics}")

    d = {"output_metrics": output_metrics, "input_metrics": input_metrics}

    model, constraints = fit_outputs_to_inputs(d, pareto_df)

    # Build param_bounds from input data
    param_bounds = {}
    for input_name in input_metrics:
        col_data = pareto_df[input_name].values
        col_data = col_data[col_data > 0]  # Filter non-positive values
        param_bounds[input_name] = {
            "min": float(col_data.min()),
            "max": float(col_data.max())
        }

    # Build full model JSON structure
    model_dict = {
        "type": "parametric",
        "param_names": input_metrics,
        "output_metrics": output_metrics,
        "input_metrics": input_metrics,
        "n_params": len(input_metrics),
        "constraints": constraints,
        "param_bounds": param_bounds,
        "usage": {
            "description": "Parametric Pareto surface model with monomial fits. Use in CVXPY with DGP."
        }
    }

    # Export model
    output_dir = os.path.dirname(pareto_csv_filename)
    base_name = os.path.basename(pareto_csv_filename).rsplit('.', 1)[0]
    model_filename = os.path.join(output_dir, f"{base_name}_model.json")

    with open(model_filename, 'w') as f:
        json.dump(model_dict, f, indent=2)

    logger.info(f"Saved model to: {model_filename}")

    fig_dir = os.path.join(output_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    sweep_tech_codesign.plot_all_predicted_vs_actual(pareto_df, model_dict, base_name, fig_dir)

    # generate cross-section plots for all output metrics
    things_to_plot = [
        ['R_avg_inv', 'C_gate'], 
        ['V_dd', 'C_gate'], 
        ['L', 'area'],
        ['Ioff', 'V_dd'],
        ['Ioff', 'L'],
        ['Ioff', 'R_avg_inv']
    ]
    for thing in things_to_plot:
        if thing[0] in output_metrics and thing[1] in output_metrics:
            cross_section_filename = os.path.join(fig_dir, f"{base_name}_{thing[0]}_vs_{thing[1]}_cross_section.png")
            sweep_tech_codesign.plot_pareto_fit_cross_section(
                pareto_df, thing[0], thing[1], model_dict, output_file=cross_section_filename
            )

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
    args = parser.parse_args()
    if args.just_prune:
        just_prune_design_space(args, args.filename)
    elif args.just_fit:
        just_fit_pareto_surface(args, args.filename)
    else:
        test_sweep_tech_codesign(args)