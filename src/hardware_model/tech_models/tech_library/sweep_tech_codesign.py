from doctest import debug
from src import codesign
import argparse
import csv
import itertools
import logging
import os
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock, current_thread
import copy
import math
import sympy as sp
import sklearn
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

    def _evaluate_configuration(self, idx, param_values, params_to_sweep, tech_model):
        """
        Evaluate a single configuration in the sweep.
        Each thread has its own tech_model copy for true parallel execution.

        Args:
            idx: Configuration index
            param_values: Tuple of parameter values for this configuration
            params_to_sweep: List of parameter names
            tech_model: Thread-local copy of the tech model

        Returns:
            Tuple of (idx, result_row) or (idx, None) on error
        """
        try:
            # Create a dictionary of current parameter values
            current_config = dict(zip(params_to_sweep, param_values))

            # Set parameter values in tech_model.base_params.tech_values
            # Each thread has its own copy, so no locking needed
            for param_name, param_value in current_config.items():
                if hasattr(tech_model.base_params, param_name):
                    param_symbol = getattr(tech_model.base_params, param_name)
                    tech_model.base_params.tech_values[param_symbol] = param_value
                else:
                    logger.warning(f"Config {idx}: Parameter '{param_name}' not found in base_params")

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
                tol = 1e-3
                slack = sim_util.xreplace_safe(constraint.slack, tech_model.base_params.tech_values)
                if (slack > tol):
                    log_info(f"CONSTRAINT VIOLATED {constraint.label} for config {idx}, slack is {slack}")
                    return (idx, None)
                else:
                    log_info(f"CONSTRAINT SATISFIED {constraint.label} for config {idx}, slack is {slack}")

            return (idx, result_row)

        except Exception as e:
            logger.error(f"Error in configuration {idx}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return (idx, None)

    def sweep_tech(self, params_to_sweep, value_ranges, output_dir="sweep_results", flush_interval=10, n_threads=1, use_processes=True):
        """
        Sweep through different tech model configurations in parallel.

        Args:
            params_to_sweep: List of parameter names to sweep (e.g., ['L', 'V_dd', 'tox'])
            value_ranges: Dictionary mapping parameter names to lists of values to sweep
                         e.g., {'L': [7e-9, 5e-9, 3e-9], 'V_dd': [0.7, 0.8, 0.9]}
            output_dir: Directory to save the CSV output file
            flush_interval: Number of iterations between CSV flushes (default: 10)
            n_threads: Number of threads/processes to use for parallel evaluation (default: 1)
            use_processes: If True, use ProcessPoolExecutor (better for CPU-bound work).
                          If False, use ThreadPoolExecutor (default: True)

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
        executor_type = "process(es)" if use_processes else "thread(s)"
        logger.info(f"Using {n_threads} {executor_type} for parallel execution")
        logger.info(f"Flushing results to CSV every {flush_interval} iterations")

        # Create a pool of tech_model copies (one per worker)
        logger.info(f"Creating {n_threads} copies of tech_model for parallel execution...")
        tech_model_pool = [copy.deepcopy(tech_model) for _ in range(n_threads)]
        logger.info(f"Tech model copies created successfully")

        # Prepare CSV file - open in append mode, but we'll track if header is written
        csv_initialized = False
        fieldnames = None
        csvfile = None
        writer = None
        results_buffer = []
        successful_count = 0
        csv_lock = Lock()
        max_num_jobs_at_once = 10000
        num_iterations = math.ceil(len(all_combinations) / max_num_jobs_at_once)
        total_completed_count = 0  # Track overall progress across all batches

        try:
            for batch_num in range(num_iterations):
                batch_start_idx = batch_num * max_num_jobs_at_once
                batch_end_idx = min((batch_num + 1) * max_num_jobs_at_once, len(all_combinations))
                current_combinations = all_combinations[batch_start_idx:batch_end_idx]

                logger.info(f"Starting batch {batch_num + 1}/{num_iterations} ({len(current_combinations)} configurations)")

                # Choose executor based on use_processes flag
                ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

                # Run sweep in parallel
                with ExecutorClass(max_workers=n_threads) as executor:
                    if use_processes:
                        # For ProcessPoolExecutor, use the global worker function
                        # Prepare args tuples: (idx, param_values, params_to_sweep, tech_model)
                        tasks = [
                            (global_idx, param_values, params_to_sweep, tech_model_pool[(global_idx - 1) % n_threads])
                            for local_idx, param_values in enumerate(current_combinations, 1)
                            for global_idx in [batch_start_idx + local_idx]
                        ]
                        future_to_idx = {
                            executor.submit(_worker_evaluate_configuration, task): task[0]
                            for task in tasks
                        }
                    else:
                        # For ThreadPoolExecutor, use the instance method
                        future_to_idx = {
                            executor.submit(
                                self._evaluate_configuration,
                                global_idx,
                                param_values,
                                params_to_sweep,
                                tech_model_pool[(global_idx - 1) % n_threads]
                            ): global_idx
                            for local_idx, param_values in enumerate(current_combinations, 1)
                            for global_idx in [batch_start_idx + local_idx]
                        }

                    # Process results as they complete
                    for future in as_completed(future_to_idx):
                        idx, result_row = future.result()
                        total_completed_count += 1
                        if (total_completed_count % 100 == 0):
                            logger.info(f"Progress: {total_completed_count}/{total_runs} configurations completed")

                        if result_row is not None:
                            with csv_lock:
                                results_buffer.append(result_row)
                                successful_count += 1

                                # Initialize CSV file and write header on first successful result
                                if not csv_initialized and results_buffer:
                                    # Get all unique column names from first result(s)
                                    all_columns = set()
                                    for result in results_buffer:
                                        all_columns.update(result.keys())

                                    # Sort columns: output metrics first, then input parameters, then others
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
                        # Do this outside the result check so it happens regardless of success/failure
                        with csv_lock:
                            if csv_initialized and len(results_buffer) > 0:
                                if len(results_buffer) >= flush_interval or total_completed_count == total_runs:
                                    writer.writerows(results_buffer)
                                    csvfile.flush()
                                    logger.info(f"Flushed {len(results_buffer)} results to CSV (total written: {successful_count}/{total_runs})")
                                    results_buffer = []

                # Flush any remaining results from this batch
                with csv_lock:
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
            n_params = min(2, len(objectives) - 1)  # Use 2 params, or n_objectives-1 if fewer objectives
            fit_results = self.fit_pareto_surface_parametric(pareto_df, output_metrics=objectives, n_params=n_params)

            # Export model for CVXPY
            model_filename = os.path.join(output_dir, f"{base_name}_pareto_model.json")
            self.export_pareto_model_for_cvxpy(fit_results, filename=model_filename)

            logger.info(f"Successfully fitted and exported Pareto surface model")
        except Exception as e:
            logger.error(f"Error fitting Pareto surface: {e}")
            logger.info("Continuing without surface fit...")

        return pareto_filename

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

        # Create constraint specifications for each output metric
        for metric, model in fit_results['models'].items():
            constraint = {
                'output': metric,
                'coefficient': model['c'],
                'exponents': model['exponents'],
                'formula': model['formula'],
                'r_squared': model['r_squared']
            }
            cvxpy_model['constraints'].append(constraint)

        # Add usage instructions
        if is_parametric:
            usage_desc = 'Parametric Pareto surface model with abstract knobs. Use in CVXPY with DGP.'
            example_code = '''
import cvxpy as cp

# Define abstract parameter knobs (must be positive)
{var_defs}

# Define output metric variables
{output_defs}

# Add Pareto surface constraints
constraints = [
{constraint_examples}
]

# Your objective (e.g., minimize energy-delay product)
objective = cp.Minimize(Edynamic * delay)

# Solve with DGP
problem = cp.Problem(objective, constraints)
problem.solve(gp=True)

# After solving, look up actual design parameters
# outputs = fit_results['evaluate_surface']({knob_values})
# design, dist = fit_results['find_nearest_design']({knob_values})
'''.format(
                var_defs='\n'.join([f"{p} = cp.Variable(pos=True, name='{p}')" for p in param_names]),
                output_defs='\n'.join([f"{m} = cp.Variable(pos=True, name='{m}')" for m in fit_results['output_metrics']]),
                constraint_examples='\n'.join([
                    f"    {m} >= {fit_results['models'][m]['c']:.3e} * " +
                    " * ".join([f"cp.power({p}, {fit_results['models'][m]['exponents'][p]:.3f})"
                               for p in param_names])
                    for m in fit_results['output_metrics']
                ]),
                knob_values=', '.join([f"{p}={p}.value" for p in param_names])
            )
        else:
            usage_desc = 'Pareto surface model with explicit independent variables. Use in CVXPY with DGP.'
            example_code = '''
import cvxpy as cp

# Define independent variables (must be positive)
{var_defs}

# Define dependent output metric variables
{output_defs}

# Add Pareto surface constraints
constraints = [
{constraint_examples}
]

# Your objective
objective = cp.Minimize(...)  # Define based on your goals

# Solve with DGP
problem = cp.Problem(objective, constraints)
problem.solve(gp=True)
'''.format(
                var_defs='\n'.join([f"{p} = cp.Variable(pos=True, name='{p}')" for p in param_names]),
                output_defs='\n'.join([f"{m} = cp.Variable(pos=True, name='{m}')"
                                      for m in fit_results['output_metrics'] if m not in param_names]),
                constraint_examples='\n'.join([
                    f"    {m} >= {fit_results['models'][m]['c']:.3e} * " +
                    " * ".join([f"cp.power({p}, {fit_results['models'][m]['exponents'][p]:.3f})"
                               for p in param_names])
                    for m in fit_results['models'].keys()
                ])
            )

        cvxpy_model['usage'] = {
            'description': usage_desc,
            'example': example_code
        }

        if filename:
            with open(filename, 'w') as f:
                # Custom JSON encoder for better readability
                json.dump(cvxpy_model, f, indent=2)
            logger.info(f"Exported CVXPY model to: {filename}")

        return cvxpy_model

    def fit_pareto_surface_parametric(self, pareto_df, output_metrics, n_params=2):
        """
        Fit a parametric representation of the Pareto surface using abstract knobs.

        Creates abstract parameters (a0, a1, ...) that serve as knobs to traverse the Pareto surface.
        Each output metric is fit as a posynomial function of these knobs:
            area = c1 * a0^p1 * a1^q1 * ...
            delay = c2 * a0^p2 * a1^q2 * ...
            Edynamic = c3 * a0^p3 * a1^q3 * ...
            Pstatic = c4 * a0^p4 * a1^q4 * ...

        In optimization, you tune the knobs (a0, a1, ...) to find optimal output metrics,
        then look up the nearest design point to get actual input parameters.

        Args:
            pareto_df: DataFrame containing Pareto-optimal designs
            output_metrics: List of output metrics to fit.
            n_params: Number of abstract parameters (knobs) to use for parameterization

        Returns:
            Dictionary with:
                - 'models': posynomial fits for each output metric
                - 'param_values': the parameter values assigned to each Pareto point
                - 'find_nearest_design': function to look up design given knob values
        """
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.neighbors import NearestNeighbors
        from sklearn.decomposition import PCA

        logger.info(f"Fitting parametric Pareto surface with {n_params} parameters for {len(output_metrics)} metrics")

        # Extract output data
        Y = pareto_df[output_metrics].values

        assert np.all(Y > 0), "Some output metrics have non-positive values"

        n_points = len(Y)
        logger.info(f"Using {n_points} valid Pareto points")

        # Create abstract parameters using PCA on log-transformed output space
        # This gives us meaningful parameters that capture variance in the Pareto front
        log_Y = np.log(Y)

        if n_params >= len(output_metrics):
            raise ValueError(f"n_params ({n_params}) must be less than number of output metrics ({len(output_metrics)})")

        pca = PCA(n_components=n_params)
        param_values = pca.fit_transform(log_Y)

        param_names = [f'a{i}' for i in range(n_params)]

        logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        logger.info(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

        results = {
            'output_metrics': output_metrics,
            'n_params': n_params,
            'param_names': param_names,
            'n_points': n_points,
            'pca': pca,
            'param_shift': {param_names[i]: float(np.log(Y).min(axis=0)[i] - 1e-9)
                           for i in range(n_params)},
            'models': {},
            'param_values': param_values,
            'pareto_data': pareto_df
        }

        # Fit each output metric as function of abstract parameters
        # y_i = c_i * a0^p_i0 * a1^p_i1 * ...
        for i, metric in enumerate(output_metrics):
            y = Y[:, i]
            log_y = np.log(y)
            log_params = np.log(param_values)

            reg = LinearRegression()
            reg.fit(log_params, log_y)

            model = {
                'c': float(np.exp(reg.intercept_)),
                'exponents': {param_names[j]: float(reg.coef_[j]) for j in range(n_params)},
                'r_squared': float(reg.score(log_params, log_y))
            }

            terms = [f"{param_names[j]}^{model['exponents'][param_names[j]]:.3f}"
                    for j in range(n_params)]
            model['formula'] = f"{metric} = {model['c']:.3e} * " + " * ".join(terms)

            # Evaluation function
            def make_eval(c, exps, params):
                def eval_func(*args, **kwargs):
                    """Evaluate metric given parameter values (positional or keyword args)."""
                    if args:
                        if len(args) != len(params):
                            raise ValueError(f"Expected {len(params)} arguments, got {len(args)}")
                        param_vals = args
                    else:
                        param_vals = [kwargs[p] for p in params]

                    result = c
                    for j, p in enumerate(params):
                        result *= np.power(param_vals[j], exps[p])
                    return result
                return eval_func

            model['eval'] = make_eval(model['c'], model['exponents'], param_names)

            results['models'][metric] = model
            logger.info(f"  {metric}: R²={model['r_squared']:.4f}")

        # Create nearest neighbor index in parameter space
        results['param_nn_index'] = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        results['param_nn_index'].fit(param_values)

        # Helper: find nearest design given parameter values
        def find_nearest_design(*args, **kwargs):
            """Find nearest Pareto design point given parameter values."""
            if args:
                query_params = np.array([args])
            else:
                query_params = np.array([[kwargs[p] for p in param_names]])

            distance, index = results['param_nn_index'].kneighbors(query_params)
            return pareto_df.iloc[index[0][0]].to_dict(), float(distance[0][0])

        results['find_nearest_design'] = find_nearest_design

        # Helper: evaluate all metrics at given parameter values
        def evaluate_surface(*args, **kwargs):
            """Evaluate all output metrics given parameter values."""
            result = {}
            for metric in output_metrics:
                result[metric] = results['models'][metric]['eval'](*args, **kwargs)
            return result

        results['evaluate_surface'] = evaluate_surface

        return results


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

        print("tox values: ", value_ranges["tox"])

    params_to_sweep = list(value_ranges.keys())
    output_dir = os.path.join(os.path.dirname(__file__), "design_spaces")
    n_threads = getattr(args, 'n_threads', 1)
    csv_filename = sweep_tech_codesign.sweep_tech(params_to_sweep, value_ranges, output_dir, n_threads=n_threads)
    if csv_filename is not None:
        pareto_csv_filename = sweep_tech_codesign.prune_design_space(csv_filename)
        just_fit_pareto_surface(args, pareto_csv_filename)
    else:
        logger.error("No results to prune")

    


def just_prune_design_space(args, filename):
    sweep_tech_codesign = SweepTechCodesign(args)
    sweep_tech_codesign.prune_design_space(filename)

def just_fit_pareto_surface(args, pareto_csv_filename, n_params=2):
    """
    Fit a parametric Pareto surface model to an existing Pareto front CSV file.

    Args:
        args: Command line arguments (for initialization)
        pareto_csv_filename: Path to Pareto front CSV file
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

    return fit_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dennard Scaling, Multi-Core Architecture Experiment",
        description="Script demonstrating recreation of historical technology/architecture trends.",
        epilog="Text at the bottom of help",
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
        "--dummy",
        type=bool,
        default=True,
        help="dummy application"
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
        "--n_threads",
        type=int,
        default=1,
        help="Number of threads to use for parallel sweep execution (default: 1)"
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