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
            tol = 1e-3
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
        csv_filename = os.path.join(output_dir, f"tech_sweep_{timestamp}.csv")

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

def test_sweep_tech_codesign(args):
    sweep_tech_codesign = SweepTechCodesign(args)
    value_ranges = {"L": list(np.linspace(15e-9, 200e-9, 4)),
                    "W": list(np.linspace(15e-9, 1e-8, 4)),
                    "V_dd": list(np.linspace(0.7, 1.2, 4)),
                    "V_th": list(np.linspace(0.5, 1.4, 4)),
                    "tox": list(np.linspace(1e-9, 100e-9, 4)),
                    "beta_p_n": list(np.linspace(2.0, 2.0, 1)),
                    "mD_fac": list(np.linspace(0.5, 0.5, 1)),
                    "mu_eff_n": list(np.linspace(250.0e-4, 250.0e-4, 1)),
                    "mu_eff_p": list(np.linspace(125.0e-4, 125.0e-4, 1)),
                    "k_gate": list(np.linspace(3.9, 25, 3)),
                    "eps_semi": list(np.linspace(11.7, 11.7, 1)),
                    "tsemi": list(np.linspace(10.0e-9, 50e-9, 4)),
                    "Lext": list(np.linspace(10.0e-9, 50e-9, 4)),
                    "Lc": list(np.linspace(20.0e-9, 100e-9, 4)),
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
    params_to_sweep = list(value_ranges.keys())
    output_dir = os.path.join(os.path.dirname(__file__), "design_spaces")
    n_threads = getattr(args, 'n_threads', 1)
    sweep_tech_codesign.sweep_tech(params_to_sweep, value_ranges, output_dir, n_threads=n_threads)

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
    args = parser.parse_args()
    test_sweep_tech_codesign(args)