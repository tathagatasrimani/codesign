from src import codesign
import argparse
import csv
import itertools
import logging
import os
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import copy
logger = logging.getLogger(__name__)

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
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)

    def _evaluate_configuration(self, idx, param_values, params_to_sweep, tech_model):
        """
        Evaluate a single configuration in the sweep.
        This method is thread-safe and can be called in parallel.

        Args:
            idx: Configuration index
            param_values: Tuple of parameter values for this configuration
            params_to_sweep: List of parameter names
            tech_model: Reference to the tech model

        Returns:
            Tuple of (idx, result_row) or (idx, None) on error
        """
        try:
            logger.info(f"Running configuration {idx}")

            # Create a dictionary of current parameter values
            current_config = dict(zip(params_to_sweep, param_values))

            # Create a deep copy of tech_model to ensure thread safety
            local_tech_model = copy.deepcopy(tech_model)

            # Set parameter values in local_tech_model.base_params.tech_values
            for param_name, param_value in current_config.items():
                if hasattr(local_tech_model.base_params, param_name):
                    param_symbol = getattr(local_tech_model.base_params, param_name)
                    local_tech_model.base_params.tech_values[param_symbol] = param_value
                    logger.info(f"Config {idx}: Set {param_name} = {param_value}")
                else:
                    logger.warning(f"Config {idx}: Parameter '{param_name}' not found in base_params")

            # Re-initialize transistor equations with new parameter values
            local_tech_model.init_transistor_equations()

            # Collect results: input parameters + output metrics from param_db
            result_row = {}

            # Helper function to format numeric values in scientific notation with 4 decimal places
            def format_scientific(value):
                """Format numeric values in scientific notation with 4 decimal places."""
                if isinstance(value, (int, float)):
                    return f"{float(value):.4e}"
                return value

            # Add input parameters
            for param_name, param_value in current_config.items():
                result_row[param_name] = format_scientific(param_value)

            # Add output metrics from param_db
            for metric_name, metric_value in local_tech_model.sweep_output_db.items():
                # Handle both numeric and symbolic values
                if hasattr(metric_value, 'evalf'):
                    # Symbolic expression - evaluate it
                    evaluated_value = float(metric_value.xreplace(local_tech_model.base_params.tech_values).evalf())
                    result_row[metric_name] = format_scientific(evaluated_value)
                    logger.info(f"Config {idx}: Evaluated {metric_name}: {evaluated_value}")
                else:
                    # Already a numeric value
                    result_row[metric_name] = format_scientific(metric_value)
                    logger.info(f"Config {idx}: Evaluated numeric {metric_name}: {metric_value}")

            logger.info(f"Configuration {idx} completed successfully")
            return (idx, result_row)

        except Exception as e:
            logger.error(f"Error in configuration {idx}: {e}")
            return (idx, None)

    def sweep_tech(self, params_to_sweep, value_ranges, output_dir="sweep_results", flush_interval=10, n_threads=1):
        """
        Sweep through different tech model configurations in parallel.

        Args:
            params_to_sweep: List of parameter names to sweep (e.g., ['L', 'V_dd', 'tox'])
            value_ranges: Dictionary mapping parameter names to lists of values to sweep
                         e.g., {'L': [7e-9, 5e-9, 3e-9], 'V_dd': [0.7, 0.8, 0.9]}
            output_dir: Directory to save the CSV output file
            flush_interval: Number of iterations between CSV flushes (default: 10)
            n_threads: Number of threads to use for parallel evaluation (default: 1)

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
        logger.info(f"Flushing results to CSV every {flush_interval} iterations")

        # Prepare CSV file - open in append mode, but we'll track if header is written
        csv_initialized = False
        fieldnames = None
        csvfile = None
        writer = None
        results_buffer = []
        successful_count = 0

        try:
            # Run sweep
            for idx, param_values in enumerate(all_combinations, 1):
                logger.info(f"Running configuration {idx}/{total_runs}")

                # Create a dictionary of current parameter values
                current_config = dict(zip(params_to_sweep, param_values))

                # Set parameter values in tech_model.base_params.tech_values
                for param_name, param_value in current_config.items():
                    # Get the symbolic parameter from base_params
                    if hasattr(tech_model.base_params, param_name):
                        param_symbol = getattr(tech_model.base_params, param_name)
                        tech_model.base_params.tech_values[param_symbol] = param_value
                        logger.info(f"Set {param_name} = {param_value}")
                    else:
                        logger.info(f"Parameter '{param_name}' not found in base_params")

                # Re-initialize transistor equations with new parameter values
                tech_model.init_transistor_equations()

                # Collect results: input parameters + output metrics from param_db
                result_row = {}

                # Helper function to format numeric values in scientific notation with 4 decimal places
                def format_scientific(value):
                    """Format numeric values in scientific notation with 4 decimal places."""
                    if isinstance(value, (int, float)):
                        return f"{float(value):.4e}"
                    return value

                # Add input parameters
                for param_name, param_value in current_config.items():
                    result_row[param_name] = format_scientific(param_value)

                # Add output metrics from param_db
                for metric_name, metric_value in tech_model.sweep_output_db.items():
                    # Handle both numeric and symbolic values
                    if hasattr(metric_value, 'evalf'):
                        # Symbolic expression - evaluate it
                        evaluated_value = float(metric_value.xreplace(tech_model.base_params.tech_values).evalf())
                        result_row[metric_name] = format_scientific(evaluated_value)
                        logger.info(f"Evaluated {metric_name}: {evaluated_value}")
                    else:
                        # Already a numeric value
                        result_row[metric_name] = format_scientific(metric_value)
                        logger.info(f"Evaluated numeric {metric_name}: {metric_value}")

                results_buffer.append(result_row)
                successful_count += 1
                logger.info(f"Configuration {idx} completed successfully")

                # Initialize CSV file and write header on first iteration
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

                # Flush to CSV every flush_interval iterations or on last iteration
                if csv_initialized and (len(results_buffer) >= flush_interval or idx == total_runs):
                    writer.writerows(results_buffer)
                    csvfile.flush()
                    logger.info(f"Flushed {len(results_buffer)} results to CSV (total written: {idx}/{total_runs})")
                    results_buffer = []

            # Final flush for any remaining results
            if csv_initialized and results_buffer:
                writer.writerows(results_buffer)
                csvfile.flush()
                logger.info(f"Final flush: {len(results_buffer)} remaining results written")

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
    value_ranges = {"L": list(np.linspace(10e-9, 200e-9, 10)), 
                    "W": list(np.linspace(10e-9, 1e-8, 10)),
                    "V_dd": list(np.linspace(0.1, 1.4, 10)), 
                    "V_th": list(np.linspace(0.1, 1.4, 10)),
                    "tox": list(np.linspace(1e-9, 100e-9, 10)),
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
    sweep_tech_codesign.sweep_tech(params_to_sweep, value_ranges, output_dir)

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
    args = parser.parse_args()
    test_sweep_tech_codesign(args)