# first party
import argparse
import logging
import time
import sys
import copy
import math
import os
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
logger = logging.getLogger(__name__)

# third party
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# custom
from src.inverse_pass.constraint import Constraint

from src import sim_util
from src.hardware_model.objective_evaluator import ObjectiveEvaluator
from src.inverse_pass.utils import DesignPointResult, visualize_top_designs


def log_info(msg, stage):
    if stage == "before optimization":
        print(msg)
    elif stage == "after optimization":
        logger.info(msg)

def satisfies_constraints(total_power, total_area, total_passive_power, design_point, max_system_power, max_system_power_density, tech_model, leakage_restriction):
    # Ensure numeric values by substituting any remaining symbolic expressions
    total_power = sim_util.xreplace_safe(total_power, tech_model.base_params.tech_values)
    total_area = sim_util.xreplace_safe(total_area, tech_model.base_params.tech_values)
    total_passive_power = sim_util.xreplace_safe(total_passive_power, tech_model.base_params.tech_values)

    if total_power > max_system_power:
        logger.info(f"total power {total_power} is greater than max system power {max_system_power} for design point {design_point}")
        return False
    total_area_mm2 = total_area *1e4
    power_density_w_cm2 = total_power / total_area_mm2
    if power_density_w_cm2 > max_system_power_density:
        logger.info(f"power density {power_density_w_cm2} ({total_area_mm2} mm^2) ({total_power} W) is greater than max system power density {max_system_power_density} for design point {design_point}")
        return False
    if leakage_restriction and (total_passive_power > total_power/3): # restriction is that passive power <= 1/3 total power
            logger.info(f"passive power {total_passive_power} is greater than 1/3 total power {total_power/3} for design point {design_point}")
            return False
    return True

def _worker_basic_optimization_chunk(args_tuple):
    """
    Worker function that evaluates all design points in its chunk and returns
    detailed metrics for each one.

    Args:
        args_tuple: (worker_id, chunk, evaluator, max_system_power, max_system_power_density)
            - worker_id: Worker identifier
            - chunk: List of (idx, design_point) tuples
            - evaluator: ObjectiveEvaluator instance (pickleable)
            - max_system_power: Maximum allowed system power
            - max_system_power_density: Maximum allowed system power density

    Returns:
        Tuple of (worker_id, list of DesignPointResult)
    """
    worker_id, chunk, evaluator, max_system_power, max_system_power_density, leakage_restriction = args_tuple

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

            constraints_satisfied = satisfies_constraints(evaluator.total_power, evaluator.total_area, evaluator.total_passive_power, design_point, max_system_power, max_system_power_density, tech_model, leakage_restriction)

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

    def basic_optimization(self, improvement, iteration, n_processes=100):
        self.iteration = iteration
        self.constraints = []
        start_time = time.time()
        cur_obj_val = sim_util.xreplace_safe(self.hw.obj, self.hw.circuit_model.tech_model.base_params.tech_values)
        filtered_pareto_df = self.hw.circuit_model.tech_model.pareto_df

        if self.hw.cfg["args"]["MUL_restriction"]:
            filtered_pareto_df = filtered_pareto_df[filtered_pareto_df["MUL"] == 1]


        total_design_points = len(filtered_pareto_df)
        logger.info(f"Starting brute force optimization with {total_design_points} design points using {n_processes} process(es)")
        logger.info(f"objective: {self.hw.obj}")
        # Parallel execution

        # Prepare design points with indices and randomize order
        design_points = [
            (i, row._asdict()) for i, row in enumerate(filtered_pareto_df.itertuples(index=False))
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
            (worker_id, chunk, evaluator, self.max_system_power, self.max_system_power_density, self.hw.cfg["args"]["leakage_restriction"])
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
            visualize_top_designs(all_results, self.iteration, obj_type=self.hw.obj_fn, top_percent=1, output_dir=self.save_dir)
        else:
            best_design_point = None
            best_obj_val = math.inf
            best_value_clk_period = None
            logger.warning("No valid designs found")

        #if best_design_point is None or best_obj_val >= cur_obj_val:
        #    logger.warning("No better solution found in this iteration")
        #    return cur_obj_val, False

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
        if opt == "basic":
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
