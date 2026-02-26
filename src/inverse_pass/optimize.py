# first party
import argparse
import logging
import time
import sys
import copy
import math
import os
import json
import pickle
import itertools
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

def compute_constraints(total_power, total_area, total_passive_power, max_system_power, max_system_power_density, tech_model, leakage_restriction):
    """Return normalized constraint violation values.

    Each value is (actual / limit - 1): <= 0 means satisfied, > 0 means violated
    with magnitude proportional to the degree of violation.
    The leakage key is always present (-1.0 = trivially satisfied when leakage_restriction=False).
    """
    total_power = sim_util.xreplace_safe(total_power, tech_model.base_params.tech_values)
    total_area = sim_util.xreplace_safe(total_area, tech_model.base_params.tech_values)
    total_passive_power = sim_util.xreplace_safe(total_passive_power, tech_model.base_params.tech_values)
    total_area_mm2 = total_area * 1e4
    power_density_w_cm2 = total_power / total_area_mm2
    leakage_violation = (total_passive_power / (total_power / 3) - 1) if leakage_restriction else -1.0
    return {
        "power":         total_power / max_system_power - 1,
        "power_density": power_density_w_cm2 / max_system_power_density - 1,
        "leakage":       leakage_violation,
    }

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

            constraint_violations = compute_constraints(evaluator.total_power, evaluator.total_area, evaluator.total_passive_power, max_system_power, max_system_power_density, tech_model, leakage_restriction)
            constraints_satisfied = all(v <= 0 for v in constraint_violations.values())
            if not constraints_satisfied:
                violated = {k: v for k, v in constraint_violations.items() if v > 0}
                logger.info(f"constraints violated: { {k: f'{v:+.2%}' for k, v in violated.items()} }")

            # Expand memory indices to full metric dicts for reporting
            expanded_dp = dict(design_point)
            memory_config = design_point.get("memory", {})
            if memory_config and evaluator.memory_models:
                expanded_memory = {}
                for mem_name, mem_idx in memory_config.items():
                    if mem_name in evaluator.memory_models:
                        mem_model = evaluator.memory_models[mem_name]
                        row = mem_model.get_design_point_row()
                        expanded_memory[mem_name] = {"index": mem_idx, "capacity": mem_model.capacity_label, **row}
                    else:
                        expanded_memory[mem_name] = {"index": mem_idx}
                expanded_dp["memory"] = expanded_memory
            fu_logic_config = design_point.get("fu_logic", {})
            if fu_logic_config and evaluator.logic_unit_models:
                expanded_fu_logic = {}
                for rsc_name, fu_idx in fu_logic_config.items():
                    if rsc_name in evaluator.logic_unit_models:
                        lum = evaluator.logic_unit_models[rsc_name]
                        expanded_fu_logic[rsc_name] = {"index": fu_idx, "function": lum.function, **lum.get_design_point_row()}
                    else:
                        expanded_fu_logic[rsc_name] = {"index": fu_idx}
                expanded_dp["fu_logic"] = expanded_fu_logic

            result = DesignPointResult(
                design_point=expanded_dp,
                obj_value=obj_value,
                constraint_violations=constraint_violations,
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
                satisfies_constraints=constraints_satisfied,
                execution_time=evaluator.execution_time,
                total_active_energy=evaluator.total_active_energy,
                total_passive_energy=evaluator.total_passive_energy,
                total_area=sim_util.xreplace_safe(evaluator.total_area, tech_model.base_params.tech_values),
                latency_breakdown=dict(evaluator.latency_breakdown),
                latency_breakdown_pct=dict(evaluator.latency_breakdown_pct),
                latency_memory_by_block=dict(evaluator.latency_memory_by_block),
                latency_memory_by_block_pct=dict(evaluator.latency_memory_by_block_pct),
                total_logic_ops=evaluator.total_logic_ops,
                total_memory_ops=evaluator.total_memory_ops,
                active_energy_breakdown=dict(evaluator.active_energy_breakdown),
                active_energy_breakdown_pct=dict(evaluator.active_energy_breakdown_pct),
                active_energy_memory_by_block=dict(evaluator.active_energy_memory_by_block),
                active_energy_memory_by_block_pct=dict(evaluator.active_energy_memory_by_block_pct),
                passive_power_breakdown=dict(evaluator.passive_power_breakdown),
                passive_power_breakdown_pct=dict(evaluator.passive_power_breakdown_pct),
                passive_power_memory_by_block=dict(evaluator.passive_power_memory_by_block),
                passive_power_memory_by_block_pct=dict(evaluator.passive_power_memory_by_block_pct),
                loop_ii_info=dict(evaluator.loop_ii_info),
            )
            results.append(result)

            # Log only when we find a new best
            if constraints_satisfied and obj_value < best_obj_val:
                best_obj_val = obj_value
                #logger.info(f"worker {worker_id} new best objective value: {obj_value}, design point: {design_point}")

    return (worker_id, results)


class DesignSpaceNN:
    """Maps PCA-space suggestions to nearest valid design point indices.

    Parameterizes each design space by its principal components so that
    Optuna's surrogate can learn geometry without corner-seeking artifacts.
    """
    def __init__(self, X, n_components=3):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        X = np.asarray(X, dtype=float)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        n_components = min(n_components, X.shape[0] - 1, X.shape[1])
        self.pca = PCA(n_components=n_components).fit(Xs)
        X_pca = self.pca.transform(Xs)
        self.lo = X_pca.min(axis=0)
        self.hi = X_pca.max(axis=0)
        self.nn = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(Xs)
        self.n_components = n_components
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_

    def suggest_and_lookup(self, trial, prefix):
        pca_vec = [
            trial.suggest_float(f"{prefix}_pc{i}", float(self.lo[i]), float(self.hi[i]))
            for i in range(self.n_components)
        ]
        Xs_query = self.pca.inverse_transform([pca_vec])
        _, idx = self.nn.kneighbors(Xs_query)
        return int(idx[0][0])


class Optimizer:
    def __init__(self, hw, tmp_dir, save_dir, max_power, max_power_density, test_config=False, opt_pipeline="block_vector"):
        self.hw = hw
        self.disabled_knobs = []
        self.test_config = test_config
        self.tmp_dir = tmp_dir
        self.save_dir = save_dir
        self.opt_pipeline = opt_pipeline
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


        n_tech_points = len(filtered_pareto_df)
        logger.info(f"objective: {self.hw.obj}")

        # Build memory design space: {mem_name: num_design_points}
        memory_design_space = {}
        for mem_name, mem_model in self.hw.memory_models.items():
            if mem_model.num_design_points > 0:
                memory_design_space[mem_name] = mem_model.num_design_points
        mem_names = sorted(memory_design_space.keys())

        fu_rsc_names = list(self.hw.logic_unit_models.keys()) if self.hw.logic_unit_models else []

        # Generate all memory index combinations
        if mem_names:
            mem_ranges = [range(memory_design_space[n]) for n in mem_names]
            memory_combinations = list(itertools.product(*mem_ranges))
        else:
            memory_combinations = [()]

        n_mem_combos = len(memory_combinations)
        total_design_points = n_tech_points * n_mem_combos
        logger.info(f"Starting brute force optimization: {n_tech_points} tech points x {n_mem_combos} memory combos = {total_design_points} total design points using {n_processes} process(es)")
        for mem_name in mem_names:
            logger.info(f"  memory '{mem_name}': capacity={self.hw.memory_models[mem_name].capacity_label}, {memory_design_space[mem_name]} design points")

        # Prepare cross-product design points with indices and randomize order
        design_points = []
        for i, row in enumerate(filtered_pareto_df.itertuples(index=False)):
            tech_dp = row._asdict()
            for mem_combo in memory_combinations:
                dp = {
                    "logic": tech_dp,
                    "memory": {name: idx for name, idx in zip(mem_names, mem_combo)},
                    "fu_logic": {rsc: i for rsc in fu_rsc_names},
                }
                design_points.append((len(design_points), dp))
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

        # Save all results to disk for system-level composition
        results_path = os.path.join(self.save_dir, f"all_design_point_results_iter_{self.iteration}.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(all_results, f)
        logger.info(f"Saved {len(all_results)} design point results to {results_path}")

        # Save JSON summary of top valid results
        summary = [{
            "obj_value": r.obj_value,
            "execution_time": r.execution_time,
            "total_active_energy": r.total_active_energy,
            "total_passive_energy": r.total_passive_energy,
            "total_area": r.total_area,
            "total_power": r.total_power,
            "clk_period": r.clk_period,
            "satisfies_constraints": r.satisfies_constraints,
            "memory": r.design_point.get("memory", {}),
            "loop_ii_info": r.loop_ii_info,
        } for r in sorted_results]
        summary_path = os.path.join(self.save_dir, f"design_point_summary_iter_{self.iteration}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

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
        # Apply best memory design points
        best_memory_config = best_design_point.get("memory", {})
        for mem_name, mem_model in self.hw.memory_models.items():
            if mem_name in best_memory_config:
                mem_model.set_design_point(best_memory_config[mem_name])
        self.hw.calculate_objective()
        logger.info(f"ending total_active_energy: {sim_util.xreplace_safe(self.hw.total_active_energy, self.hw.circuit_model.tech_model.base_params.tech_values)}")
        logger.info(f"ending total_passive_power: {sim_util.xreplace_safe(self.hw.total_passive_energy/self.hw.execution_time, self.hw.circuit_model.tech_model.base_params.tech_values)}")
        return 1, False


    def bayesian_optimization(self, improvement, iteration, n_trials=500, n_parallel=50, fu_grouping="shared", mem_grouping="by_capacity", sampler="qmc", n_pca_components_logic=5, n_pca_components_memory=3):
        """
        Sample-efficient alternative to brute-force enumeration using Bayesian
        optimisation (Optuna TPE sampler).

        Instead of exhaustively evaluating every design point in the joint
        logic × memory space, this method iteratively proposes promising
        candidates guided by a surrogate model built from previously observed
        objective values.  Evaluations are batched so that each batch is run
        in parallel using the same worker pool as ``basic_optimization``.

        Algorithm
        ---------
        1. Initialise an Optuna ``Study`` (minimise) with a TPE sampler.
        2. Repeat for ``ceil(n_trials / n_parallel)`` batches:
           a. Ask Optuna for ``n_parallel`` candidate design points.  Each
              trial proposes a logic technology index (into the Pareto front)
              and one memory design-point index per memory block.
           b. Evaluate the batch in parallel via ``ProcessPoolExecutor``,
              reusing ``_worker_basic_optimization_chunk``.
           c. Report the observed objective (or ``inf`` for constraint
              violations) back to Optuna so the surrogate model is updated.
        3. Apply the globally best design point found to the hardware model.

        Args:
            improvement:  Unused regularisation parameter (kept for API
                          compatibility with ``basic_optimization``).
            iteration:    Current outer optimisation iteration index.  Used for
                          naming saved artefacts (pkl, JSON, plots).
            n_trials:     Total number of design points to evaluate across all
                          batches (default 500).
            n_parallel:   Number of design points evaluated in parallel per
                          batch, i.e. the Optuna batch size and the number of
                          worker processes per batch (default 50).
            fu_grouping:  How FU logic tech points are suggested.
                          "shared"      - all FUs reuse the global logic_idx (default).
                          "by_function" - one suggest per unique function type (e.g. Mult16, Add16).
                          "independent" - one suggest per FU resource.
            mem_grouping: How memory blocks are suggested.
                          "by_capacity" - one suggest per unique capacity label (default).
                          "independent" - one suggest per memory block.
            sampler:      Optuna sampler. "qmc" (default) uses QMCSampler (Sobol) —
                          constant ask cost, good coverage. "tpe" uses TPESampler —
                          adaptive but ask cost grows O(n_completed × d); only
                          recommended with fu_grouping "shared" or "by_function".
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("optuna is required for Bayesian optimization: pip install optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.iteration = iteration
        self.constraints = []
        filtered_pareto_df = self.hw.circuit_model.tech_model.pareto_df
        if self.hw.cfg["args"]["MUL_restriction"]:
            filtered_pareto_df = filtered_pareto_df[filtered_pareto_df["MUL"] == 1]

        pareto_rows = [row._asdict() for row in filtered_pareto_df.itertuples(index=False)]
        n_tech_points = len(pareto_rows)

        memory_design_space = {}
        for mem_name, mem_model in self.hw.memory_models.items():
            if mem_model.num_design_points > 0:
                memory_design_space[mem_name] = mem_model.num_design_points
        mem_names = sorted(memory_design_space.keys())

        # Per-FU logic design space: all FU models share the same pareto front size
        fu_logic_design_space = {}
        if self.hw.logic_unit_models:
            any_lum = next(iter(self.hw.logic_unit_models.values()))
            n_fu_dp = any_lum.num_design_points
            for rsc_name in self.hw.logic_unit_models:
                fu_logic_design_space[rsc_name] = n_fu_dp

        logger.info(f"Starting Bayesian optimization: {n_tech_points} tech points, memory space: {memory_design_space}, fu_logic FUs: {len(fu_logic_design_space)}")
        logger.info(f"Running {n_trials} trials with {n_parallel} parallel workers per batch")

        evaluator = ObjectiveEvaluator.from_hardware_model(self.hw)
        leakage_restriction = self.hw.cfg["args"]["leakage_restriction"]

        # Build PCA-based NN models using only quantities that enter the objective.
        # Logic/FU space: delay, E_act_inv, P_pass_inv, area (from _precomputed, which
        # evaluates the tech model's symbolic expressions at each pareto row) and V_dd
        # (for wire energy). filtered_indices maps filtered rows back to _precomputed arrays.
        filtered_indices = filtered_pareto_df.index.to_numpy()
        assert self.hw.logic_unit_models, "bayesian_optimization requires logic_unit_models to be populated"
        any_lum = next(iter(self.hw.logic_unit_models.values()))
        p = any_lum._precomputed
        X_logic = np.column_stack([
            p["delay"][filtered_indices],
            p["E_act_inv"][filtered_indices],
            p["P_pass_inv"][filtered_indices],
            p["area"][filtered_indices],
            filtered_pareto_df["V_dd"].to_numpy(dtype=float),
        ])
        logic_nn = DesignSpaceNN(X_logic, n_components=n_pca_components_logic)
        logger.info(f"Logic NN: {logic_nn.n_components} PCA components, "
                    f"variance explained: {logic_nn.explained_variance_ratio_.cumsum()[-1]:.1%}")

        assert fu_grouping in ("shared", "by_function", "independent"), f"Invalid fu_grouping: {fu_grouping}"
        assert mem_grouping in ("by_capacity", "independent"), f"Invalid mem_grouping: {mem_grouping}"

        # Memory: one NN per group key; group key is capacity_label or mem_name.
        _MEM_FEATS = ["cacheHitLatency_ns", "cacheWriteLatency_ns",
                      "cacheHitDynamicEnergy_nJ", "cacheWriteDynamicEnergy_nJ",
                      "cacheLeakage_mW", "cacheArea_mm2",
                      "retentionTime_ns", "refreshEnergy_nJ"]
        mem_group_nns = {}   # group_key -> DesignSpaceNN
        mem_to_group = {}    # mem_name -> group_key
        for mem_name, mem_model in self.hw.memory_models.items():
            if mem_model.num_design_points > 0:
                gk = mem_model.capacity_label if mem_grouping == "by_capacity" else mem_name
                mem_to_group[mem_name] = gk
                if gk not in mem_group_nns:
                    cols = [c for c in _MEM_FEATS if c in mem_model.pareto_df.columns]
                    X_mem = mem_model.pareto_df[cols].to_numpy(dtype=float)
                    mem_group_nns[gk] = DesignSpaceNN(X_mem, n_components=n_pca_components_memory)
                    logger.info(f"Memory group '{gk}' NN: {mem_group_nns[gk].n_components} PCA components, "
                                f"variance explained: {mem_group_nns[gk].explained_variance_ratio_.cumsum()[-1]:.1%}")

        # FU logic draws from the same pareto front as global logic — reuse logic_nn.
        # Build fu_groups (group_key -> [rsc_names]) and rsc_to_fu_group for ask loop.
        fu_groups = {}        # group_key -> [rsc_names]
        rsc_to_fu_group = {}  # rsc_name -> group_key
        if fu_grouping == "by_function":
            for rsc_name, lum in self.hw.logic_unit_models.items():
                fu_groups.setdefault(lum.function, []).append(rsc_name)
                rsc_to_fu_group[rsc_name] = lum.function
        elif fu_grouping == "independent":
            for rsc_name in fu_logic_design_space:
                fu_groups[rsc_name] = [rsc_name]
                rsc_to_fu_group[rsc_name] = rsc_name
        # "shared": fu_groups stays empty — FUs reuse logic_idx in ask loop


        assert sampler in ("qmc", "tpe"), f"Invalid sampler: {sampler}"
        if sampler == "tpe":
            # constraints_func reads the "constraint" user attr set during tell.
            # TPE models feasibility separately from the objective, so infeasible
            # trials don't pollute the surrogate for the objective.
            _sampler = optuna.samplers.TPESampler(
                seed=42,
                constraints_func=lambda trial: trial.user_attrs.get("constraints", [0.0, 0.0, 0.0]),
            )
        else:
            _sampler = optuna.samplers.QMCSampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=_sampler)
        all_results: List[DesignPointResult] = []
        logger.info(f"Starting Bayesian optimization with {sampler} sampler, {n_trials} trials and {n_parallel} parallel workers per batch")

        n_batches = math.ceil(n_trials / n_parallel)
        total_ask_time = 0.0
        total_eval_time = 0.0
        total_tell_time = 0.0
        for batch_idx in range(n_batches):
            batch_size = min(n_parallel, n_trials - batch_idx * n_parallel)

            # Ask optuna for a batch of trials
            t0 = time.time()
            batch_trials = []
            batch_dps = []
            for i in range(batch_size):
                #t1 = time.time()
                trial = study.ask()
                logic_idx = logic_nn.suggest_and_lookup(trial, "logic")
                group_indices = {
                    gk: mem_group_nns[gk].suggest_and_lookup(trial, f"mem_{gk}")
                    for gk in mem_group_nns
                }
                mem_indices = {name: group_indices[mem_to_group[name]] for name in memory_design_space}
                #logger.info(f"Suggested memory indices after time: {time.time() - t1:.2f}s: {mem_indices}")
                if fu_grouping == "shared" or not fu_logic_design_space:
                    fu_logic_indices = {rsc: logic_idx for rsc in fu_logic_design_space}
                else:
                    fu_group_indices = {
                        gk: logic_nn.suggest_and_lookup(trial, f"fu_{gk}")
                        for gk in fu_groups
                    }
                    fu_logic_indices = {
                        rsc: fu_group_indices[rsc_to_fu_group[rsc]]
                        for rsc in fu_logic_design_space
                    }
                #logger.info(f"Suggested FU logic indices after time: {time.time() - t1:.2f}s: {fu_logic_indices}")
                dp = {
                    "logic": pareto_rows[logic_idx],
                    "memory": mem_indices,
                    "fu_logic": fu_logic_indices,
                }
                #logger.info(f"time taken to suggest trial {i} of {batch_size}: {time.time() - t1:.2f}s")
                batch_trials.append(trial)
                batch_dps.append(dp)

            ask_time = time.time() - t0
            total_ask_time += ask_time

            # Partition batch into chunks for parallel evaluation
            t0 = time.time()
            indexed_batch = list(enumerate(batch_dps))
            n_workers = min(n_parallel, batch_size)
            chunk_size = math.ceil(batch_size / n_workers)
            chunks = [indexed_batch[i * chunk_size:(i + 1) * chunk_size] for i in range(n_workers)]
            chunks = [c for c in chunks if c]

            tasks = [
                (wid, chunk, evaluator, self.max_system_power, self.max_system_power_density, leakage_restriction)
                for wid, chunk in enumerate(chunks)
            ]

            results_by_idx = {}
            with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
                future_to_chunk = {executor.submit(_worker_basic_optimization_chunk, task): task[1] for task in tasks}
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    _, worker_results = future.result()
                    for (dp_idx, _), result in zip(chunk, worker_results):
                        results_by_idx[dp_idx] = result
            eval_time = time.time() - t0
            total_eval_time += eval_time

            # Tell results to optuna.
            # For TPE: set "constraint" user attr (0.0=feasible, 1.0=violated) so
            # constraints_func can separate feasibility from the objective model.
            # For QMC: user attrs are ignored; obj is still the real value.
            t0 = time.time()
            for dp_idx, trial in enumerate(batch_trials):
                result = results_by_idx[dp_idx]
                trial.set_user_attr("constraints", list(result.constraint_violations.values()))
                study.tell(trial, result.obj_value)
            tell_time = time.time() - t0
            total_tell_time += tell_time

            batch_results = [results_by_idx[i] for i in range(batch_size)]
            all_results.extend(batch_results)

            valid_in_batch = [r for r in batch_results if r.satisfies_constraints]
            if valid_in_batch:
                batch_best = min(valid_in_batch, key=lambda r: r.obj_value)
                logger.info(f"Batch {batch_idx+1}/{n_batches}: best={batch_best.obj_value:.6g}, valid={len(valid_in_batch)}/{batch_size} | ask={ask_time:.2f}s eval={eval_time:.2f}s tell={tell_time:.2f}s")
            else:
                logger.info(f"Batch {batch_idx+1}/{n_batches}: no valid designs in batch | ask={ask_time:.2f}s eval={eval_time:.2f}s tell={tell_time:.2f}s")

        logger.info(f"Bayesian optimization timing summary: ask={total_ask_time:.2f}s ({100*total_ask_time/(total_ask_time+total_eval_time+total_tell_time):.1f}%), eval={total_eval_time:.2f}s ({100*total_eval_time/(total_ask_time+total_eval_time+total_tell_time):.1f}%), tell={total_tell_time:.2f}s ({100*total_tell_time/(total_ask_time+total_eval_time+total_tell_time):.1f}%)")

        valid_results = [r for r in all_results if r.satisfies_constraints]
        sorted_results = sorted(valid_results, key=lambda r: r.obj_value)

        logger.info(f"Total designs evaluated: {len(all_results)}, valid designs: {len(valid_results)}")

        results_path = os.path.join(self.save_dir, f"all_design_point_results_iter_{self.iteration}.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(all_results, f)
        logger.info(f"Saved {len(all_results)} design point results to {results_path}")

        def _group_mem(mem_dict):
            out = {}
            for name, info in mem_dict.items():
                gk = mem_to_group.get(name, name)
                if gk not in out:
                    out[gk] = info
            return out

        summary = [{
            "obj_value": r.obj_value,
            "execution_time": r.execution_time,
            "total_active_energy": r.total_active_energy,
            "total_passive_energy": r.total_passive_energy,
            "total_area": r.total_area,
            "total_power": r.total_power,
            "clk_period": r.clk_period,
            "satisfies_constraints": r.satisfies_constraints,
            "memory": _group_mem(r.design_point.get("memory", {})),
            "loop_ii_info": r.loop_ii_info,
        } for r in sorted_results]
        summary_path = os.path.join(self.save_dir, f"design_point_summary_iter_{self.iteration}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        if sorted_results:
            best_result = sorted_results[0]
            best_design_point = best_result.design_point
            best_obj_val = best_result.obj_value
            best_value_clk_period = best_result.clk_period
            logger.info(f"Global best objective value: {best_obj_val}, design point: {best_design_point}")
            visualize_top_designs(all_results, self.iteration, obj_type=self.hw.obj_fn, top_percent=1, output_dir=self.save_dir)
        else:
            best_design_point = None
            best_obj_val = math.inf
            best_value_clk_period = None
            logger.warning("No valid designs found")

        self.hw.circuit_model.tech_model.set_params_from_design_point(best_design_point)
        self.hw.circuit_model.tech_model.base_params.set_symbol_value(self.hw.circuit_model.tech_model.base_params.clk_period, best_value_clk_period)
        best_memory_config = best_design_point.get("memory", {})
        for mem_name, mem_model in self.hw.memory_models.items():
            if mem_name in best_memory_config:
                mem_model.set_design_point(best_memory_config[mem_name])
        best_fu_logic_config = best_design_point.get("fu_logic", {})
        for rsc_name, lum in self.hw.logic_unit_models.items():
            if rsc_name in best_fu_logic_config:
                lum.set_design_point(best_fu_logic_config[rsc_name])
        self.hw.calculate_objective()
        return 1, False


    # note: improvement/regularization parameter currently only for inverse pass validation, so only using it for ipopt
    # example: improvement of 1.1 = 10% improvement
    def optimize(self, opt, improvement=10, disabled_knobs=[], iteration=0, **kwargs):
        self.disabled_knobs = disabled_knobs
        """
        Optimize the hardware model using the specified optimization method.

        Args:

        Returns:
            None
        """
        if opt == "basic":
            return self.basic_optimization(improvement, iteration)
        elif opt == "bayesian":
            return self.bayesian_optimization(
                improvement, iteration,
                n_trials=kwargs.get("n_trials", 500),
                n_parallel=kwargs.get("n_parallel", 50),
                fu_grouping=kwargs.get("fu_grouping", "shared"),
                mem_grouping=kwargs.get("mem_grouping", "by_capacity"),
                sampler=kwargs.get("sampler", "qmc"),
                n_pca_components_logic=kwargs.get("n_pca_components_logic", 5),
                n_pca_components_memory=kwargs.get("n_pca_components_memory", 3),
            )
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
