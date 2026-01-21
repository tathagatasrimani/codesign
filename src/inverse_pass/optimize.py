# first party
import argparse
import logging
import time
import sys
import copy
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
logger = logging.getLogger(__name__)

# third party
import pyomo.environ as pyo
import sympy as sp
import cvxpy as cp
import numpy as np
# custom
from src.inverse_pass.preprocess import Preprocessor
from src.inverse_pass import curve_fit
from src.inverse_pass.constraint import Constraint

from src import sim_util
from src.hardware_model.hardwareModel import BlockVector

multistart = False
import warnings
warnings.filterwarnings("ignore")

def log_info(msg, stage):
    if stage == "before optimization":
        print(msg)
    elif stage == "after optimization":
        logger.info(msg)


def _worker_evaluate_design_point(args_tuple):
    """
    Worker function for parallel brute force optimization.
    Must be defined at module level to be picklable.

    Args:
        args_tuple: (design_point_index, design_point_dict, tech_model, hw_obj, constraints)

    Returns:
        (design_point_index, design_point_dict, obj_value, variables_dict) or (design_point_index, None, None, None) on failure
    """
    idx, design_point, tech_model, hw_obj, constraints = args_tuple

    try:
        # Set parameters from design point
        tech_model.set_params_from_design_point(design_point)
        tech_model.set_param_constant_constraints()

        #logger.info(f"evaluating design point {idx}: {design_point}")

        # Build and solve the cvxpy problem
        #logger.info(f"param constant constraints: {tech_model.param_constant_constraints}")
        param_constraints = [constr.constraint for constr in tech_model.param_constant_constraints]
        prob = cp.Problem(cp.Minimize(hw_obj), constraints + param_constraints)
        prob.solve(gp=True)

        

        # Extract variable values as a dict for returning
        variables_dict = {var.name(): var.value for var in prob.variables()}

        return (idx, design_point, prob.value, variables_dict)

    except Exception as e:
        logger.error(f"Worker error solving cvxpy problem for design point {idx}: {e}")
        return (idx, None, None, None)


class Optimizer:
    def __init__(self, hw, tmp_dir, max_power, max_power_density, test_config=False, opt_pipeline="block_vector"):
        self.hw = hw
        self.disabled_knobs = []
        self.objective_constraint_inds = []
        self.initial_alpha = None
        self.test_config = test_config
        self.tmp_dir = tmp_dir
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
        prob.solve(gp=True)
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
        lower_bound = sim_util.xreplace_safe(self.hw.obj, self.hw.circuit_model.tech_model.base_params.tech_values) / improvement
        self.constraints = self.hw.constraints + self.hw.circuit_model.constraints + self.hw.circuit_model.tech_model.constraints_cvxpy
        constraints = [constraint.constraint for constraint in self.constraints]
        for constraint in constraints:
            assert constraint.is_dgp(), f"constraint is not DGP: {constraint}"
            print(f"constraint: {constraint}")
        print(f"objective: {self.hw.obj}")

        total_design_points = len(self.hw.circuit_model.tech_model.pareto_df)
        logger.info(f"Starting brute force optimization with {total_design_points} design points using {n_processes} process(es)")

        if n_processes <= 1:
            # Sequential execution (original behavior)
            best_design_point, best_obj_val, best_design_variables = self._brute_force_sequential(constraints)
        else:
            # Parallel execution
            best_design_point, best_obj_val, best_design_variables = self._brute_force_parallel(
                constraints, n_processes
            )

        if best_design_point is None or best_obj_val == math.inf:
            logger.error("No valid solution found across all design points")
            return math.inf, False

        # Re-solve with the best design point to get the actual cvxpy variable objects
        # This ensures we have proper variable references for set_symbol_value
        logger.info(f"Re-solving with best design point: {best_design_point}")
        self.hw.circuit_model.tech_model.set_params_from_design_point(best_design_point)
        self.hw.circuit_model.tech_model.set_param_constant_constraints()
        prob = cp.Problem(
            cp.Minimize(self.hw.obj),
            constraints + [constr.constraint for constr in self.hw.circuit_model.tech_model.param_constant_constraints]
        )
        prob.solve(gp=True)

        # Apply the solution to the tech model using the actual variable objects
        for var in prob.variables():
            print(f"variable: {var.name()}, value: {var.value}")
            self.hw.circuit_model.tech_model.base_params.set_symbol_value(var, var.value)

        logger.info(f"time to run cvxpy optimization: {time.time()-start_time}")
        self.hw.circuit_model.tech_model.process_optimization_results()

        return best_obj_val / lower_bound, False

    def _brute_force_sequential(self, constraints):
        """
        Sequential brute force optimization (original implementation).

        Returns:
            (best_design_point, best_obj_val, best_design_variables_dict)
        """
        best_design_point = self.hw.circuit_model.tech_model.pareto_df.iloc[0].to_dict()
        best_obj_val = math.inf
        best_design_variables = None

        total_design_points = len(self.hw.circuit_model.tech_model.pareto_df)
        for i, row in enumerate(self.hw.circuit_model.tech_model.pareto_df.itertuples(index=False)):
            cur_design_point = row._asdict()
            logger.info(f"evaluating design point {i} of {total_design_points}: {cur_design_point}")
            self.hw.circuit_model.tech_model.set_params_from_design_point(cur_design_point)
            self.hw.circuit_model.tech_model.set_param_constant_constraints()
            for constraint in self.hw.circuit_model.tech_model.param_constant_constraints:
                logger.info(f"constraint: {constraint}")
            prob = cp.Problem(cp.Minimize(self.hw.obj), constraints + [constr.constraint for constr in self.hw.circuit_model.tech_model.param_constant_constraints])
            try:
                prob.solve(gp=True)
            except Exception as e:
                logger.error(f"error solving cvxpy problem: {e}")
                continue
            logger.info(f"problem value is: {prob.value}")
            logger.info(f"variables are: {prob.variables()}")
            logger.info(f"constraints are: {prob.constraints}")
            if prob.value is not None and prob.value < best_obj_val:
                logger.info(f"new best design point found: {cur_design_point} with value {prob.value}")
                best_design_point = cur_design_point
                best_obj_val = prob.value
                best_design_variables = {var.name(): var.value for var in prob.variables()}

        return best_design_point, best_obj_val, best_design_variables

    def _brute_force_parallel(self, constraints, n_processes):
        """
        Parallel brute force optimization using ProcessPoolExecutor.

        Args:
            constraints: List of cvxpy constraints.
            n_processes: Number of parallel processes.

        Returns:
            (best_design_point, best_obj_val, best_design_variables_dict)
        """
        total_design_points = len(self.hw.circuit_model.tech_model.pareto_df)

        # Create a pool of tech_model copies (one per worker)
        logger.info(f"Creating {n_processes} copies of tech_model for parallel execution...")
        logger.info(f"Tech model copies created successfully")

        # Prepare design points
        design_points = [
            row._asdict() for row in self.hw.circuit_model.tech_model.pareto_df.itertuples(index=False)
        ]

        # Track best result across all workers
        best_design_point = design_points[0] if design_points else None
        best_obj_val = math.inf
        best_design_variables = None
        completed_count = 0

        # Process in batches to avoid memory issues with large design spaces
        max_batch_size = 10000
        num_batches = math.ceil(total_design_points / max_batch_size)

        for batch_num in range(num_batches):
            batch_start = batch_num * max_batch_size
            batch_end = min((batch_num + 1) * max_batch_size, total_design_points)
            batch_design_points = design_points[batch_start:batch_end]

            logger.info(f"Starting batch {batch_num + 1}/{num_batches} ({len(batch_design_points)} design points)")

            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                # Create tasks: each task gets a design point and a tech_model from the pool
                tasks = [
                    (batch_start + local_idx, dp, self.hw.circuit_model.tech_model, self.hw.obj, constraints)
                    for local_idx, dp in enumerate(batch_design_points)
                ]

                future_to_idx = {
                    executor.submit(_worker_evaluate_design_point, task): task[0]
                    for task in tasks
                }

                # Process results as they complete
                for future in as_completed(future_to_idx):
                    idx, design_point, obj_val, variables_dict = future.result()
                    completed_count += 1

                    if completed_count % 100 == 0:
                        logger.info(f"Progress: {completed_count}/{total_design_points} design points completed")

                    if design_point is not None and obj_val is not None:
                        logger.info(f"Design point {idx}: objective value = {obj_val}")
                        if obj_val < best_obj_val:
                            logger.info(f"New best design point found: {design_point} with value {obj_val}")
                            best_design_point = design_point
                            best_obj_val = obj_val
                            best_design_variables = variables_dict
                    else:
                        logger.warning(f"Design point {idx} failed or returned invalid result")

            logger.info(f"Batch {batch_num + 1}/{num_batches} complete")

        logger.info(f"Parallel optimization complete. Best objective value: {best_obj_val}")
        return best_design_point, best_obj_val, best_design_variables
    

    def cvxpy_optimization(self, improvement, n_processes=100):
        if self.opt_pipeline == "fit":
            return self.fit_optimization(improvement)
        elif self.opt_pipeline == "brute_force":
            return self.brute_force_optimization(improvement, n_processes=n_processes)
        else:
            raise ValueError(f"Invalid optimization pipeline: {self.opt_pipeline}")
    # note: improvement/regularization parameter currently only for inverse pass validation, so only using it for ipopt
    # example: improvement of 1.1 = 10% improvement
    def optimize(self, opt, improvement=10, disabled_knobs=[]):
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
