# first party
import argparse
import logging
import time
import sys
import copy
import math
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

def log_info(msg, stage):
    if stage == "before optimization":
        print(msg)
    elif stage == "after optimization":
        logger.info(msg)


class Optimizer:
    def __init__(self, hw, tmp_dir, test_config=False, opt_pipeline="block_vector"):
        self.hw = hw
        self.disabled_knobs = []
        self.objective_constraint_inds = []
        self.initial_alpha = None
        self.test_config = test_config
        self.tmp_dir = tmp_dir
        self.opt_pipeline = opt_pipeline
        self.bbv_op_delay_constraints = []
        self.bbv_path_constraints = []
        self.max_system_power = None

    def initialize_max_system_power(self, power):
        logger.info(f"initializing max system power to {power*150}")
        #self.max_system_power = power * 150
        self.max_system_power = 3.0e-4 

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
        # ensure that forward pass can't add more than 10x parallelism in the next iteration. power scale is based on the amount we scale area down by,
        # because in the next forward pass we assume that much parallelism will be added, and therefore increase power
        #if not self.test_config:
        constraints.append(Constraint(self.hw.circuit_model.tech_model.capped_power_scale <= improvement, "capped_power_scale <= improvement"))

        assert len(self.hw.circuit_model.tech_model.constraints) > 0, "tech model constraints are empty"
        constraints.extend(self.hw.circuit_model.tech_model.base_params.constraints)
        constraints.extend(self.hw.circuit_model.tech_model.constraints)
        constraints.extend(self.bbv_op_delay_constraints)
        constraints.extend(self.bbv_path_constraints)

        if not approx_problem:
            constraints.extend(self.hw.circuit_model.constraints)
            constraints.extend(self.hw.constraints)

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
        self.hw.calculate_passive_power_vitis(execution_time)
        self.hw.save_obj_vals(execution_time)
        if self.opt_pipeline == "block_vector":
            self.hw.calculate_sensitivity_analysis(blackbox=True)
        print(f"obj: {self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)}, obj scaled: {self.hw.obj_scaled.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)}")
        lower_bound = sim_util.xreplace_safe(self.hw.obj_scaled, self.hw.circuit_model.tech_model.base_params.tech_values) / improvement
        self.constraints = self.create_constraints(improvement, lower_bound, approx_problem=(self.opt_pipeline != "block_vector"))
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
                    
                    # if bbv optimization, clk period updated in solver, so need to update it here
                    if self.opt_pipeline == "block_vector":
                        print(f"value of clk period: {self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.clk_period]}")
                        self.hw.calculate_objective(clk_period_opt=False, form_dfg=False)
                    else:
                        self.hw.calculate_objective(clk_period_opt=True, form_dfg=False)
                        print(f"setting clk period to {self.hw.circuit_model.clk_period_cvx.value}")
                        self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.clk_period] = self.hw.circuit_model.clk_period_cvx.value

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
        ahmdal_limit = sim_util.xreplace_safe(getattr(self.hw.circuit_model.tech_model.base_params, op_type + "_ahmdal_limit"), self.hw.circuit_model.tech_model.base_params.tech_values)
        if ahmdal_limit == math.inf:
            print(f"ahmdal limit is infinite for {op_type}, skipping constraint")
            return []
        if ahmdal_limit < improvement: # skip because we will eventually need to optimize a path with this op type anyways
            print(f"ahmdal limit is less than improvement for {op_type}, skipping constraint")
            return []
        op_delay_ratio = op_delay / sim_util.xreplace_safe(op_delay, self.hw.circuit_model.tech_model.base_params.tech_values)
        delay_contrib = ((1/ahmdal_limit) * op_delay) * op_delay_ratio
        self.hw.save_obj_vals(self.hw.execution_time, execution_time_override=True, execution_time_override_val=delay_contrib)
        obj_scaled_op = self.hw.obj_scaled
        obj_scaled_op_init = sim_util.xreplace_safe(obj_scaled_op, self.hw.circuit_model.tech_model.base_params.tech_values)
        # this constraint should always start out as feasible because ahmdal limit >= improvement
        constr = obj_scaled_op <= obj_scaled_op_init * (ahmdal_limit/improvement)

        # reset hw model state
        self.hw.save_obj_vals(self.hw.execution_time)
        assert self.hw.obj_scaled != obj_scaled_op, "obj scaled should not be the same as the original obj scaled"

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
            self.hw.calculate_objective(clk_period_opt=False, form_dfg=False)
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
        self.hw.calculate_objective(clk_period_opt=False, form_dfg=False)
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
        assert opt == "ipopt"
        return self.ipopt(improvement)


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
