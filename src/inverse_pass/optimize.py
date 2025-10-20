# first party
import argparse
import logging
import time
import sys
import copy
logger = logging.getLogger(__name__)

# third party
import pyomo.environ as pyo
import sympy as sp
import cvxpy as cp
import numpy as np
# custom
from src.inverse_pass.preprocess import Preprocessor
from src.inverse_pass import curve_fit

from src import sim_util


multistart = False

def log_info(msg, stage):
    if stage == "before optimization":
        print(msg)
    elif stage == "after optimization":
        logger.info(msg)


class Optimizer:
    def __init__(self, hw, tmp_dir, test_config=False):
        self.hw = hw
        self.disabled_knobs = []
        self.objective_constraint_inds = []
        self.initial_alpha = None
        self.test_config = test_config
        self.tmp_dir = tmp_dir

    def evaluate_constraints(self, constraints, stage):
        for constraint in constraints:
            value = 0
            log_info(f"constraint: {constraint}", stage)
            if isinstance(constraint, sp.Ge):
                value = sim_util.xreplace_safe(constraint.rhs - constraint.lhs, self.hw.circuit_model.tech_model.base_params.tech_values)
            elif isinstance(constraint, sp.Le):
                value = sim_util.xreplace_safe(constraint.lhs - constraint.rhs, self.hw.circuit_model.tech_model.base_params.tech_values)
            log_info(f"constraint value: {value}", stage)
            tol = 1e-3
            if value > tol:
                log_info(f"CONSTRAINT VIOLATED {stage}", stage)

    def create_constraints(self, improvement, lower_bound, approx_problem=False):
        # system level and objective constraints, and pull in tech model constraints

        constraints = []
        constraints.append(self.hw.obj >= lower_bound)
        self.objective_constraint_inds = [len(constraints)-1]

        # don't want a leakage-dominated design
        #constraints.append(self.hw.total_active_energy >= 2*self.hw.total_passive_energy*self.hw.circuit_model.tech_model.capped_power_scale)
        for knob in self.disabled_knobs:
            constraints.append(sp.Eq(knob, knob.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)))
        if not self.test_config:
            total_power = self.hw.total_passive_power*self.hw.circuit_model.tech_model.capped_power_scale + self.hw.total_active_energy / (self.hw.execution_time* self.hw.circuit_model.tech_model.capped_delay_scale)
        else:
            total_power = self.hw.total_passive_power*self.hw.circuit_model.tech_model.capped_power_scale_total + self.hw.total_active_energy / (self.hw.execution_time* self.hw.circuit_model.tech_model.capped_delay_scale_total)
        constraints.append(total_power <= 1e-2) # hard limit on power
        # ensure that forward pass can't add more than 10x parallelism in the next iteration. power scale is based on the amount we scale area down by,
        # because in the next forward pass we assume that much parallelism will be added, and therefore increase power
        #if not self.test_config:
        constraints.append(self.hw.circuit_model.tech_model.capped_power_scale <= improvement)

        assert len(self.hw.circuit_model.tech_model.constraints) > 0, "tech model constraints are empty"
        constraints.extend(self.hw.circuit_model.tech_model.base_params.constraints)
        constraints.extend(self.hw.circuit_model.tech_model.constraints)
        if not approx_problem:
            constraints.extend(self.hw.circuit_model.constraints)
            constraints.extend(self.hw.constraints)

        #self.evaluate_constraints(constraints, "before optimization")

        #print(f"constraints: {constraints}")
        return constraints
    
    def create_opt_model(self, improvement, lower_bound):
        constraints = self.create_constraints(improvement, lower_bound)
        model = pyo.ConcreteModel()
        self.preprocessor = Preprocessor(self.hw.circuit_model.tech_model.base_params, out_file=f"{self.tmp_dir}/solver_out.txt")
        opt, scaled_model, model, multistart_options = (
            self.preprocessor.begin(model, self.hw.obj_scaled, improvement, multistart=multistart, constraints=constraints)
        )
        return opt, scaled_model, model, multistart_options
    
    # can be used as a starting point for the optimizer
    def generate_approximate_solution(self, improvement, delay_factor, iteration, multistart=False):
        execution_time = self.hw.circuit_model.tech_model.delay * (sim_util.xreplace_safe(self.hw.execution_time, self.hw.circuit_model.tech_model.base_params.tech_values)/sim_util.xreplace_safe(self.hw.circuit_model.tech_model.delay, self.hw.circuit_model.tech_model.base_params.tech_values))
        print(f"execution time: {execution_time.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)}")
        print(f"delay factor: {delay_factor}")

        # passive energy consumption is dependent on execution time, so we need to recalculate it
        self.hw.calculate_passive_power_vitis(execution_time)
        self.hw.save_obj_vals(execution_time, execution_time_override=True, execution_time_override_val=self.hw.circuit_model.tech_model.delay)
        print(f"obj: {self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)}, obj scaled: {self.hw.obj_scaled.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)}")
        lower_bound = sim_util.xreplace_safe(self.hw.obj_scaled, self.hw.circuit_model.tech_model.base_params.tech_values) / improvement
        self.constraints = self.create_constraints(improvement, lower_bound, approx_problem=True)
        model = pyo.ConcreteModel()
        self.approx_preprocessor = Preprocessor(self.hw.circuit_model.tech_model.base_params, out_file=f"{self.tmp_dir}/solver_out_approx_{iteration}.txt", solver_name="ipopt")
        opt, scaled_model, model, multistart_options = (
            self.approx_preprocessor.begin(model, self.hw.obj_scaled, improvement, multistart=multistart, constraints=self.constraints)
        )

        if self.test_config:
            self.hw.execution_time = execution_time

        return opt, scaled_model, model, multistart_options

    # sets of delay, energy, and leakage power values are provided, fit a pareto front to them
    def fit_ed_pareto_front(self, delay_vals, energy_vals, p_leakage_vals):
        c, a = curve_fit.fit_ed_curve(delay_vals, energy_vals) # c, a for y=c*x^a
        
        return c, a

    def generate_design_points(self, count, improvement):
        #delay_factors = np.linspace(0.9, 1.1, count)
        delay_factors = [1.0]
        tech_param_sets = []
        obj_vals = []
        scaled_obj_vals = []
        original_tech_values = copy.deepcopy(self.hw.circuit_model.tech_model.base_params.tech_values)
        for i in range(count):
            stdout = sys.stdout
            Error = False
            with open(f"{self.tmp_dir}/ipopt_out_approx_{i}.txt", "w") as f:
                sys.stdout = f
                opt_approx, scaled_model_approx, model_approx, multistart_options_approx = self.generate_approximate_solution(improvement, delay_factors[i], i)
                try:
                    results = opt_approx.solve(scaled_model_approx, symbolic_solver_labels=True)
                except Exception as e:
                    print(f"Error: {e}")
                    Error = True
                # just let "infeasible" solutions through for now, often they are not violating any constraints
                if (Error or results.solver.termination_condition not in ["optimal", "acceptable", "infeasible", "maxIterations"]) and delay_factors[i] == 1.0:
                    print(f"First solve attempt failed, trying again...")
                    #raise Exception("First solve attempt failed")
                    Error = False
                    opt_approx, scaled_model_approx, model_approx, multistart_options_approx = self.generate_approximate_solution(improvement, delay_factors[i], i, multistart=True)
                    # Try with more relaxed tolerances
                    #opt_approx.options["constr_viol_tol"] = 1e-4
                    #opt_approx.options["acceptable_constr_viol_tol"] = 1e-2
                    #opt_approx.options["acceptable_tol"] = 1e-4
                    try:
                        results = opt_approx.solve(scaled_model_approx, **multistart_options_approx)
                    except Exception as e:
                        print(f"Error: {e}")
                        Error = True
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
                f = open(f"{self.tmp_dir}/ipopt_out_approx_{i}.txt", "r")
                sim_util.parse_output(f, self.hw)
                print(f"scaled objective used in approximation is now: {self.hw.obj_scaled.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)}")
                print(f"objective used in approximation is now: {self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)}")
                if not self.test_config:
                    self.hw.circuit_model.update_circuit_values()
                    self.hw.calculate_objective(clk_period_opt=True, form_dfg=False)
                    print(f"setting clk period to {self.hw.circuit_model.clk_period_cvx.value}")
                    self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.clk_period] = self.hw.circuit_model.clk_period_cvx.value
                scaled_obj_vals.append(self.hw.obj_scaled.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values))
                obj_vals.append(self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values))
                self.hw.display_objective("after approximate solver")
                tech_param_sets.append(self.hw.circuit_model.tech_model.base_params.tech_values)
                self.hw.circuit_model.tech_model.base_params.tech_values = copy.deepcopy(original_tech_values)
        return tech_param_sets, obj_vals, scaled_obj_vals


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

        lower_bound = float(self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values) / improvement)
        original_tech_values = self.hw.circuit_model.tech_model.base_params.tech_values.copy()

        start_time = time.time()
        num_solver_iterations = 0
        tech_param_sets, obj_vals, scaled_obj_vals = self.generate_design_points(1, improvement)
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

        logger.info(f"time to run optimization: {time.time()-start_time}")

        lag_factor = obj_vals[optimal_design_idx] / lower_bound
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
