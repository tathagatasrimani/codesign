# first party
import argparse
import logging
import time
import sys
logger = logging.getLogger(__name__)

# third party
import pyomo.environ as pyo
import sympy as sp
# custom
from src.inverse_pass.preprocess import Preprocessor

from src import sim_util


multistart = False

def log_info(msg, stage):
    if stage == "before optimization":
        print(msg)
    elif stage == "after optimization":
        logger.info(msg)


class Optimizer:
    def __init__(self, hw):
        self.hw = hw
        self.disabled_knobs = []
        self.objective_constraint_inds = []
        self.initial_alpha = None

    def evaluate_constraints(self, constraints, stage):
        for constraint in constraints:
            value = "N/A"
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
        constraints.append(self.hw.total_active_energy >= 2*self.hw.total_passive_energy)
        for knob in self.disabled_knobs:
            constraints.append(sp.Eq(knob, knob.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)))
        total_power = (self.hw.total_passive_energy + self.hw.total_active_energy) / self.hw.execution_time
        constraints.append(total_power <= 1e-5) # hard limit on power
        # ensure that forward pass can't add more than 10x parallelism in the next iteration. power scale is based on the amount we scale area down by,
        # because in the next forward pass we assume that much parallelism will be added, and therefore increase power
        constraints.append(self.hw.circuit_model.tech_model.power_scale_current_iter <= 10)

        assert len(self.hw.circuit_model.tech_model.constraints) > 0, "tech model constraints are empty"
        constraints.extend(self.hw.circuit_model.tech_model.base_params.constraints)
        constraints.extend(self.hw.circuit_model.tech_model.constraints)
        if not approx_problem:
            constraints.extend(self.hw.circuit_model.constraints)
            constraints.extend(self.hw.constraints)

        self.evaluate_constraints(constraints, "before optimization")

        #print(f"constraints: {constraints}")
        return constraints
    
    def create_opt_model(self, improvement, lower_bound):
        constraints = self.create_constraints(improvement, lower_bound)
        model = pyo.ConcreteModel()
        self.preprocessor = Preprocessor(self.hw.circuit_model.tech_model.base_params, out_file="src/tmp/solver_out.txt")
        opt, scaled_model, model, multistart_options = (
            self.preprocessor.begin(model, self.hw.obj_scaled, improvement, multistart=multistart, constraints=constraints)
        )
        return opt, scaled_model, model, multistart_options
    
    # can be used as a starting point for the optimizer
    def generate_approximate_solution(self, improvement):
        lower_bound = float(self.hw.circuit_model.tech_model.delay.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values) / improvement)
        self.constraints = self.create_constraints(improvement, lower_bound, approx_problem=True)
        execution_time = self.hw.circuit_model.tech_model.delay_var
        if self.hw.obj_fn == "edp":
            self.hw.obj = (self.hw.total_passive_energy + self.hw.total_active_energy) * execution_time
            self.hw.obj_scaled = (self.hw.total_passive_energy * self.hw.circuit_model.tech_model.capped_power_scale + self.hw.total_active_energy) * execution_time * self.hw.circuit_model.tech_model.capped_delay_scale
        elif self.hw.obj_fn == "ed2":
            self.hw.obj = (self.hw.total_passive_energy + self.hw.total_active_energy) * (execution_time)**2
            self.hw.obj_scaled = (self.hw.total_passive_energy * self.hw.circuit_model.tech_model.capped_power_scale + self.hw.total_active_energy) * (execution_time * self.hw.circuit_model.tech_model.capped_delay_scale)**2
        elif self.hw.obj_fn == "delay":
            self.hw.obj = execution_time
            self.hw.obj_scaled = execution_time * self.hw.circuit_model.tech_model.capped_delay_scale
        elif self.hw.obj_fn == "energy":
            self.hw.obj = self.hw.total_active_energy + self.hw.total_passive_energy
            self.hw.obj_scaled = (self.hw.total_active_energy + self.hw.total_passive_energy * self.hw.circuit_model.tech_model.capped_power_scale)
        else:
            raise ValueError(f"Objective function {self.hw.obj_fn} not supported")
        model = pyo.ConcreteModel()
        self.approx_preprocessor = Preprocessor(self.hw.circuit_model.tech_model.base_params, out_file="src/tmp/solver_out_approx.txt")
        opt, scaled_model, model, multistart_options = (
            self.approx_preprocessor.begin(model, self.hw.obj_scaled, improvement, multistart=False, constraints=self.constraints)
        )
        return opt, scaled_model, model, multistart_options

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
        while num_solver_iterations < 10:
            print("running approximate solver")
            stdout = sys.stdout
            try:
                with open("src/tmp/ipopt_out_approx.txt", "w") as f:
                    sys.stdout = f
                    opt_approx, scaled_model_approx, model_approx, multistart_options_approx = self.generate_approximate_solution(improvement)
                    results = opt_approx.solve(scaled_model_approx, symbolic_solver_labels=True)
                    if results.solver.termination_condition == "optimal":
                        print("approximate solver found optimal solution, will update starting point")
                    pyo.TransformationFactory("core.scale_model").propagate_solution(
                        scaled_model_approx, model_approx
                    )
                    model_approx.display()
            finally:
                sys.stdout = stdout
            f = open("src/tmp/ipopt_out_approx.txt", "r")
            sim_util.parse_output(f, self.hw)
            self.hw.calculate_objective()
            print(f"tech params after approximate solver: {self.hw.circuit_model.tech_model.base_params.tech_values}")

            print("running regular solver")
            opt, scaled_model, model, multistart_options = self.create_opt_model(improvement, lower_bound)
            Error = False
            try:
                # Pass multistart options if available
                if multistart_options:
                    print(f"Running multistart solver in iteration {num_solver_iterations}")
                    # For multistart solver, only pass multistart-specific options
                    results = opt.solve(scaled_model, **multistart_options)
                else:
                    print(f"Running regular solver in iteration {num_solver_iterations}")
                    # For regular solver, pass Pyomo-specific options
                    results = opt.solve(
                        scaled_model, symbolic_solver_labels=True
                    )
                            
                    #print("original ipopt failed, running multistart solver")
                    #self.preprocessor.multistart = True
                    #opt = self.preprocessor.get_solver()
                    #multistart_options = self.preprocessor.multistart_options
                    #opt, scaled_model, model, multistart_options = self.generate_approximate_solution()
                    #print(f"Running multistart solver in iteration {num_solver_iterations}")
                    #results = opt.solve(scaled_model, **multistart_options)

                pyo.TransformationFactory("core.scale_model").propagate_solution(
                    scaled_model, model
                )
            except Exception as e:
                print(f"Error: {e}")
                Error = True
            if results.solver.termination_condition == "optimal":
                print("solver found optimal solution")
                break
            else:
                num_solver_iterations += 1
        logger.info(f"time to run IPOPT: {time.time()-start_time}")

        if not Error:
            self.hw.circuit_model.tech_model.base_params.tech_values = original_tech_values
            print(results.solver.termination_condition)
            print("======================")
        model.display()
        lag_factor = 1
        for ind in self.objective_constraint_inds:
            lag_factor *= pyo.value(model.Constraints[ind]) / pyo.value(model.Constraints[ind].lower)
        # guard against bad max iterations failures
        if lag_factor > improvement * len(self.objective_constraint_inds):
            Error = True
            lag_factor = improvement * len(self.objective_constraint_inds)
        print(f"lag factor: {lag_factor}")
        return lag_factor, Error

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
