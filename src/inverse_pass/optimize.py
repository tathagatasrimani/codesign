# first party
import argparse
import logging
import time
logger = logging.getLogger(__name__)

# third party
import pyomo.environ as pyo
import sympy as sp
# custom
from src.inverse_pass.preprocess import Preprocessor


multistart = False

class Optimizer:
    def __init__(self, hw):
        self.hw = hw
        self.disabled_knobs = []
        self.objective_constraint_inds = []
        self.initial_alpha = None

    def create_constraints(self, improvement):
        # system level and objective constraints, and pull in tech model constraints

        constraints = []
        lower_bound = float(self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values) / improvement)
        constraints.append(self.hw.obj >= lower_bound)
        self.objective_constraint_inds = [len(constraints)-1]

        # don't want a leakage-dominated design
        constraints.append(self.hw.total_active_energy >= 2*self.hw.total_passive_energy)
        for knob in self.disabled_knobs:
            constraints.append(sp.Eq(knob, knob.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)))
        total_power = (self.hw.total_passive_energy + self.hw.total_active_energy) / self.hw.execution_time
        constraints.append(total_power <= 150) # hard limit on power

        assert len(self.hw.circuit_model.tech_model.constraints) > 0, "tech model constraints are empty"
        constraints.extend(self.hw.circuit_model.tech_model.base_params.constraints)
        constraints.extend(self.hw.circuit_model.tech_model.constraints)
        constraints.extend(self.hw.circuit_model.constraints)

        #print(f"constraints: {constraints}")
        return constraints
    
    def create_opt_model(self, improvement):
        constraints = self.create_constraints(improvement)
        model = pyo.ConcreteModel()
        preprocessor = Preprocessor(self.hw.circuit_model.tech_model.base_params)
        opt, scaled_model, model = (
            preprocessor.begin(model, self.hw.obj_scaled, improvement, multistart=multistart, constraints=constraints)
        )
        return opt, scaled_model, model

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

        opt, scaled_model, model = self.create_opt_model(improvement)

        Error = False

        start_time = time.time()
        try:
            results = opt.solve(
                scaled_model, keepfiles=True, tee=True, symbolic_solver_labels=True
            )
            pyo.TransformationFactory("core.scale_model").propagate_solution(
                scaled_model, model
            )
        except Exception as e:
            print(f"Error: {e}")
            Error = True
        logger.info(f"time to run IPOPT: {time.time()-start_time}")

        if not Error:
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
