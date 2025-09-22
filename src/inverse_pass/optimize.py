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
        self.dennard_scaling_type = "constant_field"
        self.objective_constraint_inds = []
        self.initial_alpha = None

    def create_constraints(self, improvement):
        # system level and objective constraints, and pull in tech model constraints

        constraints = []
        #constraints.append(1/self.hw.circuit_model.tech_model.S <= 1e5)
        #constraints.append(1/self.hw.circuit_model.tech_model.V_ox <= 1e3)
        #constraints.append(1/self.hw.circuit_model.tech_model.n <= 1e30)
        #constraints.append(1/self.hw.circuit_model.tech_model.Q_ix0.xreplace(self.hw.circuit_model.tech_model.off_state) <= 1e30)
        #constraints.append(1/self.hw.circuit_model.tech_model.v.xreplace(self.hw.circuit_model.tech_model.off_state) <= 1e30)
        #constraints.append(1/self.hw.circuit_model.tech_model.F_s.xreplace(self.hw.circuit_model.tech_model.off_state) <= 1e30)
        #constraints.append(1/self.hw.circuit_model.tech_model.I_sub <= 1e30)
        #constraints.append(1/self.hw.circuit_model.tech_model.I_tunnel >= 1e-10)
        #constraints.append(1/self.hw.circuit_model.tech_model.I_d_on <= 1e10)
        lower_bound = float(self.hw.symbolic_obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values) / improvement)
        constraints.append(self.hw.symbolic_obj >= lower_bound)
        self.objective_constraint_inds = [len(constraints)-1]

        # don't want a leakage-dominated design
        constraints.append(self.hw.total_active_energy >= 2*self.hw.total_passive_energy)
        for knob in self.disabled_knobs:
            constraints.append(sp.Eq(knob, knob.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)))
        total_power = (self.hw.total_passive_energy + self.hw.total_active_energy) / self.hw.execution_time
        constraints.append(total_power <= 150) # hard limit on power

        if self.hw.model_cfg["scaling_mode"] == "dennard_implicit":
            constraints[0] = self.hw.execution_time >= self.hw.execution_time.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)/1.3 # replace objective constraint with latency constraint
            constraints.append(self.hw.total_active_energy + self.hw.total_passive_energy >= (self.hw.total_active_energy + self.hw.total_passive_energy).xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)/2.7)
            self.objective_constraint_inds.append(len(constraints)-1)

        #self.hw.circuit_model.tech_model.create_constraints(self.dennard_scaling_type)
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
            preprocessor.begin(model, self.hw.symbolic_obj, improvement, multistart=multistart, constraints=constraints)
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

        param_replace = {param: sp.Abs(param, evaluate=False) for param in self.hw.circuit_model.tech_model.base_params.tech_values}
        #print("param_replace: ", param_replace)
        #print("symbolic obj before abs: ", self.hw.symbolic_obj)
        self.hw.symbolic_obj = self.hw.symbolic_obj.xreplace(param_replace)
        #print("symbolic obj after abs: ", self.hw.symbolic_obj)

        opt, scaled_model, model = self.create_opt_model(improvement)

        Error = False

        start_time = time.time()
        if self.hw.model_cfg["scaling_mode"].startswith("dennard"):
            results = opt.solve(
                scaled_model, keepfiles=True, tee=True, symbolic_solver_labels=True
            )
            pyo.TransformationFactory("core.scale_model").propagate_solution(
                scaled_model, model
            )
            final_value = pyo.value(model.Constraints[0]) # assume that the first constraint is the objective
            print(f"obj value: {final_value}")
            lower_bound = pyo.value(model.Constraints[0].lower)
            print(f"lower bound: {lower_bound}")
            if final_value > lower_bound*1.1: # if the objective is not improving within some margin, run with constant voltage scaling
                print(f"obj value: {final_value} > {lower_bound*1.1}")
                print(f"running with constant voltage scaling")
                self.dennard_scaling_type = "constant_voltage"
                opt, scaled_model, model = self.create_opt_model(improvement)
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
                """
                print(f"obj value: {final_value}")
                print(f"lower bound: {lower_bound}")
                if final_value > lower_bound*1.1: # if the objective is not improving within some margin, run with general scaling
                    print(f"obj value: {final_value} > {lower_bound*1.1}")
                    self.dennard_scaling_type = "generalized"
                    opt, scaled_model, model = self.create_opt_model(improvement)
                    results = opt.solve(
                        scaled_model, keepfiles=True, tee=True, symbolic_solver_labels=True
                    )
                    pyo.TransformationFactory("core.scale_model").propagate_solution(
                        scaled_model, model
                    )"""
        elif multistart:
            try:
                results = opt.solve(
                    scaled_model,
                    solver_args={
                        "keepfiles": True,
                        "tee": True,
                        "symbolic_solver_labels": True,
                    },
                )
                pyo.TransformationFactory("core.scale_model").propagate_solution(
                    scaled_model, model
                )
            except Exception as e:
                print(f"Error: {e}")
                Error = True
        else:
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
    """
    TODO. Add an entrypoint that is not main.
    For my inverse pass, I don't want to call this 
    """

    """hw = HardwareModel(cfg=args.architecture_config)
    hw.init_memory(
        sim_util.find_nearest_power_2(131072),
        sim_util.find_nearest_power_2(0),
    )

    rcs = hw.get_optimization_params_from_tech_params()
    logger.info(f"optimize.__main__.rcs: {rcs}")
    initial_params = sim_util.generate_init_params_from_rcs_as_symbols(rcs)
    edp = open("src/tmp/symbolic_edp.txt", "r")
    edp = sympify(edp.readline(), locals=hw_symbols.symbol_table)

    results = optimize(initial_params, edp, args.opt)

    return results"""
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
