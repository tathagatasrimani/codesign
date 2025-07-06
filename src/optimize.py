# first party
import argparse
import logging
import time
logger = logging.getLogger(__name__)

# third party
import pyomo.environ as pyo
import sympy as sp
# custom
from .preprocess import Preprocessor


multistart = False

class Optimizer:
    def __init__(self, hw):
        self.hw = hw
        self.disabled_knobs = []
        self.dennard_scaling_type = "constant_field"
        self.objective_constraint_inds = []

    def create_constraints(self, improvement):
        constraints = []
        self.objective_constraint_inds = [0]
        constraints.append(self.hw.symbolic_obj >= float(self.hw.symbolic_obj.subs(self.hw.params.tech_values) / improvement))
        constraints.append(self.hw.params.V_dd >= self.hw.params.V_th_eff)
        if self.hw.params.V_th_eff != self.hw.params.V_th:
            constraints.append(self.hw.params.V_th_eff >= 0)
        constraints.append(self.hw.params.V_dd <= 5)
        constraints.append(self.hw.params.V_ox >= 0)
        for knob in self.disabled_knobs:
            constraints.append(sp.Eq(knob, self.hw.params.tech_values[knob]))
        total_power = (self.hw.total_passive_energy + self.hw.total_active_energy) / self.hw.execution_time
        constraints.append(total_power <= 50) # hard limit on power
        if self.hw.params.f in self.hw.params.tech_values:
            constraints.append(self.hw.params.delay <= 1e9/self.hw.params.f)
        constraints.append(sp.Eq(self.hw.params.t_ox_, self.hw.params.e_ox/self.hw.params.Cox))
        constraints.append(self.hw.params.I_off/(self.hw.params.W*self.hw.params.L) <= 100e-9 / (1e-6 * 1e-6))

        if self.hw.model_cfg["scaling_mode"] == "dennard":
            if self.dennard_scaling_type == "constant_field":
                constraints.append(sp.Eq(self.hw.params.W/self.hw.params.tech_values[self.hw.params.W], 1/self.hw.params.alpha_dennard))
                constraints.append(sp.Eq(self.hw.params.L/self.hw.params.tech_values[self.hw.params.L], 1/self.hw.params.alpha_dennard))
                constraints.append(sp.Eq(self.hw.params.V_dd/self.hw.params.tech_values[self.hw.params.V_dd], 1/self.hw.params.alpha_dennard))
                constraints.append(sp.Eq(self.hw.params.V_th_eff/self.hw.params.tech_values[self.hw.params.V_th_eff], 1/self.hw.params.alpha_dennard))
                constraints.append(sp.Eq(self.hw.params.Cox/self.hw.params.tech_values[self.hw.params.Cox], self.hw.params.alpha_dennard))
            else:
                constraints.append(sp.Eq(self.hw.params.alpha_dennard, self.hw.params.tech_values[self.hw.params.alpha_dennard]))
                constraints.append(sp.Eq(self.hw.params.W/self.hw.params.tech_values[self.hw.params.W], 1/self.hw.params.alpha_dennard))
                constraints.append(sp.Eq(self.hw.params.L/self.hw.params.tech_values[self.hw.params.L], 1/self.hw.params.alpha_dennard))
                constraints.append(sp.Eq(self.hw.params.V_dd, self.hw.params.tech_values[self.hw.params.V_dd]))
                constraints.append(sp.Eq(self.hw.params.V_th_eff, self.hw.params.tech_values[self.hw.params.V_th_eff]))
                #constraints.append(sp.Eq(self.hw.params.V_dd/self.hw.params.tech_values[self.hw.params.V_dd], self.hw.params.epsilon_dennard/self.hw.params.alpha_dennard))
                #constraints.append(sp.Eq(self.hw.params.V_th_eff/self.hw.params.tech_values[self.hw.params.V_th_eff], self.hw.params.epsilon_dennard/self.hw.params.alpha_dennard))
                constraints.append(sp.Eq(self.hw.params.Cox/self.hw.params.tech_values[self.hw.params.Cox], self.hw.params.alpha_dennard))

            #elif self.dennard_scaling_type == "generalized":
            #    constraints.append(sp.Eq(self.hw.params.alpha_dennard, 1))
        elif self.hw.model_cfg["scaling_mode"] == "dennard_implicit":
            constraints.append(self.hw.params.t_ox_ <= self.hw.params.t_ox_.subs(self.hw.params.tech_values))
            constraints.append(self.hw.params.L <= self.hw.params.L.subs(self.hw.params.tech_values))
            constraints.append(self.hw.params.W <= self.hw.params.W.subs(self.hw.params.tech_values))
            constraints.append(self.hw.params.V_dd <= self.hw.params.V_dd.subs(self.hw.params.tech_values))
            constraints.append(self.hw.params.V_th_eff <= self.hw.params.V_th_eff.subs(self.hw.params.tech_values))
            constraints.append(sp.Eq(self.hw.params.W/self.hw.params.W.subs(self.hw.params.tech_values), self.hw.params.L/self.hw.params.L.subs(self.hw.params.tech_values)))
            constraints.append(sp.Eq(self.hw.params.V_dd/self.hw.params.V_dd.subs(self.hw.params.tech_values) , self.hw.params.L/self.hw.params.L.subs(self.hw.params.tech_values))) # lateral electric field scaling
            constraints.append(sp.Eq(self.hw.params.V_dd/self.hw.params.V_dd.subs(self.hw.params.tech_values) , self.hw.params.t_ox_/self.hw.params.t_ox_.subs(self.hw.params.tech_values))) # lateral electric field scaling
            constraints[0] = self.hw.execution_time >= self.hw.execution_time.subs(self.hw.params.tech_values)/1.3 # replace objective constraint with latency constraint
            constraints.append(self.hw.total_active_energy + self.hw.total_passive_energy >= (self.hw.total_active_energy + self.hw.total_passive_energy).subs(self.hw.params.tech_values)/2.7)
            self.objective_constraint_inds.append(len(constraints)-1)

        print(f"constraints: {constraints}")
        #constraints.append(self.hw.params.L >= 15e-9)
        #constraints.append(self.hw.params.W >= 15e-9)
        return constraints
    
    def create_opt_model(self, improvement):
        constraints = self.create_constraints(improvement)
        model = pyo.ConcreteModel()
        preprocessor = Preprocessor(self.hw.params)
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

        param_replace = {param: sp.Abs(param, evaluate=False) for param in self.hw.params.tech_values}
        print("param_replace: ", param_replace)
        print("symbolic obj before abs: ", self.hw.symbolic_obj)
        self.hw.symbolic_obj = self.hw.symbolic_obj.xreplace(param_replace)
        print("symbolic obj after abs: ", self.hw.symbolic_obj)

        constraints = self.create_constraints(improvement)

        opt, scaled_model, model = self.create_opt_model(improvement)


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
                results = opt.solve(
                    scaled_model, keepfiles=True, tee=True, symbolic_solver_labels=True
                )
                pyo.TransformationFactory("core.scale_model").propagate_solution(
                    scaled_model, model
                )
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
        else:
            results = opt.solve(
                scaled_model, keepfiles=True, tee=True, symbolic_solver_labels=True
            )
            pyo.TransformationFactory("core.scale_model").propagate_solution(
                scaled_model, model
            )
        logger.info(f"time to run IPOPT: {time.time()-start_time}")

        print(results.solver.termination_condition)
        print("======================")
        model.display()
        lag_factor = 1
        for ind in self.objective_constraint_inds:
            lag_factor *= pyo.value(model.Constraints[ind]) / pyo.value(model.Constraints[ind].lower)
        print(f"lag factor: {lag_factor}")
        return lag_factor

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
