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

    def create_constraints(self, improvement):
        constraints = []
        constraints.append(self.hw.symbolic_obj >= float(self.hw.symbolic_obj.subs(self.hw.params.tech_values) / improvement))
        constraints.append(self.hw.params.V_dd >= self.hw.params.V_th)
        constraints.append(self.hw.params.V_dd <= 5)
        return constraints

    def ipopt(self, improvement, regularization):
        """
        Run the IPOPT optimization routine for the hardware model using Pyomo.

        Args:
            tech_params (dict): Technology parameters for optimization.
            edp (sympy.Expr): Symbolic EDP Expression.
            improvement (float): Improvement factor for optimization.
            regularization (float): Regularization parameter for inverse pass validation.
            cacti_subs (dict): Substitution dictionary for CACTI parameters.

        Returns:
            None
        """
        logger.info("Optimizing using IPOPT")

        constraints = self.create_constraints(improvement)

        model = pyo.ConcreteModel()
        opt, scaled_model, model = (
            Preprocessor(self.hw.params).begin(model, self.hw.symbolic_obj, improvement, multistart=multistart, regularization=regularization, constraints=constraints)
        )


        start_time = time.time()
        if multistart:
            results = opt.solve(
                scaled_model,
                solver_args={
                    "keepfiles": True,
                    "tee": True,
                    "symbolic_solver_labels": True,
                },
            )
        else:
            results = opt.solve(
                scaled_model, keepfiles=True, tee=True, symbolic_solver_labels=True
            )
        logger.info(f"time to run IPOPT: {time.time()-start_time}")
        pyo.TransformationFactory("core.scale_model").propagate_solution(
            scaled_model, model
        )

        print(results.solver.termination_condition)
        print("======================")
        model.display()

    # note: improvement/regularization parameter currently only for inverse pass validation, so only using it for ipopt
    # example: improvement of 1.1 = 10% improvement
    def optimize(self, opt, improvement=10, regularization=0.1):
        """
        Optimize the hardware model using the specified optimization method.

        Args:

        Returns:
            None
        """
        assert opt == "ipopt"
        return self.ipopt(improvement, regularization)


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
