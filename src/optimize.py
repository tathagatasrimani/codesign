# first party
import yaml
import argparse

# third party
import pyomo.environ as pyo
from sympy import sympify
import sympy
import cvxpy as cp
import numpy as np

# custom
from preprocess import Preprocessor
from sim_util import generate_init_params_from_rcs_as_symbols
from hardwareModel import HardwareModel
import hw_symbols


multistart = False


def ipopt(tech_params, edp):
    initial_params = {}
    for key in tech_params:
        initial_params[key.name] = tech_params[key]

    model = pyo.ConcreteModel()
    opt, scaled_preproc_model, preproc_model, free_symbols, mapping = (
        Preprocessor().begin(model, edp, initial_params, multistart=multistart)
    )

    if multistart:
        results = opt.solve(
            scaled_preproc_model,
            solver_args={
                "keepfiles": True,
                "tee": True,
                "symbolic_solver_labels": True,
            },
        )
    else:
        results = opt.solve(
            scaled_preproc_model, keepfiles=True, tee=True, symbolic_solver_labels=True
        )
    pyo.TransformationFactory("core.scale_model").propagate_solution(
        scaled_preproc_model, preproc_model
    )

    print(results.solver.termination_condition)
    print("======================")
    preproc_model.display()


def get_grad(grad_var_starting_val, args_arr, jmod):
    import equinox as eqx

    return eqx.filter_grad(
        lambda gv, arr0, arr, a: a(
            grad_var=gv,
            V_dd=arr0,
            f=arr[1],
            MemReadL=arr[2],
            MemWriteL=arr[3],
            MemReadPact=arr[4],
            MemWritePact=arr[5],
            MemPpass=arr[6],
            Reff_And=arr[7],
            Reff_Or=arr[8],
            Reff_Add=arr[9],
            Reff_Sub=arr[10],
            Reff_Mult=arr[11],
            Reff_FloorDiv=arr[12],
            Reff_Mod=arr[13],
            Reff_LShift=arr[14],
            Reff_RShift=arr[15],
            Reff_BitOr=arr[16],
            Reff_BitXor=arr[17],
            Reff_BitAnd=arr[18],
            Reff_Eq=arr[19],
            Reff_NotEq=arr[20],
            Reff_Lt=arr[21],
            Reff_LtE=arr[22],
            Reff_Gt=arr[23],
            Reff_GtE=arr[24],
            Reff_USub=arr[25],
            Reff_UAdd=arr[26],
            Reff_IsNot=arr[27],
            Reff_Not=arr[28],
            Reff_Invert=arr[29],
            Reff_Regs=arr[30],
            Ceff_And=arr[31],
            Ceff_Or=arr[32],
            Ceff_Add=arr[33],
            Ceff_Sub=arr[34],
            Ceff_Mult=arr[35],
            Ceff_FloorDiv=arr[36],
            Ceff_Mod=arr[37],
            Ceff_LShift=arr[38],
            Ceff_RShift=arr[39],
            Ceff_BitOr=arr[40],
            Ceff_BitXor=arr[41],
            Ceff_BitAnd=arr[42],
            Ceff_Eq=arr[43],
            Ceff_NotEq=arr[44],
            Ceff_Lt=arr[45],
            Ceff_LtE=arr[46],
            Ceff_Gt=arr[47],
            Ceff_GtE=arr[48],
            Ceff_USub=arr[49],
            Ceff_UAdd=arr[50],
            Ceff_IsNot=arr[51],
            Ceff_Not=arr[52],
            Ceff_Invert=arr[53],
            Ceff_Regs=arr[54],
        )
    )(grad_var_starting_val, args_arr[0], args_arr, jmod)


def scp_opt(tech_params, edp):
    import sympy2jax
    import jax.numpy as jnp
    # print(tech_params)
    initial_val = edp.subs(tech_params)
    current_val = initial_val
    Ceff_scale = 1e10
    scaled_edp_map = {}
    for name in hw_symbols.symbol_table:
        if name.startswith("Ceff") or name.startswith("Mem"):
            scaled_edp_map[hw_symbols.symbol_table[name]] = (
                hw_symbols.symbol_table[name] / Ceff_scale
            )
    edp_scaled = edp.subs(scaled_edp_map)
    # print(edp_scaled)
    for i in range(10):
        args_arr = []
        starting_vals = np.zeros(len(hw_symbols.symbol_table))
        i = 0
        for name in hw_symbols.symbol_table:
            value = tech_params[hw_symbols.symbol_table[name]]
            if name.startswith("Ceff") or name.startswith("Mem"):
                value *= Ceff_scale
            args_arr.append(jnp.array(value))
            starting_vals[i] = value
            i += 1
        grad_map = {}
        grad_var = sympy.symbols("grad_var")
        ind = 0
        for name in hw_symbols.symbol_table:
            edp_cur = edp_scaled.subs({hw_symbols.symbol_table[name]: grad_var})
            jmod = sympy2jax.SymbolicModule(edp_cur)
            grad_map[name] = get_grad(args_arr[ind], args_arr, jmod)
            # print(name, grad_map[name])
            ind += 1
        # print(args_arr)

        # print(f"Grad map:\n {grad_map}")
        x = cp.Variable(len(hw_symbols.symbol_table), pos=True)
        lam = 0.0000001
        obj = lam * cp.norm1(starting_vals - x)
        for i in range(len(hw_symbols.symbol_table)):
            obj += grad_map[list(hw_symbols.symbol_table)[i]] * x[i]
        constr = []
        for i in range(len(hw_symbols.symbol_table)):
            constr += [x[i] >= starting_vals[i] * 0.95, x[i] <= starting_vals[i] * 1.05]
        prob = cp.Problem(cp.Minimize(obj), constr)
        prob.solve(solver=cp.GLPK)
        for i in range(len(hw_symbols.symbol_table)):
            if list(hw_symbols.symbol_table)[i].startswith("Ceff") or list(
                hw_symbols.symbol_table
            )[i].startswith("Mem"):
                tech_params[
                    hw_symbols.symbol_table[list(hw_symbols.symbol_table)[i]]
                ] = (x.value[i] / Ceff_scale)
            else:
                tech_params[
                    hw_symbols.symbol_table[list(hw_symbols.symbol_table)[i]]
                ] = x.value[i]
        # print(f"tech params: {tech_params}")
        current_val = edp.subs(tech_params)
        # print("result of problem:", current_val)
        if current_val <= initial_val / 2:
            break
    # print(f"result: {x.value}")
    # print(tech_params)
    return tech_params


def optimize(tech_params, edp, opt):
    if opt == "scp":
        return scp_opt(tech_params, edp)
    else:
        return ipopt(tech_params, edp)


def main():

    hw = HardwareModel(cfg=args.architecture_config)

    rcs = hw.get_optimization_params_from_tech_params()
    print(rcs)
    initial_params = generate_init_params_from_rcs_as_symbols(rcs)
    edp = open("sympy.txt", "r")
    edp = sympify(edp.readline())

    results = optimize(initial_params, edp, args.opt)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Optimize",
        description="Optimization part of the Inverse Pass. This runs after an analytic equation for the cost is created.",
        epilog="Text at the bottom of help",
    )

    parser.add_argument(
        "-c",
        "--architecture_config",
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
