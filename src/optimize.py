import yaml

import pyomo.environ as pyo
from sympy import sympify

from preprocess import Preprocessor
from sim_util import generate_init_params_from_rcs_as_symbols
from hardwareModel import HardwareModel
import hw_symbols
import sympy2jax
import equinox as eqx
import jax
import jax.numpy as jnp
import cvxpy as cp

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

def get_grad(args_arr, jmod):
    return eqx.filter_grad(lambda arr0, arr, a: a(Reff_Add=arr0, 
                                                  Ceff_Add=arr[1], 
                                                  Reff_Regs=arr[2], 
                                                  Ceff_Regs=arr[3], 
                                                  Reff_Not=arr[4], 
                                                  MemReadL=arr[5], 
                                                  MemWriteL=arr[6], 
                                                  MemReadPact=arr[7], 
                                                  MemWritePact=arr[8], 
                                                  MemPpass=arr[9], 
                                                  f=arr[10], 
                                                  V_dd=arr[11]))(args_arr[0], args_arr, jmod)

def rotate_arr(args_arr):
    next_val = args_arr[0]
    for i in range(len(args_arr))[::-1]:    
        tmp = next_val
        next_val = args_arr[(i-1)%len(args_arr)]
        args_arr[(i-1)%len(args_arr)] = tmp
    return args_arr

def scp_opt(tech_params, edp):
    #print(tech_params)
    jmod = sympy2jax.SymbolicModule(edp)
    starting_vals = [
        tech_params[hw_symbols.Reff["Add"]],
        tech_params[hw_symbols.Ceff["Add"]],
        tech_params[hw_symbols.Reff["Regs"]],
        tech_params[hw_symbols.Ceff["Regs"]],
        tech_params[hw_symbols.Reff["Not"]],
        tech_params[hw_symbols.MemReadL],
        tech_params[hw_symbols.MemWriteL],
        tech_params[hw_symbols.MemReadPact],
        tech_params[hw_symbols.MemWritePact],
        tech_params[hw_symbols.MemPpass],
        tech_params[hw_symbols.f],
        tech_params[hw_symbols.V_dd]
    ]
    Reff_Add_init = jnp.array(tech_params[hw_symbols.Reff["Add"]])
    Ceff_Add_init = jnp.array(tech_params[hw_symbols.Ceff["Add"]])
    Reff_Regs_init = jnp.array(tech_params[hw_symbols.Reff["Regs"]])
    Ceff_Regs_init = jnp.array(tech_params[hw_symbols.Ceff["Regs"]])
    Reff_Not_init = jnp.array(tech_params[hw_symbols.Reff["Not"]])
    MemReadL_init = jnp.array(tech_params[hw_symbols.MemReadL])
    MemWriteL_init = jnp.array(tech_params[hw_symbols.MemWriteL])
    MemReadPact_init = jnp.array(tech_params[hw_symbols.MemReadPact])
    MemWritePact_init = jnp.array(tech_params[hw_symbols.MemWritePact])
    MemPpass_init = jnp.array(tech_params[hw_symbols.MemPpass])
    f_init = jnp.array(tech_params[hw_symbols.f])
    V_dd_init = jnp.array(tech_params[hw_symbols.V_dd])
    
    args_arr = [
        Reff_Add_init,
        Ceff_Add_init,
        Reff_Regs_init,
        Ceff_Regs_init,
        Reff_Not_init,
        MemReadL_init,
        MemWriteL_init,
        MemReadPact_init,
        MemWritePact_init,
        MemPpass_init,
        f_init,
        V_dd_init
    ]
    grad_names = [
        "Reff_Add",
        "Ceff_Add",
        "Reff_Regs",
        "Ceff_Regs",
        "Reff_Not",
        "MemReadL",
        "MemWriteL",
        "MemReadPact",
        "MemWritePact",
        "MemPpass",
        "f",
        "V_dd"
    ]
    grad_map = {}
    for name in grad_names:
        grad_map[name] = get_grad(args_arr, jmod)
        rotate_arr(args_arr)


    print(f"Grad map:\n {grad_map}")
    x = cp.Variable(len(grad_names))
    lam = 1000
    obj = lam * cp.norm1(starting_vals-x)
    for i in range(len(grad_names)):
        obj += grad_map[grad_names[i]] * x[i]
    constr = []
    for i in range(len(grad_names)):
        constr += [x[i] >= starting_vals[i]*0.95, x[i] <= starting_vals[i]*1.05]
    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve(solver=cp.GLPK)
    print(f"result: {x.value}")
    return x, grad_names

def optimize(tech_params, edp, opt):
    if opt == "scp":
        return scp_opt(tech_params, edp)
    else:
        return ipopt(tech_params, edp)


def main():

    hw_cfg = "aladdin_const_with_mem" # this needs to be a cli arg at somepoint
    hw = HardwareModel(cfg=hw_cfg)

    rcs = hw.get_optimization_params_from_tech_params()
    print(rcs)
    initial_params = generate_init_params_from_rcs_as_symbols(rcs)
    
    results, initial_params, free_symbols, mapping = optimize(initial_params)

    return initial_params, free_symbols, mapping


if __name__ == "__main__":
    main()
