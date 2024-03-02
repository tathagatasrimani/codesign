import yaml

import pyomo.environ as pyo
from sympy import sympify

from preprocess import Preprocessor
from sim_util import generate_init_params_from_rcs_as_strings
from hardwareModel import HardwareModel

multistart = False
def main():
    with open("sympy.txt") as f:
        s = f.read()
    edp = sympify(s)

    hw_cfg = "aladdin_const_with_mem" # this needs to be a cli arg at somepoint
    hw = HardwareModel(cfg=hw_cfg)

    rcs = hw.get_optimization_params_from_tech_params()
    initial_params = generate_init_params_from_rcs_as_strings(rcs)

    model = pyo.ConcreteModel()
    opt, scaled_preproc_model, preproc_model, free_symbols, mapping = Preprocessor().begin(model, edp, initial_params, multistart=multistart) 
    if multistart:
        results = opt.solve(scaled_preproc_model, solver_args={'keepfiles':True, 'tee':True, 'symbolic_solver_labels':True})
    else:
        results = opt.solve(scaled_preproc_model, keepfiles=True, tee=True, symbolic_solver_labels=True)
    pyo.TransformationFactory('core.scale_model').propagate_solution(scaled_preproc_model, preproc_model)
    print(results.solver.termination_condition)  
    print("======================")
    preproc_model.display()
    return initial_params, free_symbols, mapping


if __name__ == "__main__":
    main()
