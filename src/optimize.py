from preprocess import Preprocessor
import pyomo.environ as pyo
from sympy import sympify
import yaml

multistart = False

def main():
    with open("sympy.txt") as f:
        s = f.read()
    new = sympify(s)
    edp = new
    initial_params = {}
    rcs = yaml.load(open("rcs_current.yaml", "r"), Loader=yaml.Loader)
    for elem in rcs["Reff"]:
        initial_params["Reff_"+elem] = rcs["Reff"][elem]
        initial_params["Ceff_"+elem] = rcs["Ceff"][elem]
    initial_params["f"] = rcs["other"]["f"]
    initial_params["V_dd"] = rcs["other"]["V_dd"]

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