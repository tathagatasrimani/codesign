from preprocess import Preprocessor
import pyomo.environ as pyo
from sympy import sympify

initial_params = {}
initial_params["f"] = 1e6 
initial_params["C_int_inv"] = 1e-8 
initial_params["V_dd"] = 1 
initial_params["C_input_inv"] = 1e-9 

multistart = False

def main():
    with open("sympy.txt") as f:
        s = f.read()
    new = sympify(s)
    edp = new

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