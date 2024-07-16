import logging
logger = logging.getLogger(__name__)

import pyomo.environ as pyo
from MyPyomoSympyBimap import MyPyomoSympyBimap
import pyomo.core.expr.sympy_tools as sympy_tools
from pyomo.opt import SolverFactory

import hw_symbols


import numpy as np
import cvxpy as cp
def l1_regularizer(self, starting_vals):
        # to promote sparsity in change (deviation from starting values)
        obj = self.obj + cp.norm1(starting_vals - self.x) # what currently exists
        return obj
    
def l2_regularizer(x, starting_vals):
    # to heavily penalize changes using the Euclidean norm: promoting general proximity to values
    obj = obj + cp.norm2(starting_vals - x)**2
    return obj

def elastic_net_regularization(alpha, starting_vals):
    # Weighing out L1 and L2 regularizers: selective agreement v/s general proximity
    obj = obj + cp.norm1(starting_vals - x) + (1 - alpha) + cp.norm2(starting_vals - x) ** 2
    return obj

def smoothness_regularizer(self):
    # Reduce total variance amongst subsequent elements: NOT RELEVANT?
    obj = 0
    for i in range(len(self.x) - 1):
        obj += cp.abs(self.x[i + 1] - self.x[i])
    return obj

def generalized_tikhonov_regularization(self, symbols_table):
    # Tikhonov Regularization: 
    L = np.eye(len(symbols_table))
    obj = self.obj + cp.quad_form(self.x, L)

def tc_norm_regularization(self):
    obj = self.obj + cp.norm(self.x, 'nuc')
    return obj

def outlier_regularization_huber(self, delta, starting_vals):
    obj = self.obj + cp.sum(cp.huber(self.x - starting_vals, delta))
    return obj

LAMBDA = 0.1 # regularization parameter

class Preprocessor:
    def __init__(self):
        self.mapping = {}
        self.pyomo_edp_exp = None
        self.initial_val = 0
        self.expr_symbols = {}
        self.free_symbols = []
        self.vars = []
        self.multistart = False
        self.obj = 0
        self.initial_params = {}
        self.obj_scale = 1

    def f(self, model):
        return model.x[self.mapping[hw_symbols.f]] >= 1e6

    def f_upper(self, model):
        return model.x[self.mapping[hw_symbols.f]] <= 5e9

    def V_dd_lower(self, model):
        return model.x[self.mapping[hw_symbols.V_dd]] >= 0.5

    def V_dd_upper(self, model):
        return model.x[self.mapping[hw_symbols.V_dd]] <= 1.7

    def add_constraints(self, model):
        logger.info("Adding Constraints")
        # this is where we say EDP_final = EDP_initial / 10
        print(f"adding constraints. initial val: {self.initial_val}; edp_exp: {self.pyomo_edp_exp}")
        model.Constraint = pyo.Constraint(expr=self.pyomo_edp_exp <= self.initial_val / 1.4)
        model.Constraint1 = pyo.Constraint(expr=self.pyomo_edp_exp >= self.initial_val / 1.6)
        model.V_dd_lower = pyo.Constraint(rule=self.V_dd_lower)
        model.V_dd_upper = pyo.Constraint(rule=self.V_dd_upper)
        # model.V_dd = pyo.Constraint(expr = model.x[self.mapping[hw_symbols.V_dd]] == self.initial_params["V_dd"])

        # all parameters can only be less than or equal to their initial values
        def max_val_orig_val_rule(model, i):
            return model.x[self.mapping[self.free_symbols[i]]] <= self.initial_params[self.free_symbols[i].name]
        model.Constraint2 = pyo.Constraint([i for i in range(len(self.free_symbols))], rule=max_val_orig_val_rule)
        
        return model

    def add_regularization_to_objective(self, model, l=1):
        """
        Parameters:
        model: pyomo model
        l: regularization hyperparameter
        """
        logger.info("Adding regularization.")
        for symbol in self.free_symbols:
            self.obj += l * (
                self.initial_params[symbol.name]
                / model.x[self.mapping[hw_symbols.symbol_table[symbol.name]]]
                - 1
            ) ** 2

    def get_solver(self):
        if self.multistart:
            opt = SolverFactory("multistart")
        else:
            opt = SolverFactory("ipopt")
            opt.options["warm_start_init_point"] = "yes"
            # opt.options['warm_start_bound_push'] = 1e-9
            # opt.options['warm_start_mult_bound_push'] = 1e-9
            # opt.options['warm_start_bound_frac'] = 1e-9
            # opt.options['warm_start_slack_bound_push'] = 1e-9
            # opt.options['warm_start_slack_bound_frac'] = 1e-9
            # opt.options['mu_init'] = 0.1
            # opt.options['acceptable_obj_change_tol'] = self.initial_val / 100
            # opt.options['tol'] = 0.5
            # opt.options['print_level'] = 5
            # opt.options['nlp_scaling_method'] = 'none'
            opt.options["bound_relax_factor"] = 0
            opt.options["max_iter"] = 100
            opt.options["print_info_string"] = "yes"
            opt.options["output_file"] = "solver_out.txt"
            opt.options["wantsol"] = 2
        return opt

    def create_scaling(self, model):
        logger.info("Creating scaling")
        model.scaling_factor[model.obj] = self.obj_scale
        print(f"mapping: {self.mapping}")
        for s in self.free_symbols:
            if s.name in self.initial_params and self.initial_params[s.name] != 0:
                print(f"symbol name: {s.name}")
                model.scaling_factor[model.x[self.mapping[s]]] = (
                    1 / self.initial_params[s.name]
                )

    def begin(self, model, edp, initial_params, multistart):
        self.multistart = multistart
        self.expr_symbols = {}
        self.free_symbols = []
        self.initial_params = initial_params
        for symbol in edp.free_symbols:
            edp = edp.subs({symbol: hw_symbols.symbol_table[symbol.name]})
        for s in edp.free_symbols:
            self.free_symbols.append(s)
            if s.name in initial_params:  # change this to just s
                self.expr_symbols[s] = initial_params[s.name]

        self.initial_val = float(edp.subs(self.expr_symbols))
        print(f"expr_symbols: {self.expr_symbols}")
        print("edp:", edp)
        print("initial val:", self.initial_val)

        model.nVars = pyo.Param(initialize=len(edp.free_symbols))
        model.N = pyo.RangeSet(model.nVars)
        model.x = pyo.Var(model.N, domain=pyo.NonNegativeReals)
        self.mapping = {}

        i = 0
        for j in model.x:
            self.mapping[self.free_symbols[i]] = j
            print(f"x[{j}] {self.free_symbols[i]}")
            i += 1

        m = MyPyomoSympyBimap()
        for symbol in edp.free_symbols:
            # create self.mapping of sympy symbols to pyomo symbols
            m.sympy2pyomo[symbol] = model.x[self.mapping[symbol]]
            # give pyomo symbols an inital value for warm start
            model.x[self.mapping[symbol]] = self.expr_symbols[symbol]
            print(f"symbol: {symbol}; initial value: {self.expr_symbols[symbol]}")
       
        self.pyomo_edp_exp = sympy_tools.sympy2pyomo_expression(edp, m)
        self.obj = self.pyomo_edp_exp

        self.add_regularization_to_objective(model, l=0.00001)

        model.obj = pyo.Objective(expr=self.obj, sense=pyo.minimize)
        model.cuts = pyo.ConstraintList()

        model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        self.create_scaling(model)
        self.add_constraints(model)

        scaled_model = pyo.TransformationFactory("core.scale_model").create_using(model)
        scaled_preproc_model = pyo.TransformationFactory(
            "contrib.constraints_to_var_bounds"
        ).create_using(scaled_model)
        preproc_model = pyo.TransformationFactory(
            "contrib.constraints_to_var_bounds"
        ).create_using(model)
        opt = self.get_solver()
        return opt, scaled_preproc_model, preproc_model, self.free_symbols, self.mapping
