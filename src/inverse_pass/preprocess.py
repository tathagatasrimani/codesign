import logging
import time
import os
logger = logging.getLogger(__name__)

import copy
import sympy as sp
import pyomo.environ as pyo
import pyomo.core.expr.sympy_tools as sympy_tools
from pyomo.opt import SolverFactory

from src.inverse_pass.MyPyomoSympyBimap import MyPyomoSympyBimap
from src.hardware_model import hardwareModel

class Preprocessor:
    """
    Prepares and processes symbolic and Pyomo-based optimization models. Handles
    mapping between symbolic and Pyomo variables, applies constraints, and manages substitutions for
    technology parameters.
    """
    def __init__(self, params):
        """
        Initialize the Preprocessor instance, setting up mappings, initial values, and constraint sets.
        """
        self.mapping = {}
        self.pyomo_obj_exp = None
        self.initial_val = 0
        self.free_symbols = []
        self.multistart = False
        self.obj = 0
        self.params = params
        self.improvement = 1.1
        self.pow_exprs_s = set()
        self.log_exprs_s = set()
        self.pow_exprs_to_constrain = []
        self.log_exprs_to_constrain = []
        self.exp_exprs_s = set()
        self.log_subs = {}
        self.pow_subs = {}
        self.exp_subs = {}
        self.regularization = 0

    def make_pow_constraint(self, model, i):
        """
        Constraint: Ensure that each power expression to constrain is non-negative.

        Args:
            model (pyomo.ConcreteModel): The Pyomo model instance.
            i (int): Index of the power expression to constrain.

        Returns:
            pyomo.Constraint: Pyomo constraint enforcing power >= 0.
        """
        return self.pow_exprs_to_constrain[i] >= 0
        
    def make_log_constraint(self, model, i):
        """
        Constraint: Ensure that each log expression to constrain is above a small positive threshold.

        Args:
            model (pyomo.ConcreteModel): The Pyomo model instance.
            i (int): Index of the log expression to constrain.

        Returns:
            pyomo.Constraint: Pyomo constraint enforcing log argument >= 0.0001.
        """
        return self.log_exprs_to_constrain[i] >= 0.0001
    

    def find_pow_exprs_to_constrain(self, expr, debug=False):
        """
        Recursively identify power expressions that should be constrained to nonnegative values.

        Args:
            expr (sympy.Expr): Symbolic expression to analyze.
            debug (bool, optional): If True, log debug information. Defaults to False.

        Returns:
            None
        """
        if debug: logger.info(f"expr.func is {expr.func}")
        if expr.func == sp.core.power.Pow and expr.base.func != sp.core.symbol.Symbol:
            if debug: logger.info(f"exponent is {expr.exp}")
            # if we are taking an even root (i.e. square root), no negative numbers allowed
            if abs(expr.exp.evalf()) == 0.5:
                if debug: logger.info(f"base func is {expr.base.func}")
                self.pow_exprs_s.add(expr.base)
        
        for arg in expr.args:
            self.find_pow_exprs_to_constrain(arg, debug=debug)

    def find_log_exprs_to_constrain(self, expr, debug=False):
        """
        Recursively identify log expressions that should be constrained to nonnegative values.

        Args:
            expr (sympy.Expr): Symbolic expression to analyze.
            debug (bool, optional): If True, log debug information. Defaults to False.

        Returns:
            None
        """
        if debug: logger.info(f"expr.func is {expr.func}")
        if expr.func == sp.log:
            if not expr.args[0].is_constant():
                self.log_exprs_s.add(expr.args[0])
                if debug: logger.info(f"arg of log is {expr.args[0]}, type is {type(expr.args[0])}")
            elif debug:
                logger.info(f"constant arg of log is {expr.args[0]}, type is {type(expr.args[0])}")
        
        for arg in expr.args:
            self.find_log_exprs_to_constrain(arg, debug=debug)

    def find_exp_exprs_to_constrain(self, expr, debug=False):
        """
        Recursively identify exp expressions that should be constrained to a maximum value.

        Args:
            expr (sympy.Expr): Symbolic expression to analyze.
            debug (bool, optional): If True, log debug information. Defaults to False.

        Returns:
            None
        """
        if debug: logger.info(f"expr.func is {expr.func}")
        if expr.func == sp.exp:
            if not expr.args[0].is_constant():
                self.exp_exprs_s.add(expr.args[0])
            elif debug:
                logger.info(f"constant arg of exp is {expr.args[0]}, type is {type(expr.args[0])}")
        
        for arg in expr.args:
            self.find_exp_exprs_to_constrain(arg, debug=debug)
    
    def pyomo_constraint(self, model, i):
        #print(f"constraint: {self.constraints[i]}")
        pyo_expr = sympy_tools.sympy2pyomo_expression(self.constraints[i], self.bimap)
        return pyo_expr

    def add_constraints(self, model):
        """
        Add all relevant constraints (obj, power, log, V_dd, etc.) to the Pyomo model.

        Args:
            model (pyomo.ConcreteModel): The Pyomo model instance.

        Returns:
            None
        """
        logger.info("Adding Constraints")
        print(f"adding constraints. initial val: {self.initial_val};") # obj_exp: {self.pyomo_obj_exp}")
        model.Constraints = pyo.Constraint([i for i in range(len(self.constraints))], rule=self.pyomo_constraint)
        #model.Regularization = pyo.Constraint(expr=self.regularization <= self.initial_val / 50)
        return model

    def add_regularization_to_objective(self, obj):
        """
        Parameters:
        obj: sympy objective function
        """
        #l = self.initial_val / 100
        l = self.initial_val / (100 + len(self.free_symbols))
        logger.info("Adding regularization.")
        self.regularization = 0
        # normal regularization for each variable
        for symbol in self.free_symbols:
            if symbol.name in self.params.symbol_table:
                self.regularization += hardwareModel.symbolic_convex_max((self.params.tech_values[symbol]/ symbol- 1), 
                                                            (symbol/self.params.tech_values[symbol] - 1)) ** 2

        obj += l * self.regularization
        return obj
        
    def get_solver(self):
        if self.multistart:
            opt = SolverFactory("multistart")
        else:
            opt = SolverFactory("ipopt")
            opt.options["warm_start_init_point"] = "yes"
            opt.options['warm_start_bound_push'] = 1e-9
            opt.options['warm_start_mult_bound_push'] = 1e-9
            opt.options['warm_start_bound_frac'] = 1e-9
            opt.options['warm_start_slack_bound_push'] = 1e-9
            opt.options['warm_start_slack_bound_frac'] = 1e-9
            opt.options['mu_init'] = 0.1
            # opt.options['acceptable_obj_change_tol'] = self.initial_val / 100
            opt.options['tol'] = 1
            # opt.options['print_level'] = 12
            # opt.options['nlp_scaling_method'] = 'none'
            opt.options["bound_relax_factor"] = 0
            opt.options["max_iter"] = 1500
            opt.options["print_info_string"] = "yes"
            opt.options["output_file"] = "src/tmp/solver_out.txt"
            opt.options["wantsol"] = 2
            opt.options["halt_on_ampl_error"] = "yes"
        return opt

    def create_scaling(self, model):
        logger.info("Creating scaling")
        model.scaling_factor[model.obj] = 1/self.initial_val
        print(f"mapping: {self.mapping}")
        for s in self.free_symbols:
            if s in self.params.tech_values and self.params.tech_values[s] != 0:
                print(f"symbol name: {s.name}: scaling factor: {1 / self.params.tech_values[s]}")
                model.scaling_factor[model.x[self.mapping[s]]] = (
                    1 / self.params.tech_values[s]
                )

    def begin(self, model, obj, improvement, multistart, constraints):
        self.multistart = multistart
        self.free_symbols = list(obj.free_symbols)
        for i in range(len(constraints)):
            print(f"constraint {i}: {constraints[i]}")
            self.free_symbols.extend(constraints[i].free_symbols)
        print(f"obj: {obj}")
        self.free_symbols = list(set(self.free_symbols))

        self.improvement = improvement
        self.constraints = constraints

        self.initial_val = float(obj.xreplace(self.params.tech_values))

        print(f"length of free symbols: {len(self.free_symbols)}")

        model.nVars = pyo.Param(initialize=len(self.free_symbols))
        model.N = pyo.RangeSet(model.nVars)
        model.x = pyo.Var(model.N, domain=pyo.NonNegativeReals)
        self.mapping = {}

        i = 0
        for j in model.x:
            self.mapping[self.free_symbols[i]] = j
            if self.free_symbols[i].name in self.params.symbol_table:
                print(f"x[{j}] {self.free_symbols[i]}")
            i += 1

        print("building bimap")
        m = MyPyomoSympyBimap()
        self.bimap = m
        for symbol in self.free_symbols:
            # create self.mapping of sympy symbols to pyomo symbols
            m.sympy2pyomo[symbol] = model.x[self.mapping[symbol]]
            # give pyomo symbols an inital value for warm start
            if symbol in self.params.tech_values:# and not symbol.name.startswith("node_arrivals_"):
                model.x[self.mapping[symbol]] = self.params.tech_values[symbol]
                print(f"symbol: {symbol}; initial value: {self.params.tech_values[symbol]}")

        print(f"converting to pyomo exp")
        start_time = time.time()
        self.pyomo_obj_exp = sympy_tools.sympy2pyomo_expression(obj, m)

        sympy_obj = self.add_regularization_to_objective(obj)
        self.regularization = sympy_tools.sympy2pyomo_expression(self.regularization, m)
        print(f"added regularization")
        print(f"value of objective after regularization: {sympy_obj.xreplace(self.params.tech_values)}")

        self.obj = sympy_tools.sympy2pyomo_expression(sympy_obj, m)

        logger.info(f"time to convert all exprs to pyomo: {time.time()-start_time}")
        start_time = time.time()


        model.obj = pyo.Objective(expr=self.obj, sense=pyo.minimize)

        model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        self.add_constraints(model)
        self.create_scaling(model)

        logger.info(f"time to add constraints and create scaling: {time.time()-start_time}")

        scaled_model = pyo.TransformationFactory("core.scale_model").create_using(model)

        scaled_preproc_model = pyo.TransformationFactory(
            "contrib.constraints_to_var_bounds"
        ).create_using(scaled_model)
        preproc_model = pyo.TransformationFactory(
            "contrib.constraints_to_var_bounds"
        ).create_using(model)
        opt = self.get_solver()
        return opt, scaled_preproc_model, preproc_model    
