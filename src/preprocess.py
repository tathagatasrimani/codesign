import logging
import time
import os
logger = logging.getLogger(__name__)

import copy
import sympy as sp
import pyomo.environ as pyo
import pyomo.core.expr.sympy_tools as sympy_tools
from pyomo.opt import SolverFactory

from .MyPyomoSympyBimap import MyPyomoSympyBimap
from . import hardwareModel

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
        print(f"constraint: {self.constraints[i]}")
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
        l = self.initial_val / 100
        logger.info("Adding regularization.")
        self.regularization = 0
        # normal regularization for each variable
        for symbol in self.free_symbols:
            self.regularization += hardwareModel.symbolic_convex_max((self.params.tech_values[symbol]/ symbol- 1), 
                                                         (symbol/self.params.tech_values[symbol] - 1)) ** 2

        # expressions inside a log/sqrt must not be negative
        """for log_expr in self.log_exprs_s:
            self.regularization += 1e15*(
                hardwareModel.symbolic_convex_max(-log_expr, 0, evaluate=False)
            ) ** 2
        for pow_expr in self.pow_exprs_s:
            self.regularization += 1e15 * (
                hardwareModel.symbolic_convex_max(-pow_expr, 0, evaluate=False)
            ) ** 2"""
        ##obj += l * self.regularization
        # alternative: minimax regularization. solver didn't really like it.
        """sym_list = [(symbol/self.params.tech_values[symbol] + self.params.tech_values[symbol]/symbol) for symbol in self.free_symbols]
        while len(sym_list) > 2:
            new_sym_list = []
            for i in range(len(sym_list)-1)[::2]:
                new_sym_list.append(hardwareModel.symbolic_convex_max(sym_list[i], sym_list[i+1]))
            if len(sym_list) % 2 == 1:
                new_sym_list.append(sym_list[-1])
            sym_list = new_sym_list
        self.regularization = hardwareModel.symbolic_convex_max(sym_list[0], sym_list[1])"""
        print(f"regularization: {self.regularization}")
                
        for symbol in self.free_symbols:
            self.regularization += hardwareModel.symbolic_convex_max(symbol, (symbol / self.params.tech_values[symbol] + self.params.tech_values[symbol] / symbol))
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
            opt.options["max_iter"] = 100
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
            self.free_symbols.extend(constraints[i].free_symbols)
        self.free_symbols = list(set(self.free_symbols))

        self.improvement = improvement
        self.constraints = constraints

        self.initial_val = float(obj.subs(self.params.tech_values))

        print(f"length of free symbols: {len(self.free_symbols)}")

        model.nVars = pyo.Param(initialize=len(self.free_symbols))
        model.N = pyo.RangeSet(model.nVars)
        model.x = pyo.Var(model.N, domain=pyo.NonNegativeReals)
        self.mapping = {}

        i = 0
        for j in model.x:
            self.mapping[self.free_symbols[i]] = j
            print(f"x[{j}] {self.free_symbols[i]}")
            i += 1

        print("building bimap")
        m = MyPyomoSympyBimap()
        self.bimap = m
        for symbol in self.free_symbols:
            # create self.mapping of sympy symbols to pyomo symbols
            m.sympy2pyomo[symbol] = model.x[self.mapping[symbol]]
            # give pyomo symbols an inital value for warm start
            model.x[self.mapping[symbol]] = self.params.tech_values[symbol]
            print(f"symbol: {symbol}; initial value: {self.params.tech_values[symbol]}")

        # find all pow/log expressions within obj equation and cacti equations, convert to pyomo
        # We shouldn't need to find any pow/log expressions in the obj expression itself. Cacti sub expressions
        # should suffice, but keep an eye on this.
        start_time = time.time()
        self.find_log_exprs_to_constrain(obj)

        logger.info(f"time to find log exprs to constrain: {time.time()-start_time}")

        start_time = time.time()

        # hotfix: substitute each log expression with max(expr, 1e-3) to avoid negatives inside log
        for log_expr in self.log_exprs_s:
            self.log_subs[log_expr] = hardwareModel.symbolic_convex_max(log_expr, 0.001, evaluate=False)
            logger.info(f"log expr: {log_expr}; sub: {self.log_subs[log_expr]}")
        

        # for overall obj expression and cacti sub expressions, we must ensure there is no
        # negative inside a sqrt/log. So substitute all log(expr) with log(max(expr, 1e-3)) and 
        # all sqrt(expr) with sqrt(abs(expr))
        obj = obj.xreplace(self.log_subs)

        logger.info(f"time to sub log exprs: {time.time()-start_time}")

        start_time = time.time()
        self.find_pow_exprs_to_constrain(obj)
        logger.info(f"time to find pow exprs to constrain: {time.time()-start_time}")

        start_time = time.time()
        
        for pow_expr in self.pow_exprs_s:
            self.pow_subs[pow_expr] = hardwareModel.symbolic_convex_max(pow_expr, 0, evaluate=False)
            #logger.info(f"pow expr: {pow_expr}; sub: {self.pow_subs[pow_expr]}")
        obj = obj.xreplace(self.pow_subs)
        logger.info(f"time to sub pow exprs: {time.time()-start_time}")


        start_time = time.time()
        self.find_exp_exprs_to_constrain(obj)
        for exp_expr in self.exp_exprs_s:
            self.exp_subs[exp_expr] = hardwareModel.symbolic_convex_min(exp_expr, 100, evaluate=False)
            obj = obj.xreplace(self.exp_subs)
        for i in range(len(self.constraints)):
            self.constraints[i] = self.constraints[i].xreplace(self.exp_subs)

        logger.info(f"time to sub exp exprs: {time.time()-start_time}")

        start_time = time.time()

        for pow_expr in self.pow_exprs_s:
            # pow expressions may have log exprs inside them, so substitute first
            pow_expr = pow_expr.xreplace(self.log_subs)
            self.pow_exprs_to_constrain.append(sympy_tools.sympy2pyomo_expression(pow_expr, m))
        for log_expr in self.log_exprs_s:
            self.log_exprs_to_constrain.append(sympy_tools.sympy2pyomo_expression(log_expr, m))
        print(f"converting to pyomo exp")
        print(f"obj: {obj}")
        print(f"m: {m}")
        self.pyomo_obj_exp = sympy_tools.sympy2pyomo_expression(obj, m)

        sympy_obj = self.add_regularization_to_objective(obj)
        self.regularization = sympy_tools.sympy2pyomo_expression(self.regularization, m)
        print(f"added regularization")

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
