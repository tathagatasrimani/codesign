import logging
logger = logging.getLogger(__name__)

import sympy as sp
import pyomo.environ as pyo
import pyomo.core.expr.sympy_tools as sympy_tools
from pyomo.opt import SolverFactory

from .MyPyomoSympyBimap import MyPyomoSympyBimap
from . import hw_symbols
from . import symbolic_simulate

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
        self.improvement = 1.1
        self.pow_exprs_s = set()
        self.log_exprs_s = set()
        self.pow_exprs_to_constrain = []
        self.log_exprs_to_constrain = []

    def f(self, model):
        return model.x[self.mapping[hw_symbols.f]] >= 1e6

    def f_upper(self, model):
        return model.x[self.mapping[hw_symbols.f]] <= 5e9

    def V_dd_lower(self, model):
        return model.x[self.mapping[hw_symbols.V_dd]] >= 0.5

    def V_dd_upper(self, model):
        return model.x[self.mapping[hw_symbols.V_dd]] <= 1.7
    
    def find_exprs_to_constrain(self, expr):
        #logger.warning(f"expr.func is {expr.func}")
        if expr.func == sp.core.power.Pow and expr.base.func != sp.core.symbol.Symbol:
            #logger.warning(f"exponent is {expr.exp}")
            # if we are taking an even root (i.e. square root), no negative numbers allowed
            if expr.exp == 0.5:
                #logger.warning(f"base func is {expr.base.func}")
                self.pow_exprs_s.add(expr.base)
        elif expr.func == sp.log:
            if not expr.args[0].is_constant():
                self.log_exprs_s.add(expr.args[0])
                #logger.warning(f"arg of log is {expr.args[0]}, type is {type(expr.args[0])}")
            #else:
                #logger.warning(f"constant arg of log is {expr.args[0]}, type is {type(expr.args[0])}")
        
        for arg in expr.args:
            self.find_exprs_to_constrain(arg)

    def add_constraints(self, model):
        logger.info("Adding Constraints")
        # this is where we say EDP_final = EDP_initial / 10
        print(f"adding constraints. initial val: {self.initial_val};") # edp_exp: {self.pyomo_edp_exp}")
        # model.Constraint = pyo.Constraint(expr=self.pyomo_edp_exp <= self.initial_val / 1.9)
        model.Constraint1 = pyo.Constraint(expr=self.pyomo_edp_exp >= self.initial_val / self.improvement)

        def make_pow_constraint(model, i):
            return self.pow_exprs_to_constrain[i] >= 0
        
        def make_log_constraint(model, i):
            return self.log_exprs_to_constrain[i] >= 0.0001

        model.PowConstraint = pyo.Constraint([i for i in range(len(self.pow_exprs_to_constrain))], rule=make_pow_constraint)

        model.LogConstraint = pyo.Constraint([i for i in range(len(self.log_exprs_to_constrain))], rule=make_log_constraint)

        #model.V_dd_lower = pyo.Constraint(rule=self.V_dd_lower)
        # model.V_dd_upper = pyo.Constraint(rule=self.V_dd_upper)
        # model.V_dd = pyo.Constraint(expr = model.x[self.mapping[hw_symbols.V_dd]] == self.initial_params["V_dd"])

        # all parameters can only be less than or equal to their initial values
        def max_val_orig_val_rule(model, i):
            return model.x[self.mapping[self.free_symbols[i]]] <= self.initial_params[self.free_symbols[i].name]
        model.Constraint2 = pyo.Constraint([i for i in range(len(self.free_symbols))], rule=max_val_orig_val_rule)

        return model

    def add_regularization_to_objective(self, obj, l):
        """
        Parameters:
        obj: sympy objective function
        l: regularization hyperparameter
        """
        logger.info("Adding regularization.")
        for symbol in self.free_symbols:
            obj += l * (
                self.initial_params[symbol.name]
                / symbol
                - 1
            ) ** 2

        """max_term = 0
        for symbol in self.free_symbols:
            max_term = symbolic_simulate.symbolic_convex_max(max_term, (symbol / self.initial_params[symbol.name]))
        obj += l * max_term"""
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
            # opt.options['print_level'] = 5
            # opt.options['nlp_scaling_method'] = 'none'
            opt.options["bound_relax_factor"] = 0
            opt.options["max_iter"] = 100
            opt.options["print_info_string"] = "yes"
            opt.options["wantsol"] = 2
            opt.options["halt_on_ampl_error"] = "yes"
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

    def symbols_in_Buf_Mem_L(self, buf_l_file, mem_l_file):
        memL_expr = sp.sympify(
            open(mem_l_file, "r").readline(), locals=hw_symbols.symbol_table
        )
        bufL_expr = sp.sympify(
            open(buf_l_file, "r").readline(), locals=hw_symbols.symbol_table
        )
        free_symbols = memL_expr.free_symbols.union(bufL_expr.free_symbols)
        return free_symbols

    def begin(self, model, edp, initial_params, improvement, multistart, regularization):
        self.multistart = multistart
        self.expr_symbols = {}
        self.free_symbols = []
        self.initial_params = initial_params
        self.improvement = improvement

        mem_buf_l_symbols = self.symbols_in_Buf_Mem_L("src/cacti/symbolic_expressions/Buf_access_time.txt", "src/cacti/symbolic_expressions/Mem_access_time.txt")
        desired_free_symbols = ["Vdd", "C_g_ideal"]#, "C_junc", "I_on_n", "vert_dielectric_constant"] #, "Vdsat"] #, "Mobility_n"]

        symbols_to_remove =  [
            sym for sym in mem_buf_l_symbols if sym.name not in desired_free_symbols
        ]
        mem_buf_l_init_params = {sym: initial_params[sym.name] for sym in symbols_to_remove}
        # edp = edp.xreplace(mem_buf_l_init_params)

        for s in edp.free_symbols:
            self.free_symbols.append(s)
            if s.name in initial_params:  # change this to just s
                self.expr_symbols[s] = initial_params[s.name]
        self.initial_val = float(edp.xreplace(self.expr_symbols))

        model.nVars = pyo.Param(initialize=len(edp.free_symbols))
        model.N = pyo.RangeSet(model.nVars)
        model.x = pyo.Var(model.N, domain=pyo.NonNegativeReals)
        self.mapping = {}

        i = 0
        for j in model.x:
            self.mapping[self.free_symbols[i]] = j
            print(f"x[{j}] {self.free_symbols[i]}")
            i += 1

        print(f"building bimap")
        m = MyPyomoSympyBimap()
        for symbol in edp.free_symbols:
            # create self.mapping of sympy symbols to pyomo symbols
            m.sympy2pyomo[symbol] = model.x[self.mapping[symbol]]
            # give pyomo symbols an inital value for warm start
            model.x[self.mapping[symbol]] = self.expr_symbols[symbol]
            print(f"symbol: {symbol}; initial value: {self.expr_symbols[symbol]}")

        # find all pow expressions within edp equation, convert to pyomo
        self.find_exprs_to_constrain(edp)
        for pow_expr in self.pow_exprs_s:
            self.pow_exprs_to_constrain.append(sympy_tools.sympy2pyomo_expression(pow_expr, m))
        for log_expr in self.log_exprs_s:
            self.log_exprs_to_constrain.append(sympy_tools.sympy2pyomo_expression(log_expr, m))
        print(f"converting to pyomo exp")
        self.pyomo_edp_exp = sympy_tools.sympy2pyomo_expression(edp, m)

        sympy_obj = self.add_regularization_to_objective(edp, l=regularization)
        print(f"added regularization")

        self.obj = sympy_tools.sympy2pyomo_expression(sympy_obj, m)
        # print(f"created pyomo expression: {self.pyomo_edp_exp}")


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
