import logging
logger = logging.getLogger(__name__)

import copy
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
        self.cacti_subs_pyo = {}
        self.cacti_sub_vars = set()
        self.log_subs = {}
        self.pow_subs = {}
        self.cacti_subs_s = {}

    def V_dd_lower(self, model):
        return model.x[self.mapping[hw_symbols.V_dd]] >= 0.5
    
    def Vdd_not_cutoff(self, model):
        return model.x[self.mapping[hw_symbols.Vdd]] >= model.x[self.mapping[hw_symbols.Vth]]

    def V_dd_upper(self, model):
        return model.x[self.mapping[hw_symbols.V_dd]] <= 1.7
    
    def make_pow_constraint(self, model, i):
        return self.pow_exprs_to_constrain[i] >= 0
        
    def make_log_constraint(self, model, i):
        return self.log_exprs_to_constrain[i] >= 0.0001
    
    def make_cacti_constraint(self, model, i):
        cacti_var_sp = list(self.cacti_subs_pyo.keys())[i]
        cacti_var = model.x[self.mapping[cacti_var_sp]]
        return cacti_var == self.cacti_subs_pyo[cacti_var_sp]

    def max_val_orig_val_rule(self, model, i):
        return model.x[self.mapping[self.free_symbols[i]]] <= self.initial_params[self.free_symbols[i].name]
    
    def find_exprs_to_constrain(self, expr, debug=False):
        if debug: logger.info(f"expr.func is {expr.func}")
        if expr.func == sp.core.power.Pow and expr.base.func != sp.core.symbol.Symbol:
            if debug: logger.info(f"exponent is {expr.exp}")
            # if we are taking an even root (i.e. square root), no negative numbers allowed
            if abs(expr.exp) == 0.5:
                if debug: logger.info(f"base func is {expr.base.func}")
                self.pow_exprs_s.add(expr.base)
        elif expr.func == sp.log:
            if not expr.args[0].is_constant():
                self.log_exprs_s.add(expr.args[0])
                if debug: logger.info(f"arg of log is {expr.args[0]}, type is {type(expr.args[0])}")
            elif debug:
                logger.info(f"constant arg of log is {expr.args[0]}, type is {type(expr.args[0])}")
        
        for arg in expr.args:
            self.find_exprs_to_constrain(arg)

    def sub_pow_exprs(self,expr, prnt=False):
        # replace each square root expression with sqrt(abs()) to avoid negative inside sqrt
        if (isinstance(expr, sp.Number)): return expr
        new_args = []
        for i in range(len(expr.args)):
            new_args.append(self.sub_pow_exprs(expr.args[i], prnt=prnt))
        if len(new_args):
            expr = expr.func(*tuple(new_args), evaluate=False)
            if expr.func == sp.core.power.Pow and expr.exp == 0.5:
                expr = sp.sqrt(sp.Abs(expr.base, evaluate=False))
            elif expr.func == sp.core.power.Pow and expr.exp == -0.5:
                expr = 1 / (sp.sqrt(sp.Abs(expr.base, evaluate=False)))
        return expr


    def add_constraints(self, model):
        logger.info("Adding Constraints")
        print(f"adding constraints. initial val: {self.initial_val};") # edp_exp: {self.pyomo_edp_exp}")
        model.Constraint1 = pyo.Constraint(expr=self.pyomo_edp_exp >= self.initial_val / self.improvement)

        model.PowConstraint = pyo.Constraint([i for i in range(len(self.pow_exprs_to_constrain))], rule=self.make_pow_constraint)

        model.LogConstraint = pyo.Constraint([i for i in range(len(self.log_exprs_to_constrain))], rule=self.make_log_constraint)

        model.CactiConstraint = pyo.Constraint([i for i in range(len(self.cacti_subs_pyo))], rule=self.make_cacti_constraint)

        if hw_symbols.V_dd in self.mapping: 
            model.V_dd_lower = pyo.Constraint(rule=self.V_dd_lower)
            model.V_dd_upper = pyo.Constraint(rule=self.V_dd_upper)
        # cacti Vdd >= Vth constraint
        if hw_symbols.Vdd in self.mapping:
            model.Vdd_not_cutoff = pyo.Constraint(rule=self.Vdd_not_cutoff)

        # all parameters can only be less than or equal to their initial values
        model.Constraint2 = pyo.Constraint([i for i in range(len(self.free_symbols))], rule=self.max_val_orig_val_rule)

        return model

    def add_regularization_to_objective(self, obj, l):
        """
        Parameters:
        obj: sympy objective function
        l: regularization hyperparameter
        """
        logger.info("Adding regularization.")
        # normal regularization for each variable
        for symbol in self.free_symbols:
            if symbol not in self.cacti_sub_vars:
                obj += l * (
                    self.initial_params[symbol.name]
                    / symbol
                    - 1
                ) ** 2
        

        # add large regularization for any constraint which we must not violate
        for cacti_var in self.cacti_sub_vars:
            # cacti expressions must equal their corresponding variable
            if cacti_var in self.mapping:
                obj += 1e15*(
                    cacti_var-self.cacti_subs_s[cacti_var]
                ) ** 2

        # expressions inside a log/sqrt must not be negative
        for log_expr in self.log_exprs_s:
            obj += 1e15*(
                symbolic_simulate.symbolic_convex_max(-log_expr, 0)
            ) ** 2
        for pow_expr in self.pow_exprs_s:
            obj += 1e15 * (
                symbolic_simulate.symbolic_convex_max(-pow_expr, 0)
            ) ** 2


        # alternative: minimax regularization. solver didn't really like it.
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
        model.scaling_factor[model.obj] = self.obj_scale
        print(f"mapping: {self.mapping}")
        for s in self.free_symbols:
            if s.name in self.initial_params and self.initial_params[s.name] != 0 and s not in self.cacti_sub_vars:
                print(f"symbol name: {s.name}")
                model.scaling_factor[model.x[self.mapping[s]]] = (
                    1 / self.initial_params[s.name]
                )
            elif s in self.cacti_sub_vars:
                print(f"cacti symbol name: {s.name}")
                new_cacti_subs = copy.copy(self.expr_symbols)
                for param in new_cacti_subs:
                    new_cacti_subs[param] = 1
                scaled_value = float(s.subs(new_cacti_subs))
                model.scaling_factor[model.x[self.mapping[s]]] = scaled_value / self.initial_params[s.name]
                print(f"scaling factor: {model.scaling_factor[model.x[self.mapping[s]]]}")
                
                

    def symbols_in_Buf_Mem_L(self, buf_l_file, mem_l_file):
        memL_expr = sp.sympify(
            open(mem_l_file, "r").readline(), locals=hw_symbols.symbol_table
        )
        bufL_expr = sp.sympify(
            open(buf_l_file, "r").readline(), locals=hw_symbols.symbol_table
        )
        free_symbols = memL_expr.free_symbols.union(bufL_expr.free_symbols)
        return free_symbols

    def begin(self, model, edp, initial_params, improvement, cacti_subs, multistart, regularization):
        self.multistart = multistart
        self.expr_symbols = {}
        self.free_symbols = []
        self.initial_params = initial_params
        self.improvement = improvement
        self.cacti_subs_s = cacti_subs

        mem_buf_l_symbols = self.symbols_in_Buf_Mem_L("src/cacti/symbolic_expressions/Buf_access_time.txt", "src/cacti/symbolic_expressions/Mem_access_time.txt")
        desired_free_symbols = ["Vdd", "C_g_ideal"]#, "C_junc", "I_on_n", "vert_dielectric_constant"] #, "Vdsat"] #, "Mobility_n"]

        symbols_to_remove =  [
            sym for sym in mem_buf_l_symbols if sym.name not in desired_free_symbols
        ]
        mem_buf_l_init_params = {sym: initial_params[sym.name] for sym in symbols_to_remove}
        # edp = edp.xreplace(mem_buf_l_init_params)

        # keep track of which cacti related variables are used in the edp expression
        cacti_free_symbols = set()

        # keep track of all the variables which will need to be substituted out of the original edp equation
        self.cacti_sub_vars = set(cacti_subs.keys())
        print(f"cacti sub vars: {self.cacti_sub_vars}")

        print(f"length of free symbols: {len(edp.free_symbols)}")

        for s in edp.free_symbols:
            self.free_symbols.append(s)
            print(f"symbol name is {s.name}")
            if s in self.cacti_sub_vars:
                cacti_exp = cacti_subs[s]
                for sub_symbol in cacti_exp.free_symbols:
                    # track each cacti variable in the expression that will be substituted
                    cacti_free_symbols.add(sub_symbol)
                    self.expr_symbols[sub_symbol] = initial_params[sub_symbol.name]
                self.expr_symbols[s] = float(cacti_subs[s].xreplace(self.expr_symbols).evalf())
                initial_params[s.name] = self.expr_symbols[s]
                print(f"symbol {s.name} has initial value {self.expr_symbols[s]}")
            elif s.name in initial_params:  # change this to just s
                self.expr_symbols[s] = initial_params[s.name]

        for free_symbol in cacti_free_symbols:
            self.free_symbols.append(free_symbol)


        self.initial_val = float(edp.xreplace(self.expr_symbols))

        model.nVars = pyo.Param(initialize=len(edp.free_symbols)+len(cacti_free_symbols))
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
        for symbol in self.expr_symbols.keys():
            # create self.mapping of sympy symbols to pyomo symbols
            m.sympy2pyomo[symbol] = model.x[self.mapping[symbol]]
            # give pyomo symbols an inital value for warm start
            model.x[self.mapping[symbol]] = self.expr_symbols[symbol]
            print(f"symbol: {symbol}; initial value: {self.expr_symbols[symbol]}")

        # find all pow/log expressions within edp equation and cacti equations, convert to pyomo
        self.find_exprs_to_constrain(edp)
        for cacti_var in cacti_subs.keys():
            if cacti_var in self.mapping:
                self.find_exprs_to_constrain(cacti_subs[cacti_var])

        # hotfix: substitute each log expression with max(expr, 1e-3) to avoid negatives inside log
        for log_expr in self.log_exprs_s:
            self.log_subs[log_expr] = symbolic_simulate.symbolic_convex_max(log_expr, 0.001)
        

        # for overall edp expression and cacti sub expressions, we must ensure there is no
        # negative inside a sqrt/log. So substitute all log(expr) with log(max(expr, 1e-3)) and 
        # all sqrt(expr) with sqrt(abs(expr))
        edp = edp.xreplace(self.log_subs)
        edp = self.sub_pow_exprs(edp)
        for cacti_var in cacti_subs.keys():
            if cacti_var in self.mapping:
                cacti_subs[cacti_var] = self.sub_pow_exprs(cacti_subs[cacti_var])
                cacti_subs[cacti_var] = cacti_subs[cacti_var].xreplace(self.log_subs)

        # save subbed expressions to tmp file for debugging purposes
        with open("src/tmp/symbolic_edp_subbed.txt", "w") as f:
            f.write(str(edp))
            f.write("\n")
            for cacti_var in cacti_subs.keys():
                if cacti_var in self.mapping:
                    f.write(f"{cacti_var.name}: {str(cacti_subs[cacti_var])}\n")

        # convert cacti subs expressions to pyomo
        for cacti_var in cacti_subs.keys():
            if cacti_var in self.mapping:
                self.cacti_subs_pyo[cacti_var] = sympy_tools.sympy2pyomo_expression(cacti_subs[cacti_var], m)

        for pow_expr in self.pow_exprs_s:
            # pow expressions may have log exprs inside them, so substitute first
            pow_expr = pow_expr.xreplace(self.log_subs)
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
        self.add_constraints(model)
        self.create_scaling(model)

        scaled_model = pyo.TransformationFactory("core.scale_model").create_using(model)
        # this transformation was having issues for some reason...
        # should be ok without it for the time being
        """scaled_preproc_model = pyo.TransformationFactory(
            "contrib.constraints_to_var_bounds"
        ).create_using(scaled_model)
        preproc_model = pyo.TransformationFactory(
            "contrib.constraints_to_var_bounds"
        ).create_using(model)"""
        opt = self.get_solver()
        return opt, scaled_model, model, self.free_symbols, self.mapping
