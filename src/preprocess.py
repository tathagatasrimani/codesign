import yaml

import pyomo.environ as pyo
from MyPyomoSympyBimap import MyPyomoSympyBimap
import pyomo.core.expr.sympy_tools as sympy_tools
from pyomo.opt import SolverFactory

import hw_symbols

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
        # this is where we say EDP_final = EDP_initial / 10
        model.Constraint = pyo.Constraint(expr=self.pyomo_edp_exp <= self.initial_val / 1.4)
        model.Constraint1 = pyo.Constraint(expr=self.pyomo_edp_exp >= self.initial_val / 1.6)
        model.V_dd_lower = pyo.Constraint(rule=self.V_dd_lower)
        model.V_dd_upper = pyo.Constraint(rule=self.V_dd_upper)
        # model.V_dd = pyo.Constraint(expr = model.x[self.mapping[hw_symbols.V_dd]] == self.initial_params["V_dd"])
        model.AddR = pyo.Constraint(
            expr=model.x[self.mapping[hw_symbols.Reff["Add"]]]
            <= self.initial_params["Reff_Add"]
        )
        model.RegsR = pyo.Constraint(
            expr=model.x[self.mapping[hw_symbols.Reff["Regs"]]]
            <= self.initial_params["Reff_Regs"]
        )
        model.NotR = pyo.Constraint(
            expr=model.x[self.mapping[hw_symbols.Reff["Not"]]]
            <= self.initial_params["Reff_Not"]
        )
        model.AddC = pyo.Constraint(
            expr=model.x[self.mapping[hw_symbols.Ceff["Add"]]]
            <= self.initial_params["Ceff_Add"]
        )
        model.RegsC = pyo.Constraint(
            expr=model.x[self.mapping[hw_symbols.Ceff["Regs"]]]
            <= self.initial_params["Ceff_Regs"]
        )
        return model

    def add_regularization_to_objective(self, model, l=1):
        """
        Parameters:
        model: pyomo model
        l: regularization hyperparameter
        """
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
        model.scaling_factor[model.obj] = self.obj_scale
        for s in self.free_symbols:
            if s.name in self.initial_params and self.initial_params[s.name] != 0:
                print(s.name, self.mapping)
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
        print(self.expr_symbols)
        print("edp:", edp)
        print("initial val:", self.initial_val)


        model.nVars = pyo.Param(initialize=len(edp.free_symbols))
        model.N = pyo.RangeSet(model.nVars)
        model.x = pyo.Var(model.N, domain=pyo.NonNegativeReals)
        self.mapping = {}

        i = 0
        for j in model.x:
            self.mapping[self.free_symbols[i]] = j
            print("x[{index}]".format(index=j), self.free_symbols[i])
            i += 1

        m = MyPyomoSympyBimap()
        for symbol in edp.free_symbols:
            # create self.mapping of sympy symbols to pyomo symbols
            m.sympy2pyomo[symbol] = model.x[self.mapping[symbol]]
            # give pyomo symbols an inital value for warm start
            model.x[self.mapping[symbol]] = self.expr_symbols[symbol]
            print(symbol, self.expr_symbols[symbol])
       
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
