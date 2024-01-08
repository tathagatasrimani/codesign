import pyomo.environ as pyo
import hw_symbols
from MyPyomoSympyBimap import MyPyomoSympyBimap
import pyomo.core.expr.sympy_tools as sympy_tools
from pyomo.opt import SolverFactory

class Preprocessor:
    def __init__(self):
        self.mapping = {}
        self.py_exp = None
        self.initial_val = 0
        self.expr_symbols = {}
        self.free_symbols = []

    def const1(self, model):
        return model.x[self.mapping[hw_symbols.f]]>=1e6 
    def const2(self, model):
        return model.x[self.mapping[hw_symbols.V_dd]]>=0.5 
    def const3(self, model):
        return model.x[self.mapping[hw_symbols.R_tr]]>=1e3 
    def const4(self, model):
        return model.x[self.mapping[hw_symbols.I_leak]]<=2e-9 
    def const5(self, model):
        return model.x[self.mapping[hw_symbols.C_eff]]>=1e-15 
    def const6(self, model):
        return model.x[self.mapping[hw_symbols.C_tr]]>=1e-13 
    def const7(self, model):
        return model.x[self.mapping[hw_symbols.V_dd]]<=1.7 

    def add_constraints(self, model):
        model.Constraint = pyo.Constraint( expr = self.py_exp <= self.initial_val/5)
        model.Constraint1 = pyo.Constraint( expr = self.py_exp >= self.initial_val/100)
        model.freq_const = pyo.Constraint( rule=self.const1)
        model.V_dd_lower = pyo.Constraint( rule=self.const2)
        model.R_tr_const = pyo.Constraint( rule=self.const3)
        model.I_leak_const = pyo.Constraint( rule=self.const4)
        model.C_eff_const = pyo.Constraint( rule=self.const5)
        model.C_tr_const = pyo.Constraint( rule=self.const6)
        model.V_dd_upper = pyo.Constraint( rule=self.const7)

    def get_solver(self):
        opt = SolverFactory('ipopt')
        opt.options['warm_start_init_point'] = 'yes'
        #opt.options['warm_start_bound_push'] = 1e-9
        #opt.options['warm_start_mult_bound_push'] = 1e-9
        #opt.options['warm_start_bound_frac'] = 1e-9
        #opt.options['warm_start_slack_bound_push'] = 1e-9
        #opt.options['warm_start_slack_bound_frac'] = 1e-9
        #opt.options['mu_init'] = 0.1
        #opt.options['acceptable_obj_change_tol'] = 0.5
        #opt.options['tol'] = 0.5
        #opt.options['print_level'] = 5
        #opt.options['nlp_scaling_method'] = 'none'
        opt.options['max_iter'] = 10000
        opt.options['output_file'] = 'solver_out.txt'
        opt.options['wantsol'] = 2
        return opt

    def begin(self, model, simulator):
        self.expr_symbols = {}
        self.free_symbols = []
        for s in simulator.edp.free_symbols:
            self.free_symbols.append(s)
            if s.name in simulator.initial_params:
                self.expr_symbols[s] = simulator.initial_params[s.name]

        self.initial_val = simulator.edp.subs(self.expr_symbols)
        print(self.expr_symbols)
        print("edp equation: ", simulator.edp)

        model.nVars = pyo.Param(initialize=len(simulator.edp.free_symbols))
        model.N = pyo.RangeSet(model.nVars)
        model.x = pyo.Var(model.N, domain=pyo.NonNegativeReals)
        self.mapping = {}
        #model.add_component("y", pyo.Var(model.N, domain=pyo.NonNegativeIntegers))
        #model.y = pyo.Var(model.N, domain=pyo.NonNegativeIntegers)

        i = 0
        for j in model.x:
            self.mapping[self.free_symbols[i]] = j
            print("x[{index}]".format(index=j), self.free_symbols[i])
            i += 1

        m = MyPyomoSympyBimap()
        for symbol in simulator.edp.free_symbols:
            # create self.mapping of sympy symbols to pyomo symbols
            m.sympy2pyomo[symbol] = model.x[self.mapping[symbol]]
            # give pyomo symbols an inital value for warm start
            model.x[self.mapping[symbol]] = self.expr_symbols[symbol]
            print(symbol, self.expr_symbols[symbol])
        #sympy_tools._operatorMap.update({sympy.Max: lambda x: nested_if(x[0], x[1:])})
        #print(self.mapping.sympyVars())
        self.py_exp = sympy_tools.sympy2pyomo_expression(simulator.edp, m)
        # py_exp = sympy_tools.sympy2pyomo_expression(hardwareModel.symbolic_latency["Add"] ** (1/2), m)
        print(self.py_exp)
        model.obj = pyo.Objective(expr=self.py_exp, sense=pyo.minimize)
        model.cuts = pyo.ConstraintList()

        # Obtain dual solutions from first solve and send to warm start
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

        model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        model.scaling_factor[model.obj] = 100 # scale the objective
        #model.scaling_factor[model.Constraint] = 100 # scale the constraint
        model.scaling_factor[model.x[self.mapping[hw_symbols.f]]] = 1e-5
        model.scaling_factor[model.x[self.mapping[hw_symbols.V_dd]]] = 1 # scale the x variable
        model.scaling_factor[model.x[self.mapping[hw_symbols.R_tr]]] = 1e-2
        model.scaling_factor[model.x[self.mapping[hw_symbols.I_leak]]] = 1e9
        model.scaling_factor[model.x[self.mapping[hw_symbols.C_eff]]] = 1e15
        model.scaling_factor[model.x[self.mapping[hw_symbols.C_tr]]] = 1e13
        

        print(self.mapping)
        scaled_model = pyo.TransformationFactory('core.scale_model').create_using(model)
        scaled_preproc_model = pyo.TransformationFactory('contrib.constraints_to_var_bounds').create_using(scaled_model)
        preproc_model = pyo.TransformationFactory('contrib.constraints_to_var_bounds').create_using(model)
        opt = self.get_solver()
        return opt, scaled_preproc_model, preproc_model