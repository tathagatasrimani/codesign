import pyomo.environ as pyo
import hw_symbols
from MyPyomoSympyBimap import MyPyomoSympyBimap
import pyomo.core.expr.sympy_tools as sympy_tools
from pyomo.opt import SolverFactory

scaling_factors = {
    hw_symbols.f: 1e-6,
    hw_symbols.V_dd: 1,
    hw_symbols.C_int_inv: 1e15,
    hw_symbols.C_input_inv: 1e13,
}
obj_scale = 1e12

class Preprocessor:
    def __init__(self):
        self.mapping = {}
        self.py_exp = None
        self.initial_val = 0
        self.expr_symbols = {}
        self.free_symbols = []
        self.vars = []
        self.model = None

    def f(self, model):
        return model.x[self.mapping[hw_symbols.f]]>=1e6 
    def V_dd_lower(self, model):
        return model.x[self.mapping[hw_symbols.V_dd]]>=0.5 
    def C_int_inv(self, model):
        return model.x[self.mapping[hw_symbols.C_int_inv]]>=1e-15 
    def C_input_inv(self, model):
        return model.x[self.mapping[hw_symbols.C_input_inv]]>=1e-13 
    def V_dd_upper(self, model):
        return model.x[self.mapping[hw_symbols.V_dd]]<=1.7 

    def add_constraints(self):
        self.model.Constraint = pyo.Constraint( expr = self.py_exp <= self.initial_val/10)
        self.model.Constraint1 = pyo.Constraint( expr = self.py_exp >= self.initial_val/10e2)
        self.model.freq_const = pyo.Constraint( rule=self.f)
        self.model.V_dd_lower = pyo.Constraint( rule=self.V_dd_lower)
        self.model.C_int_inv_constr = pyo.Constraint( rule=self.C_int_inv)
        self.model.C_input_inv_constr = pyo.Constraint( rule=self.C_input_inv)
        self.model.V_dd_upper = pyo.Constraint( rule=self.V_dd_upper)
        return self.model

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
    
    def create_scaling(self):
        self.model.scaling_factor[self.model.obj] = obj_scale
        for var in scaling_factors:
            self.model.scaling_factor[self.model.x[self.mapping[var]]] = scaling_factors[var]

    def begin(self, model, simulator):
        self.model = model
        self.expr_symbols = {}
        self.free_symbols = []
        for s in simulator.edp.free_symbols:
            self.free_symbols.append(s)
            if s.name in simulator.initial_params:
                self.expr_symbols[s] = simulator.initial_params[s.name]

        self.initial_val = simulator.edp.subs(self.expr_symbols)
        print(self.expr_symbols)
        print("edp equation: ", simulator.edp)

        self.model.nVars = pyo.Param(initialize=len(simulator.edp.free_symbols))
        self.model.N = pyo.RangeSet(self.model.nVars)
        self.model.x = pyo.Var(self.model.N, domain=pyo.NonNegativeReals)
        self.mapping = {}
        #self.model.add_component("y", pyo.Var(self.model.N, domain=pyo.NonNegativeIntegers))
        #self.model.y = pyo.Var(self.model.N, domain=pyo.NonNegativeIntegers)

        i = 0
        for j in self.model.x:
            self.mapping[self.free_symbols[i]] = j
            print("x[{index}]".format(index=j), self.free_symbols[i])
            i += 1

        m = MyPyomoSympyBimap()
        for symbol in simulator.edp.free_symbols:
            # create self.mapping of sympy symbols to pyomo symbols
            m.sympy2pyomo[symbol] = self.model.x[self.mapping[symbol]]
            # give pyomo symbols an inital value for warm start
            self.model.x[self.mapping[symbol]] = self.expr_symbols[symbol]
            print(symbol, self.expr_symbols[symbol])
        #sympy_tools._operatorMap.update({sympy.Max: lambda x: nested_if(x[0], x[1:])})
        #print(self.mapping.sympyVars())
        self.py_exp = sympy_tools.sympy2pyomo_expression(simulator.edp, m)
        # py_exp = sympy_tools.sympy2pyomo_expression(hardwareself.model.symbolic_latency["Add"] ** (1/2), m)
        print(self.py_exp)
        self.model.obj = pyo.Objective(expr=self.py_exp, sense=pyo.minimize)
        self.model.cuts = pyo.ConstraintList()

        # Obtain dual solutions from first solve and send to warm start
        self.model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

        self.model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        self.create_scaling()
        self.add_constraints()

        print(self.mapping)
        print(self.model)
        scaled_model = pyo.TransformationFactory('core.scale_model').create_using(self.model)
        scaled_preproc_model = pyo.TransformationFactory('contrib.constraints_to_var_bounds').create_using(scaled_model)
        preproc_model = pyo.TransformationFactory('contrib.constraints_to_var_bounds').create_using(self.model)
        opt = self.get_solver()
        return opt, scaled_preproc_model, preproc_model