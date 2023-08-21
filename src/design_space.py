from sympy import *
class DesignSpace:
    def __init__(self, expr, symbols, diffs) -> None:
        self.functions=[]
        # in this case the variable we want to optimize 
        # is the same as the variable we use in simulation, so no extra function needed
        self.expr=expr
        self.symbols=symbols
        self.diffs=diffs
        self.initials=[1 for i in range(len(self.symbols))]
        
    
    def solve(self):
        # can't directly solve this because of Heaviside functions
        # just try to make all diffs equal to 0
        # expr_symbols_with_cost.subs(expr_symbols)
        print("self.diffs", self.diffs)
        
        print("self.symbols", self.symbols)
        import numpy as np
        # modules = [{'Heaviside': lambda x: np.heaviside(x, 1)}, 'numpy']
        modules = ['numpy']
        # focx_lambda = lambdify((self.symbols[0], self.symbols[1]), self.expr.subs({self.symbols[0].name: self.symbols[0]}), modules=modules)
        # focy_lambda = lambdify((self.symbols[0], self.symbols[1]), self.expr, modules=modules)
        # print(focx_lambda(0.3, 0.4))  # we need to check that the lambdify works, so this should print a floating point number
        # print(focy_lambda(0.3, 0.4))
        # f_lambda = lambdify((self.symbols[0], self.symbols[1]), self.expr, modules=modules)
        # print(f_lambda(0.3, 0.4))
        f0_lambda = lambdify((self.symbols[0], self.symbols[1]), self.diffs[0], modules=modules)
        f1_lambda = lambdify((self.symbols[0], self.symbols[1]), self.diffs[1], modules=modules)
        # print("self.expr", self.expr)
        # print("self.diffs[0]", self.diffs[0])
        # print(f0_lambda(0.3, 0.4))  # we need to check that the lambdify works, so this should print a floating point number
        # # print(f1_lambda(0.3, 0.4))
        
        from scipy.optimize import fsolve
        def equations(p):
            x, y = p
            print("x, y", x, y)
            return [f0_lambda(x, y), f1_lambda(x, y)]

        sol = fsolve(equations, [0, 0])
        # start 1 1
        # -10.5242741229133
        # -2.34805725514641
        # 144.450784309381
        # start [0, 0]
        # 9.76996261670138e-15
        # 1.11910480882216e-13
        # 134.878365760103
        print(sol)  # [0.64701372 0.61726372]
        print("self.expr", self.expr)
        data_dict = {self.symbols[0]:sol[0], self.symbols[1]:sol[1]}
        print(data_dict)
        print(self.diffs[0].subs(data_dict))
        print(self.diffs[1].subs(data_dict))
        
        print(self.expr.subs(data_dict))
        

        # print(nsolve(self.diffs, self.symbols, self.initials, modules=modules))