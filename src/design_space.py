from sympy import *
class DesignSpace:
    def __init__(self, expr, symbols, diffs, initials) -> None:
        self.functions=[]
        # in this case the variable we want to optimize 
        # is the same as the variable we use in simulation, so no extra function needed
        self.expr=expr
        self.symbols=symbols
        self.diffs=diffs
        self.initials=initials
        
    
    def solve(self):
        # can't directly solve this because of Heaviside functions
        # just try to make all diffs equal to 0
        # expr_symbols_with_cost.subs(expr_symbols)
        import numpy as np
        modules = ['numpy']
        f0_lambda = lambdify((self.symbols[0], self.symbols[1]), self.diffs[0], modules=modules)
        f1_lambda = lambdify((self.symbols[0], self.symbols[1]), self.diffs[1], modules=modules)
        from scipy.optimize import fsolve
        def equations(p):
            x, y = p
            return [f0_lambda(x, y), f1_lambda(x, y)]

        sol = fsolve(equations, self.initials)
        data_dict = {self.symbols[0]:sol[0], self.symbols[1]:sol[1]}