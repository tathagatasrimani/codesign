class DesignSpace:
    def __init__(self, expr, symbols, diffs) -> None:
        self.functions=[]
        # in this case the variable we want to optimize 
        # is the same as the variable we use in simulation, so no extra function needed
        self.expr=expr
        self.symbols=symbols
        self.diffs=diffs
        self.initials=[1 for i in range(self.symbols)]
        
    
    def solve(self):
        # can't directly solve this because of Heaviside functions
        # just try to make all diffs equal to 0
        # expr_symbols_with_cost.subs(expr_symbols)