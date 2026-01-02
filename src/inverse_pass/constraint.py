import sympy as sp

class Constraint:
    def __init__(self, constraint, label):
        self.constraint = constraint
        if isinstance(constraint, sp.Ge):
            self.slack = constraint.rhs - constraint.lhs
        elif isinstance(constraint, sp.Le):
            self.slack = constraint.lhs - constraint.rhs
        elif isinstance(constraint, sp.Eq):
            self.slack = 0.0
        else:
            raise ValueError(f"Invalid constraint type: {type(constraint)}")
        self.label = label