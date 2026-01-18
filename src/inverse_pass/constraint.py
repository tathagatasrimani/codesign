import sympy as sp
import cvxpy as cp
import numpy as np

class Constraint:
    def __init__(self, constraint, label):
        self.constraint = constraint
        self.is_cvxpy = isinstance(constraint, cp.Constraint)
        if isinstance(constraint, sp.Ge):
            self.slack = constraint.rhs - constraint.lhs
        elif isinstance(constraint, sp.Le):
            self.slack = constraint.lhs - constraint.rhs
        elif isinstance(constraint, sp.Eq):
            self.slack = 0.0
        elif self.is_cvxpy:
            self.slack = 0 # set later
        else:
            raise ValueError(f"Invalid constraint type: {type(constraint)}")
        self.label = label

    def set_slack_cvxpy(self):
        """
        Set the current slack/violation value for this constraint.

        Returns:
            float: The slack value (positive means violation, negative means satisfied)
        """
        # For cvxpy, get the violation from the constraint
        # constraint.violation() returns how much the constraint is violated
        # We return negative violation as slack (positive = satisfied)
        val = self.constraint.violation()
        print(f"slack for {self.label}: {float(val)}")
        self.slack = float(val)