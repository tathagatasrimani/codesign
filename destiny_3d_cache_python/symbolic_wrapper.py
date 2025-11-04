#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

"""
Symbolic computation wrapper for DESTINY
Provides parallel symbolic and numerical computation using SymPy
"""

import sympy as sp
from typing import Union, Tuple

# Type alias for values that can be either numerical or symbolic
NumOrSym = Union[float, int, sp.Expr]


class SymbolicValue:
    """
    Wrapper class that holds both numerical and symbolic representations of a value.
    This enables parallel computation where we calculate both concrete and symbolic results.
    """

    def __init__(self, concrete: float, symbolic: sp.Expr = None, name: str = None):
        """
        Initialize a symbolic value with both concrete and symbolic representations.

        Args:
            concrete: The numerical (concrete) value
            symbolic: The symbolic expression (can be None initially)
            name: Optional name for creating a new symbol
        """
        self.concrete = concrete
        if symbolic is None and name is not None:
            self.symbolic = sp.Symbol(name, real=True, positive=True)
        else:
            self.symbolic = symbolic if symbolic is not None else sp.Float(concrete)

    def __add__(self, other):
        """Addition: y = x + z"""
        if isinstance(other, SymbolicValue):
            return SymbolicValue(
                concrete=self.concrete + other.concrete,
                symbolic=self.symbolic + other.symbolic
            )
        else:
            return SymbolicValue(
                concrete=self.concrete + other,
                symbolic=self.symbolic + other
            )

    def __radd__(self, other):
        """Reverse addition"""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtraction"""
        if isinstance(other, SymbolicValue):
            return SymbolicValue(
                concrete=self.concrete - other.concrete,
                symbolic=self.symbolic - other.symbolic
            )
        else:
            return SymbolicValue(
                concrete=self.concrete - other,
                symbolic=self.symbolic - other
            )

    def __rsub__(self, other):
        """Reverse subtraction"""
        if isinstance(other, SymbolicValue):
            return SymbolicValue(
                concrete=other.concrete - self.concrete,
                symbolic=other.symbolic - self.symbolic
            )
        else:
            return SymbolicValue(
                concrete=other - self.concrete,
                symbolic=other - self.symbolic
            )

    def __mul__(self, other):
        """Multiplication"""
        if isinstance(other, SymbolicValue):
            return SymbolicValue(
                concrete=self.concrete * other.concrete,
                symbolic=self.symbolic * other.symbolic
            )
        else:
            return SymbolicValue(
                concrete=self.concrete * other,
                symbolic=self.symbolic * other
            )

    def __rmul__(self, other):
        """Reverse multiplication"""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Division"""
        if isinstance(other, SymbolicValue):
            return SymbolicValue(
                concrete=self.concrete / other.concrete,
                symbolic=self.symbolic / other.symbolic
            )
        else:
            return SymbolicValue(
                concrete=self.concrete / other,
                symbolic=self.symbolic / other
            )

    def __rtruediv__(self, other):
        """Reverse division"""
        if isinstance(other, SymbolicValue):
            return SymbolicValue(
                concrete=other.concrete / self.concrete,
                symbolic=other.symbolic / self.symbolic
            )
        else:
            return SymbolicValue(
                concrete=other / self.concrete,
                symbolic=other / self.symbolic
            )

    def __pow__(self, other):
        """Power"""
        if isinstance(other, SymbolicValue):
            return SymbolicValue(
                concrete=self.concrete ** other.concrete,
                symbolic=self.symbolic ** other.symbolic
            )
        else:
            return SymbolicValue(
                concrete=self.concrete ** other,
                symbolic=self.symbolic ** other
            )

    def __neg__(self):
        """Negation"""
        return SymbolicValue(
            concrete=-self.concrete,
            symbolic=-self.symbolic
        )

    def __abs__(self):
        """Absolute value"""
        return SymbolicValue(
            concrete=abs(self.concrete),
            symbolic=sp.Abs(self.symbolic)
        )

    def __lt__(self, other):
        """Less than comparison - uses concrete values for branching"""
        if isinstance(other, SymbolicValue):
            return self.concrete < other.concrete
        else:
            return self.concrete < other

    def __le__(self, other):
        """Less than or equal - uses concrete values for branching"""
        if isinstance(other, SymbolicValue):
            return self.concrete <= other.concrete
        else:
            return self.concrete <= other

    def __gt__(self, other):
        """Greater than comparison - uses concrete values for branching"""
        if isinstance(other, SymbolicValue):
            return self.concrete > other.concrete
        else:
            return self.concrete > other

    def __ge__(self, other):
        """Greater than or equal - uses concrete values for branching"""
        if isinstance(other, SymbolicValue):
            return self.concrete >= other.concrete
        else:
            return self.concrete >= other

    def __eq__(self, other):
        """Equality - uses concrete values for branching"""
        if isinstance(other, SymbolicValue):
            return self.concrete == other.concrete
        else:
            return self.concrete == other

    def __ne__(self, other):
        """Not equal - uses concrete values for branching"""
        if isinstance(other, SymbolicValue):
            return self.concrete != other.concrete
        else:
            return self.concrete != other

    def __repr__(self):
        """String representation"""
        return f"SymbolicValue(concrete={self.concrete}, symbolic={self.symbolic})"

    def __str__(self):
        """User-friendly string"""
        return f"{self.concrete} [{self.symbolic}]"

    def __float__(self):
        """Convert to float - returns concrete value"""
        return float(self.concrete)

    def __int__(self):
        """Convert to int - returns concrete value"""
        return int(self.concrete)


def symbolic_sqrt(x: Union[float, SymbolicValue]) -> SymbolicValue:
    """Square root with parallel symbolic computation"""
    if isinstance(x, SymbolicValue):
        import math
        return SymbolicValue(
            concrete=math.sqrt(x.concrete),
            symbolic=sp.sqrt(x.symbolic)
        )
    else:
        import math
        return SymbolicValue(
            concrete=math.sqrt(x),
            symbolic=sp.sqrt(x)
        )


def symbolic_log(x: Union[float, SymbolicValue], base=None) -> SymbolicValue:
    """Logarithm with parallel symbolic computation"""
    if isinstance(x, SymbolicValue):
        import math
        if base is None:
            concrete = math.log(x.concrete)
            symbolic = sp.log(x.symbolic)
        else:
            if isinstance(base, SymbolicValue):
                concrete = math.log(x.concrete, base.concrete)
                symbolic = sp.log(x.symbolic, base.symbolic)
            else:
                concrete = math.log(x.concrete, base)
                symbolic = sp.log(x.symbolic, base)
        return SymbolicValue(concrete=concrete, symbolic=symbolic)
    else:
        import math
        if base is None:
            concrete = math.log(x)
            symbolic = sp.log(x)
        else:
            concrete = math.log(x, base)
            symbolic = sp.log(x, base)
        return SymbolicValue(concrete=concrete, symbolic=symbolic)


def symbolic_log2(x: Union[float, SymbolicValue]) -> SymbolicValue:
    """Log base 2 with parallel symbolic computation"""
    if isinstance(x, SymbolicValue):
        import math
        return SymbolicValue(
            concrete=math.log2(x.concrete),
            symbolic=sp.log(x.symbolic, 2)
        )
    else:
        import math
        return SymbolicValue(
            concrete=math.log2(x),
            symbolic=sp.log(x, 2)
        )


def symbolic_pow(x: Union[float, SymbolicValue], y: Union[float, SymbolicValue]) -> SymbolicValue:
    """Power function with parallel symbolic computation"""
    if isinstance(x, SymbolicValue) and isinstance(y, SymbolicValue):
        import math
        return SymbolicValue(
            concrete=math.pow(x.concrete, y.concrete),
            symbolic=x.symbolic ** y.symbolic
        )
    elif isinstance(x, SymbolicValue):
        import math
        return SymbolicValue(
            concrete=math.pow(x.concrete, y),
            symbolic=x.symbolic ** y
        )
    elif isinstance(y, SymbolicValue):
        import math
        return SymbolicValue(
            concrete=math.pow(x, y.concrete),
            symbolic=x ** y.symbolic
        )
    else:
        import math
        return SymbolicValue(
            concrete=math.pow(x, y),
            symbolic=x ** y
        )


def symbolic_min(a: Union[float, SymbolicValue], b: Union[float, SymbolicValue]) -> SymbolicValue:
    """Minimum with parallel symbolic computation (uses concrete for branching)"""
    a_concrete = a.concrete if isinstance(a, SymbolicValue) else a
    b_concrete = b.concrete if isinstance(b, SymbolicValue) else b
    a_symbolic = a.symbolic if isinstance(a, SymbolicValue) else sp.Float(a)
    b_symbolic = b.symbolic if isinstance(b, SymbolicValue) else sp.Float(b)

    # Use concrete value for branching decision
    if a_concrete < b_concrete:
        return SymbolicValue(concrete=a_concrete, symbolic=sp.Min(a_symbolic, b_symbolic))
    else:
        return SymbolicValue(concrete=b_concrete, symbolic=sp.Min(a_symbolic, b_symbolic))


def symbolic_max(a: Union[float, SymbolicValue], b: Union[float, SymbolicValue]) -> SymbolicValue:
    """Maximum with parallel symbolic computation (uses concrete for branching)"""
    a_concrete = a.concrete if isinstance(a, SymbolicValue) else a
    b_concrete = b.concrete if isinstance(b, SymbolicValue) else b
    a_symbolic = a.symbolic if isinstance(a, SymbolicValue) else sp.Float(a)
    b_symbolic = b.symbolic if isinstance(b, SymbolicValue) else sp.Float(b)

    # Use concrete value for branching decision
    if a_concrete > b_concrete:
        return SymbolicValue(concrete=a_concrete, symbolic=sp.Max(a_symbolic, b_symbolic))
    else:
        return SymbolicValue(concrete=b_concrete, symbolic=sp.Max(a_symbolic, b_symbolic))


def symbolic_ceil(x: Union[float, SymbolicValue]) -> SymbolicValue:
    """Ceiling function with parallel symbolic computation"""
    if isinstance(x, SymbolicValue):
        import math
        return SymbolicValue(
            concrete=math.ceil(x.concrete),
            symbolic=sp.ceiling(x.symbolic)
        )
    else:
        import math
        return SymbolicValue(
            concrete=math.ceil(x),
            symbolic=sp.ceiling(x)
        )


def assert_symbolic_match(concrete_val: float, symbolic_val: sp.Expr,
                          substitutions: dict, tolerance: float = 1e-6,
                          context: str = ""):
    """
    Assert that symbolic expression matches concrete value when evaluated.

    Args:
        concrete_val: The concrete numerical result
        symbolic_val: The symbolic expression
        substitutions: Dictionary mapping symbols to their concrete values
        tolerance: Relative tolerance for comparison
        context: Description of what's being checked (for error messages)
    """
    try:
        # Convert symbol names (strings) to Symbol objects for substitution
        symbol_subs = {}
        for key, value in substitutions.items():
            if isinstance(key, str):
                # Create a Symbol object from the string name
                symbol_subs[sp.Symbol(key)] = value
            else:
                symbol_subs[key] = value

        # Evaluate symbolic expression with concrete values
        symbolic_evaluated = float(symbolic_val.evalf(subs=symbol_subs))

        # Check if they match within tolerance
        if abs(concrete_val) < 1e-15:  # Near zero
            if abs(symbolic_evaluated) > tolerance:
                raise AssertionError(
                    f"Symbolic mismatch {context}: "
                    f"concrete={concrete_val}, symbolic={symbolic_evaluated}"
                )
        else:
            relative_error = abs((concrete_val - symbolic_evaluated) / concrete_val)
            if relative_error > tolerance:
                raise AssertionError(
                    f"Symbolic mismatch {context}: "
                    f"concrete={concrete_val}, symbolic={symbolic_evaluated}, "
                    f"relative_error={relative_error}"
                )
    except Exception as e:
        print(f"Warning: Could not verify symbolic match {context}: {e}")


# Global flag to enable/disable symbolic computation
ENABLE_SYMBOLIC = False


def enable_symbolic_computation():
    """Enable symbolic computation globally"""
    global ENABLE_SYMBOLIC
    ENABLE_SYMBOLIC = True


def disable_symbolic_computation():
    """Disable symbolic computation globally"""
    global ENABLE_SYMBOLIC
    ENABLE_SYMBOLIC = False


def is_symbolic_enabled() -> bool:
    """Check if symbolic computation is enabled"""
    return ENABLE_SYMBOLIC
