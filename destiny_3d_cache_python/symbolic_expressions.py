#!/usr/bin/env python3
"""
Symbolic Expressions Module

Provides SymPy expressions for memory access time components that can be:
1. Used by other parts of the framework
2. Evaluated with specific parameter values
3. Exported for analysis/optimization
4. Serialized for storage/transmission
"""

from sympy import symbols, log, simplify, lambdify
from typing import Dict, Any
import json


class MemoryAccessTimeSymbols:
    """
    Defines all symbolic variables used in memory access time modeling
    """
    def __init__(self):
        # Technology parameters
        self.V_dd = symbols('V_dd', positive=True, real=True)
        self.I_on = symbols('I_on', positive=True, real=True)
        self.R_eff = symbols('R_eff', positive=True, real=True)

        # Capacitance components
        self.C_gate = symbols('C_gate', positive=True, real=True)
        self.C_wire = symbols('C_wire', positive=True, real=True)
        self.C_junction = symbols('C_junction', positive=True, real=True)
        self.C_access = symbols('C_access', positive=True, real=True)
        self.C_mux = symbols('C_mux', positive=True, real=True)
        self.C_load = symbols('C_load', positive=True, real=True)

        # Resistance components
        self.R_access = symbols('R_access', positive=True, real=True)
        self.R_pulldown = symbols('R_pulldown', positive=True, real=True)
        self.R_pass = symbols('R_pass', positive=True, real=True)

        # Transistor sizing
        self.W = symbols('W', positive=True, real=True)
        self.W_pass = symbols('W_pass', positive=True, real=True)

        # Array geometry
        self.rows = symbols('rows', positive=True, integer=True)
        self.cols = symbols('cols', positive=True, integer=True)

        # Voltage parameters
        self.V_precharge = symbols('V_precharge', positive=True, real=True)
        self.V_sense = symbols('V_sense', positive=True, real=True)
        self.V_swing = symbols('V_swing', positive=True, real=True)

        # Current parameters
        self.I_amp = symbols('I_amp', positive=True, real=True)

        # Per-cell parameters
        self.R_per_cell = symbols('R_per_cell', positive=True, real=True)
        self.C_per_cell = symbols('C_per_cell', positive=True, real=True)


class MemoryAccessTimeExpressions:
    """
    SymPy expressions for memory access time components
    Can be used by other framework components
    """

    def __init__(self):
        self.symbols = MemoryAccessTimeSymbols()
        self._build_expressions()

    def _build_expressions(self):
        """Build all symbolic expressions"""
        s = self.symbols

        # Row Decoder expressions
        self.R_decoder_stage = s.R_eff * s.V_dd / (s.I_on * s.W)
        self.C_decoder_stage = s.C_gate + s.C_wire
        self.t_decoder_stage = self.R_decoder_stage * self.C_decoder_stage
        # Note: Total decoder = sum of stages (depends on number of stages)

        # Bitline expressions (per-cell components)
        self.R_bitline = s.R_per_cell * s.rows
        self.C_bitline = s.C_per_cell * s.rows

        # Bitline delay - SRAM full formula
        # τ = (R_access + R_pulldown) × (C_access + C_bitline + C_mux) +
        #     R_bitline × (C_mux + C_bitline/2)
        self.tau_bitline = (
            (s.R_access + s.R_pulldown) * (s.C_access + self.C_bitline + s.C_mux) +
            self.R_bitline * (s.C_mux + self.C_bitline / 2)
        )

        # With voltage sensing logarithm
        self.tau_bitline_with_log = self.tau_bitline * log(
            s.V_precharge / (s.V_precharge - s.V_sense / 2)
        )

        # Expanded form showing scaling terms
        # This expands to: A + B*rows + C*rows^2
        self.tau_bitline_expanded = simplify(self.tau_bitline)

        # Sense amplifier
        self.t_senseamp = s.V_swing * s.C_load / s.I_amp

        # Multiplexer
        self.R_mux_pass = s.R_eff * s.V_dd / (s.I_on * s.W_pass)
        self.t_mux_level = self.R_mux_pass * s.C_load

        # Total access time (symbolic sum)
        # Note: t_decoder and t_bitline need Horowitz transformation in practice
        # This is the idealized form
        self.t_total_symbolic = symbols('t_decoder') + symbols('t_bitline') + self.t_senseamp + symbols('t_mux')

    def get_expression(self, component: str):
        """
        Get a specific expression by name

        Args:
            component: Name of component (e.g., 'tau_bitline', 't_senseamp', etc.)

        Returns:
            SymPy expression
        """
        return getattr(self, component, None)

    def get_all_expressions(self) -> Dict[str, Any]:
        """
        Get all expressions as a dictionary

        Returns:
            Dictionary mapping component names to SymPy expressions
        """
        return {
            # Decoder
            'R_decoder_stage': self.R_decoder_stage,
            'C_decoder_stage': self.C_decoder_stage,
            't_decoder_stage': self.t_decoder_stage,

            # Bitline
            'R_bitline': self.R_bitline,
            'C_bitline': self.C_bitline,
            'tau_bitline': self.tau_bitline,
            'tau_bitline_with_log': self.tau_bitline_with_log,
            'tau_bitline_expanded': self.tau_bitline_expanded,

            # Senseamp
            't_senseamp': self.t_senseamp,

            # Mux
            'R_mux_pass': self.R_mux_pass,
            't_mux_level': self.t_mux_level,

            # Total
            't_total_symbolic': self.t_total_symbolic,
        }

    def evaluate(self, component: str, params: Dict[str, float]) -> float:
        """
        Evaluate an expression with specific parameter values

        Args:
            component: Name of component expression
            params: Dictionary of parameter values

        Returns:
            Numerical result
        """
        expr = self.get_expression(component)
        if expr is None:
            raise ValueError(f"Unknown component: {component}")

        # Get free symbols in the expression
        free_symbols = expr.free_symbols

        # Build substitution dictionary
        subs_dict = {}
        for sym in free_symbols:
            sym_name = str(sym)
            if sym_name in params:
                subs_dict[sym] = params[sym_name]

        # Substitute and evaluate
        result = expr.subs(subs_dict)

        # If still symbolic, can't evaluate
        if result.free_symbols:
            missing = [str(s) for s in result.free_symbols]
            raise ValueError(f"Missing parameters for evaluation: {missing}")

        return float(result)

    def to_latex(self, component: str) -> str:
        """
        Convert expression to LaTeX string

        Args:
            component: Name of component expression

        Returns:
            LaTeX representation
        """
        from sympy import latex
        expr = self.get_expression(component)
        if expr is None:
            raise ValueError(f"Unknown component: {component}")
        return latex(expr)

    def to_python_function(self, component: str):
        """
        Convert expression to Python callable function

        Args:
            component: Name of component expression

        Returns:
            Callable Python function
        """
        expr = self.get_expression(component)
        if expr is None:
            raise ValueError(f"Unknown component: {component}")

        # Get all symbols used in expression
        free_symbols = sorted(expr.free_symbols, key=lambda s: str(s))

        # Create lambda function
        return lambdify(free_symbols, expr, modules='numpy')

    def export_expressions(self, filename: str):
        """
        Export expressions to a file (as strings)

        Args:
            filename: Output filename
        """
        expressions = self.get_all_expressions()

        # Convert expressions to strings
        export_data = {
            name: {
                'str': str(expr),
                'latex': self.to_latex(name),
                'symbols': [str(s) for s in expr.free_symbols]
            }
            for name, expr in expressions.items()
        }

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)


class MemoryBlockSymbolicModel:
    """
    Complete symbolic model for a memory block
    Combines expressions with actual C++ DESTINY results
    """

    def __init__(self, cpp_config):
        """
        Initialize with C++ DESTINY configuration

        Args:
            cpp_config: OptimalConfiguration from parse_cpp_output
        """
        self.config = cpp_config
        self.expressions = MemoryAccessTimeExpressions()
        self.symbols = self.expressions.symbols

        # Store numerical results from C++ DESTINY
        self.numerical_results = {
            't_decoder': cpp_config.row_decoder_latency,
            't_bitline': cpp_config.bitline_latency,
            't_senseamp': cpp_config.senseamp_latency,
            't_mux': cpp_config.mux_latency,
            't_total': cpp_config.subarray_latency,
        }

        # Store configuration parameters
        self.config_params = {
            'rows': cpp_config.subarray_rows,
            'cols': cpp_config.subarray_cols,
            'num_banks_x': cpp_config.num_banks_x,
            'num_banks_y': cpp_config.num_banks_y,
            'num_stacks': cpp_config.num_stacks,
            'senseamp_mux': cpp_config.senseamp_mux,
            'output_mux_l1': cpp_config.output_mux_l1,
            'output_mux_l2': cpp_config.output_mux_l2,
        }

    def get_symbolic_expression(self, component: str):
        """
        Get symbolic expression for a component

        Args:
            component: Component name

        Returns:
            SymPy expression
        """
        return self.expressions.get_expression(component)

    def get_numerical_value(self, component: str) -> float:
        """
        Get numerical value from C++ DESTINY

        Args:
            component: Component name

        Returns:
            Numerical value in seconds
        """
        return self.numerical_results.get(component)

    def get_complete_model(self) -> Dict[str, Any]:
        """
        Get complete model with both symbolic and numerical data

        Returns:
            Dictionary containing expressions, numerical results, and config
        """
        return {
            'symbolic_expressions': self.expressions.get_all_expressions(),
            'numerical_results': self.numerical_results,
            'config_params': self.config_params,
        }

    def get_evaluated_expressions(self, params: dict) -> dict:
        """
        Evaluate symbolic expressions with actual parameter values
        (Uses xreplace for substitution as recommended)

        Args:
            params: Dictionary of parameter values (e.g., from extract_tech_params.py)
                   Keys should match symbolic variable names (rows, R_per_cell, etc.)

        Returns:
            Dictionary mapping expression names to numerical values (or None if missing params)
        """
        evaluated = {}
        all_exprs = self.expressions.get_all_expressions()

        for name, expr in all_exprs.items():
            try:
                value = self.expressions.evaluate(name, params)
                evaluated[name] = value
            except (ValueError, KeyError, TypeError) as e:
                # Missing parameters - skip this expression
                evaluated[name] = None

        return evaluated

    def export_to_json(self, filename: str):
        """
        Export complete model to JSON (for framework integration)

        Args:
            filename: Output filename
        """
        model = self.get_complete_model()

        # Convert SymPy expressions to strings for JSON serialization
        export_data = {
            'symbolic_expressions': {
                name: {
                    'expression': str(expr),
                    'latex': self.expressions.to_latex(name),
                    'symbols': [str(s) for s in expr.free_symbols]
                }
                for name, expr in model['symbolic_expressions'].items()
            },
            'numerical_results': model['numerical_results'],
            'config_params': model['config_params'],
        }

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Exported symbolic model to: {filename}")

    def __repr__(self):
        return f"MemoryBlockSymbolicModel(rows={self.config_params['rows']}, " \
               f"cols={self.config_params['cols']}, " \
               f"t_total={self.numerical_results['t_total']*1e9:.3f}ns)"


# Convenience function for framework integration
def create_symbolic_model_from_destiny_output(cpp_output_file: str):
    """
    Create a symbolic model from C++ DESTINY output

    Args:
        cpp_output_file: Path to C++ DESTINY output file

    Returns:
        MemoryBlockSymbolicModel instance
    """
    from parse_cpp_output import parse_cpp_destiny_output

    cpp_config = parse_cpp_destiny_output(cpp_output_file)
    return MemoryBlockSymbolicModel(cpp_config)


if __name__ == "__main__":
    # Example usage
    print("Memory Access Time Symbolic Expressions")
    print("=" * 60)

    # Create expressions
    exprs = MemoryAccessTimeExpressions()

    print("\nAvailable expressions:")
    for name in exprs.get_all_expressions().keys():
        print(f"  - {name}")

    print("\n📐 Example: Bitline delay expression")
    print(f"   tau_bitline = {exprs.tau_bitline}")

    print("\n📐 Example: Simplified bitline (expanded)")
    print(f"   tau_bitline_expanded = {exprs.tau_bitline_expanded}")

    print("\n📐 Example: Sense amplifier delay")
    print(f"   t_senseamp = {exprs.t_senseamp}")

    print("\n✓ These expressions can be:")
    print("  • Passed to other framework components")
    print("  • Evaluated with specific parameters")
    print("  • Exported to JSON")
    print("  • Converted to LaTeX")
    print("  • Converted to Python functions")
