#!/usr/bin/env python3
"""
Display all available symbolic equations
Shows how to access and use them
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from symbolic_expressions import MemoryAccessTimeExpressions, MemoryAccessTimeSymbols
from sympy import latex, pretty


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def main():
    print("="*80)
    print("ALL SYMBOLIC EQUATIONS FOR MEMORY ACCESS TIME")
    print("="*80)

    # Create expression and symbol objects
    exprs = MemoryAccessTimeExpressions()
    syms = MemoryAccessTimeSymbols()

    # =========================================================================
    # 1. Show all available symbols
    # =========================================================================
    print_section("STEP 1: Available Symbolic Variables")

    print("\nThese are the variables you can use in expressions:\n")

    symbol_descriptions = {
        'rows': 'Number of rows in subarray',
        'cols': 'Number of columns in subarray',
        'R_per_cell': 'Wire resistance per cell (Ω)',
        'C_per_cell': 'Wire capacitance per cell (F)',
        'R_access': 'Cell access transistor resistance (Ω)',
        'C_access': 'Cell access transistor capacitance (F)',
        'R_pulldown': 'SRAM pull-down transistor resistance (Ω)',
        'C_mux': 'Multiplexer capacitance (F)',
        'V_precharge': 'Bitline precharge voltage (V)',
        'V_sense': 'Sense amplifier sensitivity (V)',
        'W': 'Transistor width (m)',
        'R_eff': 'Effective resistance coefficient',
        'V_dd': 'Supply voltage (V)',
        'I_on': 'On-current (A)',
        'C_gate': 'Gate capacitance (F)',
        'C_wire': 'Wire capacitance (F)',
        'C_load': 'Load capacitance (F)',
        'V_swing': 'Voltage swing (V)',
        'I_amp': 'Amplifier current (A)',
        'W_pass': 'Pass transistor width (m)',
    }

    for attr in dir(syms):
        if not attr.startswith('_'):
            sym = getattr(syms, attr)
            desc = symbol_descriptions.get(attr, 'Parameter')
            print(f"  {str(sym):<20} - {desc}")

    # =========================================================================
    # 2. Show all available expressions
    # =========================================================================
    print_section("STEP 2: Available Symbolic Expressions")

    print("\nThese are the pre-built expressions you can use:\n")

    expression_list = [
        ('R_bitline', 'Bitline total resistance'),
        ('C_bitline', 'Bitline total capacitance'),
        ('tau_bitline', 'Bitline RC time constant'),
        ('tau_bitline_with_log', 'Bitline tau including voltage swing'),
        ('tau_bitline_expanded', 'Bitline tau (expanded form)'),
        ('R_decoder_stage', 'Decoder stage resistance'),
        ('C_decoder_stage', 'Decoder stage capacitance'),
        ('t_decoder_stage', 'Decoder stage delay'),
        ('t_senseamp', 'Sense amplifier delay'),
        ('R_mux_pass', 'Mux pass transistor resistance'),
        ('t_mux_level', 'Mux level delay'),
        ('t_total_symbolic', 'Total access time (symbolic)'),
    ]

    for expr_name, description in expression_list:
        if hasattr(exprs, expr_name):
            print(f"  {expr_name:<30} - {description}")

    # =========================================================================
    # 3. Display key expressions in detail
    # =========================================================================
    print_section("STEP 3: Key Expressions in Detail")

    # Bitline Resistance
    print("\n📐 Bitline Resistance:")
    print(f"   Expression: {exprs.R_bitline}")
    print(f"   LaTeX: {latex(exprs.R_bitline)}")
    print(f"   Meaning: Total resistance = per-cell resistance × number of rows")

    # Bitline Capacitance
    print("\n📐 Bitline Capacitance:")
    print(f"   Expression: {exprs.C_bitline}")
    print(f"   LaTeX: {latex(exprs.C_bitline)}")
    print(f"   Meaning: Total capacitance = per-cell capacitance × number of rows")

    # Bitline Time Constant (tau)
    print("\n📐 Bitline Time Constant (tau):")
    print(f"   Expression: {exprs.tau_bitline}")
    print(f"   LaTeX: {latex(exprs.tau_bitline)}")
    print(f"   Meaning: RC delay combining access resistance, pull-down, and bitline")

    # Tau with logarithmic factor
    print("\n📐 Bitline Time Constant with Voltage Swing:")
    print(f"   Expression: {exprs.tau_bitline_with_log}")
    print(f"   LaTeX (shortened): tau × log(V_pre / (V_pre - V_sense/2))")
    print(f"   Meaning: Accounts for partial voltage swing required for sensing")

    # =========================================================================
    # 4. Show how to use expressions
    # =========================================================================
    print_section("STEP 4: How to Use These Expressions")

    print("""
1. ACCESS AN EXPRESSION:
   ----------------------
   from symbolic_expressions import MemoryAccessTimeExpressions

   exprs = MemoryAccessTimeExpressions()
   tau = exprs.tau_bitline
   print(tau)  # Shows: R_per_cell*rows*(C_mux + C_per_cell*rows/2) + ...

2. SUBSTITUTE VALUES:
   ------------------
   from sympy import symbols

   # Define parameter values
   param_values = {
       exprs.symbols.rows: 1024,
       exprs.symbols.R_per_cell: 1.432e-3,
       exprs.symbols.C_per_cell: 2.789e-16,
       exprs.symbols.R_access: 32601.0,
       exprs.symbols.R_pulldown: 20532.0,
       exprs.symbols.C_access: 9.664e-17,
       exprs.symbols.C_mux: 0.0
   }

   # Substitute
   tau_value = tau.subs(param_values)
   print(float(tau_value))  # Gets actual number

3. EXPORT TO LATEX:
   ----------------
   from sympy import latex

   latex_str = latex(exprs.tau_bitline)
   print(latex_str)  # Use in LaTeX documents/papers

4. ANALYZE DEPENDENCIES:
   ---------------------
   from sympy import diff

   # How does tau change with rows?
   dtau_drows = diff(exprs.tau_bitline, exprs.symbols.rows)
   print(dtau_drows)  # Shows sensitivity to rows

5. SIMPLIFY/EXPAND:
   ----------------
   from sympy import simplify, expand

   simplified = simplify(exprs.tau_bitline)
   expanded = expand(exprs.tau_bitline)
   print(simplified)
   print(expanded)

6. EXPORT TO FILE:
   ---------------
   import json
   from sympy import latex

   equations = {}
   for name in ['R_bitline', 'C_bitline', 'tau_bitline']:
       expr = getattr(exprs, name)
       equations[name] = {
           'expression': str(expr),
           'latex': latex(expr),
           'symbols': [str(s) for s in expr.free_symbols]
       }

   with open('equations.json', 'w') as f:
       json.dump(equations, f, indent=2)
""")

    # =========================================================================
    # 5. Practical example
    # =========================================================================
    print_section("STEP 5: Complete Example")

    print("""
Complete example with real values:

```python
from symbolic_expressions import MemoryAccessTimeExpressions
import globals as g
from InputParameter import InputParameter
from Technology import Technology
from MemCell import MemCell
from Wire import Wire
from typedef import WireType, WireRepeaterType
import os

# Initialize DESTINY to get parameter values
g.inputParameter = InputParameter()
g.inputParameter.ReadInputParameterFromFile("config/sample_SRAM_2layer.cfg")

g.tech = Technology()
g.tech.Initialize(g.inputParameter.processNode,
                 g.inputParameter.deviceRoadmap, g.inputParameter)

g.devtech = Technology()
g.devtech.Initialize(g.inputParameter.processNode,
                    g.inputParameter.deviceRoadmap, g.inputParameter)

g.localWire = Wire()
g.localWire.Initialize(g.inputParameter.processNode, WireType.local_aggressive,
                      WireRepeaterType.repeated_none,
                      g.inputParameter.temperature, False)

g.cell = MemCell()
if len(g.inputParameter.fileMemCell) > 0:
    cellFile = g.inputParameter.fileMemCell[0]
    if '/' not in cellFile:
        cellFile = os.path.join('config', cellFile)
    g.cell.ReadCellFromFile(cellFile)

# Get symbolic expressions
exprs = MemoryAccessTimeExpressions()

# Calculate parameter values
R_per_cell = (g.localWire.resWirePerUnit *
              g.cell.heightInFeatureSize * g.devtech.featureSize)
C_per_cell = (g.localWire.capWirePerUnit *
              g.cell.heightInFeatureSize * g.devtech.featureSize)

# Substitute
R_bitline_val = exprs.R_bitline.subs({
    exprs.symbols.rows: 1024,
    exprs.symbols.R_per_cell: R_per_cell
})

print(f"R_bitline = {float(R_bitline_val):.3f} Ω")
# Output: R_bitline = 1466.075 Ω
```
""")

    # =========================================================================
    # 6. Summary
    # =========================================================================
    print_section("SUMMARY")

    print("""
📚 HOW TO ACCESS ALL EQUATIONS:

1. Import the module:
   from symbolic_expressions import MemoryAccessTimeExpressions

2. Create expression object:
   exprs = MemoryAccessTimeExpressions()

3. Access any expression:
   tau = exprs.tau_bitline
   R = exprs.R_bitline
   C = exprs.C_bitline

4. See all available expressions:
   for attr in dir(exprs):
       if not attr.startswith('_') and attr != 'symbols':
           print(attr, getattr(exprs, attr))

5. Export to LaTeX:
   from sympy import latex
   print(latex(exprs.tau_bitline))

6. Substitute values:
   result = exprs.tau_bitline.subs({...})

📁 KEY FILES:
   - symbolic_expressions.py - All symbolic expressions
   - final_sram_2layer_model.json - Pre-computed model
   - SYMBOLIC_VS_NUMERICAL_GUIDE.md - This guide
   - walkthrough_symbolic_vs_numerical.py - Example usage

🎯 USE CASES:
   ✅ Design space exploration
   ✅ Parameter sensitivity analysis
   ✅ Export to papers/documentation
   ✅ Understanding design trade-offs
   ✅ Building optimization frameworks
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
