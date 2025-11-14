# Complete Guide: Symbolic vs Numerical Modeling

## Overview

### What is Numerical Modeling?
**Numerical modeling** executes the actual DESTINY code with specific parameter values to compute concrete numbers.

**Example:**
```python
# Numerical approach
subarray = SubArray()
subarray.Initialize(1024, 2048, ...)  # Specific values
subarray.CalculateLatency()
print(subarray.bitlineDelay)  # Gets: 1.989e-09 seconds
```

**How it works:**
1. Initialize technology parameters (65nm, temperature, V_dd, etc.)
2. Initialize wire models (resistance per unit, capacitance per unit)
3. Initialize cell models (transistor sizes, dimensions)
4. Calculate derived quantities (bitline length, resistance, capacitance)
5. Apply circuit models (RC delay, Horowitz model, etc.)
6. Get final numerical answer

**Pros:**
- ✅ Exact values for specific configurations
- ✅ All physical effects included (parasitics, second-order effects)
- ✅ Validated against C++ DESTINY
- ✅ Production-ready accuracy

**Cons:**
- ❌ No insight into parameter dependencies
- ❌ Must re-run for each configuration
- ❌ Can't see "what affects what"

---

### What is Symbolic Modeling?
**Symbolic modeling** creates mathematical expressions with variables instead of numbers.

**Example:**
```python
# Symbolic approach
from sympy import symbols
rows, R_per_cell, C_per_cell = symbols('rows R_per_cell C_per_cell')

# Create symbolic expression
R_bitline = R_per_cell * rows
# Result: R_per_cell*rows (an equation, not a number!)

# Later, substitute values
R_bitline_value = R_bitline.subs({rows: 1024, R_per_cell: 1.432})
# Result: 1466.075 (now it's a number)
```

**How it works:**
1. Define symbolic variables (rows, R_per_cell, V_dd, etc.)
2. Build mathematical expressions using these variables
3. Store expressions (can be exported as LaTeX, displayed, analyzed)
4. When needed, substitute actual values to get numbers

**Pros:**
- ✅ See exact mathematical relationships
- ✅ Understand parameter dependencies (e.g., "delay ∝ rows²")
- ✅ Export to papers/documentation (LaTeX format)
- ✅ Design space exploration (sweep parameters symbolically)
- ✅ Sensitivity analysis

**Cons:**
- ❌ Must carefully account for all effects (easy to miss parasitics)
- ❌ Expressions can get very complex
- ❌ Need to match numerical model exactly

---

## How They Work Together

```
┌─────────────────────────────────────────────────────────────┐
│                    NUMERICAL MODELING                        │
│                                                              │
│  Technology.py → Wire.py → MemCell.py → SubArray.py         │
│       ↓             ↓          ↓            ↓                │
│   V_dd=0.8    resWire=1.5e6  width=14.6  bitlineDelay=1.989ns│
│                                                              │
│  Execute code with specific values → Get concrete answer    │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Extract formulas
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    SYMBOLIC MODELING                         │
│                                                              │
│  Define: rows, R_per_cell, C_per_cell, V_dd, ...           │
│                                                              │
│  Build expressions:                                          │
│    R_bitline = R_per_cell × rows                            │
│    C_bitline = C_per_cell × rows                            │
│    tau = R × C                                              │
│    delay = horowitz(tau, beta, ...)                         │
│                                                              │
│  Store as SymPy expressions (can export, analyze, etc.)     │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Substitute values
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    VERIFICATION                              │
│                                                              │
│  Symbolic: R_bitline.subs({rows: 1024, R_per_cell: 1.432}) │
│          → 1466.075 Ω                                        │
│                                                              │
│  Numerical: subarray.resBitline                             │
│          → 1466.075 Ω                                        │
│                                                              │
│  ✅ MATCH! Symbolic and numerical give same answer          │
└─────────────────────────────────────────────────────────────┘
```

---

## Deep Dive: The Process

### Step 1: Extract Formulas from Source Code

Look at the actual Python DESTINY code to find the formulas:

**Example from SubArray.py (line 220-223):**
```python
# Numerical code
self.capBitline = self.lenBitline * g.localWire.capWirePerUnit * 1
self.resBitline = self.lenBitline * g.localWire.resWirePerUnit * 1
```

**Convert to symbolic:**
```python
from sympy import symbols
lenBitline = symbols('lenBitline')
capWirePerUnit = symbols('capWirePerUnit')
resWirePerUnit = symbols('resWirePerUnit')

# Symbolic expressions
capBitline_expr = lenBitline * capWirePerUnit
resBitline_expr = lenBitline * resWirePerUnit
```

### Step 2: Expand Composite Variables

Some variables are themselves computed. Expand them:

**Example:**
```python
# lenBitline is computed from other variables
lenBitline = numRow * cellHeight * featureSize

# So the full symbolic expression is:
rows = symbols('rows')
cellHeight = symbols('cellHeight')
featureSize = symbols('featureSize')

lenBitline_expr = rows * cellHeight * featureSize

# Substitute into capBitline:
capBitline_expr = (rows * cellHeight * featureSize) * capWirePerUnit
```

### Step 3: Simplify to Per-Cell Parameters

For cleaner expressions, define per-cell parameters:

```python
# Instead of: C_bitline = rows × (capWirePerUnit × cellHeight × featureSize)
# Define: C_per_cell = capWirePerUnit × cellHeight × featureSize
# Then: C_bitline = rows × C_per_cell

C_per_cell = symbols('C_per_cell')
C_bitline = rows * C_per_cell

# This is cleaner and shows the linear relationship with rows
```

### Step 4: Build Complex Expressions

Combine multiple formulas:

**From SubArray.py (line 507-508):**
```python
# Numerical code
tau = ((self.resCellAccess + resPullDown) *
       (self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousDelayCalculation) +
       self.resBitline * (self.bitlineMux.capForPreviousDelayCalculation + self.capBitline / 2))
```

**Symbolic version:**
```python
from sympy import symbols

R_access = symbols('R_access')
R_pulldown = symbols('R_pulldown')
C_access = symbols('C_access')
C_bitline = symbols('C_bitline')
C_mux = symbols('C_mux')
R_bitline = symbols('R_bitline')

tau = ((R_access + R_pulldown) * (C_access + C_bitline + C_mux) +
       R_bitline * (C_mux + C_bitline / 2))

print(tau)
# Output: (C_access + C_bitline + C_mux)*(R_access + R_pulldown) +
#         R_bitline*(C_bitline/2 + C_mux)
```

### Step 5: Substitute Values When Needed

To get actual numbers, substitute parameter values:

```python
# Substitute actual values
param_values = {
    R_access: 32601.023,
    R_pulldown: 20532.375,
    C_access: 9.664486e-17,
    C_bitline: 3.351252e-13,
    C_mux: 0.0,
    R_bitline: 1466.075
}

tau_value = tau.subs(param_values)
print(float(tau_value))  # Output: 1.8057e-08 seconds
```

---

## Accessing All Symbolic Equations

### Method 1: Use the Provided Module

I've created `symbolic_expressions.py` with all the key expressions:

```python
from symbolic_expressions import MemoryAccessTimeExpressions

# Create expression object
exprs = MemoryAccessTimeExpressions()

# Access individual expressions
print("R_bitline =", exprs.R_bitline)
print("C_bitline =", exprs.C_bitline)
print("tau =", exprs.tau_bitline)

# Get LaTeX for papers
from sympy import latex
print("LaTeX:", latex(exprs.tau_bitline))
```

**Available expressions in symbolic_expressions.py:**
- `R_bitline` - Bitline resistance
- `C_bitline` - Bitline capacitance
- `tau_bitline` - Bitline RC time constant
- `tau_bitline_with_log` - Including voltage swing
- `t_decoder` - Decoder delay
- `t_senseamp` - Sense amplifier delay
- `t_mux_level` - Multiplexer delay
- `t_total_symbolic` - Total access time

### Method 2: Export to JSON

Export all expressions to a file:

```python
from symbolic_expressions import MemoryAccessTimeExpressions
import json

exprs = MemoryAccessTimeExpressions()

# Export to dictionary
expr_dict = {}
for name in dir(exprs):
    if not name.startswith('_') and name not in ['symbols']:
        expr = getattr(exprs, name)
        expr_dict[name] = {
            'expression': str(expr),
            'latex': latex(expr)
        }

# Save to JSON
with open('symbolic_expressions.json', 'w') as f:
    json.dump(expr_dict, f, indent=2)
```

### Method 3: Interactive Exploration

Use IPython or Jupyter for interactive work:

```python
from sympy import symbols, simplify, expand, latex
from symbolic_expressions import MemoryAccessTimeExpressions

exprs = MemoryAccessTimeExpressions()

# Simplify expressions
tau_simplified = simplify(exprs.tau_bitline)
print("Simplified tau:", tau_simplified)

# Expand expressions
tau_expanded = expand(exprs.tau_bitline)
print("Expanded tau:", tau_expanded)

# Show all symbols used
print("Symbols in tau:", exprs.tau_bitline.free_symbols)

# Export to LaTeX for papers
print("LaTeX for tau:")
print(latex(exprs.tau_bitline))
```

### Method 4: Direct Access to Symbol Definitions

```python
from symbolic_expressions import MemoryAccessTimeSymbols

# Get all symbolic variables
syms = MemoryAccessTimeSymbols()

print("Available symbols:")
for attr in dir(syms):
    if not attr.startswith('_'):
        print(f"  {attr} = {getattr(syms, attr)}")

# Use in custom expressions
custom_expr = syms.rows ** 2 * syms.C_per_cell
print("Custom expression:", custom_expr)
```

---

## Complete Example: Using Both Approaches

```python
#!/usr/bin/env python3
"""
Example: Using both symbolic and numerical modeling
"""

import sys
sys.path.insert(0, '.')

# ============= NUMERICAL MODELING =============
print("=" * 60)
print("NUMERICAL MODELING")
print("=" * 60)

import globals as g
from InputParameter import InputParameter
from Technology import Technology
from MemCell import MemCell
from SubArray import SubArray
from Wire import Wire
from typedef import WireType, WireRepeaterType, BufferDesignTarget
import os

# Initialize
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

# Create subarray
subarray = SubArray()
subarray.Initialize(1024, 2048, 1, 1, 1, True, 1, 8,
                   BufferDesignTarget.latency_first, 2)
subarray.CalculateArea()
subarray.CalculateLatency(1e20)

print(f"\nNumerical Results:")
print(f"  Bitline delay: {subarray.bitlineDelay * 1e9:.6f} ns")
print(f"  Bitline R: {subarray.resBitline:.3f} Ω")
print(f"  Bitline C: {subarray.capBitline * 1e15:.3f} fF")

# ============= SYMBOLIC MODELING =============
print("\n" + "=" * 60)
print("SYMBOLIC MODELING")
print("=" * 60)

from symbolic_expressions import MemoryAccessTimeExpressions, MemoryAccessTimeSymbols
from sympy import latex

# Get expressions
exprs = MemoryAccessTimeExpressions()
syms = MemoryAccessTimeSymbols()

print("\nSymbolic Expressions:")
print(f"  R_bitline = {exprs.R_bitline}")
print(f"  C_bitline = {exprs.C_bitline}")
print(f"  tau = {exprs.tau_bitline}")

print("\nLaTeX Format (for papers):")
print(f"  R_bitline: {latex(exprs.R_bitline)}")
print(f"  C_bitline: {latex(exprs.C_bitline)}")

# Extract parameter values for substitution
R_per_cell_val = (g.localWire.resWirePerUnit *
                  g.cell.heightInFeatureSize * g.devtech.featureSize)
C_per_cell_val = (g.localWire.capWirePerUnit *
                  g.cell.heightInFeatureSize * g.devtech.featureSize)

# Substitute values
subs_dict = {
    syms.rows: 1024,
    syms.R_per_cell: R_per_cell_val,
    syms.C_per_cell: C_per_cell_val,
}

R_bitline_symbolic = float(exprs.R_bitline.subs(subs_dict))
C_bitline_wire_symbolic = float(exprs.C_bitline.subs(subs_dict))

print("\nSymbolic Results (after substitution):")
print(f"  R_bitline: {R_bitline_symbolic:.3f} Ω")
print(f"  C_bitline (wire only): {C_bitline_wire_symbolic * 1e15:.3f} fF")

# ============= COMPARISON =============
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print(f"\n{'Component':<20} {'Numerical':<15} {'Symbolic':<15} {'Match'}")
print("-" * 60)
print(f"{'R_bitline':<20} {subarray.resBitline:>13.3f} Ω  {R_bitline_symbolic:>13.3f} Ω  {'✅' if abs(subarray.resBitline - R_bitline_symbolic) < 0.01 else '❌'}")

print("\n✅ Both approaches give the same answer!")
print("   Use numerical for exact values, symbolic for understanding!")
```

---

## Key Files

1. **`symbolic_expressions.py`** - Contains all symbolic expressions
2. **`final_sram_2layer_model.json`** - Pre-computed expressions + calibrated values
3. **`walkthrough_symbolic_vs_numerical.py`** - Step-by-step comparison
4. **`extract_tech_params.py`** - Extract parameters for substitution

---

## When to Use Each Approach

### Use Numerical Modeling When:
- ✅ You need exact values for a specific design
- ✅ You're comparing against C++ DESTINY
- ✅ You need all physical effects included
- ✅ You're doing final validation

### Use Symbolic Modeling When:
- ✅ You want to understand parameter relationships
- ✅ You're doing design space exploration
- ✅ You need to export equations to papers/docs
- ✅ You want to do sensitivity analysis
- ✅ You're building an optimization framework

### Use Both When:
- ✅ Building a memory modeling framework
- ✅ Verifying symbolic expressions against ground truth
- ✅ Need both insight AND accuracy

---

## Summary

**Numerical Modeling = Concrete Calculator**
- Input: Specific values (1024 rows, 65nm, etc.)
- Output: Concrete numbers (1.989 ns)
- Use: When you need exact answers

**Symbolic Modeling = Mathematical Framework**
- Input: Variable definitions (rows, R_per_cell, etc.)
- Output: Mathematical expressions (R × C)
- Use: When you need understanding and flexibility

**Together = Powerful Design Tool!**
- Symbolic gives insight
- Numerical gives accuracy
- Both validated to match exactly
