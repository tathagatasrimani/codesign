# Symbolic and Numerical Modeling Guide

## Quick Start

### Access All Symbolic Equations
```python
from symbolic_expressions import MemoryAccessTimeExpressions

# Get all expressions
exprs = MemoryAccessTimeExpressions()

# Use them
print("Bitline resistance:", exprs.R_bitline)
print("Bitline capacitance:", exprs.C_bitline)
print("Time constant:", exprs.tau_bitline)
```

### Run Numerical Simulation
```python
import globals as g
from SubArray import SubArray
# ... initialize DESTINY ...

subarray = SubArray()
subarray.Initialize(1024, 2048, ...)
subarray.CalculateLatency(1e20)

print("Bitline delay:", subarray.bitlineDelay)
```

---

## Understanding the Two Approaches

### Numerical Modeling
**What it does:** Runs actual DESTINY code with specific values

**Input:**
- Process node: 65nm
- Rows: 1024
- Columns: 2048
- Temperature: 350K

**Output:**
- Bitline delay: 1.989 ns
- Resistance: 1466.075 Ω
- Capacitance: 335.125 fF

**Workflow:**
```
Config File → Technology → Wire → Cell → SubArray → Numbers
```

---

### Symbolic Modeling
**What it does:** Creates mathematical expressions with variables

**Input:**
- Variables: rows, R_per_cell, C_per_cell

**Output:**
- Expressions: `R_bitline = R_per_cell × rows`
- LaTeX: `R_{bitline} = R_{per cell} \times rows`

**Workflow:**
```
Define Variables → Build Expressions → Substitute Values (optional)
```

---

## The Process (Detailed)

### How Numerical Modeling Works

```
STEP 1: Read Configuration
   ↓
   config/sample_SRAM_2layer.cfg
   - ProcessNode: 65
   - DeviceRoadmap: LOP
   - Temperature: 350

STEP 2: Initialize Technology
   ↓
   Technology.py extracts:
   - V_dd = 0.8 V
   - I_on = 524.5 μA/μm
   - Feature size = 65 nm

STEP 3: Initialize Wire Model
   ↓
   Wire.py calculates:
   - resWirePerUnit = 1.509e6 Ω/m
   - capWirePerUnit = 2.939e-10 F/m

STEP 4: Initialize Cell Model
   ↓
   MemCell reads:
   - heightInFeatureSize = 14.6 F
   - widthInFeatureSize = 10.0 F

STEP 5: Create SubArray
   ↓
   SubArray.py computes:
   - lenBitline = 1024 × 14.6 × 65e-9 = 9.718e-4 m
   - resBitline = 9.718e-4 × 1.509e6 = 1466.075 Ω
   - capBitline = 9.718e-4 × 2.939e-10 + drain_cap = 335.125 fF

STEP 6: Calculate Delay
   ↓
   SubArray.CalculateLatency():
   - tau = (R_access + R_pulldown) × C + R_bitline × C/2
   - tau = 18.057 ns (before log)
   - tau = 1.903 ns (after voltage swing)
   - delay = horowitz(tau, beta) = 1.989 ns

OUTPUT: Concrete number: 1.989 ns
```

### How Symbolic Modeling Works

```
STEP 1: Define Symbols
   ↓
   from sympy import symbols
   rows = symbols('rows')
   R_per_cell = symbols('R_per_cell')
   C_per_cell = symbols('C_per_cell')

STEP 2: Build Expressions
   ↓
   R_bitline = R_per_cell * rows
   C_bitline = C_per_cell * rows
   tau = R_bitline * C_bitline

STEP 3: Simplify/Analyze
   ↓
   from sympy import simplify
   tau_simplified = simplify(tau)
   # Result: C_per_cell*R_per_cell*rows^2

STEP 4 (Optional): Substitute Values
   ↓
   values = {rows: 1024, R_per_cell: 1.432, C_per_cell: 2.789e-16}
   R_bitline_val = R_bitline.subs(values)
   # Result: 1466.075 Ω

OUTPUT: Either equation OR number (your choice)
```

---

## Key Files You Need

### For Symbolic Modeling

1. **`symbolic_expressions.py`** - Main module with all expressions
   ```python
   from symbolic_expressions import MemoryAccessTimeExpressions
   exprs = MemoryAccessTimeExpressions()
   ```

2. **`show_all_equations.py`** - Shows all available equations
   ```bash
   python show_all_equations.py
   ```

3. **`final_sram_2layer_model.json`** - Pre-computed model
   ```json
   {
     "symbolic_expressions": {...},
     "calibrated_values": {...}
   }
   ```

### For Numerical Modeling

1. **`SubArray.py`** - Main calculation engine
2. **`Technology.py`** - Technology parameters
3. **`Wire.py`** - Wire models
4. **`MemCell.py`** - Cell models

### For Verification

1. **`walkthrough_symbolic_vs_numerical.py`** - Complete walkthrough
2. **`final_comparison.py`** - Quick comparison
3. **`verify_all_match.py`** - Comprehensive verification

---

## Available Symbolic Expressions

### Bitline Expressions
- `R_bitline` = R_per_cell × rows
- `C_bitline` = C_per_cell × rows
- `tau_bitline` = Full RC delay formula
- `tau_bitline_with_log` = Including voltage swing

### Decoder Expressions
- `R_decoder_stage` = Decoder resistance
- `C_decoder_stage` = Decoder capacitance
- `t_decoder_stage` = Decoder delay

### Sense Amplifier
- `t_senseamp` = C_load × V_swing / I_amp

### Multiplexer
- `R_mux_pass` = Pass transistor resistance
- `t_mux_level` = Mux level delay

### Total Access Time
- `t_total_symbolic` = Sum of all components

---

## How to Use Equations

### Example 1: Get LaTeX for Paper
```python
from symbolic_expressions import MemoryAccessTimeExpressions
from sympy import latex

exprs = MemoryAccessTimeExpressions()

# Get LaTeX
latex_tau = latex(exprs.tau_bitline)
print(latex_tau)

# Output:
# R_{per cell} rows \left(C_{mux} + \frac{C_{per cell} rows}{2}\right) +
# \left(R_{access} + R_{pulldown}\right) \left(C_{access} + C_{mux} + C_{per cell} rows\right)
```

### Example 2: Sensitivity Analysis
```python
from sympy import diff

# How does delay change with rows?
dtau_drows = diff(exprs.tau_bitline, exprs.symbols.rows)
print("Sensitivity to rows:", dtau_drows)

# Shows: tau ∝ rows² for large arrays
```

### Example 3: Design Space Exploration
```python
import numpy as np
import matplotlib.pyplot as plt

# Sweep number of rows
rows_values = np.logspace(2, 4, 50)  # 100 to 10,000
tau_values = []

for n_rows in rows_values:
    tau_val = exprs.tau_bitline.subs({
        exprs.symbols.rows: n_rows,
        exprs.symbols.R_per_cell: 1.432,
        exprs.symbols.C_per_cell: 2.789e-16,
        exprs.symbols.R_access: 32601,
        exprs.symbols.R_pulldown: 20532,
        exprs.symbols.C_access: 9.664e-17,
        exprs.symbols.C_mux: 0
    })
    tau_values.append(float(tau_val))

plt.loglog(rows_values, tau_values)
plt.xlabel('Number of Rows')
plt.ylabel('Tau (s)')
plt.title('Bitline Delay vs Array Size')
plt.grid(True)
plt.show()
```

### Example 4: Export to JSON
```python
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

with open('my_equations.json', 'w') as f:
    json.dump(equations, f, indent=2)
```

---

## Complete Workflow Example

### Scenario: Design Space Exploration for Cache

```python
#!/usr/bin/env python3
"""
Find optimal subarray size for minimum delay
"""

from symbolic_expressions import MemoryAccessTimeExpressions
import numpy as np

# Initialize
exprs = MemoryAccessTimeExpressions()

# Fixed parameters (from 65nm technology)
R_per_cell = 1.432e-3  # Ω
C_per_cell = 2.789e-16  # F
R_access = 32601  # Ω
R_pulldown = 20532  # Ω
C_access = 9.664e-17  # F

# Sweep rows from 256 to 2048
best_delay = float('inf')
best_rows = 0

for rows in [256, 512, 1024, 1536, 2048]:
    # Calculate tau symbolically, then evaluate
    tau = exprs.tau_bitline.subs({
        exprs.symbols.rows: rows,
        exprs.symbols.R_per_cell: R_per_cell,
        exprs.symbols.C_per_cell: C_per_cell,
        exprs.symbols.R_access: R_access,
        exprs.symbols.R_pulldown: R_pulldown,
        exprs.symbols.C_access: C_access,
        exprs.symbols.C_mux: 0
    })

    tau_val = float(tau)

    print(f"Rows: {rows:4d}, tau: {tau_val*1e9:.3f} ns")

    if tau_val < best_delay:
        best_delay = tau_val
        best_rows = rows

print(f"\nOptimal: {best_rows} rows with delay {best_delay*1e9:.3f} ns")

# Now validate with numerical DESTINY
print("\nValidating with numerical DESTINY...")
# ... run actual DESTINY simulation ...
```

---

## Summary

### Use Numerical When:
✅ Need exact value for specific design
✅ Comparing against C++ DESTINY
✅ Final validation

### Use Symbolic When:
✅ Understanding relationships
✅ Design space exploration
✅ Writing papers
✅ Sensitivity analysis

### Use Both When:
✅ Building optimization framework
✅ Research & development
✅ Need insight AND accuracy

---

## Getting Help

1. **See all equations:**
   ```bash
   python show_all_equations.py
   ```

2. **Run walkthrough:**
   ```bash
   python walkthrough_symbolic_vs_numerical.py
   ```

3. **Quick verification:**
   ```bash
   python final_comparison.py
   ```

4. **Read guide:**
   ```bash
   cat SYMBOLIC_VS_NUMERICAL_GUIDE.md
   ```

---

## Files Reference

| File | Purpose |
|------|---------|
| `symbolic_expressions.py` | All symbolic expressions |
| `show_all_equations.py` | Display all available equations |
| `walkthrough_symbolic_vs_numerical.py` | Complete step-by-step example |
| `final_comparison.py` | Quick C++ vs Python comparison |
| `verify_all_match.py` | Comprehensive verification |
| `SYMBOLIC_VS_NUMERICAL_GUIDE.md` | Complete documentation |
| `final_sram_2layer_model.json` | Pre-computed model |
| `FIX_SUMMARY.md` | Bug fix documentation |
| `FINAL_VERIFICATION.md` | Verification results |

---

**Questions? Check the guides or run the example scripts!**
