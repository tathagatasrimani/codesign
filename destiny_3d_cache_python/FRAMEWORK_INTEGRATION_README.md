# Framework Integration: Symbolic DESTINY Models

## Overview

This module provides **SymPy symbolic expressions** for memory access time components that can be **passed to other parts of your framework** for analysis, optimization, and visualization.

## Quick Start

```python
from symbolic_expressions import create_symbolic_model_from_destiny_output

# Create symbolic model from C++ DESTINY output
model = create_symbolic_model_from_destiny_output('cpp_output.txt')

# Get symbolic expression (SymPy object)
bitline_expr = model.get_symbolic_expression('tau_bitline')

# Get numerical value (from C++ DESTINY)
bitline_value = model.get_numerical_value('t_bitline')  # in seconds

# Export for other tools
model.export_to_json('memory_model.json')
```

## Main Files

| File | Purpose |
|------|---------|
| **`symbolic_expressions.py`** | Core module with SymPy expressions |
| **`run_symbolic_destiny.py`** | Main script to generate and export models |
| **`example_framework_usage.py`** | 7 examples showing how to use in your framework |
| **`symbolic_analysis_accurate.py`** | Human-readable analysis output |

## Available Symbolic Expressions

### 1. Bitline Delay (SRAM)

```python
tau_bitline = R_per_cell*rows*(C_mux + C_per_cell*rows/2) +
              (R_access + R_pulldown)*(C_access + C_mux + C_per_cell*rows)
```

**LaTeX:**
```latex
R_{per cell} rows \left(C_{mux} + \frac{C_{per cell} rows}{2}\right) +
\left(R_{access} + R_{pulldown}\right) \left(C_{access} + C_{mux} + C_{per cell} rows\right)
```

**Symbols:** `rows`, `R_per_cell`, `C_per_cell`, `R_access`, `R_pulldown`, `C_access`, `C_mux`

### 2. Sense Amplifier Delay

```python
t_senseamp = C_load*V_swing/I_amp
```

**Symbols:** `C_load`, `V_swing`, `I_amp`

### 3. Decoder Stage Delay

```python
t_decoder_stage = R_eff*V_dd*(C_gate + C_wire)/(I_on*W)
```

**Symbols:** `R_eff`, `V_dd`, `C_gate`, `C_wire`, `I_on`, `W`

### 4. Multiplexer Delay

```python
t_mux_level = C_load*R_eff*V_dd/(I_on*W_pass)
```

**Symbols:** `C_load`, `R_eff`, `V_dd`, `I_on`, `W_pass`

## Exported JSON Structure

When you export a model using `model.export_to_json()`, you get:

```json
{
  "symbolic_expressions": {
    "tau_bitline": {
      "expression": "R_per_cell*rows*(C_mux + C_per_cell*rows/2) + ...",
      "latex": "R_{per cell} rows \\left(C_{mux} + \\frac{C_{per cell} rows}{2}\\right) + ...",
      "symbols": ["rows", "R_per_cell", "C_per_cell", ...]
    },
    ...
  },
  "numerical_results": {
    "t_decoder": 1.2e-09,
    "t_bitline": 1.99e-09,
    "t_senseamp": 6.755e-12,
    "t_mux": 2.4213e-11,
    "t_total": 3.22e-09
  },
  "config_params": {
    "rows": 1024,
    "cols": 2048,
    "num_banks_x": 1,
    "num_banks_y": 1,
    "num_stacks": 2,
    ...
  }
}
```

## Usage Patterns

### Pattern 1: Get Expression for Analysis

```python
model = create_symbolic_model_from_destiny_output('output.txt')

# Get SymPy expression
expr = model.get_symbolic_expression('tau_bitline')

# Analyze it
print(f"Expression: {expr}")
print(f"Symbols: {expr.free_symbols}")
print(f"Simplified: {simplify(expr)}")
```

### Pattern 2: Evaluate with Custom Parameters

```python
# Define parameters
params = {
    'rows': 512,
    'R_per_cell': 1.5e3,
    'C_per_cell': 0.3e-15,
    'R_access': 500,
    'R_pulldown': 300,
    'C_access': 1e-15,
    'C_mux': 0.5e-15
}

# Evaluate
result = model.expressions.evaluate('tau_bitline', params)
print(f"τ_bitline = {result*1e9:.3f} ns")
```

### Pattern 3: Convert to Python Function

```python
# Convert expression to callable function
C_bitline_func = model.expressions.to_python_function('C_bitline')

# Use with different parameters
for rows in [256, 512, 1024, 2048]:
    C = C_bitline_func(0.3e-15, rows)  # C_per_cell, rows
    print(f"{rows} rows → {C*1e15:.2f} fF")
```

### Pattern 4: Export for Other Tools

```python
# Export to JSON (can be read by Julia, JavaScript, MATLAB, etc.)
model.export_to_json('memory_model.json')

# Other tools can load JSON and use expressions
import json
with open('memory_model.json') as f:
    data = json.load(f)

bitline_expr_str = data['symbolic_expressions']['tau_bitline']['expression']
bitline_latex = data['symbolic_expressions']['tau_bitline']['latex']
```

### Pattern 5: Parametric Studies

```python
# Study how delay scales with rows
import numpy as np
import matplotlib.pyplot as plt

rows_range = np.arange(256, 4096, 256)
delays = []

for rows in rows_range:
    params = {'rows': rows, 'R_per_cell': 1e3, 'C_per_cell': 0.3e-15, ...}
    tau = model.expressions.evaluate('tau_bitline', params)
    delays.append(tau * 1e9)  # Convert to ns

plt.plot(rows_range, delays)
plt.xlabel('Rows')
plt.ylabel('Bitline Delay (ns)')
plt.title('Bitline Delay vs Subarray Size')
```

### Pattern 6: Integration with Optimization

```python
from scipy.optimize import minimize

def objective(x):
    """Minimize access time given constraints"""
    rows = x[0]

    # Evaluate symbolic expression
    params = {'rows': rows, ...}
    tau = model.expressions.evaluate('tau_bitline', params)

    return tau

# Optimize
result = minimize(objective, x0=[1024], bounds=[(256, 4096)])
optimal_rows = result.x[0]
```

## Integration Points for Your Framework

### 1. Analysis Tools
- Use expressions to understand component relationships
- Generate sensitivity analysis
- Identify bottlenecks automatically

### 2. Optimization Tools
- Use expressions as objective functions
- Evaluate design space efficiently
- Perform gradient-based optimization

### 3. Visualization Tools
- Plot expressions vs parameters
- Show symbolic formulas in UI
- Generate LaTeX for reports

### 4. Design Space Exploration
- Evaluate expressions across parameter ranges
- Compare different configurations
- Generate Pareto frontiers

### 5. Documentation Generation
- Export LaTeX for papers
- Generate equation sheets
- Create interactive notebooks

## Running for Different Memory Types

```bash
# SRAM 2-layer
python run_symbolic_destiny.py ../destiny_3d_cache-master/cpp_output_sram2layer.txt \
    --export sram_2layer.json

# eDRAM
python run_symbolic_destiny.py ../destiny_3d_cache-master/cpp_output_edram.txt \
    --export edram.json

# PCRAM
python run_symbolic_destiny.py ../destiny_3d_cache-master/cpp_output_pcram.txt \
    --export pcram.json

# STT-RAM
python run_symbolic_destiny.py ../destiny_3d_cache-master/cpp_output_sttram.txt \
    --export sttram.json
```

Each will generate:
1. SymPy expressions specific to that memory type
2. Numerical values from C++ DESTINY
3. JSON export for framework integration

## API Reference

### `MemoryBlockSymbolicModel`

**Methods:**
- `get_symbolic_expression(component)` → SymPy expression
- `get_numerical_value(component)` → float (seconds)
- `get_complete_model()` → dict with all data
- `export_to_json(filename)` → saves to file

### `MemoryAccessTimeExpressions`

**Methods:**
- `get_expression(component)` → SymPy expression
- `get_all_expressions()` → dict of all expressions
- `evaluate(component, params)` → numerical result
- `to_latex(component)` → LaTeX string
- `to_python_function(component)` → callable function
- `export_expressions(filename)` → saves to file

## Example: Complete Framework Integration

```python
# Step 1: Run C++ DESTINY (or use existing output)
# ./destiny config.cfg -o output.txt

# Step 2: Create symbolic model
from symbolic_expressions import create_symbolic_model_from_destiny_output
model = create_symbolic_model_from_destiny_output('output.txt')

# Step 3: Export for your framework
model.export_to_json('model.json')

# Step 4: In your framework's analysis module
import json
with open('model.json') as f:
    destiny_model = json.load(f)

# Step 5: Use the expressions
bitline_expr = destiny_model['symbolic_expressions']['tau_bitline']['expression']
bitline_value = destiny_model['numerical_results']['t_bitline']  # seconds

# Step 6: Display to user (with LaTeX support)
latex_expr = destiny_model['symbolic_expressions']['tau_bitline']['latex']
print(f"Bitline: ${latex_expr}$ = {bitline_value*1e9:.3f} ns")
```

## Benefits

✅ **Symbolic expressions are REAL** - From actual DESTINY source code
✅ **Numerical values are ACCURATE** - From validated C++ DESTINY
✅ **Expressions are REUSABLE** - Can be passed to other framework components
✅ **Multiple export formats** - SymPy, JSON, LaTeX, Python functions
✅ **Type safety** - SymPy ensures mathematical correctness
✅ **Extensible** - Easy to add new expressions or memory types

## Questions?

See `example_framework_usage.py` for 7 complete examples showing different integration patterns.
