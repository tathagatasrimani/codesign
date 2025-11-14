# Symbolic Modeling Walkthrough - DESTINY vs CACTI

## How It Works: Step-by-Step

### Step 1: Create Symbolic Variables (Similar to CACTI)

**CACTI Approach (from your example):**
```python
class BaseParameters:
    def __init__(self, tech_node, dat_file):
        # Create sympy variables
        self.V_dd = symbols("V_dd", positive=True)
        self.V_th = symbols("V_th", positive=True)
        self.C_g_ideal = symbols("C_g_ideal", positive=True)
        self.I_on_n = symbols("I_on_n", positive=True)
        # ... more variables
```

**DESTINY Approach (what we built):**
```python
class BaseParameters:
    def __init__(self):
        # Create sympy variables - SAME APPROACH!
        self.vdd = symbols("vdd", positive=True, real=True)
        self.vth = symbols("vth", positive=True, real=True)
        self.capIdealGate = symbols("capIdealGate", positive=True, real=True)
        self.currentOnNmos = symbols("currentOnNmos", positive=True, real=True)
        # ... 87 total variables
```

**Key Point:** ✅ Identical approach - both use SymPy `symbols()` to create symbolic variables.

---

### Step 2: Build Symbol Table (Similar to CACTI)

**CACTI:**
```python
def init_symbol_table(self):
    self.symbol_table = {
        "V_dd": self.V_dd,
        "V_th": self.V_th,
        "C_g_ideal": self.C_g_ideal,
        # ...
    }
```

**DESTINY:**
```python
def build_symbol_table(self):
    self.symbol_table = {
        'vdd': self.vdd,
        'vth': self.vth,
        'capIdealGate': self.capIdealGate,
        # ...
    }
```

**Key Point:** ✅ Same pattern - map string names to symbol objects for easy lookup.

---

### Step 3: Store Concrete Values (Similar to CACTI)

**CACTI:**
```python
# CACTI stores concrete values in tech_values dictionary
self.tech_values = {}
self.tech_values[self.V_dd] = 1.1  # Voltage
self.tech_values[self.I_on_n] = 1211.4  # Current
```

**DESTINY:**
```python
# DESTINY does the same thing!
self.tech_values = {}
self.tech_values[self.vdd] = 1.1  # Voltage
self.tech_values[self.currentOnNmos] = 1211.4  # Current

# Or populate from DESTINY objects
bp.populate_from_technology(g.tech)
bp.populate_from_memcell(g.cell)
```

**Key Point:** ✅ Identical approach - use dictionary with symbols as keys, concrete values as values.

---

### Step 4: Create Symbolic Expressions (Similar to CACTI)

**CACTI (example from your code):**
```python
# CACTI creates expressions using symbolic variables
gate_cap = (self.C_g_ideal + self.C_fringe) * self.W
resistance = self.nmos_effective_resistance_multiplier * self.Vdd / self.I_on_n
```

**DESTINY:**
```python
# DESTINY creates expressions the same way!
gate_cap = (bp.capIdealGate + bp.capFringe) * width
resistance = bp.effectiveResistanceMultiplier * bp.vdd / bp.currentOnNmos
```

**Key Point:** ✅ Identical approach - just use symbolic variables in normal Python expressions!

---

### Step 5: Evaluate Expressions (Similar to CACTI)

**CACTI:**
```python
# Evaluate symbolic expression with concrete values
result = expression.xreplace(self.tech_values)
# or
result = expression.evalf(subs=self.tech_values)
```

**DESTINY:**
```python
# Same thing!
result = expression.evalf(subs=bp.tech_values)
```

**Key Point:** ✅ Identical approach - substitute concrete values into symbolic expressions.

---

## Complete Example Comparison

### CACTI-Style Calculation

```python
# In CACTI BaseParameters class
self.V_dd = symbols("V_dd", positive=True)
self.I_on_n = symbols("I_on_n", positive=True)
self.nmos_effective_resistance_multiplier = symbols("nmos_eff_res_mult", positive=True)

# Populate concrete values
self.tech_values[self.V_dd] = 1.1
self.tech_values[self.I_on_n] = 1211.4
self.tech_values[self.nmos_effective_resistance_multiplier] = 1.5

# Create symbolic expression
resistance = self.nmos_effective_resistance_multiplier * self.V_dd / self.I_on_n

# Symbolic form
print(resistance)
# → nmos_eff_res_mult*V_dd/I_on_n

# Evaluate
result = resistance.evalf(subs=self.tech_values)
print(result)
# → 0.00136...
```

### DESTINY Equivalent (What We Built)

```python
# In DESTINY BaseParameters class
bp = BaseParameters()
bp.vdd = symbols("vdd", positive=True)
bp.currentOnNmos = symbols("currentOnNmos", positive=True)
bp.effectiveResistanceMultiplier = symbols("effectiveResistanceMultiplier", positive=True)

# Populate concrete values (from DESTINY objects or manually)
bp.tech_values[bp.vdd] = 1.1
bp.tech_values[bp.currentOnNmos] = 1211.4
bp.tech_values[bp.effectiveResistanceMultiplier] = 1.5

# Create symbolic expression - SAME WAY!
resistance = bp.effectiveResistanceMultiplier * bp.vdd / bp.currentOnNmos

# Symbolic form
print(resistance)
# → effectiveResistanceMultiplier*vdd/currentOnNmos

# Evaluate - SAME WAY!
result = resistance.evalf(subs=bp.tech_values)
print(result)
# → 0.00136...
```

**Key Point:** ✅ **EXACT SAME APPROACH!** Just different variable names.

---

## Key Similarities

| Feature | CACTI | DESTINY | Status |
|---------|-------|---------|--------|
| Create symbols with `symbols()` | ✅ | ✅ | Identical |
| Symbol table mapping | ✅ | ✅ | Identical |
| `tech_values` dictionary | ✅ | ✅ | Identical |
| Direct symbolic expressions | ✅ | ✅ | Identical |
| `.evalf(subs=...)` evaluation | ✅ | ✅ | Identical |
| Sensitivity via `diff()` | ✅ | ✅ | Identical |
| Store in global instance | ✅ | ✅ | Identical |

---

## Detailed Workflow Comparison

### CACTI Workflow

1. **Initialize BaseParameters**
   ```python
   bp = BaseParameters(tech_node="22nm", dat_file="...")
   ```

2. **Symbols are created automatically**
   ```python
   # In __init__:
   self.V_dd = symbols("V_dd", positive=True)
   self.I_on_n = symbols("I_on_n", positive=True)
   ```

3. **Concrete values populated from CACTI**
   ```python
   # In __init__:
   cacti_params = {}
   dat.scan_dat(cacti_params, dat_file, ...)
   for key, value in cacti_params.items():
       self.tech_values[key] = value
   ```

4. **Use in calculations**
   ```python
   # In circuit model:
   cap = self.C_g_ideal * self.W
   result = cap.evalf(subs=self.tech_values)
   ```

### DESTINY Workflow (What We Built)

1. **Initialize BaseParameters**
   ```python
   bp = BaseParameters()
   ```

2. **Symbols are created automatically** (SAME!)
   ```python
   # In __init__:
   self.vdd = symbols("vdd", positive=True)
   self.currentOnNmos = symbols("currentOnNmos", positive=True)
   ```

3. **Concrete values populated from DESTINY**
   ```python
   # After DESTINY initialization:
   g.tech.Initialize(65, DeviceRoadmap.HP, ...)
   bp.populate_from_technology(g.tech)
   bp.populate_from_memcell(g.cell)
   ```

4. **Use in calculations** (SAME!)
   ```python
   # In formula.py or circuit calculations:
   cap = bp.capIdealGate * width
   result = cap.evalf(subs=bp.tech_values)
   ```

**Key Point:** ✅ **WORKFLOW IS IDENTICAL!** Both follow same pattern.

---

## Advanced Features (Both Support)

### 1. Sensitivity Analysis

**CACTI & DESTINY:**
```python
from sympy import diff

# Take derivative
sensitivity = diff(expression, bp.vdd)
print(f"∂result/∂vdd = {sensitivity}")

# Evaluate sensitivity at operating point
sens_value = sensitivity.evalf(subs=bp.tech_values)
```

### 2. Parameter Sweeps

**CACTI & DESTINY:**
```python
# Sweep voltage
for vdd in [0.8, 0.9, 1.0, 1.1, 1.2]:
    bp.tech_values[bp.vdd] = vdd
    result = expression.evalf(subs=bp.tech_values)
    print(f"vdd={vdd}V → result={result}")
```

### 3. Optimization Constraints

**CACTI & DESTINY:**
```python
# Create constraint expression
power_constraint = dynamic_power + leakage_power < max_power

# Use in optimization
# (both can interface with scipy.optimize or similar)
```

---

## Differences (Minor)

| Aspect | CACTI | DESTINY | Impact |
|--------|-------|---------|--------|
| Number of variables | ~150 | 87 | DESTINY covers core params |
| Initialization | From .dat files | From Technology/MemCell objects | Different data sources |
| Variable names | CACTI_style | DESTINY_style | Cosmetic only |
| Temperature arrays | Yes | Simplified (single ref) | DESTINY uses 300K reference |

**Key Point:** Differences are only in **what** parameters are modeled, not **how** they're modeled.

---

## Why This Approach Works

### Benefits (Same for Both CACTI and DESTINY)

1. **Symbolic expressions show exact relationships**
   ```
   R = effectiveResistanceMultiplier*vdd/currentOnNmos
   ```
   You can SEE that R ∝ vdd and R ∝ 1/current

2. **Automatic differentiation**
   ```python
   diff(R, vdd)  → effectiveResistanceMultiplier/currentOnNmos
   ```

3. **Design space exploration**
   ```python
   # Sweep multiple parameters symbolically before evaluating
   ```

4. **Verification**
   ```python
   # Compare symbolic vs numerical to catch bugs
   assert abs(symbolic_result - numerical_result) < tolerance
   ```

5. **Optimization**
   ```python
   # Use symbolic expressions as objective functions
   ```

---

## Practical Example: Gate Capacitance

### CACTI Version
```python
# In CACTI BaseParameters
self.C_g_ideal = symbols("C_g_ideal", positive=True)
self.C_fringe = symbols("C_fringe", positive=True)
self.W = symbols("W", positive=True)

# Populate
self.tech_values[self.C_g_ideal] = 4.7e-10
self.tech_values[self.C_fringe] = 2.4e-10
self.tech_values[self.W] = 100e-9

# Calculate
gate_cap = self.C_g_ideal * self.W + 3 * self.C_fringe * self.W

# Evaluate
result = gate_cap.evalf(subs=self.tech_values)
# → 1.19e-16 F
```

### DESTINY Version (Our Implementation)
```python
# In DESTINY BaseParameters
bp.capIdealGate = symbols("capIdealGate", positive=True)
bp.capFringe = symbols("capFringe", positive=True)
width = 100e-9

# Populate
bp.tech_values[bp.capIdealGate] = 4.7e-10
bp.tech_values[bp.capFringe] = 2.4e-10

# Calculate
gate_cap = bp.capIdealGate * width + 3 * bp.capFringe * width

# Evaluate
result = gate_cap.evalf(subs=bp.tech_values)
# → 1.19e-16 F
```

**IDENTICAL RESULTS AND APPROACH!** ✅

---

## Summary

### Is DESTINY symbolic modeling similar to CACTI?

**Answer: YES, EXACTLY THE SAME!**

Both use:
- ✅ SymPy `symbols()` for variable creation
- ✅ `tech_values` dictionary for concrete values
- ✅ Direct symbolic expressions in Python
- ✅ `.evalf(subs=...)` for evaluation
- ✅ `diff()` for sensitivity analysis
- ✅ Same workflow and patterns

The **ONLY** differences are:
1. Variable names (CACTI vs DESTINY naming conventions)
2. Data sources (CACTI .dat files vs DESTINY Technology objects)
3. Number of parameters (based on what each tool models)

The **methodology is identical** - I followed the exact same approach you showed me from CACTI!

---

## Try It Yourself

Run these to see it in action:

```bash
# Basic test
python test_base_params.py

# Full examples
python example_symbolic_destiny.py
```

You'll see outputs like:
```
📐 Symbolic: effectiveResistanceMultiplier*vdd/currentOnNmos
📊 Numerical: 0.000681 Ω
🔍 ∂R/∂vdd = effectiveResistanceMultiplier/currentOnNmos
```

This is **exactly** what CACTI does!
