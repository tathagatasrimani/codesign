# Quick Start: Symbolic Modeling in DESTINY

## Run the Examples

```bash
cd destiny_3d_cache_python

# Basic test
python test_base_params.py

# Comprehensive examples
python example_symbolic_destiny.py
```

## Example Output Highlights

### Example 1: Gate Capacitance
```
📐 Symbolic Expression:
   3.0e-7*capFringe + 1.0e-7*capIdealGate + 1.0e-7*capOverlap + capPolywire*phyGateLength

📊 Numerical Result:
   Gate Capacitance = 0.128 fF
```

### Example 2: Transistor Resistance
```
📐 Symbolic Expression:
   0.5*effectiveResistanceMultiplier*vdd/currentOnNmos

🔍 Sensitivity Analysis:
   ∂R/∂vdd = 0.5*effectiveResistanceMultiplier/currentOnNmos
   ∂R/∂I_on = -0.5*effectiveResistanceMultiplier*vdd/currentOnNmos**2
```

### Example 3: Power Scaling
```
💡 Voltage Scaling Analysis:
   Power ∝ V²
   Original: V = 1.10V, P = 0.155 μW
   Scaled:   V = 0.88V, P = 0.099 μW
   Reduction: 36.0%
```

## Basic Usage Pattern

```python
from base_parameters import BaseParameters
import globals as g
from Technology import Technology

# 1. Create base parameters
bp = BaseParameters()

# 2. Initialize DESTINY technology
g.tech = Technology()
g.tech.Initialize(65, DeviceRoadmap.HP, InputParameter())

# 3. Populate concrete values
bp.populate_from_technology(g.tech)

# 4. Create symbolic expression
resistance = bp.effectiveResistanceMultiplier * bp.vdd / (bp.currentOnNmos * width)

# 5. Show symbolic form
print(f"Symbolic: {resistance}")

# 6. Evaluate numerically
result = resistance.evalf(subs=bp.tech_values)
print(f"Numerical: {result}")

# 7. Sensitivity analysis
from sympy import diff
sensitivity = diff(resistance, bp.vdd)
print(f"∂R/∂vdd = {sensitivity}")
```

## Available Examples

The `example_symbolic_destiny.py` demonstrates:

1. **Gate Capacitance** - Basic symbolic calculation from formula.py
2. **Transistor Resistance** - With sensitivity analysis
3. **Dynamic Power** - Voltage scaling effects
4. **RC Delay** - Technology scaling comparison
5. **Memory Cell Energy** - Write energy optimization
6. **Complete Inverter** - Full circuit analysis

## Key Features Demonstrated

✅ **Symbolic Expressions** - See exact mathematical relationships
```python
R = effectiveResistanceMultiplier*vdd/(currentOnNmos*width)
```

✅ **Numerical Evaluation** - Get concrete values
```python
result = R.evalf(subs=bp.tech_values)  # → 0.681 mΩ
```

✅ **Sensitivity Analysis** - Understand parameter impact
```python
dR_dvdd = diff(R, bp.vdd)  # → effectiveResistanceMultiplier/currentOnNmos
```

✅ **Technology Scaling** - Compare different nodes
```python
# 65nm: delay = 0.000 ps
# 45nm: delay = 0.000 ps
# Speedup: 1.60x
```

✅ **Design Exploration** - Vary parameters symbolically
```python
# Try different voltages
bp.tech_values[bp.vdd] = 0.88  # 20% reduction
power_scaled = power.evalf(subs=bp.tech_values)  # → 36% power savings
```

## Symbolic Variables Available

**87 total variables** across:
- Technology parameters (vdd, vth, capacitances, currents)
- Memory cell parameters (resistance, area, voltages)
- Wire parameters (dimensions, parasitics)
- TSV parameters (3D stacking)

See `base_parameters.py` for complete list.

## Next Steps

1. Try modifying the examples
2. Create your own symbolic expressions
3. Integrate into DESTINY main simulation
4. Use for optimization or design space exploration

## Tips

- Use `.evalf(subs=bp.tech_values)` to evaluate
- Use `simplify()` to simplify expressions
- Use `diff()` for derivatives/sensitivity
- Access symbols via `bp.vdd`, `bp.resistanceOn`, etc.
- Access values via `bp.tech_values[bp.vdd]`

## References

- `base_parameters.py` - Main symbolic parameters file
- `SYMBOLIC_MODELING_README.md` - Full documentation
- Based on CACTI codesign framework approach
