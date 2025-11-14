# DESTINY Symbolic Modeling

This directory contains the symbolic modeling framework for DESTINY, enabling symbolic computation and optimization.

## Overview

The symbolic modeling approach uses **SymPy** to create symbolic variables for all DESTINY parameters (technology, memory cell, wires, etc.), allowing:
- **Symbolic expressions** in calculation outputs
- **Optimization** with symbolic constraints
- **Sensitivity analysis** by taking derivatives
- **Design space exploration** with symbolic parameters

## Files

### Core Files

- **`base_parameters.py`**: Main base parameters class with all symbolic variables
  - 87 symbolic variables covering technology, memory cells, wires, TSVs
  - Methods to populate concrete values from DESTINY objects
  - Symbol table for string → symbol mapping

- **`symbolic_wrapper.py`**: Helper utilities for symbolic computation (optional)
  - `SymbolicValue` class for parallel numerical/symbolic computation
  - Symbolic math functions (sqrt, log, min, max, etc.)
  - Assertion utilities to verify symbolic vs numerical match

### Test Files

- **`test_base_params.py`**: Test the base parameters approach
  - Creates symbolic variables
  - Performs symbolic calculations
  - Populates and evaluates with concrete values

- **`test_symbolic.py`**: Test the symbolic wrapper utilities (optional)
  - Demonstrates parallel symbolic/numerical computation
  - Shows branching with concrete values

## Usage

### Basic Usage

```python
from base_parameters import BaseParameters
import globals as g
from Technology import Technology
from MemCell import MemCell
from InputParameter import InputParameter

# Create base parameters object
bp = BaseParameters()

# Symbolic variables are now available
print(bp.vdd)           # Symbol: vdd
print(bp.featureSize)   # Symbol: featureSize
print(bp.resistanceOn)  # Symbol: resistanceOn

# Create symbolic expressions
gate_cap = ((bp.capIdealGate + bp.capOverlap + 3*bp.capFringe) * bp.cellWidthInFeatureSize +
            bp.phyGateLength * bp.capPolywire)

print(gate_cap)  # Symbolic expression
```

### Populating Concrete Values

```python
# Initialize DESTINY objects
g.tech = Technology()
g.tech.Initialize(65, DeviceRoadmap.HP, g.inputParameter)
g.cell = MemCell()
g.cell.ReadCellFromFile("config/sample_SRAM.cell")

# Populate concrete values
bp.populate_from_technology(g.tech)
bp.populate_from_memcell(g.cell)
bp.populate_from_input_parameter(g.inputParameter)

# Evaluate symbolic expression with concrete values
gate_cap_value = gate_cap.evalf(subs=bp.tech_values)
print(f"Gate capacitance: {gate_cap_value} F")
```

### Symbol Table Access

```python
# Access symbols by string name
vdd_symbol = bp.symbol_table['vdd']
resistance_symbol = bp.symbol_table['resistanceOn']

# Get concrete value
vdd_value = bp.tech_values[bp.vdd]
```

## Symbolic Variables

### Technology Parameters (from Technology.py)
- **Voltage**: `vdd`, `vpp`, `vth`, `vdsatNmos`, `vdsatPmos`
- **Dimensions**: `featureSize`, `featureSizeInNano`, `phyGateLength`
- **Capacitances**: `capIdealGate`, `capFringe`, `capJunction`, `capOverlap`, `capSidewall`, `capDrainToChannel`, `capOx`, `capPolywire`
- **Mobility**: `effectiveElectronMobility`, `effectiveHoleMobility`
- **Currents**: `currentOnNmos`, `currentOnPmos`, `currentOffNmos`, `currentOffPmos`
- **Ratios**: `pnSizeRatio`, `effectiveResistanceMultiplier`

### Memory Cell Parameters (from MemCell.py)
- **Dimensions**: `cellArea`, `cellAspectRatio`, `cellWidthInFeatureSize`, `cellHeightInFeatureSize`
- **Resistance**: `resistanceOn`, `resistanceOff`
- **Capacitance**: `capacitanceOn`, `capacitanceOff`, `capDRAMCell`
- **Read**: `readVoltage`, `readCurrent`, `readPower`, `minSenseVoltage`, `wordlineBoostRatio`
- **Write (Reset)**: `resetVoltage`, `resetCurrent`, `resetPulse`, `resetEnergy`
- **Write (Set)**: `setVoltage`, `setCurrent`, `setPulse`, `setEnergy`
- **Access Device**: `widthAccessCMOS`, `widthSRAMCellNMOS`, `widthSRAMCellPMOS`, `voltageDropAccessDevice`, `leakageCurrentAccessDevice`
- **FBRAM**: `gateOxThicknessFactor`, `widthSOIDevice`

### Wire Parameters (from Wire.py)
- **Dimensions**: `wirePitch`, `wireWidth`, `wireThickness`, `wireSpacing`
- **Material**: `wireResistivity`, `barrierThickness`, `dishingThickness`, `alphaScatter`
- **Dielectric**: `ildThickness`, `millerValue`, `horizontalDielectric`, `verticalDielectric`, `fringeCap`
- **Parasitics**: `wireResistance`, `wireCapacitance`

### TSV Parameters (for 3D integration)
- **Dimensions**: `tsvPitch`, `tsvDiameter`, `tsvLength`
- **Dielectric**: `tsvDielecThickness`, `tsvContactResistance`, `tsvDepletionWidth`, `tsvLinerDielectricConstant`
- **Parasitics**: `tsvResistance`, `tsvCapacitance`, `tsvArea`

### Input Parameters (from InputParameter.py)
- **Simulation**: `temperature`, `processNode`, `maxNmosSize`, `maxDriverCurrent`
- **Memory**: `capacity`, `wordWidth`, `associativity`
- **3D**: `stackedDieCount`

## Example Calculations

### Gate Capacitance (from formula.py)
```python
# Symbolic formula
gate_cap = ((bp.capIdealGate + bp.capOverlap + 3*bp.capFringe) * width +
            bp.phyGateLength * bp.capPolywire)

# Output: capPolywire*phyGateLength + width*(3*capFringe + capIdealGate + capOverlap)
```

### On-Resistance (from formula.py)
```python
# Symbolic formula
resistance = (bp.effectiveResistanceMultiplier * bp.vdd /
              (bp.currentOnNmos * width))

# Output: effectiveResistanceMultiplier*vdd/(currentOnNmos*width)
```

### TSV Resistance (from Technology.py)
```python
from sympy import pi

# Symbolic formula
tsv_resistance = (bp.wireResistivity * bp.tsvLength /
                  (pi * (bp.tsvDiameter/2)**2) +
                  bp.tsvContactResistance)

# Output: wireResistivity*tsvLength/(pi*(tsvDiameter/2)**2) + tsvContactResistance
```

## Running Tests

```bash
cd destiny_3d_cache_python

# Test base parameters (main approach)
python test_base_params.py

# Test symbolic wrapper (optional utilities)
python test_symbolic.py
```

## Integration with DESTINY

To integrate symbolic modeling into DESTINY main simulation:

1. **Initialize base parameters in main.py**:
```python
from base_parameters import initialize_base_params

# After initializing technology and cell
bp = initialize_base_params()
bp.populate_from_technology(g.tech)
bp.populate_from_memcell(g.cell)
bp.populate_from_input_parameter(g.inputParameter)
```

2. **Use symbolic expressions in calculations**:
```python
# In formula.py or other calculation files
from base_parameters import get_base_params

bp = get_base_params()
gate_cap_expr = ((bp.capIdealGate + bp.capOverlap + 3*bp.capFringe) * width +
                 bp.phyGateLength * bp.capPolywire)
```

3. **Output symbolic results**:
```python
# In Result.py or output functions
print(f"Gate Capacitance (symbolic): {gate_cap_expr}")
print(f"Gate Capacitance (numerical): {gate_cap_expr.evalf(subs=bp.tech_values)}")
```

## Benefits

1. **Optimization**: Can use symbolic expressions as objectives/constraints
2. **Sensitivity Analysis**: Take derivatives to understand parameter impact
3. **Design Space Exploration**: Express designs symbolically before evaluating
4. **Documentation**: Symbolic expressions show exact relationships
5. **Verification**: Compare symbolic vs numerical to catch errors

## Notes

- Symbolic computation is slower than numerical, so use selectively
- The `tech_values` dictionary maps symbols to concrete values
- Use `.evalf(subs=tech_values)` to evaluate symbolic expressions
- Temperature-dependent currents use single reference value (300K) in symbolic mode
- For branches/conditionals, use concrete values (evaluate first, then branch)

## References

- DESTINY original: https://github.com/cag-zhangshuai/DESTINY
- SymPy documentation: https://docs.sympy.org/
- Based on approach from CACTI codesign framework
