# Hybrid C++ DESTINY + Python DESTINY Workflow

## Overview

This hybrid workflow combines the strengths of both C++ DESTINY and Python DESTINY:

1. **C++ DESTINY**: Performs fast Design Space Exploration (DSE) to find optimal cache configurations
2. **Python DESTINY**: Uses the optimal configuration to compute symbolic expressions for memory access time

```
Input Config → C++ DESTINY (DSE) → Optimal Config → Python DESTINY (Symbolic) → Expressions
```

## Why This Approach?

### C++ DESTINY Advantages:
- Fast execution (compiled C++)
- Comprehensive design space exploration
- Proven and validated models
- Explores 100,000+ configurations efficiently

### Python DESTINY Advantages:
- Symbolic mathematics with SymPy
- Sensitivity analysis (derivatives)
- Technology scaling predictions
- Design space insights
- Easy parameter sweeps

### Combined Benefits:
✓ Get optimal configuration quickly (C++)
✓ Understand WHY it's optimal (Python symbolic)
✓ Perform "what-if" analysis on optimal design
✓ Generate mathematical expressions for papers

## Quick Start

### Run Complete Workflow

```bash
cd /path/to/cacti_destiny_old
python hybrid_destiny_workflow.py -c config/sample_SRAM_2layer.cfg
```

This will:
1. Run C++ DESTINY DSE (may take several minutes)
2. Extract optimal configuration
3. Compute symbolic expressions for access time
4. Display sensitivity analysis

### Run Only Python Symbolic Analysis

If you already have C++ DESTINY output:

```bash
cd destiny_3d_cache_python
python symbolic_access_time.py \
    ../destiny_3d_cache-master/cpp_output.txt \
    config/sample_SRAM_2layer.cfg
```

## Workflow Components

### 1. C++ DESTINY (Design Space Exploration)

**Input**: Configuration file (e.g., `sample_SRAM_2layer.cfg`)

**Process**:
- Explores bank organizations (1x1 to 64x64)
- Tests mat configurations (1x1 to 8x8)
- Varies subarray sizes
- Tries different mux levels
- Evaluates ~14M designs, finds ~464K valid solutions

**Output**: Text file with optimal configuration
- Bank organization: 1 x 1 x 2
- Mat organization: 2 x 2
- Subarray size: 1024 Rows x 2048 Columns
- Timing breakdown for all components

### 2. Configuration Parser (`parse_cpp_output.py`)

Extracts from C++ output:
- Bank/Mat/Subarray organization
- Mux levels
- Timing breakdown (decoder, bitline, senseamp, mux)
- Cell parameters
- Wire types

### 3. Symbolic Access Time Calculator (`symbolic_access_time.py`)

Using optimal configuration, computes symbolic expressions for:

#### Row Decoder Delay
```
t_decoder = num_stages × R_transistor × (C_gate + C_wire)
```

#### Bitline Delay
```
t_bitline = 0.5 × R_bitline × C_bitline
where R_bitline = R_cell × num_rows
      C_bitline = C_cell × num_rows
```

#### Sense Amplifier Delay
```
t_senseamp = V_dd × C_load / I_amp
```

#### Multiplexer Delay
```
t_mux = num_mux_levels × R_pass × C_load
```

#### Total Access Time
```
t_access = t_decoder + t_bitline + t_senseamp + t_mux
```

### 4. Integration Script (`hybrid_destiny_workflow.py`)

Orchestrates the complete workflow:
- Runs C++ DESTINY
- Parses output
- Calls Python symbolic analysis
- Displays results

## Example Output

```
================================================================================
SYMBOLIC ACCESS TIME ANALYSIS
================================================================================

ROW DECODER DELAY
─────────────────────────────────────────────────────────────────────────────
Configuration:
  Subarray rows: 1024
  Address bits to decode: 10

📐 Symbolic Expression:
   Total Decoder Delay = 4.5*effectiveResistanceMultiplier*featureSize*vdd*
                         (6.0*capFringe + 2.0*capIdealGate + 2.0*capOverlap)/
                         currentOnNmos

📊 Numerical Result:
   Row Decoder Delay = 0.XXX ns
   C++ DESTINY Result = 1.200 ns

BITLINE DELAY
─────────────────────────────────────────────────────────────────────────────
📐 Symbolic Expression:
   Bitline Delay = 0.5 × R_bitline × C_bitline
                 = 400220*effectiveResistanceMultiplier*featureSize*vdd*
                   (capIdealGate + capOverlap)/currentOnNmos

📊 Numerical Result:
   C++ DESTINY Result = 1.990 ns

SUMMARY
─────────────────────────────────────────────────────────────────────────────
Complete symbolic expression for total access time with all parameters

🔍 Sensitivity Analysis:
   ∂t/∂V_dd = ... (shows how delay changes with voltage)
   ∂t/∂I_on = ... (shows how delay changes with current)

📝 LaTeX Expression: (ready for papers)
```

## Use Cases

### 1. Technology Scaling Analysis

```python
# After getting symbolic expressions, evaluate at different nodes
for node in [65, 45, 32, 22]:
    g.tech.Initialize(node, DeviceRoadmap.HP, InputParameter())
    bp.populate_from_technology(g.tech)
    delay = t_access.evalf(subs=bp.tech_values)
    print(f"{node}nm: {delay*1e9:.3f} ns")
```

### 2. Voltage Scaling

```python
# Understand power-performance tradeoffs
for v_scale in [0.8, 0.9, 1.0, 1.1]:
    bp.tech_values[bp.vdd] *= v_scale
    delay = t_access.evalf(subs=bp.tech_values)
    power = dynamic_power.evalf(subs=bp.tech_values)
    print(f"V×{v_scale}: delay={delay*1e9:.3f}ns, power={power*1e3:.3f}mW")
```

### 3. Design Optimization

```python
# Find bottleneck components
from sympy import diff
sensitivity_vdd = diff(t_access, bp.vdd)
sensitivity_ion = diff(t_access, bp.currentOnNmos)

# Shows which parameter changes have biggest impact
```

### 4. Generate Expressions for Papers

```python
from sympy import latex
print(latex(simplify(t_access)))
# Copy-paste LaTeX into your paper
```

## Files Created

```
cacti_destiny_old/
├── hybrid_destiny_workflow.py          # Main integration script
└── destiny_3d_cache_python/
    ├── parse_cpp_output.py            # Parse C++ output
    ├── symbolic_access_time.py        # Compute symbolic expressions
    └── symbolic_results.txt           # Example output
```

## Command Reference

### Full Workflow
```bash
python hybrid_destiny_workflow.py -c config/sample_SRAM_2layer.cfg
```

### Skip C++ DSE (use existing output)
```bash
python hybrid_destiny_workflow.py -c config.cfg --skip-cpp -o existing_output.txt
```

### Custom paths
```bash
python hybrid_destiny_workflow.py \
    -c my_config.cfg \
    --cpp-destiny path/to/destiny \
    --output my_results.txt
```

### Parse existing C++ output
```bash
python destiny_3d_cache_python/parse_cpp_output.py cpp_output.txt
```

### Symbolic analysis only
```bash
python destiny_3d_cache_python/symbolic_access_time.py \
    cpp_output.txt \
    config.cfg
```

## Performance

- **C++ DESTINY DSE**: ~8 minutes for 2MB SRAM cache (explores 14M designs)
- **Python Symbolic Analysis**: ~1 second
- **Total Workflow**: ~8-9 minutes

## Future Enhancements

Possible extensions:
- [ ] Add power analysis (static + dynamic)
- [ ] Include area modeling
- [ ] Add H-tree interconnect delay
- [ ] Model TSV delay for 3D stacks
- [ ] Generate optimization constraints
- [ ] Export to Jupyter notebook for interactive analysis
- [ ] Add plotting capabilities for parameter sweeps

## Comparison: Numerical vs Symbolic

| Aspect | C++ DESTINY (Numerical) | Python DESTINY (Symbolic) |
|--------|------------------------|--------------------------|
| Speed | Fast (compiled) | Moderate (interpreted) |
| DSE | ✓ Full exploration | ✗ Single point |
| Symbolic Math | ✗ | ✓ Full expressions |
| Derivatives | ✗ | ✓ Automatic |
| Tech Scaling | ✗ Must re-run | ✓ Just substitute |
| Optimization | ✓ Find optimal | ✓ Understand why |
| Paper Writing | Numbers only | LaTeX expressions |

## Troubleshooting

### Issue: C++ DESTINY not found
```bash
# Check path
ls destiny_3d_cache-master/destiny

# Rebuild if needed
cd destiny_3d_cache-master
make clean
make
```

### Issue: Python import errors
```bash
# Ensure you're in correct directory
cd destiny_3d_cache_python
python symbolic_access_time.py ...
```

### Issue: Mismatched results
The symbolic model uses simplified approximations for:
- Wire capacitance (fanout-based instead of length-based)
- Bitline capacitance (gate cap model instead of detailed SPICE)
- Decoder stages (logarithmic estimate)

This is intentional - the goal is mathematical insight, not exact matching.

## References

- Original C++ DESTINY: https://github.com/git-pb/destiny_3d_cache
- SymPy Documentation: https://docs.sympy.org
- CACTI Paper: "CACTI: An Enhanced Cache Access and Cycle Time Model"

## License

Same as DESTINY - see LICENSE file

## Authors

- C++ DESTINY: Original DESTINY team
- Python DESTINY Port: [Your Name]
- Hybrid Workflow: [Your Name]
