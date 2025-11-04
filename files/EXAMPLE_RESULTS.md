# Example: Complete Hybrid Workflow Results
## Sample SRAM 2-Layer Cache Configuration

This document shows the complete results from running the hybrid C++ DESTINY + Python DESTINY workflow on `sample_SRAM_2layer.cfg`.

---

## STEP 1: C++ DESTINY Design Space Exploration (DSE)

### Input Configuration

```
Design Target: Cache
Capacity: 2MB
Cache Line Size: 32 Bytes
Cache Associativity: 1 Way (Direct-mapped)
Technology: 65nm
Optimization Target: Write Energy-Delay-Product
```

### DSE Statistics

```
Design Space Explored: 13,977,846 designs
Valid Solutions Found: 463,939 solutions
Time: ~8 minutes
```

### Optimal Configuration Found

#### Cache Data Array

**Organization:**
```
Bank Organization: 1 × 1 × 2 (X × Y × Stacks)
  - Row Activation: 1/1 × 1
  - Column Activation: 1/1 × 1

Mat Organization: 2 × 2
  - Row Activation: 1/2
  - Column Activation: 1/2
  - Subarray Size: 1024 Rows × 2048 Columns

Mux Levels:
  - Senseamp Mux: 1
  - Output Level-1 Mux: 1
  - Output Level-2 Mux: 8
```

**Memory Cell:**
```
Type: SRAM
Cell Area: 146 F² (14.6F × 10.0F)
Cell Aspect Ratio: 1.460
Access Transistor Width: 1.310F
NMOS Width: 2.080F
PMOS Width: 1.230F
```

**Wire Configuration:**
```
Local Wire: Local Aggressive, No Repeaters
Global Wire: Global Aggressive, No Repeaters
Buffer Style: Latency-Optimized
```

### Performance Results from C++ DESTINY

**Area:**
```
Total Data Array Area: 1.956mm × 2.856mm = 5.592 mm²
  Mat Area: 1.956mm × 2.856mm = 5.587 mm² (185.241% efficiency)
  Subarray Area: 978.243μm × 1.410mm = 1.379 mm²
  TSV Area: 10.240 μm²
Area Efficiency: 185.061%
```

**Timing (Cache Data Array):**
```
Read Latency: 3.740 ns
  ├─ TSV Latency:        0.003 ps
  ├─ H-Tree Latency:     0.000 ps
  └─ Mat Latency:        3.740 ns
      ├─ Predecoder:     519.873 ps
      └─ Subarray:       3.220 ns
          ├─ Row Decoder:    1.200 ns  ← Decoding row address
          ├─ Bitline:        1.990 ns  ← Charging bitlines
          ├─ Senseamp:       6.755 ps  ← Sensing voltage difference
          ├─ Mux:           24.213 ps  ← Multiplexing output
          └─ Precharge:      3.104 ns  ← Precharging for next access

Write Latency: 3.740 ns
```

**Power:**
```
Read Dynamic Energy:    122.409 pJ per access
Write Dynamic Energy:    20.774 pJ per access
Leakage Power:          113.063 mW
```

**Bandwidth:**
```
Read Bandwidth:  6.245 GB/s
Write Bandwidth: 9.937 GB/s
```

---

## STEP 2: Python DESTINY Symbolic Analysis

Using the optimal configuration from C++ DESTINY, Python DESTINY computes symbolic expressions for each component:

### 2.1 Row Decoder Delay

**Configuration Analyzed:**
- Subarray rows: 1024
- Address bits to decode: 10
- Number of decoder stages: 3 (hierarchical decoding)

**Symbolic Expression:**

```
R_decoder = (effectiveResistanceMultiplier × Vdd) / (2 × I_on_nmos)

C_gate = featureSize × (6.0×C_fringe + 2.0×C_gate_ideal + 2.0×C_overlap)

C_wire ≈ 2 × C_gate  [simplified fanout-based model]

Delay_per_stage = R_decoder × (C_gate + C_wire)
                = 1.5 × R_eff × featureSize × Vdd ×
                  (6.0×C_fringe + 2.0×C_gate + 2.0×C_overlap) / I_on

Total_Decoder_Delay = 3 stages × Delay_per_stage
                    = 4.5 × effectiveResistanceMultiplier × featureSize × Vdd ×
                      (6.0×capFringe + 2.0×capIdealGate + 2.0×capOverlap) / currentOnNmos
```

**Physical Interpretation:**
- **Linear in Vdd**: Higher voltage → longer delay (charging time increases)
- **Inverse in I_on**: Higher drive current → shorter delay
- **Linear in featureSize**: Smaller technology → faster
- **3 stages**: Hierarchical decoding for 1024 rows (2^10)

**Comparison with C++ DESTINY:**
- C++ DESTINY: 1.200 ns (detailed SPICE-level model)
- Symbolic Model: Simplified analytical model

---

### 2.2 Bitline Delay

**Configuration Analyzed:**
- Bitline length: 1024 rows
- Each cell adds resistance and capacitance

**Symbolic Expression:**

```
R_cell = 0.763 × effectiveResistanceMultiplier × Vdd / currentOnNmos

C_cell = featureSize × (capIdealGate + capOverlap)

R_bitline = R_cell × 1024 rows
          = 781.68 × R_eff × Vdd / I_on

C_bitline = C_cell × 1024 rows
          = 1024 × featureSize × (C_gate + C_overlap)

Bitline_Delay = 0.5 × R_bitline × C_bitline  [distributed RC line]
              = 400,220 × effectiveResistanceMultiplier × featureSize × Vdd ×
                (capIdealGate + capOverlap) / currentOnNmos
```

**Physical Interpretation:**
- **Distributed RC delay**: 0.5 factor accounts for distributed line
- **Quadratic in rows**: Delay ∝ rows² (both R and C increase linearly)
- **Dominant component**: Large coefficient (400K) shows bitline is slowest
- **1024 rows**: Long bitline → significant RC delay

**Comparison with C++ DESTINY:**
- C++ DESTINY: 1.990 ns (61.8% of total subarray delay)
- This confirms bitline is the critical path component

---

### 2.3 Sense Amplifier Delay

**Symbolic Expression:**

```
C_load = featureSize × (12×capFringe + 4×capIdealGate + 4×capOverlap)
       [Load capacitance of next stage]

I_amp = 4 × currentOnNmos
      [Amplifier uses 4F-width transistor]

Senseamp_Delay = Vdd × C_load / I_amp
               = featureSize × Vdd × (3×capFringe + capIdealGate + capOverlap) / currentOnNmos
```

**Physical Interpretation:**
- **CV/I charging model**: Time to charge load capacitance
- **Linear in Vdd**: More voltage swing → more time
- **Small delay**: Fast because small load capacitance

**Comparison with C++ DESTINY:**
- C++ DESTINY: 6.755 ps (0.2% of total - negligible)

---

### 2.4 Multiplexer Delay

**Configuration Analyzed:**
- Senseamp Mux: 1 (no muxing at senseamp)
- Output L2 Mux: 8:1 (one level of muxing)
- Total mux levels: 1

**Symbolic Expression:**

```
R_pass = effectiveResistanceMultiplier × Vdd / (2 × currentOnNmos)
       [Pass transistor resistance]

C_load = featureSize × (6×capFringe + 2×capIdealGate + 2×capOverlap)

Mux_Delay_per_level = R_pass × C_load
                    = effectiveResistanceMultiplier × featureSize × Vdd ×
                      (6×capFringe + 2×capIdealGate + 2×capOverlap) / (2×currentOnNmos)

Total_Mux_Delay = 1 level × Delay_per_level
                = effectiveResistanceMultiplier × featureSize × Vdd ×
                  (3×capFringe + capIdealGate + capOverlap) / currentOnNmos
```

**Physical Interpretation:**
- **Pass gate RC delay**: Simple RC model
- **Single level**: 8:1 mux in one stage
- **Small delay**: Only ~0.75% of total

**Comparison with C++ DESTINY:**
- C++ DESTINY: 24.213 ps (0.75% of total)

---

### 2.5 Total Access Time (Symbolic)

**Complete Expression:**

```
t_access = t_decoder + t_bitline + t_senseamp + t_mux

Simplified form:
t_access = (featureSize × Vdd / currentOnNmos) × [
    30.0 × capFringe × effectiveResistanceMultiplier +
    3.0 × capFringe +
    400,229.85 × capIdealGate × effectiveResistanceMultiplier +
    1.0 × capIdealGate +
    400,229.85 × capOverlap × effectiveResistanceMultiplier +
    1.0 × capOverlap
]
```

**Factored form showing dependencies:**

```
t_access = K × (featureSize × Vdd / I_on) × (capacitances)

where K ≈ 400,230 (dominated by bitline terms)
```

**Comparison:**
- **C++ DESTINY Subarray Latency**: 3.220 ns
- **Symbolic Model**: Analytical approximation

---

## STEP 3: Sensitivity Analysis

### 3.1 Voltage Sensitivity (∂t/∂Vdd)

**Expression:**

```
∂t_access/∂Vdd = (featureSize / currentOnNmos) × [
    30.0 × capFringe × effectiveResistanceMultiplier +
    3.0 × capFringe +
    400,229.85 × capIdealGate × effectiveResistanceMultiplier +
    1.0 × capIdealGate +
    400,229.85 × capOverlap × effectiveResistanceMultiplier +
    1.0 × capOverlap
]
```

**Interpretation:**
- **Positive derivative**: Increasing voltage increases delay
- **Linear relationship**: ∂t/∂V is constant (delay ∝ Vdd linearly)
- **Design implication**: Can't reduce delay by increasing Vdd (unusual!)
- **Reason**: In this model, R ∝ Vdd (threshold voltage effects dominate)

### 3.2 Current Sensitivity (∂t/∂I_on)

**Expression:**

```
∂t_access/∂I_on = -(featureSize × Vdd / I_on²) × [
    30.0 × capFringe × effectiveResistanceMultiplier +
    3.0 × capFringe +
    400,229.85 × capIdealGate × effectiveResistanceMultiplier +
    1.0 × capIdealGate +
    400,229.85 × capOverlap × effectiveResistanceMultiplier +
    1.0 × capOverlap
]
```

**Interpretation:**
- **Negative derivative**: Increasing current decreases delay (expected!)
- **Inverse square relationship**: t ∝ 1/I_on, so ∂t/∂I ∝ 1/I²
- **Design implication**: Doubling drive current → 2× speedup
- **Strongest optimization lever**: Improve I_on (wider transistors, better process)

---

## STEP 4: Design Insights

### Component Breakdown (from C++ DESTINY)

```
Component          | Delay (ns) | % of Total | Optimization Priority
-------------------|------------|------------|----------------------
Row Decoder        | 1.200      | 37.3%      | Medium
Bitline            | 1.990      | 61.8%      | ★★★ HIGH (Critical!)
Sense Amplifier    | 0.007      | 0.2%       | Low (negligible)
Multiplexer        | 0.024      | 0.7%       | Low
-------------------|------------|------------|----------------------
TOTAL (Subarray)   | 3.220      | 100%       |
```

### Key Bottleneck: **BITLINE DELAY**

**Why bitline dominates:**
1. **Long bitline**: 1024 rows = ~100μm length at 65nm
2. **RC delay quadratic**: Both R and C scale with length
3. **Coefficient 400K**: Much larger than other components

**Optimization strategies:**
```
Strategy                          | Impact on Bitline Delay
----------------------------------|----------------------------------
Reduce subarray rows              | Quadratic improvement (best!)
  - 1024 → 512 rows              | → 4× faster bitline
  - 1024 → 256 rows              | → 16× faster bitline

Increase drive current (I_on)     | Linear improvement
  - Better process technology     | → inversely proportional
  - Wider access transistors      | → but increases area

Reduce capacitance                | Linear improvement
  - Better interconnect tech      | → lower C_bitline
  - Low-k dielectrics            | → but expensive

Multi-level bitlines              | Logarithmic improvement
  - Hierarchical sensing          | → adds complexity
```

### Symbolic Expressions Enable:

1. **Technology Scaling Prediction**
   ```python
   # No need to re-run C++ DSE!
   for node in [65, 45, 32, 22, 16]:
       # Just substitute new tech parameters
       delay_at_node = t_access.evalf(subs=tech_values[node])
   ```

2. **Voltage Optimization**
   ```python
   # Explore V_dd scaling for power-performance tradeoff
   for v_scale in [0.7, 0.8, 0.9, 1.0]:
       delay = t_access.evalf(V_dd = nominal * v_scale)
       power = p_dynamic.evalf(V_dd = nominal * v_scale)
       # Power ∝ V², Delay ∝ V (in our model)
   ```

3. **Design Space Insights**
   ```python
   # Which parameter has biggest impact?
   sensitivities = {
       'V_dd': abs(diff(t_access, V_dd)),
       'I_on': abs(diff(t_access, I_on)),
       'C_gate': abs(diff(t_access, C_gate)),
       # ... etc
   }
   # → Shows bitline capacitance is critical
   ```

---

## LaTeX Expression (for Papers)

For academic papers, the complete symbolic access time expression in LaTeX:

```latex
t_{access} = \frac{f \cdot V_{dd}}{I_{on}} \left(
    30 C_{fringe} R_{eff} + 3 C_{fringe} +
    400{,}230 C_{gate} R_{eff} + C_{gate} +
    400{,}230 C_{overlap} R_{eff} + C_{overlap}
\right)
```

where:
- $f$ = feature size
- $V_{dd}$ = supply voltage
- $I_{on}$ = on-current per micrometer
- $R_{eff}$ = effective resistance multiplier
- $C_{gate}$, $C_{fringe}$, $C_{overlap}$ = technology-dependent capacitances

---

## Summary

### C++ DESTINY Contributions:
✓ Explored 14M designs in 8 minutes
✓ Found optimal configuration: 1×1×2 banks, 2×2 mats, 1024×2048 subarray
✓ Detailed timing: 3.74ns total, 3.22ns subarray (1.99ns bitline critical)
✓ Area: 5.59 mm², Power: 113 mW leakage

### Python DESTINY Contributions:
✓ Symbolic expressions showing t_access = f(V_dd, I_on, C_gate, ...)
✓ Sensitivity: ∂t/∂V_dd (positive), ∂t/∂I_on (negative, inverse square)
✓ Design insights: Bitline delay dominates (61.8%), reduce rows for speedup
✓ LaTeX formulas ready for publication

### Hybrid Workflow Benefits:
🎯 **Fast optimization** (C++) + **Mathematical insight** (Python)
🎯 **Numerical accuracy** + **Symbolic understanding**
🎯 **One-time DSE** + **Infinite parameter exploration**

This is the power of combining both approaches!
