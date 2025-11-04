# Quick Start: Hybrid C++ + Python DESTINY Workflow

## TL;DR

Run this single command to get symbolic expressions for memory access time:

```bash
python hybrid_destiny_workflow.py -c destiny_3d_cache-master/sample_SRAM_2layer.cfg
```

## What This Does

1. **C++ DESTINY** explores 14M cache designs in ~8 minutes
2. **Finds optimal**: 1×1×2 banks, 2×2 mats, 1024×2048 subarray
3. **Python DESTINY** computes symbolic expressions in ~1 second
4. **Shows**:
   - Mathematical formulas for access time
   - Sensitivity analysis (∂t/∂V_dd, ∂t/∂I_on)
   - LaTeX expressions for papers

## Example Output

```
ROW DECODER DELAY: t = 4.5×R_eff×V_dd×C_gate/I_on
BITLINE DELAY: t = 0.5×R_cell×C_cell×rows²
SENSEAMP DELAY: t = V_dd×C_load/I_amp
MUX DELAY: t = R_pass×C_load

TOTAL: t_access = f(V_dd, I_on, C_gate, ...)

Sensitivity:
  ∂t/∂V_dd: Shows impact of voltage on delay
  ∂t/∂I_on: Shows impact of current on delay
```

## Skip C++ DSE (If You Already Have Results)

```bash
cd destiny_3d_cache_python
python symbolic_access_time.py \
    ../destiny_3d_cache-master/cpp_output_sram2layer.txt \
    config/sample_SRAM_2layer.cfg
```

## What You Get

✓ Symbolic expressions showing exact mathematical relationships
✓ Ability to evaluate at different technology nodes instantly
✓ Derivatives for sensitivity analysis
✓ LaTeX formulas ready for papers
✓ Understanding of bottleneck components

## Next Steps

See [HYBRID_WORKFLOW_README.md](HYBRID_WORKFLOW_README.md) for:
- Detailed explanation
- Technology scaling examples
- Voltage scaling analysis
- Design optimization techniques
- Troubleshooting

## Files You Need

```
cacti_destiny_old/
├── hybrid_destiny_workflow.py          ← Main script
├── destiny_3d_cache-master/
│   └── destiny                          ← C++ executable
└── destiny_3d_cache_python/
    ├── parse_cpp_output.py             ← Parser
    ├── symbolic_access_time.py         ← Symbolic engine
    └── config/
        └── sample_SRAM_2layer.cfg      ← Example config
```

## Requirements

- Python 3.x
- SymPy: `pip install sympy`
- C++ DESTINY compiled (should be already done)

---

**Estimated Time**: 8-9 minutes for complete workflow
**Result**: Symbolic expressions + sensitivity analysis + LaTeX formulas
