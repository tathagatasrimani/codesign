# Fix Summary: Python DESTINY Now Matches C++ DESTINY

## Problem
Python DESTINY was producing bitline delays that were 2× higher than C++ DESTINY:
- Python: 4.038 ns
- C++: 1.990 ns
- Ratio: 2.029×

## Root Cause
The bug was **NOT** in the Wire resistance/capacitance calculations (those were identical).

The bug was in **SubArray.py line 217-220**: Python was incorrectly multiplying wire R/C by `num3DLevels = 2` when it should use `1` for intra-die wires.

### Why This Happened
- Config file specifies `-StackedDieCount: 2` for 2-layer 3D stacking
- Python read this and passed `num3DLevels = 2` to SubArray
- Python then multiplied bitline/wordline wire R/C by 2
- **But:** `num3DLevels` multiplier is for TSV (Through-Silicon Via) connections between dies, NOT for wire segments within a single die
- C++ correctly uses `num3DLevels = 1` for these calculations

## The Fix

**File:** `SubArray.py` lines 217-223

**Before:**
```python
self.capWordline = self.lenWordline * g.localWire.capWirePerUnit * self.num3DLevels
self.resWordline = self.lenWordline * g.localWire.resWirePerUnit * self.num3DLevels
self.capBitline = self.lenBitline * g.localWire.capWirePerUnit * self.num3DLevels
self.resBitline = self.lenBitline * g.localWire.resWirePerUnit * self.num3DLevels
```

**After:**
```python
# NOTE: For intra-die wires (wordline/bitline within a subarray), we use 1 instead of num3DLevels
# The num3DLevels multiplier is for TSV (Through-Silicon Via) connections between dies,
# not for the wire segments within a single die. This matches C++ DESTINY behavior.
self.capWordline = self.lenWordline * g.localWire.capWirePerUnit * 1
self.resWordline = self.lenWordline * g.localWire.resWirePerUnit * 1
self.capBitline = self.lenBitline * g.localWire.capWirePerUnit * 1
self.resBitline = self.lenBitline * g.localWire.resWirePerUnit * 1
```

## Results

### Before Fix
```
Python:
  resBitline = 2.932e+03 Ω (2.000× higher than C++)
  capBitline = 6.208e-13 F (1.852× higher than C++)
  bitlineDelay = 4.038 ns (2.029× higher than C++)
```

### After Fix
```
Python:
  resBitline = 1.466e+03 Ω (1.000× ratio - MATCHES C++)
  capBitline = 3.351e-13 F (1.000× ratio - MATCHES C++)
  bitlineDelay = 1.989 ns  (1.000× ratio - MATCHES C++)

C++:
  resBitline = 1.466e+03 Ω
  capBitline = 3.351e-13 F
  bitlineDelay = 1.990 ns

Error: 0.05% (essentially identical!)
```

## Verification

Wire calculations were verified to be identical:
```
Python Wire.py (65nm local_aggressive):
  resWirePerUnit = 1.509e+06 Ω/m ✅
  capWirePerUnit = 2.939e-10 F/m ✅

C++ Wire.cpp (65nm local_aggressive):
  resWirePerUnit = 1.509e+06 Ω/m ✅
  capWirePerUnit = 2.939e-10 F/m ✅
```

## Impact

✅ Python DESTINY now produces numerically identical results to C++ DESTINY
✅ Symbolic expressions remain correct and real (extracted from actual DESTINY formulas)
✅ No calibration factors needed
✅ Root cause fixed at the source

## Test Configuration

- **Process Node:** 65nm LOP
- **Subarray:** 1024 rows × 2048 columns
- **Memory Type:** SRAM
- **Config:** sample_SRAM_2layer.cfg
