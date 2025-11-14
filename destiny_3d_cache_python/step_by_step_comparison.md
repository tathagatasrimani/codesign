# Step-by-Step Comparison: C++ vs Python DESTINY

## Goal
Find ALL differences causing the 2× discrepancy between C++ and Python DESTINY for bitline delay.

## Configuration
- **ProcessNode:** 65nm
- **DeviceRoadmap:** LOP
- **Temperature:** 350K
- **Subarray:** 1024×2048

## Results Summary

| Component | C++ DESTINY | Python DESTINY | Ratio |
|-----------|-------------|----------------|-------|
| Bitline delay | 1.990 ns | 4.038 ns | 2.029× |
| Decoder delay | 1.200 ns | 2.532 ns | 2.110× |

## Investigation Plan

### ✅ Step 1: Technology Parameters (DONE)
**Finding:** currentOnNmos differs by 1.073×
- C++ uses correct temperature-indexed values
- Python may have indexing bug or different source data

### 🔄 Step 2: Wire Resistance/Capacitance
**Need to check:**
- Wire.cpp vs Wire.py initialization
- resBitlinePerRow calculation
- capBitlinePerRow calculation

### 🔄 Step 3: Cell Parameters
**Need to check:**
- MemCell loading from .cell file
- widthAccessCMOS, widthSRAMCellNMOS values

### 🔄 Step 4: SubArray Bitline Calculation
**Need to check:**
- resCellAccess calculation
- capCellAccess calculation
- resBitline total
- capBitline total
- tau calculation
- Horowitz model application

## Findings

### Finding #1: currentOnNmos Mismatch
- **C++ value:** 524.5 A (@ 350K)
- **Python value:** 562.9 A (@ 350K)
- **Impact:** Affects transistor on-resistance → ~7% difference
- **Root cause:** Python using wrong array indexing or different source data

This alone doesn't explain 2× - need to keep investigating!

## Next Steps
1. Check Wire model differences
2. Check if bitline R/C calculation methods differ
3. Check Horowitz model implementation
4. Look for systematic scaling factors
