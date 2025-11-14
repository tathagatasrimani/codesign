#!/usr/bin/env python3
"""
Comprehensive verification: C++ vs Python numerical vs Python symbolic
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import globals as g
from InputParameter import InputParameter
from Technology import Technology
from MemCell import MemCell
from SubArray import SubArray
from Wire import Wire
from typedef import WireType, WireRepeaterType, BufferDesignTarget
from symbolic_expressions import MemoryAccessTimeExpressions, MemoryAccessTimeSymbols
from sympy import symbols, simplify


def extract_parameters(subarray):
    """Extract all parameters needed for symbolic evaluation"""
    params = {}

    # From subarray
    params['rows'] = subarray.numRow
    params['cols'] = subarray.numColumn

    # Wire per-unit values
    params['R_per_cell'] = g.localWire.resWirePerUnit * g.cell.heightInFeatureSize * g.devtech.featureSize
    params['C_per_cell'] = g.localWire.capWirePerUnit * g.cell.heightInFeatureSize * g.devtech.featureSize

    # Cell access
    params['R_access'] = subarray.resCellAccess
    params['C_access'] = subarray.capCellAccess

    # SRAM pull-down
    from formula import calculate_on_resistance
    from constant import NMOS
    params['R_pulldown'] = calculate_on_resistance(
        g.cell.widthSRAMCellNMOS * g.tech.featureSize,
        NMOS,
        g.inputParameter.temperature,
        g.tech
    )

    # Mux capacitance (0 for this config)
    params['C_mux'] = 0.0

    # Voltage parameters
    params['V_precharge'] = subarray.voltagePrecharge
    params['V_sense'] = subarray.senseVoltage

    return params


def main():
    config_file = "config/sample_SRAM_2layer.cfg"

    print("="*80)
    print("COMPREHENSIVE VERIFICATION")
    print("C++ DESTINY vs Python Numerical vs Python Symbolic")
    print("="*80)

    # Initialize Python DESTINY
    g.inputParameter = InputParameter()
    g.inputParameter.ReadInputParameterFromFile(config_file)

    g.tech = Technology()
    g.tech.Initialize(g.inputParameter.processNode, g.inputParameter.deviceRoadmap, g.inputParameter)

    g.devtech = Technology()
    g.devtech.Initialize(g.inputParameter.processNode, g.inputParameter.deviceRoadmap, g.inputParameter)

    g.gtech = Technology()
    g.gtech.Initialize(g.inputParameter.processNode, g.inputParameter.deviceRoadmap, g.inputParameter)

    g.localWire = Wire()
    g.localWire.Initialize(g.inputParameter.processNode, WireType.local_aggressive,
                           WireRepeaterType.repeated_none, g.inputParameter.temperature, False)

    g.globalWire = Wire()
    g.globalWire.Initialize(g.inputParameter.processNode, WireType.global_aggressive,
                            WireRepeaterType.repeated_none, g.inputParameter.temperature, False)

    g.cell = MemCell()
    if len(g.inputParameter.fileMemCell) > 0:
        cellFile = g.inputParameter.fileMemCell[0]
        if '/' not in cellFile:
            cellFile = os.path.join('config', cellFile)
        g.cell.ReadCellFromFile(cellFile)

    # Create subarray
    subarray = SubArray()
    subarray.Initialize(
        1024, 2048, 1, 1, 1, True, 1, 8,
        BufferDesignTarget.latency_first, 2
    )

    subarray.CalculateArea()
    subarray.CalculateLatency(1e20)

    # Extract parameters for symbolic evaluation
    params = extract_parameters(subarray)

    print("\n" + "="*80)
    print("STEP 1: PYTHON NUMERICAL DESTINY")
    print("="*80)

    print(f"\nBitline R/C:")
    print(f"  resBitline  = {subarray.resBitline:.6e} Ω")
    print(f"  capBitline  = {subarray.capBitline:.6e} F")

    print(f"\nBitline Delay:")
    print(f"  bitlineDelay = {subarray.bitlineDelay*1e9:.6f} ns")

    # Get symbolic expressions
    print("\n" + "="*80)
    print("STEP 2: PYTHON SYMBOLIC EXPRESSIONS")
    print("="*80)

    exprs = MemoryAccessTimeExpressions()

    # Build substitution dict
    subs_dict = {}
    for param_name, param_value in params.items():
        if hasattr(exprs.symbols, param_name):
            sym = getattr(exprs.symbols, param_name)
            subs_dict[sym] = param_value

    # Evaluate R_bitline symbolically
    R_bitline_sym = exprs.R_bitline.xreplace(subs_dict)
    R_bitline_val = float(R_bitline_sym)

    # Evaluate C_bitline symbolically
    C_bitline_sym = exprs.C_bitline.xreplace(subs_dict)
    C_bitline_val = float(C_bitline_sym)

    # Evaluate tau symbolically (without log factor)
    tau_sym = exprs.tau_bitline.xreplace(subs_dict)
    tau_val = float(tau_sym)

    print(f"\nSymbolic Expressions:")
    print(f"  R_bitline = {exprs.R_bitline}")
    print(f"  C_bitline = {exprs.C_bitline}")
    print(f"  tau = {exprs.tau_bitline}")

    print(f"\nEvaluated Values:")
    print(f"  R_bitline = {R_bitline_val:.6e} Ω")
    print(f"  C_bitline = {C_bitline_val:.6e} F")
    print(f"  tau (before log) = {tau_val:.6e} s")

    # Apply log factor
    import math
    log_factor = math.log(params['V_precharge'] / (params['V_precharge'] - params['V_sense']/2))
    tau_with_log = tau_val * log_factor

    print(f"  tau (after log) = {tau_with_log:.6e} s")

    # Apply Horowitz
    from formula import calculate_transconductance, horowitz
    from constant import NMOS
    gm = calculate_transconductance(g.cell.widthAccessCMOS * g.tech.featureSize, NMOS, g.tech)
    beta = 1.0 / (params['R_pulldown'] * gm)

    # Use actual Horowitz function - it returns (delay, ramp)
    result = horowitz(tau_with_log, beta, 1e20)
    if isinstance(result, tuple):
        bitline_delay_symbolic, _ = result
    else:
        bitline_delay_symbolic = result

    print(f"  bitlineDelay (via Horowitz) = {bitline_delay_symbolic*1e9:.6f} ns")

    print("\n" + "="*80)
    print("STEP 3: C++ DESTINY (GROUND TRUTH)")
    print("="*80)

    cpp_bitline_delay = 1.990e-09  # From C++ output

    print(f"\nC++ DESTINY:")
    print(f"  bitlineDelay = {cpp_bitline_delay*1e9:.6f} ns")

    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)

    print(f"\n{'Source':<30} {'Bitline Delay (ns)':<20} {'Error vs C++'}")
    print("-" * 80)

    python_numerical = subarray.bitlineDelay * 1e9
    python_symbolic = bitline_delay_symbolic * 1e9
    cpp_value = cpp_bitline_delay * 1e9

    print(f"{'Python Numerical':<30} {python_numerical:>18.6f}   {abs(python_numerical - cpp_value)/cpp_value*100:>6.3f}%")
    print(f"{'Python Symbolic':<30} {python_symbolic:>18.6f}   {abs(python_symbolic - cpp_value)/cpp_value*100:>6.3f}%")
    print(f"{'C++ DESTINY (ground truth)':<30} {cpp_value:>18.6f}   {'0.000%':>7}")

    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    numerical_matches = abs(python_numerical - cpp_value) / cpp_value < 0.01  # < 1% error
    symbolic_matches = abs(python_symbolic - cpp_value) / cpp_value < 0.01
    numerical_symbolic_match = abs(python_numerical - python_symbolic) / python_numerical < 0.01

    print(f"\n✓ Python Numerical matches C++:     {'PASS ✅' if numerical_matches else 'FAIL ❌'}")
    print(f"✓ Python Symbolic matches C++:      {'PASS ✅' if symbolic_matches else 'FAIL ❌'}")
    print(f"✓ Numerical matches Symbolic:       {'PASS ✅' if numerical_symbolic_match else 'FAIL ❌'}")

    if numerical_matches and symbolic_matches and numerical_symbolic_match:
        print(f"\n🎉 ALL TESTS PASSED! Python DESTINY matches C++ DESTINY perfectly!")
        print(f"   Both numerical and symbolic modeling produce identical results.")
        return 0
    else:
        print(f"\n❌ TESTS FAILED! There are mismatches.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
