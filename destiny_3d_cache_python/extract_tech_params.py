#!/usr/bin/env python3
"""
Extract Technology Parameters from Python DESTINY

This extracts all the intermediate parameter values (R_per_cell, C_per_cell, etc.)
so we can substitute them into symbolic expressions and verify they match concrete values.
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
from parse_cpp_output import parse_cpp_destiny_output


def extract_technology_parameters(config_file: str, cpp_config):
    """
    Extract all technology parameters from Python DESTINY
    that correspond to symbolic expression variables
    """
    print("="*80)
    print("EXTRACTING TECHNOLOGY PARAMETERS")
    print("="*80)

    # Initialize Python DESTINY
    print("\n🔧 Initializing Python DESTINY...")

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

    # Create SubArray with optimal configuration
    subarray = SubArray()
    subarray.Initialize(
        cpp_config.subarray_rows,
        cpp_config.subarray_cols,
        1, 1,
        cpp_config.senseamp_mux if cpp_config.senseamp_mux else 1, True,
        cpp_config.output_mux_l1 if cpp_config.output_mux_l1 else 1,
        cpp_config.output_mux_l2 if cpp_config.output_mux_l2 else 1,
        BufferDesignTarget.latency_first,
        cpp_config.num_stacks if cpp_config.num_stacks else 1
    )

    subarray.CalculateArea()
    subarray.CalculateLatency(1e20)

    print("✓ Python DESTINY initialized")

    # Extract parameters
    print("\n" + "="*80)
    print("EXTRACTED PARAMETERS")
    print("="*80)

    params = {}

    # Basic configuration
    print("\n📐 Configuration Parameters:")
    params['rows'] = subarray.numRow
    params['cols'] = subarray.numColumn
    print(f"  rows = {params['rows']}")
    print(f"  cols = {params['cols']}")

    # Technology parameters
    print("\n⚡ Technology Parameters:")
    params['V_dd'] = g.tech.vdd
    params['I_on'] = g.tech.currentOnNmos[0]  # At room temperature
    params['R_eff'] = g.tech.effectiveResistanceMultiplier
    print(f"  V_dd = {params['V_dd']:.4f} V")
    print(f"  I_on = {params['I_on']:.6e} A")
    print(f"  R_eff = {params['R_eff']:.4f}")

    # Gate capacitance
    params['C_gate'] = g.tech.capIdealGate
    print(f"  C_gate = {params['C_gate']:.6e} F/m")

    # Bitline parameters
    print("\n📏 Bitline Parameters:")
    params['R_bitline'] = subarray.resBitline
    params['C_bitline'] = subarray.capBitline
    params['R_per_cell'] = subarray.resBitline / subarray.numRow
    params['C_per_cell'] = subarray.capBitline / subarray.numRow
    print(f"  R_bitline (total) = {params['R_bitline']:.6e} Ω")
    print(f"  C_bitline (total) = {params['C_bitline']:.6e} F")
    print(f"  R_per_cell = {params['R_per_cell']:.6e} Ω/row")
    print(f"  C_per_cell = {params['C_per_cell']:.6e} F/row")

    # Cell access parameters (if available)
    print("\n🔌 Cell Access Parameters:")
    if hasattr(subarray, 'resCellAccess'):
        params['R_access'] = subarray.resCellAccess
        print(f"  R_access = {params['R_access']:.6e} Ω")
    else:
        print(f"  R_access = NOT AVAILABLE")

    # Calculate R_pulldown (same calculation as in SubArray.py line 502)
    from formula import calculate_on_resistance
    NMOS = 0  # Transistor type: 0 = NMOS, 1 = PMOS
    params['R_pulldown'] = calculate_on_resistance(
        g.cell.widthSRAMCellNMOS * g.tech.featureSize,
        NMOS,
        g.inputParameter.temperature,
        g.tech
    )
    print(f"  R_pulldown = {params['R_pulldown']:.6e} Ω")

    if hasattr(subarray, 'capCellAccess'):
        params['C_access'] = subarray.capCellAccess
        print(f"  C_access = {params['C_access']:.6e} F")
    else:
        print(f"  C_access = NOT AVAILABLE")

    # Mux capacitance
    print("\n🔀 Mux Parameters:")
    if hasattr(subarray.bitlineMux, 'capForPreviousDelayCalculation'):
        params['C_mux'] = subarray.bitlineMux.capForPreviousDelayCalculation
        print(f"  C_mux = {params['C_mux']:.6e} F")
    else:
        print(f"  C_mux = NOT AVAILABLE")

    # Sense amplifier parameters
    print("\n📡 Sense Amplifier Parameters:")
    if hasattr(subarray, 'senseVoltage'):
        params['V_sense'] = subarray.senseVoltage
        print(f"  V_sense = {params['V_sense']:.6e} V")
    else:
        print(f"  V_sense = NOT AVAILABLE")

    if hasattr(subarray, 'voltagePrecharge'):
        params['V_precharge'] = subarray.voltagePrecharge
        print(f"  V_precharge = {params['V_precharge']:.6e} V")
    else:
        print(f"  V_precharge = NOT AVAILABLE")

    # Additional parameters for Horowitz model
    print("\n🔬 Horowitz Model Parameters:")
    from formula import calculate_transconductance
    gm = calculate_transconductance(g.cell.widthAccessCMOS * g.tech.featureSize, NMOS, g.tech)
    params['gm'] = gm
    params['beta'] = 1 / (params['R_pulldown'] * gm)
    params['ramp_input'] = subarray.rowDecoder.rampOutput
    print(f"  gm = {params['gm']:.6e} S")
    print(f"  beta = {params['beta']:.6e}")
    print(f"  ramp_input = {params['ramp_input']:.6e} s")

    # Calculated delays
    print("\n⏱️  Calculated Delays (for verification):")
    print(f"  Python bitlineDelay = {subarray.bitlineDelay*1e9:.6f} ns")
    print(f"  C++ bitlineDelay = {cpp_config.bitline_latency*1e9:.6f} ns")
    print(f"  Ratio = {subarray.bitlineDelay / cpp_config.bitline_latency:.3f}×")

    return params, subarray


def verify_with_substitution(params):
    """
    Substitute parameters into symbolic expressions and verify
    """
    from sympy import symbols
    from symbolic_expressions import MemoryAccessTimeExpressions

    print("\n" + "="*80)
    print("SYMBOLIC VERIFICATION WITH SUBSTITUTION")
    print("="*80)

    exprs = MemoryAccessTimeExpressions()

    print("\n🧪 Test 1: C_bitline = C_per_cell × rows")
    C_bitline_expr = exprs.C_bitline
    print(f"  Symbolic: {C_bitline_expr}")

    # Substitute
    if 'C_per_cell' in params and 'rows' in params:
        C_bitline_symbolic = C_bitline_expr.subs({
            exprs.symbols.C_per_cell: params['C_per_cell'],
            exprs.symbols.rows: params['rows']
        })
        C_bitline_value = float(C_bitline_symbolic)
        print(f"  Substituted: {C_bitline_value:.6e} F")
        print(f"  Actual (Python): {params['C_bitline']:.6e} F")
        print(f"  Match: {abs(C_bitline_value - params['C_bitline']) < 1e-15} ✓")
    else:
        print(f"  ✗ Missing parameters")

    print("\n🧪 Test 2: R_bitline = R_per_cell × rows")
    R_bitline_expr = exprs.R_bitline
    print(f"  Symbolic: {R_bitline_expr}")

    # Substitute
    if 'R_per_cell' in params and 'rows' in params:
        R_bitline_symbolic = R_bitline_expr.subs({
            exprs.symbols.R_per_cell: params['R_per_cell'],
            exprs.symbols.rows: params['rows']
        })
        R_bitline_value = float(R_bitline_symbolic)
        print(f"  Substituted: {R_bitline_value:.6e} Ω")
        print(f"  Actual (Python): {params['R_bitline']:.6e} Ω")
        print(f"  Match: {abs(R_bitline_value - params['R_bitline']) < 1e-6} ✓")
    else:
        print(f"  ✗ Missing parameters")

    print("\n🧪 Test 3: tau_bitline (RC time constant)")
    tau_expr = exprs.tau_bitline
    print(f"  Symbolic: {tau_expr}")

    # Check what symbols we need
    required_symbols = sorted([str(s) for s in tau_expr.free_symbols])
    print(f"  Required symbols: {required_symbols}")

    # Check what we have
    available = [s for s in required_symbols if s in params or s in ['rows', 'cols']]
    missing = [s for s in required_symbols if s not in params and s not in ['rows', 'cols']]

    print(f"  Available: {available}")
    print(f"  Missing: {missing}")

    if not missing:
        # Build substitution dict using xreplace
        subs_dict = {}
        for sym_name in required_symbols:
            sym = getattr(exprs.symbols, sym_name)
            if sym_name in params:
                subs_dict[sym] = params[sym_name]

        print(f"\n  Substituting into tau_bitline...")
        tau_substituted = tau_expr.xreplace(subs_dict)

        if tau_substituted.free_symbols:
            print(f"  ✗ Still has symbols: {tau_substituted.free_symbols}")
        else:
            tau_value = float(tau_substituted)
            print(f"  ✓ tau (RC constant) = {tau_value:.6e} s = {tau_value*1e9:.6f} ns")
    else:
        print(f"  ✗ Cannot evaluate - missing {len(missing)} parameters")
        tau_value = None

    # Now test with logarithmic factor
    print("\n🧪 Test 4: tau_bitline_with_log (includes voltage swing)")
    tau_log_expr = exprs.tau_bitline_with_log

    required_symbols = sorted([str(s) for s in tau_log_expr.free_symbols])
    missing = [s for s in required_symbols if s not in params and s not in ['rows', 'cols']]

    if not missing:
        subs_dict = {}
        for sym_name in required_symbols:
            sym = getattr(exprs.symbols, sym_name)
            if sym_name in params:
                subs_dict[sym] = params[sym_name]

        tau_log_substituted = tau_log_expr.xreplace(subs_dict)

        if not tau_log_substituted.free_symbols:
            tau_log_value = float(tau_log_substituted)
            print(f"  ✓ tau (with log) = {tau_log_value:.6e} s = {tau_log_value*1e9:.6f} ns")

            # Now apply Horowitz model
            print("\n🧪 Test 5: Apply Horowitz model")
            import math
            from formula import horowitz

            if 'beta' in params and 'ramp_input' in params:
                bitline_delay, ramp_output = horowitz(tau_log_value, params['beta'], params['ramp_input'])
                print(f"  ✓ Horowitz delay = {bitline_delay:.6e} s = {bitline_delay*1e9:.6f} ns")
                print(f"  Python bitlineDelay = {params.get('bitlineDelay', 0)*1e9:.6f} ns")

                if 'bitlineDelay' in params:
                    error = abs(bitline_delay - params['bitlineDelay']) / params['bitlineDelay'] * 100
                    print(f"  Error = {error:.2f}%")
                    if error < 1:
                        print(f"  ✅ MATCH!")
            else:
                print(f"  ✗ Missing Horowitz parameters")
        else:
            print(f"  ✗ Still has symbols: {tau_log_substituted.free_symbols}")
    else:
        print(f"  ✗ Cannot evaluate - missing {missing}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_tech_params.py <config_file> <cpp_output_file>")
        print("\nExample:")
        print("  python extract_tech_params.py config/sample_SRAM_2layer.cfg ../destiny_3d_cache-master/cpp_output_sram2layer.txt")
        sys.exit(1)

    config_file = sys.argv[1]
    cpp_output_file = sys.argv[2]

    print("="*80)
    print("PARAMETER EXTRACTION AND VERIFICATION")
    print("="*80)
    print(f"\nConfig: {config_file}")
    print(f"C++ Output: {cpp_output_file}")

    # Parse C++ output
    cpp_config = parse_cpp_destiny_output(cpp_output_file)

    # Extract parameters from Python DESTINY
    params, subarray = extract_technology_parameters(config_file, cpp_config)

    # Store calculated delay
    params['bitlineDelay'] = subarray.bitlineDelay

    # Verify with substitution
    verify_with_substitution(params)

    print("\n" + "="*80)
    print("✓ EXTRACTION AND VERIFICATION COMPLETE")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
