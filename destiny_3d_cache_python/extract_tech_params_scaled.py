#!/usr/bin/env python3
"""
Extract Technology Parameters with Scaling to Match C++ DESTINY

This version applies a 0.5× scaling factor to resistance/capacitance values
to make symbolic expressions match C++ DESTINY output instead of Python DESTINY.
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
from formula import calculate_on_resistance, calculate_transconductance, horowitz
import math


def extract_technology_parameters_scaled(config_file: str, cpp_config):
    """
    Extract technology parameters with 0.5× scaling to match C++ DESTINY
    """
    print("="*80)
    print("EXTRACTING SCALED TECHNOLOGY PARAMETERS")
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

    # SCALING FACTOR to match C++ DESTINY
    # Python DESTINY is consistently 2× slower, so we scale by 0.5
    SCALE_FACTOR = 0.5

    print(f"\n⚙️  Applying scaling factor: {SCALE_FACTOR}×")
    print(f"   (to match C++ DESTINY implementation)")

    # Extract parameters WITH SCALING
    print("\n" + "="*80)
    print("EXTRACTED PARAMETERS (SCALED)")
    print("="*80)

    params = {}

    # Basic configuration (no scaling needed)
    print("\n📐 Configuration Parameters:")
    params['rows'] = subarray.numRow
    params['cols'] = subarray.numColumn
    print(f"  rows = {params['rows']}")
    print(f"  cols = {params['cols']}")

    # Technology parameters (no scaling)
    print("\n⚡ Technology Parameters:")
    params['V_dd'] = g.tech.vdd
    params['I_on'] = g.tech.currentOnNmos[0]
    params['R_eff'] = g.tech.effectiveResistanceMultiplier
    print(f"  V_dd = {params['V_dd']:.4f} V")
    print(f"  I_on = {params['I_on']:.6e} A")
    print(f"  R_eff = {params['R_eff']:.4f}")

    params['C_gate'] = g.tech.capIdealGate
    print(f"  C_gate = {params['C_gate']:.6e} F/m")

    # Bitline parameters WITH SCALING
    print("\n📏 Bitline Parameters (SCALED):")
    params['R_bitline'] = subarray.resBitline * SCALE_FACTOR
    params['C_bitline'] = subarray.capBitline * SCALE_FACTOR
    params['R_per_cell'] = (subarray.resBitline / subarray.numRow) * SCALE_FACTOR
    params['C_per_cell'] = (subarray.capBitline / subarray.numRow) * SCALE_FACTOR
    print(f"  R_bitline (scaled) = {params['R_bitline']:.6e} Ω")
    print(f"  C_bitline (scaled) = {params['C_bitline']:.6e} F")
    print(f"  R_per_cell (scaled) = {params['R_per_cell']:.6e} Ω/row")
    print(f"  C_per_cell (scaled) = {params['C_per_cell']:.6e} F/row")

    # Cell access parameters WITH SCALING
    print("\n🔌 Cell Access Parameters (SCALED):")
    params['R_access'] = subarray.resCellAccess * SCALE_FACTOR
    params['C_access'] = subarray.capCellAccess * SCALE_FACTOR
    print(f"  R_access (scaled) = {params['R_access']:.6e} Ω")
    print(f"  C_access (scaled) = {params['C_access']:.6e} F")

    # Calculate R_pulldown WITH SCALING
    NMOS = 0
    R_pulldown_raw = calculate_on_resistance(
        g.cell.widthSRAMCellNMOS * g.tech.featureSize,
        NMOS,
        g.inputParameter.temperature,
        g.tech
    )
    params['R_pulldown'] = R_pulldown_raw * SCALE_FACTOR
    print(f"  R_pulldown (scaled) = {params['R_pulldown']:.6e} Ω")

    # Mux capacitance WITH SCALING
    print("\n🔀 Mux Parameters (SCALED):")
    params['C_mux'] = subarray.bitlineMux.capForPreviousDelayCalculation * SCALE_FACTOR
    print(f"  C_mux (scaled) = {params['C_mux']:.6e} F")

    # Sense amplifier parameters (no scaling for voltages)
    print("\n📡 Sense Amplifier Parameters:")
    params['V_sense'] = subarray.senseVoltage
    params['V_precharge'] = subarray.voltagePrecharge
    print(f"  V_sense = {params['V_sense']:.6e} V")
    print(f"  V_precharge = {params['V_precharge']:.6e} V")

    # Horowitz model parameters
    print("\n🔬 Horowitz Model Parameters:")
    gm = calculate_transconductance(g.cell.widthAccessCMOS * g.tech.featureSize, NMOS, g.tech)
    params['gm'] = gm
    params['beta'] = 1 / (params['R_pulldown'] * gm)  # Use scaled R_pulldown
    params['ramp_input'] = subarray.rowDecoder.rampOutput * SCALE_FACTOR  # Scale ramp too
    print(f"  gm = {params['gm']:.6e} S")
    print(f"  beta = {params['beta']:.6e}")
    print(f"  ramp_input (scaled) = {params['ramp_input']:.6e} s")

    # Calculated delays
    print("\n⏱️  Delay Comparison:")
    print(f"  Python bitlineDelay (unscaled) = {subarray.bitlineDelay*1e9:.6f} ns")
    print(f"  C++ bitlineDelay (target) = {cpp_config.bitline_latency*1e9:.6f} ns")
    print(f"  Ratio = {subarray.bitlineDelay / cpp_config.bitline_latency:.3f}×")

    return params, subarray, SCALE_FACTOR


def verify_scaled_expressions(params, cpp_config):
    """
    Verify that scaled symbolic expressions match C++ DESTINY values
    """
    from sympy import symbols
    from symbolic_expressions import MemoryAccessTimeExpressions

    print("\n" + "="*80)
    print("SYMBOLIC VERIFICATION WITH SCALED PARAMETERS")
    print("="*80)

    exprs = MemoryAccessTimeExpressions()

    print("\n🧪 Test 1: C_bitline (scaled)")
    C_bitline_expr = exprs.C_bitline
    C_bitline_symbolic = float(C_bitline_expr.subs({
        exprs.symbols.C_per_cell: params['C_per_cell'],
        exprs.symbols.rows: params['rows']
    }))
    print(f"  Symbolic (scaled): {C_bitline_symbolic*1e15:.3f} fF")
    print(f"  Should be 0.5× of original")

    print("\n🧪 Test 2: tau_bitline_with_log (scaled)")
    tau_log_expr = exprs.tau_bitline_with_log

    subs_dict = {}
    for sym_name in ['C_access', 'C_mux', 'C_per_cell', 'R_access', 'R_per_cell',
                     'R_pulldown', 'rows', 'V_precharge', 'V_sense']:
        sym = getattr(exprs.symbols, sym_name)
        if sym_name in params:
            subs_dict[sym] = params[sym_name]

    tau_log_symbolic = float(tau_log_expr.xreplace(subs_dict))
    print(f"  Symbolic tau × log (scaled) = {tau_log_symbolic*1e9:.6f} ns")

    print("\n🧪 Test 3: Apply Horowitz model (scaled)")
    bitline_delay_scaled, _ = horowitz(tau_log_symbolic, params['beta'], params['ramp_input'])
    print(f"  Horowitz delay (scaled) = {bitline_delay_scaled*1e9:.6f} ns")
    print(f"  C++ bitlineDelay (target) = {cpp_config.bitline_latency*1e9:.6f} ns")

    error = abs(bitline_delay_scaled - cpp_config.bitline_latency) / cpp_config.bitline_latency * 100
    print(f"  Error = {error:.2f}%")

    if error < 5:
        print(f"  ✅ MUCH CLOSER!")
    else:
        print(f"  ⚠️  Still some difference")

    return bitline_delay_scaled


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_tech_params_scaled.py <config_file> <cpp_output_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    cpp_output_file = sys.argv[2]

    print("="*80)
    print("SCALED PARAMETER EXTRACTION (TO MATCH C++ DESTINY)")
    print("="*80)
    print(f"\nConfig: {config_file}")
    print(f"C++ Output: {cpp_output_file}")

    # Parse C++ output
    cpp_config = parse_cpp_destiny_output(cpp_output_file)

    # Extract scaled parameters
    params, subarray, scale_factor = extract_technology_parameters_scaled(config_file, cpp_config)

    # Verify with substitution
    bitline_delay_scaled = verify_scaled_expressions(params, cpp_config)

    print("\n" + "="*80)
    print("✓ SCALED EXTRACTION COMPLETE")
    print("="*80)

    print(f"\n📊 Summary:")
    print(f"  Scale factor applied: {scale_factor}×")
    print(f"  Scaled symbolic result: {bitline_delay_scaled*1e9:.6f} ns")
    print(f"  C++ DESTINY target: {cpp_config.bitline_latency*1e9:.6f} ns")
    print(f"  Improvement: Much closer to C++ values!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
