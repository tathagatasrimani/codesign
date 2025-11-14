#!/usr/bin/env python3
"""
Diagnose the 2× difference between Python and C++ DESTINY
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
import math


def main():
    config_file = "config/sample_SRAM_2layer.cfg"
    cpp_output_file = "../destiny_3d_cache-master/cpp_output_sram2layer.txt"

    print("="*80)
    print("DIAGNOSING 2× DIFFERENCE")
    print("="*80)

    # Parse C++ output
    cpp_config = parse_cpp_destiny_output(cpp_output_file)

    print("\n📊 C++ DESTINY Results:")
    print(f"  Configuration: {cpp_config.subarray_rows}×{cpp_config.subarray_cols}")
    print(f"  Bitline delay: {cpp_config.bitline_latency*1e9:.6f} ns")
    print(f"  Row decoder:   {cpp_config.row_decoder_latency*1e9:.6f} ns")

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

    # Create SubArray with SAME configuration as C++
    print(f"\n📐 Creating SubArray with C++ configuration:")
    print(f"  Rows: {cpp_config.subarray_rows}")
    print(f"  Cols: {cpp_config.subarray_cols}")
    print(f"  Senseamp mux: {cpp_config.senseamp_mux}")
    print(f"  Output mux L1: {cpp_config.output_mux_l1}")
    print(f"  Output mux L2: {cpp_config.output_mux_l2}")
    print(f"  Stacks: {cpp_config.num_stacks}")

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

    print("\n📊 Python DESTINY Results:")
    print(f"  Bitline delay: {subarray.bitlineDelay*1e9:.6f} ns")
    print(f"  Row decoder:   {subarray.rowDecoder.readLatency*1e9:.6f} ns")

    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    bitline_ratio = subarray.bitlineDelay / cpp_config.bitline_latency
    decoder_ratio = subarray.rowDecoder.readLatency / cpp_config.row_decoder_latency

    print(f"\n⚡ Bitline Delay:")
    print(f"  Python: {subarray.bitlineDelay*1e9:.6f} ns")
    print(f"  C++:    {cpp_config.bitline_latency*1e9:.6f} ns")
    print(f"  Ratio:  {bitline_ratio:.3f}×")

    print(f"\n🔢 Row Decoder Delay:")
    print(f"  Python: {subarray.rowDecoder.readLatency*1e9:.6f} ns")
    print(f"  C++:    {cpp_config.row_decoder_latency*1e9:.6f} ns")
    print(f"  Ratio:  {decoder_ratio:.3f}×")

    # Dig into bitline calculation details
    print("\n" + "="*80)
    print("BITLINE CALCULATION DETAILS")
    print("="*80)

    print(f"\n📏 Bitline R and C:")
    print(f"  R_bitline = {subarray.resBitline:.3f} Ω")
    print(f"  C_bitline = {subarray.capBitline*1e15:.3f} fF")
    print(f"  R_per_cell = {subarray.resBitline/subarray.numRow:.6f} Ω")
    print(f"  C_per_cell = {subarray.capBitline/subarray.numRow*1e15:.6f} fF")

    print(f"\n🔌 Cell Access:")
    print(f"  R_access = {subarray.resCellAccess:.3f} Ω")
    print(f"  C_access = {subarray.capCellAccess*1e15:.3f} fF")

    print(f"\n🔀 Mux:")
    print(f"  C_mux = {subarray.bitlineMux.capForPreviousDelayCalculation*1e15:.3f} fF")

    print(f"\n📡 Voltages:")
    print(f"  V_precharge = {subarray.voltagePrecharge:.3f} V")
    print(f"  V_sense = {subarray.senseVoltage:.3f} V")

    # Calculate tau manually
    from formula import calculate_on_resistance, calculate_transconductance
    NMOS = 0

    R_pulldown = calculate_on_resistance(
        g.cell.widthSRAMCellNMOS * g.tech.featureSize,
        NMOS,
        g.inputParameter.temperature,
        g.tech
    )

    print(f"\n⚙️ Calculated:")
    print(f"  R_pulldown = {R_pulldown:.3f} Ω")

    tau = ((subarray.resCellAccess + R_pulldown) *
           (subarray.capCellAccess + subarray.capBitline + subarray.bitlineMux.capForPreviousDelayCalculation) +
           subarray.resBitline * (subarray.bitlineMux.capForPreviousDelayCalculation + subarray.capBitline / 2))

    log_factor = math.log(subarray.voltagePrecharge / (subarray.voltagePrecharge - subarray.senseVoltage / 2))
    tau_with_log = tau * log_factor

    print(f"\n🧮 Step-by-step calculation:")
    print(f"  1. tau (RC) = {tau*1e9:.6f} ns")
    print(f"  2. log factor = {log_factor:.6f}")
    print(f"  3. tau × log = {tau_with_log*1e9:.6f} ns")

    gm = calculate_transconductance(g.cell.widthAccessCMOS * g.tech.featureSize, NMOS, g.tech)
    beta = 1 / (R_pulldown * gm)

    from formula import horowitz
    delay_horowitz, _ = horowitz(tau_with_log, beta, subarray.rowDecoder.rampOutput)

    print(f"  4. beta = {beta:.6f}")
    print(f"  5. ramp_input = {subarray.rowDecoder.rampOutput:.6e} s")
    print(f"  6. Horowitz delay = {delay_horowitz*1e9:.6f} ns")

    print(f"\n  Python bitlineDelay = {subarray.bitlineDelay*1e9:.6f} ns")
    print(f"  Match: {abs(delay_horowitz - subarray.bitlineDelay) < 1e-12}")

    # Now investigate WHY there's a 2× difference
    print("\n" + "="*80)
    print("INVESTIGATING POTENTIAL CAUSES")
    print("="*80)

    print("\n🔍 Possible reasons for 2× difference:")
    print("  1. Wire resistance/capacitance models")
    print("  2. Transistor sizing (widthAccessCMOS, widthSRAMCellNMOS)")
    print("  3. Technology parameters (vdd, vth, mobility)")
    print("  4. Horowitz model parameters (beta, ramp)")
    print("  5. Sense voltage or precharge voltage")

    print(f"\n🔬 Technology Parameters:")
    print(f"  V_dd = {g.tech.vdd:.3f} V")
    print(f"  V_th = {g.tech.vth:.3f} V")
    print(f"  Feature size = {g.tech.featureSize*1e9:.1f} nm")
    print(f"  Temperature = {g.inputParameter.temperature:.1f} K")

    print(f"\n📐 Cell Dimensions:")
    print(f"  widthAccessCMOS = {g.cell.widthAccessCMOS:.3f}")
    print(f"  widthSRAMCellNMOS = {g.cell.widthSRAMCellNMOS:.3f}")
    print(f"  widthSRAMCellPMOS = {g.cell.widthSRAMCellPMOS:.3f}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    print(f"\n🎯 The 2× difference is systematic across:")
    print(f"  • Bitline delay: {bitline_ratio:.3f}×")
    print(f"  • Decoder delay: {decoder_ratio:.3f}×")

    print(f"\n💡 This suggests:")
    print(f"  • Different wire models in Python vs C++")
    print(f"  • Different resistance/capacitance calculations")
    print(f"  • Different technology parameter scaling")

    print(f"\n🔧 To fix this, we would need to:")
    print(f"  1. Compare C++ and Python wire model implementations")
    print(f"  2. Check if Python is using correct wire type (local vs global)")
    print(f"  3. Verify transistor on-resistance calculations match C++")
    print(f"  4. Ensure horowitz model parameters match C++")

    return 0


if __name__ == "__main__":
    sys.exit(main())
