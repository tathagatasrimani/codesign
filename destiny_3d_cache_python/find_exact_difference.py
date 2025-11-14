#!/usr/bin/env python3
"""
Find the EXACT source of the 2× difference
Compare every single intermediate value between Python and C++
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


def main():
    config_file = "config/sample_SRAM_2layer.cfg"
    cpp_output_file = "../destiny_3d_cache-master/cpp_output_sram2layer.txt"

    print("="*80)
    print("FINDING EXACT SOURCE OF 2× DIFFERENCE")
    print("="*80)

    # Parse C++ results
    cpp_config = parse_cpp_destiny_output(cpp_output_file)

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

    print("\n" + "="*80)
    print("HYPOTHESIS: Check Wire Resistance/Capacitance Per Unit")
    print("="*80)

    print(f"\nPython Wire (localWire):")
    print(f"  resWirePerUnit = {g.localWire.resWirePerUnit:.6e} Ω/m")
    print(f"  capWirePerUnit = {g.localWire.capWirePerUnit:.6e} F/m")

    print(f"\n🔍 If C++ uses HALF these values, that would explain 2× difference!")

    print(f"\nPython Bitline R/C:")
    print(f"  resBitline = {subarray.resBitline:.6f} Ω")
    print(f"  capBitline = {subarray.capBitline*1e15:.6f} fF")
    print(f"  Bitline delay = {subarray.bitlineDelay*1e9:.6f} ns")

    print(f"\nC++ Bitline delay:")
    print(f"  Bitline delay = {cpp_config.bitline_latency*1e9:.6f} ns")

    print(f"\nRatio:")
    print(f"  Python / C++ = {subarray.bitlineDelay / cpp_config.bitline_latency:.3f}×")

    # Check R×C
    RC_python = subarray.resBitline * subarray.capBitline
    print(f"\n🔍 R×C Analysis:")
    print(f"  Python R×C = {RC_python*1e9:.6f} ns")

    # If we scale by 0.5²=0.25, we'd get closer
    RC_scaled = RC_python * 0.25
    print(f"  If R and C both scaled by 0.5:")
    print(f"    R×C scaled = {RC_scaled*1e9:.6f} ns")

    print("\n" + "="*80)
    print("CHECKING WIRE PARAMETERS")
    print("="*80)

    print(f"\nWire Type: {g.localWire.wireType}")
    print(f"Wire Width: {g.localWire.wireWidth*1e9:.3f} nm")
    print(f"Wire Thickness: {g.localWire.wireThickness*1e9:.3f} nm")
    print(f"Wire Spacing: {g.localWire.wireSpacing*1e9:.3f} nm")
    print(f"Wire Pitch: {g.localWire.wirePitch*1e9:.3f} nm")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    print("\n💡 To find exact issue, need to:")
    print("  1. Print C++ wire resWirePerUnit and capWirePerUnit")
    print("  2. Compare with Python values")
    print("  3. Check if C++ uses different wire model")
    print("  4. Check SubArray R/C calculation formulas")

    print(f"\n📊 Current Status:")
    print(f"  Python bitline: {subarray.bitlineDelay*1e9:.6f} ns")
    print(f"  C++ bitline:    {cpp_config.bitline_latency*1e9:.6f} ns")
    print(f"  Need to reduce Python by: {subarray.bitlineDelay / cpp_config.bitline_latency:.3f}×")

    return 0


if __name__ == "__main__":
    sys.exit(main())
