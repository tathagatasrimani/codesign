#!/usr/bin/env python3
"""
Final comparison: C++ vs Python DESTINY
Simple verification that they match
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


def main():
    print("="*80)
    print("FINAL VERIFICATION: C++ DESTINY vs Python DESTINY")
    print("="*80)

    # Initialize Python DESTINY
    g.inputParameter = InputParameter()
    g.inputParameter.ReadInputParameterFromFile("config/sample_SRAM_2layer.cfg")

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

    # Create 1024×2048 subarray
    subarray = SubArray()
    subarray.Initialize(1024, 2048, 1, 1, 1, True, 1, 8,
                       BufferDesignTarget.latency_first, 2)

    subarray.CalculateArea()
    subarray.CalculateLatency(1e20)

    # Results
    print(f"\nConfiguration:")
    print(f"  Process Node: 65nm LOP")
    print(f"  Subarray: 1024 rows × 2048 columns")
    print(f"  Memory Type: SRAM")

    print(f"\n{'Component':<30} {'Python DESTINY':<20} {'C++ DESTINY':<20} {'Match'}")
    print("-" * 90)

    # C++ values (from actual run)
    cpp_res = 1.466e+03
    cpp_cap = 3.351e-13
    cpp_delay = 1.990e-09

    # Python values
    python_res = subarray.resBitline
    python_cap = subarray.capBitline
    python_delay = subarray.bitlineDelay

    res_match = abs(python_res - cpp_res) / cpp_res < 0.01
    cap_match = abs(python_cap - cpp_cap) / cpp_cap < 0.01
    delay_match = abs(python_delay - cpp_delay) / cpp_delay < 0.01

    print(f"{'Bitline Resistance':<30} {python_res:>18.3f} Ω   {cpp_res:>18.3f} Ω   {'✅' if res_match else '❌'}")
    print(f"{'Bitline Capacitance':<30} {python_cap*1e15:>18.3f} fF  {cpp_cap*1e15:>18.3f} fF  {'✅' if cap_match else '❌'}")
    print(f"{'Bitline Delay':<30} {python_delay*1e9:>18.6f} ns  {cpp_delay*1e9:>18.6f} ns  {'✅' if delay_match else '❌'}")

    print(f"\n{'Errors:':<30}")
    print(f"  Resistance error: {abs(python_res - cpp_res)/cpp_res*100:>6.3f}%")
    print(f"  Capacitance error: {abs(python_cap - cpp_cap)/cpp_cap*100:>6.3f}%")
    print(f"  Delay error: {abs(python_delay - cpp_delay)/cpp_delay*100:>6.3f}%")

    print("\n" + "="*80)
    if res_match and cap_match and delay_match:
        print("✅ VERIFICATION PASSED!")
        print("Python DESTINY matches C++ DESTINY within 1% error")
        return 0
    else:
        print("❌ VERIFICATION FAILED!")
        print("Python DESTINY does not match C++ DESTINY")
        return 1


if __name__ == "__main__":
    sys.exit(main())
