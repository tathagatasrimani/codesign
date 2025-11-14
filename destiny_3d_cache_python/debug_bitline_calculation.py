#!/usr/bin/env python3
"""
Debug bitline delay calculation to compare with C++
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
    config_file = "config/sample_SRAM_2layer.cfg"

    print("="*80)
    print("PYTHON BITLINE DELAY CALCULATION DEBUG")
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

    # Create 1024×2048 subarray to match C++
    subarray = SubArray()
    subarray.Initialize(
        1024,  # numRow
        2048,  # numColumn
        1,     # numBankX
        1,     # numBankY
        1,     # muxSenseAmp
        True,  # internalSenseAmp
        1,     # muxOutputLev1
        8,     # muxOutputLev2
        BufferDesignTarget.latency_first,
        2      # num3DLevels
    )

    subarray.CalculateArea()
    subarray.CalculateLatency(1e20)

    print(f"\n[DEBUG] Python Bitline Delay Calculation:")
    print(f"  resCellAccess = {subarray.resCellAccess:.3e} Ω")
    print(f"  capCellAccess = {subarray.capCellAccess:.3e} F")
    print(f"  capBitline = {subarray.capBitline:.3e} F")
    print(f"  resBitline = {subarray.resBitline:.3e} Ω")
    print(f"  bitlineDelay = {subarray.bitlineDelay:.3e} s")

    print(f"\n" + "="*80)
    print(f"COMPARISON WITH C++")
    print(f"="*80)

    cpp_capBitline = 3.351e-13
    cpp_resBitline = 1.466e+03
    cpp_bitlineDelay = 1.990e-09

    print(f"\nCapacitance:")
    print(f"  Python: {subarray.capBitline:.3e} F")
    print(f"  C++:    {cpp_capBitline:.3e} F")
    print(f"  Ratio:  {subarray.capBitline / cpp_capBitline:.3f}×")

    print(f"\nResistance:")
    print(f"  Python: {subarray.resBitline:.3e} Ω")
    print(f"  C++:    {cpp_resBitline:.3e} Ω")
    print(f"  Ratio:  {subarray.resBitline / cpp_resBitline:.3f}×")

    print(f"\nDelay:")
    print(f"  Python: {subarray.bitlineDelay*1e9:.6f} ns")
    print(f"  C++:    {cpp_bitlineDelay*1e9:.6f} ns")
    print(f"  Ratio:  {subarray.bitlineDelay / cpp_bitlineDelay:.3f}×")

    return 0


if __name__ == "__main__":
    sys.exit(main())
