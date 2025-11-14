#!/usr/bin/env python3
"""
Debug lenBitline calculation
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

    print("="*80)
    print("DEBUGGING lenBitline CALCULATION")
    print("="*80)

    print(f"\nInput:")
    print(f"  numRow = {subarray.numRow}")
    print(f"  cell.heightInFeatureSize = {g.cell.heightInFeatureSize}")
    print(f"  devtech.featureSize = {g.devtech.featureSize*1e9:.3f} nm")
    print(f"  num3DLevels = {subarray.num3DLevels}")

    print(f"\nCalculation:")
    print(f"  lenBitline = numRow × cell.heightInFeatureSize × devtech.featureSize")
    print(f"             = {subarray.numRow} × {g.cell.heightInFeatureSize} × {g.devtech.featureSize*1e9:.3f}e-9")
    print(f"             = {subarray.lenBitline:.6e} m")

    print(f"\nWire per unit:")
    print(f"  resWirePerUnit = {g.localWire.resWirePerUnit:.6e} Ω/m")
    print(f"  capWirePerUnit = {g.localWire.capWirePerUnit:.6e} F/m")

    print(f"\nBitline R/C:")
    print(f"  resBitline = lenBitline × resWirePerUnit × num3DLevels")
    print(f"             = {subarray.lenBitline:.6e} × {g.localWire.resWirePerUnit:.6e} × {subarray.num3DLevels}")
    print(f"             = {subarray.resBitline:.6e} Ω")

    print(f"\n  capBitline = lenBitline × capWirePerUnit × num3DLevels")
    print(f"             = {subarray.lenBitline:.6e} × {g.localWire.capWirePerUnit:.6e} × {subarray.num3DLevels}")
    print(f"             = {subarray.capBitline:.6e} F")

    print(f"\n" + "="*80)
    print(f"C++ VALUES (from debug output)")
    print(f"="*80)
    cpp_resBitline = 1.466e+03
    cpp_capBitline = 3.351e-13

    print(f"\n  resBitline = {cpp_resBitline:.6e} Ω")
    print(f"  capBitline = {cpp_capBitline:.6e} F")

    print(f"\nRatios:")
    print(f"  Python / C++ resistance = {subarray.resBitline / cpp_resBitline:.3f}×")
    print(f"  Python / C++ capacitance = {subarray.capBitline / cpp_capBitline:.3f}×")

    # Check if num3DLevels is the issue
    print(f"\n" + "="*80)
    print(f"HYPOTHESIS: num3DLevels might be different")
    print(f"="*80)

    print(f"\nIf C++ uses num3DLevels = 1 instead of 2:")
    res_with_1_level = subarray.lenBitline * g.localWire.resWirePerUnit * 1
    cap_with_1_level = subarray.lenBitline * g.localWire.capWirePerUnit * 1

    print(f"  resBitline = {res_with_1_level:.6e} Ω (ratio: {res_with_1_level / cpp_resBitline:.3f}×)")
    print(f"  capBitline = {cap_with_1_level:.6e} F (ratio: {cap_with_1_level / cpp_capBitline:.3f}×)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
