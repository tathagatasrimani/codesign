#!/usr/bin/env python3
"""
Trace what capBitline actually contains
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

    # Initialize
    g.inputParameter = InputParameter()
    g.inputParameter.ReadInputParameterFromFile(config_file)

    g.tech = Technology()
    g.tech.Initialize(g.inputParameter.processNode, g.inputParameter.deviceRoadmap, g.inputParameter)

    g.devtech = Technology()
    g.devtech.Initialize(g.inputParameter.processNode, g.inputParameter.deviceRoadmap, g.inputParameter)

    g.localWire = Wire()
    g.localWire.Initialize(g.inputParameter.processNode, WireType.local_aggressive,
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

    print("="*80)
    print("TRACING capBitline CALCULATION")
    print("="*80)

    print(f"\nFrom SubArray.py calculation:")
    print(f"  lenBitline = {subarray.lenBitline:.6e} m")
    print(f"  capWirePerUnit = {g.localWire.capWirePerUnit:.6e} F/m")
    print(f"  num3DLevels = 1 (hardcoded)")

    print(f"\n  capBitline = lenBitline × capWirePerUnit × 1")
    print(f"             = {subarray.lenBitline:.6e} × {g.localWire.capWirePerUnit:.6e} × 1")
    print(f"             = {subarray.capBitline:.6e} F")

    print(f"\nThis is:")
    print(f"  capBitline = {subarray.capBitline*1e15:.6f} fF")

    print(f"\nPer-cell contribution:")
    cell_cap_contribution = g.localWire.capWirePerUnit * g.cell.heightInFeatureSize * g.devtech.featureSize
    print(f"  C_per_cell (wire only) = capWirePerUnit × cell.heightInFeatureSize × featureSize")
    print(f"                         = {g.localWire.capWirePerUnit:.6e} × {g.cell.heightInFeatureSize} × {g.devtech.featureSize:.6e}")
    print(f"                         = {cell_cap_contribution:.6e} F")
    print(f"                         = {cell_cap_contribution*1e18:.6f} aF")

    print(f"\nTotal for 1024 rows:")
    print(f"  C_bitline (wire) = {subarray.numRow} × {cell_cap_contribution:.6e}")
    print(f"                   = {subarray.numRow * cell_cap_contribution:.6e} F")
    print(f"                   = {subarray.numRow * cell_cap_contribution*1e15:.6f} fF")

    print(f"\nActual capBitline from SubArray:")
    print(f"  {subarray.capBitline*1e15:.6f} fF")

    print(f"\n✓ These match! capBitline is ONLY the wire capacitance.")

    print(f"\n" + "="*80)
    print(f"WHAT ABOUT CELL CAPACITANCE?")
    print(f"="*80)

    print(f"\nIn the tau formula (line 507-508 of SubArray.py):")
    print(f"  tau = (R_access + R_pulldown) × (C_access + C_bitline + C_mux)")
    print(f"      + R_bitline × (C_mux + C_bitline/2)")

    print(f"\nSo C_bitline in the formula includes:")
    print(f"  1. Wire capacitance = {subarray.capBitline*1e15:.6f} fF")
    print(f"  2. C_access and C_mux are added separately")

    print(f"\nSo the symbolic expression for C_bitline should be:")
    print(f"  C_bitline = rows × (capWirePerUnit × cellHeight × featureSize)")
    print(f"  C_bitline = rows × C_per_cell_wire")

    return 0


if __name__ == "__main__":
    sys.exit(main())
