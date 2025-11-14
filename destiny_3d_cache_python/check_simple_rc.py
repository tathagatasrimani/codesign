#!/usr/bin/env python3
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

# Test with multiple row counts
config_file = 'config/sample_SRAM_2layer.cfg'

for num_rows in [256, 512, 1024]:
    print(f"\n{'='*60}")
    print(f"Testing {num_rows} rows")
    print('='*60)

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

    # Create SubArray
    subarray = SubArray()
    subarray.Initialize(num_rows, 2048, 1, 1, 1, True, 1, 8, BufferDesignTarget.latency_first, 2)
    subarray.CalculateArea()

    # BEFORE CalculateLatency
    simple_rc_delay = 0.5 * subarray.resBitline * subarray.capBitline * 1e9

    print(f'\nPhysical parameters:')
    print(f'  resBitline = {subarray.resBitline:.3e} Ω')
    print(f'  capBitline = {subarray.capBitline:.3e} F')
    print(f'  R × C = {subarray.resBitline * subarray.capBitline:.3e} s')

    print(f'\nSimple RC/2 prediction:')
    print(f'  t = 0.5 × R × C = {simple_rc_delay:.6f} ns')

    # Now calculate actual
    subarray.CalculateLatency(1e20)

    print(f'\nActual DESTINY result:')
    print(f'  bitlineDelay = {subarray.bitlineDelay * 1e9:.6f} ns')

    ratio = subarray.bitlineDelay / (0.5 * subarray.resBitline * subarray.capBitline)
    print(f'\nRatio (actual / simple RC/2):')
    print(f'  {ratio:.3f}× different')

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)
print("\nIf the symbolic formula t ∝ R×C/2 were correct,")
print("then (actual / RC/2) ratio should be CONSTANT across all sizes.")
print("\nIf the ratio changes significantly, then the symbolic")
print("formula is INCOMPLETE or INCORRECT.")
