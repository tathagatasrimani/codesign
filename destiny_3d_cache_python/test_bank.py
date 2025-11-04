#!/usr/bin/env python
"""Test Bank initialization and calculation"""

import sys
import os
import globals as g
from Technology import Technology
from InputParameter import InputParameter
from MemCell import MemCell
from Wire import Wire
from typedef import WireType, WireRepeaterType, BufferDesignTarget, MemoryType
from BankWithHtree import BankWithHtree

# Initialize globals
g.inputParameter = InputParameter()
g.tech = Technology()
g.devtech = Technology()  # Device technology - MISSING!
g.gtech = Technology()
g.cell = MemCell()
g.localWire = Wire()
g.globalWire = Wire()

# Read config
g.inputParameter.ReadInputParameterFromFile("config/sample_SRAM_2layer.cfg")

# Initialize tech
g.tech.Initialize(g.inputParameter.processNode, g.inputParameter.deviceRoadmap, g.inputParameter)
# Initialize devtech (device technology) - same as tech
g.devtech.Initialize(g.inputParameter.processNode, g.inputParameter.deviceRoadmap, g.inputParameter)
# Initialize gtech (global technology) - same as tech
g.gtech.Initialize(g.inputParameter.processNode, g.inputParameter.deviceRoadmap, g.inputParameter)

# Initialize cell
cellFile = g.inputParameter.fileMemCell[0]
if '/' not in cellFile:
    cellFile = os.path.join(os.path.dirname("config/sample_SRAM_2layer.cfg"), cellFile)
g.cell.ReadCellFromFile(cellFile)

# Initialize wires
g.localWire.Initialize(g.inputParameter.processNode, WireType.local_aggressive,
                      WireRepeaterType.repeated_none, g.inputParameter.temperature, False)
g.globalWire.Initialize(g.inputParameter.processNode, WireType.global_aggressive,
                        WireRepeaterType.repeated_none, g.inputParameter.temperature, False)

print(f"Globals initialized:")
print(f"  g.cell.memCellType = {g.cell.memCellType}")
print(f"  g.tech.initialized = {g.tech.initialized}")

# Create a simple bank configuration
print("\nCreating bank...")
bank = BankWithHtree()

# Use simple parameters
numRowMat = 2
numColumnMat = 2
capacity = 2 * 1024 * 1024 * 8  # 2MB in bits
blockSize = 256  # bits
associativity = 1
numRowPerSet = 1
numActiveMatPerRow = 1
numActiveMatPerColumn = 1
muxSenseAmp = 1
muxOutputLev1 = 1
muxOutputLev2 = 1
numRowSubarray = 1
numColumnSubarray = 1
numActiveSubarrayPerRow = 1
numActiveSubarrayPerColumn = 1
areaOptimizationLevel = BufferDesignTarget.latency_first
memoryType = MemoryType.data
stackedDieCount = 2
partitionGranularity = 0
monolithicStackCount = 1

print(f"Initializing bank with:")
print(f"  numRowMat={numRowMat}, numColumnMat={numColumnMat}")
print(f"  capacity={capacity} bits, blockSize={blockSize} bits")
print(f"  stackedDieCount={stackedDieCount}")

try:
    bank.Initialize(
        numRowMat, numColumnMat, capacity, blockSize, associativity,
        numRowPerSet, numActiveMatPerRow, numActiveMatPerColumn, muxSenseAmp,
        True,  # internalSensing
        muxOutputLev1, muxOutputLev2,
        numRowSubarray, numColumnSubarray,
        numActiveSubarrayPerRow, numActiveSubarrayPerColumn,
        areaOptimizationLevel, memoryType, stackedDieCount,
        partitionGranularity, monolithicStackCount
    )
    print(f"Bank initialized: {bank.initialized}, invalid: {bank.invalid}")
except Exception as e:
    print(f"ERROR during Initialize: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if bank.invalid:
    print("Bank is invalid, skipping calculations")
    sys.exit(0)

print("\nCalculating area...")
try:
    bank.CalculateArea()
    print(f"Area calculated: {bank.area}")
except Exception as e:
    print(f"ERROR during CalculateArea: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nCalculating RC...")
try:
    bank.CalculateRC()
    print("RC calculated")
except Exception as e:
    print(f"ERROR during CalculateRC: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nCalculating latency and power...")
print(f"DEBUG: Before CalculateLatencyAndPower, g.cell = {g.cell}")
try:
    bank.CalculateLatencyAndPower()
    print(f"Latency and power calculated!")
    print(f"  Read Latency: {bank.readLatency}")
    print(f"  Write Latency: {bank.writeLatency}")
except Exception as e:
    print(f"ERROR during CalculateLatencyAndPower: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll tests passed!")
