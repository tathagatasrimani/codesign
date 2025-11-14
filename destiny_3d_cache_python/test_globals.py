#!/usr/bin/env python
"""Test script to verify globals work correctly"""

import sys
import globals as g
from Technology import Technology
from InputParameter import InputParameter
from MemCell import MemCell

print("Step 1: Check initial globals state")
print(f"g.cell = {g.cell}")
print(f"g.tech = {g.tech}")
print(f"g.inputParameter = {g.inputParameter}")

print("\nStep 2: Initialize globals")
g.inputParameter = InputParameter()
g.tech = Technology()
g.cell = MemCell()

print(f"g.cell = {g.cell}")
print(f"g.tech = {g.tech}")
print(f"g.inputParameter = {g.inputParameter}")

print("\nStep 3: Read config")
g.inputParameter.ReadInputParameterFromFile("config/sample_SRAM_2layer.cfg")
print(f"Process Node: {g.inputParameter.processNode}")

print("\nStep 4: Initialize tech")
g.tech.Initialize(g.inputParameter.processNode, g.inputParameter.deviceRoadmap, g.inputParameter)
print(f"g.tech initialized: {g.tech.initialized}")

print("\nStep 5: Initialize cell")
import os
cellFile = g.inputParameter.fileMemCell[0]
if '/' not in cellFile:
    cellFile = os.path.join(os.path.dirname("config/sample_SRAM_2layer.cfg"), cellFile)
g.cell.ReadCellFromFile(cellFile)
print(f"g.cell memCellType: {g.cell.memCellType}")

print("\nStep 6: Import BankWithHtree and check if g.cell is visible")
from BankWithHtree import BankWithHtree
bank = BankWithHtree()
print(f"Created bank: {bank}")

print("\nStep 7: Check g.cell from within a function")
import globals as g2
print(f"g.cell from second import: {g2.cell}")
print(f"g.cell.memCellType: {g2.cell.memCellType if g2.cell else 'None'}")

print("\nStep 8: Try to access g.cell.memCellType like BankWithHtree does")
try:
    from typedef import MemCellType
    if g.cell.memCellType == MemCellType.eDRAM:
        print("Cell is eDRAM")
    else:
        print(f"Cell is {g.cell.memCellType}")
except AttributeError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print("\nAll tests passed!")
