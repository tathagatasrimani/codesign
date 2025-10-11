#!/usr/bin/env python
"""Test script to verify Technology initialization"""

import sys
import globals as g
from Technology import Technology
from InputParameter import InputParameter
from typedef import DeviceRoadmap

# Initialize globals
g.inputParameter = InputParameter()
g.tech = Technology()

# Read config to get parameters
g.inputParameter.ReadInputParameterFromFile("config/sample_SRAM_2layer.cfg")

print(f"Process Node: {g.inputParameter.processNode}")
print(f"Device Roadmap: {g.inputParameter.deviceRoadmap}")
print(f"Temperature: {g.inputParameter.temperature}")

# Initialize technology
g.tech.Initialize(
    g.inputParameter.processNode,
    g.inputParameter.deviceRoadmap,
    g.inputParameter
)

# Check if current arrays are populated
temp_idx = g.inputParameter.temperature - 300
print(f"\nTemperature index: {temp_idx}")
print(f"currentOnNmos[{temp_idx}] = {g.tech.currentOnNmos[temp_idx]}")
print(f"currentOnPmos[{temp_idx}] = {g.tech.currentOnPmos[temp_idx]}")
print(f"currentOffNmos[{temp_idx}] = {g.tech.currentOffNmos[temp_idx]}")
print(f"currentOffPmos[{temp_idx}] = {g.tech.currentOffPmos[temp_idx]}")

# Test the specific line that's failing
from formula import calculate_gate_cap
testWidth = 1e-6
try:
    cap = calculate_gate_cap(testWidth, g.tech)
    print(f"\ncalculate_gate_cap({testWidth}, tech) = {cap}")
    print("SUCCESS: calculate_gate_cap works!")
except Exception as e:
    print(f"\nERROR in calculate_gate_cap: {e}")
    sys.exit(1)

# Test the division that's failing
try:
    maxBitlineCurrent = 10e-6
    minWidth = maxBitlineCurrent / g.tech.currentOnNmos[temp_idx]
    print(f"\nTest division: {maxBitlineCurrent} / {g.tech.currentOnNmos[temp_idx]} = {minWidth}")
    print("SUCCESS: Division works!")
except ZeroDivisionError as e:
    print(f"\nERROR: Division by zero! currentOnNmos[{temp_idx}] is zero!")
    print(f"Full currentOnNmos array (first 11 values): {g.tech.currentOnNmos[0:101:10]}")
    sys.exit(1)

print("\nAll tests passed!")
