#!/usr/bin/env python3
"""
Debug wire resistance and capacitance calculation
Print all intermediate values to find where Python differs from C++
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import globals as g
from InputParameter import InputParameter
from Technology import Technology
from Wire import Wire
from typedef import WireType, WireRepeaterType
from constant import COPPER_RESISTIVITY, COPPER_RESISTIVITY_TEMPERATURE_COEFFICIENT, PERMITTIVITY
from formula import calculate_wire_resistance, calculate_wire_capacitance


def main():
    config_file = "config/sample_SRAM_2layer.cfg"

    print("="*80)
    print("DEBUGGING WIRE CALCULATION")
    print("="*80)

    # Initialize Python DESTINY
    g.inputParameter = InputParameter()
    g.inputParameter.ReadInputParameterFromFile(config_file)

    g.tech = Technology()
    g.tech.Initialize(g.inputParameter.processNode, g.inputParameter.deviceRoadmap, g.inputParameter)

    # Create wire with detailed printing
    print("\n📍 Creating localWire...")
    g.localWire = Wire()

    # Manually initialize to track values
    featureSizeInNano = g.inputParameter.processNode
    wireType = WireType.local_aggressive
    temperature = g.inputParameter.temperature

    print(f"\nInput Parameters:")
    print(f"  featureSizeInNano = {featureSizeInNano} nm")
    print(f"  wireType = {wireType}")
    print(f"  temperature = {temperature} K")

    # Initialize the wire
    g.localWire.Initialize(featureSizeInNano, wireType,
                           WireRepeaterType.repeated_none, temperature, False)

    print(f"\n65nm local_aggressive Wire Parameters:")
    print(f"  barrierThickness = {g.localWire.barrierThickness*1e9:.6f} nm")
    print(f"  horizontalDielectric = {g.localWire.horizontalDielectric:.6f}")
    print(f"  wirePitch = {g.localWire.wirePitch*1e9:.6f} nm")
    print(f"  aspectRatio = {g.localWire.aspectRatio:.6f}")
    print(f"  ildThickness = {g.localWire.ildThickness*1e9:.6f} nm")

    print(f"\nCalculated Wire Dimensions:")
    print(f"  wireWidth = wirePitch / 2 = {g.localWire.wireWidth*1e9:.6f} nm")
    print(f"  wireThickness = aspectRatio × wireWidth = {g.localWire.wireThickness*1e9:.6f} nm")
    print(f"  wireSpacing = wirePitch - wireWidth = {g.localWire.wireSpacing*1e9:.6f} nm")

    # Calculate resistivity with temperature adjustment
    copper_resistivity_base = COPPER_RESISTIVITY
    print(f"\nCopper Resistivity:")
    print(f"  Base (293K) = {copper_resistivity_base*1e8:.6f} × 10⁻⁸ Ω·m")

    copper_resistivity = copper_resistivity_base * (
        1 + COPPER_RESISTIVITY_TEMPERATURE_COEFFICIENT * (temperature - 293))
    print(f"  Adjusted ({temperature}K) = {copper_resistivity*1e8:.6f} × 10⁻⁸ Ω·m")
    print(f"  Temperature coefficient = {COPPER_RESISTIVITY_TEMPERATURE_COEFFICIENT}")

    # Calculate wire resistance
    print(f"\nWire Resistance Calculation:")
    print(f"  Formula: R = alpha_scatter × resistivity / (thickness - barrier - dishing) / (width - 2×barrier)")
    print(f"  alpha_scatter = 1")
    print(f"  dishing_thickness = 0")

    wire_res = calculate_wire_resistance(
        copper_resistivity,
        g.localWire.wireWidth,
        g.localWire.wireThickness,
        g.localWire.barrierThickness,
        0,  # dishing thickness
        1   # alpha scatter
    )

    print(f"\n  Numerator: {copper_resistivity*1e8:.6f} × 10⁻⁸")
    print(f"  Denominator (thickness): ({g.localWire.wireThickness*1e9:.6f} - {g.localWire.barrierThickness*1e9:.6f} - 0) nm = {(g.localWire.wireThickness - g.localWire.barrierThickness)*1e9:.6f} nm")
    print(f"  Denominator (width): ({g.localWire.wireWidth*1e9:.6f} - 2×{g.localWire.barrierThickness*1e9:.6f}) nm = {(g.localWire.wireWidth - 2*g.localWire.barrierThickness)*1e9:.6f} nm")
    print(f"\n  resWirePerUnit = {wire_res:.6e} Ω/m")
    print(f"  Python (with calibration) = {g.localWire.resWirePerUnit:.6e} Ω/m")

    # Calculate wire capacitance
    print(f"\nWire Capacitance Calculation:")
    print(f"  Formula: C = vertical_cap + sidewall_cap + fringe_cap")
    print(f"  Permittivity = {PERMITTIVITY*1e12:.6f} × 10⁻¹² F/m")
    print(f"  miller_value = 1.5")
    print(f"  vertical_dielectric = 3.9")
    print(f"  fringe_cap = 1.15e-10 F/m")

    vertical_cap = 2 * PERMITTIVITY * 3.9 * g.localWire.wireWidth / g.localWire.ildThickness
    sidewall_cap = 2 * PERMITTIVITY * 1.5 * g.localWire.horizontalDielectric * g.localWire.wireThickness / g.localWire.wireSpacing
    fringe_cap = 1.15e-10

    print(f"\n  Vertical cap = 2 × permittivity × 3.9 × wireWidth / ildThickness")
    print(f"              = 2 × {PERMITTIVITY*1e12:.6f}e-12 × 3.9 × {g.localWire.wireWidth*1e9:.6f}e-9 / {g.localWire.ildThickness*1e9:.6f}e-9")
    print(f"              = {vertical_cap*1e12:.6f} × 10⁻¹² F/m")

    print(f"\n  Sidewall cap = 2 × permittivity × 1.5 × horizontalDielectric × wireThickness / wireSpacing")
    print(f"               = 2 × {PERMITTIVITY*1e12:.6f}e-12 × 1.5 × {g.localWire.horizontalDielectric:.6f} × {g.localWire.wireThickness*1e9:.6f}e-9 / {g.localWire.wireSpacing*1e9:.6f}e-9")
    print(f"               = {sidewall_cap*1e12:.6f} × 10⁻¹² F/m")

    print(f"\n  Fringe cap = {fringe_cap*1e12:.6f} × 10⁻¹² F/m")

    wire_cap = vertical_cap + sidewall_cap + fringe_cap

    print(f"\n  Total capWirePerUnit = {wire_cap:.6e} F/m")
    print(f"  Python (with calibration) = {g.localWire.capWirePerUnit:.6e} F/m")

    print(f"\n" + "="*80)
    print(f"SUMMARY")
    print(f"="*80)
    print(f"\nPython BEFORE calibration:")
    print(f"  resWirePerUnit = {wire_res:.6e} Ω/m")
    print(f"  capWirePerUnit = {wire_cap:.6e} F/m")

    print(f"\nPython AFTER 0.5× calibration:")
    print(f"  resWirePerUnit = {g.localWire.resWirePerUnit:.6e} Ω/m")
    print(f"  capWirePerUnit = {g.localWire.capWirePerUnit:.6e} F/m")

    print(f"\n🔍 These values should match C++ DESTINY exactly!")
    print(f"   Need to compare with C++ output to find discrepancy")

    return 0


if __name__ == "__main__":
    sys.exit(main())
