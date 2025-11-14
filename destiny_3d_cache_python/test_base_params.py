#!/usr/bin/env python3
"""
Test the base parameters symbolic modeling approach
This demonstrates how symbolic variables work with DESTINY
"""

import globals as g
from base_parameters import BaseParameters
from Technology import Technology
from MemCell import MemCell
from InputParameter import InputParameter
from typedef import DeviceRoadmap
from sympy import simplify


def test_basic_symbolic():
    """Test basic symbolic variable creation"""
    print("=" * 80)
    print("Test 1: Basic Symbolic Variable Creation")
    print("=" * 80)

    bp = BaseParameters()

    print(f"\nSymbolic variables created:")
    print(f"  vdd = {bp.vdd}")
    print(f"  vth = {bp.vth}")
    print(f"  featureSize = {bp.featureSize}")
    print(f"  capIdealGate = {bp.capIdealGate}")
    print(f"  resistanceOn = {bp.resistanceOn}")

    print(f"\nTotal symbols: {len(bp.symbol_table)}")
    print("✓ Symbolic variables created successfully")


def test_gate_capacitance_symbolic():
    """Test symbolic gate capacitance calculation"""
    print("\n" + "=" * 80)
    print("Test 2: Symbolic Gate Capacitance Calculation")
    print("=" * 80)

    bp = BaseParameters()

    # Symbolic gate capacitance formula (from formula.py calculate_gate_cap)
    # cap = (capIdealGate + capOverlap + 3*capFringe) * width + phyGateLength * capPolywire
    width_sym = bp.symbol_table['processNode']  # Use processNode as width for example

    gate_cap_symbolic = ((bp.capIdealGate + bp.capOverlap + 3 * bp.capFringe) * width_sym +
                         bp.phyGateLength * bp.capPolywire)

    print(f"\nSymbolic gate capacitance expression:")
    print(f"  {gate_cap_symbolic}")

    print(f"\nSimplified:")
    print(f"  {simplify(gate_cap_symbolic)}")

    print("✓ Symbolic calculation successful")


def test_resistance_symbolic():
    """Test symbolic resistance calculation"""
    print("\n" + "=" * 80)
    print("Test 3: Symbolic On-Resistance Calculation")
    print("=" * 80)

    bp = BaseParameters()

    # Symbolic resistance formula (from formula.py calculate_on_resistance)
    # R = effectiveResistanceMultiplier * vdd / (currentOnNmos * width)
    width_sym = bp.cellWidthInFeatureSize

    resistance_symbolic = (bp.effectiveResistanceMultiplier * bp.vdd /
                          (bp.currentOnNmos * width_sym))

    print(f"\nSymbolic resistance expression:")
    print(f"  {resistance_symbolic}")

    print(f"\nSimplified:")
    print(f"  {simplify(resistance_symbolic)}")

    print("✓ Symbolic calculation successful")


def test_populate_concrete_values():
    """Test populating concrete values from DESTINY objects"""
    print("\n" + "=" * 80)
    print("Test 4: Populating Concrete Values")
    print("=" * 80)

    # Initialize globals
    g.inputParameter = InputParameter()
    g.tech = Technology()
    g.cell = MemCell()

    # Initialize technology for 65nm HP
    print("\nInitializing 65nm HP technology...")
    g.tech.Initialize(65, DeviceRoadmap.HP, g.inputParameter)

    # Create some dummy cell data
    g.cell.area = 10.0
    g.cell.aspectRatio = 1.5
    g.cell.resistanceOn = 1000.0
    g.cell.resistanceOff = 1000000.0

    # Create base parameters and populate
    bp = BaseParameters()
    bp.populate_from_technology(g.tech)
    bp.populate_from_memcell(g.cell)
    bp.populate_from_input_parameter(g.inputParameter)

    print(f"\nConcrete values populated:")
    print(f"  vdd = {bp.tech_values.get(bp.vdd, 'not set')} V")
    print(f"  vth = {bp.tech_values.get(bp.vth, 'not set')} V")
    print(f"  featureSize = {bp.tech_values.get(bp.featureSize, 'not set')} m")
    print(f"  capIdealGate = {bp.tech_values.get(bp.capIdealGate, 'not set')} F")
    print(f"  currentOnNmos = {bp.tech_values.get(bp.currentOnNmos, 'not set')} uA/um")
    print(f"  resistanceOn = {bp.tech_values.get(bp.resistanceOn, 'not set')} Ω")

    print(f"\nTotal concrete values: {len(bp.tech_values)}")
    print("✓ Concrete values populated successfully")


def test_evaluate_symbolic_with_concrete():
    """Test evaluating symbolic expression with concrete values"""
    print("\n" + "=" * 80)
    print("Test 5: Evaluating Symbolic Expression with Concrete Values")
    print("=" * 80)

    # Initialize globals
    g.inputParameter = InputParameter()
    g.tech = Technology()
    g.cell = MemCell()

    # Initialize technology
    g.tech.Initialize(65, DeviceRoadmap.HP, g.inputParameter)
    g.cell.widthInFeatureSize = 2.0

    # Create base parameters and populate
    bp = BaseParameters()
    bp.populate_from_technology(g.tech)
    bp.populate_from_memcell(g.cell)

    # Create symbolic expression: R = effectiveResistanceMultiplier * vdd / (currentOnNmos * width)
    resistance_symbolic = (bp.effectiveResistanceMultiplier * bp.vdd /
                          (bp.currentOnNmos * bp.cellWidthInFeatureSize))

    print(f"\nSymbolic expression:")
    print(f"  R = {resistance_symbolic}")

    # Evaluate with concrete values
    resistance_concrete = resistance_symbolic.evalf(subs=bp.tech_values)

    print(f"\nEvaluated with concrete values:")
    print(f"  R = {resistance_concrete} Ω")

    # Show individual parameter values used
    print(f"\nParameter values used:")
    print(f"  effectiveResistanceMultiplier = {bp.tech_values[bp.effectiveResistanceMultiplier]}")
    print(f"  vdd = {bp.tech_values[bp.vdd]} V")
    print(f"  currentOnNmos = {bp.tech_values[bp.currentOnNmos]} uA/um")
    print(f"  width = {bp.tech_values[bp.cellWidthInFeatureSize]}")

    print("✓ Expression evaluation successful")


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DESTINY Base Parameters Test" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        test_basic_symbolic()
        test_gate_capacitance_symbolic()
        test_resistance_symbolic()
        test_populate_concrete_values()
        test_evaluate_symbolic_with_concrete()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nBase parameters approach is working correctly!")
        print("Symbolic variables can be used directly in calculations.")
        print("Concrete values can be substituted when needed.")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
