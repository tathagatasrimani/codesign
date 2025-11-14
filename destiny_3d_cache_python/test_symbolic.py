#!/usr/bin/env python3
"""
Test and demonstration of symbolic modeling in DESTINY
This shows how symbolic computation works in parallel with numerical computation
"""

import sympy as sp
from symbolic_wrapper import (
    SymbolicValue, symbolic_sqrt, symbolic_log, symbolic_pow,
    symbolic_min, symbolic_max, assert_symbolic_match,
    enable_symbolic_computation, is_symbolic_enabled
)


def test_basic_operations():
    """Test basic arithmetic operations with symbolic values"""
    print("=" * 80)
    print("Test 1: Basic Arithmetic Operations")
    print("=" * 80)

    # Create symbolic variables
    x = SymbolicValue(concrete=5.0, name='x')
    z = SymbolicValue(concrete=3.0, name='z')

    print(f"\nx = {x}")
    print(f"z = {z}")

    # Addition
    y = x + z
    print(f"\ny = x + z")
    print(f"  Concrete: {y.concrete}")
    print(f"  Symbolic: {y.symbolic}")

    # Verify they match
    subs = {'x': 5.0, 'z': 3.0}
    assert_symbolic_match(y.concrete, y.symbolic, subs, context="addition")
    print("  ✓ Symbolic and concrete match!")

    # Multiplication
    y2 = x * z
    print(f"\ny2 = x * z")
    print(f"  Concrete: {y2.concrete}")
    print(f"  Symbolic: {y2.symbolic}")
    assert_symbolic_match(y2.concrete, y2.symbolic, subs, context="multiplication")
    print("  ✓ Symbolic and concrete match!")

    # Division
    y3 = x / z
    print(f"\ny3 = x / z")
    print(f"  Concrete: {y3.concrete}")
    print(f"  Symbolic: {y3.symbolic}")
    assert_symbolic_match(y3.concrete, y3.symbolic, subs, context="division")
    print("  ✓ Symbolic and concrete match!")

    # Power
    y4 = x ** 2
    print(f"\ny4 = x ** 2")
    print(f"  Concrete: {y4.concrete}")
    print(f"  Symbolic: {y4.symbolic}")
    assert_symbolic_match(y4.concrete, y4.symbolic, {'x': 5.0}, context="power")
    print("  ✓ Symbolic and concrete match!")


def test_gate_capacitance_example():
    """
    Test symbolic modeling with gate capacitance calculation
    This mirrors the calculate_gate_cap function from formula.py
    """
    print("\n" + "=" * 80)
    print("Test 2: Gate Capacitance Calculation (from formula.py)")
    print("=" * 80)

    # Create symbolic technology parameters
    capIdealGate = SymbolicValue(concrete=6.38e-10, name='capIdealGate')
    capOverlap = SymbolicValue(concrete=1.276e-10, name='capOverlap')  # 20% of capIdealGate
    capFringe = SymbolicValue(concrete=2.5e-10, name='capFringe')
    phyGateLength = SymbolicValue(concrete=0.037e-6, name='phyGateLength')
    capPolywire = SymbolicValue(concrete=0.0, name='capPolywire')
    width = SymbolicValue(concrete=100e-9, name='width')

    print(f"\nTechnology Parameters:")
    print(f"  capIdealGate  = {capIdealGate.concrete} F")
    print(f"  capOverlap    = {capOverlap.concrete} F")
    print(f"  capFringe     = {capFringe.concrete} F")
    print(f"  phyGateLength = {phyGateLength.concrete} m")
    print(f"  width         = {width.concrete} m")

    # Calculate gate capacitance: cap = (capIdealGate + capOverlap + 3*capFringe) * width + phyGateLength * capPolywire
    gate_cap = (capIdealGate + capOverlap + 3 * capFringe) * width + phyGateLength * capPolywire

    print(f"\nGate Capacitance Calculation:")
    print(f"  cap = (capIdealGate + capOverlap + 3*capFringe) * width + phyGateLength * capPolywire")
    print(f"  Concrete result: {gate_cap.concrete:.6e} F")
    print(f"  Symbolic result: {gate_cap.symbolic}")

    # Verify they match
    subs = {
        'capIdealGate': 6.38e-10,
        'capOverlap': 1.276e-10,
        'capFringe': 2.5e-10,
        'phyGateLength': 0.037e-6,
        'capPolywire': 0.0,
        'width': 100e-9
    }
    assert_symbolic_match(gate_cap.concrete, gate_cap.symbolic, subs, context="gate_cap")
    print("  ✓ Symbolic and concrete match!")


def test_resistance_calculation():
    """
    Test symbolic modeling with resistance calculation
    This mirrors calculate_on_resistance from formula.py
    """
    print("\n" + "=" * 80)
    print("Test 3: On-Resistance Calculation (from formula.py)")
    print("=" * 80)

    # Create symbolic parameters
    effectiveResistanceMultiplier = SymbolicValue(concrete=1.54, name='R_eff_mult')
    vdd = SymbolicValue(concrete=1.2, name='vdd')
    currentOnNmos = SymbolicValue(concrete=1050.5, name='I_on')  # uA/um at 300K
    width = SymbolicValue(concrete=100e-9, name='width')  # 100nm

    print(f"\nParameters:")
    print(f"  effectiveResistanceMultiplier = {effectiveResistanceMultiplier.concrete}")
    print(f"  vdd = {vdd.concrete} V")
    print(f"  currentOnNmos = {currentOnNmos.concrete} uA/um")
    print(f"  width = {width.concrete} m")

    # Calculate resistance: R = R_eff_mult * vdd / (I_on * width)
    resistance = effectiveResistanceMultiplier * vdd / (currentOnNmos * width)

    print(f"\nResistance Calculation:")
    print(f"  R = R_eff_mult * vdd / (I_on * width)")
    print(f"  Concrete result: {resistance.concrete:.6e} Ω")
    print(f"  Symbolic result: {resistance.symbolic}")

    # Simplify the symbolic expression
    print(f"  Simplified: {sp.simplify(resistance.symbolic)}")

    # Verify they match
    subs = {
        'R_eff_mult': 1.54,
        'vdd': 1.2,
        'I_on': 1050.5,
        'width': 100e-9
    }
    assert_symbolic_match(resistance.concrete, resistance.symbolic, subs, context="resistance")
    print("  ✓ Symbolic and concrete match!")


def test_complex_expression():
    """Test a more complex expression with square roots and logarithms"""
    print("\n" + "=" * 80)
    print("Test 4: Complex Expression with sqrt and log")
    print("=" * 80)

    # Create symbolic variables
    a = SymbolicValue(concrete=4.0, name='a')
    b = SymbolicValue(concrete=16.0, name='b')
    c = SymbolicValue(concrete=2.0, name='c')

    print(f"\na = {a.concrete}")
    print(f"b = {b.concrete}")
    print(f"c = {c.concrete}")

    # Complex calculation: result = sqrt(a) + log(b) / c
    result = symbolic_sqrt(a) + symbolic_log(b) / c

    print(f"\nresult = sqrt(a) + log(b) / c")
    print(f"  Concrete: {result.concrete:.6f}")
    print(f"  Symbolic: {result.symbolic}")
    print(f"  Simplified: {sp.simplify(result.symbolic)}")

    # Verify
    subs = {'a': 4.0, 'b': 16.0, 'c': 2.0}
    assert_symbolic_match(result.concrete, result.symbolic, subs, context="complex")
    print("  ✓ Symbolic and concrete match!")


def test_branching_with_concrete():
    """
    Test that branching decisions use concrete values
    This demonstrates the user's requirement: "for branches just use the concrete values"
    """
    print("\n" + "=" * 80)
    print("Test 5: Branching with Concrete Values")
    print("=" * 80)

    # Create symbolic variables
    x = SymbolicValue(concrete=5.0, name='x')
    y = SymbolicValue(concrete=3.0, name='y')
    threshold = SymbolicValue(concrete=4.0, name='threshold')

    print(f"\nx = {x.concrete}")
    print(f"y = {y.concrete}")
    print(f"threshold = {threshold.concrete}")

    # Branch using concrete values
    print(f"\nif x > threshold:")
    if x > threshold:  # Uses concrete values: 5.0 > 4.0 = True
        result = x + y
        print(f"  Branch taken: result = x + y")
    else:
        result = x - y
        print(f"  Branch NOT taken")

    print(f"  Concrete: {result.concrete}")
    print(f"  Symbolic: {result.symbolic}")

    # Verify
    subs = {'x': 5.0, 'y': 3.0}
    assert_symbolic_match(result.concrete, result.symbolic, subs, context="branching")
    print("  ✓ Symbolic and concrete match!")


def test_min_max():
    """Test min/max operations"""
    print("\n" + "=" * 80)
    print("Test 6: Min/Max Operations")
    print("=" * 80)

    a = SymbolicValue(concrete=10.0, name='a')
    b = SymbolicValue(concrete=7.0, name='b')

    min_val = symbolic_min(a, b)
    max_val = symbolic_max(a, b)

    print(f"\na = {a.concrete}")
    print(f"b = {b.concrete}")
    print(f"\nmin(a, b):")
    print(f"  Concrete: {min_val.concrete}")
    print(f"  Symbolic: {min_val.symbolic}")

    print(f"\nmax(a, b):")
    print(f"  Concrete: {max_val.concrete}")
    print(f"  Symbolic: {max_val.symbolic}")

    # Verify
    subs = {'a': 10.0, 'b': 7.0}
    assert_symbolic_match(min_val.concrete, min_val.symbolic, subs, context="min")
    assert_symbolic_match(max_val.concrete, max_val.symbolic, subs, context="max")
    print("  ✓ All symbolic and concrete match!")


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 18 + "DESTINY Symbolic Modeling Demonstration" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")

    enable_symbolic_computation()
    print(f"\nSymbolic computation enabled: {is_symbolic_enabled()}")

    try:
        test_basic_operations()
        test_gate_capacitance_example()
        test_resistance_calculation()
        test_complex_expression()
        test_branching_with_concrete()
        test_min_max()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nSymbolic modeling framework is working correctly.")
        print("This demonstrates parallel symbolic and numerical computation.")
        print("\nNext steps:")
        print("  1. Add symbolic attributes to Technology class")
        print("  2. Add symbolic attributes to MemCell class")
        print("  3. Modify formula.py functions to use SymbolicValue")
        print("  4. Run full DESTINY simulation with symbolic tracking")
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
