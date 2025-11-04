#!/usr/bin/env python3
"""
Comprehensive example of symbolic modeling with DESTINY
Shows real circuit calculations with symbolic parameters
"""

import globals as g
from base_parameters import BaseParameters
from Technology import Technology
from MemCell import MemCell
from InputParameter import InputParameter
from typedef import DeviceRoadmap, MemCellType
from sympy import simplify, diff, latex, pprint
import math


def example_1_gate_capacitance():
    """Example 1: Gate Capacitance Calculation (from formula.py)"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Gate Capacitance Calculation")
    print("=" * 80)
    print("Formula: cap = (capIdealGate + capOverlap + 3*capFringe) * width")
    print("                + phyGateLength * capPolywire")

    # Initialize
    bp = BaseParameters()
    g.tech = Technology()
    g.tech.Initialize(65, DeviceRoadmap.HP, InputParameter())
    bp.populate_from_technology(g.tech)

    # Define transistor width (symbolic)
    width = 100e-9  # 100nm

    # Symbolic calculation
    gate_cap_sym = ((bp.capIdealGate + bp.capOverlap + 3*bp.capFringe) * width +
                    bp.phyGateLength * bp.capPolywire)

    print(f"\n📐 Symbolic Expression:")
    print(f"   {gate_cap_sym}")

    # Evaluate with concrete values
    gate_cap_val = gate_cap_sym.evalf(subs=bp.tech_values)

    print(f"\n📊 Numerical Result:")
    print(f"   Gate Capacitance = {float(gate_cap_val):.6e} F")
    print(f"   Gate Capacitance = {float(gate_cap_val)*1e15:.3f} fF")

    # Show parameter values
    print(f"\n🔧 Parameter Values Used:")
    print(f"   capIdealGate = {bp.tech_values[bp.capIdealGate]:.3e} F")
    print(f"   capOverlap   = {bp.tech_values[bp.capOverlap]:.3e} F")
    print(f"   capFringe    = {bp.tech_values[bp.capFringe]:.3e} F")
    print(f"   phyGateLength = {bp.tech_values[bp.phyGateLength]:.3e} m")
    print(f"   width        = {width:.3e} m")

    # Sensitivity analysis
    print(f"\n🔍 Sensitivity Analysis (derivatives):")
    sens_idealGate = diff(gate_cap_sym, bp.capIdealGate)
    sens_fringe = diff(gate_cap_sym, bp.capFringe)
    print(f"   ∂cap/∂capIdealGate = {sens_idealGate}")
    print(f"   ∂cap/∂capFringe    = {sens_fringe}")

    return gate_cap_sym


def example_2_transistor_resistance():
    """Example 2: Transistor On-Resistance (from formula.py)"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Transistor On-Resistance Calculation")
    print("=" * 80)
    print("Formula: R = effectiveResistanceMultiplier * vdd / (currentOnNmos * width)")

    # Initialize
    bp = BaseParameters()
    g.tech = Technology()
    g.tech.Initialize(65, DeviceRoadmap.HP, InputParameter())
    bp.populate_from_technology(g.tech)

    # Define transistor width
    width = 2.0  # Feature sizes

    # Symbolic calculation
    resistance_sym = (bp.effectiveResistanceMultiplier * bp.vdd /
                     (bp.currentOnNmos * width))

    print(f"\n📐 Symbolic Expression:")
    print(f"   {resistance_sym}")

    print(f"\n   Simplified:")
    print(f"   {simplify(resistance_sym)}")

    # Evaluate with concrete values
    resistance_val = resistance_sym.evalf(subs=bp.tech_values)

    print(f"\n📊 Numerical Result:")
    print(f"   On-Resistance = {float(resistance_val):.6e} Ω")
    print(f"   On-Resistance = {float(resistance_val)*1e3:.3f} mΩ")

    # Show parameter values
    print(f"\n🔧 Parameter Values Used:")
    print(f"   effectiveResistanceMultiplier = {bp.tech_values[bp.effectiveResistanceMultiplier]}")
    print(f"   vdd = {bp.tech_values[bp.vdd]} V")
    print(f"   currentOnNmos = {bp.tech_values[bp.currentOnNmos]} μA/μm")
    print(f"   width = {width} (feature sizes)")

    # Sensitivity analysis
    print(f"\n🔍 Sensitivity Analysis:")
    sens_vdd = diff(resistance_sym, bp.vdd)
    sens_current = diff(resistance_sym, bp.currentOnNmos)
    print(f"   ∂R/∂vdd = {simplify(sens_vdd)}")
    print(f"   ∂R/∂I_on = {simplify(sens_current)}")

    # Show that resistance decreases with current
    print(f"\n💡 Insight: Resistance is inversely proportional to current")
    print(f"   If current doubles, resistance halves")

    return resistance_sym


def example_3_power_calculation():
    """Example 3: Dynamic Power Calculation"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Dynamic Power Calculation")
    print("=" * 80)
    print("Formula: P = C * V² * f")
    print("         where C is capacitance, V is voltage, f is frequency")

    # Initialize
    bp = BaseParameters()
    g.tech = Technology()
    g.tech.Initialize(65, DeviceRoadmap.HP, InputParameter())
    bp.populate_from_technology(g.tech)

    # Define parameters
    width = 100e-9  # 100nm
    frequency = 1e9  # 1 GHz

    # Capacitance (from example 1)
    capacitance_sym = ((bp.capIdealGate + bp.capOverlap + 3*bp.capFringe) * width +
                       bp.phyGateLength * bp.capPolywire)

    # Power calculation
    power_sym = capacitance_sym * bp.vdd**2 * frequency

    print(f"\n📐 Symbolic Expression:")
    print(f"   P = {power_sym}")

    # Evaluate
    power_val = power_sym.evalf(subs=bp.tech_values)

    print(f"\n📊 Numerical Result:")
    print(f"   Dynamic Power = {float(power_val):.6e} W")
    print(f"   Dynamic Power = {float(power_val)*1e6:.3f} μW")

    # Show how power scales with voltage
    print(f"\n💡 Voltage Scaling Analysis:")
    print(f"   Power ∝ V²")
    print(f"   If V reduces by 20% (0.8V), power reduces by 36%")

    # Demonstrate
    vdd_original = bp.tech_values[bp.vdd]
    vdd_scaled = vdd_original * 0.8

    power_original = power_sym.evalf(subs=bp.tech_values)
    bp.tech_values[bp.vdd] = vdd_scaled
    power_scaled = power_sym.evalf(subs=bp.tech_values)

    print(f"   Original: V = {vdd_original:.2f}V, P = {float(power_original)*1e6:.3f} μW")
    print(f"   Scaled:   V = {vdd_scaled:.2f}V, P = {float(power_scaled)*1e6:.3f} μW")
    print(f"   Reduction: {(1 - float(power_scaled)/float(power_original))*100:.1f}%")

    # Restore original value
    bp.tech_values[bp.vdd] = vdd_original

    return power_sym


def example_4_delay_calculation():
    """Example 4: RC Delay Calculation"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: RC Delay Calculation")
    print("=" * 80)
    print("Formula: delay = R * C")
    print("         where R is resistance, C is capacitance")

    # Initialize
    bp = BaseParameters()
    g.tech = Technology()
    g.tech.Initialize(65, DeviceRoadmap.HP, InputParameter())
    bp.populate_from_technology(g.tech)

    # Define parameters
    width = 2.0
    transistor_width = 100e-9

    # Resistance calculation
    resistance_sym = (bp.effectiveResistanceMultiplier * bp.vdd /
                     (bp.currentOnNmos * width))

    # Capacitance calculation
    capacitance_sym = ((bp.capIdealGate + bp.capOverlap + 3*bp.capFringe) * transistor_width +
                       bp.phyGateLength * bp.capPolywire)

    # Delay calculation
    delay_sym = resistance_sym * capacitance_sym

    print(f"\n📐 Symbolic Expression:")
    print(f"   delay = R * C")
    print(f"   delay = ({resistance_sym}) * ({capacitance_sym})")

    # Evaluate
    delay_val = delay_sym.evalf(subs=bp.tech_values)

    print(f"\n📊 Numerical Result:")
    print(f"   Delay = {float(delay_val):.6e} s")
    print(f"   Delay = {float(delay_val)*1e12:.3f} ps")

    # Technology scaling
    print(f"\n💡 Technology Scaling:")
    print(f"   65nm: delay = {float(delay_val)*1e12:.3f} ps")

    # Try 45nm
    g.tech.Initialize(45, DeviceRoadmap.HP, InputParameter())
    bp.populate_from_technology(g.tech)
    delay_45nm = delay_sym.evalf(subs=bp.tech_values)
    print(f"   45nm: delay = {float(delay_45nm)*1e12:.3f} ps")
    print(f"   Speedup: {float(delay_val)/float(delay_45nm):.2f}x")

    return delay_sym


def example_5_memory_cell_energy():
    """Example 5: Memory Cell Write Energy"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Memory Cell Write Energy Calculation")
    print("=" * 80)
    print("Formula: E = V * (V - V_drop) / R * t_pulse")
    print("         (for voltage-mode write)")

    # Initialize
    bp = BaseParameters()
    g.cell = MemCell()

    # Set up example SRAM cell parameters
    bp.tech_values[bp.setVoltage] = 1.2  # V
    bp.tech_values[bp.voltageDropAccessDevice] = 0.1  # V
    bp.tech_values[bp.resistanceOn] = 1000.0  # Ω
    bp.tech_values[bp.setPulse] = 1e-9  # 1 ns

    # Symbolic calculation
    write_energy_sym = (bp.setVoltage * (bp.setVoltage - bp.voltageDropAccessDevice) /
                       bp.resistanceOn * bp.setPulse)

    print(f"\n📐 Symbolic Expression:")
    print(f"   {write_energy_sym}")

    # Evaluate
    energy_val = write_energy_sym.evalf(subs=bp.tech_values)

    print(f"\n📊 Numerical Result:")
    print(f"   Write Energy = {float(energy_val):.6e} J")
    print(f"   Write Energy = {float(energy_val)*1e12:.3f} pJ")
    print(f"   Write Energy = {float(energy_val)*1e15:.3f} fJ")

    # Show parameter values
    print(f"\n🔧 Parameter Values Used:")
    print(f"   setVoltage = {bp.tech_values[bp.setVoltage]} V")
    print(f"   voltageDropAccessDevice = {bp.tech_values[bp.voltageDropAccessDevice]} V")
    print(f"   resistanceOn = {bp.tech_values[bp.resistanceOn]} Ω")
    print(f"   setPulse = {bp.tech_values[bp.setPulse]*1e9} ns")

    # Energy optimization
    print(f"\n💡 Energy Optimization:")
    print(f"   Energy ∝ V² (quadratic)")
    print(f"   Energy ∝ 1/R (inverse)")
    print(f"   Energy ∝ t_pulse (linear)")

    # Sensitivity
    sens_voltage = diff(write_energy_sym, bp.setVoltage)
    print(f"\n🔍 Sensitivity:")
    print(f"   ∂E/∂V = {simplify(sens_voltage)}")

    return write_energy_sym


def example_6_complete_circuit():
    """Example 6: Complete Circuit Analysis"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Complete CMOS Inverter Analysis")
    print("=" * 80)

    # Initialize
    bp = BaseParameters()
    g.tech = Technology()
    g.tech.Initialize(65, DeviceRoadmap.HP, InputParameter())
    bp.populate_from_technology(g.tech)

    # Inverter sizing (PMOS is pnSizeRatio times wider than NMOS)
    width_nmos = 2.0
    width_pmos = bp.pnSizeRatio * width_nmos
    width_transistor = 100e-9

    print(f"Inverter Configuration:")
    print(f"  NMOS width: {width_nmos} F")
    print(f"  PMOS width: {width_pmos} F (ratio = {bp.tech_values[bp.pnSizeRatio]})")

    # Input capacitance
    cap_nmos = (bp.capIdealGate + bp.capOverlap + 3*bp.capFringe) * width_transistor
    cap_pmos = (bp.capIdealGate + bp.capOverlap + 3*bp.capFringe) * width_transistor
    input_cap = cap_nmos + cap_pmos

    # On-resistance
    res_nmos = (bp.effectiveResistanceMultiplier * bp.vdd /
               (bp.currentOnNmos * width_nmos))
    res_pmos = (bp.effectiveResistanceMultiplier * bp.vdd /
               (bp.currentOnPmos * width_pmos))

    # Average delay (using average resistance)
    avg_resistance = (res_nmos + res_pmos) / 2
    output_cap = input_cap  # Assume driving same load
    delay = avg_resistance * output_cap

    # Dynamic power
    frequency = 1e9  # 1 GHz
    dynamic_power = input_cap * bp.vdd**2 * frequency

    # Leakage power
    leakage_power = (bp.currentOffNmos * width_nmos +
                    bp.currentOffPmos * width_pmos) * bp.vdd

    print(f"\n📐 Symbolic Expressions:")
    print(f"   Input Cap:  {simplify(input_cap)}")
    print(f"   Delay:      {simplify(delay)}")
    print(f"   Dyn Power:  {simplify(dynamic_power)}")
    print(f"   Leak Power: {simplify(leakage_power)}")

    # Evaluate all
    input_cap_val = input_cap.evalf(subs=bp.tech_values)
    delay_val = delay.evalf(subs=bp.tech_values)
    dynamic_power_val = dynamic_power.evalf(subs=bp.tech_values)
    leakage_power_val = leakage_power.evalf(subs=bp.tech_values)

    print(f"\n📊 Numerical Results:")
    print(f"   Input Capacitance: {float(input_cap_val)*1e15:.3f} fF")
    print(f"   Propagation Delay: {float(delay_val)*1e12:.3f} ps")
    print(f"   Dynamic Power:     {float(dynamic_power_val)*1e6:.3f} μW")
    print(f"   Leakage Power:     {float(leakage_power_val)*1e9:.3f} nW")
    print(f"   Total Power:       {(float(dynamic_power_val)+float(leakage_power_val))*1e6:.3f} μW")

    # Energy-Delay Product
    edp = dynamic_power_val * delay_val
    print(f"\n💡 Energy-Delay Product:")
    print(f"   EDP = {float(edp)*1e24:.3f} fJ·ps")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "DESTINY Symbolic Parameter Examples" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\nThis demonstrates symbolic modeling with real DESTINY calculations")

    try:
        example_1_gate_capacitance()
        example_2_transistor_resistance()
        example_3_power_calculation()
        example_4_delay_calculation()
        example_5_memory_cell_energy()
        example_6_complete_circuit()

        print("\n" + "=" * 80)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  • Symbolic expressions show exact mathematical relationships")
        print("  • Can evaluate with different technology nodes")
        print("  • Enables sensitivity analysis (derivatives)")
        print("  • Supports design space exploration")
        print("  • Useful for optimization and verification")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n✗ EXAMPLE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
