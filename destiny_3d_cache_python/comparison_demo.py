#!/usr/bin/env python3
"""
Side-by-side comparison showing DESTINY symbolic modeling
follows EXACT same pattern as CACTI
"""

from base_parameters import BaseParameters
import globals as g
from Technology import Technology
from InputParameter import InputParameter
from typedef import DeviceRoadmap
from sympy import diff, simplify


def show_comparison():
    print("=" * 80)
    print("SYMBOLIC MODELING: CACTI vs DESTINY Comparison")
    print("=" * 80)
    
    print("\n" + "─" * 80)
    print("STEP 1: Create Symbolic Variables")
    print("─" * 80)
    
    print("\n📘 CACTI Approach:")
    print("""
    class BaseParameters:
        def __init__(self):
            self.V_dd = symbols("V_dd", positive=True)
            self.I_on_n = symbols("I_on_n", positive=True)
            self.C_g_ideal = symbols("C_g_ideal", positive=True)
    """)
    
    print("📗 DESTINY Approach (IDENTICAL!):")
    print("""
    class BaseParameters:
        def __init__(self):
            self.vdd = symbols("vdd", positive=True)
            self.currentOnNmos = symbols("currentOnNmos", positive=True)
            self.capIdealGate = symbols("capIdealGate", positive=True)
    """)
    
    # Actually create them
    bp = BaseParameters()
    print(f"\n✅ Created: {bp.vdd}, {bp.currentOnNmos}, {bp.capIdealGate}")
    
    print("\n" + "─" * 80)
    print("STEP 2: Store Concrete Values")
    print("─" * 80)
    
    print("\n📘 CACTI Approach:")
    print("""
    self.tech_values = {}
    self.tech_values[self.V_dd] = 1.1
    self.tech_values[self.I_on_n] = 1211.4
    """)
    
    print("📗 DESTINY Approach (IDENTICAL!):")
    print("""
    bp.tech_values = {}
    bp.tech_values[bp.vdd] = 1.1
    bp.tech_values[bp.currentOnNmos] = 1211.4
    """)
    
    # Actually do it
    g.tech = Technology()
    g.tech.Initialize(65, DeviceRoadmap.HP, InputParameter())
    bp.populate_from_technology(g.tech)
    
    print(f"\n✅ Stored: vdd={bp.tech_values[bp.vdd]}, I_on={bp.tech_values[bp.currentOnNmos]}")
    
    print("\n" + "─" * 80)
    print("STEP 3: Create Symbolic Expression")
    print("─" * 80)
    
    print("\n📘 CACTI Approach:")
    print("""
    # Transistor resistance
    R = self.nmos_eff_res_mult * self.V_dd / self.I_on_n
    """)
    
    print("📗 DESTINY Approach (IDENTICAL!):")
    print("""
    # Transistor resistance
    R = bp.effectiveResistanceMultiplier * bp.vdd / bp.currentOnNmos
    """)
    
    # Actually create it
    R = bp.effectiveResistanceMultiplier * bp.vdd / bp.currentOnNmos
    print(f"\n✅ Symbolic Expression: {R}")
    
    print("\n" + "─" * 80)
    print("STEP 4: Evaluate with Concrete Values")
    print("─" * 80)
    
    print("\n📘 CACTI Approach:")
    print("""
    result = R.evalf(subs=self.tech_values)
    print(f"Resistance = {result} Ω")
    """)
    
    print("📗 DESTINY Approach (IDENTICAL!):")
    print("""
    result = R.evalf(subs=bp.tech_values)
    print(f"Resistance = {result} Ω")
    """)
    
    # Actually evaluate
    result = R.evalf(subs=bp.tech_values)
    print(f"\n✅ Evaluated: Resistance = {result} Ω = {float(result)*1e3:.3f} mΩ")
    
    print("\n" + "─" * 80)
    print("STEP 5: Sensitivity Analysis")
    print("─" * 80)
    
    print("\n📘 CACTI Approach:")
    print("""
    dR_dV = diff(R, self.V_dd)
    print(f"∂R/∂V_dd = {dR_dV}")
    """)
    
    print("📗 DESTINY Approach (IDENTICAL!):")
    print("""
    dR_dV = diff(R, bp.vdd)
    print(f"∂R/∂vdd = {dR_dV}")
    """)
    
    # Actually compute
    dR_dV = diff(R, bp.vdd)
    print(f"\n✅ Derivative: ∂R/∂vdd = {simplify(dR_dV)}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION: DESTINY uses EXACT SAME APPROACH as CACTI!")
    print("=" * 80)
    print("\nBoth frameworks:")
    print("  ✓ Use SymPy symbols()")
    print("  ✓ Store values in tech_values dictionary")
    print("  ✓ Create expressions directly")
    print("  ✓ Evaluate with .evalf(subs=...)")
    print("  ✓ Take derivatives with diff()")
    print("\nOnly differences: variable names and data sources")
    print("=" * 80 + "\n")


def show_complete_example():
    print("\n" + "=" * 80)
    print("COMPLETE EXAMPLE: Gate Capacitance")
    print("=" * 80)
    
    # Setup
    bp = BaseParameters()
    g.tech = Technology()
    g.tech.Initialize(65, DeviceRoadmap.HP, InputParameter())
    bp.populate_from_technology(g.tech)
    
    width = 100e-9  # 100nm
    
    print("\n1️⃣ Define symbolic expression (same as CACTI):")
    gate_cap = ((bp.capIdealGate + bp.capOverlap + 3*bp.capFringe) * width +
                bp.phyGateLength * bp.capPolywire)
    print(f"   cap = {gate_cap}")
    
    print("\n2️⃣ Evaluate (same as CACTI):")
    result = gate_cap.evalf(subs=bp.tech_values)
    print(f"   cap = {float(result)*1e15:.3f} fF")
    
    print("\n3️⃣ Sensitivity (same as CACTI):")
    dcap_dCg = diff(gate_cap, bp.capIdealGate)
    print(f"   ∂cap/∂C_g_ideal = {dcap_dCg}")
    
    print("\n4️⃣ Parameter sweep (same as CACTI):")
    original_cap = bp.tech_values[bp.capIdealGate]
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        bp.tech_values[bp.capIdealGate] = original_cap * scale
        cap_val = gate_cap.evalf(subs=bp.tech_values)
        print(f"   C_g scaled by {scale:.1f}x → cap = {float(cap_val)*1e15:.3f} fF")
    
    print("\n✅ Everything works exactly like CACTI!")


if __name__ == "__main__":
    show_comparison()
    show_complete_example()
