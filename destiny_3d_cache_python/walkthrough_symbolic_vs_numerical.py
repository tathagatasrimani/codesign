#!/usr/bin/env python3
"""
Step-by-step walkthrough: Symbolic vs Numerical Modeling
Shows how both approaches give the same answer for bitline delay
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import math
import globals as g
from InputParameter import InputParameter
from Technology import Technology
from MemCell import MemCell
from SubArray import SubArray
from Wire import Wire
from typedef import WireType, WireRepeaterType, BufferDesignTarget
from formula import calculate_on_resistance, calculate_transconductance, horowitz
from constant import NMOS

# For symbolic
from sympy import symbols, simplify


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def main():
    print("="*80)
    print("WALKTHROUGH: SYMBOLIC vs NUMERICAL MODELING")
    print("Configuration: 65nm SRAM, 1024×2048 subarray, 2-layer stack")
    print("="*80)

    # =========================================================================
    # PART 1: Initialize Python DESTINY (Numerical)
    # =========================================================================
    print_section("PART 1: NUMERICAL MODELING (Python DESTINY)")

    print("\n📦 Initializing Python DESTINY...")

    g.inputParameter = InputParameter()
    g.inputParameter.ReadInputParameterFromFile("config/sample_SRAM_2layer.cfg")

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

    print("✓ Initialized!")

    # Create subarray
    print("\n📦 Creating subarray (1024 rows × 2048 columns)...")
    subarray = SubArray()
    subarray.Initialize(1024, 2048, 1, 1, 1, True, 1, 8,
                       BufferDesignTarget.latency_first, 2)

    subarray.CalculateArea()
    subarray.CalculateLatency(1e20)

    print("✓ Subarray created and analyzed!")

    # =========================================================================
    # NUMERICAL: Step 1 - Extract Parameters
    # =========================================================================
    print_section("NUMERICAL Step 1: Extract Physical Parameters")

    print("\n🔍 From Technology & Wire:")
    print(f"  Process node: {g.inputParameter.processNode} nm")
    print(f"  Temperature: {g.inputParameter.temperature} K")
    print(f"  V_dd: {g.tech.vdd} V")
    print(f"  Wire R/C per unit:")
    print(f"    resWirePerUnit = {g.localWire.resWirePerUnit:.6e} Ω/m")
    print(f"    capWirePerUnit = {g.localWire.capWirePerUnit:.6e} F/m")

    print("\n🔍 From Cell:")
    print(f"  Cell dimensions:")
    print(f"    heightInFeatureSize = {g.cell.heightInFeatureSize} F")
    print(f"    widthInFeatureSize = {g.cell.widthInFeatureSize} F")
    print(f"  SRAM transistor widths:")
    print(f"    widthAccessCMOS = {g.cell.widthAccessCMOS} × min")
    print(f"    widthSRAMCellNMOS = {g.cell.widthSRAMCellNMOS} × min")

    # =========================================================================
    # NUMERICAL: Step 2 - Calculate Bitline R/C
    # =========================================================================
    print_section("NUMERICAL Step 2: Calculate Bitline R/C")

    print("\n📐 Bitline length:")
    print(f"  lenBitline = numRow × cellHeight × featureSize")
    print(f"             = {subarray.numRow} × {g.cell.heightInFeatureSize} × {g.devtech.featureSize*1e9:.1f}e-9")
    print(f"             = {subarray.lenBitline:.6e} m")

    print("\n📐 Bitline wire capacitance:")
    print(f"  C_wire = lenBitline × capWirePerUnit")
    wire_cap = subarray.lenBitline * g.localWire.capWirePerUnit
    print(f"         = {subarray.lenBitline:.6e} × {g.localWire.capWirePerUnit:.6e}")
    print(f"         = {wire_cap:.6e} F")
    print(f"         = {wire_cap*1e15:.3f} fF")

    print("\n📐 Bitline drain capacitance (shared contact):")
    print(f"  C_drain = capCellAccess × numRow / 2")
    drain_cap = subarray.capCellAccess * subarray.numRow / 2
    print(f"          = {subarray.capCellAccess:.6e} × {subarray.numRow} / 2")
    print(f"          = {drain_cap:.6e} F")
    print(f"          = {drain_cap*1e15:.3f} fF")

    print("\n📐 Total bitline capacitance:")
    print(f"  capBitline = C_wire + C_drain")
    print(f"             = {wire_cap*1e15:.3f} + {drain_cap*1e15:.3f}")
    print(f"             = {subarray.capBitline*1e15:.3f} fF")

    print("\n📐 Bitline resistance:")
    print(f"  resBitline = lenBitline × resWirePerUnit")
    print(f"             = {subarray.lenBitline:.6e} × {g.localWire.resWirePerUnit:.6e}")
    print(f"             = {subarray.resBitline:.3f} Ω")

    # =========================================================================
    # NUMERICAL: Step 3 - Calculate Access & Pull-down Resistances
    # =========================================================================
    print_section("NUMERICAL Step 3: Access & Pull-down Resistances")

    print("\n📐 Cell access resistance:")
    print(f"  resCellAccess = {subarray.resCellAccess:.3f} Ω")

    print("\n📐 SRAM pull-down resistance:")
    resPullDown = calculate_on_resistance(
        g.cell.widthSRAMCellNMOS * g.tech.featureSize,
        NMOS,
        g.inputParameter.temperature,
        g.tech
    )
    print(f"  resPullDown = CalculateOnResistance(widthNMOS, NMOS, T)")
    print(f"              = {resPullDown:.3f} Ω")

    # =========================================================================
    # NUMERICAL: Step 4 - Calculate Time Constant (tau)
    # =========================================================================
    print_section("NUMERICAL Step 4: Calculate Time Constant (tau)")

    print("\n📐 Tau formula (from SubArray.py line 507-508):")
    print("  tau = (R_access + R_pulldown) × (C_access + C_bitline + C_mux)")
    print("      + R_bitline × (C_mux + C_bitline/2)")

    print(f"\n📊 Values:")
    print(f"  R_access = {subarray.resCellAccess:.3f} Ω")
    print(f"  R_pulldown = {resPullDown:.3f} Ω")
    print(f"  C_access = {subarray.capCellAccess*1e15:.3f} fF")
    print(f"  C_bitline = {subarray.capBitline*1e15:.3f} fF")
    print(f"  C_mux = 0.000 fF (no mux in this config)")
    print(f"  R_bitline = {subarray.resBitline:.3f} Ω")

    C_mux = 0.0  # No mux in this configuration
    term1 = (subarray.resCellAccess + resPullDown) * (subarray.capCellAccess + subarray.capBitline + C_mux)
    term2 = subarray.resBitline * (C_mux + subarray.capBitline / 2)
    tau_before_log = term1 + term2

    print(f"\n📐 Calculation:")
    print(f"  Term 1: (R_access + R_pulldown) × (C_access + C_bitline + C_mux)")
    print(f"        = ({subarray.resCellAccess:.1f} + {resPullDown:.1f}) × ({subarray.capCellAccess*1e15:.3f} + {subarray.capBitline*1e15:.3f} + 0)")
    print(f"        = {term1*1e9:.6f} ns")

    print(f"\n  Term 2: R_bitline × (C_mux + C_bitline/2)")
    print(f"        = {subarray.resBitline:.1f} × (0 + {subarray.capBitline*1e15/2:.3f})")
    print(f"        = {term2*1e9:.6f} ns")

    print(f"\n  tau (before log) = {term1*1e9:.6f} + {term2*1e9:.6f}")
    print(f"                   = {tau_before_log*1e9:.6f} ns")

    # =========================================================================
    # NUMERICAL: Step 5 - Apply Logarithmic Factor
    # =========================================================================
    print_section("NUMERICAL Step 5: Apply Voltage Swing (Log Factor)")

    print("\n📐 Voltage parameters:")
    print(f"  V_precharge = {subarray.voltagePrecharge} V")
    print(f"  V_sense = {subarray.senseVoltage} V")

    log_factor = math.log(subarray.voltagePrecharge / (subarray.voltagePrecharge - subarray.senseVoltage / 2))
    tau_after_log = tau_before_log * log_factor

    print(f"\n📐 Log factor:")
    print(f"  log(V_pre / (V_pre - V_sense/2))")
    print(f"  = log({subarray.voltagePrecharge} / ({subarray.voltagePrecharge} - {subarray.senseVoltage}/2))")
    print(f"  = log({subarray.voltagePrecharge} / {subarray.voltagePrecharge - subarray.senseVoltage/2})")
    print(f"  = {log_factor:.6f}")

    print(f"\n📐 tau (after log) = {tau_before_log*1e9:.6f} × {log_factor:.6f}")
    print(f"                   = {tau_after_log*1e9:.6f} ns")

    # =========================================================================
    # NUMERICAL: Step 6 - Apply Horowitz Model
    # =========================================================================
    print_section("NUMERICAL Step 6: Apply Horowitz Model")

    gm = calculate_transconductance(g.cell.widthAccessCMOS * g.tech.featureSize, NMOS, g.tech)
    beta = 1.0 / (resPullDown * gm)

    print(f"\n📐 Transconductance:")
    print(f"  gm = CalculateTransconductance(widthAccessCMOS)")
    print(f"     = {gm:.6e} S")

    print(f"\n📐 Beta parameter:")
    print(f"  beta = 1 / (R_pulldown × gm)")
    print(f"       = 1 / ({resPullDown:.1f} × {gm:.6e})")
    print(f"       = {beta:.6f}")

    bitline_delay, _ = horowitz(tau_after_log, beta, 1e20)

    print(f"\n📐 Horowitz delay:")
    print(f"  delay = horowitz(tau, beta, rampInput)")
    print(f"        = {bitline_delay*1e9:.6f} ns")

    print(f"\n✅ NUMERICAL RESULT: {bitline_delay*1e9:.6f} ns")

    # =========================================================================
    # PART 2: SYMBOLIC MODELING
    # =========================================================================
    print_section("PART 2: SYMBOLIC MODELING")

    print("\n📝 Define symbolic variables:")

    # Define symbols
    rows = symbols('rows')
    R_per_cell = symbols('R_per_cell')
    C_per_cell_wire = symbols('C_per_cell_wire')
    R_access = symbols('R_access')
    R_pulldown = symbols('R_pulldown')
    C_access = symbols('C_access')
    C_mux_sym = symbols('C_mux')

    print("  rows, R_per_cell, C_per_cell_wire,")
    print("  R_access, R_pulldown, C_access, C_mux")

    # =========================================================================
    # SYMBOLIC: Build Expressions
    # =========================================================================
    print_section("SYMBOLIC Step 1: Build Symbolic Expressions")

    print("\n📐 Bitline resistance (symbolic):")
    R_bitline_expr = R_per_cell * rows
    print(f"  R_bitline = R_per_cell × rows")
    print(f"  R_bitline = {R_bitline_expr}")

    print("\n📐 Bitline wire capacitance (symbolic):")
    C_bitline_wire_expr = C_per_cell_wire * rows
    print(f"  C_bitline_wire = C_per_cell_wire × rows")
    print(f"  C_bitline_wire = {C_bitline_wire_expr}")

    print("\n📐 Bitline drain capacitance (symbolic):")
    C_bitline_drain_expr = C_access * rows / 2
    print(f"  C_bitline_drain = C_access × rows / 2")
    print(f"  C_bitline_drain = {C_bitline_drain_expr}")

    print("\n📐 Total bitline capacitance (symbolic):")
    C_bitline_expr = C_bitline_wire_expr + C_bitline_drain_expr
    print(f"  C_bitline = C_bitline_wire + C_bitline_drain")
    print(f"  C_bitline = {simplify(C_bitline_expr)}")

    print("\n📐 Time constant tau (symbolic):")
    tau_expr = ((R_access + R_pulldown) * (C_access + C_bitline_expr + C_mux_sym) +
                R_bitline_expr * (C_mux_sym + C_bitline_expr / 2))
    tau_simplified = simplify(tau_expr)
    print(f"  tau = (R_access + R_pulldown) × (C_access + C_bitline + C_mux)")
    print(f"      + R_bitline × (C_mux + C_bitline/2)")
    print(f"\n  Simplified:")
    print(f"  tau = {tau_simplified}")

    # =========================================================================
    # SYMBOLIC: Substitute Values
    # =========================================================================
    print_section("SYMBOLIC Step 2: Substitute Numerical Values")

    # Calculate per-cell values
    R_per_cell_val = g.localWire.resWirePerUnit * g.cell.heightInFeatureSize * g.devtech.featureSize
    C_per_cell_wire_val = g.localWire.capWirePerUnit * g.cell.heightInFeatureSize * g.devtech.featureSize

    print("\n📊 Parameter values:")
    print(f"  rows = {subarray.numRow}")
    print(f"  R_per_cell = resWirePerUnit × cellHeight × featureSize")
    print(f"             = {g.localWire.resWirePerUnit:.6e} × {g.cell.heightInFeatureSize} × {g.devtech.featureSize:.6e}")
    print(f"             = {R_per_cell_val:.6e} Ω")
    print(f"  C_per_cell_wire = capWirePerUnit × cellHeight × featureSize")
    print(f"                  = {g.localWire.capWirePerUnit:.6e} × {g.cell.heightInFeatureSize} × {g.devtech.featureSize:.6e}")
    print(f"                  = {C_per_cell_wire_val:.6e} F")
    print(f"  R_access = {subarray.resCellAccess:.3f} Ω")
    print(f"  R_pulldown = {resPullDown:.3f} Ω")
    print(f"  C_access = {subarray.capCellAccess:.6e} F")
    print(f"  C_mux = 0.0 F")

    # Substitute
    subs_dict = {
        rows: subarray.numRow,
        R_per_cell: R_per_cell_val,
        C_per_cell_wire: C_per_cell_wire_val,
        R_access: subarray.resCellAccess,
        R_pulldown: resPullDown,
        C_access: subarray.capCellAccess,
        C_mux_sym: 0.0
    }

    print("\n📐 Evaluating symbolic expressions:")

    R_bitline_symbolic = float(R_bitline_expr.subs(subs_dict))
    print(f"  R_bitline = {R_bitline_symbolic:.3f} Ω")

    C_bitline_symbolic = float(C_bitline_expr.subs(subs_dict))
    print(f"  C_bitline = {C_bitline_symbolic*1e15:.3f} fF")

    tau_symbolic_before_log = float(tau_expr.subs(subs_dict))
    print(f"  tau (before log) = {tau_symbolic_before_log*1e9:.6f} ns")

    tau_symbolic_after_log = tau_symbolic_before_log * log_factor
    print(f"  tau (after log) = {tau_symbolic_after_log*1e9:.6f} ns")

    # Apply Horowitz
    bitline_delay_symbolic, _ = horowitz(tau_symbolic_after_log, beta, 1e20)
    print(f"  delay (Horowitz) = {bitline_delay_symbolic*1e9:.6f} ns")

    print(f"\n✅ SYMBOLIC RESULT: {bitline_delay_symbolic*1e9:.6f} ns")

    # =========================================================================
    # FINAL COMPARISON
    # =========================================================================
    print_section("FINAL COMPARISON: Numerical vs Symbolic")

    print(f"\n{'Component':<30} {'Numerical':<20} {'Symbolic':<20} {'Match'}")
    print("-" * 90)

    res_match = abs(subarray.resBitline - R_bitline_symbolic) / subarray.resBitline < 0.001
    cap_match = abs(subarray.capBitline - C_bitline_symbolic) / subarray.capBitline < 0.001
    tau_match = abs(tau_after_log - tau_symbolic_after_log) / tau_after_log < 0.001
    delay_match = abs(bitline_delay - bitline_delay_symbolic) / bitline_delay < 0.001

    print(f"{'R_bitline':<30} {subarray.resBitline:>18.3f} Ω   {R_bitline_symbolic:>18.3f} Ω   {'✅' if res_match else '❌'}")
    print(f"{'C_bitline':<30} {subarray.capBitline*1e15:>18.3f} fF  {C_bitline_symbolic*1e15:>18.3f} fF  {'✅' if cap_match else '❌'}")
    print(f"{'tau (after log)':<30} {tau_after_log*1e9:>18.6f} ns  {tau_symbolic_after_log*1e9:>18.6f} ns  {'✅' if tau_match else '❌'}")
    print(f"{'Bitline Delay':<30} {bitline_delay*1e9:>18.6f} ns  {bitline_delay_symbolic*1e9:>18.6f} ns  {'✅' if delay_match else '❌'}")

    print("\n" + "="*80)
    if res_match and cap_match and tau_match and delay_match:
        print("✅ PERFECT MATCH!")
        print("Symbolic and Numerical modeling produce identical results!")
        print("\nKey insight: Symbolic expressions, when evaluated with actual")
        print("parameter values, give exactly the same answer as numerical simulation.")
        return 0
    else:
        print("❌ MISMATCH DETECTED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
