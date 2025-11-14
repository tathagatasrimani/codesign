#!/usr/bin/env python3
"""
Symbolic Access Time Calculator
Takes optimal configuration from C++ DESTINY DSE and computes symbolic expressions
for memory access time
"""

import globals as g
from base_parameters import BaseParameters
from Technology import Technology
from InputParameter import InputParameter
from MemCell import MemCell
from typedef import DeviceRoadmap, MemCellType, WireType, WireRepeaterType
from Wire import Wire
from parse_cpp_output import OptimalConfiguration, parse_cpp_destiny_output
from sympy import symbols, simplify, latex, pprint, sqrt, log
import sys


class SymbolicAccessTimeAnalyzer:
    """Computes symbolic expressions for memory access time components"""

    def __init__(self, config: OptimalConfiguration, input_param: InputParameter):
        self.config = config
        self.input_param = input_param
        self.bp = BaseParameters()

        # Initialize technology
        g.tech = Technology()
        g.tech.Initialize(
            input_param.processNode,
            input_param.deviceRoadmap,
            input_param
        )

        # Populate symbolic parameters
        self.bp.populate_from_technology(g.tech)

        # Initialize memory cell
        g.cell = MemCell()
        if len(input_param.fileMemCell) > 0:
            cellFile = input_param.fileMemCell[0]
            if '/' not in cellFile:
                import os
                cellFile = os.path.join('config', cellFile)
            g.cell.ReadCellFromFile(cellFile)

    def compute_row_decoder_delay(self):
        """Compute symbolic expression for row decoder delay"""
        print("\n" + "─" * 80)
        print("ROW DECODER DELAY")
        print("─" * 80)

        # Number of address bits to decode
        num_addr_bits = int(log(self.config.subarray_rows, 2).evalf())

        print(f"\nConfiguration:")
        print(f"  Subarray rows: {self.config.subarray_rows}")
        print(f"  Address bits to decode: {num_addr_bits}")

        # Decoder stages (typically log2 stages for hierarchical decoder)
        # Each stage: R * C delay
        # R = transistor resistance, C = wire + gate capacitance

        # Transistor resistance (using width = 2F)
        width_f = 2.0
        R_decoder = self.bp.effectiveResistanceMultiplier * self.bp.vdd / (self.bp.currentOnNmos * width_f)

        # Gate capacitance per stage
        C_gate = (self.bp.capIdealGate + self.bp.capOverlap + 3*self.bp.capFringe) * width_f * self.bp.featureSize

        # Wire capacitance (approximate - simplified model)
        # For row decoder, use estimated wire cap based on fanout
        fanout = 4  # Typical fanout per decoder stage
        C_wire = fanout * C_gate / 2  # Simplified model

        # Total capacitance per stage
        C_total = C_gate + C_wire

        # Delay per stage
        delay_per_stage = R_decoder * C_total

        # Total decoder delay (num_stages * delay_per_stage)
        # For hierarchical decoder, typically ~log2(rows) stages
        num_stages = max(1, int(log(num_addr_bits, 2).evalf())) if num_addr_bits > 1 else 1
        decoder_delay_sym = num_stages * delay_per_stage

        print(f"\n📐 Symbolic Expression:")
        print(f"   R_decoder = {R_decoder}")
        print(f"\n   C_gate = {C_gate}")
        print(f"\n   C_wire ≈ {C_wire}")
        print(f"\n   Delay per stage = R × C = {delay_per_stage}")
        print(f"\n   Number of stages: {num_stages}")
        print(f"\n   Total Decoder Delay = {simplify(decoder_delay_sym)}")

        # Evaluate
        delay_val = decoder_delay_sym.evalf(subs=self.bp.tech_values)

        print(f"\n📊 Numerical Result:")
        print(f"   Row Decoder Delay = {float(delay_val)*1e9:.3f} ns")
        print(f"   C++ DESTINY Result = {self.config.row_decoder_latency*1e9:.3f} ns")
        print(f"   Ratio (Symbolic/C++): {float(delay_val)/self.config.row_decoder_latency:.2f}")

        return decoder_delay_sym

    def compute_bitline_delay(self):
        """Compute symbolic expression for bitline delay"""
        print("\n" + "─" * 80)
        print("BITLINE DELAY")
        print("─" * 80)

        print(f"\nConfiguration:")
        print(f"  Bitline length (rows): {self.config.subarray_rows}")
        print(f"  Columns: {self.config.subarray_cols}")

        # Bitline is modeled as RC line
        # R = resistance per cell × num_cells
        # C = capacitance per cell × num_cells

        # Cell access transistor resistance
        # Use default SRAM access transistor width (typical ~1.31F)
        access_width = 1.31 if not hasattr(self.config, 'access_transistor_width') or self.config.access_transistor_width is None else self.config.access_transistor_width
        R_cell = self.bp.effectiveResistanceMultiplier * self.bp.vdd / (self.bp.currentOnNmos * access_width)

        # Bitline capacitance per cell (simplified model using available parameters)
        # Approximate bitline cap as gate capacitance of storage cells
        C_bitline_per_cell = (self.bp.capIdealGate + self.bp.capOverlap) * self.bp.featureSize

        # Total bitline RC
        R_bitline = R_cell * self.config.subarray_rows
        C_bitline = C_bitline_per_cell * self.config.subarray_rows

        # RC delay (using distributed RC: delay = 0.5 * R * C for line)
        bitline_delay_sym = 0.5 * R_bitline * C_bitline

        print(f"\n📐 Symbolic Expression:")
        print(f"   R_cell = {R_cell}")
        print(f"\n   C_cell = {C_bitline_per_cell}")
        print(f"\n   R_bitline = R_cell × {self.config.subarray_rows} = {simplify(R_bitline)}")
        print(f"\n   C_bitline = C_cell × {self.config.subarray_rows} = {simplify(C_bitline)}")
        print(f"\n   Bitline Delay = 0.5 × R × C = {simplify(bitline_delay_sym)}")

        # Evaluate
        delay_val = bitline_delay_sym.evalf(subs=self.bp.tech_values)

        print(f"\n📊 Numerical Result:")
        print(f"   Bitline Delay = {float(delay_val)*1e9:.3f} ns")
        print(f"   C++ DESTINY Result = {self.config.bitline_latency*1e9:.3f} ns")
        print(f"   Ratio (Symbolic/C++): {float(delay_val)/self.config.bitline_latency:.2f}")

        return bitline_delay_sym

    def compute_senseamp_delay(self):
        """Compute symbolic expression for sense amplifier delay"""
        print("\n" + "─" * 80)
        print("SENSE AMPLIFIER DELAY")
        print("─" * 80)

        # Sense amp delay depends on:
        # 1. Voltage swing detection time
        # 2. Amplification time

        # Typical sense amp: differential amplifier
        # Delay ~ C_load / I_amp

        # Load capacitance (output of sense amp)
        C_load = (self.bp.capIdealGate + self.bp.capOverlap + 3*self.bp.capFringe) * 4 * self.bp.featureSize

        # Amplifier current (NMOS width ~ 4F)
        I_amp = self.bp.currentOnNmos * 4

        # Sense amp delay
        senseamp_delay_sym = self.bp.vdd * C_load / I_amp

        print(f"\n📐 Symbolic Expression:")
        print(f"   C_load = {C_load}")
        print(f"\n   I_amp = {I_amp}")
        print(f"\n   Senseamp Delay = V × C / I = {simplify(senseamp_delay_sym)}")

        # Evaluate
        delay_val = senseamp_delay_sym.evalf(subs=self.bp.tech_values)

        print(f"\n📊 Numerical Result:")
        print(f"   Senseamp Delay = {float(delay_val)*1e12:.3f} ps")
        print(f"   C++ DESTINY Result = {self.config.senseamp_latency*1e12:.3f} ps")
        print(f"   Ratio (Symbolic/C++): {float(delay_val)/self.config.senseamp_latency:.2f}")

        return senseamp_delay_sym

    def compute_mux_delay(self):
        """Compute symbolic expression for multiplexer delay"""
        print("\n" + "─" * 80)
        print("MULTIPLEXER DELAY")
        print("─" * 80)

        print(f"\nConfiguration:")
        print(f"  Senseamp Mux: {self.config.senseamp_mux}")
        print(f"  Output L2 Mux: {self.config.output_mux_l2}")

        # Mux is pass transistor with load
        # Delay = R_pass * C_load

        # Pass transistor resistance
        R_pass = self.bp.effectiveResistanceMultiplier * self.bp.vdd / (self.bp.currentOnNmos * 2)

        # Load capacitance (next stage input + wire)
        C_load = (self.bp.capIdealGate + self.bp.capOverlap + 3*self.bp.capFringe) * 2 * self.bp.featureSize

        # Mux delay per level
        mux_delay_per_level = R_pass * C_load

        # Total mux levels
        total_mux_levels = (1 if self.config.senseamp_mux > 1 else 0) + \
                          (1 if self.config.output_mux_l2 > 1 else 0)

        mux_delay_sym = total_mux_levels * mux_delay_per_level

        print(f"\n📐 Symbolic Expression:")
        print(f"   R_pass = {R_pass}")
        print(f"\n   C_load = {C_load}")
        print(f"\n   Delay per level = {mux_delay_per_level}")
        print(f"\n   Total levels: {total_mux_levels}")
        print(f"\n   Total Mux Delay = {simplify(mux_delay_sym)}")

        # Evaluate
        delay_val = mux_delay_sym.evalf(subs=self.bp.tech_values)

        print(f"\n📊 Numerical Result:")
        print(f"   Mux Delay = {float(delay_val)*1e12:.3f} ps")
        print(f"   C++ DESTINY Result = {self.config.mux_latency*1e12:.3f} ps")
        print(f"   Ratio (Symbolic/C++): {float(delay_val)/self.config.mux_latency:.2f}")

        return mux_delay_sym

    def compute_total_access_time(self):
        """Compute total symbolic access time"""
        print("\n" + "=" * 80)
        print("TOTAL MEMORY ACCESS TIME")
        print("=" * 80)

        # Compute all components
        t_decoder = self.compute_row_decoder_delay()
        t_bitline = self.compute_bitline_delay()
        t_senseamp = self.compute_senseamp_delay()
        t_mux = self.compute_mux_delay()

        # Total access time (these happen sequentially)
        t_total_sym = t_decoder + t_bitline + t_senseamp + t_mux

        print("\n" + "=" * 80)
        print("SUMMARY: SYMBOLIC ACCESS TIME EXPRESSION")
        print("=" * 80)

        print(f"\n📐 Complete Symbolic Expression:")
        print(f"\n   t_access = t_decoder + t_bitline + t_senseamp + t_mux")
        print(f"\n   Simplified: {simplify(t_total_sym)}")

        # Evaluate
        total_val = t_total_sym.evalf(subs=self.bp.tech_values)

        print(f"\n📊 Numerical Results:")
        print(f"   Symbolic Total Access Time = {float(total_val)*1e9:.3f} ns")
        print(f"   C++ DESTINY Subarray Latency = {self.config.subarray_latency*1e9:.3f} ns")
        print(f"   Ratio (Symbolic/C++): {float(total_val)/self.config.subarray_latency:.2f}")

        # Sensitivity analysis
        print(f"\n🔍 Sensitivity Analysis:")
        from sympy import diff
        dt_dvdd = diff(t_total_sym, self.bp.vdd)
        dt_dIon = diff(t_total_sym, self.bp.currentOnNmos)

        print(f"\n   ∂t/∂V_dd (impact of voltage):")
        print(f"   {simplify(dt_dvdd)}")

        print(f"\n   ∂t/∂I_on (impact of drive current):")
        print(f"   {simplify(dt_dIon)}")

        # LaTeX output for papers
        print(f"\n📝 LaTeX Expression (for papers):")
        print(f"   {latex(simplify(t_total_sym))}")

        print("\n" + "=" * 80)

        return t_total_sym


def main():
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage: python symbolic_access_time.py <cpp_output_file> <config_file>")
        sys.exit(1)

    cpp_output_file = sys.argv[1]
    config_file = sys.argv[2]

    print("=" * 80)
    print("SYMBOLIC ACCESS TIME ANALYSIS")
    print("Using optimal configuration from C++ DESTINY DSE")
    print("=" * 80)

    # Parse C++ DESTINY output
    print(f"\nParsing C++ DESTINY output: {cpp_output_file}")
    opt_config = parse_cpp_destiny_output(cpp_output_file)

    # Read input parameters
    print(f"Reading configuration: {config_file}")
    input_param = InputParameter()
    input_param.ReadInputParameterFromFile(config_file)

    # Create analyzer
    analyzer = SymbolicAccessTimeAnalyzer(opt_config, input_param)

    # Compute symbolic access time
    symbolic_expr = analyzer.compute_total_access_time()

    print("\n✓ Symbolic analysis complete!")
    print("\nKey Insights:")
    print("  • Symbolic expressions show exact mathematical relationships")
    print("  • Can perform sensitivity analysis on any parameter")
    print("  • Enables technology scaling predictions")
    print("  • Useful for design space exploration and optimization")

    return 0


if __name__ == "__main__":
    sys.exit(main())
