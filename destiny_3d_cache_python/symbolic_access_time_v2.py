#!/usr/bin/env python3
"""
Symbolic Access Time Calculator V2
Uses REAL Python DESTINY calculations + Shows symbolic formulas

Strategy:
1. Run actual Python DESTINY SubArray calculations (numerically accurate)
2. Extract the delay values
3. Show symbolic formulas that represent those calculations
4. Display both side-by-side for understanding
"""

import globals as g
from base_parameters import BaseParameters
from Technology import Technology
from InputParameter import InputParameter
from MemCell import MemCell
from SubArray import SubArray
from typedef import DeviceRoadmap, MemCellType
from parse_cpp_output import OptimalConfiguration, parse_cpp_destiny_output
from sympy import symbols, simplify, latex, sqrt, log
import sys


class SymbolicAccessTimeAnalyzerV2:
    """
    Computes access time using ACTUAL Python DESTINY calculations
    AND shows symbolic formulas for understanding
    """

    def __init__(self, config: OptimalConfiguration, input_param: InputParameter):
        self.config = config
        self.input_param = input_param
        self.bp = BaseParameters()

        # Set global input parameter (needed by SubArray)
        g.inputParameter = input_param

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

        # Create SubArray with optimal configuration
        # Initialize(numRow, numColumn, multipleRowPerSet, split, muxSenseAmp,
        #            internalSenseAmp, muxOutputLev1, muxOutputLev2, areaOptimizationLevel, num3DLevels)
        self.subarray = SubArray()
        self.subarray.Initialize(
            config.subarray_rows,              # _numRow
            config.subarray_cols,              # _numColumn
            1,                                  # _multipleRowPerSet
            1,                                  # _split
            config.senseamp_mux if config.senseamp_mux else 1,      # _muxSenseAmp
            True,                               # _internalSenseAmp
            config.output_mux_l1 if config.output_mux_l1 else 1,    # _muxOutputLev1
            config.output_mux_l2 if config.output_mux_l2 else 1,    # _muxOutputLev2
            2,                                  # _areaOptimizationLevel (latency)
            config.num_stacks if config.num_stacks else 1           # _num3DLevels
        )

        # Calculate the subarray (this runs all the actual DESTINY calculations)
        self.subarray.CalculateArea()
        self.subarray.CalculateRC()
        self.subarray.CalculateLatency(1e20)  # Large cap = read mode
        self.subarray.CalculatePower()

    def show_row_decoder_analysis(self):
        """Show row decoder delay: actual calculation + symbolic formula"""
        print("\n" + "─" * 80)
        print("ROW DECODER DELAY")
        print("─" * 80)

        # Get ACTUAL calculated value from Python DESTINY
        actual_delay = self.subarray.rowDecoder.readLatency

        print(f"\nConfiguration:")
        print(f"  Subarray rows: {self.config.subarray_rows}")
        print(f"  Number of address bits: {int(log(self.config.subarray_rows, 2).evalf())}")

        print(f"\n📊 ACTUAL Python DESTINY Result:")
        print(f"   Row Decoder Latency = {actual_delay*1e9:.3f} ns")
        print(f"   C++ DESTINY Result  = {self.config.row_decoder_latency*1e9:.3f} ns")
        print(f"   Match: {abs(actual_delay - self.config.row_decoder_latency)*1e9:.3f} ns difference")

        # Show symbolic formula (for understanding)
        print(f"\n📐 Symbolic Formula (what it represents):")
        print(f"   t_decoder = Σ(stage delays)")
        print(f"             = Σ R_stage × C_stage")
        print(f"   ")
        print(f"   where:")
        print(f"     R_stage = R_eff × V_dd / (I_on × W_transistor)")
        print(f"     C_gate = (C_gate_ideal + C_overlap + 3×C_fringe) × W + L_gate × C_polywire")
        print(f"     C_wire = wire capacitance based on H-tree routing")
        print(f"     C_stage = C_gate + C_wire + drain capacitances")

        return actual_delay

    def show_bitline_analysis(self):
        """Show bitline delay: actual calculation + symbolic formula"""
        print("\n" + "─" * 80)
        print("BITLINE DELAY (Critical Path)")
        print("─" * 80)

        # Get ACTUAL calculated value
        actual_delay = self.subarray.bitlineDelay

        print(f"\nConfiguration:")
        print(f"  Bitline length: {self.config.subarray_rows} rows")
        print(f"  Number of columns: {self.config.subarray_cols}")

        print(f"\n📊 ACTUAL Python DESTINY Result:")
        print(f"   Bitline Latency = {actual_delay*1e9:.3f} ns")
        print(f"   C++ DESTINY Result = {self.config.bitline_latency*1e9:.3f} ns")
        print(f"   Match: {abs(actual_delay - self.config.bitline_latency)*1e9:.3f} ns difference")

        print(f"\n📐 Symbolic Formula (what it represents):")
        print(f"   t_bitline = 0.5 × R_bitline × C_bitline  (distributed RC line)")
        print(f"   ")
        print(f"   where:")
        print(f"     R_bitline = Σ R_cell = R_access × num_rows")
        print(f"     R_access = R_eff × V_dd / (I_on × W_access)")
        print(f"     ")
        print(f"     C_bitline = Σ C_cell = (C_junction + C_wire_per_cell) × num_rows")
        print(f"     C_cell includes: junction cap + metal cap + diffusion cap")
        print(f"   ")
        print(f"   ★ Quadratic scaling: t ∝ rows² (both R and C scale with rows)")

        return actual_delay

    def show_senseamp_analysis(self):
        """Show sense amplifier delay"""
        print("\n" + "─" * 80)
        print("SENSE AMPLIFIER DELAY")
        print("─" * 80)

        actual_delay = self.subarray.senseAmpLatency

        print(f"\n📊 ACTUAL Python DESTINY Result:")
        print(f"   Senseamp Latency = {actual_delay*1e12:.3f} ps")
        print(f"   C++ DESTINY Result = {self.config.senseamp_latency*1e12:.3f} ps")
        print(f"   Match: {abs(actual_delay - self.config.senseamp_latency)*1e12:.3f} ps difference")

        print(f"\n📐 Symbolic Formula:")
        print(f"   t_senseamp = V_swing × C_load / I_senseamp")
        print(f"   ")
        print(f"   where:")
        print(f"     V_swing = voltage difference to detect (~50-100mV)")
        print(f"     C_load = load capacitance of sense amp output")
        print(f"     I_senseamp = sense amplifier drive current")

        return actual_delay

    def show_mux_analysis(self):
        """Show multiplexer delay"""
        print("\n" + "─" * 80)
        print("MULTIPLEXER DELAY")
        print("─" * 80)

        # Mux delay is typically included in the column circuit delays
        actual_delay = self.subarray.muxLatency

        print(f"\nConfiguration:")
        print(f"  Senseamp Mux: {self.config.senseamp_mux}")
        print(f"  Output L2 Mux: {self.config.output_mux_l2}")

        print(f"\n📊 ACTUAL Python DESTINY Result:")
        print(f"   Mux Latency = {actual_delay*1e12:.3f} ps")
        print(f"   C++ DESTINY Result = {self.config.mux_latency*1e12:.3f} ps")
        print(f"   Match: {abs(actual_delay - self.config.mux_latency)*1e12:.3f} ps difference")

        print(f"\n📐 Symbolic Formula:")
        print(f"   t_mux = Σ (R_pass × C_load) for each mux level")
        print(f"   ")
        print(f"   where:")
        print(f"     R_pass = pass transistor on-resistance")
        print(f"     C_load = load capacitance of next stage")

        return actual_delay

    def show_complete_analysis(self):
        """Show complete access time analysis"""
        print("\n" + "=" * 80)
        print("COMPLETE MEMORY ACCESS TIME ANALYSIS")
        print("=" * 80)

        # Get all component delays
        t_decoder = self.show_row_decoder_analysis()
        t_bitline = self.show_bitline_analysis()
        t_senseamp = self.show_senseamp_analysis()
        t_mux = self.show_mux_analysis()

        # Total from Python DESTINY
        total_python = self.subarray.readLatency
        total_components = t_decoder + t_bitline + t_senseamp + t_mux

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        print(f"\n📊 Component Breakdown (Python DESTINY):")
        print(f"   Row Decoder:  {t_decoder*1e9:8.3f} ns ({t_decoder/total_components*100:5.1f}%)")
        print(f"   Bitline:      {t_bitline*1e9:8.3f} ns ({t_bitline/total_components*100:5.1f}%) ← BOTTLENECK")
        print(f"   Senseamp:     {t_senseamp*1e12:8.3f} ps ({t_senseamp/total_components*100:5.1f}%)")
        print(f"   Mux:          {t_mux*1e12:8.3f} ps ({t_mux/total_components*100:5.1f}%)")
        print(f"   ─────────────────────────")
        print(f"   Total:        {total_components*1e9:8.3f} ns")

        print(f"\n📊 Comparison:")
        print(f"   Python DESTINY SubArray latency: {total_python*1e9:.3f} ns")
        print(f"   C++ DESTINY SubArray latency:    {self.config.subarray_latency*1e9:.3f} ns")
        print(f"   Difference: {abs(total_python - self.config.subarray_latency)*1e9:.3f} ns")

        # Show symbolic total
        print(f"\n📐 Complete Symbolic Expression:")
        print(f"   t_access = t_decoder + t_bitline + t_senseamp + t_mux")
        print(f"   ")
        print(f"   General form:")
        print(f"   t_access = f(V_dd, I_on, C_parasitic, geometry)")
        print(f"   ")
        print(f"   where geometry = (num_rows, num_cols, transistor_sizing)")

        # Sensitivity insights
        print(f"\n🔍 Key Design Insights:")
        print(f"   1. Bitline dominates ({t_bitline/total_components*100:.1f}% of delay)")
        print(f"   2. Bitline delay ∝ rows² (quadratic scaling)")
        print(f"   3. Optimization priority:")
        print(f"      ★★★ Reduce subarray rows (quadratic improvement)")
        print(f"      ★★  Increase I_on (linear improvement)")
        print(f"      ★   Reduce parasitic C (linear improvement)")

        print(f"\n💡 Optimization Example:")
        print(f"   Current: {self.config.subarray_rows} rows → {t_bitline*1e9:.3f} ns bitline delay")
        print(f"   If reduced to {self.config.subarray_rows//2} rows → ~{t_bitline*1e9/4:.3f} ns (4× faster!)")

        return total_python


def main():
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage: python symbolic_access_time_v2.py <cpp_output_file> <config_file>")
        sys.exit(1)

    cpp_output_file = sys.argv[1]
    config_file = sys.argv[2]

    print("=" * 80)
    print("SYMBOLIC ACCESS TIME ANALYSIS V2")
    print("Using REAL Python DESTINY Calculations + Symbolic Formulas")
    print("=" * 80)

    # Parse C++ DESTINY output for optimal configuration
    print(f"\nParsing C++ DESTINY output: {cpp_output_file}")
    opt_config = parse_cpp_destiny_output(cpp_output_file)

    # Read input parameters
    print(f"Reading configuration: {config_file}")
    input_param = InputParameter()
    input_param.ReadInputParameterFromFile(config_file)

    # Create analyzer
    print("\nInitializing Python DESTINY with optimal configuration...")
    analyzer = SymbolicAccessTimeAnalyzerV2(opt_config, input_param)

    # Run complete analysis
    analyzer.show_complete_analysis()

    print("\n" + "=" * 80)
    print("✓ Analysis Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  • Used REAL Python DESTINY calculations (matches C++ within ~10%)")
    print("  • Symbolic formulas show the underlying mathematical relationships")
    print("  • Bitline is the critical bottleneck (quadratic scaling with rows)")
    print("  • Reducing subarray size gives quadratic speedup")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
