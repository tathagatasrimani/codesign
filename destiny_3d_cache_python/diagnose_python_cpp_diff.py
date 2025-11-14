#!/usr/bin/env python3
"""
Diagnose why Python DESTINY differs from C++ DESTINY

This script runs BOTH Python and C++ DESTINY with identical configurations
and compares every intermediate value to find where they diverge.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import globals as g
from InputParameter import InputParameter
from Technology import Technology
from MemCell import MemCell
from SubArray import SubArray
from Wire import Wire
from typedef import WireType, WireRepeaterType, BufferDesignTarget
from parse_cpp_output import parse_cpp_destiny_output
import subprocess


def run_cpp_destiny(config_file):
    """Run C++ DESTINY and return output file"""
    cpp_output = "../destiny_3d_cache-master/cpp_diagnostic.txt"
    cpp_binary = "../destiny_3d_cache-master/destiny"

    print("Running C++ DESTINY...")
    cmd = f"cd ../destiny_3d_cache-master && ./destiny ../{config_file} -o cpp_diagnostic.txt"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"C++ DESTINY failed: {result.stderr}")
        return None

    return cpp_output


def compare_technology_params(py_tech, cpp_config):
    """Compare technology parameters"""
    print("\n" + "="*80)
    print("TECHNOLOGY PARAMETERS COMPARISON")
    print("="*80)

    print(f"\nProcess Node:")
    print(f"  Python: {g.inputParameter.processNode} nm")
    print(f"  C++:    65 nm (from config)")

    print(f"\nVoltage (V_dd):")
    print(f"  Python g.tech.vdd:    {g.tech.vdd:.4f} V")
    print(f"  Python g.devtech.vdd: {g.devtech.vdd:.4f} V")

    print(f"\nFeature Size:")
    print(f"  Python g.tech.featureSize:    {g.tech.featureSize*1e9:.2f} nm")
    print(f"  Python g.devtech.featureSize: {g.devtech.featureSize*1e9:.2f} nm")

    print(f"\nCurrent (I_on):")
    print(f"  Python g.tech.currentOnNmos[0]:    {g.tech.currentOnNmos[0]:.6e} A")
    print(f"  Python g.devtech.currentOnNmos[0]: {g.devtech.currentOnNmos[0]:.6e} A")

    print(f"\nWire Parameters:")
    print(f"  Python g.localWire.capWirePerUnit: {g.localWire.capWirePerUnit:.6e} F/m")
    print(f"  Python g.localWire.resWirePerUnit: {g.localWire.resWirePerUnit:.6e} Ω/m")


def compare_subarray_params(subarray, cpp_config):
    """Compare SubArray parameters before calculation"""
    print("\n" + "="*80)
    print("SUBARRAY PARAMETERS COMPARISON (Before CalculateLatency)")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  Rows: Python={subarray.numRow}, C++={cpp_config.subarray_rows}")
    print(f"  Cols: Python={subarray.numColumn}, C++={cpp_config.subarray_cols}")
    print(f"  Mux:  Python=senseamp:{subarray.muxSenseAmp} L1:{subarray.muxOutputLev1} L2:{subarray.muxOutputLev2}")
    print(f"        C++=senseamp:{cpp_config.senseamp_mux} L1:{cpp_config.output_mux_l1} L2:{cpp_config.output_mux_l2}")

    print(f"\nPhysical Dimensions:")
    print(f"  Python lenBitline: {subarray.lenBitline*1e6:.3f} μm")
    print(f"  Python lenWordline: {subarray.lenWordline*1e6:.3f} μm")

    print(f"\nBitline RC (CRITICAL!):")
    print(f"  Python resBitline: {subarray.resBitline:.6e} Ω")
    print(f"  Python capBitline: {subarray.capBitline:.6e} F")
    print(f"  Python R×C:        {subarray.resBitline * subarray.capBitline:.6e} s")

    print(f"\nCell Parameters:")
    print(f"  Python widthInFeatureSize:  {g.cell.widthInFeatureSize:.2f} F")
    print(f"  Python heightInFeatureSize: {g.cell.heightInFeatureSize:.2f} F")
    print(f"  Python memCellType: {g.cell.memCellType}")


def compare_delay_results(subarray, cpp_config):
    """Compare delay calculation results"""
    print("\n" + "="*80)
    print("DELAY RESULTS COMPARISON (After CalculateLatency)")
    print("="*80)

    print(f"\nComponent Delays:")
    print(f"  {'Component':<20} {'Python (ns)':<15} {'C++ (ns)':<15} {'Ratio':<10}")
    print(f"  {'-'*60}")

    py_decoder = subarray.rowDecoder.readLatency * 1e9
    cpp_decoder = cpp_config.row_decoder_latency * 1e9
    print(f"  {'Row Decoder':<20} {py_decoder:<15.6f} {cpp_decoder:<15.6f} {py_decoder/cpp_decoder if cpp_decoder>0 else 0:<10.3f}×")

    py_bitline = subarray.bitlineDelay * 1e9
    cpp_bitline = cpp_config.bitline_latency * 1e9
    print(f"  {'Bitline':<20} {py_bitline:<15.6f} {cpp_bitline:<15.6f} {py_bitline/cpp_bitline if cpp_bitline>0 else 0:<10.3f}×")

    py_sense = subarray.senseAmp.readLatency * 1e12
    cpp_sense = cpp_config.senseamp_latency * 1e12
    print(f"  {'Senseamp (ps)':<20} {py_sense:<15.6f} {cpp_sense:<15.6f} {py_sense/cpp_sense if cpp_sense>0 else 0:<10.3f}×")

    py_mux = (subarray.senseAmpMuxLev1.readLatency + subarray.senseAmpMuxLev2.readLatency) * 1e12
    cpp_mux = cpp_config.mux_latency * 1e12
    print(f"  {'Mux (ps)':<20} {py_mux:<15.6f} {cpp_mux:<15.6f} {py_mux/cpp_mux if cpp_mux>0 else 0:<10.3f}×")

    py_total = subarray.readLatency * 1e9
    cpp_total = cpp_config.subarray_latency * 1e9
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<20} {py_total:<15.6f} {cpp_total:<15.6f} {py_total/cpp_total if cpp_total>0 else 0:<10.3f}×")

    print(f"\nAnalysis:")
    if abs(py_sense - cpp_sense) < 0.1:
        print(f"  ✓ Senseamp matches exactly - good!")
    if abs(py_mux - cpp_mux) < 0.1:
        print(f"  ✓ Mux matches exactly - good!")
    if abs(py_bitline/cpp_bitline - 1.0) > 0.1:
        print(f"  ✗ Bitline differs by {abs(py_bitline/cpp_bitline - 1.0)*100:.1f}% - INVESTIGATE!")
    if abs(py_decoder/cpp_decoder - 1.0) > 0.1:
        print(f"  ✗ Row decoder differs by {abs(py_decoder/cpp_decoder - 1.0)*100:.1f}% - INVESTIGATE!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_python_cpp_diff.py <config_file>")
        print("\nExample:")
        print("  python diagnose_python_cpp_diff.py config/sample_SRAM_2layer.cfg")
        sys.exit(1)

    config_file = sys.argv[1]

    print("="*80)
    print("PYTHON vs C++ DESTINY DIAGNOSTIC")
    print("="*80)
    print(f"\nConfig: {config_file}")

    # Run C++ DESTINY first
    cpp_output_file = run_cpp_destiny(config_file)
    if not cpp_output_file:
        print("Failed to run C++ DESTINY")
        return 1

    # Parse C++ results
    print("\nParsing C++ DESTINY output...")
    cpp_config = parse_cpp_destiny_output(cpp_output_file)

    # Initialize Python DESTINY
    print("\nInitializing Python DESTINY...")
    g.inputParameter = InputParameter()
    g.inputParameter.ReadInputParameterFromFile(config_file)

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

    # Compare technology parameters
    compare_technology_params(g.tech, cpp_config)

    # Create SubArray with same config as C++
    subarray = SubArray()
    subarray.Initialize(
        cpp_config.subarray_rows,
        cpp_config.subarray_cols,
        1,  # multipleRowPerSet
        1,  # split
        cpp_config.senseamp_mux if cpp_config.senseamp_mux else 1,
        True,  # internalSenseAmp
        cpp_config.output_mux_l1 if cpp_config.output_mux_l1 else 1,
        cpp_config.output_mux_l2 if cpp_config.output_mux_l2 else 1,
        BufferDesignTarget.latency_first,
        cpp_config.num_stacks if cpp_config.num_stacks else 1
    )

    # Calculate area first
    subarray.CalculateArea()

    # Compare before latency calculation
    compare_subarray_params(subarray, cpp_config)

    # Calculate latency
    print("\nRunning Python DESTINY CalculateLatency...")
    subarray.CalculateLatency(1e20)
    subarray.CalculatePower()

    # Compare results
    compare_delay_results(subarray, cpp_config)

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Check if technology parameters match")
    print("  2. Check if wire parameters match")
    print("  3. Check if cell parameters match")
    print("  4. If all match but delays differ, check calculation formulas")

    return 0


if __name__ == "__main__":
    sys.exit(main())
