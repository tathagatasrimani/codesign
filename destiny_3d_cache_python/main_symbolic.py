#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

"""
DESTINY - 3D DRAM/NVM Cache Simulator with Symbolic Modeling
Main entry point for the simulator with symbolic computation enabled
"""

import sys
import argparse
import globals as g
from InputParameter import InputParameter
from Technology import Technology
from MemCell import MemCell
from Wire import Wire
from Result import Result
from typedef import OptimizationTarget
from nvsim import nvsim, applyConstraint
from symbolic_wrapper import enable_symbolic_computation, disable_symbolic_computation
import sympy as sp


def initialize_globals():
    """Initialize global variables"""
    g.inputParameter = InputParameter()
    g.tech = Technology()
    g.devtech = Technology()
    g.cell = MemCell()
    g.gtech = Technology()
    g.localWire = Wire()
    g.globalWire = Wire()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='DESTINY - 3D DRAM/NVM Cache Simulator with Symbolic Modeling'
    )
    parser.add_argument(
        '-config',
        type=str,
        help='Configuration file path',
        required=True
    )
    parser.add_argument(
        '-output',
        type=str,
        help='Output file path',
        default='output_symbolic.txt'
    )
    parser.add_argument(
        '--symbolic',
        action='store_true',
        help='Enable symbolic computation tracking',
        default=True
    )
    parser.add_argument(
        '--compare-cpp',
        type=str,
        help='C++ output file to compare against',
        default=None
    )
    return parser.parse_args()


def extract_cpp_results(cpp_output_file):
    """
    Extract key metrics from C++ DESTINY output for comparison

    Returns dict with:
        - total_area
        - hit_latency
        - miss_latency
        - write_latency
        - leakage_power
        - hit_dynamic_energy
        - etc.
    """
    results = {}
    try:
        with open(cpp_output_file, 'r') as f:
            content = f.read()

        # Parse key metrics from C++ output
        import re

        # Extract total area
        match = re.search(r'Total Area = ([\d.]+)mm\^2', content)
        if match:
            results['total_area_mm2'] = float(match.group(1))

        # Extract hit latency
        match = re.search(r'Cache Hit Latency\s+=\s+([\d.]+)ns', content)
        if match:
            results['hit_latency_ns'] = float(match.group(1))

        # Extract miss latency
        match = re.search(r'Cache Miss Latency\s+=\s+([\d.]+)ns', content)
        if match:
            results['miss_latency_ns'] = float(match.group(1))

        # Extract write latency
        match = re.search(r'Cache Write Latency\s+=\s+([\d.]+)ns', content)
        if match:
            results['write_latency_ns'] = float(match.group(1))

        # Extract leakage power
        match = re.search(r'Cache Total Leakage Power\s+=\s+([\d.]+)mW', content)
        if match:
            results['leakage_power_mw'] = float(match.group(1))

        # Extract hit dynamic energy
        match = re.search(r'Cache Hit Dynamic Energy\s+=\s+([\d.]+)nJ', content)
        if match:
            results['hit_dynamic_energy_nj'] = float(match.group(1))

        print(f"\nExtracted C++ Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"Warning: Could not parse C++ output file: {e}")

    return results


def compare_results(python_results, cpp_results):
    """
    Compare Python symbolic results with C++ concrete results
    """
    print("\n" + "="*80)
    print("COMPARISON: Python (Symbolic) vs C++ (Concrete)")
    print("="*80)

    for key in cpp_results:
        if key in python_results:
            py_val = python_results[key]
            cpp_val = cpp_results[key]

            # Calculate relative difference
            if abs(cpp_val) > 1e-10:
                rel_diff = abs((py_val - cpp_val) / cpp_val) * 100
            else:
                rel_diff = 0 if abs(py_val) < 1e-10 else float('inf')

            status = "✓" if rel_diff < 5.0 else "✗"  # 5% tolerance

            print(f"\n{key}:")
            print(f"  Python:  {py_val}")
            print(f"  C++:     {cpp_val}")
            print(f"  Diff:    {rel_diff:.2f}% {status}")
        else:
            print(f"\n{key}: Missing in Python results")


def extract_python_results(bestDataResults, bestTagResults):
    """
    Extract key metrics from Python DESTINY results
    """
    results = {}

    # Get the write_edp_optimized result (index 4 based on OptimizationTarget enum)
    write_edp_idx = 4  # write_edp_optimized

    data_result = bestDataResults[write_edp_idx]
    tag_result = bestTagResults[write_edp_idx]

    # Extract metrics
    if hasattr(data_result, 'area') and hasattr(tag_result, 'area'):
        # Convert to mm^2
        total_area = (data_result.area + tag_result.area) * 1e-6  # um^2 to mm^2
        results['total_area_mm2'] = total_area

    if hasattr(data_result, 'readLatency'):
        # Assume readLatency is in seconds, convert to ns
        results['hit_latency_ns'] = data_result.readLatency * 1e9

    if hasattr(data_result, 'writeLatency'):
        results['write_latency_ns'] = data_result.writeLatency * 1e9

    if hasattr(data_result, 'leakage') and hasattr(tag_result, 'leakage'):
        # Convert to mW
        total_leakage = (data_result.leakage + tag_result.leakage) * 1e3  # W to mW
        results['leakage_power_mw'] = total_leakage

    if hasattr(data_result, 'readDynamicEnergy') and hasattr(tag_result, 'readDynamicEnergy'):
        # Convert to nJ
        total_read_energy = (data_result.readDynamicEnergy + tag_result.readDynamicEnergy) * 1e9  # J to nJ
        results['hit_dynamic_energy_nj'] = total_read_energy

    return results


def main():
    """Main simulation function with symbolic modeling"""
    print("=" * 80)
    print("DESTINY - 3D DRAM/NVM Cache Simulator (Symbolic Modeling)")
    print("=" * 80)

    # Parse arguments
    args = parse_arguments()

    # Enable symbolic computation if requested
    if args.symbolic:
        print("\n[SYMBOLIC MODE ENABLED]")
        print("Running parallel symbolic + concrete calculations...")
        enable_symbolic_computation()

    # Initialize global variables
    initialize_globals()

    # Read input parameters
    print(f"\nReading configuration from: {args.config}")
    g.inputParameter.ReadInputParameterFromFile(args.config)
    g.inputParameter.PrintInputParameter()

    # Initialize technology
    print("\nInitializing technology parameters...")
    g.tech.Initialize(
        g.inputParameter.processNode,
        g.inputParameter.deviceRoadmap,
        g.inputParameter
    )
    # Initialize device and global technology (same as tech)
    g.devtech.Initialize(
        g.inputParameter.processNode,
        g.inputParameter.deviceRoadmap,
        g.inputParameter
    )
    g.gtech.Initialize(
        g.inputParameter.processNode,
        g.inputParameter.deviceRoadmap,
        g.inputParameter
    )

    # Initialize memory cell
    print("Initializing memory cell...")
    if len(g.inputParameter.fileMemCell) > 0:
        # Use the cell file specified in the config
        cellFile = g.inputParameter.fileMemCell[0]
        # If it doesn't have a path, assume it's in the config directory
        if '/' not in cellFile:
            import os
            cellFile = os.path.join(os.path.dirname(args.config), cellFile)
        g.cell.ReadCellFromFile(cellFile)
    else:
        # Fall back to reading from the main config (shouldn't happen)
        g.cell.ReadCellFromFile(args.config)

    # Initialize wires - will be properly initialized in nvsim()
    print("Initializing wire models...")
    from typedef import WireType, WireRepeaterType
    g.localWire.Initialize(g.inputParameter.processNode, WireType.local_aggressive,
                           WireRepeaterType.repeated_none, g.inputParameter.temperature, False)
    g.globalWire.Initialize(g.inputParameter.processNode, WireType.global_aggressive,
                            WireRepeaterType.repeated_none, g.inputParameter.temperature, False)

    # DEBUG: Verify globals are initialized
    print(f"\nDEBUG: g.cell = {g.cell}, g.cell.memCellType = {g.cell.memCellType if g.cell else 'None'}")
    print(f"DEBUG: g.tech.initialized = {g.tech.initialized if g.tech else 'None'}")

    # Prepare result arrays for all optimization targets
    bestDataResults = [Result() for _ in range(int(OptimizationTarget.full_exploration))]
    bestTagResults = [Result() for _ in range(int(OptimizationTarget.full_exploration))]

    # Run full DESTINY simulation with design space exploration
    print("\nRunning DESTINY simulation with design space exploration...")
    print("This may take a while as it explores the full design space...")
    if args.symbolic:
        print("Tracking symbolic expressions alongside concrete values...")

    with open(args.output, 'w') as outputFile:
        outputFile.write("="*80 + "\n")
        outputFile.write("DESTINY SYMBOLIC MODELING OUTPUT\n")
        outputFile.write("="*80 + "\n\n")

        numSolution, returnCode = nvsim(outputFile, args.config, bestDataResults, bestTagResults)

    if returnCode != 0:
        print("\nSimulation failed!")
        if args.symbolic:
            disable_symbolic_computation()
        return returnCode

    print(f"\nSimulation complete! Found {numSolution} valid solutions.")

    # Extract Python results for comparison
    python_results = extract_python_results(bestDataResults, bestTagResults)

    # Print results for the primary optimization target
    if g.inputParameter.optimizationTarget == OptimizationTarget.full_exploration:
        # Print write_edp_optimized result (most relevant)
        write_edp_idx = 4
        print("\n" + "=" * 80)
        print("Results for write_edp_optimized:")
        print("=" * 80)
        if g.inputParameter.designTarget == 0:  # cache
            bestDataResults[write_edp_idx].printAsCache(bestTagResults[write_edp_idx],
                                                       g.inputParameter.cacheAccessMode)
        else:
            bestDataResults[write_edp_idx].print()
    else:
        # Print result for specific optimization target
        targetIdx = int(g.inputParameter.optimizationTarget)
        print("\n" + "=" * 80)
        print(f"Results for {g.inputParameter.optimizationTarget.name}:")
        print("=" * 80)
        if g.inputParameter.designTarget == 0:  # cache
            bestDataResults[targetIdx].printAsCache(bestTagResults[targetIdx],
                                                   g.inputParameter.cacheAccessMode)
        else:
            bestDataResults[targetIdx].print()

    # Compare with C++ if requested
    if args.compare_cpp:
        cpp_results = extract_cpp_results(args.compare_cpp)
        if cpp_results:
            compare_results(python_results, cpp_results)

    print(f"\nDetailed results written to: {args.output}")

    # Disable symbolic computation
    if args.symbolic:
        disable_symbolic_computation()

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
