#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

"""
DESTINY - 3D DRAM/NVM Cache Simulator
Main entry point for the simulator
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
        description='DESTINY - 3D DRAM/NVM Cache Simulator'
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
        default='output.txt'
    )
    return parser.parse_args()


def main():
    """Main simulation function"""
    print("=" * 80)
    print("DESTINY - 3D DRAM/NVM Cache Simulator")
    print("=" * 80)

    # Parse arguments
    args = parse_arguments()

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

    # Prepare result arrays for all optimization targets
    bestDataResults = [Result() for _ in range(int(OptimizationTarget.full_exploration))]
    bestTagResults = [Result() for _ in range(int(OptimizationTarget.full_exploration))]

    # Run full DESTINY simulation with design space exploration
    print("\nRunning DESTINY simulation with design space exploration...")
    print("This may take a while as it explores the full design space...")

    with open(args.output, 'w') as outputFile:
        numSolution, returnCode = nvsim(outputFile, args.config, bestDataResults, bestTagResults)

    if returnCode != 0:
        print("\nSimulation failed!")
        return returnCode

    print(f"\nSimulation complete! Found {numSolution} valid solutions.")

    # Print results for the primary optimization target
    if g.inputParameter.optimizationTarget == OptimizationTarget.full_exploration:
        # Print all optimization results
        print("\n" + "=" * 80)
        print("Results for all optimization targets:")
        print("=" * 80)
        for i in range(int(OptimizationTarget.full_exploration)):
            print(f"\n{OptimizationTarget(i).name}:")
            if g.inputParameter.designTarget == 0:  # cache
                bestDataResults[i].printAsCache(bestTagResults[i], g.inputParameter.cacheAccessMode)
            else:
                bestDataResults[i].print()
    else:
        # Print result for specific optimization target
        targetIdx = int(g.inputParameter.optimizationTarget)
        print("\n" + "=" * 80)
        print(f"Results for {g.inputParameter.optimizationTarget.name}:")
        print("=" * 80)
        if g.inputParameter.designTarget == 0:  # cache
            bestDataResults[targetIdx].printAsCache(bestTagResults[targetIdx], g.inputParameter.cacheAccessMode)
        else:
            bestDataResults[targetIdx].print()

    print(f"\nDetailed results written to: {args.output}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
