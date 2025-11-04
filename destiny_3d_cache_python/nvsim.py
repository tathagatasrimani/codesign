# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
# and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
# No part of DESTINY Project, including this file, may be copied,
# modified, propagated, or distributed except according to the terms
# contained in the LICENSE file.

"""
Python port of the nvsim() function from main.cpp

This module provides the main DESTINY simulation function with design space exploration
for cache and memory systems.
"""

import math
import sys

# Import required modules
from InputParameter import InputParameter
from Technology import Technology
from MemCell import MemCell
from Wire import Wire
from Result import Result
from BankWithHtree import BankWithHtree
from BankWithoutHtree import BankWithoutHtree
from typedef import (
    OptimizationTarget,
    MemoryType,
    RoutingMode,
    WireType,
    WireRepeaterType,
    BufferDesignTarget,
    MemCellType,
    DesignTarget,
    CacheAccessMode
)
from formula import is_pow2
from macros import (
    initial_basic_wire,
    reduce_search_size,
    restore_search_size,
    bigfor,
    calculate,
    verify_tag_capacity,
    verify_data_capacity,
    update_best_tag,
    update_best_data,
    refine_local_wire_forloop,
    refine_global_wire_forloop,
    load_global_wire,
    load_local_wire,
    try_and_update,
    apply_limit
)
import globals as g

# Constants
TOTAL_ADDRESS_BIT = 64  # Typical 64-bit address space


def nvsim(outputFile, inputFileName, bestDataResults, bestTagResults):
    """Main DESTINY simulation function with design space exploration

    Args:
        outputFile: Output file handle for CSV results (or None if not full_exploration)
        inputFileName: Input configuration file path
        bestDataResults: List of Result objects for data array (length = full_exploration)
        bestTagResults: List of Result objects for tag array (length = full_exploration)

    Returns:
        (numSolution, return_code) tuple where:
            numSolution: Number of valid solutions found
            return_code: 0 for success, 1 for failure
    """
    # Use global variables from globals module
    inputParameter = g.inputParameter
    tech = g.tech
    cell = g.cell
    localWire = g.localWire
    globalWire = g.globalWire

    # Apply constraints
    applyConstraint()

    # Initialize local variables
    numRowMat = 0
    numColumnMat = 0
    numActiveMatPerRow = 0
    numActiveMatPerColumn = 0
    numRowSubarray = 0
    numColumnSubarray = 0
    numActiveSubarrayPerRow = 0
    numActiveSubarrayPerColumn = 0
    muxSenseAmp = 0
    muxOutputLev1 = 0
    muxOutputLev2 = 0
    numRowPerSet = 0
    areaOptimizationLevel = 0
    localWireType = 0
    globalWireType = 0
    localWireRepeaterType = 0
    globalWireRepeaterType = 0
    isLocalWireLowSwing = 0
    isGlobalWireLowSwing = 0
    stackedDieCount = 0
    partitionGranularity = 0

    capacity = 0
    blockSize = 0
    associativity = 0

    numDesigns = 0
    numSolution = 0

    # Initialize bestDataResults optimization targets
    for i in range(int(OptimizationTarget.full_exploration)):
        bestDataResults[i].optimizationTarget = OptimizationTarget(i)
        bestDataResults[i].cellTech = cell

    # Initialize bestTagResults optimization targets
    tagBank = None
    for i in range(int(OptimizationTarget.full_exploration)):
        bestTagResults[i].optimizationTarget = OptimizationTarget(i)
        bestTagResults[i].cellTech = cell

    # Create wire objects
    localWire = Wire()
    globalWire = Wire()

    partitionGranularity = inputParameter.partitionGranularity

    # Search tag first if cache design
    if inputParameter.designTarget == DesignTarget.cache:
        # Need to design the tag array
        reduce_search_size(inputParameter)

        # Calculate the tag configuration
        numDataSet = inputParameter.capacity * 8 // inputParameter.wordWidth // inputParameter.associativity
        numIndexBit = int(math.log2(numDataSet) + 0.1)
        numOffsetBit = int(math.log2(inputParameter.wordWidth / 8) + 0.1)

        initial_basic_wire(inputParameter, localWire, globalWire)

        # Simulate tag
        for (numRowMat, numColumnMat, stackedDieCount, numActiveMatPerRow,
             numActiveMatPerColumn, numRowSubarray, numColumnSubarray,
             numActiveSubarrayPerRow, numActiveSubarrayPerColumn, muxSenseAmp,
             muxOutputLev1, muxOutputLev2, numRowPerSet, areaOptimizationLevel) in bigfor(inputParameter):

            tech.SetLayerCount(inputParameter, stackedDieCount)

            blockSize = TOTAL_ADDRESS_BIT - numIndexBit - numOffsetBit
            blockSize += 2  # add dirty bits and valid bits

            if blockSize // (numActiveMatPerRow * numActiveMatPerColumn *
                            numActiveSubarrayPerRow * numActiveSubarrayPerColumn) == 0:
                # Too aggressive partitioning
                continue

            if blockSize % (numActiveMatPerRow * numActiveMatPerColumn *
                           numActiveSubarrayPerRow * numActiveSubarrayPerColumn):
                blockSize = ((blockSize // (numActiveMatPerRow * numActiveMatPerColumn *
                                           numActiveSubarrayPerRow * numActiveSubarrayPerColumn) + 1) *
                            (numActiveMatPerRow * numActiveMatPerColumn *
                             numActiveSubarrayPerRow * numActiveSubarrayPerColumn))

            capacity = inputParameter.capacity * 8 // inputParameter.wordWidth * blockSize
            associativity = inputParameter.associativity

            # Calculate tag bank
            tagBank = calculate(
                inputParameter, capacity, blockSize, associativity,
                numRowMat, numColumnMat, stackedDieCount, numRowPerSet,
                numActiveMatPerRow, numActiveMatPerColumn, muxSenseAmp,
                muxOutputLev1, muxOutputLev2, numRowSubarray, numColumnSubarray,
                numActiveSubarrayPerRow, numActiveSubarrayPerColumn,
                areaOptimizationLevel, MemoryType.tag, partitionGranularity
            )

            numDesigns += 1

            if not tagBank.invalid:
                tempResult = Result()
                verify_tag_capacity(tagBank, capacity, stackedDieCount)
                numSolution += 1
                update_best_tag(tagBank, localWire, globalWire, tempResult, bestTagResults)

            del tagBank

        # Refine wire types if solutions found
        if numSolution > 0:
            trialBank = None
            tempResult = Result()

            # Refine local wire type
            for localWireType, localWireRepeaterType, isLocalWireLowSwing in refine_local_wire_forloop(inputParameter):
                localWire.Initialize(inputParameter.processNode, WireType(localWireType),
                                    WireRepeaterType(localWireRepeaterType), inputParameter.temperature,
                                    bool(isLocalWireLowSwing))
                for i in range(int(OptimizationTarget.full_exploration)):
                    load_global_wire(bestTagResults[i], inputParameter, globalWire)
                    try_and_update(bestTagResults[i], MemoryType.tag, inputParameter,
                                 localWire, globalWire, tempResult)

            # Refine global wire type
            for globalWireType, globalWireRepeaterType, isGlobalWireLowSwing in refine_global_wire_forloop(inputParameter):
                globalWire.Initialize(inputParameter.processNode, WireType(globalWireType),
                                     WireRepeaterType(globalWireRepeaterType), inputParameter.temperature,
                                     bool(isGlobalWireLowSwing))
                for i in range(int(OptimizationTarget.full_exploration)):
                    load_local_wire(bestTagResults[i], inputParameter, localWire)
                    try_and_update(bestTagResults[i], MemoryType.tag, inputParameter,
                                 localWire, globalWire, tempResult)

        # Check if valid tag solutions found
        if numSolution == 0:
            print("No valid solutions for tags.")
            print()
            print("Finished!")
            if localWire:
                del localWire
            if globalWire:
                del globalWire
            return (numSolution, 1)
        else:
            numSolution = 0
            numDesigns = 0
            restore_search_size(inputParameter)
            inputParameter.ReadInputParameterFromFile(inputFileName)  # Restore search space
            applyConstraint()

    # Adjust cache data array parameters according to the access mode
    capacity = inputParameter.capacity * 8
    blockSize = inputParameter.wordWidth
    associativity = inputParameter.associativity

    if inputParameter.designTarget == DesignTarget.cache:
        if inputParameter.cacheAccessMode == CacheAccessMode.sequential_access_mode:
            # Already knows which way to access
            associativity = 1
        elif inputParameter.cacheAccessMode == CacheAccessMode.fast_access_mode:
            # Load the entire set as a single word
            blockSize *= associativity
            associativity = 1
        else:  # Normal access mode
            # Normal access does not allow one set be distributed into multiple rows
            # otherwise, the row activation has to be delayed until the hit signals arrive.
            inputParameter.minNumRowPerSet = inputParameter.maxNumRowPerSet = 1

    # Adjust block size if it is SLC NAND flash or DRAM memory chip
    if inputParameter.designTarget == DesignTarget.RAM_chip and (
        cell.memCellType == MemCellType.SLCNAND or cell.memCellType == MemCellType.DRAM):
        blockSize = inputParameter.pageSize
        associativity = 1

    # Simulate data array
    initial_basic_wire(inputParameter, localWire, globalWire)

    for (numRowMat, numColumnMat, stackedDieCount, numActiveMatPerRow,
         numActiveMatPerColumn, numRowSubarray, numColumnSubarray,
         numActiveSubarrayPerRow, numActiveSubarrayPerColumn, muxSenseAmp,
         muxOutputLev1, muxOutputLev2, numRowPerSet, areaOptimizationLevel) in bigfor(inputParameter):

        if blockSize // (numActiveMatPerRow * numActiveMatPerColumn *
                        numActiveSubarrayPerRow * numActiveSubarrayPerColumn) == 0:
            # Too aggressive partitioning
            continue

        # Calculate data bank
        dataBank = calculate(
            inputParameter, capacity, blockSize, associativity,
            numRowMat, numColumnMat, stackedDieCount, numRowPerSet,
            numActiveMatPerRow, numActiveMatPerColumn, muxSenseAmp,
            muxOutputLev1, muxOutputLev2, numRowSubarray, numColumnSubarray,
            numActiveSubarrayPerRow, numActiveSubarrayPerColumn,
            areaOptimizationLevel, MemoryType.data, partitionGranularity
        )

        numDesigns += 1

        if not dataBank.invalid:
            tempResult = Result()
            verify_data_capacity(dataBank, capacity, stackedDieCount)
            numSolution += 1
            update_best_data(dataBank, localWire, globalWire, tempResult, bestDataResults)

            if (inputParameter.optimizationTarget == OptimizationTarget.full_exploration and
                not inputParameter.isPruningEnabled and outputFile is not None):
                # OUTPUT_TO_FILE macro - write to CSV
                output_to_file(outputFile, tempResult, inputParameter)

        del dataBank

    # Refine wire types if solutions found
    if numSolution > 0:
        trialBank = None
        tempResult = Result()

        # Refine local wire type
        for localWireType, localWireRepeaterType, isLocalWireLowSwing in refine_local_wire_forloop(inputParameter):
            localWire.Initialize(inputParameter.processNode, WireType(localWireType),
                                WireRepeaterType(localWireRepeaterType), inputParameter.temperature,
                                bool(isLocalWireLowSwing))
            for i in range(int(OptimizationTarget.full_exploration)):
                load_global_wire(bestDataResults[i], inputParameter, globalWire)
                try_and_update(bestDataResults[i], MemoryType.data, inputParameter,
                             localWire, globalWire, tempResult)

            if (inputParameter.optimizationTarget == OptimizationTarget.full_exploration and
                not inputParameter.isPruningEnabled and outputFile is not None):
                output_to_file(outputFile, tempResult, inputParameter)

        # Refine global wire type
        for globalWireType, globalWireRepeaterType, isGlobalWireLowSwing in refine_global_wire_forloop(inputParameter):
            globalWire.Initialize(inputParameter.processNode, WireType(globalWireType),
                                 WireRepeaterType(globalWireRepeaterType), inputParameter.temperature,
                                 bool(isGlobalWireLowSwing))
            for i in range(int(OptimizationTarget.full_exploration)):
                load_local_wire(bestDataResults[i], inputParameter, localWire)
                try_and_update(bestDataResults[i], MemoryType.data, inputParameter,
                             localWire, globalWire, tempResult)

            if (inputParameter.optimizationTarget == OptimizationTarget.full_exploration and
                not inputParameter.isPruningEnabled and outputFile is not None):
                output_to_file(outputFile, tempResult, inputParameter)

    # Handle pruning if enabled
    if (inputParameter.optimizationTarget == OptimizationTarget.full_exploration and
        inputParameter.isPruningEnabled):
        # Pruning is enabled
        # Create 4D pruning results array
        pruningResults = [[[[None for k in range(3)]
                           for j in range(int(OptimizationTarget.full_exploration))]
                          for i in range(int(OptimizationTarget.full_exploration))]]

        for i in range(int(OptimizationTarget.full_exploration)):
            pruningResults[i] = [[None for k in range(3)]
                                for j in range(int(OptimizationTarget.full_exploration))]
            for j in range(int(OptimizationTarget.full_exploration)):
                pruningResults[i][j] = [None for k in range(3)]
                for k in range(3):
                    pruningResults[i][j][k] = Result()

        # Assign the constraints
        for i in range(int(OptimizationTarget.full_exploration)):
            for j in range(int(OptimizationTarget.full_exploration)):
                for k in range(3):
                    pruningResults[i][j][k].optimizationTarget = OptimizationTarget(i)
                    pruningResults[i][j][k].localWire.__assign__(bestDataResults[i].localWire)
                    pruningResults[i][j][k].globalWire.__assign__(bestDataResults[i].globalWire)

                    # Set limits based on optimization target j
                    target = OptimizationTarget(j)
                    if target == OptimizationTarget.read_latency_optimized:
                        pruningResults[i][j][k].limitReadLatency = (
                            bestDataResults[j].bank.readLatency * (1 + (k + 1.0) / 10))
                    elif target == OptimizationTarget.write_latency_optimized:
                        pruningResults[i][j][k].limitWriteLatency = (
                            bestDataResults[j].bank.writeLatency * (1 + (k + 1.0) / 10))
                    elif target == OptimizationTarget.read_energy_optimized:
                        pruningResults[i][j][k].limitReadDynamicEnergy = (
                            bestDataResults[j].bank.readDynamicEnergy * (1 + (k + 1.0) / 10))
                    elif target == OptimizationTarget.write_energy_optimized:
                        pruningResults[i][j][k].limitWriteDynamicEnergy = (
                            bestDataResults[j].bank.writeDynamicEnergy * (1 + (k + 1.0) / 10))
                    elif target == OptimizationTarget.read_edp_optimized:
                        pruningResults[i][j][k].limitReadEdp = (
                            bestDataResults[j].bank.readLatency *
                            bestDataResults[j].bank.readDynamicEnergy * (1 + (k + 1.0) / 10))
                    elif target == OptimizationTarget.write_edp_optimized:
                        pruningResults[i][j][k].limitWriteEdp = (
                            bestDataResults[j].bank.writeLatency *
                            bestDataResults[j].bank.writeDynamicEnergy * (1 + (k + 1.0) / 10))
                    elif target == OptimizationTarget.area_optimized:
                        pruningResults[i][j][k].limitArea = (
                            bestDataResults[j].bank.area * (1 + (k + 1.0) / 10))
                    elif target == OptimizationTarget.leakage_optimized:
                        pruningResults[i][j][k].limitLeakage = (
                            bestDataResults[j].bank.leakage * (1 + (k + 1.0) / 10))

        # Output best results to CSV
        for i in range(int(OptimizationTarget.full_exploration)):
            if outputFile is not None:
                bestDataResults[i].printAsCacheToCsvFile(bestTagResults[i],
                                                        inputParameter.cacheAccessMode,
                                                        outputFile)

        print("Pruning done")
        # Run pruning here - TO-DO in future

        # Cleanup pruning results
        del pruningResults

    # If design constraint is applied
    if (inputParameter.optimizationTarget != OptimizationTarget.full_exploration and
        inputParameter.isConstraintApplied):

        # Calculate allowed metrics based on best results
        allowedDataReadLatency = (bestDataResults[int(OptimizationTarget.read_latency_optimized)].bank.readLatency *
                                 (inputParameter.readLatencyConstraint + 1))
        allowedDataWriteLatency = (bestDataResults[int(OptimizationTarget.write_latency_optimized)].bank.writeLatency *
                                  (inputParameter.writeLatencyConstraint + 1))
        allowedDataReadDynamicEnergy = (bestDataResults[int(OptimizationTarget.read_energy_optimized)].bank.readDynamicEnergy *
                                       (inputParameter.readDynamicEnergyConstraint + 1))
        allowedDataWriteDynamicEnergy = (bestDataResults[int(OptimizationTarget.write_energy_optimized)].bank.writeDynamicEnergy *
                                        (inputParameter.writeDynamicEnergyConstraint + 1))
        allowedDataLeakage = (bestDataResults[int(OptimizationTarget.leakage_optimized)].bank.leakage *
                             (inputParameter.leakageConstraint + 1))
        allowedDataArea = (bestDataResults[int(OptimizationTarget.area_optimized)].bank.area *
                          (inputParameter.areaConstraint + 1))
        allowedDataReadEdp = (bestDataResults[int(OptimizationTarget.read_edp_optimized)].bank.readLatency *
                             bestDataResults[int(OptimizationTarget.read_edp_optimized)].bank.readDynamicEnergy *
                             (inputParameter.readEdpConstraint + 1))
        allowedDataWriteEdp = (bestDataResults[int(OptimizationTarget.write_edp_optimized)].bank.writeLatency *
                              bestDataResults[int(OptimizationTarget.write_edp_optimized)].bank.writeDynamicEnergy *
                              (inputParameter.writeEdpConstraint + 1))

        # Apply limits to all results
        for i in range(int(OptimizationTarget.full_exploration)):
            apply_limit(bestDataResults[i], allowedDataReadLatency, allowedDataWriteLatency,
                       allowedDataReadDynamicEnergy, allowedDataWriteDynamicEnergy,
                       allowedDataReadEdp, allowedDataWriteEdp,
                       allowedDataArea, allowedDataLeakage)

        # Re-run simulation with constraints
        numSolution = 0
        initial_basic_wire(inputParameter, localWire, globalWire)

        for (numRowMat, numColumnMat, stackedDieCount, numActiveMatPerRow,
             numActiveMatPerColumn, numRowSubarray, numColumnSubarray,
             numActiveSubarrayPerRow, numActiveSubarrayPerColumn, muxSenseAmp,
             muxOutputLev1, muxOutputLev2, numRowPerSet, areaOptimizationLevel) in bigfor(inputParameter):

            if blockSize // (numActiveMatPerRow * numActiveMatPerColumn *
                            numActiveSubarrayPerRow * numActiveSubarrayPerColumn) == 0:
                # Too aggressive partitioning
                continue

            # Calculate data bank
            dataBank = calculate(
                inputParameter, capacity, blockSize, associativity,
                numRowMat, numColumnMat, stackedDieCount, numRowPerSet,
                numActiveMatPerRow, numActiveMatPerColumn, muxSenseAmp,
                muxOutputLev1, muxOutputLev2, numRowSubarray, numColumnSubarray,
                numActiveSubarrayPerRow, numActiveSubarrayPerColumn,
                areaOptimizationLevel, MemoryType.data, partitionGranularity
            )

            numDesigns += 1

            # Check constraints
            if (not dataBank.invalid and
                dataBank.readLatency <= allowedDataReadLatency and
                dataBank.writeLatency <= allowedDataWriteLatency and
                dataBank.readDynamicEnergy <= allowedDataReadDynamicEnergy and
                dataBank.writeDynamicEnergy <= allowedDataWriteDynamicEnergy and
                dataBank.leakage <= allowedDataLeakage and
                dataBank.area <= allowedDataArea and
                dataBank.readLatency * dataBank.readDynamicEnergy <= allowedDataReadEdp and
                dataBank.writeLatency * dataBank.writeDynamicEnergy <= allowedDataWriteEdp):

                tempResult = Result()
                verify_data_capacity(dataBank, capacity, stackedDieCount)
                numSolution += 1
                update_best_data(dataBank, localWire, globalWire, tempResult, bestDataResults)

            del dataBank

    print(f"numSolutions = {numSolution} / numDesigns = {numDesigns}")

    # Cleanup
    if localWire:
        del localWire
    if globalWire:
        del globalWire

    return (numSolution, 0)


def output_to_file(outputFile, result, inputParameter):
    """
    Output result to CSV file.

    This is a placeholder for the OUTPUT_TO_FILE macro functionality.
    The actual implementation would write CSV-formatted results.

    Args:
        outputFile: File handle to write to
        result: Result object to output
        inputParameter: InputParameter object
    """
    # TO-DO: Implement CSV output formatting
    # This would format the result as a CSV row and write to outputFile
    pass


def applyConstraint():
    """Apply design constraints and validate parameters"""
    inputParameter = g.inputParameter
    cell = g.cell

    # Check functions that are not yet implemented
    if inputParameter.designTarget == DesignTarget.CAM_chip:
        print("[ERROR] CAM model is still under development")
        sys.exit(-1)

    if cell.memCellType == MemCellType.DRAM:
        print("[ERROR] DRAM model is still under development")
        sys.exit(-1)

    if cell.memCellType == MemCellType.MLCNAND:
        print("[ERROR] MLC NAND flash model is still under development")
        sys.exit(-1)

    # Validate associativity for non-cache designs
    if inputParameter.designTarget != DesignTarget.cache and inputParameter.associativity > 1:
        print("[WARNING] Associativity setting is ignored for non-cache designs")
        inputParameter.associativity = 1

    # Validate associativity is power of 2
    if not is_pow2(inputParameter.associativity):
        print("[ERROR] The associativity value has to be a power of 2 in this version")
        # Note: Original C++ code commented out exit(-1)

    # Validate routing mode and sensing scheme compatibility
    if inputParameter.routingMode == RoutingMode.h_tree and inputParameter.internalSensing == False:
        print("[ERROR] H-tree does not support external sensing scheme in this version")
        sys.exit(-1)

    # Note: The following check was commented out in original C++ code
    # if inputParameter.globalWireRepeaterType != WireRepeaterType.repeated_none and inputParameter.internalSensing == False:
    #     print("[ERROR] Repeated global wire does not support external sensing scheme")
    #     sys.exit(-1)

    # TO-DO: more rules to add here
