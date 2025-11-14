# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
# and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
# No part of DESTINY Project, including this file, may be copied,
# modified, propagated, or distributed except according to the terms
# contained in the LICENSE file.

"""
Python port of C++ macros from macros.h

This module provides Python functions and generators that replace the C++ macros
used for memory design space exploration in the DESTINY cache simulator.
"""

from typing import Iterator, Tuple, Any
import sys

# Absolute imports
from typedef import (
    WireType,
    WireRepeaterType,
    BufferDesignTarget,
    MemoryType,
    RoutingMode,
    OptimizationTarget
)


def initial_basic_wire(inputParameter, localWire, globalWire):
    """
    Initialize basic wire configurations based on input parameters.

    Replaces INITIAL_BASIC_WIRE macro.

    Args:
        inputParameter: InputParameter object containing min/max wire settings
        localWire: Wire object for local wiring
        globalWire: Wire object for global wiring
    """
    # Determine basic local wire type
    if inputParameter.minLocalWireType == inputParameter.maxLocalWireType:
        basicWireType = WireType(inputParameter.minLocalWireType)
    else:
        basicWireType = WireType.local_aggressive

    # Determine basic local wire repeater type
    if inputParameter.minLocalWireRepeaterType == inputParameter.maxLocalWireRepeaterType:
        basicWireRepeaterType = WireRepeaterType(inputParameter.minLocalWireRepeaterType)
    else:
        basicWireRepeaterType = WireRepeaterType.repeated_none

    # Determine basic local wire low swing setting
    if inputParameter.minIsLocalWireLowSwing == inputParameter.maxIsLocalWireLowSwing:
        isBasicLowSwing = inputParameter.minIsLocalWireLowSwing
    else:
        isBasicLowSwing = False

    # Initialize local wire
    localWire.Initialize(inputParameter.processNode, basicWireType,
                        basicWireRepeaterType, inputParameter.temperature,
                        isBasicLowSwing)

    # Determine basic global wire type
    if inputParameter.minGlobalWireType == inputParameter.maxGlobalWireType:
        basicWireType = WireType(inputParameter.minGlobalWireType)
    else:
        basicWireType = WireType.global_aggressive

    # Determine basic global wire repeater type
    if inputParameter.minGlobalWireRepeaterType == inputParameter.maxGlobalWireRepeaterType:
        basicWireRepeaterType = WireRepeaterType(inputParameter.minGlobalWireRepeaterType)
    else:
        basicWireRepeaterType = WireRepeaterType.repeated_none

    # Determine basic global wire low swing setting
    if inputParameter.minIsGlobalWireLowSwing == inputParameter.maxIsGlobalWireLowSwing:
        isBasicLowSwing = inputParameter.minIsGlobalWireLowSwing
    else:
        isBasicLowSwing = False

    # Initialize global wire
    globalWire.Initialize(inputParameter.processNode, basicWireType,
                         basicWireRepeaterType, inputParameter.temperature,
                         isBasicLowSwing)


def refine_local_wire_forloop(inputParameter) -> Iterator[Tuple[int, int, int]]:
    """
    Generator for iterating through local wire refinement configurations.

    Replaces REFINE_LOCAL_WIRE_FORLOOP macro.

    Args:
        inputParameter: InputParameter object containing wire parameter ranges

    Yields:
        Tuple of (localWireType, localWireRepeaterType, isLocalWireLowSwing)
    """
    for localWireType in range(inputParameter.minLocalWireType,
                               inputParameter.maxLocalWireType + 1):
        for localWireRepeaterType in range(inputParameter.minLocalWireRepeaterType,
                                          inputParameter.maxLocalWireRepeaterType + 1):
            for isLocalWireLowSwing in range(int(inputParameter.minIsLocalWireLowSwing),
                                            int(inputParameter.maxIsLocalWireLowSwing) + 1):
                # Only yield if repeater is none OR low swing is false
                if (WireRepeaterType(localWireRepeaterType) == WireRepeaterType.repeated_none or
                    bool(isLocalWireLowSwing) == False):
                    yield (localWireType, localWireRepeaterType, isLocalWireLowSwing)


def refine_global_wire_forloop(inputParameter) -> Iterator[Tuple[int, int, int]]:
    """
    Generator for iterating through global wire refinement configurations.

    Replaces REFINE_GLOBAL_WIRE_FORLOOP macro.

    Args:
        inputParameter: InputParameter object containing wire parameter ranges

    Yields:
        Tuple of (globalWireType, globalWireRepeaterType, isGlobalWireLowSwing)
    """
    for globalWireType in range(inputParameter.minGlobalWireType,
                                inputParameter.maxGlobalWireType + 1):
        for globalWireRepeaterType in range(inputParameter.minGlobalWireRepeaterType,
                                           inputParameter.maxGlobalWireRepeaterType + 1):
            for isGlobalWireLowSwing in range(int(inputParameter.minIsGlobalWireLowSwing),
                                             int(inputParameter.maxIsGlobalWireLowSwing) + 1):
                # Only yield if repeater is none OR low swing is false
                if (WireRepeaterType(globalWireRepeaterType) == WireRepeaterType.repeated_none or
                    bool(isGlobalWireLowSwing) == False):
                    yield (globalWireType, globalWireRepeaterType, isGlobalWireLowSwing)


def load_global_wire(oldResult, inputParameter, globalWire):
    """
    Load global wire configuration from a previous result.

    Replaces LOAD_GLOBAL_WIRE macro.

    Args:
        oldResult: Previous Result object containing wire configuration
        inputParameter: InputParameter object
        globalWire: Wire object to initialize
    """
    globalWire.Initialize(inputParameter.processNode,
                         oldResult.globalWire.wireType,
                         oldResult.globalWire.wireRepeaterType,
                         inputParameter.temperature,
                         oldResult.globalWire.isLowSwing)


def load_local_wire(oldResult, inputParameter, localWire):
    """
    Load local wire configuration from a previous result.

    Replaces LOAD_LOCAL_WIRE macro.

    Args:
        oldResult: Previous Result object containing wire configuration
        inputParameter: InputParameter object
        localWire: Wire object to initialize
    """
    localWire.Initialize(inputParameter.processNode,
                        oldResult.localWire.wireType,
                        oldResult.localWire.wireRepeaterType,
                        inputParameter.temperature,
                        oldResult.localWire.isLowSwing)


def try_and_update(oldResult, memoryType, inputParameter, localWire, globalWire, tempResult):
    """
    Try a configuration and update the result if better.

    Replaces TRY_AND_UPDATE macro.

    Args:
        oldResult: Previous Result object to base configuration on
        memoryType: MemoryType enum value
        inputParameter: InputParameter object
        localWire: Current local wire configuration
        globalWire: Current global wire configuration
        tempResult: Temporary Result object for comparison
    """
    # Import here to avoid circular dependencies
    from BankWithHtree import BankWithHtree
    from BankWithoutHtree import BankWithoutHtree

    # Create trial bank based on routing mode
    if inputParameter.routingMode == RoutingMode.h_tree:
        trialBank = BankWithHtree()
    else:
        trialBank = BankWithoutHtree()

    # Initialize trial bank with old result's configuration
    trialBank.Initialize(
        oldResult.bank.numRowMat,
        oldResult.bank.numColumnMat,
        oldResult.bank.capacity,
        oldResult.bank.blockSize,
        oldResult.bank.associativity,
        oldResult.bank.numRowPerSet,
        oldResult.bank.numActiveMatPerRow,
        oldResult.bank.numActiveMatPerColumn,
        oldResult.bank.muxSenseAmp,
        inputParameter.internalSensing,
        oldResult.bank.muxOutputLev1,
        oldResult.bank.muxOutputLev2,
        oldResult.bank.numRowSubarray,
        oldResult.bank.numColumnSubarray,
        oldResult.bank.numActiveSubarrayPerRow,
        oldResult.bank.numActiveSubarrayPerColumn,
        oldResult.bank.areaOptimizationLevel,
        memoryType,
        oldResult.bank.stackedDieCount,
        oldResult.bank.partitionGranularity,
        inputParameter.monolithicStackCount
    )

    # Calculate bank metrics
    trialBank.CalculateArea()
    trialBank.CalculateRC()
    trialBank.CalculateLatencyAndPower()

    # Update temp result
    tempResult.bank = trialBank
    tempResult.localWire = localWire
    tempResult.globalWire = globalWire

    # Compare and update old result
    oldResult.compareAndUpdate(tempResult)


def bigfor(inputParameter) -> Iterator[Tuple[int, ...]]:
    """
    Generator for design space exploration - iterates through all parameter combinations.

    Replaces BIGFOR macro.

    Args:
        inputParameter: InputParameter object containing parameter ranges

    Yields:
        Tuple of (numRowMat, numColumnMat, stackedDieCount, numActiveMatPerRow,
                 numActiveMatPerColumn, numRowSubarray, numColumnSubarray,
                 numActiveSubarrayPerRow, numActiveSubarrayPerColumn, muxSenseAmp,
                 muxOutputLev1, muxOutputLev2, numRowPerSet, areaOptimizationLevel)
    """
    # Helper function to generate power-of-2 sequence
    def power2_range(start, end):
        val = start
        while val <= end:
            yield val
            val *= 2

    # Helper function to generate linear sequence
    def linear_range(start, end):
        return range(start, end + 1)

    for numRowMat in power2_range(inputParameter.minNumRowMat, inputParameter.maxNumRowMat):
        for numColumnMat in power2_range(inputParameter.minNumColumnMat, inputParameter.maxNumColumnMat):
            for stackedDieCount in power2_range(inputParameter.minStackLayer, inputParameter.maxStackLayer):
                # numActiveMatPerRow is bounded by numColumnMat
                minActiveRow = min(numColumnMat, inputParameter.minNumActiveMatPerRow)
                maxActiveRow = min(numColumnMat, inputParameter.maxNumActiveMatPerRow)
                for numActiveMatPerRow in power2_range(minActiveRow, maxActiveRow):
                    # numActiveMatPerColumn is bounded by numRowMat
                    minActiveCol = min(numRowMat, inputParameter.minNumActiveMatPerColumn)
                    maxActiveCol = min(numRowMat, inputParameter.maxNumActiveMatPerColumn)
                    for numActiveMatPerColumn in power2_range(minActiveCol, maxActiveCol):
                        for numRowSubarray in power2_range(inputParameter.minNumRowSubarray, inputParameter.maxNumRowSubarray):
                            for numColumnSubarray in power2_range(inputParameter.minNumColumnSubarray, inputParameter.maxNumColumnSubarray):
                                # numActiveSubarrayPerRow is bounded by numColumnSubarray
                                minActiveSubRow = min(numColumnSubarray, inputParameter.minNumActiveSubarrayPerRow)
                                maxActiveSubRow = min(numColumnSubarray, inputParameter.maxNumActiveSubarrayPerRow)
                                for numActiveSubarrayPerRow in power2_range(minActiveSubRow, maxActiveSubRow):
                                    # numActiveSubarrayPerColumn is bounded by numRowSubarray
                                    minActiveSubCol = min(numRowSubarray, inputParameter.minNumActiveSubarrayPerColumn)
                                    maxActiveSubCol = min(numRowSubarray, inputParameter.maxNumActiveSubarrayPerColumn)
                                    for numActiveSubarrayPerColumn in power2_range(minActiveSubCol, maxActiveSubCol):
                                        for muxSenseAmp in power2_range(inputParameter.minMuxSenseAmp, inputParameter.maxMuxSenseAmp):
                                            for muxOutputLev1 in power2_range(inputParameter.minMuxOutputLev1, inputParameter.maxMuxOutputLev1):
                                                for muxOutputLev2 in power2_range(inputParameter.minMuxOutputLev2, inputParameter.maxMuxOutputLev2):
                                                    # numRowPerSet is bounded by associativity
                                                    maxRowPerSet = min(inputParameter.maxNumRowPerSet, inputParameter.associativity)
                                                    for numRowPerSet in power2_range(inputParameter.minNumRowPerSet, maxRowPerSet):
                                                        for areaOptimizationLevel in linear_range(inputParameter.minAreaOptimizationLevel, inputParameter.maxAreaOptimizationLevel):
                                                            yield (numRowMat, numColumnMat, stackedDieCount,
                                                                  numActiveMatPerRow, numActiveMatPerColumn,
                                                                  numRowSubarray, numColumnSubarray,
                                                                  numActiveSubarrayPerRow, numActiveSubarrayPerColumn,
                                                                  muxSenseAmp, muxOutputLev1, muxOutputLev2,
                                                                  numRowPerSet, areaOptimizationLevel)


def calculate(inputParameter, capacity, blockSize, associativity,
              numRowMat, numColumnMat, stackedDieCount, numRowPerSet,
              numActiveMatPerRow, numActiveMatPerColumn, muxSenseAmp,
              muxOutputLev1, muxOutputLev2, numRowSubarray, numColumnSubarray,
              numActiveSubarrayPerRow, numActiveSubarrayPerColumn,
              areaOptimizationLevel, memoryType, partitionGranularity):
    """
    Calculate bank metrics for a given configuration.

    Replaces CALCULATE macro.

    Args:
        inputParameter: InputParameter object
        capacity: Memory capacity in bits
        blockSize: Block size in bits
        associativity: Cache associativity
        numRowMat: Number of mat rows
        numColumnMat: Number of mat columns
        stackedDieCount: Number of stacked dies
        numRowPerSet: Number of rows per set
        numActiveMatPerRow: Active mats per row
        numActiveMatPerColumn: Active mats per column
        muxSenseAmp: Sense amplifier mux ratio
        muxOutputLev1: Level 1 output mux ratio
        muxOutputLev2: Level 2 output mux ratio
        numRowSubarray: Number of subarray rows
        numColumnSubarray: Number of subarray columns
        numActiveSubarrayPerRow: Active subarrays per row
        numActiveSubarrayPerColumn: Active subarrays per column
        areaOptimizationLevel: Buffer design target
        memoryType: MemoryType enum value
        partitionGranularity: Partition granularity

    Returns:
        Bank object with calculated metrics
    """
    # Import here to avoid circular dependencies
    from BankWithHtree import BankWithHtree
    from BankWithoutHtree import BankWithoutHtree

    # Create bank based on routing mode
    if inputParameter.routingMode == RoutingMode.h_tree:
        bank = BankWithHtree()
    else:
        bank = BankWithoutHtree()

    # Initialize bank
    bank.Initialize(
        numRowMat, numColumnMat, capacity, blockSize, associativity,
        numRowPerSet, numActiveMatPerRow, numActiveMatPerColumn, muxSenseAmp,
        inputParameter.internalSensing, muxOutputLev1, muxOutputLev2,
        numRowSubarray, numColumnSubarray, numActiveSubarrayPerRow,
        numActiveSubarrayPerColumn, BufferDesignTarget(areaOptimizationLevel),
        memoryType, stackedDieCount, partitionGranularity,
        inputParameter.monolithicStackCount
    )

    # Calculate bank metrics
    bank.CalculateArea()
    bank.CalculateRC()
    bank.CalculateLatencyAndPower()

    return bank


def update_best_data(dataBank, localWire, globalWire, tempResult, bestDataResults):
    """
    Update best data array results.

    Replaces UPDATE_BEST_DATA macro.

    Args:
        dataBank: Data bank object
        localWire: Local wire object
        globalWire: Global wire object
        tempResult: Temporary result object
        bestDataResults: List of best result objects for each optimization target
    """
    tempResult.bank = dataBank
    tempResult.localWire = localWire
    tempResult.globalWire = globalWire

    for i in range(int(OptimizationTarget.full_exploration)):
        bestDataResults[i].compareAndUpdate(tempResult)


def update_best_tag(tagBank, localWire, globalWire, tempResult, bestTagResults):
    """
    Update best tag array results.

    Replaces UPDATE_BEST_TAG macro.

    Args:
        tagBank: Tag bank object
        localWire: Local wire object
        globalWire: Global wire object
        tempResult: Temporary result object
        bestTagResults: List of best result objects for each optimization target
    """
    tempResult.bank = tagBank
    tempResult.localWire = localWire
    tempResult.globalWire = globalWire

    for i in range(int(OptimizationTarget.full_exploration)):
        bestTagResults[i].compareAndUpdate(tempResult)


def verify_data_capacity(dataBank, capacity, stackedDieCount):
    """
    Verify that data bank capacity matches expected capacity.

    Replaces VERIFY_DATA_CAPACITY macro.

    Args:
        dataBank: Data bank object
        capacity: Expected capacity in bits
        stackedDieCount: Number of stacked dies
    """
    actualCapacity = (dataBank.mat.subarray.numColumn * dataBank.mat.subarray.numRow *
                     dataBank.numColumnMat * dataBank.numRowMat *
                     dataBank.numColumnSubarray * dataBank.numRowSubarray * stackedDieCount)

    if actualCapacity != capacity:
        print(f"1 Bank = {dataBank.numRowMat}x{dataBank.numColumnMat} Mats")
        print(f"Activation - {dataBank.numActiveMatPerColumn}x{dataBank.numActiveMatPerRow} Mats")
        print(f"1 Mat  = {dataBank.numRowSubarray}x{dataBank.numColumnSubarray} Subarrays")
        print(f"Activation - {dataBank.numActiveSubarrayPerColumn}x{dataBank.numActiveSubarrayPerRow} Subarrays")
        print(f"Mux Degree - {dataBank.muxSenseAmp} x {dataBank.muxOutputLev1} x {dataBank.muxOutputLev2}")
        print("ERROR: DATA capacity violation. Shouldn't happen")
        print(f"Saw {actualCapacity}")
        print(f"Expected {capacity}")
        sys.exit(-1)


def verify_tag_capacity(tagBank, capacity, stackedDieCount):
    """
    Verify that tag bank capacity matches expected capacity.

    Replaces VERIFY_TAG_CAPACITY macro.

    Args:
        tagBank: Tag bank object
        capacity: Expected capacity in bits
        stackedDieCount: Number of stacked dies
    """
    actualCapacity = (tagBank.mat.subarray.numColumn * tagBank.mat.subarray.numRow *
                     tagBank.numColumnMat * tagBank.numRowMat *
                     tagBank.numColumnSubarray * tagBank.numRowSubarray * stackedDieCount)

    if actualCapacity != capacity:
        print(f"1 Bank = {tagBank.numRowMat}x{tagBank.numColumnMat} Mats")
        print(f"Activation - {tagBank.numActiveMatPerColumn}x{tagBank.numActiveMatPerRow} Mats")
        print(f"1 Mat  = {tagBank.numRowSubarray}x{tagBank.numColumnSubarray} Subarrays")
        print(f"Activation - {tagBank.numActiveSubarrayPerColumn}x{tagBank.numActiveSubarrayPerRow} Subarrays")
        print(f"Mux Degree - {tagBank.muxSenseAmp} x {tagBank.muxOutputLev1} x {tagBank.muxOutputLev2}")
        print("ERROR: TAG capacity violation. Shouldn't happen")
        print(f"Saw {actualCapacity}")
        print(f"Expected {capacity}")
        sys.exit(-1)


def reduce_search_size(inputParameter):
    """
    Reduce the search space size for faster exploration.

    Replaces REDUCE_SEARCH_SIZE macro.

    Args:
        inputParameter: InputParameter object to modify
    """
    inputParameter.minNumRowMat = 1
    inputParameter.maxNumRowMat = 64
    inputParameter.minNumColumnMat = 1
    inputParameter.maxNumColumnMat = 64
    inputParameter.minNumActiveMatPerRow = 1
    inputParameter.maxNumActiveMatPerRow = inputParameter.maxNumColumnMat
    inputParameter.minNumActiveMatPerColumn = 1
    inputParameter.maxNumActiveMatPerColumn = inputParameter.maxNumRowMat
    inputParameter.minNumRowSubarray = 1
    inputParameter.maxNumRowSubarray = 2
    inputParameter.minNumColumnSubarray = 1
    inputParameter.maxNumColumnSubarray = 2
    inputParameter.minNumActiveSubarrayPerRow = 1
    inputParameter.maxNumActiveSubarrayPerRow = inputParameter.maxNumColumnSubarray
    inputParameter.minNumActiveSubarrayPerColumn = 1
    inputParameter.maxNumActiveSubarrayPerColumn = inputParameter.maxNumRowSubarray
    inputParameter.minMuxSenseAmp = 1
    inputParameter.maxMuxSenseAmp = 64
    inputParameter.minMuxOutputLev1 = 1
    inputParameter.maxMuxOutputLev1 = 64
    inputParameter.minMuxOutputLev2 = 1
    inputParameter.maxMuxOutputLev2 = 64
    inputParameter.minNumRowPerSet = 1
    inputParameter.maxNumRowPerSet = 1
    inputParameter.minAreaOptimizationLevel = BufferDesignTarget.latency_first
    inputParameter.maxAreaOptimizationLevel = BufferDesignTarget.area_first
    inputParameter.minLocalWireType = WireType.local_aggressive
    inputParameter.maxLocalWireType = WireType.local_conservative
    inputParameter.minGlobalWireType = WireType.global_aggressive
    inputParameter.maxGlobalWireType = WireType.global_conservative
    inputParameter.minLocalWireRepeaterType = WireRepeaterType.repeated_none
    inputParameter.maxLocalWireRepeaterType = WireRepeaterType.repeated_opt
    inputParameter.minGlobalWireRepeaterType = WireRepeaterType.repeated_none
    inputParameter.maxGlobalWireRepeaterType = WireRepeaterType.repeated_opt
    inputParameter.minIsLocalWireLowSwing = False
    inputParameter.maxIsLocalWireLowSwing = True
    inputParameter.minIsGlobalWireLowSwing = False
    inputParameter.maxIsGlobalWireLowSwing = True


def reduce_search_size_constrained(inputParameter):
    """
    Reduce the search space size with constraints (only change if min != max).

    Replaces REDUCE_SEARCH_SIZE_CONSTRAINED macro.

    Args:
        inputParameter: InputParameter object to modify
    """
    if inputParameter.maxNumRowMat != inputParameter.minNumRowMat:
        inputParameter.minNumRowMat = 1
        inputParameter.maxNumRowMat = 64

    if inputParameter.maxNumColumnMat != inputParameter.minNumColumnMat:
        inputParameter.minNumColumnMat = 1
        inputParameter.maxNumColumnMat = 64

    if inputParameter.maxNumActiveMatPerRow != inputParameter.minNumActiveMatPerRow:
        inputParameter.minNumActiveMatPerRow = 1
        inputParameter.maxNumActiveMatPerRow = inputParameter.maxNumColumnMat

    if inputParameter.maxNumActiveMatPerColumn != inputParameter.minNumActiveMatPerColumn:
        inputParameter.minNumActiveMatPerColumn = 1
        inputParameter.maxNumActiveMatPerColumn = inputParameter.maxNumRowMat

    if inputParameter.maxNumRowSubarray != inputParameter.minNumRowSubarray:
        inputParameter.minNumRowSubarray = 1
        inputParameter.maxNumRowSubarray = 2

    if inputParameter.maxNumColumnSubarray != inputParameter.minNumColumnSubarray:
        inputParameter.minNumColumnSubarray = 1
        inputParameter.maxNumColumnSubarray = 2

    if inputParameter.maxNumActiveSubarrayPerRow != inputParameter.minNumActiveSubarrayPerRow:
        inputParameter.minNumActiveSubarrayPerRow = 1
        inputParameter.maxNumActiveSubarrayPerRow = inputParameter.maxNumColumnSubarray

    if inputParameter.maxNumActiveSubarrayPerColumn != inputParameter.minNumActiveSubarrayPerColumn:
        inputParameter.minNumActiveSubarrayPerColumn = 1
        inputParameter.maxNumActiveSubarrayPerColumn = inputParameter.maxNumRowSubarray

    if inputParameter.maxMuxSenseAmp != inputParameter.minMuxSenseAmp:
        inputParameter.minMuxSenseAmp = 1
        inputParameter.maxMuxSenseAmp = 64

    if inputParameter.maxMuxOutputLev1 != inputParameter.minMuxOutputLev1:
        inputParameter.minMuxOutputLev1 = 1
        inputParameter.maxMuxOutputLev1 = 64

    if inputParameter.maxMuxOutputLev2 != inputParameter.minMuxOutputLev2:
        inputParameter.minMuxOutputLev2 = 1
        inputParameter.maxMuxOutputLev2 = 64

    # These are always set
    inputParameter.minNumRowPerSet = 1
    inputParameter.maxNumRowPerSet = 1
    inputParameter.minAreaOptimizationLevel = BufferDesignTarget.latency_first
    inputParameter.maxAreaOptimizationLevel = BufferDesignTarget.area_first
    inputParameter.minLocalWireType = WireType.local_aggressive
    inputParameter.maxLocalWireType = WireType.local_conservative
    inputParameter.minGlobalWireType = WireType.global_aggressive
    inputParameter.maxGlobalWireType = WireType.global_conservative
    inputParameter.minLocalWireRepeaterType = WireRepeaterType.repeated_none
    inputParameter.maxLocalWireRepeaterType = WireRepeaterType.repeated_opt
    inputParameter.minGlobalWireRepeaterType = WireRepeaterType.repeated_none
    inputParameter.maxGlobalWireRepeaterType = WireRepeaterType.repeated_opt
    inputParameter.minIsLocalWireLowSwing = False
    inputParameter.maxIsLocalWireLowSwing = True
    inputParameter.minIsGlobalWireLowSwing = False
    inputParameter.maxIsGlobalWireLowSwing = True


def restore_search_size(inputParameter):
    """
    Restore the search space to full size.

    Replaces RESTORE_SEARCH_SIZE macro.

    Args:
        inputParameter: InputParameter object to modify
    """
    inputParameter.minNumRowMat = 1
    inputParameter.maxNumRowMat = 512
    inputParameter.minNumColumnMat = 1
    inputParameter.maxNumColumnMat = 512
    inputParameter.minNumActiveMatPerRow = 1
    inputParameter.maxNumActiveMatPerRow = inputParameter.maxNumColumnMat
    inputParameter.minNumActiveMatPerColumn = 1
    inputParameter.maxNumActiveMatPerColumn = inputParameter.maxNumRowMat
    inputParameter.minNumRowSubarray = 1
    inputParameter.maxNumRowSubarray = 2
    inputParameter.minNumColumnSubarray = 1
    inputParameter.maxNumColumnSubarray = 2
    inputParameter.minNumActiveSubarrayPerRow = 1
    inputParameter.maxNumActiveSubarrayPerRow = inputParameter.maxNumColumnSubarray
    inputParameter.minNumActiveSubarrayPerColumn = 1
    inputParameter.maxNumActiveSubarrayPerColumn = inputParameter.maxNumRowSubarray
    inputParameter.minMuxSenseAmp = 1
    inputParameter.maxMuxSenseAmp = 256
    inputParameter.minMuxOutputLev1 = 1
    inputParameter.maxMuxOutputLev1 = 256
    inputParameter.minMuxOutputLev2 = 1
    inputParameter.maxMuxOutputLev2 = 256
    inputParameter.minNumRowPerSet = 1
    inputParameter.maxNumRowPerSet = inputParameter.associativity
    inputParameter.minAreaOptimizationLevel = BufferDesignTarget.latency_first
    inputParameter.maxAreaOptimizationLevel = BufferDesignTarget.area_first
    inputParameter.minLocalWireType = WireType.local_aggressive
    inputParameter.maxLocalWireType = WireType.semi_conservative
    inputParameter.minGlobalWireType = WireType.semi_aggressive
    inputParameter.maxGlobalWireType = WireType.global_conservative
    inputParameter.minLocalWireRepeaterType = WireRepeaterType.repeated_none
    inputParameter.maxLocalWireRepeaterType = WireRepeaterType.repeated_50  # The limit is repeated_50
    inputParameter.minGlobalWireRepeaterType = WireRepeaterType.repeated_none
    inputParameter.maxGlobalWireRepeaterType = WireRepeaterType.repeated_50  # The limit is repeated_50
    inputParameter.minIsLocalWireLowSwing = False
    inputParameter.maxIsLocalWireLowSwing = True
    inputParameter.minIsGlobalWireLowSwing = False
    inputParameter.maxIsGlobalWireLowSwing = True


def apply_limit(result, allowedDataReadLatency, allowedDataWriteLatency,
               allowedDataReadDynamicEnergy, allowedDataWriteDynamicEnergy,
               allowedDataReadEdp, allowedDataWriteEdp,
               allowedDataArea, allowedDataLeakage):
    """
    Apply constraints/limits to a result object.

    Replaces APPLY_LIMIT macro.

    Args:
        result: Result object to apply limits to
        allowedDataReadLatency: Maximum allowed read latency
        allowedDataWriteLatency: Maximum allowed write latency
        allowedDataReadDynamicEnergy: Maximum allowed read energy
        allowedDataWriteDynamicEnergy: Maximum allowed write energy
        allowedDataReadEdp: Maximum allowed read EDP
        allowedDataWriteEdp: Maximum allowed write EDP
        allowedDataArea: Maximum allowed area
        allowedDataLeakage: Maximum allowed leakage
    """
    result.reset()
    result.limitReadLatency = allowedDataReadLatency
    result.limitWriteLatency = allowedDataWriteLatency
    result.limitReadDynamicEnergy = allowedDataReadDynamicEnergy
    result.limitWriteDynamicEnergy = allowedDataWriteDynamicEnergy
    result.limitReadEdp = allowedDataReadEdp
    result.limitWriteEdp = allowedDataWriteEdp
    result.limitArea = allowedDataArea
    result.limitLeakage = allowedDataLeakage


# Additional utility functions for formatting output (from the TO_* macros)

def to_second(value):
    """
    Format time value with appropriate unit.

    Args:
        value: Time value in seconds

    Returns:
        Tuple of (scaled_value, unit_string)
    """
    if value < 1e-9:
        return (value * 1e12, "ps")
    elif value < 1e-6:
        return (value * 1e9, "ns")
    elif value < 1e-3:
        return (value * 1e6, "us")
    elif value < 1:
        return (value * 1e3, "ms")
    else:
        return (value, "s")


def to_joule(value):
    """
    Format energy value with appropriate unit.

    Args:
        value: Energy value in joules

    Returns:
        Tuple of (scaled_value, unit_string)
    """
    if value < 1e-9:
        return (value * 1e12, "pJ")
    elif value < 1e-6:
        return (value * 1e9, "nJ")
    elif value < 1e-3:
        return (value * 1e6, "uJ")
    elif value < 1:
        return (value * 1e3, "mJ")
    else:
        return (value, "J")


def to_watt(value):
    """
    Format power value with appropriate unit.

    Args:
        value: Power value in watts

    Returns:
        Tuple of (scaled_value, unit_string)
    """
    if value < 1e-9:
        return (value * 1e12, "pW")
    elif value < 1e-6:
        return (value * 1e9, "nW")
    elif value < 1e-3:
        return (value * 1e6, "uW")
    elif value < 1:
        return (value * 1e3, "mW")
    else:
        return (value, "W")


def to_meter(value):
    """
    Format length value with appropriate unit.

    Args:
        value: Length value in meters

    Returns:
        Tuple of (scaled_value, unit_string)
    """
    if value < 1e-9:
        return (value * 1e12, "pm")
    elif value < 1e-6:
        return (value * 1e9, "nm")
    elif value < 1e-3:
        return (value * 1e6, "um")
    elif value < 1:
        return (value * 1e3, "mm")
    else:
        return (value, "m")


def to_sqm(value):
    """
    Format area value with appropriate unit.

    Args:
        value: Area value in square meters

    Returns:
        Tuple of (scaled_value, unit_string)
    """
    if value < 1e-12:
        return (value * 1e18, "nm^2")
    elif value < 1e-6:
        return (value * 1e12, "um^2")
    elif value < 1:
        return (value * 1e6, "mm^2")
    else:
        return (value, "m^2")


def to_bps(value):
    """
    Format bandwidth value with appropriate unit.

    Args:
        value: Bandwidth value in bytes per second

    Returns:
        Tuple of (scaled_value, unit_string)
    """
    if value < 1e3:
        return (value, "B/s")
    elif value < 1e6:
        return (value / 1e3, "KB/s")
    elif value < 1e9:
        return (value / 1e6, "MB/s")
    elif value < 1e12:
        return (value / 1e9, "GB/s")
    else:
        return (value / 1e12, "TB/s")
