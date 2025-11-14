#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

import math
from FunctionUnit import FunctionUnit
from SubArray import SubArray
from PredecodeBlock import PredecodeBlock
from Comparator import Comparator
from TSV import TSV
from typedef import BufferDesignTarget, MemoryType, TSV_type
import globals as g


class Mat(FunctionUnit):
    """
    Mat class represents a matrix of subarrays with associated decoders and logic.
    Inherits from FunctionUnit to provide area, latency, and power calculations.
    """

    def __init__(self):
        """Constructor - initializes all properties"""
        super().__init__()

        # Initialization and validation flags
        self.initialized = False
        self.invalid = False

        # Configuration properties
        self.internalSenseAmp = False
        self.numRowSubarray = 0
        self.numColumnSubarray = 0
        self.numAddressBit = 0
        self.numDataBit = 0
        self.numWay = 0
        self.numRowPerSet = 0
        self.split = False
        self.numActiveSubarrayPerRow = 0
        self.numActiveSubarrayPerColumn = 0
        self.muxSenseAmp = 1
        self.muxOutputLev1 = 1
        self.muxOutputLev2 = 1
        self.areaOptimizationLevel = BufferDesignTarget.latency_first
        self.memoryType = MemoryType.data
        self.stackedDieCount = 1
        self.partitionGranularity = 0

        # Derived properties
        self.totalPredecoderOutputBits = 0
        self.predecoderLatency = 0.0
        self.areaAllLogicBlocks = 0.0

        # Sub-components
        self.subarray = SubArray()
        self.rowPredecoderBlock1 = PredecodeBlock()
        self.rowPredecoderBlock2 = PredecodeBlock()
        self.bitlineMuxPredecoderBlock1 = PredecodeBlock()
        self.bitlineMuxPredecoderBlock2 = PredecodeBlock()
        self.senseAmpMuxLev1PredecoderBlock1 = PredecodeBlock()
        self.senseAmpMuxLev1PredecoderBlock2 = PredecodeBlock()
        self.senseAmpMuxLev2PredecoderBlock1 = PredecodeBlock()
        self.senseAmpMuxLev2PredecoderBlock2 = PredecodeBlock()
        self.comparator = Comparator()
        self.tsvArray = TSV()

    def Initialize(self, _numRowSubarray, _numColumnSubarray, _numAddressBit, _numDataBit,
                   _numWay, _numRowPerSet, _split, _numActiveSubarrayPerRow, _numActiveSubarrayPerColumn,
                   _muxSenseAmp, _internalSenseAmp, _muxOutputLev1, _muxOutputLev2,
                   _areaOptimizationLevel, _memoryType, _stackedDieCount,
                   _partitionGranularity, monolithicStackCount):
        """
        Initialize the Mat with all configuration parameters.

        Args:
            _numRowSubarray: Number of subarray rows in a mat
            _numColumnSubarray: Number of subarray columns in a mat
            _numAddressBit: Number of mat address bits
            _numDataBit: Number of mat data bits
            _numWay: Number of cache ways distributed to this mat
            _numRowPerSet: For cache design, the number of wordlines which a set is partitioned into
            _split: Whether the row decoder is at the middle of subarrays
            _numActiveSubarrayPerRow: For different access types
            _numActiveSubarrayPerColumn: For different access types
            _muxSenseAmp: How many bitlines connect to one sense amplifier
            _internalSenseAmp: Whether sense amplifier is internal
            _muxOutputLev1: How many sense amplifiers connect to one output bit, level-1
            _muxOutputLev2: How many sense amplifiers connect to one output bit, level-2
            _areaOptimizationLevel: Buffer design target
            _memoryType: Type of memory (data, tag, CAM)
            _stackedDieCount: Number of stacked dies
            _partitionGranularity: Partition granularity
            monolithicStackCount: Count for monolithic 3D stacking
        """
        if self.initialized:
            print("[Mat] Warning: Already initialized!")

        # Store configuration parameters
        self.numRowSubarray = _numRowSubarray
        self.numColumnSubarray = _numColumnSubarray
        self.numAddressBit = _numAddressBit
        self.numDataBit = _numDataBit
        self.numWay = _numWay
        self.numRowPerSet = _numRowPerSet
        self.split = _split
        self.internalSenseAmp = _internalSenseAmp
        self.areaOptimizationLevel = _areaOptimizationLevel
        self.memoryType = _memoryType
        self.stackedDieCount = _stackedDieCount
        self.partitionGranularity = _partitionGranularity

        # Validate and adjust active subarray counts
        if _numActiveSubarrayPerRow > self.numColumnSubarray:
            print(f"[Mat] Warning: The number of active subarray per row is larger than the number of subarray per row!")
            print(f"{_numActiveSubarrayPerRow} > {self.numColumnSubarray}")
            self.numActiveSubarrayPerRow = self.numColumnSubarray
        else:
            self.numActiveSubarrayPerRow = _numActiveSubarrayPerRow

        if _numActiveSubarrayPerColumn > self.numRowSubarray:
            print(f"[Mat] Warning: The number of active subarray per column is larger than the number of subarray per column!")
            print(f"{_numActiveSubarrayPerColumn} > {self.numRowSubarray}")
            self.numActiveSubarrayPerColumn = self.numRowSubarray
        else:
            self.numActiveSubarrayPerColumn = _numActiveSubarrayPerColumn

        self.muxSenseAmp = _muxSenseAmp
        self.muxOutputLev1 = _muxOutputLev1
        self.muxOutputLev2 = _muxOutputLev2

        # Calculate number of rows and columns in a subarray
        numRow = 0
        numColumn = 0

        # The number of address bits that are used to power gate inactive subarrays
        numAddressForGating = int(math.log2(self.numRowSubarray * self.numColumnSubarray /
                                            self.numActiveSubarrayPerColumn / self.numActiveSubarrayPerRow) + 0.1)
        _numAddressBit -= numAddressForGating  # Only use the effective address bits in the following calculation

        if _numAddressBit <= 0:
            # Too aggressive partitioning
            self.invalid = True
            self.initialized = True
            return

        # Determine the number of rows in a subarray
        numRow = 1 << _numAddressBit
        if self.memoryType == MemoryType.data:
            numRow *= self.numWay  # Only for cache design that partitions a set into multiple rows
        numRow //= (self.muxSenseAmp * self.muxOutputLev1 * self.muxOutputLev2)  # Distribute to column decoding

        if numRow == 0:
            self.invalid = True
            self.initialized = True
            return

        numColumn = self.numDataBit // (self.numActiveSubarrayPerRow * self.numActiveSubarrayPerColumn)

        if numColumn == 0:
            self.invalid = True
            self.initialized = True
            return

        numColumn *= self.muxSenseAmp * self.muxOutputLev1 * self.muxOutputLev2
        if self.memoryType == MemoryType.tag:
            numColumn *= self.numWay

        # Initialize subarray
        self.subarray.Initialize(numRow, numColumn, self.numRowPerSet > 1, True,  # TO-DO: need to correct
                                self.muxSenseAmp, self.internalSenseAmp, self.muxOutputLev1, self.muxOutputLev2,
                                self.areaOptimizationLevel, monolithicStackCount)

        if self.subarray.invalid:
            self.invalid = True
            self.initialized = True
            return

        # Calculate subarray area during initialization (needed for size dimension calculations)
        self.subarray.CalculateArea()

        # Calculate row predecoder address bits
        numAddressRowPredecoderBlock1 = _numAddressBit - int(math.log2(self.muxSenseAmp * self.muxOutputLev1 * self.muxOutputLev2) + 0.1)

        if numAddressRowPredecoderBlock1 < 0:
            self.invalid = True
            self.initialized = True
            return

        numAddressRowPredecoderBlock2 = 0
        if numAddressRowPredecoderBlock1 > 3:  # Block 2 is needed
            numAddressRowPredecoderBlock2 = numAddressRowPredecoderBlock1 // 2
            numAddressRowPredecoderBlock1 = numAddressRowPredecoderBlock1 - numAddressRowPredecoderBlock2

        self.totalPredecoderOutputBits = 1 << numAddressRowPredecoderBlock1
        self.totalPredecoderOutputBits += 1 << numAddressRowPredecoderBlock2

        # Calculate capacitive load for row predecoder
        capLoadRowPredecoder = (self.subarray.height * g.localWire.capWirePerUnit * self.numRowSubarray / 2 +
                               self.subarray.width * g.localWire.capWirePerUnit * self.numColumnSubarray / 2)

        self.rowPredecoderBlock1.Initialize(numAddressRowPredecoderBlock1, capLoadRowPredecoder, 0)  # TO-DO
        self.rowPredecoderBlock2.Initialize(numAddressRowPredecoderBlock2, capLoadRowPredecoder, 0)  # TO-DO

        # Calculate capacitive load for mux predecoder
        capLoadMuxPredecoder = (max(0, self.subarray.height * g.localWire.capWirePerUnit * (self.numRowSubarray - 2) / 2) +
                               max(0, self.subarray.width * g.localWire.capWirePerUnit * (self.numColumnSubarray - 2) / 2))

        # Bitline mux predecoder
        numAddressBitlineMuxPredecoderBlock1 = int(math.log2(self.muxSenseAmp) + 0.1)
        numAddressBitlineMuxPredecoderBlock2 = 0
        if numAddressBitlineMuxPredecoderBlock1 > 3:  # Block 2 is needed
            numAddressBitlineMuxPredecoderBlock2 = numAddressBitlineMuxPredecoderBlock1 // 2
            numAddressBitlineMuxPredecoderBlock1 = numAddressBitlineMuxPredecoderBlock1 - numAddressBitlineMuxPredecoderBlock2

        self.bitlineMuxPredecoderBlock1.Initialize(numAddressBitlineMuxPredecoderBlock1, capLoadMuxPredecoder, 0)  # TO-DO
        self.bitlineMuxPredecoderBlock2.Initialize(numAddressBitlineMuxPredecoderBlock2, capLoadMuxPredecoder, 0)  # TO-DO

        self.totalPredecoderOutputBits += 1 << numAddressBitlineMuxPredecoderBlock1
        self.totalPredecoderOutputBits += 1 << numAddressBitlineMuxPredecoderBlock2

        # Sense amp mux level 1 predecoder
        numAddressSenseAmpMuxLev1PredecoderBlock1 = int(math.log2(self.muxOutputLev1) + 0.1)
        numAddressSenseAmpMuxLev1PredecoderBlock2 = 0
        if numAddressSenseAmpMuxLev1PredecoderBlock1 > 3:  # Block 2 is needed
            numAddressSenseAmpMuxLev1PredecoderBlock2 = numAddressSenseAmpMuxLev1PredecoderBlock1 // 2
            numAddressSenseAmpMuxLev1PredecoderBlock1 = numAddressSenseAmpMuxLev1PredecoderBlock1 - numAddressSenseAmpMuxLev1PredecoderBlock2

        self.senseAmpMuxLev1PredecoderBlock1.Initialize(numAddressSenseAmpMuxLev1PredecoderBlock1, capLoadMuxPredecoder, 0)  # TO-DO
        self.senseAmpMuxLev1PredecoderBlock2.Initialize(numAddressSenseAmpMuxLev1PredecoderBlock2, capLoadMuxPredecoder, 0)  # TO-DO

        self.totalPredecoderOutputBits += 1 << numAddressSenseAmpMuxLev1PredecoderBlock1
        self.totalPredecoderOutputBits += 1 << numAddressSenseAmpMuxLev1PredecoderBlock2

        # Sense amp mux level 2 predecoder
        numAddressSenseAmpMuxLev2PredecoderBlock1 = int(math.log2(self.muxOutputLev2) + 0.1)
        numAddressSenseAmpMuxLev2PredecoderBlock2 = 0
        if numAddressSenseAmpMuxLev2PredecoderBlock1 > 3:  # Block 2 is needed
            numAddressSenseAmpMuxLev2PredecoderBlock2 = numAddressSenseAmpMuxLev2PredecoderBlock1 // 2
            numAddressSenseAmpMuxLev2PredecoderBlock1 = numAddressSenseAmpMuxLev2PredecoderBlock1 - numAddressSenseAmpMuxLev2PredecoderBlock2

        self.senseAmpMuxLev2PredecoderBlock1.Initialize(numAddressSenseAmpMuxLev2PredecoderBlock1, capLoadMuxPredecoder, 0)  # TO-DO
        self.senseAmpMuxLev2PredecoderBlock2.Initialize(numAddressSenseAmpMuxLev2PredecoderBlock2, capLoadMuxPredecoder, 0)  # TO-DO

        self.totalPredecoderOutputBits += 1 << numAddressSenseAmpMuxLev2PredecoderBlock1
        self.totalPredecoderOutputBits += 1 << numAddressSenseAmpMuxLev2PredecoderBlock2

        # Initialize comparator for tag memory with internal sense amplifier
        if self.memoryType == MemoryType.tag and self.internalSenseAmp:
            self.comparator.Initialize(self.numDataBit, 0)  # TO-DO: need to fix

        # Initialize TSV connections
        if self.stackedDieCount > 1 and self.partitionGranularity != 0:
            tsv_type = g.tech.WireTypeToTSVType(g.inputParameter.maxLocalWireType)
            self.tsvArray.Initialize(tsv_type)

        self.initialized = True

    def CalculateArea(self):
        """Calculate the area of the Mat including all sub-components"""
        if not self.initialized:
            print("[Mat] Error: Require initialization first!")
        elif self.invalid:
            self.height = self.width = self.area = g.invalid_value
        else:
            # Subarray CalculateArea() is already called during initialization
            self.rowPredecoderBlock1.CalculateArea()
            self.rowPredecoderBlock2.CalculateArea()
            self.bitlineMuxPredecoderBlock1.CalculateArea()
            self.bitlineMuxPredecoderBlock2.CalculateArea()
            self.senseAmpMuxLev1PredecoderBlock1.CalculateArea()
            self.senseAmpMuxLev1PredecoderBlock2.CalculateArea()
            self.senseAmpMuxLev2PredecoderBlock1.CalculateArea()
            self.senseAmpMuxLev2PredecoderBlock2.CalculateArea()

            # Calculate total area of all predecoder blocks
            areaAllPredecoderBlocks = (self.rowPredecoderBlock1.area + self.rowPredecoderBlock2.area +
                                      self.bitlineMuxPredecoderBlock1.area + self.bitlineMuxPredecoderBlock2.area +
                                      self.senseAmpMuxLev1PredecoderBlock1.area + self.senseAmpMuxLev1PredecoderBlock2.area +
                                      self.senseAmpMuxLev2PredecoderBlock1.area + self.senseAmpMuxLev2PredecoderBlock2.area)

            self.width = self.subarray.width * self.numColumnSubarray
            self.height = self.subarray.height * self.numRowSubarray

            self.areaAllLogicBlocks = areaAllPredecoderBlocks

            # For any partition granularity besides coarse grained, predecoders go on logic layer
            if self.stackedDieCount > 1 and self.partitionGranularity == 1:
                # Add TSV area for predecoders
                self.tsvArray.CalculateArea()
                redundancyFactor = g.inputParameter.tsvRedundancy
                areaTSV = self.tsvArray.area * self.totalPredecoderOutputBits * redundancyFactor
                self.tsvArray.numTotalBits = int(self.totalPredecoderOutputBits * redundancyFactor + 0.1)
                self.tsvArray.numAccessBits = self.tsvArray.numTotalBits

                # Area of logic layer is computed during result output
                if self.width > self.height:
                    self.width += math.sqrt(areaTSV)
                else:
                    self.height += math.sqrt(areaTSV)
            else:
                # Add the predecoders' area
                if self.width > self.height:
                    self.width += math.sqrt(areaAllPredecoderBlocks)  # We don't want to have too much white space here
                else:
                    self.height += math.sqrt(areaAllPredecoderBlocks)

            # Add comparator area for tag memory
            if self.memoryType == MemoryType.tag and self.internalSenseAmp:
                self.comparator.CalculateArea()
                self.areaAllLogicBlocks += self.comparator.area
                # TSVs for comparator are added above in previous conditional
                if self.stackedDieCount <= 1 or self.partitionGranularity != 1:
                    self.height += self.numWay * self.comparator.area / self.width

            self.area = self.height * self.width

    def CalculateRC(self):
        """Calculate resistance and capacitance for all sub-components"""
        if not self.initialized:
            print("[Mat] Error: Require initialization first!")
        elif not self.invalid:
            # Subarray does not have CalculateRC() function, since it is integrated as a part of initialization
            self.rowPredecoderBlock1.CalculateRC()
            self.rowPredecoderBlock2.CalculateRC()
            self.bitlineMuxPredecoderBlock1.CalculateRC()
            self.bitlineMuxPredecoderBlock2.CalculateRC()
            self.senseAmpMuxLev1PredecoderBlock1.CalculateRC()
            self.senseAmpMuxLev1PredecoderBlock2.CalculateRC()
            self.senseAmpMuxLev2PredecoderBlock1.CalculateRC()
            self.senseAmpMuxLev2PredecoderBlock2.CalculateRC()

            if self.memoryType == MemoryType.tag and self.internalSenseAmp:
                self.comparator.CalculateRC()

    def CalculateLatency(self, _rampInput):
        """
        Calculate latency for read and write operations.

        Args:
            _rampInput: Input ramp time
        """
        if not self.initialized:
            print("[Mat] Error: Require initialization first!")
        elif self.invalid:
            self.readLatency = self.writeLatency = g.invalid_value
        else:
            # Calculate the predecoder blocks latency
            self.rowPredecoderBlock1.CalculateLatency(_rampInput)
            self.rowPredecoderBlock2.CalculateLatency(_rampInput)
            self.bitlineMuxPredecoderBlock1.CalculateLatency(_rampInput)
            self.bitlineMuxPredecoderBlock2.CalculateLatency(_rampInput)
            self.senseAmpMuxLev1PredecoderBlock1.CalculateLatency(_rampInput)
            self.senseAmpMuxLev1PredecoderBlock2.CalculateLatency(_rampInput)
            self.senseAmpMuxLev2PredecoderBlock1.CalculateLatency(_rampInput)
            self.senseAmpMuxLev2PredecoderBlock2.CalculateLatency(_rampInput)

            # Calculate maximum latency across all predecoder types
            rowPredecoderLatency = max(self.rowPredecoderBlock1.readLatency, self.rowPredecoderBlock2.readLatency)
            bitlineMuxPredecoderLatency = max(self.bitlineMuxPredecoderBlock1.readLatency,
                                             self.bitlineMuxPredecoderBlock2.readLatency)
            senseAmpMuxLev1PredecoderLatency = max(self.senseAmpMuxLev1PredecoderBlock1.readLatency,
                                                   self.senseAmpMuxLev1PredecoderBlock2.readLatency)
            senseAmpMuxLev2PredecoderLatency = max(self.senseAmpMuxLev2PredecoderBlock1.readLatency,
                                                   self.senseAmpMuxLev2PredecoderBlock2.readLatency)
            self.predecoderLatency = max(max(rowPredecoderLatency, bitlineMuxPredecoderLatency),
                                        max(senseAmpMuxLev1PredecoderLatency, senseAmpMuxLev2PredecoderLatency))

            # Add TSV latency for 3D stacking
            if self.stackedDieCount > 1 and self.partitionGranularity != 0:
                # Add TSV latency here -- Once for address, once for data
                tsvReadRampInput = 1e20  # Normally senseAmpMuxLev2 is the last driver from Mat
                tsvWriteRampInput = g.infinite_ramp  # Write TSVs should be driven by predecoders

                # Add TSV energy ~ Assume outside of bank area
                self.tsvArray.CalculateLatencyAndPower(tsvReadRampInput, tsvWriteRampInput)

                # Address TSV latency
                self.predecoderLatency += (self.stackedDieCount - 1) * self.tsvArray.writeLatency

            # Calculate subarray latency
            self.subarray.CalculateLatency(min(self.rowPredecoderBlock1.rampOutput, self.rowPredecoderBlock2.rampOutput))

            # Add them together
            self.readLatency = self.predecoderLatency + self.subarray.readLatency
            self.writeLatency = self.predecoderLatency + self.subarray.writeLatency

            # For RESET and SET only
            self.resetLatency = self.predecoderLatency + self.subarray.resetLatency
            self.setLatency = self.predecoderLatency + self.subarray.setLatency

            # Valid for DRAM and eDRAM only
            self.refreshLatency = self.predecoderLatency + self.subarray.refreshLatency
            self.refreshLatency *= self.numColumnSubarray  # TOTAL refresh time for all subarrays

            # Add comparator latency for tag memory
            if self.memoryType == MemoryType.tag and self.internalSenseAmp:
                self.comparator.CalculateLatency(_rampInput)
                self.readLatency += self.comparator.readLatency

    def CalculatePower(self):
        """Calculate power consumption for all operations"""
        if not self.initialized:
            print("[Mat] Error: Require initialization first!")
        elif self.invalid:
            self.readDynamicEnergy = self.writeDynamicEnergy = self.leakage = g.invalid_value
        else:
            # Calculate power for all predecoder blocks
            self.rowPredecoderBlock1.CalculatePower()
            self.rowPredecoderBlock2.CalculatePower()
            self.bitlineMuxPredecoderBlock1.CalculatePower()
            self.bitlineMuxPredecoderBlock2.CalculatePower()
            self.senseAmpMuxLev1PredecoderBlock1.CalculatePower()
            self.senseAmpMuxLev1PredecoderBlock2.CalculatePower()
            self.senseAmpMuxLev2PredecoderBlock1.CalculatePower()
            self.senseAmpMuxLev2PredecoderBlock2.CalculatePower()
            self.subarray.CalculatePower()

            # Sum up predecoder energy
            self.readDynamicEnergy = (self.rowPredecoderBlock1.readDynamicEnergy + self.rowPredecoderBlock2.readDynamicEnergy +
                                     self.bitlineMuxPredecoderBlock1.readDynamicEnergy + self.bitlineMuxPredecoderBlock2.readDynamicEnergy +
                                     self.senseAmpMuxLev1PredecoderBlock1.readDynamicEnergy + self.senseAmpMuxLev1PredecoderBlock2.readDynamicEnergy +
                                     self.senseAmpMuxLev2PredecoderBlock1.readDynamicEnergy + self.senseAmpMuxLev2PredecoderBlock2.readDynamicEnergy)

            self.writeDynamicEnergy = (self.rowPredecoderBlock1.writeDynamicEnergy + self.rowPredecoderBlock2.writeDynamicEnergy +
                                      self.bitlineMuxPredecoderBlock1.writeDynamicEnergy + self.bitlineMuxPredecoderBlock2.writeDynamicEnergy +
                                      self.senseAmpMuxLev1PredecoderBlock1.writeDynamicEnergy + self.senseAmpMuxLev1PredecoderBlock2.writeDynamicEnergy +
                                      self.senseAmpMuxLev2PredecoderBlock1.writeDynamicEnergy + self.senseAmpMuxLev2PredecoderBlock2.writeDynamicEnergy)

            # Assume the predecoder bits are broadcast, so we don't need to multiply by total subarrays / active
            self.refreshDynamicEnergy = (self.rowPredecoderBlock1.readDynamicEnergy + self.rowPredecoderBlock2.readDynamicEnergy)
            self.refreshDynamicEnergy *= self.subarray.numRow * self.numRowSubarray  # Total predecoder energy for all REFs

            self.leakage = (self.rowPredecoderBlock1.leakage + self.rowPredecoderBlock2.leakage +
                           self.bitlineMuxPredecoderBlock1.leakage + self.bitlineMuxPredecoderBlock2.leakage +
                           self.senseAmpMuxLev1PredecoderBlock1.leakage + self.senseAmpMuxLev1PredecoderBlock2.leakage +
                           self.senseAmpMuxLev2PredecoderBlock1.leakage + self.senseAmpMuxLev2PredecoderBlock2.leakage)

            # Add subarray energy
            self.readDynamicEnergy += self.subarray.readDynamicEnergy * self.numActiveSubarrayPerRow * self.numActiveSubarrayPerColumn

            # This is now the total refresh energy for this Mat
            self.refreshDynamicEnergy += self.subarray.refreshDynamicEnergy * self.numRowSubarray * self.numColumnSubarray

            # Energy consumption on cells
            self.cellReadEnergy = self.subarray.cellReadEnergy * self.numActiveSubarrayPerRow * self.numActiveSubarrayPerColumn
            self.cellSetEnergy = self.subarray.cellSetEnergy * self.numActiveSubarrayPerRow * self.numActiveSubarrayPerColumn
            self.cellResetEnergy = self.subarray.cellResetEnergy * self.numActiveSubarrayPerRow * self.numActiveSubarrayPerColumn

            # For RESET and SET only
            self.resetDynamicEnergy = self.writeDynamicEnergy + self.subarray.resetDynamicEnergy * self.numActiveSubarrayPerRow * self.numActiveSubarrayPerColumn
            self.setDynamicEnergy = self.writeDynamicEnergy + self.subarray.setDynamicEnergy * self.numActiveSubarrayPerRow * self.numActiveSubarrayPerColumn

            # Total write energy
            self.writeDynamicEnergy += self.subarray.writeDynamicEnergy * self.numActiveSubarrayPerRow * self.numActiveSubarrayPerColumn
            self.leakage += self.subarray.leakage * self.numRowSubarray * self.numColumnSubarray

            # Add TSV energy for 3D stacking
            if self.stackedDieCount > 1 and self.partitionGranularity != 0:
                # Add address TSV energy
                self.readDynamicEnergy += (self.stackedDieCount - 1) * self.totalPredecoderOutputBits * self.tsvArray.readDynamicEnergy
                self.writeDynamicEnergy += (self.stackedDieCount - 1) * self.totalPredecoderOutputBits * self.tsvArray.writeDynamicEnergy
                self.resetDynamicEnergy += (self.stackedDieCount - 1) * self.totalPredecoderOutputBits * self.tsvArray.resetDynamicEnergy
                self.setDynamicEnergy += (self.stackedDieCount - 1) * self.totalPredecoderOutputBits * self.tsvArray.setDynamicEnergy
                self.refreshDynamicEnergy += (self.stackedDieCount - 1) * self.totalPredecoderOutputBits * self.tsvArray.readDynamicEnergy

                self.leakage += self.tsvArray.numTotalBits * (self.stackedDieCount - 1) * self.tsvArray.leakage

            # Add comparator power for tag memory
            if self.memoryType == MemoryType.tag and self.internalSenseAmp:
                self.comparator.CalculatePower()
                self.readDynamicEnergy += self.comparator.readDynamicEnergy * self.numWay
                self.writeDynamicEnergy += self.comparator.writeDynamicEnergy * self.numWay
                self.leakage += self.comparator.leakage * self.numWay

    def PrintProperty(self):
        """Print the properties of the Mat"""
        print("Mat Properties:")
        super().PrintProperty()

    def assign(self, rhs):
        """
        Assignment operator to copy all properties from another Mat instance.
        Python implementation of C++ operator=.

        Args:
            rhs: The Mat object to copy from

        Returns:
            self: For chaining
        """
        # Copy FunctionUnit base class properties
        self.height = rhs.height
        self.width = rhs.width
        self.area = rhs.area
        self.readLatency = rhs.readLatency
        self.writeLatency = rhs.writeLatency
        self.refreshLatency = rhs.refreshLatency
        self.readDynamicEnergy = rhs.readDynamicEnergy
        self.writeDynamicEnergy = rhs.writeDynamicEnergy
        self.resetLatency = rhs.resetLatency
        self.setLatency = rhs.setLatency
        self.resetDynamicEnergy = rhs.resetDynamicEnergy
        self.setDynamicEnergy = rhs.setDynamicEnergy
        self.refreshDynamicEnergy = rhs.refreshDynamicEnergy
        self.cellReadEnergy = rhs.cellReadEnergy
        self.cellSetEnergy = rhs.cellSetEnergy
        self.cellResetEnergy = rhs.cellResetEnergy
        self.leakage = rhs.leakage

        # Copy Mat-specific properties
        self.initialized = rhs.initialized
        self.invalid = rhs.invalid
        self.numRowSubarray = rhs.numRowSubarray
        self.numColumnSubarray = rhs.numColumnSubarray
        self.numAddressBit = rhs.numAddressBit
        self.numDataBit = rhs.numDataBit
        self.numWay = rhs.numWay
        self.numRowPerSet = rhs.numRowPerSet
        self.split = rhs.split
        self.internalSenseAmp = rhs.internalSenseAmp
        self.numActiveSubarrayPerRow = rhs.numActiveSubarrayPerRow
        self.numActiveSubarrayPerColumn = rhs.numActiveSubarrayPerColumn
        self.muxSenseAmp = rhs.muxSenseAmp
        self.muxOutputLev1 = rhs.muxOutputLev1
        self.muxOutputLev2 = rhs.muxOutputLev2
        self.areaOptimizationLevel = rhs.areaOptimizationLevel
        self.memoryType = rhs.memoryType
        self.stackedDieCount = rhs.stackedDieCount
        self.partitionGranularity = rhs.partitionGranularity
        self.totalPredecoderOutputBits = rhs.totalPredecoderOutputBits
        self.predecoderLatency = rhs.predecoderLatency
        self.areaAllLogicBlocks = rhs.areaAllLogicBlocks

        # Copy sub-component objects
        self.subarray.assign(rhs.subarray)

        # Copy predecoder blocks - use __copy__ or assign method if available
        if hasattr(rhs.rowPredecoderBlock1, 'assign'):
            self.rowPredecoderBlock1.assign(rhs.rowPredecoderBlock1)
            self.rowPredecoderBlock2.assign(rhs.rowPredecoderBlock2)
            self.bitlineMuxPredecoderBlock1.assign(rhs.bitlineMuxPredecoderBlock1)
            self.bitlineMuxPredecoderBlock2.assign(rhs.bitlineMuxPredecoderBlock2)
            self.senseAmpMuxLev1PredecoderBlock1.assign(rhs.senseAmpMuxLev1PredecoderBlock1)
            self.senseAmpMuxLev1PredecoderBlock2.assign(rhs.senseAmpMuxLev1PredecoderBlock2)
            self.senseAmpMuxLev2PredecoderBlock1.assign(rhs.senseAmpMuxLev2PredecoderBlock1)
            self.senseAmpMuxLev2PredecoderBlock2.assign(rhs.senseAmpMuxLev2PredecoderBlock2)
        else:
            # Fallback: create new instances if assign not available
            import copy
            self.rowPredecoderBlock1 = copy.copy(rhs.rowPredecoderBlock1)
            self.rowPredecoderBlock2 = copy.copy(rhs.rowPredecoderBlock2)
            self.bitlineMuxPredecoderBlock1 = copy.copy(rhs.bitlineMuxPredecoderBlock1)
            self.bitlineMuxPredecoderBlock2 = copy.copy(rhs.bitlineMuxPredecoderBlock2)
            self.senseAmpMuxLev1PredecoderBlock1 = copy.copy(rhs.senseAmpMuxLev1PredecoderBlock1)
            self.senseAmpMuxLev1PredecoderBlock2 = copy.copy(rhs.senseAmpMuxLev1PredecoderBlock2)
            self.senseAmpMuxLev2PredecoderBlock1 = copy.copy(rhs.senseAmpMuxLev2PredecoderBlock1)
            self.senseAmpMuxLev2PredecoderBlock2 = copy.copy(rhs.senseAmpMuxLev2PredecoderBlock2)

        # Copy comparator if applicable
        if self.memoryType == MemoryType.tag and self.internalSenseAmp:
            if hasattr(rhs.comparator, 'assign'):
                self.comparator.assign(rhs.comparator)
            else:
                import copy
                self.comparator = copy.copy(rhs.comparator)

        # Copy TSV array
        if hasattr(rhs.tsvArray, 'assign'):
            self.tsvArray.assign(rhs.tsvArray)
        else:
            import copy
            self.tsvArray = copy.copy(rhs.tsvArray)

        return self
