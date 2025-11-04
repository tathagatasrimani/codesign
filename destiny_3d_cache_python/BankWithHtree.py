#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

import math
from Bank import Bank
from Mat import Mat
from TSV import TSV
from typedef import (
    BufferDesignTarget, MemoryType, MemCellType, TSV_type, DesignTarget, CacheAccessMode
)
from constant import CONSTRAINT_ASPECT_RATIO_BANK
import globals as g


class BankWithHtree(Bank):
    """Bank implementation with H-tree routing for interconnects"""

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.invalid = False

        # Address and data bit routing
        self.numAddressBit = 0
        self.numDataDistributeBit = 0
        self.numDataBroadcastBit = 0

        # H-tree level information
        self.levelHorizontal = 0
        self.levelVertical = 0

        # Horizontal wire arrays - initialized to None (like NULL in C++)
        self.numHorizontalAddressBitToRoute = None
        self.numHorizontalDataDistributeBitToRoute = None
        self.numHorizontalDataBroadcastBitToRoute = None
        self.numHorizontalWire = None
        self.numSumHorizontalWire = None
        self.numActiveHorizontalWire = None
        self.lengthHorizontalWire = None

        # Vertical wire arrays - initialized to None (like NULL in C++)
        self.numVerticalAddressBitToRoute = None
        self.numVerticalDataDistributeBitToRoute = None
        self.numVerticalDataBroadcastBitToRoute = None
        self.numVerticalWire = None
        self.numSumVerticalWire = None
        self.numActiveVerticalWire = None
        self.lengthVerticalWire = None

    def Initialize(self, _numRowMat, _numColumnMat, _capacity, _blockSize, _associativity,
                  _numRowPerSet, _numActiveMatPerRow, _numActiveMatPerColumn, _muxSenseAmp,
                  _internalSenseAmp, _muxOutputLev1, _muxOutputLev2, _numRowSubarray,
                  _numColumnSubarray, _numActiveSubarrayPerRow, _numActiveSubarrayPerColumn,
                  _areaOptimizationLevel, _memoryType, _stackedDieCount, _partitionGranularity,
                  monolithicStackCount):
        """Initialize bank with H-tree routing"""

        # Reset if already initialized
        if self.initialized:
            # Reset all arrays to None
            self.numHorizontalAddressBitToRoute = None
            self.numHorizontalDataDistributeBitToRoute = None
            self.numHorizontalDataBroadcastBitToRoute = None
            self.numHorizontalWire = None
            self.numSumHorizontalWire = None
            self.numActiveHorizontalWire = None
            self.lengthHorizontalWire = None
            self.numVerticalAddressBitToRoute = None
            self.numVerticalDataDistributeBitToRoute = None
            self.numVerticalDataBroadcastBitToRoute = None
            self.numVerticalWire = None
            self.numSumVerticalWire = None
            self.numActiveVerticalWire = None
            self.lengthVerticalWire = None
            self.initialized = False
            self.invalid = False

        # Validate input - H-tree does not support external sense amplification
        if not _internalSenseAmp:
            self.invalid = True
            print("[Bank] Htree organization does not support external sense amplification scheme")
            return

        if self.initialized:
            print("[Bank] Warning: Already initialized!")

        # Store basic parameters
        self.numRowMat = _numRowMat
        self.numColumnMat = _numColumnMat
        self.capacity = _capacity
        self.blockSize = _blockSize
        self.associativity = _associativity
        self.numRowPerSet = _numRowPerSet
        self.internalSenseAmp = _internalSenseAmp
        self.areaOptimizationLevel = _areaOptimizationLevel
        self.memoryType = _memoryType
        self.stackedDieCount = _stackedDieCount
        self.partitionGranularity = _partitionGranularity
        numWay = 1  # default value for non-cache design

        # Calculate the physical signals that are required in routing
        # Use double during the calculation to avoid overflow
        if self.stackedDieCount > 1:
            self.numAddressBit = int(math.log2(float(_capacity) / _blockSize / _associativity / _stackedDieCount) + 0.1)
        else:
            self.numAddressBit = int(math.log2(float(_capacity) / _blockSize / _associativity) + 0.1)

        # Determine data distribution and broadcast bits based on memory type
        if _memoryType == MemoryType.data:
            self.numDataDistributeBit = _blockSize
            self.numDataBroadcastBit = int(math.log2(_associativity))
        elif _memoryType == MemoryType.tag:
            self.numDataDistributeBit = _associativity
            self.numDataBroadcastBit = _blockSize
        else:  # CAM
            self.numDataDistributeBit = 0
            self.numDataBroadcastBit = _blockSize

        # Validate and set active mat counts
        if _numActiveMatPerRow > _numColumnMat:
            print(f"[Bank] Warning: The number of active subarray per row is larger than the number of subarray per row!")
            print(f"{_numActiveMatPerRow} > {_numColumnMat}")
            self.numActiveMatPerRow = _numColumnMat
        else:
            self.numActiveMatPerRow = _numActiveMatPerRow

        if _numActiveMatPerColumn > _numRowMat:
            print(f"[Bank] Warning: The number of active subarray per column is larger than the number of subarray per column!")
            print(f"{_numActiveMatPerColumn} > {_numRowMat}")
            self.numActiveMatPerColumn = _numRowMat
        else:
            self.numActiveMatPerColumn = _numActiveMatPerColumn

        self.muxSenseAmp = _muxSenseAmp
        self.muxOutputLev1 = _muxOutputLev1
        self.muxOutputLev2 = _muxOutputLev2

        # Set subarray parameters
        self.numRowSubarray = _numRowSubarray
        self.numColumnSubarray = _numColumnSubarray

        if _numActiveSubarrayPerRow > _numColumnSubarray:
            print(f"[Bank] Warning: The number of active subarray per row is larger than the number of subarray per row!")
            print(f"{_numActiveSubarrayPerRow} > {_numColumnSubarray}")
            self.numActiveSubarrayPerRow = _numColumnSubarray
        else:
            self.numActiveSubarrayPerRow = _numActiveSubarrayPerRow

        if _numActiveSubarrayPerColumn > _numRowSubarray:
            print(f"[Bank] Warning: The number of active subarray per column is larger than the number of subarray per column!")
            print(f"{_numActiveSubarrayPerColumn} > {_numRowSubarray}")
            self.numActiveSubarrayPerColumn = _numRowSubarray
        else:
            self.numActiveSubarrayPerColumn = _numActiveSubarrayPerColumn

        # Calculate H-tree levels
        self.levelHorizontal = int(math.log2(self.numColumnMat) + 0.1)
        self.levelVertical = int(math.log2(self.numRowMat) + 0.1)

        # Allocate arrays for horizontal levels
        if self.levelHorizontal > 0:
            self.numHorizontalAddressBitToRoute = [0] * self.levelHorizontal
            self.numHorizontalDataDistributeBitToRoute = [0] * self.levelHorizontal
            self.numHorizontalDataBroadcastBitToRoute = [0] * self.levelHorizontal
            self.numHorizontalWire = [0] * self.levelHorizontal
            self.numSumHorizontalWire = [0] * self.levelHorizontal
            self.numActiveHorizontalWire = [0] * self.levelHorizontal
            self.lengthHorizontalWire = [0.0] * self.levelHorizontal

        # Allocate arrays for vertical levels
        if self.levelVertical > 0:
            self.numVerticalAddressBitToRoute = [0] * self.levelVertical
            self.numVerticalDataDistributeBitToRoute = [0] * self.levelVertical
            self.numVerticalDataBroadcastBitToRoute = [0] * self.levelVertical
            self.numVerticalWire = [0] * self.levelVertical
            self.numSumVerticalWire = [0] * self.levelVertical
            self.numActiveVerticalWire = [0] * self.levelVertical
            self.lengthVerticalWire = [0.0] * self.levelVertical

        # Initialize H-tree routing algorithm
        h = self.levelHorizontal
        v = self.levelVertical
        rowToActive = self.numActiveMatPerColumn
        columnToActive = self.numActiveMatPerRow
        numAddressBitToRoute = self.numAddressBit
        numDataDistributeBitToRoute = self.numDataDistributeBit
        numDataBroadcastBitToRoute = self.numDataBroadcastBit

        # Always route H as the first step
        if h > 0:
            if numDataDistributeBitToRoute + numDataBroadcastBitToRoute == 0 or numAddressBitToRoute == 0:
                self.invalid = True
                self.initialized = True
                return
            self.numHorizontalAddressBitToRoute[0] = numAddressBitToRoute
            self.numHorizontalDataDistributeBitToRoute[0] = numDataDistributeBitToRoute
            self.numHorizontalDataBroadcastBitToRoute[0] = numDataBroadcastBitToRoute
            self.numHorizontalWire[0] = 1
            self.numSumHorizontalWire[0] = 1
            self.numActiveHorizontalWire[0] = 1
            h -= 1

        hTemp = 1
        vTemp = 1

        # If H is larger than V, then reduce H to V
        while h > v:
            if numDataDistributeBitToRoute + numDataBroadcastBitToRoute == 0 or numAddressBitToRoute == 0:
                self.invalid = True
                self.initialized = True
                return
            # If there is possibility to reduce the data bits
            if columnToActive > 1:
                numDataDistributeBitToRoute //= 2
                columnToActive //= 2
                self.numActiveHorizontalWire[self.levelHorizontal - h] = 2 * self.numActiveHorizontalWire[self.levelHorizontal - h - 1]
            else:
                numAddressBitToRoute -= 1
                self.numActiveHorizontalWire[self.levelHorizontal - h] = self.numActiveHorizontalWire[self.levelHorizontal - h - 1]

            self.numHorizontalAddressBitToRoute[self.levelHorizontal - h] = numAddressBitToRoute
            self.numHorizontalDataDistributeBitToRoute[self.levelHorizontal - h] = numDataDistributeBitToRoute
            self.numHorizontalDataBroadcastBitToRoute[self.levelHorizontal - h] = numDataBroadcastBitToRoute
            self.numHorizontalWire[self.levelHorizontal - h] = 1
            self.numSumHorizontalWire[self.levelHorizontal - h] = 2 * self.numSumHorizontalWire[self.levelHorizontal - h - 1]
            h -= 1
            vTemp *= 2

        # If V is larger than H, then reduce V to H
        while v > h:
            if numDataDistributeBitToRoute + numDataBroadcastBitToRoute == 0 or numAddressBitToRoute == 0:
                self.invalid = True
                self.initialized = True
                return
            # If there is possibility to reduce the data bits on vertical
            if rowToActive > 1:
                numDataDistributeBitToRoute //= 2
                rowToActive //= 2
                if v == self.levelVertical:
                    self.numActiveVerticalWire[0] = 2
                else:
                    self.numActiveVerticalWire[self.levelVertical - v] = 2 * self.numActiveVerticalWire[self.levelVertical - v - 1]
            else:
                numAddressBitToRoute -= 1
                if v == self.levelVertical:
                    self.numActiveVerticalWire[0] = 1
                else:
                    self.numActiveVerticalWire[self.levelVertical - v] = self.numActiveVerticalWire[self.levelVertical - v - 1]

            self.numVerticalAddressBitToRoute[self.levelVertical - v] = numAddressBitToRoute
            self.numVerticalDataDistributeBitToRoute[self.levelVertical - v] = numDataDistributeBitToRoute
            self.numVerticalDataBroadcastBitToRoute[self.levelVertical - v] = numDataBroadcastBitToRoute
            self.numVerticalWire[self.levelVertical - v] = 1
            if v == self.levelVertical:
                self.numSumVerticalWire[0] = 2
            else:
                self.numSumVerticalWire[self.levelVertical - v] = 2 * self.numSumVerticalWire[self.levelVertical - v - 1]
            v -= 1
            hTemp *= 2

        # Reduce H and V to zero
        while h > 0:
            if numDataDistributeBitToRoute + numDataBroadcastBitToRoute == 0 or numAddressBitToRoute == 0:
                self.invalid = True
                self.initialized = True
                return
            # If there is possibility to reduce the data bits
            if columnToActive > 1:
                numDataDistributeBitToRoute //= 2
                columnToActive //= 2
                if v == self.levelVertical:
                    self.numActiveHorizontalWire[self.levelHorizontal - h] = 2 * self.numActiveHorizontalWire[self.levelHorizontal - h - 1]
                else:
                    self.numActiveHorizontalWire[self.levelHorizontal - h] = 2 * self.numActiveVerticalWire[self.levelVertical - v - 1]
            else:
                numAddressBitToRoute -= 1
                if v == self.levelVertical:
                    self.numActiveHorizontalWire[self.levelHorizontal - h] = self.numActiveHorizontalWire[self.levelHorizontal - h - 1]
                else:
                    self.numActiveHorizontalWire[self.levelHorizontal - h] = self.numActiveVerticalWire[self.levelVertical - v - 1]

            self.numHorizontalAddressBitToRoute[self.levelHorizontal - h] = numAddressBitToRoute
            self.numHorizontalDataDistributeBitToRoute[self.levelHorizontal - h] = numDataDistributeBitToRoute
            self.numHorizontalDataBroadcastBitToRoute[self.levelHorizontal - h] = numDataBroadcastBitToRoute
            self.numHorizontalWire[self.levelHorizontal - h] = hTemp
            if v == self.levelVertical:
                self.numSumHorizontalWire[self.levelHorizontal - h] = 2 * self.numSumHorizontalWire[self.levelHorizontal - h - 1]
            else:
                self.numSumHorizontalWire[self.levelHorizontal - h] = 2 * self.numSumVerticalWire[self.levelVertical - v - 1]

            if numDataDistributeBitToRoute + numDataBroadcastBitToRoute == 0 or numAddressBitToRoute == 0:
                self.invalid = True
                self.initialized = True
                return
            # If there is possibility to reduce the data bits on vertical
            if rowToActive > 1:
                numDataDistributeBitToRoute //= 2
                rowToActive //= 2
                self.numActiveVerticalWire[self.levelVertical - v] = 2 * self.numActiveHorizontalWire[self.levelHorizontal - h]
            else:
                numAddressBitToRoute -= 1
                self.numActiveVerticalWire[self.levelVertical - v] = self.numActiveHorizontalWire[self.levelHorizontal - h]

            self.numVerticalAddressBitToRoute[self.levelVertical - v] = numAddressBitToRoute
            self.numVerticalDataDistributeBitToRoute[self.levelVertical - v] = numDataDistributeBitToRoute
            self.numVerticalDataBroadcastBitToRoute[self.levelVertical - v] = numDataBroadcastBitToRoute
            if self.levelHorizontal == 2:
                self.numVerticalWire[self.levelVertical - v] = vTemp
            else:
                self.numVerticalWire[self.levelVertical - v] = 2 * vTemp
            self.numSumVerticalWire[self.levelVertical - v] = 2 * self.numSumHorizontalWire[self.levelHorizontal - h]
            h -= 1
            v -= 1
            hTemp *= 2
            vTemp *= 2

        # Final bit reduction
        if numDataDistributeBitToRoute + numDataBroadcastBitToRoute == 0 or numAddressBitToRoute == 0:
            self.invalid = True
            self.initialized = True
            return

        if columnToActive > 1:
            numDataDistributeBitToRoute //= 2
            columnToActive //= 2
        else:
            if self.levelHorizontal > 0:
                numAddressBitToRoute -= 1

        # Validate cache data array configuration
        if self.memoryType == MemoryType.data:
            if self.numRowPerSet > int(pow(2, numDataBroadcastBitToRoute)):
                # There is no enough ways to distribute into multiple rows
                self.invalid = True
                self.initialized = True
                return

        # Validate cache tag array configuration
        if self.memoryType == MemoryType.tag:
            if self.numRowPerSet > 1:
                # tag array cannot have multiple rows to contain ways in a set
                self.invalid = True
                self.initialized = True
                return
            if numDataDistributeBitToRoute == 0:
                # This mat does not contain at least one way
                self.invalid = True
                self.initialized = True
                return

        # Determine the number of columns in a mat
        if self.memoryType == MemoryType.data:  # Data array
            matBlockSize = numDataDistributeBitToRoute
            numWay = int(pow(2, numDataBroadcastBitToRoute))
            # Consider the case if each mat is a cache data array that contains multiple ways
            numWayPerRow = numWay // self.numRowPerSet
            if numWayPerRow > 1:  # multiple ways per row, needs extra mux level
                # Do mux level recalculation to contain the multiple ways
                if g.cell.memCellType == MemCellType.DRAM or g.cell.memCellType == MemCellType.eDRAM:
                    # for DRAM, mux before sense amp has to be 1
                    numWayPerRowInLog = int(math.log2(float(numWayPerRow)) + 0.1)
                    extraMuxOutputLev2 = int(pow(2, numWayPerRowInLog // 2))
                    extraMuxOutputLev1 = numWayPerRow // extraMuxOutputLev2
                    self.muxOutputLev1 *= extraMuxOutputLev1
                    self.muxOutputLev2 *= extraMuxOutputLev2
                else:
                    # for non-DRAM, all mux levels can be used
                    numWayPerRowInLog = int(math.log2(float(numWayPerRow)) + 0.1)
                    extraMuxOutputLev2 = int(pow(2, numWayPerRowInLog // 3))
                    extraMuxOutputLev1 = extraMuxOutputLev2
                    extraMuxSenseAmp = numWayPerRow // extraMuxOutputLev1 // extraMuxOutputLev2
                    self.muxSenseAmp *= extraMuxSenseAmp
                    self.muxOutputLev1 *= extraMuxOutputLev1
                    self.muxOutputLev2 *= extraMuxOutputLev2
        elif self.memoryType == MemoryType.tag:  # Tag array
            matBlockSize = numDataBroadcastBitToRoute
            numWay = numDataDistributeBitToRoute
        else:  # CAM
            matBlockSize = numDataBroadcastBitToRoute
            numWay = 1

        # Initialize the mat
        self.mat.Initialize(
            self.numRowSubarray, self.numColumnSubarray, numAddressBitToRoute, matBlockSize,
            numWay, self.numRowPerSet, False, self.numActiveSubarrayPerRow, self.numActiveSubarrayPerColumn,
            self.muxSenseAmp, self.internalSenseAmp, self.muxOutputLev1, self.muxOutputLev2,
            self.areaOptimizationLevel, self.memoryType, self.stackedDieCount,
            self.partitionGranularity, monolithicStackCount
        )

        # Check if mat is under a legal configuration
        if self.mat.invalid:
            self.invalid = True
            self.initialized = True
            return

        # Reset the mux values for correct printing
        self.muxSenseAmp = _muxSenseAmp
        self.muxOutputLev1 = _muxOutputLev1
        self.muxOutputLev2 = _muxOutputLev2

        # Initialize TSV connections
        if self.stackedDieCount > 0:
            tsv_type = g.tech.WireTypeToTSVType(g.inputParameter.maxGlobalWireType)
            self.tsvArray.Initialize(tsv_type)

        self.initialized = True

    def CalculateArea(self):
        """Calculate bank area with H-tree routing"""
        if not self.initialized:
            print("[Bank] Error: Require initialization first!")
        elif self.invalid:
            self.height = self.width = self.area = g.invalid_value
        else:
            # Calculate mat area first
            self.mat.CalculateArea()
            self.height = self.mat.height * self.numRowMat
            self.width = self.mat.width * self.numColumnMat

            # Add wire area
            numWireSharingWidth = 1
            effectivePitch = 0.0

            if g.globalWire.wireRepeaterType == 0:  # repeated_none
                numWireSharingWidth = 1
                effectivePitch = 0  # assume wire built on another metal layer
            else:
                numWireSharingWidth = int(g.globalWire.repeaterSpacing / g.globalWire.repeaterHeight)
                effectivePitch = g.globalWire.repeatedWirePitch

            # Add horizontal wire area
            for i in range(self.levelHorizontal):
                self.height += math.ceil(
                    float(self.numHorizontalAddressBitToRoute[i] +
                          self.numHorizontalDataDistributeBitToRoute[i] +
                          self.numHorizontalDataBroadcastBitToRoute[i]) *
                    self.numHorizontalWire[i] / numWireSharingWidth
                ) * effectivePitch

            # Add vertical wire area
            for i in range(self.levelVertical):
                self.width += math.ceil(
                    float(self.numVerticalAddressBitToRoute[i] +
                          self.numVerticalDataDistributeBitToRoute[i] +
                          self.numVerticalDataBroadcastBitToRoute[i]) *
                    self.numVerticalWire[i] / numWireSharingWidth
                ) * effectivePitch

            # Determine if the aspect ratio meets the constraint
            if self.memoryType == MemoryType.data:
                if (self.height / self.width > CONSTRAINT_ASPECT_RATIO_BANK or
                    self.width / self.height > CONSTRAINT_ASPECT_RATIO_BANK):
                    # illegal
                    self.invalid = True
                    self.height = self.width = self.area = g.invalid_value
                    return

            self.area = self.height * self.width

            # Calculate the length of each H-tree wire
            h = self.levelHorizontal - 1
            v = self.levelVertical - 1

            # Process vertical wires when v > h
            while v > h:
                if v == self.levelVertical - 1:
                    self.lengthVerticalWire[v] = self.mat.height / 2
                else:
                    self.lengthVerticalWire[v] = self.lengthVerticalWire[v + 1] * 2
                v -= 1

            numHorizontalBitToRoute = 0.0
            numVerticalBitToRoute = 0.0

            # Process both horizontal and vertical wires
            while v >= 0:
                if v == self.levelVertical - 1:
                    self.lengthVerticalWire[v] = self.mat.height / 2
                else:
                    if h == self.levelHorizontal - 1:
                        self.lengthVerticalWire[v] = self.lengthVerticalWire[v + 1] * 2
                    else:
                        numHorizontalBitToRoute = (self.numHorizontalAddressBitToRoute[h + 1] +
                                                  self.numHorizontalDataDistributeBitToRoute[h + 1] +
                                                  self.numHorizontalDataBroadcastBitToRoute[h + 1])
                        self.lengthVerticalWire[v] = (self.lengthVerticalWire[v + 1] * 2 +
                                                     math.ceil(numHorizontalBitToRoute / numWireSharingWidth) * effectivePitch / 2)

                if h == self.levelHorizontal - 1:
                    self.lengthHorizontalWire[h] = self.mat.width
                    for i in range(v, self.levelVertical):
                        numVerticalBitToRoute = (self.numVerticalAddressBitToRoute[i] +
                                               self.numVerticalDataDistributeBitToRoute[i] +
                                               self.numVerticalDataBroadcastBitToRoute[i])
                        self.lengthHorizontalWire[h] += math.ceil(numVerticalBitToRoute / numWireSharingWidth) * effectivePitch / 2
                else:
                    numVerticalBitToRoute = (self.numVerticalAddressBitToRoute[v] +
                                           self.numVerticalDataDistributeBitToRoute[v] +
                                           self.numVerticalDataBroadcastBitToRoute[v])
                    self.lengthHorizontalWire[h] = (self.lengthHorizontalWire[h + 1] * 2 +
                                                   math.ceil(numVerticalBitToRoute / numWireSharingWidth) * effectivePitch / 2)
                v -= 1
                h -= 1

            # Process remaining horizontal wires
            while h >= 0:
                if h == self.levelHorizontal - 1:
                    self.lengthHorizontalWire[h] = self.mat.width
                else:
                    self.lengthHorizontalWire[h] = self.lengthHorizontalWire[h + 1] * 2
                h -= 1

            # Initialize TSV connections
            if self.stackedDieCount > 1:
                self.tsvArray.CalculateArea()

                numControlBits = self.stackedDieCount
                numAddressBits = int(math.log2(float(self.capacity) / self.blockSize /
                                              self.associativity / self.stackedDieCount) + 0.1)
                numDataBits = self.blockSize * 2  # Read and write TSVs

                # Fine-granularity has predecoders on logic layer
                if self.partitionGranularity == 1:
                    numAddressBits = 0

                redundancyFactor = g.inputParameter.tsvRedundancy
                self.tsvArray.numTotalBits = int(float(numControlBits + numAddressBits + numDataBits) * redundancyFactor)
                self.tsvArray.numAccessBits = int(float(numControlBits + numAddressBits + self.blockSize) * redundancyFactor)

                # We're not adding in a particular dimension so increase the total
                self.area += self.tsvArray.numTotalBits * self.tsvArray.area

    def CalculateRC(self):
        """Calculate RC parameters"""
        if not self.initialized:
            print("[Bank] Error: Require initialization first!")
        elif not self.invalid:
            self.mat.CalculateRC()

    def CalculateLatencyAndPower(self):
        """Calculate latency and power consumption"""
        if not self.initialized:
            print("[Bank] Error: Require initialization first!")
        elif self.invalid:
            self.readLatency = self.writeLatency = g.invalid_value
            self.readDynamicEnergy = self.writeDynamicEnergy = g.invalid_value
            self.leakage = g.invalid_value
        else:
            beta = 1  # Default value. For fast access mode cache, equals associativity

            # Calculate mat latency and power
            self.mat.CalculateLatency(g.infinite_ramp)
            self.mat.CalculatePower()

            self.readLatency = self.mat.readLatency
            self.writeLatency = self.mat.writeLatency
            self.refreshLatency = self.mat.refreshLatency * self.numColumnMat  # TOTAL refresh time for all Mats
            self.readDynamicEnergy = self.mat.readDynamicEnergy * self.numActiveMatPerRow * self.numActiveMatPerColumn
            self.writeDynamicEnergy = self.mat.writeDynamicEnergy * self.numActiveMatPerRow * self.numActiveMatPerColumn
            self.refreshDynamicEnergy = self.mat.refreshDynamicEnergy * self.numRowMat * self.numColumnMat
            self.leakage = self.mat.leakage * self.numRowMat * self.numColumnMat

            # Energy consumption on cells
            self.cellReadEnergy = self.mat.cellReadEnergy * self.numActiveMatPerRow * self.numActiveMatPerColumn
            self.cellSetEnergy = self.mat.cellSetEnergy * self.numActiveMatPerRow * self.numActiveMatPerColumn
            self.cellResetEnergy = self.mat.cellResetEnergy * self.numActiveMatPerRow * self.numActiveMatPerColumn

            # For asymmetric RESET/SET only
            self.resetLatency = self.mat.resetLatency
            self.setLatency = self.mat.setLatency
            self.resetDynamicEnergy = self.mat.resetDynamicEnergy * self.numActiveMatPerRow * self.numActiveMatPerColumn
            self.setDynamicEnergy = self.mat.setDynamicEnergy * self.numActiveMatPerRow * self.numActiveMatPerColumn

            if (g.inputParameter.designTarget == DesignTarget.cache and
                g.inputParameter.cacheAccessMode == CacheAccessMode.fast_access_mode):
                beta = g.inputParameter.associativity

            # Calculate horizontal wire latency and power
            for i in range(self.levelHorizontal):
                latency, energy, leakageWire = g.globalWire.CalculateLatencyAndPower(self.lengthHorizontalWire[i])

                self.readLatency += latency * 2  # 2 due to in/out
                self.writeLatency += latency  # only in
                self.resetLatency += latency
                self.setLatency += latency
                self.refreshLatency += latency

                # Read and write energy for H-tree should be the same
                self.readDynamicEnergy += (energy * self.numActiveHorizontalWire[i] *
                    (self.numHorizontalAddressBitToRoute[i] + self.numHorizontalDataDistributeBitToRoute[i] +
                     self.numHorizontalDataBroadcastBitToRoute[i]))
                self.writeDynamicEnergy += (energy * self.numActiveHorizontalWire[i] *
                    (self.numHorizontalAddressBitToRoute[i] + self.numHorizontalDataDistributeBitToRoute[i] +
                     self.numHorizontalDataBroadcastBitToRoute[i]) / beta)
                self.resetDynamicEnergy += (energy * self.numActiveHorizontalWire[i] *
                    (self.numHorizontalAddressBitToRoute[i] + self.numHorizontalDataDistributeBitToRoute[i] +
                     self.numHorizontalDataBroadcastBitToRoute[i]) / beta)
                self.setDynamicEnergy += (energy * self.numActiveHorizontalWire[i] *
                    (self.numHorizontalAddressBitToRoute[i] + self.numHorizontalDataDistributeBitToRoute[i] +
                     self.numHorizontalDataBroadcastBitToRoute[i]) / beta)
                self.refreshDynamicEnergy += (energy * self.numActiveHorizontalWire[i] *
                    (self.numHorizontalAddressBitToRoute[i] + self.numHorizontalDataDistributeBitToRoute[i] +
                     self.numHorizontalDataBroadcastBitToRoute[i]) / beta)
                self.leakage += (leakageWire * self.numSumHorizontalWire[i] *
                    (self.numHorizontalAddressBitToRoute[i] + self.numHorizontalDataDistributeBitToRoute[i] +
                     self.numHorizontalDataBroadcastBitToRoute[i]))

            # Calculate vertical wire latency and power
            for i in range(self.levelVertical):
                latency, energy, leakageWire = g.globalWire.CalculateLatencyAndPower(self.lengthVerticalWire[i])

                self.readLatency += latency * 2  # 2 due to in/out
                self.writeLatency += latency  # only in
                self.resetLatency += latency
                self.setLatency += latency
                self.refreshLatency += latency

                self.readDynamicEnergy += (energy * self.numActiveVerticalWire[i] *
                    (self.numVerticalAddressBitToRoute[i] + self.numVerticalDataDistributeBitToRoute[i] +
                     self.numVerticalDataBroadcastBitToRoute[i]))
                self.writeDynamicEnergy += (energy * self.numActiveVerticalWire[i] *
                    (self.numVerticalAddressBitToRoute[i] + self.numVerticalDataDistributeBitToRoute[i] +
                     self.numVerticalDataBroadcastBitToRoute[i]) / beta)
                self.resetDynamicEnergy += (energy * self.numActiveVerticalWire[i] *
                    (self.numVerticalAddressBitToRoute[i] + self.numVerticalDataDistributeBitToRoute[i] +
                     self.numVerticalDataBroadcastBitToRoute[i]) / beta)
                self.setDynamicEnergy += (energy * self.numActiveVerticalWire[i] *
                    (self.numVerticalAddressBitToRoute[i] + self.numVerticalDataDistributeBitToRoute[i] +
                     self.numVerticalDataBroadcastBitToRoute[i]) / beta)
                self.refreshDynamicEnergy += (energy * self.numActiveVerticalWire[i] *
                    (self.numVerticalAddressBitToRoute[i] + self.numVerticalDataDistributeBitToRoute[i] +
                     self.numVerticalDataBroadcastBitToRoute[i]) / beta)
                self.leakage += (leakageWire * self.numSumVerticalWire[i] *
                    (self.numVerticalAddressBitToRoute[i] + self.numVerticalDataDistributeBitToRoute[i] +
                     self.numVerticalDataBroadcastBitToRoute[i]))

            # Calculate routing overhead
            self.routingReadLatency = self.readLatency - self.mat.readLatency
            self.routingWriteLatency = self.writeLatency - self.mat.writeLatency
            self.routingResetLatency = self.resetLatency - self.mat.resetLatency
            self.routingSetLatency = self.setLatency - self.mat.setLatency
            self.routingRefreshLatency = self.refreshLatency - self.mat.refreshLatency

            self.routingReadDynamicEnergy = (self.readDynamicEnergy -
                self.mat.readDynamicEnergy * self.numActiveMatPerColumn * self.numActiveMatPerRow)
            self.routingWriteDynamicEnergy = (self.writeDynamicEnergy -
                self.mat.writeDynamicEnergy * self.numActiveMatPerColumn * self.numActiveMatPerRow)
            self.routingResetDynamicEnergy = (self.resetDynamicEnergy -
                self.mat.resetDynamicEnergy * self.numActiveMatPerColumn * self.numActiveMatPerRow)
            self.routingSetDynamicEnergy = (self.setDynamicEnergy -
                self.mat.setDynamicEnergy * self.numActiveMatPerColumn * self.numActiveMatPerRow)
            self.routingRefreshDynamicEnergy = (self.refreshDynamicEnergy -
                self.mat.refreshDynamicEnergy * self.numActiveMatPerColumn * self.numActiveMatPerRow)

            self.routingLeakage = self.leakage - self.mat.leakage * self.numColumnMat * self.numRowMat

            # For Htree bank, each layer contains an exact copy of this bank
            if self.stackedDieCount > 1:
                self.leakage *= self.stackedDieCount

                # Normally senseAmpMuxLev2 is the last driver from Mat
                tsvReadRampInput = 1e20
                # Bank is the end unit for NVSIM, so we assume something external
                # is fully driving the input data values
                tsvWriteRampInput = g.infinite_ramp

                # Add TSV energy
                self.tsvArray.CalculateLatencyAndPower(tsvReadRampInput, tsvWriteRampInput)

                numControlBits = self.stackedDieCount
                numAddressBits = int(math.log2(float(self.capacity) / self.blockSize /
                                              self.associativity / self.stackedDieCount) + 0.1)
                numDataBits = self.blockSize * 2  # Read and write TSVs

                # Fine-granularity has predecoders on logic layer
                if self.partitionGranularity == 1:
                    numAddressBits = 0

                redundancyFactor = g.inputParameter.tsvRedundancy
                self.tsvArray.numTotalBits = int(float(numControlBits + numAddressBits + numDataBits) * redundancyFactor)
                self.tsvArray.numAccessBits = int(float(numControlBits + numAddressBits + self.blockSize) * redundancyFactor)
                self.tsvArray.numReadBits = int(float(numControlBits + numAddressBits) * redundancyFactor)
                self.tsvArray.numDataBits = int(float(self.blockSize) * redundancyFactor)

                # Always assume worst case going to furthest die
                self.readLatency += ((self.stackedDieCount - 1) * self.tsvArray.readLatency +
                                    (self.stackedDieCount - 1) * self.tsvArray.writeLatency)
                self.writeLatency += (self.stackedDieCount - 1) * self.tsvArray.writeLatency
                self.resetLatency += (self.stackedDieCount - 1) * self.tsvArray.resetLatency
                self.setLatency += (self.stackedDieCount - 1) * self.tsvArray.setLatency

                # Also assume worst energy
                self.readDynamicEnergy += (self.tsvArray.numReadBits * (self.stackedDieCount - 1) *
                    self.tsvArray.writeDynamicEnergy + self.tsvArray.numDataBits *
                    self.tsvArray.readDynamicEnergy * (self.stackedDieCount - 1))
                self.writeDynamicEnergy += (self.tsvArray.numAccessBits * (self.stackedDieCount - 1) *
                    self.tsvArray.writeDynamicEnergy)
                self.resetDynamicEnergy += (self.tsvArray.numAccessBits * (self.stackedDieCount - 1) *
                    self.tsvArray.resetDynamicEnergy)
                self.setDynamicEnergy += (self.tsvArray.numAccessBits * (self.stackedDieCount - 1) *
                    self.tsvArray.setDynamicEnergy)
                self.refreshDynamicEnergy += (self.tsvArray.numReadBits * (self.stackedDieCount - 1) *
                    self.tsvArray.writeDynamicEnergy)

                self.leakage += self.tsvArray.numTotalBits * (self.stackedDieCount - 1) * self.tsvArray.leakage

        # Check if eDRAM refresh time is valid
        if g.cell.memCellType == MemCellType.eDRAM:
            if self.refreshLatency > g.cell.retentionTime:
                self.invalid = True
