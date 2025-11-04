#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

from FunctionUnit import FunctionUnit
from Mat import Mat
from TSV import TSV
from typedef import BufferDesignTarget, MemoryType


class Bank(FunctionUnit):
    """
    Bank class - Base class for memory bank implementations.
    This is an abstract base class that must be inherited by concrete implementations.
    """

    def __init__(self):
        """Constructor - Initialize all properties to default values."""
        super().__init__()

        # Initialization flags
        self.initialized = False
        self.invalid = False

        # Configuration properties
        self.internalSenseAmp = False
        self.numRowMat = 0              # Number of mat rows in a bank
        self.numColumnMat = 0           # Number of mat columns in a bank
        self.capacity = 0               # The capacity of this bank, Unit: bit
        self.blockSize = 0              # The basic block size in this bank, Unit: bit
        self.associativity = 0          # Associativity, for cache design only
        self.numRowPerSet = 0           # For cache design, the number of wordlines which a set is partitioned into
        self.numActiveMatPerRow = 0     # For different access types
        self.numActiveMatPerColumn = 0  # For different access types
        self.muxSenseAmp = 0            # How many bitlines connect to one sense amplifier
        self.muxOutputLev1 = 0          # How many sense amplifiers connect to one output bit, level-1
        self.muxOutputLev2 = 0          # How many sense amplifiers connect to one output bit, level-2
        self.numRowSubarray = 0         # Number of subarray rows in a mat
        self.numColumnSubarray = 0      # Number of subarray columns in a mat
        self.numActiveSubarrayPerRow = 0     # For different access types
        self.numActiveSubarrayPerColumn = 0  # For different access types

        # Design optimization and memory type
        self.areaOptimizationLevel = BufferDesignTarget.latency_first
        self.memoryType = MemoryType.data

        # 3D stacking properties
        self.stackedDieCount = 1
        self.partitionGranularity = 0

        # Routing latencies
        self.routingReadLatency = 0.0
        self.routingWriteLatency = 0.0
        self.routingResetLatency = 0.0
        self.routingSetLatency = 0.0
        self.routingRefreshLatency = 0.0

        # Routing dynamic energies (Non-TSV routing energy)
        self.routingReadDynamicEnergy = 0.0
        self.routingWriteDynamicEnergy = 0.0
        self.routingResetDynamicEnergy = 0.0
        self.routingSetDynamicEnergy = 0.0
        self.routingRefreshDynamicEnergy = 0.0

        # Routing leakage
        self.routingLeakage = 0.0

        # Component objects
        self.mat = Mat()
        self.tsvArray = TSV()

    def PrintProperty(self):
        """Print bank properties."""
        print("Bank Properties:")
        super().PrintProperty()

    def Initialize(self, _numRowMat, _numColumnMat, _capacity, _blockSize, _associativity,
                  _numRowPerSet, _numActiveMatPerRow, _numActiveMatPerColumn, _muxSenseAmp,
                  _internalSenseAmp, _muxOutputLev1, _muxOutputLev2, _numRowSubarray,
                  _numColumnSubarray, _numActiveSubarrayPerRow, _numActiveSubarrayPerColumn,
                  _areaOptimizationLevel, _memoryType, _stackedDieCount, _partitionGranularity,
                  monolithicStackCount):
        """
        Initialize bank configuration.
        This is an abstract method that must be implemented by subclasses.

        Args:
            _numRowMat: Number of mat rows in the bank
            _numColumnMat: Number of mat columns in the bank
            _capacity: Capacity of the bank in bits
            _blockSize: Basic block size in bits
            _associativity: Cache associativity
            _numRowPerSet: Number of wordlines per set
            _numActiveMatPerRow: Number of active mats per row
            _numActiveMatPerColumn: Number of active mats per column
            _muxSenseAmp: Bitlines per sense amplifier
            _internalSenseAmp: Whether to use internal sense amplifier
            _muxOutputLev1: Level-1 output mux ratio
            _muxOutputLev2: Level-2 output mux ratio
            _numRowSubarray: Number of subarray rows in a mat
            _numColumnSubarray: Number of subarray columns in a mat
            _numActiveSubarrayPerRow: Number of active subarrays per row
            _numActiveSubarrayPerColumn: Number of active subarrays per column
            _areaOptimizationLevel: Buffer design target
            _memoryType: Type of memory
            _stackedDieCount: Number of stacked dies
            _partitionGranularity: Partition granularity
            monolithicStackCount: Monolithic stack count
        """
        raise NotImplementedError("This method must be implemented in subclasses")

    def CalculateArea(self):
        """
        Calculate the area of the bank.
        This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented in subclasses")

    def CalculateRC(self):
        """
        Calculate resistance and capacitance.
        This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented in subclasses")

    def CalculateLatencyAndPower(self):
        """
        Calculate latency and power consumption.
        This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented in subclasses")

    def __assign__(self, rhs):
        """
        Assignment operator equivalent.
        Copies all properties from another Bank instance.

        Args:
            rhs: The Bank instance to copy from

        Returns:
            self: The current instance
        """
        # Copy FunctionUnit properties
        self.height = rhs.height
        self.width = rhs.width
        self.area = rhs.area
        self.readLatency = rhs.readLatency
        self.writeLatency = rhs.writeLatency
        self.readDynamicEnergy = rhs.readDynamicEnergy
        self.writeDynamicEnergy = rhs.writeDynamicEnergy
        self.resetLatency = rhs.resetLatency
        self.setLatency = rhs.setLatency
        self.refreshLatency = rhs.refreshLatency
        self.resetDynamicEnergy = rhs.resetDynamicEnergy
        self.setDynamicEnergy = rhs.setDynamicEnergy
        self.refreshDynamicEnergy = rhs.refreshDynamicEnergy
        self.cellReadEnergy = rhs.cellReadEnergy
        self.cellSetEnergy = rhs.cellSetEnergy
        self.cellResetEnergy = rhs.cellResetEnergy
        self.leakage = rhs.leakage

        # Copy Bank-specific properties
        self.initialized = rhs.initialized
        self.invalid = rhs.invalid
        self.numRowMat = rhs.numRowMat
        self.numColumnMat = rhs.numColumnMat
        self.capacity = rhs.capacity
        self.blockSize = rhs.blockSize
        self.associativity = rhs.associativity
        self.numRowPerSet = rhs.numRowPerSet
        self.numActiveMatPerRow = rhs.numActiveMatPerRow
        self.numActiveMatPerColumn = rhs.numActiveMatPerColumn
        self.muxSenseAmp = rhs.muxSenseAmp
        self.internalSenseAmp = rhs.internalSenseAmp
        self.muxOutputLev1 = rhs.muxOutputLev1
        self.muxOutputLev2 = rhs.muxOutputLev2
        self.areaOptimizationLevel = rhs.areaOptimizationLevel
        self.memoryType = rhs.memoryType
        self.numRowSubarray = rhs.numRowSubarray
        self.numColumnSubarray = rhs.numColumnSubarray
        self.numActiveSubarrayPerRow = rhs.numActiveSubarrayPerRow
        self.numActiveSubarrayPerColumn = rhs.numActiveSubarrayPerColumn
        self.stackedDieCount = rhs.stackedDieCount
        self.partitionGranularity = rhs.partitionGranularity

        # Copy routing properties
        self.routingReadLatency = rhs.routingReadLatency
        self.routingWriteLatency = rhs.routingWriteLatency
        self.routingResetLatency = rhs.routingResetLatency
        self.routingSetLatency = rhs.routingSetLatency
        self.routingRefreshLatency = rhs.routingRefreshLatency
        self.routingReadDynamicEnergy = rhs.routingReadDynamicEnergy
        self.routingWriteDynamicEnergy = rhs.routingWriteDynamicEnergy
        self.routingResetDynamicEnergy = rhs.routingResetDynamicEnergy
        self.routingSetDynamicEnergy = rhs.routingSetDynamicEnergy
        self.routingRefreshDynamicEnergy = rhs.routingRefreshDynamicEnergy
        self.routingLeakage = rhs.routingLeakage

        # Copy component objects
        self.mat = rhs.mat
        self.tsvArray = rhs.tsvArray

        return self
