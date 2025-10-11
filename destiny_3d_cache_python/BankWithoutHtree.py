#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

from Bank import Bank
from Mat import Mat
from Mux import Mux
from SenseAmp import SenseAmp
from Comparator import Comparator
from TSV import TSV
from typedef import BufferDesignTarget, MemoryType, MemCellType, WireRepeaterType, DesignTarget, CacheAccessMode
import globals as g
from constant import CONSTRAINT_ASPECT_RATIO_BANK
import math


class BankWithoutHtree(Bank):
    """Bank implementation without H-tree routing

    This class implements a bank without hierarchical tree (H-tree) routing,
    using direct wire connections between mats. This is simpler but may have
    longer wire delays for larger banks.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.invalid = False

        # Bank-specific attributes
        self.numAddressBit = 0          # Number of bank address bits
        self.numWay = 0                 # Number of ways in a mat
        self.numAddressBitRouteToMat = 0  # Number of address bits routed to mat
        self.numDataBitRouteToMat = 0   # Number of data bits routed to mat

        # Global components (used when not using internal sense amps)
        self.globalBitlineMux = Mux()
        self.globalSenseAmp = SenseAmp()
        self.globalComparator = Comparator()

    def Initialize(self, _numRowMat, _numColumnMat, _capacity, _blockSize, _associativity,
                  _numRowPerSet, _numActiveMatPerRow, _numActiveMatPerColumn, _muxSenseAmp,
                  _internalSenseAmp, _muxOutputLev1, _muxOutputLev2, _numRowSubarray,
                  _numColumnSubarray, _numActiveSubarrayPerRow, _numActiveSubarrayPerColumn,
                  _areaOptimizationLevel, _memoryType, _stackedDieCount, _partitionGranularity,
                  monolithicStackCount):
        """Initialize bank without H-tree routing

        Args:
            _numRowMat: Number of mat rows
            _numColumnMat: Number of mat columns
            _capacity: Total capacity in bits
            _blockSize: Block size in bits
            _associativity: Cache associativity
            _numRowPerSet: Number of rows per set
            _numActiveMatPerRow: Number of active mats per row
            _numActiveMatPerColumn: Number of active mats per column
            _muxSenseAmp: Mux before sense amp
            _internalSenseAmp: Whether to use internal sense amps
            _muxOutputLev1: First level output mux
            _muxOutputLev2: Second level output mux
            _numRowSubarray: Number of subarray rows
            _numColumnSubarray: Number of subarray columns
            _numActiveSubarrayPerRow: Number of active subarrays per row
            _numActiveSubarrayPerColumn: Number of active subarrays per column
            _areaOptimizationLevel: Area optimization level
            _memoryType: Type of memory (data, tag, or CAM)
            _stackedDieCount: Number of stacked dies
            _partitionGranularity: Partition granularity
            monolithicStackCount: Monolithic stack count
        """
        if self.initialized:
            # Reset the class for re-initialization
            self.initialized = False
            self.invalid = False

        if not _internalSenseAmp:
            if g.cell.memCellType == MemCellType.DRAM or g.cell.memCellType == MemCellType.eDRAM:
                self.invalid = True
                print("[BankWithoutHtree] Error: DRAM does not support external sense amplification!")
                return
            elif g.globalWire.wireRepeaterType != WireRepeaterType.repeated_none:
                self.invalid = True
                self.initialized = True
                return

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
        self.numWay = 1  # default value for non-cache design

        # Calculate the physical signals that are required in routing. Use double during the calculation to avoid overflow
        if self.stackedDieCount > 1:
            self.numAddressBit = int(math.log2(float(_capacity) / _blockSize / _associativity / _stackedDieCount) + 0.1)
        else:
            self.numAddressBit = int(math.log2(float(_capacity) / _blockSize / _associativity) + 0.1)

        if _numActiveMatPerRow > self.numColumnMat:
            print("[Bank] Warning: The number of active subarray per row is larger than the number of subarray per row!")
            print(f"{_numActiveMatPerRow} > {self.numColumnMat}")
            self.numActiveMatPerRow = self.numColumnMat
        else:
            self.numActiveMatPerRow = _numActiveMatPerRow

        if _numActiveMatPerColumn > self.numRowMat:
            print("[Bank] Warning: The number of active subarray per column is larger than the number of subarray per column!")
            print(f"{_numActiveMatPerColumn} > {self.numRowMat}")
            self.numActiveMatPerColumn = self.numRowMat
        else:
            self.numActiveMatPerColumn = _numActiveMatPerColumn

        self.muxSenseAmp = _muxSenseAmp
        self.muxOutputLev1 = _muxOutputLev1
        self.muxOutputLev2 = _muxOutputLev2

        self.numRowSubarray = _numRowSubarray
        self.numColumnSubarray = _numColumnSubarray

        if _numActiveSubarrayPerRow > self.numColumnSubarray:
            print("[Bank] Warning: The number of active subarray per row is larger than the number of subarray per row!")
            print(f"{_numActiveSubarrayPerRow} > {self.numColumnSubarray}")
            self.numActiveSubarrayPerRow = self.numColumnSubarray
        else:
            self.numActiveSubarrayPerRow = _numActiveSubarrayPerRow

        if _numActiveSubarrayPerColumn > self.numRowSubarray:
            print("[Bank] Warning: The number of active subarray per column is larger than the number of subarray per column!")
            print(f"{_numActiveSubarrayPerColumn} > {self.numRowSubarray}")
            self.numActiveSubarrayPerColumn = self.numRowSubarray
        else:
            self.numActiveSubarrayPerColumn = _numActiveSubarrayPerColumn

        # The number of address bits that are used to power gate inactive mats
        numAddressForGating = int(math.log2(self.numRowMat * self.numColumnMat / self.numActiveMatPerColumn / self.numActiveMatPerRow) + 0.1)
        self.numAddressBitRouteToMat = self.numAddressBit - numAddressForGating  # Only use the effective address bits in the following calculation
        self.numDataBitRouteToMat = _blockSize

        if self.memoryType == MemoryType.data:  # Data array
            self.numDataBitRouteToMat = _blockSize // self.numActiveMatPerColumn // self.numActiveMatPerRow
            if self.numRowPerSet > _associativity:
                # There is no enough ways to distribute into multiple rows
                self.invalid = True
                self.initialized = True
                return
            self.numWay = _associativity
            numWayPerRow = self.numWay // self.numRowPerSet  # At least 1, otherwise it is invalid, and returned already
            if numWayPerRow > 1:  # multiple ways per row, needs extra mux level
                # Do mux level recalculation to contain the multiple ways
                if g.cell.memCellType == MemCellType.DRAM or g.cell.memCellType == MemCellType.eDRAM:
                    # for DRAM, mux before sense amp has to be 1, only mux output1 and mux output2 can be used
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
            if self.numRowPerSet > 1:
                # tag array cannot have multiple rows to contain ways in a set, otherwise the bitline has to be shared
                self.invalid = True
                self.initialized = True
                return
            self.numDataBitRouteToMat = _blockSize
            self.numWay = _associativity // self.numActiveMatPerColumn // self.numActiveMatPerRow
            if self.numWay < 1:
                # This mat does not contain at least one way
                self.invalid = True
                self.initialized = True
                return
        else:  # CAM
            self.numDataBitRouteToMat = _blockSize
            self.numWay = 1

        # Initialize mat
        self.mat.Initialize(self.numRowSubarray, self.numColumnSubarray, self.numAddressBitRouteToMat,
                           self.numDataBitRouteToMat, self.numWay, self.numRowPerSet, False,
                           self.numActiveSubarrayPerRow, self.numActiveSubarrayPerColumn,
                           self.muxSenseAmp, _internalSenseAmp, self.muxOutputLev1, self.muxOutputLev2,
                           _areaOptimizationLevel, _memoryType, _stackedDieCount, _partitionGranularity,
                           monolithicStackCount)

        # Check if mat is under a legal configuration
        if self.mat.invalid:
            self.invalid = True
            self.initialized = True
            return

        self.mat.CalculateArea()

        if not _internalSenseAmp:
            voltageSense = True
            senseVoltage = g.cell.minSenseVoltage

            if g.cell.memCellType == MemCellType.SRAM:
                # SRAM, DRAM, and eDRAM all use voltage sensing
                voltageSense = True
            elif (g.cell.memCellType == MemCellType.MRAM or g.cell.memCellType == MemCellType.PCRAM or
                  g.cell.memCellType == MemCellType.memristor or g.cell.memCellType == MemCellType.FBRAM):
                voltageSense = g.cell.readMode
            else:  # NAND flash
                # TO-DO
                pass

            if self.memoryType == MemoryType.data:
                numSenseAmp = _blockSize
            else:
                numSenseAmp = _blockSize * _associativity

            self.globalSenseAmp.Initialize(numSenseAmp, not voltageSense, senseVoltage,
                                          self.mat.width * self.numColumnMat / numSenseAmp)
            if self.globalSenseAmp.invalid:
                self.invalid = True
                self.initialized = True
                return

            self.globalSenseAmp.CalculateRC()
            self.globalBitlineMux.Initialize(self.numRowMat * self.numColumnMat // self.numActiveMatPerColumn // self.numActiveMatPerRow,
                                            numSenseAmp, self.globalSenseAmp.capLoad, self.globalSenseAmp.capLoad, 0)
            self.globalBitlineMux.CalculateRC()

            if self.memoryType == MemoryType.tag:
                self.globalComparator.Initialize(_blockSize, 0)  # TO-DO: only for test

        # Reset the mux values for correct printing
        self.muxSenseAmp = _muxSenseAmp
        self.muxOutputLev1 = _muxOutputLev1
        self.muxOutputLev2 = _muxOutputLev2

        # Initialize TSV connections
        if self.stackedDieCount > 1:
            tsv_type = g.tech.WireTypeToTSVType(g.inputParameter.maxGlobalWireType)
            self.tsvArray.Initialize(tsv_type)

        self.initialized = True

    def CalculateArea(self):
        """Calculate bank area without H-tree routing"""
        if not self.initialized:
            print("[BankWithoutHtree] Error: Require initialization first!")
        elif self.invalid:
            self.height = self.width = self.area = g.invalid_value
        else:
            self.height = self.mat.height * self.numRowMat
            self.width = self.mat.width * self.numColumnMat

            numWireSharingWidth = 0
            effectivePitch = 0.0

            if g.globalWire.wireRepeaterType == WireRepeaterType.repeated_none:
                numWireSharingWidth = 1
                effectivePitch = 0  # assume that the wire is built on another metal layer, there does not cause silicon area
            else:
                numWireSharingWidth = int(math.floor(g.globalWire.repeaterSpacing / g.globalWire.repeaterHeight))
                effectivePitch = g.globalWire.repeatedWirePitch

            self.width += math.ceil(float(self.numRowMat * self.numColumnMat * self.numAddressBitRouteToMat) / numWireSharingWidth) * effectivePitch

            if not self.internalSenseAmp:
                self.globalSenseAmp.CalculateArea()
                self.height += self.globalSenseAmp.height
                self.globalBitlineMux.CalculateArea()
                self.height += self.globalBitlineMux.height
                if self.memoryType == MemoryType.tag:
                    self.globalComparator.CalculateArea()
                    self.height += self.associativity * self.globalComparator.area / self.width

            # Determine if the aspect ratio meets the constraint
            if self.memoryType == MemoryType.data:
                if self.height / self.width > CONSTRAINT_ASPECT_RATIO_BANK or self.width / self.height > CONSTRAINT_ASPECT_RATIO_BANK:
                    # illegal
                    self.invalid = True
                    self.height = self.width = self.area = g.invalid_value
                    return

            self.area = self.height * self.width

            # Initialize TSV connections
            if self.stackedDieCount > 1:
                self.tsvArray.CalculateArea()

                numControlBits = self.stackedDieCount
                numAddressBits = int(math.log2(float(self.capacity) / self.blockSize / self.associativity / self.stackedDieCount) + 0.1)
                numDataBits = self.blockSize * 2  # Read and write TSVs

                # Fine-granularity has predecoders on logic layer
                if self.partitionGranularity == 1:
                    numAddressBits = 0

                redundancyFactor = g.inputParameter.tsvRedundancy
                self.tsvArray.numTotalBits = int(float(numControlBits + numAddressBits + numDataBits) * redundancyFactor)
                self.tsvArray.numAccessBits = int(float(numControlBits + numAddressBits + self.blockSize) * redundancyFactor)

                # We're not adding in a particular dimension (width/height) so increase the total
                self.area += self.tsvArray.numTotalBits * self.tsvArray.area

    def CalculateRC(self):
        """Calculate RC parameters for the bank"""
        if not self.initialized:
            print("[BankWithoutHtree] Error: Require initialization first!")
        elif not self.invalid:
            self.mat.CalculateRC()
            if not self.internalSenseAmp:
                self.globalBitlineMux.CalculateRC()
                self.globalSenseAmp.CalculateRC()
                if self.memoryType == MemoryType.tag:
                    self.globalComparator.CalculateRC()

    def CalculateLatencyAndPower(self):
        """Calculate latency and power consumption"""
        if not self.initialized:
            print("[BankWithoutHtree] Error: Require initialization first!")
        elif self.invalid:
            self.readLatency = self.writeLatency = g.invalid_value
            self.readDynamicEnergy = self.writeDynamicEnergy = g.invalid_value
            self.leakage = g.invalid_value
        else:
            latency = 0.0
            energy = 0.0
            leakageWire = 0.0

            self.mat.CalculateLatency(g.infinite_ramp)
            self.mat.CalculatePower()
            self.readLatency = self.resetLatency = self.setLatency = self.writeLatency = 0.0
            self.refreshLatency = self.mat.refreshLatency * self.numColumnMat  # TOTAL refresh time for all Mats
            self.readDynamicEnergy = self.writeDynamicEnergy = self.resetDynamicEnergy = self.setDynamicEnergy = 0.0
            self.refreshDynamicEnergy = self.mat.refreshDynamicEnergy * self.numRowMat * self.numColumnMat
            self.leakage = 0.0

            lengthWire = self.mat.height * (self.numRowMat + 1)
            for i in range(self.numRowMat):
                lengthWire -= self.mat.height
                if self.internalSenseAmp:
                    numBitRouteToMat = 0.0
                    latency, energy, leakageWire = g.globalWire.CalculateLatencyAndPower(lengthWire)
                    if i == 0:
                        self.readLatency += latency
                        self.writeLatency += latency
                        self.refreshLatency += latency
                    if i < self.numActiveMatPerColumn:
                        if self.memoryType == MemoryType.tag:
                            numBitRouteToMat = self.numAddressBitRouteToMat + self.numDataBitRouteToMat + self.numWay
                        else:
                            numBitRouteToMat = self.numAddressBitRouteToMat + self.numDataBitRouteToMat
                        self.readDynamicEnergy += energy * numBitRouteToMat * self.numActiveMatPerRow
                        self.writeDynamicEnergy += energy * numBitRouteToMat * self.numActiveMatPerRow
                        self.refreshDynamicEnergy += energy * numBitRouteToMat * self.numActiveMatPerRow
                    self.leakage += leakageWire * numBitRouteToMat * self.numColumnMat
                else:
                    # External sense amp case
                    capBitlineMux = self.globalBitlineMux.capNMOSPassTransistor
                    resBitlineMux = self.globalBitlineMux.resNMOSPassTransistor
                    resLocalBitline = self.mat.subarray.resBitline + 3 * resBitlineMux
                    capLocalBitline = self.mat.subarray.capBitline + 6 * capBitlineMux

                    resGlobalBitline = lengthWire * g.globalWire.resWirePerUnit
                    capGlobalBitline = lengthWire * g.globalWire.capWirePerUnit
                    capGlobalBitlineMux = self.globalBitlineMux.capForPreviousDelayCalculation

                    if g.cell.memCellType == MemCellType.SRAM:
                        vpre = g.cell.readVoltage  # This value should be equal to resetVoltage and setVoltage for SRAM
                        if i == 0:
                            latency = resLocalBitline * capGlobalBitline / 2 + \
                                     (resLocalBitline + resGlobalBitline) * (capGlobalBitline / 2 + capGlobalBitlineMux)
                            latency *= math.log(vpre / (vpre - self.globalSenseAmp.senseVoltage))
                            latency += resLocalBitline * capGlobalBitline / 2
                            self.globalBitlineMux.CalculateLatency(1e20)
                            latency += self.globalBitlineMux.readLatency
                            self.globalSenseAmp.CalculateLatency(1e20)
                            self.writeLatency += latency
                            latency += self.globalSenseAmp.readLatency
                            self.readLatency += latency
                        if i < self.numActiveMatPerColumn:
                            energy = capGlobalBitline * g.tech.vdd * g.tech.vdd * self.numAddressBitRouteToMat
                            self.readDynamicEnergy += energy
                            self.writeDynamicEnergy += energy
                            self.readDynamicEnergy += capGlobalBitline * vpre * vpre * self.numWay
                            self.writeDynamicEnergy += capGlobalBitline * vpre * vpre * self.numDataBitRouteToMat

                    elif (g.cell.memCellType == MemCellType.MRAM or g.cell.memCellType == MemCellType.PCRAM or
                          g.cell.memCellType == MemCellType.memristor or g.cell.memCellType == MemCellType.FBRAM):
                        vWrite = max(abs(g.cell.resetVoltage), abs(g.cell.setVoltage))
                        vPre = self.mat.subarray.voltagePrecharge
                        vOn = self.mat.subarray.voltageMemCellOn
                        vOff = self.mat.subarray.voltageMemCellOff

                        if i == 0:
                            tau = resBitlineMux * capGlobalBitline / 2 + (resBitlineMux + resGlobalBitline) * \
                                  (capGlobalBitline + capLocalBitline) / 2 + (resBitlineMux + resGlobalBitline + \
                                  resLocalBitline) * capLocalBitline / 2
                            self.writeLatency += 0.63 * tau

                            if g.cell.readMode == False:  # current-sensing
                                # Use ICCAD 2009 model
                                resLocalBitline += self.mat.subarray.resMemCellOff
                                tau = resGlobalBitline * capGlobalBitline / 2 * \
                                      (resLocalBitline + resGlobalBitline / 3) / (resLocalBitline + resGlobalBitline)
                                self.readLatency += 0.63 * tau
                            else:  # voltage-sensing
                                if g.cell.readVoltage == 0:  # Current-in voltage sensing
                                    resLocalBitline += self.mat.subarray.resMemCellOn
                                    tau = resLocalBitline * capGlobalBitline + (resLocalBitline + resGlobalBitline) * capGlobalBitline / 2
                                    latencyOn = tau * math.log((vPre - vOn) / (vPre - vOn - self.globalSenseAmp.senseVoltage))
                                    resLocalBitline += g.cell.resistanceOff - g.cell.resistanceOn
                                    tau = resLocalBitline * capGlobalBitline + (resLocalBitline + resGlobalBitline) * capGlobalBitline / 2
                                    latencyOff = tau * math.log((vOff - vPre) / (vOff - vPre - self.globalSenseAmp.senseVoltage))
                                else:  # Voltage-in voltage sensing
                                    resLocalBitline += self.mat.subarray.resEquivalentOn
                                    tau = resLocalBitline * capGlobalBitline + (resLocalBitline + resGlobalBitline) * capGlobalBitline / 2
                                    latencyOn = tau * math.log((vPre - vOn) / (vPre - vOn - self.globalSenseAmp.senseVoltage))
                                    resLocalBitline += self.mat.subarray.resEquivalentOff - self.mat.subarray.resEquivalentOn
                                    tau = resLocalBitline * capGlobalBitline + (resLocalBitline + resGlobalBitline) * capGlobalBitline / 2
                                    latencyOff = tau * math.log((vOff - vPre) / (vOff - vPre - self.globalSenseAmp.senseVoltage))

                                self.readLatency -= self.mat.subarray.bitlineDelay
                                if (latencyOn + self.mat.subarray.bitlineDelayOn) > (latencyOff + self.mat.subarray.bitlineDelayOff):
                                    self.readLatency += latencyOn + self.mat.subarray.bitlineDelayOn
                                else:
                                    self.readLatency += latencyOff + self.mat.subarray.bitlineDelayOff

                        if i < self.numActiveMatPerColumn:
                            energy = capGlobalBitline * g.tech.vdd * g.tech.vdd * self.numAddressBitRouteToMat
                            self.readDynamicEnergy += energy
                            self.writeDynamicEnergy += energy
                            self.writeDynamicEnergy += capGlobalBitline * vWrite * vWrite * self.numDataBitRouteToMat
                            if g.cell.readMode:  # Voltage-in voltage sensing
                                self.readDynamicEnergy += capGlobalBitline * (vPre * vPre - vOn * vOn) * self.numDataBitRouteToMat

            if not self.internalSenseAmp:
                self.globalBitlineMux.CalculateLatency(1e40)
                self.globalSenseAmp.CalculateLatency(1e40)
                self.readLatency += self.globalBitlineMux.readLatency + self.globalSenseAmp.readLatency
                self.writeLatency += self.globalBitlineMux.writeLatency + self.globalSenseAmp.writeLatency
                self.globalBitlineMux.CalculatePower()
                self.globalSenseAmp.CalculatePower()
                self.readDynamicEnergy += (self.globalBitlineMux.readDynamicEnergy + self.globalSenseAmp.readDynamicEnergy) * self.numActiveMatPerRow
                self.writeDynamicEnergy += (self.globalBitlineMux.writeDynamicEnergy + self.globalSenseAmp.writeDynamicEnergy) * self.numActiveMatPerRow
                self.leakage += (self.globalBitlineMux.leakage + self.globalSenseAmp.leakage) * self.numColumnMat
                if self.memoryType == MemoryType.tag:
                    self.globalComparator.CalculateLatency(1e40)
                    self.readLatency += self.globalComparator.readLatency
                    self.globalComparator.CalculatePower()
                    self.readDynamicEnergy += self.numWay * self.globalComparator.readDynamicEnergy
                    self.leakage += self.associativity * self.globalComparator.leakage

        # only 1/A wires are activated in fast mode cache write
        if g.inputParameter.designTarget == DesignTarget.cache and g.inputParameter.cacheAccessMode == CacheAccessMode.fast_access_mode:
            self.writeDynamicEnergy /= g.inputParameter.associativity

        self.readLatency += self.mat.readLatency
        self.resetLatency = self.writeLatency + self.mat.resetLatency
        self.setLatency = self.writeLatency + self.mat.setLatency
        self.writeLatency += self.mat.writeLatency
        self.readDynamicEnergy += self.mat.readDynamicEnergy * self.numActiveMatPerRow * self.numActiveMatPerColumn
        self.cellReadEnergy = self.mat.cellReadEnergy * self.numActiveMatPerRow * self.numActiveMatPerColumn
        self.cellSetEnergy = self.mat.cellSetEnergy * self.numActiveMatPerRow * self.numActiveMatPerColumn
        self.cellResetEnergy = self.mat.cellResetEnergy * self.numActiveMatPerRow * self.numActiveMatPerColumn
        self.resetDynamicEnergy = self.writeDynamicEnergy + self.mat.resetDynamicEnergy * self.numActiveMatPerRow * self.numActiveMatPerColumn
        self.setDynamicEnergy = self.writeDynamicEnergy + self.mat.setDynamicEnergy * self.numActiveMatPerRow * self.numActiveMatPerColumn
        self.writeDynamicEnergy += self.mat.writeDynamicEnergy * self.numActiveMatPerRow * self.numActiveMatPerColumn
        self.leakage += self.mat.leakage * self.numRowMat * self.numColumnMat

        # Calculate routing metrics
        self.routingReadLatency = self.readLatency - self.mat.readLatency
        self.routingWriteLatency = self.writeLatency - self.mat.writeLatency
        self.routingResetLatency = self.resetLatency - self.mat.resetLatency
        self.routingSetLatency = self.setLatency - self.mat.setLatency
        self.routingRefreshLatency = self.refreshLatency - self.mat.refreshLatency

        self.routingReadDynamicEnergy = self.readDynamicEnergy - self.mat.readDynamicEnergy * self.numActiveMatPerColumn * self.numActiveMatPerRow
        self.routingWriteDynamicEnergy = self.writeDynamicEnergy - self.mat.writeDynamicEnergy * self.numActiveMatPerColumn * self.numActiveMatPerRow
        self.routingResetDynamicEnergy = self.resetDynamicEnergy - self.mat.resetDynamicEnergy * self.numActiveMatPerColumn * self.numActiveMatPerRow
        self.routingSetDynamicEnergy = self.setDynamicEnergy - self.mat.setDynamicEnergy * self.numActiveMatPerColumn * self.numActiveMatPerRow
        self.routingRefreshDynamicEnergy = self.refreshDynamicEnergy - self.mat.refreshDynamicEnergy * self.numActiveMatPerColumn * self.numActiveMatPerRow

        self.routingLeakage = self.leakage - self.mat.leakage * self.numColumnMat * self.numRowMat

        # For non-Htree bank, each layer contains an exact copy of this bank
        if self.initialized and not self.invalid and self.stackedDieCount > 1:
            self.leakage *= self.stackedDieCount

            # Normally senseAmpMuxLev2 is the last driver from Mat
            # or mux from global sense amp if used
            tsvReadRampInput = 1e20

            # Bank is the end unit for NVSIM, so we assume something external
            # is fully driving the input data values
            tsvWriteRampInput = g.infinite_ramp

            # Add TSV energy ~ Assume outside of bank area
            # Use comparator for tag read ramp input with internal sensing
            self.tsvArray.CalculateLatencyAndPower(tsvReadRampInput, tsvWriteRampInput)

            numControlBits = self.stackedDieCount
            numAddressBits = int(math.log2(float(self.capacity) / self.blockSize / self.associativity / self.stackedDieCount) + 0.1)
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
            self.readLatency += (self.stackedDieCount - 1) * self.tsvArray.readLatency + \
                               (self.stackedDieCount - 1) * self.tsvArray.writeLatency
            self.writeLatency += (self.stackedDieCount - 1) * self.tsvArray.writeLatency
            self.resetLatency += (self.stackedDieCount - 1) * self.tsvArray.writeLatency
            self.setLatency += (self.stackedDieCount - 1) * self.tsvArray.writeLatency
            self.refreshLatency += (self.stackedDieCount - 1) * self.tsvArray.writeLatency

            # Also assume worst energy
            self.readDynamicEnergy += self.tsvArray.numReadBits * (self.stackedDieCount - 1) * self.tsvArray.writeDynamicEnergy + \
                                     self.tsvArray.numDataBits * self.tsvArray.readDynamicEnergy * (self.stackedDieCount - 1)
            self.writeDynamicEnergy += self.tsvArray.numAccessBits * (self.stackedDieCount - 1) * self.tsvArray.writeDynamicEnergy
            self.resetDynamicEnergy += self.tsvArray.numAccessBits * (self.stackedDieCount - 1) * self.tsvArray.resetDynamicEnergy
            self.setDynamicEnergy += self.tsvArray.numAccessBits * (self.stackedDieCount - 1) * self.tsvArray.setDynamicEnergy
            self.refreshDynamicEnergy += self.tsvArray.numReadBits * (self.stackedDieCount - 1) * self.tsvArray.writeDynamicEnergy

            self.leakage += self.tsvArray.numTotalBits * (self.stackedDieCount - 1) * self.tsvArray.leakage

        if g.cell.memCellType == MemCellType.eDRAM:
            if self.refreshLatency > g.cell.retentionTime:
                self.invalid = True

    def assign(self, rhs):
        """Assignment operator to copy from another BankWithoutHtree instance

        Args:
            rhs: Another BankWithoutHtree instance to copy from

        Returns:
            self: Returns self to allow chaining
        """
        # Call parent class assignment
        # Copy all Bank properties
        self.height = rhs.height
        self.width = rhs.width
        self.area = rhs.area
        self.readLatency = rhs.readLatency
        self.writeLatency = rhs.writeLatency
        self.readDynamicEnergy = rhs.readDynamicEnergy
        self.writeDynamicEnergy = rhs.writeDynamicEnergy
        self.resetLatency = rhs.resetLatency
        self.setLatency = rhs.setLatency
        self.resetDynamicEnergy = rhs.resetDynamicEnergy
        self.setDynamicEnergy = rhs.setDynamicEnergy
        self.cellReadEnergy = rhs.cellReadEnergy
        self.cellSetEnergy = rhs.cellSetEnergy
        self.cellResetEnergy = rhs.cellResetEnergy
        self.leakage = rhs.leakage
        self.initialized = rhs.initialized
        self.invalid = rhs.invalid
        self.internalSenseAmp = rhs.internalSenseAmp
        self.numRowMat = rhs.numRowMat
        self.numColumnMat = rhs.numColumnMat
        self.capacity = rhs.capacity
        self.blockSize = rhs.blockSize
        self.associativity = rhs.associativity
        self.numRowPerSet = rhs.numRowPerSet
        self.numActiveMatPerRow = rhs.numActiveMatPerRow
        self.numActiveMatPerColumn = rhs.numActiveMatPerColumn
        self.muxSenseAmp = rhs.muxSenseAmp
        self.muxOutputLev1 = rhs.muxOutputLev1
        self.muxOutputLev2 = rhs.muxOutputLev2
        self.numRowSubarray = rhs.numRowSubarray
        self.numColumnSubarray = rhs.numColumnSubarray
        self.numActiveSubarrayPerRow = rhs.numActiveSubarrayPerRow
        self.numActiveSubarrayPerColumn = rhs.numActiveSubarrayPerColumn
        self.areaOptimizationLevel = rhs.areaOptimizationLevel
        self.memoryType = rhs.memoryType
        self.stackedDieCount = rhs.stackedDieCount
        self.partitionGranularity = rhs.partitionGranularity
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

        # Copy BankWithoutHtree-specific properties
        self.numAddressBit = rhs.numAddressBit
        self.numAddressBitRouteToMat = rhs.numAddressBitRouteToMat
        self.numDataBitRouteToMat = rhs.numDataBitRouteToMat
        self.numWay = rhs.numWay

        return self
