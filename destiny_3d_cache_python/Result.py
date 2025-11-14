#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.


# Enums from typedef.h
class MemCellType:
    SRAM = 0
    DRAM = 1
    eDRAM = 2
    MRAM = 3
    PCRAM = 4
    MEMRISTOR = 5
    FBRAM = 6
    SLCNAND = 7
    MLCNAND = 8


class WireType:
    LOCAL_AGGRESSIVE = 0
    LOCAL_CONSERVATIVE = 1
    SEMI_AGGRESSIVE = 2
    SEMI_CONSERVATIVE = 3
    GLOBAL_AGGRESSIVE = 4
    GLOBAL_CONSERVATIVE = 5
    DRAM_WORDLINE = 6


class WireRepeaterType:
    REPEATED_NONE = 0
    REPEATED_OPT = 1
    REPEATED_5 = 2
    REPEATED_10 = 3
    REPEATED_20 = 4
    REPEATED_30 = 5
    REPEATED_40 = 6
    REPEATED_50 = 7


class BufferDesignTarget:
    LATENCY_FIRST = 0
    LATENCY_AREA_TRADE_OFF = 1
    AREA_FIRST = 2


class MemoryType:
    DATA = 0
    TAG = 1
    CAM = 2


class RoutingMode:
    H_TREE = 0
    NON_H_TREE = 1


class OptimizationTarget:
    READ_LATENCY_OPTIMIZED = 0
    WRITE_LATENCY_OPTIMIZED = 1
    READ_ENERGY_OPTIMIZED = 2
    WRITE_ENERGY_OPTIMIZED = 3
    READ_EDP_OPTIMIZED = 4
    WRITE_EDP_OPTIMIZED = 5
    LEAKAGE_OPTIMIZED = 6
    AREA_OPTIMIZED = 7
    FULL_EXPLORATION = 8


class CacheAccessMode:
    NORMAL_ACCESS_MODE = 0
    SEQUENTIAL_ACCESS_MODE = 1
    FAST_ACCESS_MODE = 2


class DesignTarget:
    CACHE = 0
    RAM_CHIP = 1
    CAM_CHIP = 2


# Simple placeholder classes for bank, wire, and cell
class Bank:
    def __init__(self):
        # Organization
        self.numRowMat = 0
        self.numColumnMat = 0
        self.numActiveMatPerRow = 0
        self.numActiveMatPerColumn = 0
        self.numRowSubarray = 0
        self.numColumnSubarray = 0
        self.numActiveSubarrayPerRow = 0
        self.numActiveSubarrayPerColumn = 0
        self.stackedDieCount = 1
        self.partitionGranularity = 0

        # Mux configuration
        self.muxSenseAmp = 1
        self.muxOutputLev1 = 1
        self.muxOutputLev2 = 1
        self.numRowPerSet = 1

        # Area
        self.height = 0.0
        self.width = 0.0
        self.area = 0.0

        # Latency
        self.readLatency = 0.0
        self.writeLatency = 0.0
        self.resetLatency = 0.0
        self.setLatency = 0.0
        self.refreshLatency = 0.0
        self.routingReadLatency = 0.0
        self.routingWriteLatency = 0.0
        self.routingResetLatency = 0.0
        self.routingSetLatency = 0.0
        self.routingRefreshLatency = 0.0

        # Energy
        self.readDynamicEnergy = 0.0
        self.writeDynamicEnergy = 0.0
        self.resetDynamicEnergy = 0.0
        self.setDynamicEnergy = 0.0
        self.refreshDynamicEnergy = 0.0
        self.routingReadDynamicEnergy = 0.0
        self.routingWriteDynamicEnergy = 0.0
        self.routingResetDynamicEnergy = 0.0
        self.routingSetDynamicEnergy = 0.0
        self.routingRefreshDynamicEnergy = 0.0

        # Power
        self.leakage = 0.0
        self.routingLeakage = 0.0

        # Configuration
        self.capacity = 0
        self.blockSize = 0
        self.associativity = 1
        self.areaOptimizationLevel = BufferDesignTarget.LATENCY_FIRST
        self.memoryType = MemoryType.DATA

        # Sub-components
        self.mat = Mat()
        self.tsvArray = TSVArray()


class Mat:
    def __init__(self):
        self.height = 0.0
        self.width = 0.0
        self.area = 0.0
        self.readLatency = 0.0
        self.writeLatency = 0.0
        self.resetLatency = 0.0
        self.setLatency = 0.0
        self.refreshLatency = 0.0
        self.predecoderLatency = 0.0
        self.readDynamicEnergy = 0.0
        self.writeDynamicEnergy = 0.0
        self.resetDynamicEnergy = 0.0
        self.setDynamicEnergy = 0.0
        self.refreshDynamicEnergy = 0.0
        self.leakage = 0.0
        self.memoryType = MemoryType.DATA
        self.internalSenseAmp = False
        self.areaAllLogicBlocks = 0.0
        self.subarray = Subarray()
        self.comparator = Comparator()
        self.tsvArray = TSVArray()


class Subarray:
    def __init__(self):
        self.height = 0.0
        self.width = 0.0
        self.area = 0.0
        self.numRow = 0
        self.numColumn = 0
        self.readLatency = 0.0
        self.writeLatency = 0.0
        self.resetLatency = 0.0
        self.setLatency = 0.0
        self.refreshLatency = 0.0
        self.bitlineDelay = 0.0
        self.chargeLatency = 0.0
        self.columnDecoderLatency = 0.0
        self.readDynamicEnergy = 0.0
        self.writeDynamicEnergy = 0.0
        self.refreshDynamicEnergy = 0.0
        self.cellReadEnergy = 0.0
        self.cellResetEnergy = 0.0
        self.cellSetEnergy = 0.0
        self.leakage = 0.0
        self.rowDecoder = RowDecoder()
        self.bitlineMux = Mux()
        self.bitlineMuxDecoder = Decoder()
        self.senseAmp = SenseAmp()
        self.senseAmpMuxLev1 = Mux()
        self.senseAmpMuxLev1Decoder = Decoder()
        self.senseAmpMuxLev2 = Mux()
        self.senseAmpMuxLev2Decoder = Decoder()
        self.precharger = Precharger()


class RowDecoder:
    def __init__(self):
        self.readLatency = 0.0
        self.writeLatency = 0.0
        self.refreshLatency = 0.0
        self.readDynamicEnergy = 0.0
        self.writeDynamicEnergy = 0.0
        self.refreshDynamicEnergy = 0.0


class Mux:
    def __init__(self):
        self.readLatency = 0.0
        self.writeLatency = 0.0
        self.readDynamicEnergy = 0.0
        self.writeDynamicEnergy = 0.0


class Decoder:
    def __init__(self):
        self.readDynamicEnergy = 0.0
        self.writeDynamicEnergy = 0.0


class SenseAmp:
    def __init__(self):
        self.readLatency = 0.0
        self.readDynamicEnergy = 0.0
        self.refreshDynamicEnergy = 0.0


class Precharger:
    def __init__(self):
        self.readLatency = 0.0
        self.refreshDynamicEnergy = 0.0
        self.readDynamicEnergy = 0.0


class Comparator:
    def __init__(self):
        self.readLatency = 0.0


class TSVArray:
    def __init__(self):
        self.area = 0.0
        self.readLatency = 0.0
        self.writeLatency = 0.0
        self.resetLatency = 0.0
        self.setLatency = 0.0
        self.readDynamicEnergy = 0.0
        self.writeDynamicEnergy = 0.0
        self.resetDynamicEnergy = 0.0
        self.setDynamicEnergy = 0.0
        self.leakage = 0.0
        self.numReadBits = 0
        self.numDataBits = 0
        self.numAccessBits = 0
        self.numTotalBits = 0


class Wire:
    def __init__(self):
        self.wireType = WireType.LOCAL_AGGRESSIVE
        self.wireRepeaterType = WireRepeaterType.REPEATED_NONE
        self.isLowSwing = False


class MemCell:
    def __init__(self):
        self.memCellType = MemCellType.SRAM
        self.accessType = 0
        self.area = 0.0
        self.aspectRatio = 0.0
        self.resetPulse = 0.0
        self.setPulse = 0.0
        self.retentionTime = 0.0

    def PrintCell(self, indent=0):
        """Print cell information"""
        spaces = ' ' * indent
        print(f"{spaces}Memory Cell: ", end='')
        if self.memCellType == MemCellType.SRAM:
            print("SRAM")
        elif self.memCellType == MemCellType.DRAM:
            print("DRAM")
        elif self.memCellType == MemCellType.eDRAM:
            print("eDRAM")
        elif self.memCellType == MemCellType.MRAM:
            print("MRAM")
        elif self.memCellType == MemCellType.PCRAM:
            print("PCRAM")
        elif self.memCellType == MemCellType.MEMRISTOR:
            print("Memristor")
        elif self.memCellType == MemCellType.FBRAM:
            print("FBRAM")
        elif self.memCellType == MemCellType.SLCNAND:
            print("SLC NAND")
        elif self.memCellType == MemCellType.MLCNAND:
            print("MLC NAND")


class Tech:
    def __init__(self):
        self.featureSize = 65e-9  # 65nm default


class InputParameter:
    def __init__(self):
        self.routingMode = RoutingMode.H_TREE
        self.designTarget = DesignTarget.RAM_CHIP
        self.internalSensing = False
        self.printLevel = 0


# Global placeholder variables
INVALID_VALUE = 1e41
cell = MemCell()
tech = Tech()
inputParameter = InputParameter()


class Result:
    def __init__(self):
        # Initialize bank
        self.bank = Bank()

        # Initialize wires
        self.localWire = Wire()
        self.globalWire = Wire()

        # Initialize worst case values
        self.bank.readLatency = INVALID_VALUE
        self.bank.writeLatency = INVALID_VALUE
        self.bank.readDynamicEnergy = INVALID_VALUE
        self.bank.writeDynamicEnergy = INVALID_VALUE
        self.bank.leakage = INVALID_VALUE
        self.bank.height = INVALID_VALUE
        self.bank.width = INVALID_VALUE
        self.bank.area = INVALID_VALUE

        # Limits (no constraints by default)
        self.limitReadLatency = INVALID_VALUE
        self.limitWriteLatency = INVALID_VALUE
        self.limitReadDynamicEnergy = INVALID_VALUE
        self.limitWriteDynamicEnergy = INVALID_VALUE
        self.limitReadEdp = INVALID_VALUE
        self.limitWriteEdp = INVALID_VALUE
        self.limitArea = INVALID_VALUE
        self.limitLeakage = INVALID_VALUE

        # Default optimization target
        self.optimizationTarget = OptimizationTarget.READ_LATENCY_OPTIMIZED

        # Cell technology
        self.cellTech = None

        # Legacy attributes for compatibility with existing RunSimplifiedSimulation
        self.bank_area = 0.0
        self.total_area = 0.0
        self.bank_height = 0.0
        self.bank_width = 0.0
        self.read_latency = 0.0
        self.write_latency = 0.0
        self.set_latency = 0.0
        self.reset_latency = 0.0
        self.refresh_latency = 0.0
        self.read_dynamic_energy = 0.0
        self.write_dynamic_energy = 0.0
        self.set_dynamic_energy = 0.0
        self.reset_dynamic_energy = 0.0
        self.refresh_dynamic_energy = 0.0
        self.leakage_power = 0.0
        self.cell_area = 0.0
        self.cell_aspect_ratio = 0.0
        self.num_row_mat = 0
        self.num_column_mat = 0
        self.num_active_mat_per_row = 0
        self.num_active_mat_per_column = 0

    def reset(self):
        """Reset bank metrics to invalid values"""
        self.bank.readLatency = INVALID_VALUE
        self.bank.writeLatency = INVALID_VALUE
        self.bank.readDynamicEnergy = INVALID_VALUE
        self.bank.writeDynamicEnergy = INVALID_VALUE
        self.bank.leakage = INVALID_VALUE
        self.bank.height = INVALID_VALUE
        self.bank.width = INVALID_VALUE
        self.bank.area = INVALID_VALUE

    def printOptimizationTarget(self):
        """Return optimization target as string"""
        if self.optimizationTarget == OptimizationTarget.READ_LATENCY_OPTIMIZED:
            return "Read Latency"
        elif self.optimizationTarget == OptimizationTarget.WRITE_LATENCY_OPTIMIZED:
            return "Write Latency"
        elif self.optimizationTarget == OptimizationTarget.READ_ENERGY_OPTIMIZED:
            return "Read Energy"
        elif self.optimizationTarget == OptimizationTarget.WRITE_ENERGY_OPTIMIZED:
            return "Write Energy"
        elif self.optimizationTarget == OptimizationTarget.READ_EDP_OPTIMIZED:
            return "Read Energy-Delay-Product"
        elif self.optimizationTarget == OptimizationTarget.WRITE_EDP_OPTIMIZED:
            return "Write Energy-Delay-Product"
        elif self.optimizationTarget == OptimizationTarget.AREA_OPTIMIZED:
            return "Area"
        elif self.optimizationTarget == OptimizationTarget.LEAKAGE_OPTIMIZED:
            return "Leakage"
        else:
            return "Unknown"

    def compareAndUpdate(self, newResult):
        """Compare new result with current and update if better"""
        toUpdate = False

        # Check if new result meets all constraints
        if (newResult.bank.readLatency <= self.limitReadLatency and
            newResult.bank.writeLatency <= self.limitWriteLatency and
            newResult.bank.readDynamicEnergy <= self.limitReadDynamicEnergy and
            newResult.bank.writeDynamicEnergy <= self.limitWriteDynamicEnergy and
            newResult.bank.readLatency * newResult.bank.readDynamicEnergy <= self.limitReadEdp and
            newResult.bank.writeLatency * newResult.bank.writeDynamicEnergy <= self.limitWriteEdp and
            newResult.bank.area <= self.limitArea and
            newResult.bank.leakage <= self.limitLeakage):

            # Check optimization target
            if self.optimizationTarget == OptimizationTarget.READ_LATENCY_OPTIMIZED:
                if newResult.bank.readLatency < self.bank.readLatency:
                    toUpdate = True
            elif self.optimizationTarget == OptimizationTarget.WRITE_LATENCY_OPTIMIZED:
                if newResult.bank.writeLatency < self.bank.writeLatency:
                    toUpdate = True
            elif self.optimizationTarget == OptimizationTarget.READ_ENERGY_OPTIMIZED:
                if newResult.bank.readDynamicEnergy < self.bank.readDynamicEnergy:
                    toUpdate = True
            elif self.optimizationTarget == OptimizationTarget.WRITE_ENERGY_OPTIMIZED:
                if newResult.bank.writeDynamicEnergy < self.bank.writeDynamicEnergy:
                    toUpdate = True
            elif self.optimizationTarget == OptimizationTarget.READ_EDP_OPTIMIZED:
                if (newResult.bank.readLatency * newResult.bank.readDynamicEnergy <
                    self.bank.readLatency * self.bank.readDynamicEnergy):
                    toUpdate = True
            elif self.optimizationTarget == OptimizationTarget.WRITE_EDP_OPTIMIZED:
                if (newResult.bank.writeLatency * newResult.bank.writeDynamicEnergy <
                    self.bank.writeLatency * self.bank.writeDynamicEnergy):
                    toUpdate = True
            elif self.optimizationTarget == OptimizationTarget.AREA_OPTIMIZED:
                if newResult.bank.area < self.bank.area:
                    toUpdate = True
            elif self.optimizationTarget == OptimizationTarget.LEAKAGE_OPTIMIZED:
                if newResult.bank.leakage < self.bank.leakage:
                    toUpdate = True

            # Update if better
            if toUpdate:
                # Deep copy bank, localWire, globalWire
                self.bank = self._copyBank(newResult.bank)
                self.localWire = self._copyWire(newResult.localWire)
                self.globalWire = self._copyWire(newResult.globalWire)

        return toUpdate

    def _copyBank(self, srcBank):
        """Deep copy a Bank object"""
        # This is a simplified copy - in production would need full deep copy
        import copy
        return copy.deepcopy(srcBank)

    def _copyWire(self, srcWire):
        """Deep copy a Wire object"""
        import copy
        return copy.deepcopy(srcWire)

    def _toSecond(self, value):
        """Format time value with appropriate unit"""
        if value < 1e-9:
            return f"{value * 1e12:.3f}ps"
        elif value < 1e-6:
            return f"{value * 1e9:.3f}ns"
        elif value < 1e-3:
            return f"{value * 1e6:.3f}us"
        elif value < 1:
            return f"{value * 1e3:.3f}ms"
        else:
            return f"{value:.3f}s"

    def _toBPS(self, value):
        """Format bandwidth with appropriate unit"""
        if value < 1e3:
            return f"{value:.3f}B/s"
        elif value < 1e6:
            return f"{value / 1e3:.3f}KB/s"
        elif value < 1e9:
            return f"{value / 1e6:.3f}MB/s"
        elif value < 1e12:
            return f"{value / 1e9:.3f}GB/s"
        else:
            return f"{value / 1e12:.3f}TB/s"

    def _toJoule(self, value):
        """Format energy value with appropriate unit"""
        if value < 1e-9:
            return f"{value * 1e12:.3f}pJ"
        elif value < 1e-6:
            return f"{value * 1e9:.3f}nJ"
        elif value < 1e-3:
            return f"{value * 1e6:.3f}uJ"
        elif value < 1:
            return f"{value * 1e3:.3f}mJ"
        else:
            return f"{value:.3f}J"

    def _toWatt(self, value):
        """Format power value with appropriate unit"""
        if value < 1e-9:
            return f"{value * 1e12:.3f}pW"
        elif value < 1e-6:
            return f"{value * 1e9:.3f}nW"
        elif value < 1e-3:
            return f"{value * 1e6:.3f}uW"
        elif value < 1:
            return f"{value * 1e3:.3f}mW"
        else:
            return f"{value:.3f}W"

    def _toMeter(self, value):
        """Format length value with appropriate unit"""
        if value < 1e-9:
            return f"{value * 1e12:.3f}pm"
        elif value < 1e-6:
            return f"{value * 1e9:.3f}nm"
        elif value < 1e-3:
            return f"{value * 1e6:.3f}um"
        elif value < 1:
            return f"{value * 1e3:.3f}mm"
        else:
            return f"{value:.3f}m"

    def _toSqM(self, value):
        """Format area value with appropriate unit"""
        if value < 1e-12:
            return f"{value * 1e18:.3f}nm^2"
        elif value < 1e-6:
            return f"{value * 1e12:.3f}um^2"
        elif value < 1:
            return f"{value * 1e6:.3f}mm^2"
        else:
            return f"{value:.3f}m^2"

    def _getWireTypeName(self, wireType):
        """Get wire type name"""
        if wireType == WireType.LOCAL_AGGRESSIVE:
            return "Local Aggressive"
        elif wireType == WireType.LOCAL_CONSERVATIVE:
            return "Local Conservative"
        elif wireType == WireType.SEMI_AGGRESSIVE:
            return "Semi-Global Aggressive"
        elif wireType == WireType.SEMI_CONSERVATIVE:
            return "Semi-Global Conservative"
        elif wireType == WireType.GLOBAL_AGGRESSIVE:
            return "Global Aggressive"
        elif wireType == WireType.GLOBAL_CONSERVATIVE:
            return "Global Conservative"
        else:
            return "DRAM Wire"

    def _getRepeaterTypeName(self, repeaterType):
        """Get repeater type name"""
        if repeaterType == WireRepeaterType.REPEATED_NONE:
            return "No Repeaters"
        elif repeaterType == WireRepeaterType.REPEATED_OPT:
            return "Fully-Optimized Repeaters"
        elif repeaterType == WireRepeaterType.REPEATED_5:
            return "Repeaters with 5% Overhead"
        elif repeaterType == WireRepeaterType.REPEATED_10:
            return "Repeaters with 10% Overhead"
        elif repeaterType == WireRepeaterType.REPEATED_20:
            return "Repeaters with 20% Overhead"
        elif repeaterType == WireRepeaterType.REPEATED_30:
            return "Repeaters with 30% Overhead"
        elif repeaterType == WireRepeaterType.REPEATED_40:
            return "Repeaters with 40% Overhead"
        elif repeaterType == WireRepeaterType.REPEATED_50:
            return "Repeaters with 50% Overhead"
        else:
            return "Unknown"

    def print(self, indent=0):
        """Print comprehensive results (ported from C++)"""
        spaces = ' ' * indent

        print()
        print(f"{spaces}=============")
        print(f"{spaces}   SUMMARY   ")
        print(f"{spaces}=============")
        print(f"{spaces}Optimized for: {self.printOptimizationTarget()}")

        if self.cellTech:
            self.cellTech.PrintCell(indent)

        print()
        print(f"{spaces}=============")
        print(f"{spaces}CONFIGURATION")
        print(f"{spaces}=============")

        # Bank Organization
        if self.bank.stackedDieCount > 1:
            print(f"{spaces}Bank Organization: {self.bank.numRowMat} x {self.bank.numColumnMat} x {self.bank.stackedDieCount}")
            print(f"{spaces} - Row Activation   : {self.bank.numActiveMatPerColumn} / {self.bank.numRowMat} x 1")
            print(f"{spaces} - Column Activation: {self.bank.numActiveMatPerRow} / {self.bank.numColumnMat} x 1")
        else:
            print(f"{spaces}Bank Organization: {self.bank.numRowMat} x {self.bank.numColumnMat}")
            print(f"{spaces} - Row Activation   : {self.bank.numActiveMatPerColumn} / {self.bank.numRowMat}")
            print(f"{spaces} - Column Activation: {self.bank.numActiveMatPerRow} / {self.bank.numColumnMat}")

        # Mat Organization
        print(f"{spaces}Mat Organization: {self.bank.numRowSubarray} x {self.bank.numColumnSubarray}")
        print(f"{spaces} - Row Activation   : {self.bank.numActiveSubarrayPerColumn} / {self.bank.numRowSubarray}")
        print(f"{spaces} - Column Activation: {self.bank.numActiveSubarrayPerRow} / {self.bank.numColumnSubarray}")
        print(f"{spaces} - Subarray Size    : {self.bank.mat.subarray.numRow} Rows x {self.bank.mat.subarray.numColumn} Columns")

        # Mux Level
        print(f"{spaces}Mux Level:")
        print(f"{spaces} - Senseamp Mux      : {self.bank.muxSenseAmp}")
        print(f"{spaces} - Output Level-1 Mux: {self.bank.muxOutputLev1}")
        print(f"{spaces} - Output Level-2 Mux: {self.bank.muxOutputLev2}")

        if inputParameter.designTarget == DesignTarget.CACHE:
            print(f"{spaces} - One set is partitioned into {self.bank.numRowPerSet} rows")

        # Local Wire
        print(f"{spaces}Local Wire:")
        print(f"{spaces} - Wire Type : {self._getWireTypeName(self.localWire.wireType)}")
        print(f"{spaces} - Repeater Type: {self._getRepeaterTypeName(self.localWire.wireRepeaterType)}")
        print(f"{spaces} - Low Swing : {'Yes' if self.localWire.isLowSwing else 'No'}")

        # Global Wire
        print(f"{spaces}Global Wire:")
        print(f"{spaces} - Wire Type : {self._getWireTypeName(self.globalWire.wireType)}")
        print(f"{spaces} - Repeater Type: {self._getRepeaterTypeName(self.globalWire.wireRepeaterType)}")
        print(f"{spaces} - Low Swing : {'Yes' if self.globalWire.isLowSwing else 'No'}")

        # Buffer Design Style
        print(f"{spaces}Buffer Design Style: ", end='')
        if self.bank.areaOptimizationLevel == BufferDesignTarget.LATENCY_FIRST:
            print("Latency-Optimized")
        elif self.bank.areaOptimizationLevel == BufferDesignTarget.AREA_FIRST:
            print("Area-Optimized")
        else:
            print("Balanced")

        print(f"{spaces}=============")
        print(f"{spaces}   RESULT")
        print(f"{spaces}=============")

        # Area
        print(f"{spaces}Area:")
        cellAreaEfficiency = (cell.area * tech.featureSize * tech.featureSize *
                            self.bank.capacity / self.bank.numRowMat / self.bank.numColumnMat /
                            self.bank.mat.area * 100) if self.bank.mat.area > 0 else 0
        subarrayAreaEfficiency = (cell.area * tech.featureSize * tech.featureSize *
                                self.bank.capacity / self.bank.numRowMat / self.bank.numColumnMat /
                                self.bank.numRowSubarray / self.bank.numColumnSubarray /
                                self.bank.mat.subarray.area * 100) if self.bank.mat.subarray.area > 0 else 0

        print(f"{spaces} - Total Area = {self._toMeter(self.bank.height)} x {self._toMeter(self.bank.width)} = {self._toSqM(self.bank.area)}")
        print(f"{spaces} |--- Mat Area      = {self._toMeter(self.bank.mat.height)} x {self._toMeter(self.bank.mat.width)} = {self._toSqM(self.bank.mat.area)}   ({cellAreaEfficiency:.1f}%)")
        print(f"{spaces} |--- Subarray Area = {self._toMeter(self.bank.mat.subarray.height)} x {self._toMeter(self.bank.mat.subarray.width)} = {self._toSqM(self.bank.mat.subarray.area)}   ({subarrayAreaEfficiency:.1f}%)")

        # TSV Area for 3D stacking
        if self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 0:
            print(f"{spaces} |--- TSV Area      = {self._toSqM(self.bank.tsvArray.area)}")
        elif self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 1:
            totalTSVArea = (self.bank.tsvArray.area +
                          self.bank.mat.tsvArray.area * self.bank.numColumnMat * self.bank.numRowMat)
            areaLogicLayer = (self.bank.mat.areaAllLogicBlocks *
                            self.bank.numColumnMat * self.bank.numRowMat)
            print(f"{spaces} |--- TSV Area      = {self._toSqM(totalTSVArea)}")
            print(f"{spaces} |--- Logic Layer Area = {self._toSqM(areaLogicLayer)}")

        areaEfficiency = (cell.area * tech.featureSize * tech.featureSize *
                        self.bank.capacity / self.bank.area * 100) if self.bank.area > 0 else 0
        print(f"{spaces} - Area Efficiency = {areaEfficiency:.1f}%")

        # Timing
        print(f"{spaces}Timing:")
        print(f"{spaces} -  Read Latency = {self._toSecond(self.bank.readLatency)}")

        # TSV Latency for 3D
        if self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 0:
            totalTSVLatency = ((self.bank.tsvArray.readLatency + self.bank.tsvArray.writeLatency) *
                             (self.bank.stackedDieCount - 1))
            print(f"{spaces} |--- TSV Latency    = {self._toSecond(totalTSVLatency)}")
        elif self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 1:
            totalTSVLatency = (self.bank.tsvArray.readLatency * (self.bank.stackedDieCount - 1) +
                             self.bank.mat.tsvArray.writeLatency * (self.bank.stackedDieCount - 1))
            print(f"{spaces} |--- TSV Latency    = {self._toSecond(totalTSVLatency)}")

        # Routing latency
        if inputParameter.routingMode == RoutingMode.H_TREE:
            print(f"{spaces} |--- H-Tree Latency = {self._toSecond(self.bank.routingReadLatency)}")
        else:
            print(f"{spaces} |--- Non-H-Tree Latency = {self._toSecond(self.bank.routingReadLatency)}")

        print(f"{spaces} |--- Mat Latency    = {self._toSecond(self.bank.mat.readLatency)}")
        print(f"{spaces}    |--- Predecoder Latency = {self._toSecond(self.bank.mat.predecoderLatency)}")
        print(f"{spaces}    |--- Subarray Latency   = {self._toSecond(self.bank.mat.subarray.readLatency)}")
        print(f"{spaces}       |--- Row Decoder Latency = {self._toSecond(self.bank.mat.subarray.rowDecoder.readLatency)}")
        print(f"{spaces}       |--- Bitline Latency     = {self._toSecond(self.bank.mat.subarray.bitlineDelay)}")

        if inputParameter.internalSensing:
            print(f"{spaces}       |--- Senseamp Latency    = {self._toSecond(self.bank.mat.subarray.senseAmp.readLatency)}")

        muxLatency = (self.bank.mat.subarray.bitlineMux.readLatency +
                     self.bank.mat.subarray.senseAmpMuxLev1.readLatency +
                     self.bank.mat.subarray.senseAmpMuxLev2.readLatency)
        print(f"{spaces}       |--- Mux Latency         = {self._toSecond(muxLatency)}")
        print(f"{spaces}       |--- Precharge Latency   = {self._toSecond(self.bank.mat.subarray.precharger.readLatency)}")

        if self.bank.mat.memoryType == MemoryType.TAG and self.bank.mat.internalSenseAmp:
            print(f"{spaces}    |--- Comparator Latency  = {self._toSecond(self.bank.mat.comparator.readLatency)}")

        # Write/Reset/Set Latency based on cell type
        if (cell.memCellType == MemCellType.PCRAM or cell.memCellType == MemCellType.FBRAM or
            cell.memCellType == MemCellType.MEMRISTOR):
            self._printResetSetLatency(spaces)
        elif cell.memCellType == MemCellType.SLCNAND:
            self._printEraseProgLatency(spaces)
        else:
            self._printWriteLatency(spaces)

        # Refresh latency for eDRAM
        if cell.memCellType == MemCellType.eDRAM:
            self._printRefreshLatency(spaces)

        # Bandwidth
        if self.bank.mat.subarray.readLatency > self.bank.mat.subarray.rowDecoder.readLatency:
            readBandwidth = (self.bank.blockSize /
                           (self.bank.mat.subarray.readLatency -
                            self.bank.mat.subarray.rowDecoder.readLatency +
                            self.bank.mat.subarray.precharger.readLatency) / 8)
        else:
            readBandwidth = 0

        writeBandwidth = (self.bank.blockSize / self.bank.mat.subarray.writeLatency / 8) if self.bank.mat.subarray.writeLatency > 0 else 0

        print(f"{spaces} - Read Bandwidth  = {self._toBPS(readBandwidth)}")
        print(f"{spaces} - Write Bandwidth = {self._toBPS(writeBandwidth)}")

        # Power
        print(f"{spaces}Power:")
        self._printReadDynamicEnergy(spaces)

        # Write/Reset/Set Energy based on cell type
        if (cell.memCellType == MemCellType.PCRAM or cell.memCellType == MemCellType.FBRAM or
            cell.memCellType == MemCellType.MEMRISTOR):
            self._printResetSetDynamicEnergy(spaces)
        elif cell.memCellType == MemCellType.SLCNAND:
            self._printEraseProgDynamicEnergy(spaces)
        else:
            self._printWriteDynamicEnergy(spaces)

        # Refresh energy for eDRAM
        if cell.memCellType == MemCellType.eDRAM:
            self._printRefreshDynamicEnergy(spaces)

        # Leakage
        print(f"{spaces} - Leakage Power = {self._toWatt(self.bank.leakage)}")

        # TSV Leakage for 3D
        if self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 0:
            tsvLeakage = self.bank.tsvArray.leakage * (self.bank.stackedDieCount - 1) * self.bank.tsvArray.numTotalBits
            print(f"{spaces} |--- TSV Leakage              = {self._toWatt(tsvLeakage)}")
        elif self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 1:
            tsvLeakage = (self.bank.tsvArray.leakage * (self.bank.stackedDieCount - 1) * self.bank.tsvArray.numTotalBits +
                        self.bank.mat.tsvArray.leakage * self.bank.numColumnMat * self.bank.numRowMat *
                        self.bank.mat.tsvArray.numTotalBits)
            print(f"{spaces} |--- TSV Leakage              = {self._toWatt(tsvLeakage)}")

        if inputParameter.routingMode == RoutingMode.H_TREE:
            print(f"{spaces} |--- H-Tree Leakage Power     = {self._toWatt(self.bank.routingLeakage)}")
        else:
            print(f"{spaces} |--- Non-H-Tree Leakage Power = {self._toWatt(self.bank.routingLeakage)}")

        print(f"{spaces} |--- Mat Leakage Power        = {self._toWatt(self.bank.mat.leakage)} per mat")

        # Refresh power for eDRAM
        if cell.memCellType == MemCellType.eDRAM and cell.retentionTime > 0:
            refreshPower = self.bank.refreshDynamicEnergy / cell.retentionTime
            print(f"{spaces} - Refresh Power = {self._toWatt(refreshPower)}")

    def _printWriteLatency(self, spaces):
        """Print write latency breakdown"""
        print(f"{spaces} - Write Latency = {self._toSecond(self.bank.writeLatency)}")

        if self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 0:
            print(f"{spaces} |--- TSV Latency    = {self._toSecond(self.bank.tsvArray.writeLatency * (self.bank.stackedDieCount - 1))}")
        elif self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 1:
            print(f"{spaces} |--- TSV Latency    = {self._toSecond(self.bank.tsvArray.writeLatency * (self.bank.stackedDieCount - 1))}")

        if inputParameter.routingMode == RoutingMode.H_TREE:
            print(f"{spaces} |--- H-Tree Latency = {self._toSecond(self.bank.routingWriteLatency)}")
        else:
            print(f"{spaces} |--- Non-H-Tree Latency = {self._toSecond(self.bank.routingWriteLatency)}")

        print(f"{spaces} |--- Mat Latency    = {self._toSecond(self.bank.mat.writeLatency)}")
        print(f"{spaces}    |--- Predecoder Latency = {self._toSecond(self.bank.mat.predecoderLatency)}")
        print(f"{spaces}    |--- Subarray Latency   = {self._toSecond(self.bank.mat.subarray.writeLatency)}")

        if cell.memCellType == MemCellType.MRAM:
            print(f"{spaces}       |--- Write Pulse Duration = {self._toSecond(cell.resetPulse)}")

        print(f"{spaces}       |--- Row Decoder Latency = {self._toSecond(self.bank.mat.subarray.rowDecoder.writeLatency)}")
        print(f"{spaces}       |--- Charge Latency      = {self._toSecond(self.bank.mat.subarray.chargeLatency)}")

    def _printResetSetLatency(self, spaces):
        """Print RESET/SET latency for PCRAM/FBRAM/Memristor"""
        print(f"{spaces} - RESET Latency = {self._toSecond(self.bank.resetLatency)}")

        if self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 0:
            print(f"{spaces} |--- TSV Latency    = {self._toSecond(self.bank.tsvArray.resetLatency * (self.bank.stackedDieCount - 1))}")
        elif self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 1:
            print(f"{spaces} |--- TSV Latency    = {self._toSecond(self.bank.tsvArray.resetLatency * (self.bank.stackedDieCount - 1))}")

        if inputParameter.routingMode == RoutingMode.H_TREE:
            print(f"{spaces} |--- H-Tree Latency = {self._toSecond(self.bank.routingResetLatency)}")
        else:
            print(f"{spaces} |--- Non-H-Tree Latency = {self._toSecond(self.bank.routingResetLatency)}")

        print(f"{spaces} |--- Mat Latency    = {self._toSecond(self.bank.mat.resetLatency)}")
        print(f"{spaces}    |--- Predecoder Latency = {self._toSecond(self.bank.mat.predecoderLatency)}")
        print(f"{spaces}    |--- Subarray Latency   = {self._toSecond(self.bank.mat.subarray.resetLatency)}")
        print(f"{spaces}       |--- RESET Pulse Duration = {self._toSecond(cell.resetPulse)}")
        print(f"{spaces}       |--- Row Decoder Latency  = {self._toSecond(self.bank.mat.subarray.rowDecoder.writeLatency)}")
        print(f"{spaces}       |--- Charge Latency   = {self._toSecond(self.bank.mat.subarray.chargeLatency)}")

        print(f"{spaces} - SET Latency   = {self._toSecond(self.bank.setLatency)}")

        if self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 0:
            print(f"{spaces} |--- TSV Latency    = {self._toSecond(self.bank.tsvArray.setLatency * (self.bank.stackedDieCount - 1))}")
        elif self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 1:
            print(f"{spaces} |--- TSV Latency    = {self._toSecond(self.bank.tsvArray.setLatency * (self.bank.stackedDieCount - 1))}")

        if inputParameter.routingMode == RoutingMode.H_TREE:
            print(f"{spaces} |--- H-Tree Latency = {self._toSecond(self.bank.routingSetLatency)}")
        else:
            print(f"{spaces} |--- Non-H-Tree Latency = {self._toSecond(self.bank.routingSetLatency)}")

        print(f"{spaces} |--- Mat Latency    = {self._toSecond(self.bank.mat.setLatency)}")
        print(f"{spaces}    |--- Predecoder Latency = {self._toSecond(self.bank.mat.predecoderLatency)}")
        print(f"{spaces}    |--- Subarray Latency   = {self._toSecond(self.bank.mat.subarray.setLatency)}")
        print(f"{spaces}       |--- SET Pulse Duration   = {self._toSecond(cell.setPulse)}")
        print(f"{spaces}       |--- Row Decoder Latency  = {self._toSecond(self.bank.mat.subarray.rowDecoder.writeLatency)}")
        print(f"{spaces}       |--- Charger Latency      = {self._toSecond(self.bank.mat.subarray.chargeLatency)}")

    def _printEraseProgLatency(self, spaces):
        """Print Erase/Programming latency for NAND"""
        print(f"{spaces} - Erase Latency = {self._toSecond(self.bank.resetLatency)}")

        if self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 0:
            print(f"{spaces} |--- TSV Latency    = {self._toSecond(self.bank.tsvArray.resetLatency * (self.bank.stackedDieCount - 1))}")
        elif self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 1:
            print(f"{spaces} |--- TSV Latency    = {self._toSecond(self.bank.tsvArray.resetLatency * (self.bank.stackedDieCount - 1))}")

        if inputParameter.routingMode == RoutingMode.H_TREE:
            print(f"{spaces} |--- H-Tree Latency = {self._toSecond(self.bank.routingResetLatency)}")
        else:
            print(f"{spaces} |--- Non-H-Tree Latency = {self._toSecond(self.bank.routingResetLatency)}")

        print(f"{spaces} |--- Mat Latency    = {self._toSecond(self.bank.mat.resetLatency)}")

        print(f"{spaces} - Programming Latency   = {self._toSecond(self.bank.setLatency)}")

        if inputParameter.routingMode == RoutingMode.H_TREE:
            print(f"{spaces} |--- H-Tree Latency = {self._toSecond(self.bank.routingSetLatency)}")
        else:
            print(f"{spaces} |--- Non-H-Tree Latency = {self._toSecond(self.bank.routingSetLatency)}")

        print(f"{spaces} |--- Mat Latency    = {self._toSecond(self.bank.mat.setLatency)}")

    def _printRefreshLatency(self, spaces):
        """Print refresh latency for eDRAM"""
        print(f"{spaces} - Refresh Latency = {self._toSecond(self.bank.refreshLatency)}")

        if self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 0:
            print(f"{spaces} |--- TSV Latency    = {self._toSecond(self.bank.tsvArray.writeLatency * (self.bank.stackedDieCount - 1))}")
        elif self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 1:
            print(f"{spaces} |--- TSV Latency    = {self._toSecond(self.bank.tsvArray.writeLatency * (self.bank.stackedDieCount - 1))}")

        if inputParameter.routingMode == RoutingMode.H_TREE:
            print(f"{spaces} |--- H-Tree Latency = {self._toSecond(self.bank.routingRefreshLatency)}")
        else:
            print(f"{spaces} |--- Non-H-Tree Latency = {self._toSecond(self.bank.routingRefreshLatency)}")

        print(f"{spaces} |--- Mat Latency    = {self._toSecond(self.bank.mat.refreshLatency)}")
        print(f"{spaces}    |--- Predecoder Latency = {self._toSecond(self.bank.mat.predecoderLatency)}")
        print(f"{spaces}    |--- Subarray Latency   = {self._toSecond(self.bank.mat.subarray.refreshLatency)}")

    def _printReadDynamicEnergy(self, spaces):
        """Print read dynamic energy breakdown"""
        print(f"{spaces} -  Read Dynamic Energy = {self._toJoule(self.bank.readDynamicEnergy)}")

        if self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 0:
            tsvEnergy = (self.bank.tsvArray.writeDynamicEnergy * (self.bank.stackedDieCount - 1) *
                        self.bank.tsvArray.numReadBits +
                        self.bank.tsvArray.readDynamicEnergy * self.bank.tsvArray.numDataBits *
                        (self.bank.stackedDieCount - 1))
            print(f"{spaces} |--- TSV Dynamic Energy    = {self._toJoule(tsvEnergy)}")
        elif self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 1:
            tsvEnergy = (self.bank.mat.tsvArray.writeDynamicEnergy * (self.bank.stackedDieCount - 1) *
                        self.bank.mat.tsvArray.numAccessBits +
                        self.bank.tsvArray.writeDynamicEnergy * (self.bank.stackedDieCount - 1) *
                        self.bank.stackedDieCount +
                        self.bank.tsvArray.readDynamicEnergy * self.bank.blockSize *
                        (self.bank.stackedDieCount - 1))
            print(f"{spaces} |--- TSV Dynamic Energy    = {self._toJoule(tsvEnergy)}")

        if inputParameter.routingMode == RoutingMode.H_TREE:
            print(f"{spaces} |--- H-Tree Dynamic Energy = {self._toJoule(self.bank.routingReadDynamicEnergy)}")
        else:
            print(f"{spaces} |--- Non-H-Tree Dynamic Energy = {self._toJoule(self.bank.routingReadDynamicEnergy)}")

        print(f"{spaces} |--- Mat Dynamic Energy    = {self._toJoule(self.bank.mat.readDynamicEnergy)} per mat")

        predecoderEnergy = (self.bank.mat.readDynamicEnergy -
                          self.bank.mat.subarray.readDynamicEnergy *
                          self.bank.numActiveSubarrayPerRow *
                          self.bank.numActiveSubarrayPerColumn)
        print(f"{spaces}    |--- Predecoder Dynamic Energy = {self._toJoule(predecoderEnergy)}")
        print(f"{spaces}    |--- Subarray Dynamic Energy   = {self._toJoule(self.bank.mat.subarray.readDynamicEnergy)} per active subarray")
        print(f"{spaces}       |--- Row Decoder Dynamic Energy = {self._toJoule(self.bank.mat.subarray.rowDecoder.readDynamicEnergy)}")

        muxDecoderEnergy = (self.bank.mat.subarray.bitlineMuxDecoder.readDynamicEnergy +
                          self.bank.mat.subarray.senseAmpMuxLev1Decoder.readDynamicEnergy +
                          self.bank.mat.subarray.senseAmpMuxLev2Decoder.readDynamicEnergy)
        print(f"{spaces}       |--- Mux Decoder Dynamic Energy = {self._toJoule(muxDecoderEnergy)}")

        if (cell.memCellType == MemCellType.PCRAM or cell.memCellType == MemCellType.FBRAM or
            cell.memCellType == MemCellType.MRAM or cell.memCellType == MemCellType.MEMRISTOR):
            print(f"{spaces}       |--- Bitline & Cell Read Energy = {self._toJoule(self.bank.mat.subarray.cellReadEnergy)}")

        if inputParameter.internalSensing:
            print(f"{spaces}       |--- Senseamp Dynamic Energy    = {self._toJoule(self.bank.mat.subarray.senseAmp.readDynamicEnergy)}")

        muxEnergy = (self.bank.mat.subarray.bitlineMux.readDynamicEnergy +
                    self.bank.mat.subarray.senseAmpMuxLev1.readDynamicEnergy +
                    self.bank.mat.subarray.senseAmpMuxLev2.readDynamicEnergy)
        print(f"{spaces}       |--- Mux Dynamic Energy         = {self._toJoule(muxEnergy)}")
        print(f"{spaces}       |--- Precharge Dynamic Energy   = {self._toJoule(self.bank.mat.subarray.precharger.readDynamicEnergy)}")

    def _printWriteDynamicEnergy(self, spaces):
        """Print write dynamic energy breakdown"""
        print(f"{spaces} - Write Dynamic Energy = {self._toJoule(self.bank.writeDynamicEnergy)}")

        if self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 0:
            tsvEnergy = self.bank.tsvArray.writeDynamicEnergy * (self.bank.stackedDieCount - 1) * self.bank.tsvArray.numAccessBits
            print(f"{spaces} |--- TSV Dynamic Energy    = {self._toJoule(tsvEnergy)}")
        elif self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 1:
            tsvEnergy = self.bank.tsvArray.writeDynamicEnergy * (self.bank.stackedDieCount - 1) * self.bank.tsvArray.numAccessBits
            print(f"{spaces} |--- TSV Dynamic Energy    = {self._toJoule(tsvEnergy)}")

        if inputParameter.routingMode == RoutingMode.H_TREE:
            print(f"{spaces} |--- H-Tree Dynamic Energy = {self._toJoule(self.bank.routingWriteDynamicEnergy)}")
        else:
            print(f"{spaces} |--- Non-H-Tree Dynamic Energy = {self._toJoule(self.bank.routingWriteDynamicEnergy)}")

        print(f"{spaces} |--- Mat Dynamic Energy    = {self._toJoule(self.bank.mat.writeDynamicEnergy)} per mat")

        predecoderEnergy = (self.bank.mat.writeDynamicEnergy -
                          self.bank.mat.subarray.writeDynamicEnergy *
                          self.bank.numActiveSubarrayPerRow *
                          self.bank.numActiveSubarrayPerColumn)
        print(f"{spaces}    |--- Predecoder Dynamic Energy = {self._toJoule(predecoderEnergy)}")
        print(f"{spaces}    |--- Subarray Dynamic Energy   = {self._toJoule(self.bank.mat.subarray.writeDynamicEnergy)} per active subarray")
        print(f"{spaces}       |--- Row Decoder Dynamic Energy = {self._toJoule(self.bank.mat.subarray.rowDecoder.writeDynamicEnergy)}")

        muxDecoderEnergy = (self.bank.mat.subarray.bitlineMuxDecoder.writeDynamicEnergy +
                          self.bank.mat.subarray.senseAmpMuxLev1Decoder.writeDynamicEnergy +
                          self.bank.mat.subarray.senseAmpMuxLev2Decoder.writeDynamicEnergy)
        print(f"{spaces}       |--- Mux Decoder Dynamic Energy = {self._toJoule(muxDecoderEnergy)}")

        muxEnergy = (self.bank.mat.subarray.bitlineMux.writeDynamicEnergy +
                    self.bank.mat.subarray.senseAmpMuxLev1.writeDynamicEnergy +
                    self.bank.mat.subarray.senseAmpMuxLev2.writeDynamicEnergy)
        print(f"{spaces}       |--- Mux Dynamic Energy         = {self._toJoule(muxEnergy)}")

        if cell.memCellType == MemCellType.MRAM:
            print(f"{spaces}       |--- Bitline & Cell Write Energy= {self._toJoule(self.bank.mat.subarray.cellResetEnergy)}")

    def _printResetSetDynamicEnergy(self, spaces):
        """Print RESET/SET dynamic energy for PCRAM/FBRAM/Memristor"""
        print(f"{spaces} - RESET Dynamic Energy = {self._toJoule(self.bank.resetDynamicEnergy)}")

        if self.bank.stackedDieCount > 1 and self.bank.partitionGranularity in [0, 1]:
            tsvEnergy = self.bank.tsvArray.resetDynamicEnergy * (self.bank.stackedDieCount - 1) * self.bank.tsvArray.numAccessBits
            print(f"{spaces} |--- TSV Dynamic Energy    = {self._toJoule(tsvEnergy)}")

        if inputParameter.routingMode == RoutingMode.H_TREE:
            print(f"{spaces} |--- H-Tree Dynamic Energy = {self._toJoule(self.bank.routingResetDynamicEnergy)}")
        else:
            print(f"{spaces} |--- H-Tree Dynamic Energy = {self._toJoule(self.bank.routingResetDynamicEnergy)}")

        print(f"{spaces} |--- Mat Dynamic Energy    = {self._toJoule(self.bank.mat.resetDynamicEnergy)} per mat")

        predecoderEnergy = (self.bank.mat.writeDynamicEnergy -
                          self.bank.mat.subarray.writeDynamicEnergy *
                          self.bank.numActiveSubarrayPerRow *
                          self.bank.numActiveSubarrayPerColumn)
        print(f"{spaces}    |--- Predecoder Dynamic Energy = {self._toJoule(predecoderEnergy)}")
        print(f"{spaces}    |--- Subarray Dynamic Energy   = {self._toJoule(self.bank.mat.subarray.writeDynamicEnergy)} per active subarray")
        print(f"{spaces}       |--- Row Decoder Dynamic Energy = {self._toJoule(self.bank.mat.subarray.rowDecoder.writeDynamicEnergy)}")

        muxDecoderEnergy = (self.bank.mat.subarray.bitlineMuxDecoder.writeDynamicEnergy +
                          self.bank.mat.subarray.senseAmpMuxLev1Decoder.writeDynamicEnergy +
                          self.bank.mat.subarray.senseAmpMuxLev2Decoder.writeDynamicEnergy)
        print(f"{spaces}       |--- Mux Decoder Dynamic Energy = {self._toJoule(muxDecoderEnergy)}")

        muxEnergy = (self.bank.mat.subarray.bitlineMux.writeDynamicEnergy +
                    self.bank.mat.subarray.senseAmpMuxLev1.writeDynamicEnergy +
                    self.bank.mat.subarray.senseAmpMuxLev2.writeDynamicEnergy)
        print(f"{spaces}       |--- Mux Dynamic Energy         = {self._toJoule(muxEnergy)}")
        print(f"{spaces}       |--- Cell RESET Dynamic Energy  = {self._toJoule(self.bank.mat.subarray.cellResetEnergy)}")

        print(f"{spaces} - SET Dynamic Energy = {self._toJoule(self.bank.setDynamicEnergy)}")

        if inputParameter.routingMode == RoutingMode.H_TREE:
            print(f"{spaces} |--- H-Tree Dynamic Energy = {self._toJoule(self.bank.routingSetDynamicEnergy)}")
        else:
            print(f"{spaces} |--- Non-H-Tree Dynamic Energy = {self._toJoule(self.bank.routingSetDynamicEnergy)}")

        print(f"{spaces} |--- Mat Dynamic Energy    = {self._toJoule(self.bank.mat.setDynamicEnergy)} per mat")
        print(f"{spaces}    |--- Predecoder Dynamic Energy = {self._toJoule(predecoderEnergy)}")
        print(f"{spaces}    |--- Subarray Dynamic Energy   = {self._toJoule(self.bank.mat.subarray.writeDynamicEnergy)} per active subarray")
        print(f"{spaces}       |--- Row Decoder Dynamic Energy = {self._toJoule(self.bank.mat.subarray.rowDecoder.writeDynamicEnergy)}")
        print(f"{spaces}       |--- Mux Decoder Dynamic Energy = {self._toJoule(muxDecoderEnergy)}")
        print(f"{spaces}       |--- Mux Dynamic Energy         = {self._toJoule(muxEnergy)}")
        print(f"{spaces}       |--- Cell SET Dynamic Energy    = {self._toJoule(self.bank.mat.subarray.cellSetEnergy)}")

    def _printEraseProgDynamicEnergy(self, spaces):
        """Print Erase/Programming dynamic energy for NAND"""
        print(f"{spaces} - Erase Dynamic Energy = {self._toJoule(self.bank.resetDynamicEnergy)} per block")

        if self.bank.stackedDieCount > 1 and self.bank.partitionGranularity in [0, 1]:
            tsvEnergy = self.bank.tsvArray.resetDynamicEnergy * (self.bank.stackedDieCount - 1) * self.bank.tsvArray.numAccessBits
            print(f"{spaces} |--- TSV Dynamic Energy    = {self._toJoule(tsvEnergy)}")

        if inputParameter.routingMode == RoutingMode.H_TREE:
            print(f"{spaces} |--- H-Tree Dynamic Energy = {self._toJoule(self.bank.routingResetDynamicEnergy)}")
        else:
            print(f"{spaces} |--- Non-H-Tree Dynamic Energy = {self._toJoule(self.bank.routingResetDynamicEnergy)}")

        print(f"{spaces} |--- Mat Dynamic Energy    = {self._toJoule(self.bank.mat.resetDynamicEnergy)} per mat")

        predecoderEnergy = (self.bank.mat.writeDynamicEnergy -
                          self.bank.mat.subarray.writeDynamicEnergy *
                          self.bank.numActiveSubarrayPerRow *
                          self.bank.numActiveSubarrayPerColumn)
        print(f"{spaces}    |--- Predecoder Dynamic Energy = {self._toJoule(predecoderEnergy)}")
        print(f"{spaces}    |--- Subarray Dynamic Energy   = {self._toJoule(self.bank.mat.subarray.writeDynamicEnergy)} per active subarray")
        print(f"{spaces}       |--- Row Decoder Dynamic Energy = {self._toJoule(self.bank.mat.subarray.rowDecoder.writeDynamicEnergy)}")

        muxDecoderEnergy = (self.bank.mat.subarray.bitlineMuxDecoder.writeDynamicEnergy +
                          self.bank.mat.subarray.senseAmpMuxLev1Decoder.writeDynamicEnergy +
                          self.bank.mat.subarray.senseAmpMuxLev2Decoder.writeDynamicEnergy)
        print(f"{spaces}       |--- Mux Decoder Dynamic Energy = {self._toJoule(muxDecoderEnergy)}")

        muxEnergy = (self.bank.mat.subarray.bitlineMux.writeDynamicEnergy +
                    self.bank.mat.subarray.senseAmpMuxLev1.writeDynamicEnergy +
                    self.bank.mat.subarray.senseAmpMuxLev2.writeDynamicEnergy)
        print(f"{spaces}       |--- Mux Dynamic Energy         = {self._toJoule(muxEnergy)}")

        print(f"{spaces} - Programming Dynamic Energy = {self._toJoule(self.bank.setDynamicEnergy)} per page")

        if inputParameter.routingMode == RoutingMode.H_TREE:
            print(f"{spaces} |--- H-Tree Dynamic Energy = {self._toJoule(self.bank.routingSetDynamicEnergy)}")
        else:
            print(f"{spaces} |--- Non-H-Tree Dynamic Energy = {self._toJoule(self.bank.routingSetDynamicEnergy)}")

        print(f"{spaces} |--- Mat Dynamic Energy    = {self._toJoule(self.bank.mat.setDynamicEnergy)} per mat")
        print(f"{spaces}    |--- Predecoder Dynamic Energy = {self._toJoule(predecoderEnergy)}")
        print(f"{spaces}    |--- Subarray Dynamic Energy   = {self._toJoule(self.bank.mat.subarray.writeDynamicEnergy)} per active subarray")
        print(f"{spaces}       |--- Row Decoder Dynamic Energy = {self._toJoule(self.bank.mat.subarray.rowDecoder.writeDynamicEnergy)}")
        print(f"{spaces}       |--- Mux Decoder Dynamic Energy = {self._toJoule(muxDecoderEnergy)}")
        print(f"{spaces}       |--- Mux Dynamic Energy         = {self._toJoule(muxEnergy)}")

    def _printRefreshDynamicEnergy(self, spaces):
        """Print refresh dynamic energy for eDRAM"""
        print(f"{spaces} - Refresh Dynamic Energy = {self._toJoule(self.bank.refreshDynamicEnergy)}")

        if self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 0:
            tsvEnergy = self.bank.tsvArray.writeDynamicEnergy * (self.bank.stackedDieCount - 1) * self.bank.tsvArray.numReadBits
            print(f"{spaces} |--- TSV Dynamic Energy    = {self._toJoule(tsvEnergy)}")
        elif self.bank.stackedDieCount > 1 and self.bank.partitionGranularity == 1:
            tsvEnergy = self.bank.tsvArray.writeDynamicEnergy * (self.bank.stackedDieCount - 1) * self.bank.tsvArray.numReadBits
            print(f"{spaces} |--- TSV Dynamic Energy    = {self._toJoule(tsvEnergy)}")

        if inputParameter.routingMode == RoutingMode.H_TREE:
            print(f"{spaces} |--- H-Tree Dynamic Energy = {self._toJoule(self.bank.routingRefreshDynamicEnergy)}")
        else:
            print(f"{spaces} |--- Non-H-Tree Dynamic Energy = {self._toJoule(self.bank.routingRefreshDynamicEnergy)}")

        print(f"{spaces} |--- Mat Dynamic Energy    = {self._toJoule(self.bank.mat.refreshDynamicEnergy)} per mat")

        predecoderEnergy = (self.bank.mat.refreshDynamicEnergy -
                          self.bank.mat.subarray.refreshDynamicEnergy *
                          self.bank.numActiveSubarrayPerRow *
                          self.bank.numActiveSubarrayPerColumn)
        print(f"{spaces}    |--- Predecoder Dynamic Energy = {self._toJoule(predecoderEnergy)}")
        print(f"{spaces}    |--- Subarray Dynamic Energy   = {self._toJoule(self.bank.mat.subarray.refreshDynamicEnergy)} per active subarray")
        print(f"{spaces}       |--- Row Decoder Dynamic Energy = {self._toJoule(self.bank.mat.subarray.rowDecoder.refreshDynamicEnergy)}")

        if inputParameter.internalSensing:
            print(f"{spaces}       |--- Senseamp Dynamic Energy    = {self._toJoule(self.bank.mat.subarray.senseAmp.refreshDynamicEnergy)}")

        print(f"{spaces}       |--- Precharge Dynamic Energy   = {self._toJoule(self.bank.mat.subarray.precharger.refreshDynamicEnergy)}")

    def printAsCache(self, tagResult, cacheAccessMode):
        """Print cache-specific results (ported from C++)"""
        if self.bank.memoryType != MemoryType.DATA or tagResult.bank.memoryType != MemoryType.TAG:
            print("This is not a valid cache configuration.")
            return

        # Calculate cache metrics based on access mode
        if cacheAccessMode == CacheAccessMode.NORMAL_ACCESS_MODE:
            # Calculate latencies
            cacheMissLatency = tagResult.bank.readLatency
            cacheHitLatency = max(tagResult.bank.readLatency, self.bank.mat.readLatency)
            cacheHitLatency += self.bank.mat.subarray.columnDecoderLatency
            cacheHitLatency += self.bank.readLatency - self.bank.mat.readLatency
            cacheWriteLatency = max(tagResult.bank.writeLatency, self.bank.writeLatency)

            # Calculate power
            cacheMissDynamicEnergy = tagResult.bank.readDynamicEnergy
            cacheMissDynamicEnergy += self.bank.readDynamicEnergy
            cacheHitDynamicEnergy = tagResult.bank.readDynamicEnergy + self.bank.readDynamicEnergy
            cacheWriteDynamicEnergy = tagResult.bank.writeDynamicEnergy + self.bank.writeDynamicEnergy

        elif cacheAccessMode == CacheAccessMode.FAST_ACCESS_MODE:
            # Calculate latencies
            cacheMissLatency = tagResult.bank.readLatency
            cacheHitLatency = max(tagResult.bank.readLatency, self.bank.readLatency)
            cacheWriteLatency = max(tagResult.bank.writeLatency, self.bank.writeLatency)

            # Calculate power
            cacheMissDynamicEnergy = tagResult.bank.readDynamicEnergy
            cacheMissDynamicEnergy += self.bank.readDynamicEnergy
            cacheHitDynamicEnergy = tagResult.bank.readDynamicEnergy + self.bank.readDynamicEnergy
            cacheWriteDynamicEnergy = tagResult.bank.writeDynamicEnergy + self.bank.writeDynamicEnergy

        else:  # Sequential access
            # Calculate latencies
            cacheMissLatency = tagResult.bank.readLatency
            cacheHitLatency = tagResult.bank.readLatency + self.bank.readLatency
            cacheWriteLatency = max(tagResult.bank.writeLatency, self.bank.writeLatency)

            # Calculate power
            cacheMissDynamicEnergy = tagResult.bank.readDynamicEnergy
            cacheHitDynamicEnergy = tagResult.bank.readDynamicEnergy + self.bank.readDynamicEnergy
            cacheWriteDynamicEnergy = tagResult.bank.writeDynamicEnergy + self.bank.writeDynamicEnergy

        # Calculate leakage and area
        cacheLeakage = tagResult.bank.leakage + self.bank.leakage
        cacheArea = tagResult.bank.area + self.bank.area

        # Start printing
        print()
        print("=======================")
        print("CACHE DESIGN -- SUMMARY")
        print("=======================")

        print("Access Mode: ", end='')
        if cacheAccessMode == CacheAccessMode.NORMAL_ACCESS_MODE:
            print("Normal")
        elif cacheAccessMode == CacheAccessMode.FAST_ACCESS_MODE:
            print("Fast")
        else:
            print("Sequential")

        print("Area:")
        print(f" - Total Area = {cacheArea * 1e6:.3f}mm^2")
        print(f" |--- Data Array Area = {self.bank.height * 1e6:.3f}um x {self.bank.width * 1e6:.3f}um = {self.bank.area * 1e6:.3f}mm^2")
        print(f" |--- Tag Array Area  = {tagResult.bank.height * 1e6:.3f}um x {tagResult.bank.width * 1e6:.3f}um = {tagResult.bank.area * 1e6:.3f}mm^2")

        print("Timing:")
        print(f" - Cache Hit Latency   = {cacheHitLatency * 1e9:.3f}ns")
        print(f" - Cache Miss Latency  = {cacheMissLatency * 1e9:.3f}ns")
        print(f" - Cache Write Latency = {cacheWriteLatency * 1e9:.3f}ns")

        if cell.memCellType == MemCellType.eDRAM:
            cacheRefreshLatency = max(tagResult.bank.refreshLatency, self.bank.refreshLatency)
            print(f" - Cache Refresh Latency = {cacheRefreshLatency * 1e6:.3f}us per bank")
            if cell.retentionTime > 0:
                availability = ((cell.retentionTime - cacheRefreshLatency) / cell.retentionTime) * 100.0
                print(f" - Cache Availability = {availability:.2f}%")

        print("Power:")
        print(f" - Cache Hit Dynamic Energy   = {cacheHitDynamicEnergy * 1e9:.3f}nJ per access")
        print(f" - Cache Miss Dynamic Energy  = {cacheMissDynamicEnergy * 1e9:.3f}nJ per access")
        print(f" - Cache Write Dynamic Energy = {cacheWriteDynamicEnergy * 1e9:.3f}nJ per access")

        if cell.memCellType == MemCellType.eDRAM:
            cacheRefreshEnergy = tagResult.bank.refreshDynamicEnergy + self.bank.refreshDynamicEnergy
            print(f" - Cache Refresh Dynamic Energy = {cacheRefreshEnergy * 1e9:.3f}nJ per bank")

        print(f" - Cache Total Leakage Power  = {cacheLeakage * 1e3:.3f}mW")
        print(f" |--- Cache Data Array Leakage Power = {self.bank.leakage * 1e3:.3f}mW")
        print(f" |--- Cache Tag Array Leakage Power  = {tagResult.bank.leakage * 1e3:.3f}mW")

        if cell.memCellType == MemCellType.eDRAM and cell.retentionTime > 0:
            refreshPower = self.bank.refreshDynamicEnergy / cell.retentionTime
            print(f" - Cache Refresh Power = {self._toWatt(refreshPower)} per bank")

        if inputParameter.printLevel > 0:
            print()
            print("CACHE DATA ARRAY DETAILS")
            self.print(4)
            print()
            print("CACHE TAG ARRAY DETAILS")
            tagResult.print(4)

    def RunSimplifiedSimulation(self, input_param, tech, cell):
        """Run a simplified simulation to get non-zero results

        This is a placeholder implementation that uses basic analytical models
        to estimate performance metrics. For accurate results, the full DESTINY
        simulation engine needs to be ported from C++.
        """
        import math

        # Basic parameters
        cache_size_bits = input_param.capacity * 8 if input_param.capacity > 0 else input_param.cacheSize * 8
        word_width = input_param.wordWidth if input_param.wordWidth > 0 else 64
        process_nm = input_param.processNode if input_param.processNode > 0 else 65

        if cache_size_bits == 0 or word_width == 0:
            return

        # Estimate number of cells
        num_cells = cache_size_bits

        # Simplified cell area estimation (F^2, where F is feature size)
        # SRAM cell is typically 100-150 F^2
        feature_size_m = process_nm * 1e-9
        cell_area_f2 = 120  # typical SRAM cell in F^2
        self.cell_area = cell_area_f2 * (feature_size_m ** 2)

        # Bank area estimation
        # Include overhead for decoders, sense amps, etc (typically 2-3x cell array area)
        array_area = num_cells * self.cell_area
        overhead_factor = 2.5
        self.bank_area = array_area * overhead_factor
        self.total_area = self.bank_area

        # Bank dimensions (assume square-ish layout)
        self.bank_width = math.sqrt(self.bank_area)
        self.bank_height = self.bank_area / self.bank_width

        # Latency estimation (simplified)
        # Based on process node scaling
        base_latency_ns = 2.0  # base latency at 65nm
        scaling_factor = process_nm / 65.0
        self.read_latency = base_latency_ns * scaling_factor * 1e-9  # convert to seconds
        self.write_latency = self.read_latency * 1.2  # write slightly slower

        # Energy estimation (simplified)
        # Energy scales with C*V^2, where C scales with process node and V with technology
        voltage_map = {22: 0.8, 32: 0.9, 45: 1.0, 65: 1.1, 90: 1.2, 120: 1.3}
        voltage = 1.1  # default
        for node in sorted(voltage_map.keys()):
            if process_nm <= node:
                voltage = voltage_map[node]
                break

        # Capacitance scales with feature size
        base_cap_pf = 0.1  # pF per bit at 65nm
        cap_per_bit = base_cap_pf * (process_nm / 65.0) * 1e-12  # convert to F

        # Energy per access = C * V^2
        energy_per_bit = cap_per_bit * voltage * voltage
        bits_per_access = word_width

        self.read_dynamic_energy = bits_per_access * energy_per_bit
        self.write_dynamic_energy = self.read_dynamic_energy * 1.5  # write uses more energy

        # Leakage power estimation
        # Scales exponentially with voltage and linearly with number of transistors
        # SRAM cell has 6 transistors
        transistors_per_cell = 6
        total_transistors = num_cells * transistors_per_cell

        # Leakage per transistor (rough estimate)
        leakage_per_transistor_nw = 1.0 * (process_nm / 65.0) * (voltage / 1.1)**2
        self.leakage_power = total_transistors * leakage_per_transistor_nw * 1e-9  # convert to W

    def CalculateEDP(self):
        """Calculate Energy-Delay Product"""
        self.read_edp = self.read_latency * self.read_dynamic_energy
        self.write_edp = self.write_latency * self.write_dynamic_energy
        return self.read_edp, self.write_edp

    def PrintResult(self):
        """Print result summary (legacy method for compatibility)"""
        print("=" * 80)
        print("DESTINY Simulation Results")
        print("=" * 80)
        print(f"\nArea Metrics:")
        print(f"  Bank Area: {self.bank_area * 1e12:.3f} um^2")
        print(f"  Bank Height: {self.bank_height * 1e6:.3f} um")
        print(f"  Bank Width: {self.bank_width * 1e6:.3f} um")
        print(f"\nTiming Metrics:")
        print(f"  Read Latency: {self.read_latency * 1e9:.3f} ns")
        print(f"  Write Latency: {self.write_latency * 1e9:.3f} ns")
        print(f"\nPower Metrics:")
        print(f"  Read Dynamic Energy: {self.read_dynamic_energy * 1e12:.3f} pJ")
        print(f"  Write Dynamic Energy: {self.write_dynamic_energy * 1e12:.3f} pJ")
        print(f"  Leakage Power: {self.leakage_power * 1e3:.3f} mW")
        print("=" * 80)

    def WriteResultToFile(self, filename):
        """Write results to output file (legacy method for compatibility)"""
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DESTINY Simulation Results\n")
            f.write("=" * 80 + "\n")
            f.write(f"\nArea Metrics:\n")
            f.write(f"  Bank Area: {self.bank_area * 1e12:.3f} um^2\n")
            f.write(f"  Bank Height: {self.bank_height * 1e6:.3f} um\n")
            f.write(f"  Bank Width: {self.bank_width * 1e6:.3f} um\n")
            f.write(f"\nTiming Metrics:\n")
            f.write(f"  Read Latency: {self.read_latency * 1e9:.3f} ns\n")
            f.write(f"  Write Latency: {self.write_latency * 1e9:.3f} ns\n")
            f.write(f"\nPower Metrics:\n")
            f.write(f"  Read Dynamic Energy: {self.read_dynamic_energy * 1e12:.3f} pJ\n")
            f.write(f"  Write Dynamic Energy: {self.write_dynamic_energy * 1e12:.3f} pJ\n")
            f.write(f"  Leakage Power: {self.leakage_power * 1e3:.3f} mW\n")
            f.write("=" * 80 + "\n")
