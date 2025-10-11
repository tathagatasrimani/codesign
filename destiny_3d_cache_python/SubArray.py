#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

import math
from FunctionUnit import FunctionUnit
from RowDecoder import RowDecoder
from Precharger import Precharger
from SenseAmp import SenseAmp
from Mux import Mux
from typedef import (MemCellType, CellAccessType, DesignTarget, CacheAccessMode,
                     BufferDesignTarget, WriteScheme)
from constant import (NMOS, PMOS, INV, MIN_NMOS_SIZE, BITLINE_LEAKAGE_TOLERANCE,
                      STITCHING_OVERHEAD, DRAM_REFRESH_PERIOD, SHAPER_EFFICIENCY_CONSERVATIVE,
                      SHAPER_EFFICIENCY_AGGRESSIVE, DELTA_V_TH, TUNNEL_CURRENT_FLOW)
import globals as g
from formula import (calculate_gate_cap, calculate_drain_cap, calculate_on_resistance,
                     calculate_transconductance, calculate_gate_leakage, horowitz,
                     calculate_fbram_gate_cap, calculate_fbram_drain_cap)


class SubArray(FunctionUnit):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.invalid = False
        self.internalSenseAmp = False
        self.numRow = 0
        self.numColumn = 0
        self.multipleRowPerSet = False
        self.split = False
        self.muxSenseAmp = 1
        self.muxOutputLev1 = 1
        self.muxOutputLev2 = 1
        self.areaOptimizationLevel = BufferDesignTarget.latency_first
        self.num3DLevels = 1

        self.voltageSense = False
        self.senseVoltage = 0.0
        self.voltagePrecharge = 0.0
        self.numSenseAmp = 0
        self.lenWordline = 0.0
        self.lenBitline = 0.0
        self.capWordline = 0.0
        self.capBitline = 0.0
        self.resWordline = 0.0
        self.resBitline = 0.0
        self.resCellAccess = 0.0
        self.capCellAccess = 0.0
        self.resMemCellOff = 0.0
        self.resMemCellOn = 0.0
        self.voltageMemCellOff = 0.0
        self.voltageMemCellOn = 0.0
        self.resInSerialForSenseAmp = 0.0
        self.resEquivalentOn = 0.0
        self.resEquivalentOff = 0.0
        self.bitlineDelay = 0.0
        self.chargeLatency = 0.0
        self.columnDecoderLatency = 0.0
        self.bitlineDelayOn = 0.0
        self.bitlineDelayOff = 0.0

        self.rowDecoder = RowDecoder()
        self.bitlineMuxDecoder = RowDecoder()
        self.bitlineMux = Mux()
        self.senseAmpMuxLev1Decoder = RowDecoder()
        self.senseAmpMuxLev1 = Mux()
        self.senseAmpMuxLev2Decoder = RowDecoder()
        self.senseAmpMuxLev2 = Mux()
        self.precharger = Precharger()
        self.senseAmp = SenseAmp()

    def Initialize(self, _numRow, _numColumn, _multipleRowPerSet, _split,
                   _muxSenseAmp, _internalSenseAmp, _muxOutputLev1, _muxOutputLev2,
                   _areaOptimizationLevel, _num3DLevels):
        """Initialize subarray with all configuration parameters"""
        if self.initialized:
            print("[Subarray] Warning: Already initialized!")

        self.numRow = _numRow
        self.numColumn = _numColumn
        self.multipleRowPerSet = _multipleRowPerSet
        self.split = _split
        self.muxSenseAmp = _muxSenseAmp
        self.muxOutputLev1 = _muxOutputLev1
        self.muxOutputLev2 = _muxOutputLev2
        self.internalSenseAmp = _internalSenseAmp
        self.areaOptimizationLevel = _areaOptimizationLevel
        self.num3DLevels = _num3DLevels

        maxWordlineCurrent = 0.0
        maxBitlineCurrent = 0.0

        # Check if the configuration is legal
        if g.inputParameter.designTarget == DesignTarget.cache and g.inputParameter.cacheAccessMode != CacheAccessMode.sequential_access_mode:
            # In these cases, each column should hold part of data in all the ways
            if self.numColumn < g.inputParameter.associativity:
                self.invalid = True
                self.initialized = True
                return

        if g.cell.memCellType == MemCellType.DRAM or g.cell.memCellType == MemCellType.eDRAM:
            if self.muxSenseAmp > 1:
                # DRAM does not allow muxed bitline because of its destructive readout
                self.invalid = True
                self.initialized = True
                return

        if g.cell.memCellType == MemCellType.SLCNAND:
            if self.numRow < g.inputParameter.flashBlockSize / g.inputParameter.pageSize:
                # SLC NAND does not have enough rows to hold the page count
                self.invalid = True
                self.initialized = True
                return
            if self.internalSenseAmp and self.muxSenseAmp < 2:
                # There is no way to put the sense amp
                self.invalid = True
                self.initialized = True
                return

        if g.cell.memCellType == MemCellType.memristor or g.cell.memCellType == MemCellType.FBRAM:
            if self.internalSenseAmp and self.muxSenseAmp < 2:
                # There is no way to put the sense amp
                self.invalid = True
                self.initialized = True
                return

        if g.cell.memCellType == MemCellType.FBRAM:
            if g.cell.resistanceOff / g.cell.resistanceOn < self.numRow / BITLINE_LEAKAGE_TOLERANCE:
                # bitline too long
                self.invalid = True
                self.initialized = True
                return
            maxBitlineCurrent = max(g.cell.resetCurrent, g.cell.setCurrent) + g.cell.leakageCurrentAccessDevice * (self.numRow - 1)

        if g.cell.memCellType == MemCellType.MRAM or g.cell.memCellType == MemCellType.PCRAM or g.cell.memCellType == MemCellType.memristor:
            if g.cell.accessType == CellAccessType.CMOS_access:
                if (g.tech.currentOnNmos[g.inputParameter.temperature - 300] /
                    g.tech.currentOffNmos[g.inputParameter.temperature - 300] < self.numRow / BITLINE_LEAKAGE_TOLERANCE):
                    # bitline too long
                    self.invalid = True
                    self.initialized = True
                    return
                maxBitlineCurrent = max(g.cell.resetCurrent, g.cell.setCurrent) + g.cell.leakageCurrentAccessDevice * (self.numRow - 1)
            else:  # non-CMOS access
                # Write half select problem limit the array size
                if g.cell.resetCurrent == 0:
                    resetCurrent = (abs(g.cell.resetVoltage) - g.cell.voltageDropAccessDevice) / g.cell.resistanceOnAtResetVoltage
                else:
                    resetCurrent = g.cell.resetCurrent
                numSelectedColumnPerRow = self.numColumn // self.muxSenseAmp // self.muxOutputLev1 // self.muxOutputLev2

                if g.cell.accessType == CellAccessType.none_access:
                    # Based on Equation (1) in DATE2011 "Design Implications of Memristor-Based RRAM Cross-Point Structures" Xu et. al
                    maxWordlineCurrent = (resetCurrent * numSelectedColumnPerRow +
                                        resetCurrent * g.cell.resistanceOnAtResetVoltage / 2 / g.cell.resistanceOnAtHalfResetVoltage *
                                        (self.numColumn - numSelectedColumnPerRow))
                    maxWordlineCurrent += (resetCurrent * g.cell.resistanceOnAtResetVoltage / 2 / g.cell.resistanceOnAtHalfResetVoltage *
                                         self.numColumn * (self.num3DLevels - 1))
                else:  # diode or BJT
                    maxWordlineCurrent = (resetCurrent * numSelectedColumnPerRow +
                                        g.cell.leakageCurrentAccessDevice * (self.numColumn - numSelectedColumnPerRow))
                    maxWordlineCurrent += g.cell.leakageCurrentAccessDevice * self.numColumn * (self.num3DLevels - 1)

                minWordlineDriverWidth = maxWordlineCurrent / g.tech.currentOnNmos[g.inputParameter.temperature - 300]
                if minWordlineDriverWidth > g.inputParameter.maxNmosSize * g.tech.featureSize:
                    self.invalid = True
                    return

                if g.cell.accessType == CellAccessType.none_access:
                    # Based on Table 1, Row 1 in DATE2011 "Design Implications of Memristor-Based RRAM Cross-Point Structures" Xu et. al
                    maxBitlineCurrent = resetCurrent + resetCurrent * g.cell.resistanceOnAtResetVoltage / 2 / g.cell.resistanceOnAtHalfResetVoltage * (self.numRow - 1)
                    maxBitlineCurrent = (resetCurrent * g.cell.resistanceOnAtResetVoltage / 2 / g.cell.resistanceOnAtHalfResetVoltage *
                                       self.numRow * (self.num3DLevels - 1))
                else:  # diode or BJT
                    maxBitlineCurrent = resetCurrent + g.cell.leakageCurrentAccessDevice * (self.numRow - 1)
                    maxBitlineCurrent += g.cell.leakageCurrentAccessDevice * self.numRow * (self.num3DLevels - 1)

        minBitlineMuxWidth = maxBitlineCurrent / g.tech.currentOnNmos[g.inputParameter.temperature - 300]
        minBitlineMuxWidth = max(MIN_NMOS_SIZE * g.tech.featureSize, minBitlineMuxWidth)
        if minBitlineMuxWidth > g.inputParameter.maxNmosSize * g.tech.featureSize:
            self.invalid = True
            return

        if self.internalSenseAmp:
            if g.cell.memCellType == MemCellType.SRAM or g.cell.memCellType == MemCellType.DRAM or g.cell.memCellType == MemCellType.eDRAM:
                # SRAM, DRAM, and eDRAM all use voltage sensing
                self.voltageSense = True
            elif (g.cell.memCellType == MemCellType.MRAM or g.cell.memCellType == MemCellType.PCRAM or
                  g.cell.memCellType == MemCellType.memristor or g.cell.memCellType == MemCellType.FBRAM):
                self.voltageSense = g.cell.readMode
            else:  # NAND flash
                self.voltageSense = True
        elif g.cell.memCellType == MemCellType.DRAM or g.cell.memCellType == MemCellType.eDRAM:
            print("[Subarray] Error: DRAM does not support external sense amplifiers!")
            exit(-1)

        # Derived parameters
        self.numSenseAmp = self.numColumn // self.muxSenseAmp
        self.lenWordline = float(self.numColumn) * g.cell.widthInFeatureSize * g.devtech.featureSize
        self.lenBitline = float(self.numRow) * g.cell.heightInFeatureSize * g.devtech.featureSize

        # Add stitching overhead if necessary
        if g.cell.stitching:
            self.lenWordline += ((self.numColumn - 1) // g.cell.stitching + 1) * STITCHING_OVERHEAD * g.devtech.featureSize

        # Add select transistors into the length calculation
        if g.cell.memCellType == MemCellType.SLCNAND:
            pageCount = g.inputParameter.flashBlockSize // g.inputParameter.pageSize
            # Two select transistor including contacts have total length of 5F
            self.lenBitline += (self.numRow // pageCount) * 5 * g.tech.featureSize

        # Calculate wire resistance/capacitance
        self.capWordline = self.lenWordline * g.localWire.capWirePerUnit * self.num3DLevels
        self.resWordline = self.lenWordline * g.localWire.resWirePerUnit * self.num3DLevels
        self.capBitline = self.lenBitline * g.localWire.capWirePerUnit * self.num3DLevels
        self.resBitline = self.lenBitline * g.localWire.resWirePerUnit * self.num3DLevels

        # Calculate the load resistance and capacitance for Mux Decoders
        resMuxLoad = self.resWordline
        capMuxLoad = calculate_gate_cap(minBitlineMuxWidth, g.tech) * self.numColumn
        capMuxLoad += self.capWordline

        if g.cell.memCellType == MemCellType.DRAM or g.cell.memCellType == MemCellType.eDRAM:
            self.senseVoltage = g.devtech.vdd / 2 * g.cell.capDRAMCell / (g.cell.capDRAMCell + self.capBitline)
            if self.senseVoltage < g.cell.minSenseVoltage:  # Bitline is too long
                self.invalid = True
                self.initialized = True
                return
        elif g.cell.memCellType == MemCellType.SLCNAND:
            # suppose the reference voltage is 0.5Vdd, the initial bitline voltage is 0.6Vdd
            # if the bitline drops to 0.4Vdd, the senseamp can tell which data is stored
            self.senseVoltage = max(g.cell.minSenseVoltage, 0.2 * g.tech.vdd)
        else:
            # TO-DO: different memory technology might have different values here
            self.senseVoltage = g.cell.minSenseVoltage

        # Add transistor resistance/capacitance
        if g.cell.memCellType == MemCellType.SRAM:
            # SRAM has two access transistors
            self.resCellAccess = calculate_on_resistance(g.cell.widthAccessCMOS * g.tech.featureSize, NMOS,
                                                        g.inputParameter.temperature, g.tech)
            self.capCellAccess = calculate_drain_cap(g.cell.widthAccessCMOS * g.tech.featureSize, NMOS,
                                                     g.cell.widthInFeatureSize * g.tech.featureSize, g.tech)
            self.capWordline += 2 * calculate_gate_cap(g.cell.widthAccessCMOS * g.tech.featureSize, g.tech) * self.numColumn
            self.capBitline += self.capCellAccess * self.numRow / 2  # Due to shared contact
            self.voltagePrecharge = g.tech.vdd / 2  # SRAM read voltage is always half of vdd

        elif g.cell.memCellType == MemCellType.DRAM or g.cell.memCellType == MemCellType.eDRAM:
            # DRAM and eDRAM only has one access transistors
            self.resCellAccess = calculate_on_resistance(g.cell.widthAccessCMOS * g.devtech.featureSize, NMOS,
                                                        g.inputParameter.temperature, g.devtech)
            self.capCellAccess = calculate_drain_cap(g.cell.widthAccessCMOS * g.devtech.featureSize, NMOS,
                                                     g.cell.widthInFeatureSize * g.devtech.featureSize, g.devtech)
            self.capWordline += calculate_gate_cap(g.cell.widthAccessCMOS * g.devtech.featureSize, g.devtech) * self.numColumn
            self.capBitline += self.capCellAccess * self.numRow / 2  # Due to shared contact
            self.voltagePrecharge = g.devtech.vdd / 2  # DRAM read voltage is always half of vdd

        elif g.cell.memCellType == MemCellType.FBRAM:  # Floating Body RAM
            self.resCellAccess = 0
            self.capCellAccess = calculate_fbram_drain_cap(g.cell.widthSOIDevice * g.tech.featureSize, g.tech)
            self.capWordline += calculate_fbram_gate_cap(g.cell.widthSOIDevice * g.tech.featureSize,
                                                        g.cell.gateOxThicknessFactor, g.tech) * self.numColumn
            self.capBitline += self.capCellAccess * self.numRow / 2  # Due to shared contact
            self.resMemCellOff = g.cell.resistanceOff
            self.resMemCellOn = g.cell.resistanceOn
            if g.cell.readMode:  # voltage-sensing
                if g.cell.readVoltage == 0:  # Current-in voltage sensing
                    self.voltageMemCellOff = g.cell.readCurrent * self.resMemCellOff
                    self.voltageMemCellOn = g.cell.readCurrent * self.resMemCellOn
                    self.voltagePrecharge = (self.voltageMemCellOff + self.voltageMemCellOn) / 2
                    self.voltagePrecharge = min(g.tech.vdd, self.voltagePrecharge)  # TO-DO: we can have charge bump to increase SA working point
                    if (self.voltagePrecharge - self.voltageMemCellOn) <= self.senseVoltage:
                        print("Error[Subarray]: Read current too large or too small that no reasonable precharge voltage existing")
                        self.invalid = True
                        return
                else:  # Voltage-divider sensing
                    self.resInSerialForSenseAmp = math.sqrt(self.resMemCellOn * self.resMemCellOff)
                    self.resEquivalentOn = self.resMemCellOn * self.resInSerialForSenseAmp / (self.resMemCellOn + self.resInSerialForSenseAmp)
                    self.resEquivalentOff = self.resMemCellOff * self.resInSerialForSenseAmp / (self.resMemCellOff + self.resInSerialForSenseAmp)
                    self.voltageMemCellOff = g.cell.readVoltage * self.resMemCellOff / (self.resMemCellOff + self.resInSerialForSenseAmp)
                    self.voltageMemCellOn = g.cell.readVoltage * self.resMemCellOn / (self.resMemCellOn + self.resInSerialForSenseAmp)
                    self.voltagePrecharge = (self.voltageMemCellOff + self.voltageMemCellOn) / 2
                    self.voltagePrecharge = min(g.tech.vdd, self.voltagePrecharge)  # TO-DO: we can have charge bump to increase SA working point
                    if (self.voltagePrecharge - self.voltageMemCellOn) <= self.senseVoltage:
                        print("Error[Subarray]: Read Voltage too large or too small that no reasonable precharge voltage existing")
                        self.invalid = True
                        return

        elif g.cell.memCellType == MemCellType.MRAM or g.cell.memCellType == MemCellType.PCRAM or g.cell.memCellType == MemCellType.memristor:
            # MRAM, PCRAM, and memristor have three types of access devices: CMOS, BJT, and diode
            if g.cell.accessType == CellAccessType.CMOS_access:
                self.resCellAccess = calculate_on_resistance(g.cell.widthAccessCMOS * g.tech.featureSize, NMOS,
                                                            g.inputParameter.temperature, g.tech)
                self.capCellAccess = calculate_drain_cap(g.cell.widthAccessCMOS * g.tech.featureSize, NMOS,
                                                         g.cell.widthInFeatureSize * g.tech.featureSize, g.tech)
                self.capWordline += calculate_gate_cap(g.cell.widthAccessCMOS * g.tech.featureSize, g.tech) * self.numColumn
                self.capBitline += self.capCellAccess * self.numRow / 2  # Due to shared contact
            elif g.cell.accessType == CellAccessType.BJT_access:
                # TO-DO
                pass
            else:  # none_access or diode_access
                self.resCellAccess = 0
                self.capCellAccess = max(g.cell.capacitanceOn, g.cell.capacitanceOff)
                self.capWordline += max(g.cell.capacitanceOff, g.cell.capacitanceOn) * self.numColumn  # TO-DO: choose the right capacitance
                self.capBitline += max(g.cell.capacitanceOff, g.cell.capacitanceOn) * self.numRow  # TO-DO: choose the right capacitance

                # Add capacitance for other monolithic layers
                self.capWordline += max(g.cell.capacitanceOff, g.cell.capacitanceOn) * self.numColumn * (self.num3DLevels - 1)  # TO-DO: choose the right capacitance
                self.capBitline += max(g.cell.capacitanceOff, g.cell.capacitanceOn) * self.numRow * (self.num3DLevels - 1)  # TO-DO: choose the right capacitance

            self.resMemCellOff = self.resCellAccess + g.cell.resistanceOff
            self.resMemCellOn = self.resCellAccess + g.cell.resistanceOn
            if g.cell.readMode:  # voltage-sensing
                if g.cell.readVoltage == 0:  # Current-in voltage sensing
                    self.voltageMemCellOff = g.cell.readCurrent * self.resMemCellOff
                    self.voltageMemCellOn = g.cell.readCurrent * self.resMemCellOn
                    self.voltagePrecharge = (self.voltageMemCellOff + self.voltageMemCellOn) / 2
                    self.voltagePrecharge = min(g.tech.vdd, self.voltagePrecharge)  # TO-DO: we can have charge bump to increase SA working point
                    if (self.voltagePrecharge - self.voltageMemCellOn) <= self.senseVoltage:
                        print("Error[Subarray]: Read current too large or too small that no reasonable precharge voltage existing")
                        self.invalid = True
                        return
                else:  # Voltage-in voltage sensing
                    self.resInSerialForSenseAmp = math.sqrt(self.resMemCellOn * self.resMemCellOff)
                    self.resEquivalentOn = self.resMemCellOn * self.resInSerialForSenseAmp / (self.resMemCellOn + self.resInSerialForSenseAmp)
                    self.resEquivalentOff = self.resMemCellOff * self.resInSerialForSenseAmp / (self.resMemCellOff + self.resInSerialForSenseAmp)
                    self.voltageMemCellOff = g.cell.readVoltage * self.resMemCellOff / (self.resMemCellOff + self.resInSerialForSenseAmp)
                    self.voltageMemCellOn = g.cell.readVoltage * self.resMemCellOn / (self.resMemCellOn + self.resInSerialForSenseAmp)
                    self.voltagePrecharge = (self.voltageMemCellOff + self.voltageMemCellOn) / 2
                    self.voltagePrecharge = min(g.tech.vdd, self.voltagePrecharge)  # TO-DO: we can have charge bump to increase SA working point
                    if (self.voltagePrecharge - self.voltageMemCellOn) <= self.senseVoltage:
                        print("Error[Subarray]: Read Voltage too large or too small that no reasonable precharge voltage existing")
                        self.invalid = True
                        return

        elif g.cell.memCellType == MemCellType.SLCNAND:
            # Calculate the NAND flash string length, which is the page count per block plus 2 (two select transistors)
            pageCount = g.inputParameter.flashBlockSize // g.inputParameter.pageSize
            stringLength = pageCount + 2
            self.resCellAccess = calculate_on_resistance(g.tech.featureSize, NMOS, g.inputParameter.temperature, g.tech) * stringLength
            self.capCellAccess = calculate_drain_cap(g.tech.featureSize, NMOS, g.cell.widthInFeatureSize * g.tech.featureSize, g.tech)
            # The capacitance of each cell at the gate terminal is the series of C_control_gate | C_floating_gate
            self.capWordline += calculate_gate_cap(g.tech.featureSize, g.tech) * self.numColumn * g.cell.gateCouplingRatio / (g.cell.gateCouplingRatio + 1)
            self.capBitline += self.capCellAccess * (self.numRow // pageCount) / 2  # 2 is due to shared contact and the effective row count is numRow/pageCount
            self.voltagePrecharge = g.tech.vdd * 0.6  # SLC NAND flash bitline precharge voltage is assumed to 0.6Vdd
        else:  # MLC NAND flash
            # TO-DO
            pass

        # Initialize sub-components
        self.precharger.Initialize(g.tech.vdd, self.numColumn, self.capBitline, self.resBitline)
        self.precharger.CalculateRC()

        self.rowDecoder.Initialize(self.numRow, self.capWordline, self.resWordline, self.multipleRowPerSet,
                                   self.areaOptimizationLevel, maxWordlineCurrent)
        if self.rowDecoder.invalid:
            self.invalid = True
            return
        self.rowDecoder.CalculateRC()

        if not self.invalid:
            self.bitlineMuxDecoder.Initialize(self.muxSenseAmp, capMuxLoad, resMuxLoad, False, self.areaOptimizationLevel, 0)
            if self.bitlineMuxDecoder.invalid:
                self.invalid = True
            else:
                self.bitlineMuxDecoder.CalculateRC()

        if not self.invalid:
            self.senseAmpMuxLev1Decoder.Initialize(self.muxOutputLev1, capMuxLoad, resMuxLoad, False, self.areaOptimizationLevel, 0)
            if self.senseAmpMuxLev1Decoder.invalid:
                self.invalid = True
            else:
                self.senseAmpMuxLev1Decoder.CalculateRC()

        if not self.invalid:
            self.senseAmpMuxLev2Decoder.Initialize(self.muxOutputLev2, capMuxLoad, resMuxLoad, False, self.areaOptimizationLevel, 0)
            if self.senseAmpMuxLev2Decoder.invalid:
                self.invalid = True
            else:
                self.senseAmpMuxLev2Decoder.CalculateRC()

        self.senseAmpMuxLev2.Initialize(self.muxOutputLev2, self.numColumn // self.muxSenseAmp // self.muxOutputLev1 // self.muxOutputLev2,
                                       0, 0, maxBitlineCurrent)
        self.senseAmpMuxLev2.CalculateRC()

        self.senseAmpMuxLev1.Initialize(self.muxOutputLev1, self.numColumn // self.muxSenseAmp // self.muxOutputLev1,
                                       self.senseAmpMuxLev2.capForPreviousDelayCalculation,
                                       self.senseAmpMuxLev2.capForPreviousPowerCalculation, maxBitlineCurrent)
        self.senseAmpMuxLev1.CalculateRC()

        if self.internalSenseAmp:
            if not self.invalid:
                self.senseAmp.Initialize(self.numSenseAmp, not self.voltageSense, self.senseVoltage,
                                        self.lenWordline / self.numColumn * self.muxSenseAmp)
                if self.senseAmp.invalid:
                    self.invalid = True
                else:
                    self.senseAmp.CalculateRC()
            if not self.invalid:
                self.bitlineMux.Initialize(self.muxSenseAmp, self.numColumn // self.muxSenseAmp,
                                          self.senseAmp.capLoad, self.senseAmp.capLoad, maxBitlineCurrent)
        else:
            if not self.invalid:
                self.bitlineMux.Initialize(self.muxSenseAmp, self.numColumn // self.muxSenseAmp,
                                          self.senseAmpMuxLev1.capForPreviousDelayCalculation,
                                          self.senseAmpMuxLev1.capForPreviousPowerCalculation, maxBitlineCurrent)

        if not self.invalid:
            self.bitlineMux.CalculateRC()

        self.initialized = True

    def CalculateArea(self):
        """Calculate subarray area"""
        if not self.initialized:
            print("[Subarray] Error: Require initialization first!")
        elif self.invalid:
            self.height = self.width = self.area = g.invalid_value
        else:
            addWidth = 0.0
            addHeight = 0.0

            self.width = self.lenWordline
            self.height = self.lenBitline

            self.rowDecoder.CalculateArea()
            if self.rowDecoder.height > self.height:
                # assume magic folding
                addWidth = self.rowDecoder.area / self.height
            else:
                # allow white space
                addWidth = self.rowDecoder.width

            self.precharger.CalculateArea()
            if self.precharger.width > self.width:
                # assume magic folding
                addHeight = self.precharger.area / self.precharger.width
            else:
                # allow white space
                addHeight = self.precharger.height

            self.bitlineMux.CalculateArea()
            addHeight += self.bitlineMux.height

            if self.internalSenseAmp:
                self.senseAmp.CalculateArea()
                if self.senseAmp.width > self.width * 1.001:
                    # should never happen
                    print("[ERROR] Sense Amplifier area calculation is wrong!")
                else:
                    addHeight += self.senseAmp.height

            self.senseAmpMuxLev1.CalculateArea()
            addHeight += self.senseAmpMuxLev1.height

            self.senseAmpMuxLev2.CalculateArea()
            addHeight += self.senseAmpMuxLev2.height

            self.bitlineMuxDecoder.CalculateArea()
            addWidth = max(addWidth, self.bitlineMuxDecoder.width)
            self.senseAmpMuxLev1Decoder.CalculateArea()
            addWidth = max(addWidth, self.senseAmpMuxLev1Decoder.width)
            self.senseAmpMuxLev2Decoder.CalculateArea()
            addWidth = max(addWidth, self.senseAmpMuxLev2Decoder.width)

            self.width += addWidth
            self.height += addHeight
            self.area = self.width * self.height

    def CalculateLatency(self, _rampInput):
        """Calculate latency for read and write operations"""
        if not self.initialized:
            print("[Subarray] Error: Require initialization first!")
        elif self.invalid:
            self.readLatency = self.writeLatency = g.invalid_value
        else:
            self.precharger.CalculateLatency(_rampInput)
            self.rowDecoder.CalculateLatency(_rampInput)
            self.bitlineMuxDecoder.CalculateLatency(_rampInput)
            self.senseAmpMuxLev1Decoder.CalculateLatency(_rampInput)
            self.senseAmpMuxLev2Decoder.CalculateLatency(_rampInput)
            self.columnDecoderLatency = max(max(self.bitlineMuxDecoder.readLatency, self.senseAmpMuxLev1Decoder.readLatency),
                                           self.senseAmpMuxLev2Decoder.readLatency)
            decoderLatency = max(self.rowDecoder.readLatency, self.columnDecoderLatency)

            # need a second thought on this equation
            capPassTransistor = (self.bitlineMux.capNMOSPassTransistor +
                               self.senseAmpMuxLev1.capNMOSPassTransistor +
                               self.senseAmpMuxLev2.capNMOSPassTransistor)
            resPassTransistor = (self.bitlineMux.resNMOSPassTransistor +
                               self.senseAmpMuxLev1.resNMOSPassTransistor +
                               self.senseAmpMuxLev2.resNMOSPassTransistor)
            tauChargeLatency = resPassTransistor * (capPassTransistor + self.capBitline) + self.resBitline * self.capBitline / 2
            self.chargeLatency, _ = horowitz(tauChargeLatency, 0, 1e20)

            if g.cell.memCellType == MemCellType.SRAM:
                # Codes below calculate the bitline latency
                resPullDown = calculate_on_resistance(g.cell.widthSRAMCellNMOS * g.tech.featureSize, NMOS,
                                                     g.inputParameter.temperature, g.tech)
                tau = ((self.resCellAccess + resPullDown) * (self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousDelayCalculation) +
                      self.resBitline * (self.bitlineMux.capForPreviousDelayCalculation + self.capBitline / 2))
                tau *= math.log(self.voltagePrecharge / (self.voltagePrecharge - self.senseVoltage / 2))  # one signal raises and the other drops, so senseVoltage/2 is enough
                gm = calculate_transconductance(g.cell.widthAccessCMOS * g.tech.featureSize, NMOS, g.tech)
                beta = 1 / (resPullDown * gm)
                self.bitlineDelay, bitlineRamp = horowitz(tau, beta, self.rowDecoder.rampOutput)
                self.bitlineMux.CalculateLatency(bitlineRamp)
                if self.internalSenseAmp:
                    self.senseAmp.CalculateLatency(self.bitlineMuxDecoder.rampOutput)
                    self.senseAmpMuxLev1.CalculateLatency(1e20)
                    self.senseAmpMuxLev2.CalculateLatency(self.senseAmpMuxLev1.rampOutput)
                else:
                    self.senseAmpMuxLev1.CalculateLatency(self.bitlineMux.rampOutput)
                    self.senseAmpMuxLev2.CalculateLatency(self.senseAmpMuxLev1.rampOutput)
                self.readLatency = (decoderLatency + self.bitlineDelay + self.bitlineMux.readLatency + self.senseAmp.readLatency +
                                  self.senseAmpMuxLev1.readLatency + self.senseAmpMuxLev2.readLatency)
                # assume symmetric read/write for SRAM bitline delay
                self.writeLatency = self.readLatency

            elif g.cell.memCellType == MemCellType.DRAM or g.cell.memCellType == MemCellType.eDRAM:
                cap = ((self.capCellAccess + g.cell.capDRAMCell) * (self.capBitline + self.bitlineMux.capForPreviousDelayCalculation) /
                      (self.capCellAccess + g.cell.capDRAMCell + self.capBitline + self.bitlineMux.capForPreviousDelayCalculation))
                res = self.resBitline + self.resCellAccess
                tau = 2.3 * res * cap
                self.bitlineDelay, bitlineRamp = horowitz(tau, 0, self.rowDecoder.rampOutput)
                self.senseAmp.CalculateLatency(bitlineRamp)
                self.senseAmpMuxLev1.CalculateLatency(1e20)
                self.senseAmpMuxLev2.CalculateLatency(self.senseAmpMuxLev1.rampOutput)

                # Refresh operation does not pass sense amplifier
                self.refreshLatency = decoderLatency + self.bitlineDelay + self.senseAmp.readLatency
                self.refreshLatency *= self.numRow  # TOTAL refresh latency for subarray
                self.readLatency = (decoderLatency + self.bitlineDelay + self.senseAmp.readLatency +
                                  self.senseAmpMuxLev1.readLatency + self.senseAmpMuxLev2.readLatency)
                # assume symmetric read/write for DRAM/eDRAM bitline delay
                self.writeLatency = self.readLatency

            elif (g.cell.memCellType == MemCellType.MRAM or g.cell.memCellType == MemCellType.PCRAM or
                  g.cell.memCellType == MemCellType.memristor or g.cell.memCellType == MemCellType.FBRAM):
                bitlineRamp = 0.0
                if g.cell.readMode == False:  # current-sensing
                    # Use ICCAD 2009 model
                    tau = self.resBitline * self.capBitline / 2 * (self.resMemCellOff + self.resBitline / 3) / (self.resMemCellOff + self.resBitline)
                    self.bitlineDelay, bitlineRamp = horowitz(tau, 0, self.rowDecoder.rampOutput)
                else:  # voltage-sensing
                    if g.cell.readVoltage == 0:  # Current-in voltage sensing
                        tau = (self.resMemCellOn * (self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousDelayCalculation) +
                              self.resBitline * (self.bitlineMux.capForPreviousDelayCalculation + self.capBitline / 2))  # time constant of LRS
                        self.bitlineDelayOn = tau * math.log((self.voltagePrecharge - self.voltageMemCellOn) /
                                                            (self.voltagePrecharge - self.voltageMemCellOn - self.senseVoltage))  # BitlineDelay of HRS
                        tau = (self.resMemCellOff * (self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousDelayCalculation) +
                              self.resBitline * (self.bitlineMux.capForPreviousDelayCalculation + self.capBitline / 2))  # time constant of HRS
                        self.bitlineDelayOff = tau * math.log((self.voltageMemCellOff - self.voltagePrecharge) /
                                                             (self.voltageMemCellOff - self.voltagePrecharge - self.senseVoltage))
                        self.bitlineDelay = max(self.bitlineDelayOn, self.bitlineDelayOff)
                    else:  # Voltage-in voltage sensing
                        tau = (self.resEquivalentOn * (self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousDelayCalculation) +
                              self.resBitline * (self.bitlineMux.capForPreviousDelayCalculation + self.capBitline / 2))  # time constant of LRS
                        self.bitlineDelayOn = tau * math.log((self.voltagePrecharge - self.voltageMemCellOn) /
                                                            (self.voltagePrecharge - self.voltageMemCellOn - self.senseVoltage))  # BitlineDelay of HRS

                        tau = (self.resEquivalentOff * (self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousDelayCalculation) +
                              self.resBitline * (self.bitlineMux.capForPreviousDelayCalculation + self.capBitline / 2))  # time constant of HRS
                        self.bitlineDelayOff = tau * math.log((self.voltageMemCellOff - self.voltagePrecharge) /
                                                             (self.voltageMemCellOff - self.voltagePrecharge - self.senseVoltage))
                        self.bitlineDelay = max(self.bitlineDelayOn, self.bitlineDelayOff)

                self.bitlineMux.CalculateLatency(bitlineRamp)
                if self.internalSenseAmp:
                    self.senseAmp.CalculateLatency(self.bitlineMuxDecoder.rampOutput)
                    self.senseAmpMuxLev1.CalculateLatency(1e20)
                    self.senseAmpMuxLev2.CalculateLatency(self.senseAmpMuxLev1.rampOutput)
                else:
                    self.senseAmpMuxLev1.CalculateLatency(self.bitlineMux.rampOutput)
                    self.senseAmpMuxLev2.CalculateLatency(self.senseAmpMuxLev1.rampOutput)
                self.readLatency = (decoderLatency + self.bitlineDelay + self.bitlineMux.readLatency + self.senseAmp.readLatency +
                                  self.senseAmpMuxLev1.readLatency + self.senseAmpMuxLev2.readLatency)

                if g.cell.memCellType == MemCellType.PCRAM:
                    if g.inputParameter.writeScheme == WriteScheme.write_and_verify:
                        # TO-DO: write and verify programming
                        pass
                    else:
                        self.writeLatency = max(self.rowDecoder.writeLatency, self.columnDecoderLatency + self.chargeLatency)  # TO-DO: why not directly use precharger latency?
                        self.resetLatency = self.writeLatency + g.cell.resetPulse
                        self.setLatency = self.writeLatency + g.cell.setPulse
                        self.writeLatency += max(g.cell.resetPulse, g.cell.setPulse)
                elif g.cell.memCellType == MemCellType.FBRAM:
                    self.writeLatency = max(self.rowDecoder.writeLatency, self.columnDecoderLatency + self.chargeLatency)
                    self.resetLatency = self.writeLatency + g.cell.resetPulse
                    self.setLatency = self.writeLatency + g.cell.setPulse
                    self.writeLatency += max(g.cell.resetPulse, g.cell.setPulse)
                else:  # memristor and MRAM
                    if g.cell.accessType == CellAccessType.diode_access or g.cell.accessType == CellAccessType.none_access:
                        if g.inputParameter.writeScheme == WriteScheme.erase_before_reset or g.inputParameter.writeScheme == WriteScheme.erase_before_set:
                            self.writeLatency = max(self.rowDecoder.writeLatency, self.chargeLatency)
                        else:
                            self.writeLatency = max(self.rowDecoder.writeLatency, self.columnDecoderLatency + self.chargeLatency)
                        self.writeLatency += self.chargeLatency
                        self.writeLatency += g.cell.resetPulse + g.cell.setPulse
                    else:  # CMOS or Bipolar access
                        self.writeLatency = max(self.rowDecoder.writeLatency, self.columnDecoderLatency + self.chargeLatency)
                        self.resetLatency = self.writeLatency + g.cell.resetPulse
                        self.setLatency = self.writeLatency + g.cell.setPulse
                        self.writeLatency += max(g.cell.resetPulse, g.cell.setPulse)

            elif g.cell.memCellType == MemCellType.SLCNAND:
                # Calculate the NAND flash string length, which is the page count per block plus 2 (two select transistors)
                pageCount = g.inputParameter.flashBlockSize // g.inputParameter.pageSize
                stringLength = pageCount + 2
                # Codes below calculate the bitline latency
                resPullDown = calculate_on_resistance(g.tech.featureSize, NMOS, g.inputParameter.temperature, g.tech) * stringLength
                tau = (resPullDown * (self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousDelayCalculation) +
                      self.resBitline * (self.bitlineMux.capForPreviousDelayCalculation + self.capBitline / 2))
                # in one case the bitline is unchanged, and in the other case the bitline drops from 0.6V to 0.4V
                tau *= math.log(self.voltagePrecharge / (self.voltagePrecharge - self.senseVoltage))
                gm = calculate_transconductance(g.tech.featureSize, NMOS, g.tech)  # minimum size transistor
                beta = 1 / (resPullDown * gm)
                self.bitlineDelay, bitlineRamp = horowitz(tau, beta, self.rowDecoder.rampOutput)
                # to correct unnecessary horowitz calculation, TO-DO: need to revisit
                self.bitlineDelay = max(self.bitlineDelay, tau * 20)
                self.bitlineMux.CalculateLatency(bitlineRamp)
                if self.internalSenseAmp:
                    self.senseAmp.CalculateLatency(self.bitlineMuxDecoder.rampOutput)
                    self.senseAmpMuxLev1.CalculateLatency(1e20)
                    self.senseAmpMuxLev2.CalculateLatency(self.senseAmpMuxLev1.rampOutput)
                else:
                    self.senseAmpMuxLev1.CalculateLatency(self.bitlineMux.rampOutput)
                    self.senseAmpMuxLev2.CalculateLatency(self.senseAmpMuxLev1.rampOutput)
                self.readLatency = (decoderLatency + self.bitlineDelay + self.bitlineMux.readLatency + self.senseAmp.readLatency +
                                  self.senseAmpMuxLev1.readLatency + self.senseAmpMuxLev2.readLatency)
                # calculate the erase time, a.k.a. reset here
                self.resetLatency = max(self.rowDecoder.readLatency, self.columnDecoderLatency + self.chargeLatency) + g.cell.flashEraseTime
                # calculate the programming time, a.k.a. set here
                self.setLatency = max(self.rowDecoder.readLatency, self.columnDecoderLatency + self.chargeLatency) + g.cell.flashProgramTime
                # use the programming latency as the write latency for SLC NAND
                self.writeLatency = self.setLatency
            else:  # MLC NAND
                # TO-DO
                pass

    def CalculatePower(self):
        """Calculate power consumption"""
        if not self.initialized:
            print("[Subarray] Error: Require initialization first!")
        elif self.invalid:
            self.readDynamicEnergy = self.writeDynamicEnergy = self.leakage = g.invalid_value
        else:
            self.precharger.CalculatePower()
            self.rowDecoder.CalculatePower()
            self.bitlineMuxDecoder.CalculatePower()
            self.senseAmpMuxLev1Decoder.CalculatePower()
            self.senseAmpMuxLev2Decoder.CalculatePower()
            self.bitlineMux.CalculatePower()
            if self.internalSenseAmp:
                self.senseAmp.CalculatePower()
            self.senseAmpMuxLev1.CalculatePower()
            self.senseAmpMuxLev2.CalculatePower()

            if g.cell.memCellType == MemCellType.SRAM:
                # Codes below calculate the SRAM bitline power
                self.readDynamicEnergy = ((self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousPowerCalculation) *
                                        self.voltagePrecharge * self.voltagePrecharge * self.numColumn)
                self.writeDynamicEnergy = ((self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousPowerCalculation) *
                                         self.voltagePrecharge * self.voltagePrecharge * self.numColumn // self.muxSenseAmp // self.muxOutputLev1 // self.muxOutputLev2)
                self.leakage = (calculate_gate_leakage(INV, 1, g.cell.widthSRAMCellNMOS * g.tech.featureSize,
                                                      g.cell.widthSRAMCellPMOS * g.tech.featureSize, g.inputParameter.temperature, g.tech) *
                              g.tech.vdd * 2)  # two inverters per SRAM cell
                self.leakage += (calculate_gate_leakage(INV, 1, g.cell.widthAccessCMOS * g.tech.featureSize, 0,
                                                       g.inputParameter.temperature, g.tech) * g.tech.vdd)  # two accesses NMOS, but combined as one with vdd crossed
                self.leakage *= self.numRow * self.numColumn

            elif g.cell.memCellType == MemCellType.DRAM or g.cell.memCellType == MemCellType.eDRAM:
                # Codes below calculate the DRAM bitline power
                self.readDynamicEnergy = ((self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousPowerCalculation) *
                                        self.senseVoltage * g.devtech.vdd * self.numColumn)
                self.refreshDynamicEnergy = self.readDynamicEnergy
                writeVoltage = g.cell.resetVoltage  # should also equal to setVoltage, for DRAM, it is Vdd
                self.writeDynamicEnergy = ((self.capBitline + self.bitlineMux.capForPreviousPowerCalculation) *
                                         writeVoltage * writeVoltage * self.numColumn)
                self.leakage = self.readDynamicEnergy / DRAM_REFRESH_PERIOD * self.numRow

            elif (g.cell.memCellType == MemCellType.MRAM or g.cell.memCellType == MemCellType.PCRAM or
                  g.cell.memCellType == MemCellType.memristor or g.cell.memCellType == MemCellType.FBRAM):
                if g.cell.readMode == False:  # current-sensing
                    # Use ICCAD 2009 model
                    resBitlineMux = self.bitlineMux.resNMOSPassTransistor
                    vpreMin = g.cell.readVoltage * resBitlineMux / (resBitlineMux + self.resBitline + self.resMemCellOn)
                    vpreMax = g.cell.readVoltage * (resBitlineMux + self.resBitline) / (resBitlineMux + self.resBitline + self.resMemCellOn)
                    self.readDynamicEnergy = (self.capCellAccess * vpreMax * vpreMax + self.bitlineMux.capForPreviousPowerCalculation *
                                            vpreMin * vpreMin + self.capBitline * (vpreMax * vpreMax + vpreMin * vpreMin + vpreMax * vpreMin) / 3)
                    self.readDynamicEnergy *= self.numColumn
                else:  # voltage-sensing
                    self.readDynamicEnergy = ((self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousPowerCalculation) *
                                            (self.voltagePrecharge * self.voltagePrecharge - self.voltageMemCellOn * self.voltageMemCellOn) * self.numColumn)

                if g.cell.readPower == 0:
                    self.cellReadEnergy = 2 * g.cell.CalculateReadPower() * self.senseAmp.readLatency  # x2 is because of the reference cell
                else:
                    self.cellReadEnergy = 2 * g.cell.readPower * self.senseAmp.readLatency
                self.cellReadEnergy *= self.numColumn // self.muxSenseAmp // self.muxOutputLev1 // self.muxOutputLev2

                # Ignore the dynamic transition during the SET/RESET operation
                # Assume that the cell resistance keeps high for worst-case power estimation
                g.cell.CalculateWriteEnergy()

                resetEnergyPerBit = g.cell.resetEnergy
                setEnergyPerBit = g.cell.setEnergy
                if g.cell.setMode:
                    setEnergyPerBit += (self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousPowerCalculation) * g.cell.setVoltage * g.cell.setVoltage
                else:
                    setEnergyPerBit += (self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousPowerCalculation) * g.tech.vdd * g.tech.vdd
                if g.cell.resetMode:
                    resetEnergyPerBit += (self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousPowerCalculation) * g.cell.resetVoltage * g.cell.resetVoltage
                else:
                    resetEnergyPerBit += (self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousPowerCalculation) * g.tech.vdd * g.tech.vdd

                if g.cell.memCellType == MemCellType.PCRAM:  # PCRAM write energy
                    if g.inputParameter.writeScheme == WriteScheme.write_and_verify:
                        # TO-DO: write and verify programming
                        pass
                    else:
                        self.cellResetEnergy = resetEnergyPerBit * self.numColumn // self.muxSenseAmp // self.muxOutputLev1 // self.muxOutputLev2
                        self.cellSetEnergy = setEnergyPerBit * self.numColumn // self.muxSenseAmp // self.muxOutputLev1 // self.muxOutputLev2
                        self.cellResetEnergy /= SHAPER_EFFICIENCY_CONSERVATIVE
                        self.cellSetEnergy /= SHAPER_EFFICIENCY_CONSERVATIVE  # Due to the shaper inefficiency
                        self.writeDynamicEnergy = max(self.cellResetEnergy, self.cellSetEnergy)
                elif g.cell.memCellType == MemCellType.FBRAM:  # FBRAM write energy
                    self.cellResetEnergy = resetEnergyPerBit * self.numColumn // self.muxSenseAmp // self.muxOutputLev1 // self.muxOutputLev2
                    self.cellSetEnergy = setEnergyPerBit * self.numColumn // self.muxSenseAmp // self.muxOutputLev1 // self.muxOutputLev2
                    self.cellResetEnergy /= SHAPER_EFFICIENCY_AGGRESSIVE
                    self.cellSetEnergy /= SHAPER_EFFICIENCY_AGGRESSIVE  # Due to the shaper inefficiency
                    self.writeDynamicEnergy = max(self.cellResetEnergy, self.cellSetEnergy)
                else:  # MRAM and memristor write energy
                    if g.cell.accessType == CellAccessType.diode_access or g.cell.accessType == CellAccessType.none_access:
                        if g.inputParameter.writeScheme == WriteScheme.erase_before_reset or g.inputParameter.writeScheme == WriteScheme.erase_before_set:
                            self.cellResetEnergy = resetEnergyPerBit * self.numColumn // self.muxSenseAmp // self.muxOutputLev1 // self.muxOutputLev2
                            self.cellSetEnergy = setEnergyPerBit * self.numColumn // self.muxSenseAmp // self.muxOutputLev1 // self.muxOutputLev2
                            self.writeDynamicEnergy = self.cellResetEnergy + self.cellSetEnergy  # TO-DO: bug here, did you consider the write pattern?
                        else:  # write scheme = set_before_reset or reset_before_set
                            self.cellResetEnergy = resetEnergyPerBit * self.numColumn // self.muxSenseAmp // self.muxOutputLev1 // self.muxOutputLev2
                            self.cellSetEnergy = setEnergyPerBit * self.numColumn // self.muxSenseAmp // self.muxOutputLev1 // self.muxOutputLev2
                            self.writeDynamicEnergy = max(self.cellResetEnergy, self.cellSetEnergy)
                    else:
                        self.cellResetEnergy = resetEnergyPerBit * self.numColumn // self.muxSenseAmp // self.muxOutputLev1 // self.muxOutputLev2
                        self.cellSetEnergy = setEnergyPerBit * self.numColumn // self.muxSenseAmp // self.muxOutputLev1 // self.muxOutputLev2
                        self.writeDynamicEnergy = max(self.cellResetEnergy, self.cellSetEnergy)
                    self.cellResetEnergy /= SHAPER_EFFICIENCY_AGGRESSIVE
                    self.cellSetEnergy /= SHAPER_EFFICIENCY_AGGRESSIVE  # Due to the shaper inefficiency
                    self.writeDynamicEnergy /= SHAPER_EFFICIENCY_AGGRESSIVE
                self.leakage = 0  # TO-DO: cell leaks during read/write operation

            elif g.cell.memCellType == MemCellType.SLCNAND:
                # Calculate the NAND flash string length, which is the page count per block plus 2 (two select transistors)
                pageCount = g.inputParameter.flashBlockSize // g.inputParameter.pageSize
                stringLength = pageCount + 2

                # === READ energy ===
                # only the selected bitline is charged during the read operation, bitline is charged to Vpre
                self.readDynamicEnergy = ((self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousPowerCalculation) *
                                        self.voltagePrecharge * self.voltagePrecharge * self.numColumn)
                # tricky thing here!
                # In SLC NAND operation, SSL, GSL, and unselected wordlines in a block are charged to Vpass,
                # but the selected wordline is not charged, which is totally different from the other cases.
                self.rowDecoder.resetDynamicEnergy = self.rowDecoder.readDynamicEnergy
                self.rowDecoder.setDynamicEnergy = self.rowDecoder.readDynamicEnergy
                actualWordlineReadEnergy = (self.rowDecoder.readDynamicEnergy / g.tech.vdd / g.tech.vdd *
                                          g.cell.flashPassVoltage * g.cell.flashPassVoltage)  # approximate calculate, the wordline is charged to Vpass instead of Vdd
                actualWordlineReadEnergy = actualWordlineReadEnergy * (self.numRow // pageCount * stringLength - 1)  # except the selected wordline itself
                self.rowDecoder.readDynamicEnergy = actualWordlineReadEnergy  # update the correct value

                # === Programming (SET) energy ===
                # first calculate the source line energy (charged to Vdd), which is a part of "bitline" in this scenario
                self.setDynamicEnergy = ((self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousPowerCalculation) *
                                       g.cell.flashProgramVoltage * g.cell.flashProgramVoltage * self.numColumn)
                # add tunneling current
                # originally it should be multiplied by numColumn/muxSenseAmp/muxOutputLev1/muxOutputLev2,
                # but it is multiplied by numColumn here because all the unselected bitlines also need to precharge to Vdd
                self.setDynamicEnergy += (DELTA_V_TH * TUNNEL_CURRENT_FLOW * g.cell.area *
                                        g.tech.featureSize * g.tech.featureSize * g.cell.flashProgramTime * self.numColumn)
                # in programming, the SSL is precharged to Vdd, which is equal to the original value calculated
                # from row decoder
                actualWordlineSetEnergy = self.rowDecoder.setDynamicEnergy
                # however, the unselected wordlines in the same block have to precharge to Vpass
                actualWordlineSetEnergy += (self.rowDecoder.setDynamicEnergy / g.tech.vdd / g.tech.vdd *
                                          g.cell.flashPassVoltage * g.cell.flashPassVoltage * (self.numRow // pageCount * stringLength - 1))
                # And the selected wordline is precharged to Vpgm
                actualWordlineSetEnergy += (self.rowDecoder.setDynamicEnergy / g.tech.vdd / g.tech.vdd *
                                          g.cell.flashProgramVoltage * g.cell.flashProgramVoltage)
                self.rowDecoder.setDynamicEnergy = actualWordlineSetEnergy  # update the correct value

                # === Erase (RESET) energy ===
                # in erase, all the bitlines (selected or unselected) and the sourceline are precharged to Vera-Vbi
                self.resetDynamicEnergy = ((self.capCellAccess + self.capBitline + self.bitlineMux.capForPreviousPowerCalculation) *
                                         (g.cell.flashEraseVoltage - g.tech.buildInPotential) * (g.cell.flashEraseVoltage - g.tech.buildInPotential))
                self.resetDynamicEnergy *= (self.numColumn + 1)  # plus 1 is due to the source line
                # the p-well shared by the selected block is precharged to Vera
                wellJunctionCap = g.tech.capJunction * g.cell.area * g.tech.featureSize * g.tech.featureSize
                wellJunctionCap *= g.inputParameter.flashBlockSize  # one block shares the same well
                self.resetDynamicEnergy += wellJunctionCap * g.cell.flashEraseVoltage * g.cell.flashEraseVoltage
                # in erase, all the wordlines, SSL, and GSL in unselected block are precharged to Vera * beta
                # in selected block, SSL and GSL are precharged to Vera * beta
                # here beta is fixed at 0.8
                beta = 0.8
                actualWordlineResetEnergy = (self.rowDecoder.resetDynamicEnergy / g.tech.vdd / g.tech.vdd *
                                           (g.cell.flashEraseVoltage * beta) * (g.cell.flashEraseVoltage * beta))
                actualWordlineResetEnergy *= (self.numRow // pageCount * stringLength - pageCount)
                self.rowDecoder.resetDynamicEnergy = actualWordlineResetEnergy

                # let write energy to be the average energy per page
                self.rowDecoder.writeDynamicEnergy = (self.rowDecoder.setDynamicEnergy + self.rowDecoder.resetDynamicEnergy / pageCount) / 2
                self.writeDynamicEnergy = (self.setDynamicEnergy + self.resetDynamicEnergy / pageCount) / 2

                # Assume NAND flash cell does not consume any leakage
                self.leakage = 0
            else:  # MLC NAND
                # TO-DO
                pass

            if g.inputParameter.designTarget == DesignTarget.cache and g.inputParameter.cacheAccessMode != CacheAccessMode.sequential_access_mode:
                self.cellResetEnergy /= g.inputParameter.associativity
                self.cellSetEnergy /= g.inputParameter.associativity
                self.writeDynamicEnergy /= g.inputParameter.associativity
                self.resetDynamicEnergy /= g.inputParameter.associativity
                self.setDynamicEnergy /= g.inputParameter.associativity

            self.readDynamicEnergy += (self.cellReadEnergy + self.rowDecoder.readDynamicEnergy + self.bitlineMuxDecoder.readDynamicEnergy +
                                     self.senseAmpMuxLev1Decoder.readDynamicEnergy + self.senseAmpMuxLev2Decoder.readDynamicEnergy +
                                     self.precharger.readDynamicEnergy + self.bitlineMux.readDynamicEnergy + self.senseAmp.readDynamicEnergy +
                                     self.senseAmpMuxLev1.readDynamicEnergy + self.senseAmpMuxLev2.readDynamicEnergy)
            self.writeDynamicEnergy += (self.rowDecoder.writeDynamicEnergy + self.bitlineMuxDecoder.writeDynamicEnergy +
                                      self.senseAmpMuxLev1Decoder.writeDynamicEnergy + self.senseAmpMuxLev2Decoder.writeDynamicEnergy +
                                      self.bitlineMux.writeDynamicEnergy + self.senseAmp.writeDynamicEnergy +
                                      self.senseAmpMuxLev1.writeDynamicEnergy + self.senseAmpMuxLev2.writeDynamicEnergy)

            # Read all column energy + row decoder + sense amp + precharger is enough for one subarray row REF.
            self.refreshDynamicEnergy += (self.rowDecoder.readDynamicEnergy + self.precharger.readDynamicEnergy +
                                        self.senseAmp.readDynamicEnergy)
            self.refreshDynamicEnergy *= self.numRow  # Energy for this entire subarray

            # for assymetric RESET and SET latency calculation only
            self.setDynamicEnergy += (self.cellSetEnergy + self.rowDecoder.setDynamicEnergy + self.bitlineMuxDecoder.writeDynamicEnergy +
                                    self.senseAmpMuxLev1Decoder.writeDynamicEnergy + self.senseAmpMuxLev2Decoder.writeDynamicEnergy +
                                    self.bitlineMux.writeDynamicEnergy + self.senseAmp.writeDynamicEnergy +
                                    self.senseAmpMuxLev1.writeDynamicEnergy + self.senseAmpMuxLev2.writeDynamicEnergy)
            self.resetDynamicEnergy += (self.setDynamicEnergy + self.rowDecoder.resetDynamicEnergy + self.bitlineMuxDecoder.writeDynamicEnergy +
                                      self.senseAmpMuxLev1Decoder.writeDynamicEnergy + self.senseAmpMuxLev2Decoder.writeDynamicEnergy +
                                      self.bitlineMux.writeDynamicEnergy + self.senseAmp.writeDynamicEnergy +
                                      self.senseAmpMuxLev1.writeDynamicEnergy + self.senseAmpMuxLev2.writeDynamicEnergy)

            if g.cell.accessType == CellAccessType.diode_access or g.cell.accessType == CellAccessType.none_access:
                self.writeDynamicEnergy += (self.bitlineMux.writeDynamicEnergy + self.senseAmp.writeDynamicEnergy +
                                          self.senseAmpMuxLev1.writeDynamicEnergy + self.senseAmpMuxLev2.writeDynamicEnergy)
            self.leakage += (self.rowDecoder.leakage + self.bitlineMuxDecoder.leakage + self.senseAmpMuxLev1Decoder.leakage +
                           self.senseAmpMuxLev2Decoder.leakage + self.precharger.leakage + self.bitlineMux.leakage +
                           self.senseAmp.leakage + self.senseAmpMuxLev1.leakage + self.senseAmpMuxLev2.leakage)

    def PrintProperty(self):
        """Print subarray properties"""
        print("Subarray Properties:")
        super().PrintProperty()
        print(f"numRow:{self.numRow} numColumn:{self.numColumn}")
        print(f"lenWordline * lenBitline = {self.lenWordline*1e6}um * {self.lenBitline*1e6}um = {self.lenWordline * self.lenBitline * 1e6}mm^2")
        print(f"Row Decoder Area:{self.rowDecoder.height*1e6}um x {self.rowDecoder.width*1e6}um = {self.rowDecoder.area*1e6}mm^2")
        print(f"Sense Amplifier Area:{self.senseAmp.height*1e6}um x {self.senseAmp.width*1e6}um = {self.senseAmp.area*1e6}mm^2")
        print(f"Subarray Area Efficiency = {self.lenWordline * self.lenBitline / self.area * 100}%")
        print(f"bitlineDelay: {self.bitlineDelay*1e12}ps")
        print(f"chargeLatency: {self.chargeLatency*1e12}ps")
        print(f"columnDecoderLatency: {self.columnDecoderLatency*1e12}ps")

    def assign(self, rhs):
        """Assignment method to copy all properties from another SubArray instance"""
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
        self.cellResetEnergy = rhs.cellResetEnergy
        self.cellSetEnergy = rhs.cellSetEnergy
        self.leakage = rhs.leakage
        self.initialized = rhs.initialized
        self.numRow = rhs.numRow
        self.numColumn = rhs.numColumn
        self.multipleRowPerSet = rhs.multipleRowPerSet
        self.split = rhs.split
        self.muxSenseAmp = rhs.muxSenseAmp
        self.internalSenseAmp = rhs.internalSenseAmp
        self.muxOutputLev1 = rhs.muxOutputLev1
        self.muxOutputLev2 = rhs.muxOutputLev2
        self.areaOptimizationLevel = rhs.areaOptimizationLevel
        self.num3DLevels = rhs.num3DLevels

        self.voltageSense = rhs.voltageSense
        self.senseVoltage = rhs.senseVoltage
        self.numSenseAmp = rhs.numSenseAmp
        self.lenWordline = rhs.lenWordline
        self.lenBitline = rhs.lenBitline
        self.capWordline = rhs.capWordline
        self.capBitline = rhs.capBitline
        self.resWordline = rhs.resWordline
        self.resBitline = rhs.resBitline
        self.resCellAccess = rhs.resCellAccess
        self.capCellAccess = rhs.capCellAccess
        self.bitlineDelay = rhs.bitlineDelay
        self.chargeLatency = rhs.chargeLatency
        self.columnDecoderLatency = rhs.columnDecoderLatency
        self.bitlineDelayOn = rhs.bitlineDelayOn
        self.bitlineDelayOff = rhs.bitlineDelayOff
        self.resInSerialForSenseAmp = rhs.resInSerialForSenseAmp
        self.resEquivalentOn = rhs.resEquivalentOn
        self.resEquivalentOff = rhs.resEquivalentOff
        self.resMemCellOff = rhs.resMemCellOff
        self.resMemCellOn = rhs.resMemCellOn

        self.rowDecoder.assign(rhs.rowDecoder)
        self.bitlineMuxDecoder.assign(rhs.bitlineMuxDecoder)
        self.bitlineMux.assign(rhs.bitlineMux)
        self.senseAmpMuxLev1Decoder.assign(rhs.senseAmpMuxLev1Decoder)
        self.senseAmpMuxLev1.assign(rhs.senseAmpMuxLev1)
        self.senseAmpMuxLev2Decoder.assign(rhs.senseAmpMuxLev2Decoder)
        self.senseAmpMuxLev2.assign(rhs.senseAmpMuxLev2)
        self.precharger.assign(rhs.precharger)
        self.senseAmp.assign(rhs.senseAmp)

        return self
