#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

from FunctionUnit import FunctionUnit
from OutputDriver import OutputDriver
from formula import *
from constant import *
from typedef import BufferDesignTarget
import globals as g


class Precharger(FunctionUnit):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.enableLatency = 0.0
        self.outputDriver = OutputDriver()
        self.voltagePrecharge = 0.0
        self.capBitline = 0.0
        self.resBitline = 0.0
        self.capLoadInv = 0.0
        self.capOutputBitlinePrecharger = 0.0
        self.capWireLoadPerColumn = 0.0
        self.resWireLoadPerColumn = 0.0
        self.numColumn = 0
        self.widthPMOSBitlinePrecharger = 0.0
        self.widthPMOSBitlineEqual = 0.0
        self.widthInvNmos = 0.0
        self.widthInvPmos = 0.0
        self.capLoadPerColumn = 0.0
        self.rampInput = 0.0
        self.rampOutput = 0.0

    def Initialize(self, _voltagePrecharge, _numColumn, _capBitline, _resBitline):
        """Initialize precharger parameters"""
        if self.initialized:
            print("[Precharger] Warning: Already initialized!")

        self.voltagePrecharge = _voltagePrecharge
        self.numColumn = _numColumn
        self.capBitline = _capBitline
        self.resBitline = _resBitline
        self.capWireLoadPerColumn = g.cell.widthInFeatureSize * g.tech.featureSize * g.localWire.capWirePerUnit
        self.resWireLoadPerColumn = g.cell.widthInFeatureSize * g.tech.featureSize * g.localWire.resWirePerUnit
        self.widthInvNmos = MIN_NMOS_SIZE * g.tech.featureSize
        self.widthInvPmos = self.widthInvNmos * g.tech.pnSizeRatio
        self.widthPMOSBitlineEqual = MIN_NMOS_SIZE * g.tech.featureSize
        self.widthPMOSBitlinePrecharger = 6 * g.tech.featureSize

        self.capLoadInv = (calculate_gate_cap(self.widthPMOSBitlineEqual, g.tech) +
                          2 * calculate_gate_cap(self.widthPMOSBitlinePrecharger, g.tech) +
                          calculate_drain_cap(self.widthInvNmos, NMOS, g.tech.featureSize * 40, g.tech) +
                          calculate_drain_cap(self.widthInvPmos, PMOS, g.tech.featureSize * 40, g.tech))

        self.capOutputBitlinePrecharger = (calculate_drain_cap(self.widthPMOSBitlinePrecharger, PMOS,
                                                                g.tech.featureSize * 40, g.tech) +
                                           calculate_drain_cap(self.widthPMOSBitlineEqual, PMOS,
                                                              g.tech.featureSize * 40, g.tech))

        capInputInv = calculate_gate_cap(self.widthInvNmos, g.tech) + calculate_gate_cap(self.widthInvPmos, g.tech)
        self.capLoadPerColumn = capInputInv + self.capWireLoadPerColumn
        capLoadOutputDriver = self.numColumn * self.capLoadPerColumn

        # Always Latency First
        self.outputDriver.Initialize(1, capInputInv, capLoadOutputDriver, 0, True,
                                     BufferDesignTarget.latency_first, 0)

        self.initialized = True

    def CalculateArea(self):
        """Calculate precharger area"""
        if not self.initialized:
            print("[Precharger] Error: Require initialization first!")
        else:
            self.outputDriver.CalculateArea()

            area, hBitlinePrecharger, wBitlinePrecharger = calculate_gate_area(
                INV, 1, 0, self.widthPMOSBitlinePrecharger, g.tech.featureSize * 40, g.tech)

            area, hBitlineEqual, wBitlineEqual = calculate_gate_area(
                INV, 1, 0, self.widthPMOSBitlineEqual, g.tech.featureSize * 40, g.tech)

            area, hInverter, wInverter = calculate_gate_area(
                INV, 1, self.widthInvNmos, self.widthInvPmos, g.tech.featureSize * 40, g.tech)

            self.width = 2 * wBitlinePrecharger + wBitlineEqual
            self.width = max(self.width, wInverter)
            self.width *= self.numColumn
            self.width = max(self.width, self.outputDriver.width)

            self.height = max(hBitlinePrecharger, hBitlineEqual)
            self.height += hInverter
            self.height = max(self.height, self.outputDriver.height)

            self.area = self.height * self.width

    def CalculateRC(self):
        """Calculate RC parameters"""
        if not self.initialized:
            print("[Precharger] Error: Require initialization first!")
        else:
            self.outputDriver.CalculateRC()
            # More accurate RC model would include drain capacitances of Precharger and Equalization PMOS transistors

    def CalculateLatency(self, _rampInput):
        """Calculate latency"""
        if not self.initialized:
            print("[Precharger] Error: Require initialization first!")
        else:
            self.rampInput = _rampInput
            self.outputDriver.CalculateLatency(self.rampInput)
            self.enableLatency = self.outputDriver.readLatency

            # Calculate pull-down latency for inverter
            resPullDown = calculate_on_resistance(self.widthInvNmos, NMOS,
                                                  g.inputParameter.temperature, g.tech)
            tr = resPullDown * self.capLoadInv
            gm = calculate_transconductance(self.widthInvNmos, NMOS, g.tech)
            beta = 1 / (resPullDown * gm)
            temp, rampTemp = horowitz(tr, beta, self.outputDriver.rampOutput)
            self.enableLatency += temp

            # Calculate precharge latency
            self.readLatency = 0
            resPullUp = calculate_on_resistance(self.widthPMOSBitlinePrecharger, PMOS,
                                               g.inputParameter.temperature, g.tech)
            tau = (resPullUp * (self.capBitline + self.capOutputBitlinePrecharger) +
                   self.resBitline * self.capBitline / 2)
            gm = calculate_transconductance(self.widthPMOSBitlinePrecharger, PMOS, g.tech)
            beta = 1 / (resPullUp * gm)
            latency, rampOutput = horowitz(tau, beta, rampTemp)
            self.readLatency += latency
            self.rampOutput = rampOutput

            self.writeLatency = self.readLatency
            self.refreshLatency = self.readLatency

    def CalculatePower(self):
        """Calculate power"""
        if not self.initialized:
            print("[Precharger] Error: Require initialization first!")
        else:
            self.outputDriver.CalculatePower()

            # Leakage power
            self.leakage = self.outputDriver.leakage
            self.leakage += (self.numColumn * g.tech.vdd *
                           calculate_gate_leakage(INV, 1, self.widthInvNmos, self.widthInvPmos,
                                                 g.inputParameter.temperature, g.tech))
            self.leakage += (self.numColumn * self.voltagePrecharge *
                           calculate_gate_leakage(INV, 1, 0, self.widthPMOSBitlinePrecharger,
                                                 g.inputParameter.temperature, g.tech))

            # Dynamic energy
            # We don't count bitline precharge energy into account because it is a charging process
            self.readDynamicEnergy = self.outputDriver.readDynamicEnergy
            self.readDynamicEnergy += self.capLoadInv * g.tech.vdd * g.tech.vdd * self.numColumn
            self.writeDynamicEnergy = 0  # No precharging is needed during the write operation
            self.refreshDynamicEnergy = self.readDynamicEnergy

    def PrintProperty(self):
        """Print precharger properties"""
        print("Precharger Properties:")
        super().PrintProperty()

    def assign(self, rhs):
        """Assignment method to copy all properties from another Precharger instance"""
        self.height = rhs.height
        self.width = rhs.width
        self.area = rhs.area
        self.readLatency = rhs.readLatency
        self.writeLatency = rhs.writeLatency
        self.refreshLatency = rhs.refreshLatency
        self.readDynamicEnergy = rhs.readDynamicEnergy
        self.writeDynamicEnergy = rhs.writeDynamicEnergy
        self.refreshDynamicEnergy = rhs.refreshDynamicEnergy
        self.resetLatency = rhs.resetLatency
        self.setLatency = rhs.setLatency
        self.resetDynamicEnergy = rhs.resetDynamicEnergy
        self.setDynamicEnergy = rhs.setDynamicEnergy
        self.cellReadEnergy = rhs.cellReadEnergy
        self.cellSetEnergy = rhs.cellSetEnergy
        self.cellResetEnergy = rhs.cellResetEnergy
        self.leakage = rhs.leakage
        self.initialized = rhs.initialized

        # Deep copy the outputDriver
        self.outputDriver.assign(rhs.outputDriver)

        self.capBitline = rhs.capBitline
        self.resBitline = rhs.resBitline
        self.capLoadInv = rhs.capLoadInv
        self.capOutputBitlinePrecharger = rhs.capOutputBitlinePrecharger
        self.capWireLoadPerColumn = rhs.capWireLoadPerColumn
        self.resWireLoadPerColumn = rhs.resWireLoadPerColumn
        self.enableLatency = rhs.enableLatency
        self.numColumn = rhs.numColumn
        self.widthPMOSBitlinePrecharger = rhs.widthPMOSBitlinePrecharger
        self.widthPMOSBitlineEqual = rhs.widthPMOSBitlineEqual
        self.capLoadPerColumn = rhs.capLoadPerColumn
        self.rampInput = rhs.rampInput
        self.rampOutput = rhs.rampOutput
        self.voltagePrecharge = rhs.voltagePrecharge
        self.widthInvNmos = rhs.widthInvNmos
        self.widthInvPmos = rhs.widthInvPmos

        return self
