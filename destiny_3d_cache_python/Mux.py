# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

from FunctionUnit import FunctionUnit
from formula import *
from constant import *
from typedef import MemCellType
import globals as g


class Mux(FunctionUnit):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.capForPreviousPowerCalculation = 0.0
        self.capForPreviousDelayCalculation = 0.0
        self.capNMOSPassTransistor = 0.0
        self.resNMOSPassTransistor = 0.0
        self.numInput = 0
        self.numMux = 0
        self.capLoad = 0.0
        self.capInputNextStage = 0.0
        self.minDriverCurrent = 0.0
        self.capOutput = 0.0
        self.widthNMOSPassTransistor = 0.0
        self.rampInput = 0.0
        self.rampOutput = 0.0

    def Initialize(self, _numInput, _numMux, _capLoad, _capInputNextStage, _minDriverCurrent):
        if self.initialized:
            print("[Mux] Warning: Already initialized!")

        self.numInput = _numInput
        self.numMux = _numMux
        self.capLoad = _capLoad
        self.capInputNextStage = _capInputNextStage
        self.minDriverCurrent = _minDriverCurrent

        if (self.numInput > 1) and (self.numMux > 0):
            minNMOSWidth = self.minDriverCurrent / g.tech.currentOnNmos[g.inputParameter.temperature - 300]
            if (g.cell.memCellType == MemCellType.MRAM or
                g.cell.memCellType == MemCellType.PCRAM or
                g.cell.memCellType == MemCellType.memristor):
                # Mux resistance should be small enough for voltage dividing
                maxResNMOSPassTransistor = g.cell.resistanceOn * IR_DROP_TOLERANCE
                self.widthNMOSPassTransistor = (CalculateOnResistance(g.tech.featureSize, NMOS,
                                                                      g.inputParameter.temperature, g.tech) *
                                               g.tech.featureSize / maxResNMOSPassTransistor)
                if self.widthNMOSPassTransistor > g.inputParameter.maxNmosSize * g.tech.featureSize:
                    self.widthNMOSPassTransistor = g.inputParameter.maxNmosSize * g.tech.featureSize
                self.widthNMOSPassTransistor = MAX(MAX(self.widthNMOSPassTransistor, minNMOSWidth),
                                                   6 * MIN_NMOS_SIZE * g.tech.featureSize)
            else:
                self.widthNMOSPassTransistor = MAX(6 * MIN_NMOS_SIZE * g.tech.featureSize, minNMOSWidth)

        self.initialized = True

    def CalculateArea(self):
        if not self.initialized:
            print("[Mux] Error: Require initialization first!")
        else:
            if (self.numInput > 1) and (self.numMux > 0):
                area, h, w = CalculateGateArea(INV, 1, self.widthNMOSPassTransistor, 0,
                                                g.tech.featureSize * 40, g.tech)
                self.width = self.numMux * self.numInput * w
                self.height = h
                self.area = self.width * self.height
            else:
                self.height = self.width = self.area = 0.0

    def CalculateRC(self):
        if not self.initialized:
            print("[Mux] Error: Require initialization first!")
        else:
            if (self.numInput > 1) and (self.numMux > 0):
                self.capNMOSPassTransistor = CalculateDrainCap(self.widthNMOSPassTransistor, NMOS,
                                                               g.tech.featureSize * 40, g.tech)
                self.capForPreviousPowerCalculation = self.capNMOSPassTransistor
                self.capOutput = self.numInput * self.capNMOSPassTransistor
                self.capForPreviousDelayCalculation = (self.capOutput + self.capNMOSPassTransistor +
                                                       self.capLoad)
                self.resNMOSPassTransistor = CalculateOnResistance(self.widthNMOSPassTransistor, NMOS,
                                                                   g.inputParameter.temperature, g.tech)

    def CalculateLatency(self, _rampInput):  # rampInput is actually useless in Mux module
        if not self.initialized:
            print("[Mux] Error: Require initialization first!")
        else:
            if (self.numInput > 1) and (self.numMux > 0):
                self.rampInput = _rampInput
                tr = self.resNMOSPassTransistor * (self.capOutput + self.capLoad)
                self.readLatency = 2.3 * tr
                self.writeLatency = self.readLatency
            else:
                self.readLatency = self.writeLatency = 0.0

    def CalculatePower(self):
        if not self.initialized:
            print("[Mux] Error: Require initialization first!")
        else:
            if (self.numInput > 1) and (self.numMux > 0):
                self.leakage = 0.0  # TO-DO
                self.readDynamicEnergy = ((self.capOutput + self.capInputNextStage) *
                                         g.tech.vdd * (g.tech.vdd - g.tech.vth))
                self.readDynamicEnergy *= self.numMux  # worst-case dynamic power analysis
                self.writeDynamicEnergy = self.readDynamicEnergy
            else:
                self.readDynamicEnergy = self.writeDynamicEnergy = self.leakage = 0.0

    def PrintProperty(self):
        print("Mux Properties:")
        super().PrintProperty()

    def copy_from(self, rhs):
        """
        Copy assignment operator equivalent (corresponds to operator= in C++)
        Copies all properties from another Mux instance to this one.
        """
        # Copy FunctionUnit properties
        self.height = rhs.height
        self.width = rhs.width
        self.area = rhs.area
        self.readLatency = rhs.readLatency
        self.writeLatency = rhs.writeLatency
        self.readDynamicEnergy = rhs.readDynamicEnergy
        self.writeDynamicEnergy = rhs.writeDynamicEnergy
        self.leakage = rhs.leakage

        # Copy Mux-specific properties
        self.initialized = rhs.initialized
        self.numInput = rhs.numInput
        self.numMux = rhs.numMux
        self.capLoad = rhs.capLoad
        self.capInputNextStage = rhs.capInputNextStage
        self.minDriverCurrent = rhs.minDriverCurrent
        self.capOutput = rhs.capOutput
        self.widthNMOSPassTransistor = rhs.widthNMOSPassTransistor
        self.resNMOSPassTransistor = rhs.resNMOSPassTransistor
        self.capNMOSPassTransistor = rhs.capNMOSPassTransistor
        self.capForPreviousDelayCalculation = rhs.capForPreviousDelayCalculation
        self.capForPreviousPowerCalculation = rhs.capForPreviousPowerCalculation
        self.rampInput = rhs.rampInput
        self.rampOutput = rhs.rampOutput

        return self
