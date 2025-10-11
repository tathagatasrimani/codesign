# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

from FunctionUnit import FunctionUnit
from formula import *
from constant import *
import globals as g
import math


class Comparator(FunctionUnit):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.numTagBits = 0
        self.capLoad = 0.0
        self.widthNMOSInv = [0.0] * COMPARATOR_INV_CHAIN_LEN
        self.widthPMOSInv = [0.0] * COMPARATOR_INV_CHAIN_LEN
        self.widthNMOSComp = 0.0
        self.widthPMOSComp = 0.0
        self.capInput = [0.0] * COMPARATOR_INV_CHAIN_LEN
        self.capOutput = [0.0] * COMPARATOR_INV_CHAIN_LEN
        self.capBottom = 0.0
        self.capTop = 0.0
        self.resBottom = 0.0
        self.resTop = 0.0
        self.rampInput = 0.0
        self.rampOutput = 1e40

    def Initialize(self, _numTagBits, _capLoad):
        if self.initialized:
            print("[Comparator] Warning: Already initialized!")

        self.numTagBits = _numTagBits // 4  # Assuming there are 4 quarter comparators. input tagbits is already a multiple of 4
        self.capLoad = _capLoad
        self.widthNMOSInv[0] = 7.5 * g.tech.featureSize
        self.widthPMOSInv[0] = 12.5 * g.tech.featureSize
        self.widthNMOSInv[1] = 15 * g.tech.featureSize
        self.widthPMOSInv[1] = 25 * g.tech.featureSize
        self.widthNMOSInv[2] = 30 * g.tech.featureSize
        self.widthPMOSInv[2] = 50 * g.tech.featureSize
        self.widthNMOSInv[3] = 50 * g.tech.featureSize
        self.widthPMOSInv[3] = 100 * g.tech.featureSize
        self.widthNMOSComp = 12.5 * g.tech.featureSize
        self.widthPMOSComp = 37.5 * g.tech.featureSize

        self.initialized = True

    def CalculateArea(self):
        if not self.initialized:
            print("[Comparator] Error: Require initialization first!")
        else:
            totalHeight = 0.0
            totalWidth = 0.0
            for i in range(COMPARATOR_INV_CHAIN_LEN):
                area, h, w = CalculateGateArea(INV, 1, self.widthNMOSInv[i], self.widthPMOSInv[i],
                                                g.tech.featureSize * 40, g.tech)
                totalHeight = MAX(totalHeight, h)
                totalWidth += w
            area, h, w = CalculateGateArea(NAND, 2, self.widthNMOSComp, 0,
                                            g.tech.featureSize * 40, g.tech)
            totalHeight += h
            totalWidth = MAX(totalWidth, self.numTagBits * w)
            self.height = totalHeight * 1  # 4 quarter comparators can have different placement, here assumes 1*4
            self.width = totalWidth * 4
            self.area = self.height * self.width

    def CalculateRC(self):
        if not self.initialized:
            print("[Comparator] Error: Require initialization first!")
        else:
            for i in range(COMPARATOR_INV_CHAIN_LEN):
                capInput, capOutput = CalculateGateCapacitance(INV, 1, self.widthNMOSInv[i], self.widthPMOSInv[i],
                                                                g.tech.featureSize * MAX_TRANSISTOR_HEIGHT, g.tech)
                self.capInput[i] = capInput
                self.capOutput[i] = capOutput

            capTemp, capComp = CalculateGateCapacitance(NAND, 2, self.widthNMOSComp, 0,
                                                        g.tech.featureSize * 40, g.tech)
            self.capBottom = self.capOutput[COMPARATOR_INV_CHAIN_LEN - 1] + self.numTagBits * capComp
            self.capTop = (self.numTagBits * capComp +
                          CalculateDrainCap(self.widthPMOSComp, PMOS,
                                           g.tech.featureSize * MAX_TRANSISTOR_HEIGHT, g.tech) +
                          self.capLoad)
            self.resBottom = CalculateOnResistance(self.widthNMOSInv[COMPARATOR_INV_CHAIN_LEN - 1],
                                                   NMOS, g.inputParameter.temperature, g.tech)
            self.resTop = 2 * CalculateOnResistance(self.widthNMOSComp, NMOS,
                                                    g.inputParameter.temperature, g.tech)

    def CalculateLatency(self, _rampInput):
        if not self.initialized:
            print("[Comparator] Error: Require initialization first!")
        else:
            self.rampInput = _rampInput
            self.readLatency = 0.0

            for i in range(COMPARATOR_INV_CHAIN_LEN - 1):
                resPullDown = CalculateOnResistance(self.widthNMOSInv[i], NMOS,
                                                    g.inputParameter.temperature, g.tech)
                capNode = self.capOutput[i] + self.capInput[i + 1]
                tr = resPullDown * capNode
                gm = CalculateTransconductance(self.widthNMOSInv[i], NMOS, g.tech)
                beta = 1 / (resPullDown * gm)
                delay, temp = horowitz(tr, beta, self.rampInput)
                self.readLatency += delay
                self.rampInput = temp  # for next stage

            tr = self.resBottom * self.capBottom + (self.resBottom + self.resTop) * self.capTop
            delay, rampOutput = horowitz(tr, 0, self.rampInput)
            self.readLatency += delay
            self.rampOutput = rampOutput
            self.rampInput = _rampInput
            self.writeLatency = self.readLatency

    def CalculatePower(self):
        if not self.initialized:
            print("[Comparator] Error: Require initialization first!")
        else:
            # Leakage power
            self.leakage = 0.0
            for i in range(COMPARATOR_INV_CHAIN_LEN):
                self.leakage += (CalculateGateLeakage(INV, 1, self.widthNMOSInv[i],
                                                      self.widthPMOSInv[i],
                                                      g.inputParameter.temperature, g.tech) *
                               g.tech.vdd)
            self.leakage += (self.numTagBits *
                           CalculateGateLeakage(NAND, 2, self.widthNMOSComp, 0,
                                               g.inputParameter.temperature, g.tech) *
                           g.tech.vdd)
            self.leakage *= 4

            # Dynamic energy
            self.readDynamicEnergy = 0.0
            for i in range(COMPARATOR_INV_CHAIN_LEN - 1):
                capNode = self.capOutput[i] + self.capInput[i + 1]
                self.readDynamicEnergy += capNode * g.tech.vdd * g.tech.vdd
            self.readDynamicEnergy += (self.capBottom + self.capTop) * g.tech.vdd * g.tech.vdd
            self.readDynamicEnergy *= 4
            self.writeDynamicEnergy = self.readDynamicEnergy

    def PrintProperty(self):
        print("Comparator Properties:")
        super().PrintProperty()

    def assign(self, rhs):
        """Python implementation of assignment operator (C++ operator=)"""
        # Copy FunctionUnit base class properties
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

        # Copy Comparator-specific properties
        self.initialized = rhs.initialized
        self.numTagBits = rhs.numTagBits
        self.capLoad = rhs.capLoad
        self.widthNMOSComp = rhs.widthNMOSComp
        self.widthPMOSComp = rhs.widthPMOSComp
        self.capBottom = rhs.capBottom
        self.capTop = rhs.capTop
        self.resBottom = rhs.resBottom
        self.resTop = rhs.resTop

        # Copy array properties
        for i in range(COMPARATOR_INV_CHAIN_LEN):
            self.widthNMOSInv[i] = rhs.widthNMOSInv[i]
            self.widthPMOSInv[i] = rhs.widthPMOSInv[i]
            self.capInput[i] = rhs.capInput[i]
            self.capOutput[i] = rhs.capOutput[i]

        self.rampInput = rhs.rampInput
        self.rampOutput = rhs.rampOutput

        return self
