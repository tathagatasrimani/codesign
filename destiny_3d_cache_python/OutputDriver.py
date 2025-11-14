#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

from FunctionUnit import FunctionUnit
from formula import *
from constant import *
from typedef import BufferDesignTarget
import globals as g
import math


class OutputDriver(FunctionUnit):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.invalid = False
        self.logicEffort = 0.0
        self.inputCap = 0.0
        self.outputCap = 0.0
        self.outputRes = 0.0
        self.inv = False
        self.numStage = 0
        self.areaOptimizationLevel = BufferDesignTarget.latency_first
        self.minDriverCurrent = 0.0
        self.widthNMOS = [0.0] * MAX_INV_CHAIN_LEN
        self.widthPMOS = [0.0] * MAX_INV_CHAIN_LEN
        self.capInput = [0.0] * MAX_INV_CHAIN_LEN
        self.capOutput = [0.0] * MAX_INV_CHAIN_LEN
        self.rampInput = 0.0
        self.rampOutput = 0.0

    def Initialize(self, _logicEffort, _inputCap, _outputCap, _outputRes,
                  _inv, _areaOptimizationLevel, _minDriverCurrent):
        if self.initialized:
            print("[Output Driver] Warning: Already initialized!")

        self.logicEffort = _logicEffort
        self.inputCap = _inputCap
        self.outputCap = _outputCap
        self.outputRes = _outputRes
        self.inv = _inv
        self.areaOptimizationLevel = _areaOptimizationLevel
        self.minDriverCurrent = _minDriverCurrent

        minNMOSDriverWidth = self.minDriverCurrent / g.tech.currentOnNmos[g.inputParameter.temperature - 300]
        minNMOSDriverWidth = MAX(MIN_NMOS_SIZE * g.tech.featureSize, minNMOSDriverWidth)

        if minNMOSDriverWidth > g.inputParameter.maxNmosSize * g.tech.featureSize:
            self.invalid = True
            return

        if self.areaOptimizationLevel == BufferDesignTarget.latency_first:
            F = MAX(1, self.logicEffort * self.outputCap / self.inputCap)
            optimalNumStage = MAX(0, int(math.log(F) / math.log(OPT_F) + 0.5) - 1)

            if (optimalNumStage % 2) ^ self.inv:  # If odd, add 1
                optimalNumStage += 1

            if optimalNumStage > MAX_INV_CHAIN_LEN:
                if WARNING:
                    print("[WARNING] Exceed maximum inverter chain length!")
                optimalNumStage = MAX_INV_CHAIN_LEN

            self.numStage = optimalNumStage

            f = pow(F, 1.0 / (optimalNumStage + 1))
            inputCapLast = self.outputCap / f

            self.widthNMOS[optimalNumStage - 1] = MAX(MIN_NMOS_SIZE * g.tech.featureSize,
                                                      inputCapLast / CalculateGateCap(1, g.tech) /
                                                      (1.0 + g.tech.pnSizeRatio))

            if self.widthNMOS[optimalNumStage - 1] > g.inputParameter.maxNmosSize * g.tech.featureSize:
                if WARNING:
                    print("[WARNING] Exceed maximum NMOS size!")
                self.widthNMOS[optimalNumStage - 1] = g.inputParameter.maxNmosSize * g.tech.featureSize
                # re-Calculate the logic effort
                capLastStage = CalculateGateCap((1 + g.tech.pnSizeRatio) * g.inputParameter.maxNmosSize * g.tech.featureSize, g.tech)
                F = self.logicEffort * capLastStage / self.inputCap
                f = pow(F, 1.0 / optimalNumStage)

            if self.widthNMOS[optimalNumStage - 1] < minNMOSDriverWidth:
                self.areaOptimizationLevel = BufferDesignTarget.latency_area_trade_off
            else:
                self.widthPMOS[optimalNumStage - 1] = self.widthNMOS[optimalNumStage - 1] * g.tech.pnSizeRatio

                for i in range(optimalNumStage - 2, -1, -1):
                    self.widthNMOS[i] = self.widthNMOS[i + 1] / f
                    if self.widthNMOS[i] < MIN_NMOS_SIZE * g.tech.featureSize:
                        if WARNING:
                            print("[WARNING] Exceed minimum NMOS size!")
                        self.widthNMOS[i] = MIN_NMOS_SIZE * g.tech.featureSize
                    self.widthPMOS[i] = self.widthNMOS[i] * g.tech.pnSizeRatio

        if self.areaOptimizationLevel == BufferDesignTarget.latency_area_trade_off:
            newOutputCap = CalculateGateCap(minNMOSDriverWidth, g.tech) * (1.0 + g.tech.pnSizeRatio)
            F = MAX(1, self.logicEffort * newOutputCap / self.inputCap)
            optimalNumStage = MAX(0, int(math.log(F) / math.log(OPT_F) + 0.5) - 1)

            if not ((optimalNumStage % 2) ^ self.inv):  # If even, add 1
                optimalNumStage += 1

            if optimalNumStage > MAX_INV_CHAIN_LEN:
                if WARNING:
                    print("[WARNING] Exceed maximum inverter chain length!")
                optimalNumStage = MAX_INV_CHAIN_LEN

            self.numStage = optimalNumStage + 1

            self.widthNMOS[optimalNumStage] = minNMOSDriverWidth
            self.widthPMOS[optimalNumStage] = self.widthNMOS[optimalNumStage] * g.tech.pnSizeRatio

            f = pow(F, 1.0 / (optimalNumStage + 1))

            for i in range(optimalNumStage - 1, -1, -1):
                self.widthNMOS[i] = self.widthNMOS[i + 1] / f
                if self.widthNMOS[i] < MIN_NMOS_SIZE * g.tech.featureSize:
                    if WARNING:
                        print("[WARNING] Exceed minimum NMOS size!")
                    self.widthNMOS[i] = MIN_NMOS_SIZE * g.tech.featureSize
                self.widthPMOS[i] = self.widthNMOS[i] * g.tech.pnSizeRatio

        elif self.areaOptimizationLevel == BufferDesignTarget.area_first:
            optimalNumStage = 1
            self.numStage = 1
            self.widthNMOS[optimalNumStage - 1] = MAX(MIN_NMOS_SIZE * g.tech.featureSize, minNMOSDriverWidth)
            if self.widthNMOS[optimalNumStage - 1] > AREA_OPT_CONSTRAIN * g.inputParameter.maxNmosSize * g.tech.featureSize:
                self.invalid = True
                return
            self.widthPMOS[optimalNumStage - 1] = self.widthNMOS[optimalNumStage - 1] * g.tech.pnSizeRatio

        # Restore the original buffer design style
        self.areaOptimizationLevel = _areaOptimizationLevel

        self.initialized = True

    def CalculateArea(self):
        if not self.initialized:
            print("[Output Driver] Error: Require initialization first!")
        elif self.invalid:
            self.height = self.width = self.area = g.invalid_value
        else:
            totalHeight = 0.0
            totalWidth = 0.0
            for i in range(self.numStage):
                area, h, w = CalculateGateArea(INV, 1, self.widthNMOS[i], self.widthPMOS[i],
                                               g.tech.featureSize * 40, g.tech)
                totalHeight = MAX(totalHeight, h)
                totalWidth += w
            self.height = totalHeight
            self.width = totalWidth
            self.area = self.height * self.width

    def CalculateRC(self):
        if not self.initialized:
            print("[Output Driver] Error: Require initialization first!")
        elif self.invalid:
            pass
        elif self.numStage == 0:
            self.capInput[0] = 0.0
        else:
            for i in range(self.numStage):
                capInput, capOutput = CalculateGateCapacitance(INV, 1, self.widthNMOS[i], self.widthPMOS[i],
                                                               g.tech.featureSize * MAX_TRANSISTOR_HEIGHT, g.tech)
                self.capInput[i] = capInput
                self.capOutput[i] = capOutput

    def CalculateLatency(self, _rampInput):
        if not self.initialized:
            print("[Output Driver] Error: Require initialization first!")
        elif self.invalid:
            self.readLatency = self.writeLatency = g.invalid_value
        else:
            self.rampInput = _rampInput
            self.readLatency = 0.0

            for i in range(self.numStage - 1):
                resPullDown = CalculateOnResistance(self.widthNMOS[i], NMOS,
                                                    g.inputParameter.temperature, g.tech)
                capLoad = self.capOutput[i] + self.capInput[i + 1]
                tr = resPullDown * capLoad
                gm = CalculateTransconductance(self.widthNMOS[i], NMOS, g.tech)
                beta = 1 / (resPullDown * gm)
                delay, temp = horowitz(tr, beta, self.rampInput)
                self.readLatency += delay
                self.rampInput = temp  # for next stage

            # Last level inverter
            resPullDown = CalculateOnResistance(self.widthNMOS[self.numStage - 1], NMOS,
                                                g.inputParameter.temperature, g.tech)
            capLoad = self.capOutput[self.numStage - 1] + self.outputCap
            tr = resPullDown * capLoad + self.outputCap * self.outputRes / 2
            gm = CalculateTransconductance(self.widthNMOS[self.numStage - 1], NMOS, g.tech)
            beta = 1 / (resPullDown * gm)
            delay, rampOutput = horowitz(tr, beta, self.rampInput)
            self.readLatency += delay
            self.rampOutput = rampOutput
            self.rampInput = _rampInput
            self.writeLatency = self.readLatency

    def CalculatePower(self):
        if not self.initialized:
            print("[Output Driver] Error: Require initialization first!")
        elif self.invalid:
            self.readDynamicEnergy = self.writeDynamicEnergy = self.leakage = g.invalid_value
        else:
            # Leakage power
            self.leakage = 0.0
            for i in range(self.numStage):
                self.leakage += (CalculateGateLeakage(INV, 1, self.widthNMOS[i], self.widthPMOS[i],
                                                      g.inputParameter.temperature, g.tech) *
                               g.tech.vdd)
            # Dynamic energy
            self.readDynamicEnergy = 0.0
            for i in range(self.numStage - 1):
                capLoad = self.capOutput[i] + self.capInput[i + 1]
                self.readDynamicEnergy += capLoad * g.tech.vdd * g.tech.vdd
            capLoad = self.capOutput[self.numStage - 1] + self.outputCap
            self.readDynamicEnergy += capLoad * g.tech.vdd * g.tech.vdd
            self.writeDynamicEnergy = self.readDynamicEnergy

    def PrintProperty(self):
        print("Output Driver Properties:")
        super().PrintProperty()
        print(f"Number of inverter stage: {self.numStage}")

    def assign(self, rhs):
        """Assignment method to copy all properties from another OutputDriver instance"""
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
        self.logicEffort = rhs.logicEffort
        self.inputCap = rhs.inputCap
        self.outputCap = rhs.outputCap
        self.outputRes = rhs.outputRes
        self.inv = rhs.inv
        self.numStage = rhs.numStage
        self.areaOptimizationLevel = rhs.areaOptimizationLevel
        self.minDriverCurrent = rhs.minDriverCurrent
        self.rampInput = rhs.rampInput
        self.rampOutput = rhs.rampOutput
        # Copy arrays
        self.widthNMOS = rhs.widthNMOS.copy()
        self.widthPMOS = rhs.widthPMOS.copy()
        self.capInput = rhs.capInput.copy()
        self.capOutput = rhs.capOutput.copy()
        return self
