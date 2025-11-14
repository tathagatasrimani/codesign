# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

import globals as g
from FunctionUnit import FunctionUnit
from OutputDriver import OutputDriver
from formula import (
    CalculateGateCap,
    CalculateGateArea,
    CalculateGateCapacitance,
    CalculateOnResistance,
    CalculateTransconductance,
    CalculateGateLeakage,
    horowitz
)
from constant import (
    MIN_NMOS_SIZE,
    NAND,
    NMOS,
    MAX_TRANSISTOR_HEIGHT
)
from typedef import BufferDesignTarget


class BasicDecoder(FunctionUnit):
    """
    BasicDecoder is a decoder module that inherits from FunctionUnit.
    It can be configured as an inverter (1 address bit) or as a NAND-based
    decoder (2 or 3 input NAND gates).
    """

    def __init__(self):
        """Constructor - initializes all properties"""
        super().__init__()
        self.initialized = False
        self.outputDriver = OutputDriver()
        self.capLoad = 0.0
        self.resLoad = 0.0
        self.numNandInput = 0
        self.numNandGate = 0
        self.widthNandN = 0.0
        self.widthNandP = 0.0
        self.capNandInput = 0.0
        self.capNandOutput = 0.0
        self.rampInput = 0.0
        self.rampOutput = 0.0

    def Initialize(self, _numAddressBit, _capLoad, _resLoad):
        """
        Initialize the BasicDecoder with the given parameters.

        Args:
            _numAddressBit: Number of address bits (1 for inverter, 2 for NAND2, 3+ for NAND3)
            _capLoad: Load capacitance in F
            _resLoad: Load resistance in ohm
        """
        # might be re-initialized by predecodeblock
        if _numAddressBit == 1:
            self.numNandInput = 0
        else:
            self.numNandInput = _numAddressBit

        self.capLoad = _capLoad
        self.resLoad = _resLoad

        if self.numNandInput == 0:
            # Inverter configuration
            self.numNandGate = 0
            logicEffortInv = 1
            widthInvN = MIN_NMOS_SIZE * g.tech.featureSize
            widthInvP = g.tech.pnSizeRatio * MIN_NMOS_SIZE * g.tech.featureSize
            capInv = CalculateGateCap(widthInvN, g.tech) + CalculateGateCap(widthInvP, g.tech)
            self.outputDriver.Initialize(logicEffortInv, capInv, self.capLoad, self.resLoad,
                                        True, BufferDesignTarget.latency_first, 0)  # Always Latency First
        else:
            # NAND configuration
            if self.numNandInput == 2:  # NAND2
                self.numNandGate = 4
                self.widthNandN = 2 * MIN_NMOS_SIZE * g.tech.featureSize
                logicEffortNand = (2 + g.tech.pnSizeRatio) / (1 + g.tech.pnSizeRatio)
            else:  # NAND3
                self.numNandGate = 8
                self.widthNandN = 3 * MIN_NMOS_SIZE * g.tech.featureSize
                logicEffortNand = (3 + g.tech.pnSizeRatio) / (1 + g.tech.pnSizeRatio)

            self.widthNandP = g.tech.pnSizeRatio * MIN_NMOS_SIZE * g.tech.featureSize
            capNand = CalculateGateCap(self.widthNandN, g.tech) + CalculateGateCap(self.widthNandP, g.tech)
            self.outputDriver.Initialize(logicEffortNand, capNand, self.capLoad, self.resLoad,
                                        True, BufferDesignTarget.latency_first, 0)  # Always Latency First

        self.initialized = True

    def CalculateArea(self):
        """Calculate the area of the BasicDecoder"""
        if not self.initialized:
            print("[Basic Decoder] Error: Require initialization first!")
        else:
            self.outputDriver.CalculateArea()
            if self.numNandInput == 0:
                # Inverter configuration
                self.height = 2 * self.outputDriver.height
                self.width = self.outputDriver.width
            else:
                # NAND configuration
                area, hNand, wNand = CalculateGateArea(NAND, self.numNandInput, self.widthNandN, self.widthNandP,
                                                        g.tech.featureSize * 40, g.tech)
                self.height = max(hNand, self.outputDriver.height)
                self.width = wNand + self.outputDriver.width
                self.height *= self.numNandGate
            self.area = self.height * self.width

    def CalculateRC(self):
        """Calculate the resistance and capacitance of the BasicDecoder"""
        if not self.initialized:
            print("[Basic Decoder] Error: Require initialization first!")
        else:
            self.outputDriver.CalculateRC()
            if self.numNandInput > 0:
                capInput, capOutput = CalculateGateCapacitance(NAND, self.numNandInput, self.widthNandN, self.widthNandP,
                                                                g.tech.featureSize * MAX_TRANSISTOR_HEIGHT, g.tech)
                self.capNandInput = capInput
                self.capNandOutput = capOutput

    def CalculateLatency(self, _rampInput):
        """
        Calculate the latency of the BasicDecoder.

        Args:
            _rampInput: Input ramp time
        """
        if not self.initialized:
            print("[Basic Decoder] Error: Require initialization first!")
        else:
            self.rampInput = _rampInput
            if self.numNandInput == 0:
                # Inverter configuration
                self.outputDriver.CalculateLatency(self.rampInput)
                self.readLatency = self.outputDriver.readLatency
                self.writeLatency = self.readLatency
            else:
                # NAND configuration
                resPullDown = (CalculateOnResistance(self.widthNandN, NMOS,
                                                    g.inputParameter.temperature, g.tech) *
                              self.numNandInput)
                capLoad = self.capNandOutput + self.outputDriver.capInput[0]
                tr = resPullDown * capLoad
                gm = CalculateTransconductance(self.widthNandN, NMOS, g.tech)
                beta = 1 / (resPullDown * gm)
                self.readLatency, rampInputForDriver = horowitz(tr, beta, self.rampInput)

                self.outputDriver.CalculateLatency(rampInputForDriver)
                self.readLatency += self.outputDriver.readLatency
                self.writeLatency = self.readLatency

            self.rampOutput = self.outputDriver.rampOutput

    def CalculatePower(self):
        """Calculate the power consumption of the BasicDecoder"""
        if not self.initialized:
            print("[Basic Decoder] Error: Require initialization first!")
        else:
            self.outputDriver.CalculatePower()
            if self.numNandInput == 0:
                # Inverter configuration
                self.leakage = 2 * self.outputDriver.leakage
                capLoad = self.outputDriver.capInput[0] + self.outputDriver.capOutput[0]
                self.readDynamicEnergy = capLoad * g.tech.vdd * g.tech.vdd
                self.readDynamicEnergy += self.outputDriver.readDynamicEnergy
                self.readDynamicEnergy *= 1  # only one row is activated each time
                self.writeDynamicEnergy = self.readDynamicEnergy
            else:
                # NAND configuration
                # Leakage power
                self.leakage = (CalculateGateLeakage(NAND, self.numNandInput, self.widthNandN,
                                                    self.widthNandP, g.inputParameter.temperature, g.tech) *
                               g.tech.vdd)
                self.leakage += self.outputDriver.leakage
                self.leakage *= self.numNandGate
                # Dynamic energy
                capLoad = self.capNandOutput + self.outputDriver.capInput[0]
                self.readDynamicEnergy = capLoad * g.tech.vdd * g.tech.vdd
                self.readDynamicEnergy += self.outputDriver.readDynamicEnergy
                self.readDynamicEnergy *= 1  # only one row is activated each time
                self.writeDynamicEnergy = self.readDynamicEnergy

    def PrintProperty(self):
        """Print the properties of the BasicDecoder"""
        print(f"{self.numNandInput} to {self.numNandGate} Decoder Properties:")
        super().PrintProperty()
