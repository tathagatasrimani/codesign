#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
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
    horowitz,
    MAX,
    MIN
)
from constant import (
    MIN_NMOS_SIZE,
    NAND,
    NMOS,
    MAX_TRANSISTOR_HEIGHT
)
from typedef import BufferDesignTarget, MemCellType


class RowDecoder(FunctionUnit):
    """
    RowDecoder is a row decoder module that inherits from FunctionUnit.
    It decodes row addresses and drives the wordlines.
    """

    def __init__(self):
        """Constructor - initializes all properties"""
        super().__init__()
        self.initialized = False
        self.invalid = False
        self.outputDriver = OutputDriver()
        self.numRow = 0
        self.multipleRowPerSet = False
        self.numNandInput = 0
        self.capLoad = 0.0
        self.resLoad = 0.0
        self.areaOptimizationLevel = BufferDesignTarget.latency_first
        self.minDriverCurrent = 0.0
        self.widthNandN = 0.0
        self.widthNandP = 0.0
        self.capNandInput = 0.0
        self.capNandOutput = 0.0
        self.rampInput = 0.0
        self.rampOutput = 0.0

    def Initialize(self, _numRow, _capLoad, _resLoad, _multipleRowPerSet,
                   _areaOptimizationLevel, _minDriverCurrent):
        """
        Initialize the RowDecoder with the given parameters.

        Args:
            _numRow: Number of rows
            _capLoad: Load capacitance (wordline capacitance) in F
            _resLoad: Load resistance in ohm
            _multipleRowPerSet: For cache design, whether a set is partitioned into multiple wordlines
            _areaOptimizationLevel: Buffer design target (0 for latency, 2 for area)
            _minDriverCurrent: Minimum driving current should be provided
        """
        if self.initialized:
            print("[Row Decoder] Warning: Already initialized!")

        self.numRow = _numRow
        self.capLoad = _capLoad
        self.resLoad = _resLoad
        self.multipleRowPerSet = _multipleRowPerSet
        self.areaOptimizationLevel = _areaOptimizationLevel
        self.minDriverCurrent = _minDriverCurrent

        if self.numRow <= 8:  # The predecoder output is used directly
            if self.multipleRowPerSet:
                self.numNandInput = 2  # NAND way-select with predecoder output
            else:
                self.numNandInput = 0  # no circuit needed
        else:
            if self.multipleRowPerSet:
                self.numNandInput = 3  # NAND way-select with two predecoder outputs
            else:
                self.numNandInput = 2  # just NAND two predecoder outputs

        if self.numNandInput > 0:
            if self.numNandInput == 2:  # NAND2
                self.widthNandN = 2 * MIN_NMOS_SIZE * g.tech.featureSize
                logicEffortNand = (2 + g.tech.pnSizeRatio) / (1 + g.tech.pnSizeRatio)
            else:  # NAND3
                self.widthNandN = 3 * MIN_NMOS_SIZE * g.tech.featureSize
                logicEffortNand = (3 + g.tech.pnSizeRatio) / (1 + g.tech.pnSizeRatio)

            self.widthNandP = g.tech.pnSizeRatio * MIN_NMOS_SIZE * g.tech.featureSize
            capNand = CalculateGateCap(self.widthNandN, g.tech) + CalculateGateCap(self.widthNandP, g.tech)
            self.outputDriver.Initialize(logicEffortNand, capNand, self.capLoad, self.resLoad,
                                        True, self.areaOptimizationLevel, self.minDriverCurrent)
        else:
            # we only need an 1-level output buffer to driver the wordline
            self.widthNandN = MIN_NMOS_SIZE * g.tech.featureSize
            self.widthNandP = g.tech.pnSizeRatio * MIN_NMOS_SIZE * g.tech.featureSize
            capInv = CalculateGateCap(self.widthNandN, g.tech) + CalculateGateCap(self.widthNandP, g.tech)
            self.outputDriver.Initialize(1, capInv, self.capLoad, self.resLoad,
                                        True, self.areaOptimizationLevel, self.minDriverCurrent)

        if self.outputDriver.invalid:
            self.invalid = True
            return

        self.initialized = True

    def CalculateArea(self):
        """Calculate the area of the RowDecoder"""
        if not self.initialized:
            print("[Row Decoder Area] Error: Require initialization first!")
        else:
            self.outputDriver.CalculateArea()
            if self.numNandInput == 0:  # no circuit needed, use predecoder outputs directly
                self.height = self.outputDriver.height
                self.width = self.outputDriver.width
            else:
                area, hNand, wNand = CalculateGateArea(NAND, self.numNandInput, self.widthNandN, self.widthNandP,
                                                        g.tech.featureSize * 40, g.tech)
                self.height = MAX(hNand, self.outputDriver.height)
                self.width = wNand + self.outputDriver.width
            self.height *= self.numRow
            self.area = self.height * self.width

    def CalculateRC(self):
        """Calculate the resistance and capacitance of the RowDecoder"""
        if not self.initialized:
            print("[Row Decoder RC] Error: Require initialization first!")
        else:
            self.outputDriver.CalculateRC()
            if self.numNandInput == 0:  # no circuit needed, use predecoder outputs directly
                self.capNandInput = 0.0
                self.capNandOutput = 0.0
            else:
                capInput, capOutput = CalculateGateCapacitance(NAND, self.numNandInput, self.widthNandN, self.widthNandP,
                                                                g.tech.featureSize * MAX_TRANSISTOR_HEIGHT, g.tech)
                self.capNandInput = capInput
                self.capNandOutput = capOutput

    def CalculateLatency(self, _rampInput):
        """
        Calculate the latency of the RowDecoder.

        Args:
            _rampInput: Input ramp time
        """
        if not self.initialized:
            print("[Row Decoder Latency] Error: Require initialization first!")
        else:
            if self.numNandInput == 0:  # no circuit needed, use predecoder outputs directly
                self.outputDriver.CalculateLatency(_rampInput)
                self.readLatency = self.outputDriver.readLatency
                self.writeLatency = self.readLatency
                self.rampOutput = self.outputDriver.rampOutput
            else:
                self.rampInput = _rampInput

                resPullDown = CalculateOnResistance(self.widthNandN, NMOS,
                                                   g.inputParameter.temperature, g.tech) * self.numNandInput
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
        """Calculate the power consumption of the RowDecoder"""
        if not self.initialized:
            print("[Row Decoder Power] Error: Require initialization first!")
        else:
            self.outputDriver.CalculatePower()
            self.leakage = self.outputDriver.leakage
            if self.numNandInput == 0:  # no circuit needed, use predecoder outputs directly
                self.readDynamicEnergy = self.outputDriver.readDynamicEnergy
                self.writeDynamicEnergy = self.readDynamicEnergy
            else:
                # Leakage power
                self.leakage += (CalculateGateLeakage(NAND, self.numNandInput, self.widthNandN,
                                                     self.widthNandP, g.inputParameter.temperature, g.tech) *
                                g.tech.vdd)
                # Dynamic energy
                capLoad = self.capNandOutput + self.outputDriver.capInput[0]
                # For DRAM types account for overdriven wordline.
                if g.cell.memCellType == MemCellType.DRAM or g.cell.memCellType == MemCellType.eDRAM:
                    self.readDynamicEnergy = capLoad * g.tech.vpp * g.tech.vpp
                else:
                    self.readDynamicEnergy = capLoad * g.tech.vdd * g.tech.vdd
                self.readDynamicEnergy += self.outputDriver.readDynamicEnergy
                self.readDynamicEnergy *= 1  # only one row is activated each time
                self.writeDynamicEnergy = self.readDynamicEnergy
            self.leakage *= self.numRow

    def PrintProperty(self):
        """Print the properties of the RowDecoder"""
        print("Row Decoder Properties:")
        super().PrintProperty()

    def assign(self, rhs):
        """
        Assignment method to copy all properties from another RowDecoder instance.
        This is the Python equivalent of the C++ assignment operator.

        Args:
            rhs: The RowDecoder instance to copy from

        Returns:
            self: For method chaining
        """
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

        # Copy RowDecoder specific properties
        self.initialized = rhs.initialized
        self.invalid = rhs.invalid
        self.outputDriver = rhs.outputDriver
        self.numRow = rhs.numRow
        self.multipleRowPerSet = rhs.multipleRowPerSet
        self.numNandInput = rhs.numNandInput
        self.capLoad = rhs.capLoad
        self.resLoad = rhs.resLoad
        self.areaOptimizationLevel = rhs.areaOptimizationLevel
        self.minDriverCurrent = rhs.minDriverCurrent
        self.widthNandN = rhs.widthNandN
        self.widthNandP = rhs.widthNandP
        self.capNandInput = rhs.capNandInput
        self.capNandOutput = rhs.capNandOutput
        self.rampInput = rhs.rampInput
        self.rampOutput = rhs.rampOutput

        return self
