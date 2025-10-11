# Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
# and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
# No part of DESTINY Project, including this file, may be copied,
# modified, propagated, or distributed except according to the terms
# contained in the LICENSE file.

from FunctionUnit import FunctionUnit
from RowDecoder import RowDecoder
from BasicDecoder import BasicDecoder
from typedef import BufferDesignTarget


class PredecodeBlock(FunctionUnit):
    """
    PredecodeBlock is a predecoder module that inherits from FunctionUnit.
    It manages multiple stages of row decoders and basic decoders to implement
    hierarchical decoding for address bits.
    """

    def __init__(self):
        """Constructor - initializes all properties"""
        super().__init__()

        # Initialization flag
        self.initialized = False

        # Row decoder stages
        self.rowDecoderStage1A = None
        self.rowDecoderStage1B = None
        self.rowDecoderStage1C = None
        self.rowDecoderStage2 = None

        # NAND input counts for each stage
        self.numNandInputStage1A = 0
        self.numNandInputStage1B = 0
        self.numNandInputStage1C = 0

        # Address bit counts for each stage
        self.numAddressBitStage1A = 0
        self.numAddressBitStage1B = 0
        self.numAddressBitStage1C = 0

        # Basic decoders
        self.basicDecoderA1 = None
        self.basicDecoderA2 = None
        self.basicDecoderB = None
        self.basicDecoderC = None

        # Load parameters
        self.capLoad = 0.0  # Load capacitance Unit: F
        self.resLoad = 0.0  # Load resistance Unit: ohm

        # Address bit parameters
        self.numAddressBit = 0  # Number of Address Bits assigned to the block
        self.numOutputAddressBit = 0

        # Decoder counts
        self.numDecoder12 = 0  # Number of 1 to 2 Decoders
        self.numDecoder24 = 0  # Number of 2 to 4 Decoders
        self.numDecoder38 = 0  # Number of 3 to 8 Decoders

        # Basic decoder counts
        self.numBasicDecoderA1 = 0
        self.numBasicDecoderA2 = 0

        # Capacitance loads for basic decoders
        self.capLoadBasicDecoderA1 = 0.0
        self.capLoadBasicDecoderA2 = 0.0
        self.capLoadBasicDecoderB = 0.0
        self.capLoadBasicDecoderC = 0.0

        # Ramp parameters
        self.rampInput = 0.0
        self.rampOutput = 0.0

    def Initialize(self, _numAddressBit, _capLoad, _resLoad):
        """
        Initialize the PredecodeBlock with the given parameters.

        Args:
            _numAddressBit: Number of address bits
            _capLoad: Load capacitance in F
            _resLoad: Load resistance in ohm
        """
        if self.initialized:
            print("[Predecoder Block] Warning: Already initialized!")

        self.numAddressBit = _numAddressBit

        if self.numAddressBit > 27:
            print("[Predecoder Block] Error: Invalid number of address bits")
            exit(-1)
        elif self.numAddressBit == 0:
            self.height = self.width = self.area = 0
            self.readLatency = self.writeLatency = 0
            self.readDynamicEnergy = self.writeDynamicEnergy = 0
            self.leakage = 0
            self.initialized = True
        else:
            self.capLoad = _capLoad
            self.resLoad = _resLoad

            self.numDecoder12 = self.numDecoder24 = self.numDecoder38 = 0
            self.numOutputAddressBit = 1 << self.numAddressBit

            if self.numAddressBit == 1:
                self.numDecoder12 = 1
            else:
                numAddressMod3 = self.numAddressBit % 3
                if numAddressMod3 == 2:
                    self.numDecoder24 = 1
                elif numAddressMod3 == 1:
                    self.numDecoder24 = 2
                self.numDecoder38 = (self.numAddressBit - 2 * self.numDecoder24) // 3

            numBasicDecoder = self.numDecoder12 + self.numDecoder24 + self.numDecoder38

            if numBasicDecoder <= 1:
                self.rowDecoderStage1A = None
                self.rowDecoderStage1B = None
                self.rowDecoderStage1C = None
                self.rowDecoderStage2 = None
            elif numBasicDecoder <= 3:
                self.numNandInputStage1A = numBasicDecoder
                self.rowDecoderStage2 = None
                self.rowDecoderStage1B = None
                self.rowDecoderStage1C = None
                self.rowDecoderStage1A = RowDecoder()
                # Note: RowDecoder Initialize signature may need updating to match C++ version
                # For now using simplified version: Initialize(_numAddressBit, _numRow)
                # C++ signature: Initialize(_numRow, _capLoad, _resLoad, _multipleRowPerSet, _areaOptimizationLevel, _minDriverCurrent)
                # TODO: Update RowDecoder.Initialize to match C++ signature
                self.rowDecoderStage1A.Initialize(self.numOutputAddressBit, self.capLoad)
                self.rowDecoderStage1A.CalculateRC()
            else:
                self.rowDecoderStage2 = RowDecoder()

                if numBasicDecoder <= 6:
                    # Initialize rowDecoderStage2
                    self.rowDecoderStage2.Initialize(self.numOutputAddressBit, self.capLoad)
                    self.rowDecoderStage2.CalculateRC()

                    self.numNandInputStage1B = numBasicDecoder // 2
                    self.numNandInputStage1A = numBasicDecoder - self.numNandInputStage1B
                    self.numAddressBitStage1A = self.numAddressBitStage1B = 1
                    i = 3 * self.numNandInputStage1A - self.numDecoder24
                    self.numAddressBitStage1A <<= i
                    self.numAddressBitStage1B <<= 3 * self.numNandInputStage1B

                    # Calculate capacitance loads
                    # Note: Assuming rowDecoderStage2 has capNandInput attribute
                    capLoadStage1A = self.numAddressBitStage1B * getattr(self.rowDecoderStage2, 'capNandInput', 0)
                    capLoadStage1B = self.numAddressBitStage1A * getattr(self.rowDecoderStage2, 'capNandInput', 0)

                    self.rowDecoderStage1C = None
                    self.rowDecoderStage1A = RowDecoder()
                    self.rowDecoderStage1A.Initialize(self.numAddressBitStage1A, capLoadStage1A)
                    self.rowDecoderStage1A.CalculateRC()

                    self.rowDecoderStage1B = RowDecoder()
                    self.rowDecoderStage1B.Initialize(self.numAddressBitStage1B, capLoadStage1B)
                    self.rowDecoderStage1B.CalculateRC()

                elif numBasicDecoder <= 9:
                    # Initialize rowDecoderStage2
                    self.rowDecoderStage2.Initialize(self.numOutputAddressBit, self.capLoad)
                    self.rowDecoderStage2.CalculateRC()

                    if numBasicDecoder == 7:
                        self.numNandInputStage1A = 3
                        self.numNandInputStage1B = 2
                        self.numNandInputStage1C = 2
                        self.numAddressBitStage1B = 64
                        self.numAddressBitStage1C = 64
                    elif numBasicDecoder == 8:
                        self.numNandInputStage1A = 3
                        self.numNandInputStage1B = 3
                        self.numNandInputStage1C = 2
                        self.numAddressBitStage1B = 512
                        self.numAddressBitStage1C = 64
                    else:  # numBasicDecoder == 9
                        self.numNandInputStage1A = 3
                        self.numNandInputStage1B = 3
                        self.numNandInputStage1C = 3
                        self.numAddressBitStage1B = 512
                        self.numAddressBitStage1C = 512

                    i = 3 * self.numNandInputStage1A - self.numDecoder24
                    self.numAddressBitStage1A = 1
                    self.numAddressBitStage1A <<= i

                    # Calculate capacitance loads
                    capNandInput = getattr(self.rowDecoderStage2, 'capNandInput', 0)
                    capLoadStage1A = self.numAddressBitStage1B * self.numAddressBitStage1C * capNandInput
                    capLoadStage1B = self.numAddressBitStage1A * self.numAddressBitStage1C * capNandInput
                    capLoadStage1C = self.numAddressBitStage1A * self.numAddressBitStage1B * capNandInput

                    self.rowDecoderStage1A = RowDecoder()
                    self.rowDecoderStage1A.Initialize(self.numAddressBitStage1A, capLoadStage1A)
                    self.rowDecoderStage1A.CalculateRC()

                    self.rowDecoderStage1B = RowDecoder()
                    self.rowDecoderStage1B.Initialize(self.numAddressBitStage1B, capLoadStage1B)
                    self.rowDecoderStage1B.CalculateRC()

                    self.rowDecoderStage1C = RowDecoder()
                    self.rowDecoderStage1C.Initialize(self.numAddressBitStage1C, capLoadStage1C)
                    self.rowDecoderStage1C.CalculateRC()

            # Initialize basic decoders
            if self.rowDecoderStage1C is not None:
                if self.numNandInputStage1C == 2:
                    self.capLoadBasicDecoderC = 8 * getattr(self.rowDecoderStage1C, 'capNandInput', 0)
                else:
                    self.capLoadBasicDecoderC = 64 * getattr(self.rowDecoderStage1C, 'capNandInput', 0)
                self.basicDecoderC = BasicDecoder()
                self.basicDecoderC.Initialize(3, self.capLoadBasicDecoderC, 0)
            else:
                self.basicDecoderC = None

            if self.rowDecoderStage1B is not None:
                if self.numNandInputStage1B == 2:
                    self.capLoadBasicDecoderB = 8 * getattr(self.rowDecoderStage1B, 'capNandInput', 0)
                else:
                    self.capLoadBasicDecoderB = 64 * getattr(self.rowDecoderStage1B, 'capNandInput', 0)
                self.basicDecoderB = BasicDecoder()
                self.basicDecoderB.Initialize(3, self.capLoadBasicDecoderB, 0)
            else:
                self.basicDecoderB = None

            if self.rowDecoderStage1A is not None:
                if self.numDecoder24 == 0:
                    self.numBasicDecoderA1 = self.numNandInputStage1A
                    self.numBasicDecoderA2 = 0
                    numCapNandA1 = 1 << (3 * (self.numNandInputStage1A - 1))
                    self.capLoadBasicDecoderA1 = numCapNandA1 * getattr(self.rowDecoderStage1A, 'capNandInput', 0)
                    self.basicDecoderA1 = BasicDecoder()
                    self.basicDecoderA1.Initialize(3, self.capLoadBasicDecoderA1, 0)
                    self.basicDecoderA2 = None
                elif self.numDecoder24 == 1:
                    self.numBasicDecoderA1 = 1
                    self.numBasicDecoderA2 = self.numNandInputStage1A - self.numBasicDecoderA1
                    numCapNandA1 = 1 << (3 * self.numBasicDecoderA2)
                    numCapNandA2 = 1 << (2 + 3 * (self.numBasicDecoderA2 - 1))
                    capNandInput = getattr(self.rowDecoderStage1A, 'capNandInput', 0)
                    self.capLoadBasicDecoderA1 = numCapNandA1 * capNandInput
                    self.capLoadBasicDecoderA2 = numCapNandA2 * capNandInput
                    self.basicDecoderA1 = BasicDecoder()
                    self.basicDecoderA1.Initialize(2, self.capLoadBasicDecoderA1, 0)
                    self.basicDecoderA2 = BasicDecoder()
                    self.basicDecoderA2.Initialize(3, self.capLoadBasicDecoderA2, 0)
                elif self.numDecoder24 == 2:
                    if self.numNandInputStage1A == 2:
                        self.numBasicDecoderA1 = 2
                        self.numBasicDecoderA2 = 0
                        self.basicDecoderA1 = BasicDecoder()
                        self.basicDecoderA1.Initialize(2, 4 * getattr(self.rowDecoderStage1A, 'capNandInput', 0), 0)
                        self.basicDecoderA2 = None
                    else:
                        self.numBasicDecoderA1 = 2
                        self.numBasicDecoderA2 = 1
                        capNandInput = getattr(self.rowDecoderStage1A, 'capNandInput', 0)
                        self.basicDecoderA1 = BasicDecoder()
                        self.basicDecoderA1.Initialize(2, 32 * capNandInput, 0)
                        self.basicDecoderA2 = BasicDecoder()
                        self.basicDecoderA2.Initialize(3, 16 * capNandInput, 0)
            else:
                self.numBasicDecoderA1 = 1
                self.numBasicDecoderA2 = 0
                self.basicDecoderA1 = BasicDecoder()
                self.basicDecoderA2 = None
                if self.numDecoder12 == 1:
                    self.basicDecoderA1.Initialize(1, self.capLoad, self.resLoad)
                elif self.numDecoder24 == 1:
                    self.basicDecoderA1.Initialize(2, self.capLoad, self.resLoad)
                elif self.numDecoder38 == 1:
                    self.basicDecoderA1.Initialize(3, self.capLoad, self.resLoad)

        self.initialized = True

    def CalculateArea(self):
        """Calculate the area of the PredecodeBlock"""
        if not self.initialized:
            print("[Predecoder Block] Error: Require initialization first!")
        elif self.numAddressBit == 0:
            self.height = self.width = self.area = 0
        else:
            hTemp = wTemp = 0

            # Calculate area for basic decoders
            if self.basicDecoderA1 is not None:
                self.basicDecoderA1.CalculateArea()
                wTemp = max(wTemp, self.basicDecoderA1.width)
                hTemp += self.numBasicDecoderA1 * self.basicDecoderA1.height

                if self.basicDecoderA2 is not None:
                    self.basicDecoderA2.CalculateArea()
                    wTemp = max(wTemp, self.basicDecoderA2.width)
                    hTemp += self.numBasicDecoderA2 * self.basicDecoderA2.height

                if self.basicDecoderB is not None:
                    self.basicDecoderB.CalculateArea()
                    wTemp = max(wTemp, self.basicDecoderB.width)
                    hTemp += self.numNandInputStage1B * self.basicDecoderB.height

                    if self.basicDecoderC is not None:
                        self.basicDecoderC.CalculateArea()
                        wTemp = max(wTemp, self.basicDecoderC.width)
                        hTemp += self.numNandInputStage1C * self.basicDecoderC.height

            self.width = wTemp
            self.height = hTemp
            hTemp = wTemp = 0

            # Calculate area for row decoders
            if self.rowDecoderStage1A is not None:
                self.rowDecoderStage1A.CalculateArea()
                wTemp = max(wTemp, self.rowDecoderStage1A.width)
                hTemp += self.rowDecoderStage1A.height

                if self.rowDecoderStage1B is not None:
                    self.rowDecoderStage1B.CalculateArea()
                    wTemp = max(wTemp, self.rowDecoderStage1B.width)
                    hTemp += self.rowDecoderStage1B.height

                    if self.rowDecoderStage1C is not None:
                        self.rowDecoderStage1C.CalculateArea()
                        wTemp = max(wTemp, self.rowDecoderStage1C.width)
                        hTemp += self.rowDecoderStage1C.height

                if self.rowDecoderStage2 is not None:
                    self.rowDecoderStage2.CalculateArea()
                    wTemp += self.rowDecoderStage2.width
                    hTemp = max(hTemp, self.rowDecoderStage2.height)

            self.width += wTemp
            self.height = max(self.height, hTemp)
            self.area = self.width * self.height

    def CalculateRC(self):
        """Calculate the resistance and capacitance of the PredecodeBlock"""
        if not self.initialized:
            print("[Predecoder Block] Error: Require initialization first!")
        elif self.numAddressBit > 0:
            if self.basicDecoderA1 is not None:
                self.basicDecoderA1.CalculateRC()

                if self.basicDecoderA2 is not None:
                    self.basicDecoderA2.CalculateRC()

                if self.basicDecoderB is not None:
                    self.basicDecoderB.CalculateRC()

                    if self.basicDecoderC is not None:
                        self.basicDecoderC.CalculateRC()

    def CalculateLatency(self, _rampInput):
        """
        Calculate the latency of the PredecodeBlock.

        Args:
            _rampInput: Input ramp time
        """
        if not self.initialized:
            print("[Predecoder Block] Error: Require initialization first!")
        elif self.numAddressBit == 0:
            self.readLatency = self.writeLatency = 0
            self.rampOutput = _rampInput
        else:
            self.rampInput = _rampInput
            delayA1 = delayA2 = delayB = delayC = 0
            maxRampOutput = 0
            self.rampOutput = 0
            self.readLatency = self.writeLatency = 0

            # Calculate delay for path A1
            if self.basicDecoderA1 is not None:
                self.basicDecoderA1.CalculateLatency(self.rampInput)
                delayA1 += self.basicDecoderA1.readLatency
                maxRampOutput = self.basicDecoderA1.rampOutput

                if self.rowDecoderStage1A is not None:
                    self.rowDecoderStage1A.CalculateLatency(self.basicDecoderA1.rampOutput)
                    delayA1 += self.rowDecoderStage1A.readLatency
                    maxRampOutput = getattr(self.rowDecoderStage1A, 'rampOutput', maxRampOutput)

                    if self.rowDecoderStage2 is not None:
                        self.rowDecoderStage2.CalculateLatency(getattr(self.rowDecoderStage1A, 'rampOutput', 0))
                        delayA1 += self.rowDecoderStage2.readLatency
                        maxRampOutput = getattr(self.rowDecoderStage2, 'rampOutput', maxRampOutput)

            self.rampOutput = max(self.rampOutput, maxRampOutput)
            self.readLatency = max(self.readLatency, delayA1)
            maxRampOutput = 0

            # Calculate delay for path A2
            if self.basicDecoderA2 is not None:
                self.basicDecoderA2.CalculateLatency(self.rampInput)
                delayA2 += self.basicDecoderA1.readLatency
                self.rowDecoderStage1A.CalculateLatency(self.basicDecoderA2.rampOutput)
                delayA2 += self.rowDecoderStage1A.readLatency
                maxRampOutput = getattr(self.rowDecoderStage1A, 'rampOutput', maxRampOutput)

                if self.rowDecoderStage2 is not None:
                    self.rowDecoderStage2.CalculateLatency(getattr(self.rowDecoderStage1A, 'rampOutput', 0))
                    delayA2 += self.rowDecoderStage2.readLatency
                    maxRampOutput = getattr(self.rowDecoderStage2, 'rampOutput', maxRampOutput)

            self.rampOutput = max(self.rampOutput, maxRampOutput)
            self.readLatency = max(self.readLatency, delayA2)
            maxRampOutput = 0

            # Calculate delay for path B
            if self.basicDecoderB is not None:
                self.basicDecoderB.CalculateLatency(self.rampInput)
                delayB += self.basicDecoderB.readLatency
                self.rowDecoderStage1B.CalculateLatency(self.basicDecoderB.rampOutput)
                delayB += self.rowDecoderStage1B.readLatency
                self.rowDecoderStage2.CalculateLatency(getattr(self.rowDecoderStage1B, 'rampOutput', 0))
                delayB += self.rowDecoderStage2.readLatency
                maxRampOutput = getattr(self.rowDecoderStage2, 'rampOutput', maxRampOutput)

            self.rampOutput = max(self.rampOutput, maxRampOutput)
            self.readLatency = max(self.readLatency, delayB)
            maxRampOutput = 0

            # Calculate delay for path C
            if self.basicDecoderC is not None:
                self.basicDecoderC.CalculateLatency(self.rampInput)
                delayC += self.basicDecoderC.readLatency
                self.rowDecoderStage1C.CalculateLatency(self.basicDecoderC.rampOutput)
                delayC += self.rowDecoderStage1C.readLatency
                self.rowDecoderStage2.CalculateLatency(getattr(self.rowDecoderStage1C, 'rampOutput', 0))
                delayC += self.rowDecoderStage2.readLatency
                maxRampOutput = getattr(self.rowDecoderStage2, 'rampOutput', maxRampOutput)

            self.rampOutput = max(self.rampOutput, maxRampOutput)
            self.readLatency = max(self.readLatency, delayC)
            self.writeLatency = self.readLatency

    def CalculatePower(self):
        """Calculate the power consumption of the PredecodeBlock"""
        if not self.initialized:
            print("[Predecoder Block] Error: Require initialization first!")
        elif self.numAddressBit == 0:
            self.leakage = self.readDynamicEnergy = self.writeDynamicEnergy = 0
        else:
            self.leakage = self.readDynamicEnergy = 0

            # Calculate power for basic decoders
            if self.basicDecoderA1 is not None:
                self.basicDecoderA1.CalculatePower()
                self.leakage += self.basicDecoderA1.leakage
                self.readDynamicEnergy += self.basicDecoderA1.readDynamicEnergy

                if self.basicDecoderA2 is not None:
                    self.basicDecoderA2.CalculatePower()
                    self.leakage += self.basicDecoderA2.leakage
                    self.readDynamicEnergy += self.basicDecoderA2.readDynamicEnergy

                if self.basicDecoderB is not None:
                    self.basicDecoderB.CalculatePower()
                    self.leakage += self.basicDecoderB.leakage
                    self.readDynamicEnergy += self.basicDecoderB.readDynamicEnergy

                    if self.basicDecoderC is not None:
                        self.basicDecoderC.CalculatePower()
                        self.leakage += self.basicDecoderC.leakage
                        self.readDynamicEnergy += self.basicDecoderC.readDynamicEnergy

            # Calculate power for row decoders
            if self.rowDecoderStage1A is not None:
                self.rowDecoderStage1A.CalculatePower()
                self.leakage += self.rowDecoderStage1A.leakage
                self.readDynamicEnergy += self.rowDecoderStage1A.readDynamicEnergy

                if self.rowDecoderStage1B is not None:
                    self.rowDecoderStage1B.CalculatePower()
                    self.leakage += self.rowDecoderStage1B.leakage
                    self.readDynamicEnergy += self.rowDecoderStage1B.readDynamicEnergy

                    if self.rowDecoderStage1C is not None:
                        self.rowDecoderStage1C.CalculatePower()
                        self.leakage += self.rowDecoderStage1C.leakage
                        self.readDynamicEnergy += self.rowDecoderStage1C.readDynamicEnergy

                if self.rowDecoderStage2 is not None:
                    self.leakage += self.rowDecoderStage2.leakage
                    self.readDynamicEnergy += self.rowDecoderStage2.readDynamicEnergy

            self.writeDynamicEnergy = self.readDynamicEnergy

    def PrintProperty(self):
        """Print the properties of the PredecodeBlock"""
        print("Predecoding Block Properties:")
        super().PrintProperty()

    def __copy__(self):
        """Copy operator equivalent"""
        new_obj = PredecodeBlock()

        # Copy FunctionUnit properties
        new_obj.height = self.height
        new_obj.width = self.width
        new_obj.area = self.area
        new_obj.readLatency = self.readLatency
        new_obj.writeLatency = self.writeLatency
        new_obj.readDynamicEnergy = self.readDynamicEnergy
        new_obj.writeDynamicEnergy = self.writeDynamicEnergy
        new_obj.resetLatency = self.resetLatency
        new_obj.setLatency = self.setLatency
        new_obj.resetDynamicEnergy = self.resetDynamicEnergy
        new_obj.setDynamicEnergy = self.setDynamicEnergy
        new_obj.cellReadEnergy = self.cellReadEnergy
        new_obj.cellSetEnergy = self.cellSetEnergy
        new_obj.cellResetEnergy = self.cellResetEnergy
        new_obj.leakage = self.leakage

        # Copy PredecodeBlock-specific properties
        new_obj.initialized = self.initialized
        new_obj.numNandInputStage1A = self.numNandInputStage1A
        new_obj.numNandInputStage1B = self.numNandInputStage1B
        new_obj.numNandInputStage1C = self.numNandInputStage1C
        new_obj.numAddressBitStage1A = self.numAddressBitStage1A
        new_obj.numAddressBitStage1B = self.numAddressBitStage1B
        new_obj.numAddressBitStage1C = self.numAddressBitStage1C
        new_obj.capLoad = self.capLoad
        new_obj.resLoad = self.resLoad
        new_obj.numAddressBit = self.numAddressBit
        new_obj.numOutputAddressBit = self.numOutputAddressBit
        new_obj.numDecoder12 = self.numDecoder12
        new_obj.numDecoder24 = self.numDecoder24
        new_obj.numDecoder38 = self.numDecoder38
        new_obj.numBasicDecoderA1 = self.numBasicDecoderA1
        new_obj.numBasicDecoderA2 = self.numBasicDecoderA2
        new_obj.capLoadBasicDecoderA1 = self.capLoadBasicDecoderA1
        new_obj.capLoadBasicDecoderA2 = self.capLoadBasicDecoderA2
        new_obj.capLoadBasicDecoderB = self.capLoadBasicDecoderB
        new_obj.capLoadBasicDecoderC = self.capLoadBasicDecoderC
        new_obj.rampInput = self.rampInput
        new_obj.rampOutput = self.rampOutput

        return new_obj
