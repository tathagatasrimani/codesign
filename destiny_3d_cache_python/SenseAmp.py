#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

from FunctionUnit import FunctionUnit
from formula import (calculate_gate_area, calculate_gate_cap, calculate_drain_cap,
                     calculate_transconductance, calculate_gate_leakage)
from constant import (W_SENSE_P, W_SENSE_N, W_SENSE_ISO, W_SENSE_EN, W_SENSE_MUX,
                      IV_CONVERTER_AREA, MIN_GAP_BET_P_AND_N_DIFFS,
                      MIN_GAP_BET_SAME_TYPE_DIFFS, INV, NMOS, PMOS)
import globals as g
import math


class SenseAmp(FunctionUnit):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.invalid = False
        self.numColumn = 0
        self.currentSense = False
        self.senseVoltage = 0.0
        self.capLoad = 0.0
        self.pitchSenseAmp = 0.0

    def Initialize(self, _numColumn, _currentSense, _senseVoltage, _pitchSenseAmp):
        """Initialize sense amplifier parameters

        Args:
            _numColumn: Number of columns
            _currentSense: Whether the sensing scheme is current-based
            _senseVoltage: Minimum sensible voltage (Unit: V)
            _pitchSenseAmp: The maximum width allowed for one sense amplifier layout
        """
        if self.initialized:
            print("[Sense Amp] Warning: Already initialized!")

        self.numColumn = _numColumn
        self.currentSense = _currentSense
        self.senseVoltage = _senseVoltage
        self.pitchSenseAmp = _pitchSenseAmp

        if self.pitchSenseAmp <= g.tech.featureSize * 3:
            # too small, cannot do the layout
            self.invalid = True

        self.initialized = True

    def CalculateArea(self):
        """Calculate sense amplifier area"""
        if not self.initialized:
            print("[Sense Amp] Error: Require initialization first!")
        elif self.invalid:
            self.height = self.width = self.area = g.invalid_value
        else:
            self.height = self.width = self.area = 0
            if self.currentSense:  # current-sensing needs IV converter
                self.area += IV_CONVERTER_AREA * g.tech.featureSize * g.tech.featureSize

            # the following codes are transformed from CACTI 6.5
            tempHeight = 0
            tempWidth = 0

            # Calculate area for PMOS sense transistors
            _, temp_h, temp_w = calculate_gate_area(INV, 1, 0, W_SENSE_P * g.tech.featureSize,
                                                     self.pitchSenseAmp, g.tech)
            # exchange width and height for senseamp layout
            tempWidth = temp_w
            tempHeight = temp_h
            self.width = max(self.width, tempWidth)
            self.height += 2 * tempHeight

            # Calculate area for ISO transistor
            _, temp_h, temp_w = calculate_gate_area(INV, 1, 0, W_SENSE_ISO * g.tech.featureSize,
                                                     self.pitchSenseAmp, g.tech)
            tempWidth = temp_w
            tempHeight = temp_h
            self.width = max(self.width, tempWidth)
            self.height += tempHeight

            self.height += 2 * MIN_GAP_BET_SAME_TYPE_DIFFS * g.tech.featureSize

            # Calculate area for NMOS sense transistors
            _, temp_h, temp_w = calculate_gate_area(INV, 1, W_SENSE_N * g.tech.featureSize, 0,
                                                     self.pitchSenseAmp, g.tech)
            tempWidth = temp_w
            tempHeight = temp_h
            self.width = max(self.width, tempWidth)
            self.height += 2 * tempHeight

            # Calculate area for enable transistor
            _, temp_h, temp_w = calculate_gate_area(INV, 1, W_SENSE_EN * g.tech.featureSize, 0,
                                                     self.pitchSenseAmp, g.tech)
            tempWidth = temp_w
            tempHeight = temp_h
            self.width = max(self.width, tempWidth)
            self.height += tempHeight

            self.height += 2 * MIN_GAP_BET_SAME_TYPE_DIFFS * g.tech.featureSize
            self.height += MIN_GAP_BET_P_AND_N_DIFFS * g.tech.featureSize

            # transformation so that width meets the pitch
            self.height = self.height * self.width / self.pitchSenseAmp
            self.width = self.pitchSenseAmp

            # Add additional area if IV converter exists
            self.height += self.area / self.width
            self.width *= self.numColumn

            self.area = self.height * self.width

    def CalculateRC(self):
        """Calculate RC parameters"""
        if not self.initialized:
            print("[Sense Amp] Error: Require initialization first!")
        elif self.invalid:
            self.readLatency = self.writeLatency = g.invalid_value
        else:
            self.capLoad = (calculate_gate_cap((W_SENSE_P + W_SENSE_N) * g.tech.featureSize, g.tech) +
                           calculate_drain_cap(W_SENSE_N * g.tech.featureSize, NMOS, self.pitchSenseAmp, g.tech) +
                           calculate_drain_cap(W_SENSE_P * g.tech.featureSize, PMOS, self.pitchSenseAmp, g.tech) +
                           calculate_drain_cap(W_SENSE_ISO * g.tech.featureSize, PMOS, self.pitchSenseAmp, g.tech) +
                           calculate_drain_cap(W_SENSE_MUX * g.tech.featureSize, NMOS, self.pitchSenseAmp, g.tech))

    def CalculateLatency(self, _rampInput):
        """Calculate latency

        Args:
            _rampInput: Input ramp (actually no use in SenseAmp)
        """
        if not self.initialized:
            print("[Sense Amp] Error: Require initialization first!")
        else:
            self.readLatency = self.writeLatency = 0
            if self.currentSense:  # current-sensing needs IV converter
                # all the following values achieved from HSPICE
                if g.tech.featureSize >= 179e-9:
                    self.readLatency += 0.46e-9  # 180nm
                elif g.tech.featureSize >= 119e-9:
                    self.readLatency += 0.49e-9  # 120nm
                elif g.tech.featureSize >= 89e-9:
                    self.readLatency += 0.53e-9  # 90nm
                elif g.tech.featureSize >= 64e-9:
                    self.readLatency += 0.62e-9  # 65nm
                elif g.tech.featureSize >= 44e-9:
                    self.readLatency += 0.80e-9  # 45nm
                elif g.tech.featureSize >= 31e-9:
                    self.readLatency += 1.07e-9  # 32nm
                else:
                    self.readLatency += 1.45e-9  # below 22nm

            # Voltage sense amplifier
            gm = (calculate_transconductance(W_SENSE_N * g.tech.featureSize, NMOS, g.tech) +
                  calculate_transconductance(W_SENSE_P * g.tech.featureSize, PMOS, g.tech))
            tau = self.capLoad / gm
            self.readLatency += tau * math.log(g.tech.vdd / self.senseVoltage)
            self.refreshLatency = self.readLatency

    def CalculatePower(self):
        """Calculate power consumption"""
        if not self.initialized:
            print("[Sense Amp] Error: Require initialization first!")
        elif self.invalid:
            self.readDynamicEnergy = self.writeDynamicEnergy = self.leakage = g.invalid_value
        else:
            self.readDynamicEnergy = self.writeDynamicEnergy = 0
            self.leakage = 0
            if self.currentSense:  # current-sensing needs IV converter
                # all the following values achieved from HSPICE
                if g.tech.featureSize >= 119e-9:  # 120nm
                    self.readDynamicEnergy += 8.52e-14  # Unit: J
                    self.leakage += 1.40e-8  # Unit: W
                elif g.tech.featureSize >= 89e-9:  # 90nm
                    self.readDynamicEnergy += 8.72e-14
                    self.leakage += 1.87e-8
                elif g.tech.featureSize >= 64e-9:  # 65nm
                    self.readDynamicEnergy += 9.00e-14
                    self.leakage += 2.57e-8
                elif g.tech.featureSize >= 44e-9:  # 45nm
                    self.readDynamicEnergy += 10.26e-14
                    self.leakage += 4.41e-9
                elif g.tech.featureSize >= 31e-9:  # 32nm
                    self.readDynamicEnergy += 12.56e-14
                    self.leakage += 12.54e-8
                else:  # TO-DO, need calibration below 22nm
                    self.readDynamicEnergy += 15e-14
                    self.leakage += 15e-8

            # Voltage sense amplifier
            self.readDynamicEnergy += self.capLoad * g.tech.vdd * g.tech.vdd
            idleCurrent = (calculate_gate_leakage(INV, 1, W_SENSE_EN * g.tech.featureSize, 0,
                                                 g.inputParameter.temperature, g.tech) * g.tech.vdd)
            self.leakage += idleCurrent * g.tech.vdd

            self.readDynamicEnergy *= self.numColumn
            self.leakage *= self.numColumn

            self.refreshDynamicEnergy = self.readDynamicEnergy

    def PrintProperty(self):
        """Print sense amplifier properties"""
        print("Sense Amplifier Properties:")
        super().PrintProperty()

    def assign(self, rhs):
        """Assignment method to copy all properties from another SenseAmp instance

        Args:
            rhs: Another SenseAmp instance to copy from

        Returns:
            self: Returns self to allow chaining
        """
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
        self.invalid = rhs.invalid
        self.numColumn = rhs.numColumn
        self.currentSense = rhs.currentSense
        self.senseVoltage = rhs.senseVoltage
        self.capLoad = rhs.capLoad
        self.pitchSenseAmp = rhs.pitchSenseAmp

        return self
