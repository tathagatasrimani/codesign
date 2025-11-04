#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

import math
import sys
from typedef import WireType, WireRepeaterType
from constant import (COPPER_RESISTIVITY, COPPER_RESISTIVITY_TEMPERATURE_COEFFICIENT,
                      PERMITTIVITY, MIN_NMOS_SIZE, MAX_NMOS_SIZE, INV, NAND, NMOS, PMOS,
                      RES_ADJ, VOL_SWING)
from formula import (calculate_gate_area as CalculateGateArea,
                     calculate_gate_capacitance as CalculateGateCapacitance,
                     calculate_gate_cap as CalculateGateCap,
                     calculate_drain_cap as CalculateDrainCap,
                     calculate_on_resistance as CalculateOnResistance,
                     calculate_transconductance as CalculateTransconductance,
                     calculate_gate_leakage as CalculateGateLeakage,
                     calculate_wire_resistance as CalculateWireResistance,
                     calculate_wire_capacitance as CalculateWireCapacitance,
                     horowitz, MIN, MAX)
import globals as g
from SenseAmp import SenseAmp


class Wire:
    """Wire model for interconnect delay and power calculation.

    This class models wire properties including resistance, capacitance,
    and repeater insertion for different wire types and process nodes.
    """

    def __init__(self):
        """Initialize Wire with default values."""
        self.initialized = False
        self.featureSizeInNano = 0
        self.featureSize = 0.0
        self.wireType = WireType.local_aggressive
        self.wireRepeaterType = WireRepeaterType.repeated_none
        self.temperature = 300
        self.isLowSwing = False

        # Wire physical properties
        self.barrierThickness = 0.0
        self.horizontalDielectric = 0.0
        self.wirePitch = 0.0
        self.aspectRatio = 0.0
        self.ildThickness = 0.0
        self.wireWidth = 0.0
        self.wireThickness = 0.0
        self.wireSpacing = 0.0

        # Repeater properties
        self.repeaterSize = 0.0
        self.repeaterSpacing = float('inf')
        self.repeaterHeight = 0.0
        self.repeaterWidth = 0.0
        self.repeatedWirePitch = 0.0

        # Electrical properties
        self.resWirePerUnit = 0.0
        self.capWirePerUnit = 0.0

        # Sense amplifier for low-swing wires
        self.senseAmp = None

    def __del__(self):
        """Destructor to clean up sense amplifier."""
        if self.senseAmp:
            del self.senseAmp

    def Initialize(self, _featureSizeInNano, _wireType, _wireRepeaterType, _temperature, _isLowSwing):
        """Initialize wire parameters based on technology and wire type.

        Args:
            _featureSizeInNano: Process feature size in nanometers
            _wireType: WireType enum value (local/semi/global, aggressive/conservative)
            _wireRepeaterType: WireRepeaterType enum value (none/opt/penalized)
            _temperature: Temperature in Kelvin
            _isLowSwing: Boolean for low-swing signaling
        """
        if self.initialized:
            # Reload new input, clear previous settings
            self.initialized = False
            if self.senseAmp:
                del self.senseAmp
                self.senseAmp = None

        self.featureSizeInNano = _featureSizeInNano
        self.featureSize = _featureSizeInNano * 1e-9
        self.wireType = _wireType
        self.wireRepeaterType = _wireRepeaterType
        self.temperature = _temperature
        self.isLowSwing = _isLowSwing

        if self.wireRepeaterType != WireRepeaterType.repeated_none and self.isLowSwing:
            print("[Wire] Error: Low Swing is not supported for repeated wires!")
            sys.exit(-1)

        copper_resistivity = COPPER_RESISTIVITY

        # Initialize wire parameters based on feature size and wire type
        if _featureSizeInNano <= 22:
            self.featureSize = 22e-9
            self._initialize_22nm(copper_resistivity)
        elif _featureSizeInNano <= 32:
            self.featureSize = 32e-9
            self._initialize_32nm(copper_resistivity)
        elif _featureSizeInNano <= 45:
            self.featureSize = 45e-9
            self._initialize_45nm(copper_resistivity)
        elif _featureSizeInNano <= 65:
            self.featureSize = 65e-9
            self._initialize_65nm(copper_resistivity)
        elif _featureSizeInNano <= 90:
            self.featureSize = 90e-9
            self._initialize_90nm(copper_resistivity)
        elif _featureSizeInNano <= 120:
            self.featureSize = 120e-9
            self._initialize_120nm(copper_resistivity)
        elif _featureSizeInNano <= 200:
            self.featureSize = 200e-9
            self._initialize_200nm(copper_resistivity)
        else:
            self.featureSize = _featureSizeInNano * 1e-9
            self._initialize_default(copper_resistivity)

        # Calculate wire dimensions
        self.wireWidth = self.wirePitch / 2
        self.wireThickness = self.aspectRatio * self.wireWidth
        self.wireSpacing = self.wirePitch - self.wireWidth

        # Adjust copper resistivity for temperature
        copper_resistivity = copper_resistivity * (
            1 + COPPER_RESISTIVITY_TEMPERATURE_COEFFICIENT * (self.temperature - 293))

        # Calculate wire resistance and capacitance per unit length
        self.resWirePerUnit = CalculateWireResistance(
            copper_resistivity, self.wireWidth, self.wireThickness,
            self.barrierThickness, 0, 1)  # 0 = dishing thickness, 1 = alpha scatter

        self.capWirePerUnit = CalculateWireCapacitance(
            PERMITTIVITY, self.wireWidth, self.wireThickness, self.wireSpacing,
            self.ildThickness, 1.5, self.horizontalDielectric, 3.9, 1.15e-10)
            # 1.5 = miller value, 3.9 = vertical dielectric, 1.15e-10 = fringe cap

        # Handle repeated wires
        if self.wireRepeaterType != WireRepeaterType.repeated_none:
            self.findOptimalRepeater()

            if self.wireRepeaterType != WireRepeaterType.repeated_opt:
                # The repeated wire is not fully latency optimized
                penalty_map = {
                    WireRepeaterType.repeated_5: 0.05,
                    WireRepeaterType.repeated_10: 0.10,
                    WireRepeaterType.repeated_20: 0.20,
                    WireRepeaterType.repeated_30: 0.30,
                    WireRepeaterType.repeated_40: 0.40,
                    WireRepeaterType.repeated_50: 0.50
                }
                penalty = penalty_map.get(self.wireRepeaterType, 0.50)
                self.findPenalizedRepeater(penalty)

            # Calculate repeated wire pitch
            area, self.repeaterHeight, self.repeaterWidth = CalculateGateArea(
                INV, 1,
                self.repeaterSize * MIN_NMOS_SIZE * g.tech.featureSize,
                self.repeaterSize * MIN_NMOS_SIZE * g.tech.featureSize * g.tech.pnSizeRatio,
                1e41, g.tech)

            if self.repeaterWidth < self.repeaterHeight:
                self.repeaterWidth, self.repeaterHeight = self.repeaterHeight, self.repeaterWidth

            self.repeatedWirePitch = self.wirePitch + self.repeaterWidth

        self.initialized = True

    def _initialize_22nm(self, copper_resistivity):
        """Initialize parameters for 22nm node."""
        if self.wireType == WireType.local_aggressive:
            self.barrierThickness = 0.00e-6
            self.horizontalDielectric = 2.55
            self.wirePitch = 2 * self.featureSize
            self.aspectRatio = 1.9
            self.ildThickness = self.aspectRatio * self.featureSize
            copper_resistivity = 6.0e-8
        elif self.wireType == WireType.local_conservative:
            self.barrierThickness = 0.0021e-6
            self.horizontalDielectric = 3
            self.wirePitch = 2 * self.featureSize
            self.aspectRatio = 1.9
            self.ildThickness = self.aspectRatio * self.featureSize
            copper_resistivity = 6.0e-8
        elif self.wireType == WireType.semi_aggressive:
            self.barrierThickness = 0.00e-6
            self.horizontalDielectric = 2.55
            self.wirePitch = 4 * self.featureSize
            self.aspectRatio = 1.9
            self.ildThickness = 2 * self.aspectRatio * self.featureSize
            copper_resistivity = 6.0e-8
        elif self.wireType == WireType.semi_conservative:
            self.barrierThickness = 0.0021e-6
            self.horizontalDielectric = 3
            self.wirePitch = 4 * self.featureSize
            self.aspectRatio = 1.9
            self.ildThickness = 2 * self.aspectRatio * self.featureSize
            copper_resistivity = 6.0e-8
        elif self.wireType == WireType.global_aggressive:
            self.barrierThickness = 0.00e-6
            self.horizontalDielectric = 2.55
            self.wirePitch = 8 * self.featureSize
            self.aspectRatio = 2.34
            self.ildThickness = 0.42e-6 * 22 / 32
            copper_resistivity = 3.0e-8
        elif self.wireType == WireType.global_conservative:
            self.barrierThickness = 0.0063e-6
            self.horizontalDielectric = 3
            self.wirePitch = 8 * self.featureSize
            self.aspectRatio = 2.34
            self.ildThickness = 0.385e-6 * 22 / 32
            copper_resistivity = 3.0e-8
        else:  # dram_wordline
            self.barrierThickness = 0e-6
            self.horizontalDielectric = 0
            self.wirePitch = 2 * self.featureSize
            self.aspectRatio = 0
            self.ildThickness = 0e-6

        return copper_resistivity

    def _initialize_32nm(self, copper_resistivity):
        """Initialize parameters for 32nm node."""
        if self.wireType == WireType.local_aggressive:
            self.barrierThickness = 0.00e-6
            self.horizontalDielectric = 2.82
            self.wirePitch = 2 * self.featureSize
            self.aspectRatio = 1.8
            self.ildThickness = self.aspectRatio * self.featureSize
            copper_resistivity = 5.0e-8
        elif self.wireType == WireType.local_conservative:
            self.barrierThickness = 0.0026e-6
            self.horizontalDielectric = 3.16
            self.wirePitch = 2 * self.featureSize
            self.aspectRatio = 1.8
            self.ildThickness = self.aspectRatio * self.featureSize
            copper_resistivity = 5.0e-8
        elif self.wireType == WireType.semi_aggressive:
            self.barrierThickness = 0.00e-6
            self.horizontalDielectric = 2.82
            self.wirePitch = 4 * self.featureSize
            self.aspectRatio = 1.9
            self.ildThickness = 2 * self.aspectRatio * self.featureSize
            copper_resistivity = 5.0e-8
        elif self.wireType == WireType.semi_conservative:
            self.barrierThickness = 0.0026e-6
            self.horizontalDielectric = 3.16
            self.wirePitch = 4 * self.featureSize
            self.aspectRatio = 1.9
            self.ildThickness = 2 * self.aspectRatio * self.featureSize
            copper_resistivity = 5.0e-8
        elif self.wireType == WireType.global_aggressive:
            self.barrierThickness = 0.00e-6
            self.horizontalDielectric = 2.82
            self.wirePitch = 8 * self.featureSize
            self.aspectRatio = 2.34
            self.ildThickness = 0.42e-6
            copper_resistivity = 2.5e-8
        elif self.wireType == WireType.global_conservative:
            self.barrierThickness = 0.0078e-6
            self.horizontalDielectric = 3.16
            self.wirePitch = 8 * self.featureSize
            self.aspectRatio = 2.34
            self.ildThickness = 0.385e-6
            copper_resistivity = 2.5e-8
        else:  # dram_wordline
            self.barrierThickness = 0e-6
            self.horizontalDielectric = 0
            self.wirePitch = 2 * self.featureSize
            self.aspectRatio = 0
            self.ildThickness = 0e-6

        return copper_resistivity

    def _initialize_45nm(self, copper_resistivity):
        """Initialize parameters for 45nm node."""
        if self.wireType == WireType.local_aggressive:
            self.barrierThickness = 0.00e-6
            self.horizontalDielectric = 2.6
            self.wirePitch = 0.102e-6
            self.aspectRatio = 1.8
            self.ildThickness = 0.0918e-6
            copper_resistivity = 4.08e-8
        elif self.wireType == WireType.local_conservative:
            self.barrierThickness = 0.0033e-6
            self.horizontalDielectric = 2.9
            self.wirePitch = 0.102e-6
            self.aspectRatio = 1.8
            self.ildThickness = 0.0918e-6
            copper_resistivity = 4.08e-8
        elif self.wireType == WireType.semi_aggressive:
            self.barrierThickness = 0.00e-6
            self.horizontalDielectric = 2.6
            self.wirePitch = 4 * self.featureSize
            self.aspectRatio = 1.8
            self.ildThickness = 2 * self.aspectRatio * self.featureSize
            copper_resistivity = 4.08e-8
        elif self.wireType == WireType.semi_conservative:
            self.barrierThickness = 0.0033e-6
            self.horizontalDielectric = 2.9
            self.wirePitch = 4 * self.featureSize
            self.aspectRatio = 1.8
            self.ildThickness = 2 * self.aspectRatio * self.featureSize
            copper_resistivity = 4.08e-8
        elif self.wireType == WireType.global_aggressive:
            self.barrierThickness = 0.00e-6
            self.horizontalDielectric = 2.6
            self.wirePitch = 8 * self.featureSize
            self.aspectRatio = 2.34
            self.ildThickness = 0.63e-6
            copper_resistivity = 2.06e-8
        elif self.wireType == WireType.global_conservative:
            self.barrierThickness = 0.01e-6
            self.horizontalDielectric = 2.9
            self.wirePitch = 8 * self.featureSize
            self.aspectRatio = 2.34
            self.ildThickness = 0.55e-6
            copper_resistivity = 2.06e-8
        else:  # dram_wordline
            self.barrierThickness = 0e-6
            self.horizontalDielectric = 0
            self.wirePitch = 2 * self.featureSize
            self.aspectRatio = 0
            self.ildThickness = 0e-6

        return copper_resistivity

    def _initialize_65nm(self, copper_resistivity):
        """Initialize parameters for 65nm node."""
        if self.wireType == WireType.local_aggressive:
            self.barrierThickness = 0.00e-6
            self.horizontalDielectric = 2.303
            self.wirePitch = 2.5 * self.featureSize
            self.aspectRatio = 2.7
            self.ildThickness = 0.405e-6
        elif self.wireType == WireType.local_conservative:
            self.barrierThickness = 0.006e-6
            self.horizontalDielectric = 2.734
            self.wirePitch = 2.5 * self.featureSize
            self.aspectRatio = 2.0
            self.ildThickness = 0.405e-6
        elif self.wireType == WireType.semi_aggressive:
            self.barrierThickness = 0.00e-6
            self.horizontalDielectric = 2.303
            self.wirePitch = 4 * self.featureSize
            self.aspectRatio = 2.7
            self.ildThickness = 0.405e-6
        elif self.wireType == WireType.semi_conservative:
            self.barrierThickness = 0.006e-6
            self.horizontalDielectric = 2.734
            self.wirePitch = 4 * self.featureSize
            self.aspectRatio = 2.0
            self.ildThickness = 0.405e-6
        elif self.wireType == WireType.global_aggressive:
            self.barrierThickness = 0.00e-6
            self.horizontalDielectric = 2.303
            self.wirePitch = 8 * self.featureSize
            self.aspectRatio = 2.8
            self.ildThickness = 0.81e-6
        elif self.wireType == WireType.global_conservative:
            self.barrierThickness = 0.006e-6
            self.horizontalDielectric = 2.734
            self.wirePitch = 8 * self.featureSize
            self.aspectRatio = 2.2
            self.ildThickness = 0.77e-6
        else:  # dram_wordline
            self.barrierThickness = 0e-6
            self.horizontalDielectric = 0
            self.wirePitch = 2 * self.featureSize
            self.aspectRatio = 0
            self.ildThickness = 0e-6

        return copper_resistivity

    def _initialize_90nm(self, copper_resistivity):
        """Initialize parameters for 90nm node."""
        if self.wireType == WireType.local_aggressive:
            self.barrierThickness = 0.01e-6
            self.horizontalDielectric = 2.709
            self.wirePitch = 2.5 * self.featureSize
            self.aspectRatio = 2.4
            self.ildThickness = 0.48e-6
        elif self.wireType == WireType.local_conservative:
            self.barrierThickness = 0.008e-6
            self.horizontalDielectric = 3.038
            self.wirePitch = 2.5 * self.featureSize
            self.aspectRatio = 2.0
            self.ildThickness = 0.48e-6
        elif self.wireType == WireType.semi_aggressive:
            self.barrierThickness = 0.01e-6
            self.horizontalDielectric = 2.709
            self.wirePitch = 4 * self.featureSize
            self.aspectRatio = 2.4
            self.ildThickness = 0.48e-6
        elif self.wireType == WireType.semi_conservative:
            self.barrierThickness = 0.008e-6
            self.horizontalDielectric = 3.038
            self.wirePitch = 4 * self.featureSize
            self.aspectRatio = 2.0
            self.ildThickness = 0.48e-6
        elif self.wireType == WireType.global_aggressive:
            self.barrierThickness = 0.01e-6
            self.horizontalDielectric = 2.709
            self.wirePitch = 8 * self.featureSize
            self.aspectRatio = 2.7
            self.ildThickness = 0.96e-6
        elif self.wireType == WireType.global_conservative:
            self.barrierThickness = 0.008e-6
            self.horizontalDielectric = 3.038
            self.wirePitch = 8 * self.featureSize
            self.aspectRatio = 2.2
            self.ildThickness = 1.1e-6
        else:  # dram_wordline
            self.barrierThickness = 0e-6
            self.horizontalDielectric = 0
            self.wirePitch = 2 * self.featureSize
            self.aspectRatio = 0
            self.ildThickness = 0e-6

        return copper_resistivity

    def _initialize_120nm(self, copper_resistivity):
        """Initialize parameters for 120nm node."""
        if self.wireType == WireType.local_aggressive:
            self.barrierThickness = 0.012e-6
            self.horizontalDielectric = 3.3
            self.wirePitch = 240e-9
            self.aspectRatio = 1.6
            self.ildThickness = 0.48e-6
        elif self.wireType == WireType.local_conservative:
            self.barrierThickness = 0.01e-6
            self.horizontalDielectric = 3.6
            self.wirePitch = 240e-9
            self.aspectRatio = 1.4
            self.ildThickness = 0.48e-6
        elif self.wireType == WireType.semi_aggressive:
            self.barrierThickness = 0.012e-6
            self.horizontalDielectric = 3.3
            self.wirePitch = 320e-9
            self.aspectRatio = 1.7
            self.ildThickness = 0.48e-6
        elif self.wireType == WireType.semi_conservative:
            self.barrierThickness = 0.01e-6
            self.horizontalDielectric = 3.6
            self.wirePitch = 320e-9
            self.aspectRatio = 1.5
            self.ildThickness = 0.48e-6
        elif self.wireType == WireType.global_aggressive:
            self.barrierThickness = 0.012e-6
            self.horizontalDielectric = 3.3
            self.wirePitch = 475e-9
            self.aspectRatio = 2.1
            self.ildThickness = 0.96e-6
        elif self.wireType == WireType.global_conservative:
            self.barrierThickness = 0.01e-6
            self.horizontalDielectric = 3.6
            self.wirePitch = 475e-9
            self.aspectRatio = 1.9
            self.ildThickness = 1.1e-6
        else:  # dram_wordline
            self.barrierThickness = 0e-6
            self.horizontalDielectric = 0
            self.wirePitch = 2 * self.featureSize
            self.aspectRatio = 0
            self.ildThickness = 0e-6

        return copper_resistivity

    def _initialize_200nm(self, copper_resistivity):
        """Initialize parameters for 200nm node."""
        if self.wireType == WireType.local_aggressive:
            self.barrierThickness = 0.016e-6
            self.horizontalDielectric = 3.75
            self.wirePitch = 0.45e-6
            self.aspectRatio = 2.4
            self.ildThickness = 1e-6
        elif self.wireType == WireType.local_conservative:
            self.barrierThickness = 0.016e-6 * 0.8
            self.horizontalDielectric = 3.75 * 3.038 / 2.709
            self.wirePitch = 0.45e-6
            self.aspectRatio = 1.2
            self.ildThickness = 1e-6
        elif self.wireType == WireType.semi_aggressive:
            self.barrierThickness = 0.016e-6
            self.horizontalDielectric = 3.75
            self.wirePitch = 0.575e-6
            self.aspectRatio = 2.1
            self.ildThickness = 1e-6
        elif self.wireType == WireType.semi_conservative:
            self.barrierThickness = 0.016e-6 * 0.8
            self.horizontalDielectric = 3.75 * 3.038 / 2.709
            self.wirePitch = 0.575e-6
            self.aspectRatio = 2.1 * 2.0 / 2.4
            self.ildThickness = 1e-6
        elif self.wireType == WireType.global_aggressive:
            self.barrierThickness = 0.016e-6
            self.horizontalDielectric = 3.75
            self.wirePitch = 0.945e-6
            self.aspectRatio = 2.1
            self.ildThickness = 2e-6
        elif self.wireType == WireType.global_conservative:
            self.barrierThickness = 0.016e-6 * 0.8
            self.horizontalDielectric = 3.75 * 3.038 / 2.709
            self.wirePitch = 0.945e-6
            self.aspectRatio = 2.1 * 2.2 / 2.7
            self.ildThickness = 2.2e-6
        else:  # dram_wordline
            self.barrierThickness = 0e-6
            self.horizontalDielectric = 0
            self.wirePitch = 2 * self.featureSize
            self.aspectRatio = 0
            self.ildThickness = 0e-6

        return copper_resistivity

    def _initialize_default(self, copper_resistivity):
        """Initialize default parameters for unlisted feature sizes."""
        self.barrierThickness = 0e-6
        self.horizontalDielectric = 0
        self.wirePitch = 2 * self.featureSize
        self.aspectRatio = 0
        self.ildThickness = 0e-6
        return copper_resistivity

    def CalculateLatencyAndPower(self, _wireLength, delay=None, dynamicEnergy=None, leakagePower=None):
        """Calculate wire delay and power for a given length.

        Args:
            _wireLength: Wire length in meters
            delay: List to store delay (pass [0] to get result)
            dynamicEnergy: List to store dynamic energy (pass [0] to get result)
            leakagePower: List to store leakage power (pass [0] to get result)

        Returns:
            Tuple of (delay, dynamicEnergy, leakagePower) if no output lists provided
        """
        if not self.initialized:
            print("[Wire] Error: Require initialization first!")
            return None

        # Create output containers if not provided
        return_tuple = False
        if delay is None:
            delay = [0.0]
            return_tuple = True
        if dynamicEnergy is None:
            dynamicEnergy = [0.0]
        if leakagePower is None:
            leakagePower = [0.0]

        if self.isLowSwing:
            # Low-swing wire implementation
            if self.wireRepeaterType == WireRepeaterType.repeated_none:
                self._calculate_low_swing_wire(_wireLength, delay, dynamicEnergy, leakagePower)
            else:
                print("Error: Low Swing Wires with Repeaters is not supported in this version!")
                sys.exit(-1)
        else:
            # Regular wire (not low-swing)
            if self.wireRepeaterType == WireRepeaterType.repeated_none:
                # Non-repeated wire: simple RC delay model
                delay[0] = 2.3 * self.resWirePerUnit * self.capWirePerUnit * _wireLength * _wireLength / 2
                dynamicEnergy[0] = self.capWirePerUnit * _wireLength * g.tech.vdd * g.tech.vdd
                leakagePower[0] = 0
            else:
                # Repeated wire: use repeater-based model
                delay[0] = self.getRepeatedWireUnitDelay() * _wireLength
                dynamicEnergy[0] = self.getRepeatedWireUnitDynamicEnergy() * _wireLength
                leakagePower[0] = self.getRepeatedWireUnitLeakage() * _wireLength

        if return_tuple:
            return (delay[0], dynamicEnergy[0], leakagePower[0])

    def _calculate_low_swing_wire(self, wireLength, delay, dynamicEnergy, leakagePower):
        """Calculate low-swing wire delay and power.

        This implements the complex low-swing signaling model with transmitter,
        wire, and sense amplifier receiver.
        """
        widthNmos = MIN_NMOS_SIZE * g.tech.featureSize
        widthPmos = widthNmos * g.tech.pnSizeRatio

        # Calculate ramp input
        capInput, capOutput = CalculateGateCapacitance(
            INV, 1, widthNmos, widthPmos, g.tech.featureSize * 40, g.tech)
        capLoad = capInput + capOutput
        resPullUp = CalculateOnResistance(widthPmos, PMOS, g.inputParameter.temperature, g.tech)
        tr = resPullUp * capLoad
        gm = CalculateTransconductance(widthPmos, PMOS, g.tech)
        beta = 1 / (resPullUp * gm)
        riseTime, temp = horowitz(tr, beta, 1e20)

        resPullDown = CalculateOnResistance(widthNmos, NMOS, g.inputParameter.temperature, g.tech)
        tr = resPullDown * capLoad
        gm = CalculateTransconductance(widthNmos, NMOS, g.tech)
        beta = 1 / (resPullDown * gm)
        fallTime, temp = horowitz(tr, beta, riseTime)
        rampInput = fallTime

        # Calculate FO4 delay
        capLoad = capOutput + 4 * capInput
        tr = resPullDown * capLoad
        delayFO4, temp = horowitz(tr, beta, 1e20)

        # Calculate the size of driver
        capWire = self.capWirePerUnit * wireLength
        resWire = self.resWirePerUnit * wireLength
        resDriver = ((-8) * delayFO4 / (math.log(0.5) * capWire)) / RES_ADJ
        widthNmosDriver = resPullDown * widthNmos / resDriver
        widthNmosDriver = MIN(widthNmosDriver, MAX_NMOS_SIZE * g.tech.featureSize)
        widthNmosDriver = MAX(widthNmosDriver, MIN_NMOS_SIZE * g.tech.featureSize)

        if resWire * capWire > 8 * delayFO4:
            widthNmosDriver = g.inputParameter.maxNmosSize * g.tech.featureSize

        # Size the inverter appropriately to minimize transmitter delay
        capGateDriver, capTemp = CalculateGateCapacitance(
            INV, 1, widthNmosDriver, 0, g.tech.featureSize * 40, g.tech)
        capInput2, capOutput2 = CalculateGateCapacitance(
            INV, 1, 2 * widthNmos, 2 * widthPmos, g.tech.featureSize * 40, g.tech)
        stageEffort = math.sqrt(
            ((2 + g.tech.pnSizeRatio) / (1 + g.tech.pnSizeRatio)) * capGateDriver / capInput2)
        reqCin = (((2 + g.tech.pnSizeRatio) / (1 + g.tech.pnSizeRatio)) * capGateDriver) / stageEffort
        capInput3, capTemp = CalculateGateCapacitance(
            INV, 1, widthNmos, widthPmos, g.tech.featureSize * 40, g.tech)
        sizeInverter = reqCin / capInput3
        sizeInverter = MAX(sizeInverter, 1)

        # NAND gate delay
        resPullDown *= 2
        beta = 1 / (resPullDown * gm)
        capNandInput, capNandOutput = CalculateGateCapacitance(
            NAND, 2, 2 * widthNmos, widthPmos, g.tech.featureSize * 40, g.tech)
        capInput4, capOutput4 = CalculateGateCapacitance(
            INV, 1, sizeInverter * widthNmos, sizeInverter * widthPmos,
            g.tech.featureSize * 40, g.tech)
        capLoad = capNandOutput + capInput4
        tr = resPullDown * capLoad
        delay[0], rampInput = horowitz(tr, beta, rampInput)
        dynamicEnergy[0] = capLoad * g.tech.vdd * g.tech.vdd

        # Inverter delay
        resPullDown = CalculateOnResistance(
            sizeInverter * widthNmos, NMOS, g.inputParameter.temperature, g.tech)
        gm = CalculateTransconductance(widthNmos, NMOS, g.tech)
        beta = 1 / (resPullDown * gm)
        capLoad = capOutput4 + capGateDriver
        tr = resPullDown * capLoad
        delayInv, rampInput = horowitz(tr, beta, rampInput)
        delay[0] += delayInv
        dynamicEnergy[0] += capLoad * g.tech.vdd * g.tech.vdd

        # Leakage power
        leakagePower[0] = 2 * g.tech.vdd * CalculateGateLeakage(
            INV, 1, sizeInverter * widthNmos, sizeInverter * widthPmos,
            g.inputParameter.temperature, g.tech)
        leakagePower[0] += 2 * g.tech.vdd * CalculateGateLeakage(
            NAND, 2, 2 * widthNmos, widthPmos, g.inputParameter.temperature, g.tech)
        leakagePower[0] *= 2

        # Initialize sense amplifier
        self.senseAmp = SenseAmp()
        self.senseAmp.Initialize(1, False, g.cell.minSenseVoltage, 1)
        self.senseAmp.CalculateRC()

        # NMOS delay + wire delay
        drainCapDriver = CalculateDrainCap(widthNmosDriver, NMOS, g.tech.featureSize * 40, g.tech)
        capLoad = capWire + drainCapDriver * 2 + self.senseAmp.capLoad
        resPullDown = CalculateOnResistance(widthNmosDriver, NMOS, g.inputParameter.temperature, g.tech)
        gm = CalculateTransconductance(widthNmosDriver, NMOS, g.tech)
        beta = 1 / (resPullDown * gm)
        tr = (resPullDown * RES_ADJ * (capWire + drainCapDriver * 2) +
              capWire * resWire / 2 + (resPullDown + resWire) * self.senseAmp.capLoad)
        delayWire, temp = horowitz(tr, beta, rampInput)
        delay[0] += delayWire
        dynamicEnergy[0] += capLoad * VOL_SWING * 0.4  # 0.4V is the overdrive voltage
        dynamicEnergy[0] *= 2
        leakagePower[0] += 4 * g.tech.vdd * CalculateGateLeakage(
            INV, 1, widthNmosDriver, 0, g.inputParameter.temperature, g.tech)

        # Sense amplifier delay and power
        delay[0] += self.senseAmp.readLatency
        dynamicEnergy[0] += self.senseAmp.readDynamicEnergy
        leakagePower[0] += self.senseAmp.leakage

    def findOptimalRepeater(self):
        """Find optimal repeater size and spacing for minimum delay."""
        # Use minimum sized inverter
        nmosSize = MIN_NMOS_SIZE * g.tech.featureSize
        pmosSize = nmosSize * g.tech.pnSizeRatio
        inputCap = CalculateGateCap(nmosSize, g.tech) + CalculateGateCap(pmosSize, g.tech)
        outputCap = (CalculateDrainCap(nmosSize, NMOS, 1, g.tech) +
                     CalculateDrainCap(pmosSize, PMOS, 1, g.tech))
        outputRes = (CalculateOnResistance(nmosSize, NMOS, g.inputParameter.temperature, g.tech) +
                     CalculateOnResistance(pmosSize, PMOS, g.inputParameter.temperature, g.tech))

        self.repeaterSize = math.sqrt(
            outputRes * self.capWirePerUnit / inputCap / self.resWirePerUnit)
        self.repeaterSpacing = math.sqrt(
            2 * outputRes * (outputCap + inputCap) / (self.resWirePerUnit * self.capWirePerUnit))

    def findPenalizedRepeater(self, _penalty):
        """Find repeater configuration with specified delay penalty.

        This method searches for a repeater configuration that meets a target
        delay (optimal + penalty) while minimizing energy and leakage.

        Args:
            _penalty: Fractional delay penalty (e.g., 0.1 for 10% slower than optimal)
        """
        targetDelay = self.getRepeatedWireUnitDelay() * (1 + _penalty)
        currentDynamicEnergy = self.getRepeatedWireUnitDynamicEnergy()
        currentLeakage = self.getRepeatedWireUnitLeakage()

        targetRepeaterSpacing = self.repeaterSpacing
        targetRepeaterSize = self.repeaterSize
        stepSpacing = 100e-6  # 100um
        endSpacing = 4 * self.repeaterSpacing
        stepSize = 1  # minimum buffer size
        endSize = 1

        # Start finding the target repeated wire
        spacing = self.repeaterSpacing
        while spacing <= endSpacing:
            size = self.repeaterSize
            while size >= endSize:
                self.repeaterSpacing = spacing
                self.repeaterSize = size

                thisDelay = self.getRepeatedWireUnitDelay()
                thisDynamicEnergy = self.getRepeatedWireUnitDynamicEnergy()
                thisLeakage = self.getRepeatedWireUnitLeakage()

                if (thisDelay <= targetDelay and
                    thisDynamicEnergy / currentDynamicEnergy + thisLeakage / currentLeakage < 2):
                    currentDynamicEnergy = thisDynamicEnergy
                    currentLeakage = thisLeakage
                    targetRepeaterSpacing = spacing
                    targetRepeaterSize = size

                size -= stepSize
            spacing += stepSpacing

        self.repeaterSpacing = targetRepeaterSpacing
        self.repeaterSize = targetRepeaterSize

    def getRepeatedWireUnitDelay(self):
        """Get delay per unit length for repeated wire.

        Returns:
            Delay per meter (s/m)
        """
        nmosSize = MIN_NMOS_SIZE * g.tech.featureSize * self.repeaterSize
        pmosSize = nmosSize * g.tech.pnSizeRatio
        inputCap = CalculateGateCap(nmosSize, g.tech) + CalculateGateCap(pmosSize, g.tech)
        outputCap = (CalculateDrainCap(nmosSize, NMOS, 1, g.tech) +
                     CalculateDrainCap(pmosSize, PMOS, 1, g.tech))
        outputRes = (CalculateOnResistance(nmosSize, NMOS, g.inputParameter.temperature, g.tech) +
                     CalculateOnResistance(pmosSize, PMOS, g.inputParameter.temperature, g.tech))
        wireCap = self.capWirePerUnit * self.repeaterSpacing
        wireRes = self.resWirePerUnit * self.repeaterSpacing

        tau = (outputRes * (inputCap + outputCap) + outputRes * wireCap +
               wireRes * outputCap + 0.5 * wireRes * wireCap)

        # Return as a unit value
        return 0.693 * tau / self.repeaterSpacing

    def getRepeatedWireUnitDynamicEnergy(self):
        """Get dynamic energy per unit length for repeated wire.

        Returns:
            Energy per meter (J/m)
        """
        nmosSize = MIN_NMOS_SIZE * g.tech.featureSize * self.repeaterSize
        pmosSize = nmosSize * g.tech.pnSizeRatio
        inputCap = CalculateGateCap(nmosSize, g.tech) + CalculateGateCap(pmosSize, g.tech)
        outputCap = (CalculateDrainCap(nmosSize, NMOS, 1, g.tech) +
                     CalculateDrainCap(pmosSize, PMOS, 1, g.tech))
        wireCap = self.capWirePerUnit * self.repeaterSpacing

        switchingEnergy = (inputCap + outputCap + wireCap) * g.tech.vdd * g.tech.vdd
        shortCircuitEnergy = 0  # No short circuit energy in this version

        return (switchingEnergy + shortCircuitEnergy) / self.repeaterSpacing

    def getRepeatedWireUnitLeakage(self):
        """Get leakage power per unit length for repeated wire.

        Returns:
            Power per meter (W/m)
        """
        nmosSize = MIN_NMOS_SIZE * g.tech.featureSize * self.repeaterSize
        pmosSize = nmosSize * g.tech.pnSizeRatio
        leakagePerRepeater = (CalculateGateLeakage(
            INV, 1, nmosSize, pmosSize, g.inputParameter.temperature, g.tech) * g.tech.vdd)

        return leakagePerRepeater / self.repeaterSpacing

    def PrintProperty(self):
        """Print wire properties."""
        if self.wireRepeaterType == WireRepeaterType.repeated_none:
            print("Wire Type: passive (without repeaters)", end="")
            if self.isLowSwing:
                print(" Low Swing")
            else:
                print()
            print(f"Wire Resistance: {self.resWirePerUnit / 1e6:.4f} ohm/um")
            print(f"Wire Capacitance: {self.capWirePerUnit / 1e6:.4e} F/um")
        else:
            print("Wire type: active (with repeaters)")
            print(f"Repeater Size: {self.repeaterSize:.4f}")
            print(f"Repeater Spacing: {self.repeaterSpacing * 1e3:.4f} mm")
            print(f"Delay: {self.getRepeatedWireUnitDelay() * 1e6:.4f} ns/mm")
            print(f"Dynamic Energy: {self.getRepeatedWireUnitDynamicEnergy() * 1e6:.4f} nJ/mm")
            print(f"Subthreshold Leakage Power: {self.getRepeatedWireUnitLeakage():.4e} W/mm")

    def assign(self, rhs):
        """Assignment operator to copy from another Wire instance.

        Args:
            rhs: Another Wire instance to copy from

        Returns:
            self: Returns self to allow chaining
        """
        self.initialized = rhs.initialized
        self.featureSizeInNano = rhs.featureSizeInNano
        self.featureSize = rhs.featureSize
        self.wireType = rhs.wireType
        self.wireRepeaterType = rhs.wireRepeaterType
        self.temperature = rhs.temperature
        self.isLowSwing = rhs.isLowSwing
        self.barrierThickness = rhs.barrierThickness
        self.horizontalDielectric = rhs.horizontalDielectric
        self.wirePitch = rhs.wirePitch
        self.aspectRatio = rhs.aspectRatio
        self.ildThickness = rhs.ildThickness
        self.wireWidth = rhs.wireWidth
        self.wireThickness = rhs.wireThickness
        self.wireSpacing = rhs.wireSpacing
        self.repeaterSize = rhs.repeaterSize
        self.repeaterSpacing = rhs.repeaterSpacing
        self.repeaterHeight = rhs.repeaterHeight
        self.repeaterWidth = rhs.repeaterWidth
        self.repeatedWirePitch = rhs.repeatedWirePitch
        self.resWirePerUnit = rhs.resWirePerUnit
        self.capWirePerUnit = rhs.capWirePerUnit

        return self
