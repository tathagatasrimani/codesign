#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
# This file contains code from CACTI-3DD, (c) 2012 Hewlett-Packard Development Company, L.P.
#See LICENSE_CACTI3DD file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

import math
from typedef import DeviceRoadmap, TSV_type, WireType
from constant import (NUMBER_INTERCONNECT_PROJECTION_TYPES,
                      BULK_CU_RESISTIVITY, PERMITTIVITY_FREE_SPACE)


class Technology:
    """
    Technology class that stores and manages technology parameters for different
    process nodes and device roadmaps (HP, LSTP, LOP, EDRAM).

    This class handles:
    - Transistor parameters (Ion, Ioff, mobility, gate capacitance, etc.)
    - Temperature-dependent current arrays (300K-400K)
    - TSV (Through-Silicon Via) parameters for 3D integration
    - Technology node-specific values from MASTAR models
    """

    def __init__(self):
        """Initialize Technology object with default values."""
        # Initialization flag
        self.initialized = False

        # Feature size
        self.featureSizeInNano = 0
        self.featureSize = 0.0

        # Device roadmap
        self.deviceRoadmap = DeviceRoadmap.HP

        # Voltage parameters
        self.vdd = 0.0
        self.vpp = 0.0
        self.vth = 0.0
        self.vdsatNmos = 0.0
        self.vdsatPmos = 0.0

        # Gate parameters
        self.phyGateLength = 0.0
        self.capIdealGate = 0.0
        self.capFringe = 0.0
        self.capJunction = 0.0
        self.capOverlap = 0.0
        self.capSidewall = 0.0
        self.capDrainToChannel = 0.0
        self.capOx = 0.0
        self.buildInPotential = 0.0

        # Mobility parameters
        self.effectiveElectronMobility = 0.0
        self.effectiveHoleMobility = 0.0

        # Device sizing
        self.pnSizeRatio = 0.0
        self.effectiveResistanceMultiplier = 0.0

        # Current arrays (temperature-dependent, 300K to 400K)
        # Index 0-100 represents temperatures from 300K to 400K
        self.currentOnNmos = [0.0] * 101
        self.currentOnPmos = [0.0] * 101
        self.currentOffNmos = [0.0] * 101
        self.currentOffPmos = [0.0] * 101

        # Poly wire capacitance
        self.capPolywire = 0.0

        # TSV parameters
        self.capTSV = [0.0] * TSV_type.NUM_TSV_TYPES
        self.resTSV = [0.0] * TSV_type.NUM_TSV_TYPES
        self.areaTSV = [0.0] * TSV_type.NUM_TSV_TYPES

        # Private TSV arrays
        self.layerCount = 0
        self.tsv_pitch = [[0.0 for _ in range(TSV_type.NUM_TSV_TYPES)]
                          for _ in range(NUMBER_INTERCONNECT_PROJECTION_TYPES)]
        self.tsv_diameter = [[0.0 for _ in range(TSV_type.NUM_TSV_TYPES)]
                            for _ in range(NUMBER_INTERCONNECT_PROJECTION_TYPES)]
        self.tsv_length = [[0.0 for _ in range(TSV_type.NUM_TSV_TYPES)]
                          for _ in range(NUMBER_INTERCONNECT_PROJECTION_TYPES)]
        self.tsv_dielec_thickness = [[0.0 for _ in range(TSV_type.NUM_TSV_TYPES)]
                                     for _ in range(NUMBER_INTERCONNECT_PROJECTION_TYPES)]
        self.tsv_contact_resistance = [[0.0 for _ in range(TSV_type.NUM_TSV_TYPES)]
                                       for _ in range(NUMBER_INTERCONNECT_PROJECTION_TYPES)]
        self.tsv_depletion_width = [[0.0 for _ in range(TSV_type.NUM_TSV_TYPES)]
                                   for _ in range(NUMBER_INTERCONNECT_PROJECTION_TYPES)]
        self.tsv_liner_dielectric_constant = [[0.0 for _ in range(TSV_type.NUM_TSV_TYPES)]
                                              for _ in range(NUMBER_INTERCONNECT_PROJECTION_TYPES)]
        self.tsv_parasitic_res = [[0.0 for _ in range(TSV_type.NUM_TSV_TYPES)]
                                 for _ in range(NUMBER_INTERCONNECT_PROJECTION_TYPES)]
        self.tsv_parasitic_cap = [[0.0 for _ in range(TSV_type.NUM_TSV_TYPES)]
                                 for _ in range(NUMBER_INTERCONNECT_PROJECTION_TYPES)]
        self.tsv_occupation_area = [[0.0 for _ in range(TSV_type.NUM_TSV_TYPES)]
                                   for _ in range(NUMBER_INTERCONNECT_PROJECTION_TYPES)]

    def Initialize(self, _featureSizeInNano, _deviceRoadmap, inputParameter):
        """
        Initialize technology parameters based on feature size and device roadmap.

        Args:
            _featureSizeInNano: Process node in nanometers (e.g., 65, 45, 32, 22)
            _deviceRoadmap: DeviceRoadmap enum (HP, LSTP, LOP, EDRAM)
            inputParameter: InputParameter object containing configuration
        """
        if self.initialized:
            print("Warning: Already initialized!")

        self.featureSizeInNano = _featureSizeInNano
        self.featureSize = _featureSizeInNano * 1e-9
        self.deviceRoadmap = _deviceRoadmap

        # Initialize technology parameters based on feature size and roadmap
        if _featureSizeInNano >= 180:
            self._init_180nm(_deviceRoadmap)
        elif _featureSizeInNano >= 120:
            self._init_100nm(_deviceRoadmap)
        elif _featureSizeInNano >= 90:
            self._init_90nm(_deviceRoadmap)
        elif _featureSizeInNano >= 65:
            self._init_65nm(_deviceRoadmap)
        elif _featureSizeInNano >= 45:
            self._init_45nm(_deviceRoadmap)
        elif _featureSizeInNano >= 32:
            self._init_32nm(_deviceRoadmap)
        elif _featureSizeInNano >= 22:
            self._init_22nm(_deviceRoadmap)
        else:
            print(f"Warning: Unsupported feature size {_featureSizeInNano}nm, using 22nm parameters")
            self._init_22nm(_deviceRoadmap)

        # Setup TSV parameters
        self._init_tsv_params(_featureSizeInNano)

        # Initialize to something -- will be changed in main loop later
        self.SetLayerCount(inputParameter, 2)

        # For non-DRAM types vpp is equal to vdd
        if _deviceRoadmap != DeviceRoadmap.EDRAM:
            self.vpp = self.vdd

        # Calculate derived capacitance values
        self.capOverlap = self.capIdealGate * 0.2
        cjd = 1e-3  # Bottom junction capacitance, Unit: F/m^2
        cjswd = 2.5e-10  # Isolation-edge sidewall junction capacitance, Unit: F/m
        cjswgd = 0.5e-10  # Gate-edge sidewall junction capacitance, Unit: F/m
        mjd = 0.5  # Bottom junction capacitance grating coefficient
        mjswd = 0.33  # Isolation-edge sidewall junction capacitance grading coefficient
        mjswgd = 0.33  # Gate-edge sidewall junction capacitance grading coefficient
        self.buildInPotential = 0.9  # This value is from BSIM4
        self.capJunction = cjd / pow(1 + self.vdd / self.buildInPotential, mjd)
        self.capSidewall = cjswd / pow(1 + self.vdd / self.buildInPotential, mjswd)
        self.capDrainToChannel = cjswgd / pow(1 + self.vdd / self.buildInPotential, mjswgd)

        # Calculate saturation voltages
        self.vdsatNmos = self.phyGateLength * 1e5 / self.effectiveElectronMobility
        self.vdsatPmos = self.phyGateLength * 1e5 / self.effectiveHoleMobility

        # Properties not used so far
        self.capPolywire = 0.0

        # Interpolate current values for intermediate temperatures
        self._interpolate_currents()

        self.initialized = True

    def _init_180nm(self, _deviceRoadmap):
        """Initialize parameters for 180nm technology node."""
        if _deviceRoadmap == DeviceRoadmap.HP:
            self.vdd = 1.5
            self.vth = 300e-3
            self.phyGateLength = 0.1e-6
            self.capIdealGate = 8e-10
            self.capFringe = 2.5e-10
            self.capJunction = 1.00e-3
            self.capOx = 1e-2
            self.effectiveElectronMobility = 320e-4
            self.effectiveHoleMobility = 80e-4
            self.pnSizeRatio = 2.45
            self.effectiveResistanceMultiplier = 1.54
            for i in range(0, 101, 10):
                self.currentOnNmos[i] = 750
                self.currentOnPmos[i] = 350
                self.currentOffNmos[i] = 8e-3
                self.currentOffPmos[i] = 1.6e-2
        elif _deviceRoadmap == DeviceRoadmap.LSTP:
            self.vdd = 1.5
            self.vth = 600e-3
            self.phyGateLength = 0.16e-6
            self.capIdealGate = 8e-10
            self.capFringe = 2.5e-10
            self.capJunction = 1.00e-3
            self.capOx = 1e-2
            self.effectiveElectronMobility = 320e-4
            self.effectiveHoleMobility = 80e-4
            self.pnSizeRatio = 2.45
            self.effectiveResistanceMultiplier = 1.54
            for i in range(0, 101, 10):
                self.currentOnNmos[i] = 330
                self.currentOnPmos[i] = 168
                self.currentOffNmos[i] = 4.25e-6
                self.currentOffPmos[i] = 8.5e-6
        elif _deviceRoadmap == DeviceRoadmap.LOP:
            self.vdd = 1.2
            self.vth = 450e-3
            self.phyGateLength = 0.135e-6
            self.capIdealGate = 8e-10
            self.capFringe = 2.5e-10
            self.capJunction = 1.00e-3
            self.capOx = 1e-2
            self.effectiveElectronMobility = 330e-4
            self.effectiveHoleMobility = 90e-4
            self.pnSizeRatio = 2.45
            self.effectiveResistanceMultiplier = 1.54
            for i in range(0, 101, 10):
                self.currentOnNmos[i] = 490
                self.currentOnPmos[i] = 230
                self.currentOffNmos[i] = 4e-4
                self.currentOffPmos[i] = 8e-4
        else:
            print("Unknown device roadmap!")
            exit(1)

    def _init_100nm(self, _deviceRoadmap):
        """Initialize parameters for 100nm technology node."""
        if _deviceRoadmap == DeviceRoadmap.HP:
            self.vdd = 1.2
            self.vth = 218.04e-3
            self.phyGateLength = 0.0451e-6
            self.capIdealGate = 7.41e-10
            self.capFringe = 2.4e-10
            self.capJunction = 1.00e-3
            self.capOx = 1.64e-2
            self.effectiveElectronMobility = 249.59e-4
            self.effectiveHoleMobility = 59.52e-4
            self.pnSizeRatio = 2.45
            self.effectiveResistanceMultiplier = 1.54
            nmos_vals = [960.9, 947.9, 935.1, 922.5, 910.0, 897.7, 885.5, 873.6, 861.8, 850.1, 838.6]
            pmos_vals = [578.4, 567.8, 557.5, 547.4, 537.5, 527.8, 518.3, 509.1, 500.0, 491.1, 482.5]
            off_nmos = [1.90e-2, 2.35e-2, 2.86e-2, 3.45e-2, 4.12e-2, 4.87e-2, 5.71e-2, 6.64e-2, 7.67e-2, 8.80e-2, 1.00e-1]
            off_pmos = [3.82e-2, 3.84e-2, 3.87e-2, 3.90e-2, 3.93e-2, 3.97e-2, 4.01e-2, 4.05e-2, 4.10e-2, 4.16e-2, 4.22e-2]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = pmos_vals[i]
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_pmos[i]
        elif _deviceRoadmap == DeviceRoadmap.LSTP:
            self.vdd = 1.2
            self.vth = 501.25e-3
            self.phyGateLength = 0.075e-6
            self.capIdealGate = 8.62e-10
            self.capFringe = 2.5e-10
            self.capJunction = 1.00e-3
            self.capOx = 1.15e-2
            self.effectiveElectronMobility = 284.97e-4
            self.effectiveHoleMobility = 61.82e-4
            self.pnSizeRatio = 2.45
            self.effectiveResistanceMultiplier = 1.54
            nmos_vals = [422.5, 415.0, 407.7, 400.5, 393.6, 386.8, 380.1, 373.7, 367.4, 361.3, 355.5]
            pmos_vals = [204.9, 200.3, 195.9, 191.7, 187.5, 183.5, 179.7, 175.9, 172.3, 168.8, 165.4]
            off_nmos = [1.01e-5, 1.04e-5, 1.06e-5, 1.09e-5, 1.12e-5, 1.16e-5, 1.20e-5, 1.24e-5, 1.28e-5, 1.32e-5, 1.37e-5]
            off_pmos = [2.21e-5, 2.27e-5, 2.33e-5, 2.40e-5, 2.47e-5, 2.54e-5, 2.62e-5, 2.71e-5, 2.80e-5, 2.90e-5, 3.01e-5]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = pmos_vals[i]
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_pmos[i]
        elif _deviceRoadmap == DeviceRoadmap.LOP:
            self.vdd = 1.0
            self.vth = 312.60e-3
            self.phyGateLength = 0.065e-6
            self.capIdealGate = 6.34e-10
            self.capFringe = 2.5e-10
            self.capJunction = 1.00e-3
            self.capOx = 1.44e-2
            self.effectiveElectronMobility = 292.43e-4
            self.effectiveHoleMobility = 64.53e-4
            self.pnSizeRatio = 2.45
            self.effectiveResistanceMultiplier = 1.54
            nmos_vals = [531.4, 522.6, 514.0, 505.5, 497.3, 489.2, 481.3, 473.6, 466.1, 458.8, 451.6]
            pmos_vals = [278.5, 272.5, 266.8, 261.2, 255.8, 250.5, 245.4, 240.4, 235.6, 231.0, 226.4]
            off_nmos = [9.69e-4, 9.87e-4, 1.01e-3, 1.03e-3, 1.05e-3, 1.08e-3, 1.10e-3, 1.13e-3, 1.16e-3, 1.19e-3, 1.23e-3]
            off_pmos = [2.20e-3, 2.25e-3, 2.29e-3, 2.34e-3, 2.39e-3, 2.45e-3, 2.51e-3, 2.57e-3, 2.64e-3, 2.72e-3, 2.79e-3]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = pmos_vals[i]
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_pmos[i]
        else:
            print("Unknown device roadmap!")
            exit(1)

    def _init_90nm(self, _deviceRoadmap):
        """Initialize parameters for 90nm technology node."""
        if _deviceRoadmap == DeviceRoadmap.HP:
            self.vdd = 1.2
            self.vth = 197.95e-3
            self.phyGateLength = 0.037e-6
            self.capIdealGate = 6.38e-10
            self.capFringe = 2.5e-10
            self.capJunction = 1.00e-3
            self.capOx = 1.73e-2
            self.effectiveElectronMobility = 243.43e-4
            self.effectiveHoleMobility = 58.32e-4
            self.pnSizeRatio = 2.45
            self.effectiveResistanceMultiplier = 1.54
            nmos_vals = [1050.5, 1037.0, 1023.6, 1010.3, 997.2, 984.2, 971.4, 958.8, 946.3, 933.9, 921.7]
            pmos_vals = [638.7, 627.5, 616.5, 605.8, 595.2, 584.9, 574.7, 564.8, 555.1, 545.5, 536.2]
            off_nmos = [1.90e-2*2.73, 2.35e-2*2.73, 2.86e-2*2.73, 3.45e-2*2.73, 4.12e-2*2.73,
                       4.87e-2*2.73, 5.71e-2*2.73, 6.64e-2*2.73, 7.67e-2*2.73, 8.80e-2*2.73, 1.00e-1*2.73]
            off_pmos = [5.26e-2, 5.26e-2, 5.26e-2, 5.27e-2, 5.28e-2, 5.29e-2, 5.31e-2, 5.34e-2, 5.36e-2, 5.40e-2, 5.43e-2]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = pmos_vals[i]
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_pmos[i]
        elif _deviceRoadmap == DeviceRoadmap.LSTP:
            self.vdd = 1.2
            self.vth = 502.36e-3
            self.phyGateLength = 0.065e-6
            self.capIdealGate = 7.73e-10
            self.capFringe = 2.4e-10
            self.capJunction = 1.00e-3
            self.capOx = 1.19e-2
            self.effectiveElectronMobility = 277.94e-4
            self.effectiveHoleMobility = 60.64e-4
            self.pnSizeRatio = 2.44
            self.effectiveResistanceMultiplier = 1.92
            nmos_vals = [446.6, 438.7, 431.2, 423.8, 416.7, 409.7, 402.9, 396.3, 389.8, 383.5, 377.3]
            pmos_vals = [221.5, 216.6, 212.0, 207.4, 203.1, 198.8, 194.7, 190.7, 186.9, 183.1, 179.5]
            off_nmos = [9.45e-6, 9.67e-6, 9.91e-5, 1.02e-5, 1.05e-5, 1.08e-5, 1.11e-5, 1.14e-5, 1.28e-5, 1.32e-5, 1.37e-5]
            off_pmos = [2.05e-5, 2.10e-5, 2.15e-5, 2.21e-5, 2.27e-5, 2.34e-5, 2.41e-5, 2.48e-5, 2.56e-5, 2.65e-5, 2.74e-5]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = pmos_vals[i]
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_pmos[i]
        elif _deviceRoadmap == DeviceRoadmap.LOP:
            self.vdd = 0.9
            self.vth = 264.54e-3
            self.phyGateLength = 0.053e-6
            self.capIdealGate = 7.95e-10
            self.capFringe = 2.4e-10
            self.capJunction = 1.00e-3
            self.capOx = 1.50e-2
            self.effectiveElectronMobility = 309.04e-4
            self.effectiveHoleMobility = 67.88e-4
            self.pnSizeRatio = 2.54
            self.effectiveResistanceMultiplier = 1.77
            nmos_vals = [534.5, 525.7, 517.0, 508.5, 500.2, 492.1, 484.1, 476.3, 468.7, 461.2, 453.9]
            pmos_vals = [294.2, 287.8, 281.7, 275.7, 269.9, 264.2, 258.7, 253.4, 248.2, 243.2, 238.3]
            off_nmos = [2.74e-3, 2.6e-3, 2.79e-3, 2.81e-3, 2.84e-3, 2.88e-3, 2.91e-3, 2.95e-3, 2.99e-3, 3.04e-3, 3.09e-3]
            off_pmos = [6.51e-3, 6.56e-3, 6.61e-3, 6.67e-3, 6.74e-3, 6.82e-3, 6.91e-3, 7.00e-3, 7.10e-3, 7.21e-3, 7.33e-3]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = pmos_vals[i]
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_pmos[i]
        elif _deviceRoadmap == DeviceRoadmap.EDRAM:
            self.vdd = 1.2
            self.vpp = 1.6
            self.vth = 454.5e-3
            self.phyGateLength = 0.12e-6
            self.capIdealGate = 1.47e-9
            self.capFringe = 0.08e-9
            self.capJunction = 1e-3
            self.capOx = 1.22e-2
            self.effectiveElectronMobility = 323.95e-4
            self.effectiveHoleMobility = 323.95e-4
            self.pnSizeRatio = 1.95
            self.effectiveResistanceMultiplier = 1.65
            for i in range(0, 101, 10):
                self.currentOnNmos[i] = 321.6
                self.currentOnPmos[i] = 203.3
            off_nmos = [1.42e-5, 2.25e-5, 3.46e-5, 5.18e-5, 7.58e-5, 1.08e-4, 1.51e-4, 2.02e-4, 2.57e-4, 3.14e-4, 3.85e-4]
            off_pmos = [1.42e-5, 2.25e-5, 3.46e-5, 5.18e-5, 7.58e-5, 1.08e-4, 1.51e-4, 2.02e-4, 2.57e-4, 3.14e-4, 3.85e-4]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_pmos[i]
        else:
            print("Unknown device roadmap!")
            exit(1)

    def _init_65nm(self, _deviceRoadmap):
        """Initialize parameters for 65nm technology node."""
        if _deviceRoadmap == DeviceRoadmap.HP:
            self.vdd = 1.1
            self.vth = 163.98e-3
            self.phyGateLength = 0.025e-6
            self.capIdealGate = 4.70e-10
            self.capFringe = 2.4e-10
            self.capJunction = 1.00e-3
            self.capOx = 1.88e-2
            self.effectiveElectronMobility = 445.74e-4
            self.effectiveHoleMobility = 113.330e-4
            self.pnSizeRatio = 2.41
            self.effectiveResistanceMultiplier = 1.50
            nmos_vals = [1211.4, 1198.4, 1185.4, 1172.5, 1156.9, 1146.7, 1133.6, 1119.9, 1104.3, 1084.6, 1059.0]
            pmos_vals = [888.7, 875.8, 861.7, 848.5, 835.4, 822.6, 809.9, 797.3, 784.8, 772.2, 759.4]
            off_nmos = [3.43e-1, 3.73e-1, 4.03e-1, 4.35e-1, 4.66e-1, 4.99e-1, 5.31e-1, 5.64e-1, 5.96e-1, 6.25e-1, 6.51e-1]
            off_pmos = [5.68e-1, 6.07e-1, 6.46e-1, 6.86e-1, 7.26e-1, 7.66e-1, 8.06e-1, 8.46e-1, 8.86e-1, 9.26e-1, 9.65e-1]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = pmos_vals[i]
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_pmos[i]
        elif _deviceRoadmap == DeviceRoadmap.LSTP:
            self.vdd = 1.1
            self.vth = 563.92e-3
            self.phyGateLength = 0.045e-6
            self.capIdealGate = 6.17e-10
            self.capFringe = 2.4e-10
            self.capJunction = 1.00e-3
            self.capOx = 1.37e-2
            self.effectiveElectronMobility = 457.86e-4
            self.effectiveHoleMobility = 102.64e-4
            self.pnSizeRatio = 2.23
            self.effectiveResistanceMultiplier = 1.96
            nmos_vals = [465.4, 458.5, 451.8, 445.1, 438.4, 431.6, 423.9, 414.2, 400.6, 383.5, 367.2]
            pmos_vals = [234.2, 229.7, 225.3, 221.0, 216.8, 212.7, 208.8, 204.8, 200.7, 196.6, 192.6]
            off_nmos = [3.03e-5, 4.46e-5, 6.43e-5, 9.06e-5, 1.25e-4, 1.70e-4, 2.25e-4, 2.90e-4, 3.61e-4, 4.35e-4, 5.20e-4]
            off_pmos = [3.85e-5, 5.64e-5, 8.09e-5, 1.14e-4, 1.57e-4, 2.12e-4, 2.82e-4, 3.70e-4, 4.78e-4, 6.09e-4, 7.66e-4]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = pmos_vals[i]
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_pmos[i]
        elif _deviceRoadmap == DeviceRoadmap.LOP:
            self.vdd = 0.8
            self.vth = 323.75e-3
            self.phyGateLength = 0.032e-6
            self.capIdealGate = 6.01e-10
            self.capFringe = 2.4e-10
            self.capJunction = 1.00e-3
            self.capOx = 1.88e-2
            self.effectiveElectronMobility = 491.59e-4
            self.effectiveHoleMobility = 110.95e-4
            self.pnSizeRatio = 2.28
            self.effectiveResistanceMultiplier = 1.82
            nmos_vals = [562.9, 555.2, 547.5, 539.8, 532.2, 524.5, 516.1, 505.7, 491.1, 471.7, 451.6]
            pmos_vals = [329.5, 323.3, 317.2, 311.2, 305.4, 299.8, 294.2, 288.7, 283.2, 277.5, 271.8]
            off_nmos = [9.08e-3, 1.11e-2, 1.35e-2, 1.62e-2, 1.92e-2, 2.25e-2, 2.62e-2, 2.99e-2, 3.35e-2, 3.67e-2, 3.98e-2]
            off_pmos = [1.30e-2, 1.57e-2, 1.89e-2, 2.24e-2, 2.64e-2, 3.08e-2, 3.56e-2, 1.09e-2, 4.65e-2, 5.26e-2, 5.91e-2]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = pmos_vals[i]
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_pmos[i]
        elif _deviceRoadmap == DeviceRoadmap.EDRAM:
            self.vdd = 1.2
            self.vpp = 1.6
            self.vth = 438.06e-3
            self.phyGateLength = 0.12e-6
            self.capIdealGate = 1.46e-9
            self.capFringe = 0.08e-9
            self.capJunction = 1e-3
            self.capOx = 1.22e-2
            self.effectiveElectronMobility = 328.32e-4
            self.effectiveHoleMobility = 328.32e-4
            self.pnSizeRatio = 2.05
            self.effectiveResistanceMultiplier = 1.65
            for i in range(0, 101, 10):
                self.currentOnNmos[i] = 399.8
                self.currentOnPmos[i] = 243.4
            off_nmos = [2.23e-5, 3.46e-5, 5.24e-5, 7.75e-5, 1.12e-4, 1.58e-4, 2.18e-4, 2.88e-4, 3.63e-4, 4.41e-4, 5.36e-4]
            off_pmos = [2.23e-5, 3.46e-5, 5.24e-5, 7.75e-5, 1.12e-4, 1.58e-4, 2.18e-4, 2.88e-4, 3.63e-4, 4.41e-4, 5.36e-4]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_pmos[i]
        else:
            print("Unknown device roadmap!")
            exit(1)

    def _init_45nm(self, _deviceRoadmap):
        """Initialize parameters for 45nm technology node."""
        if _deviceRoadmap == DeviceRoadmap.HP:
            self.vdd = 1.0
            self.vth = 126.79e-3
            self.phyGateLength = 0.018e-6
            self.capIdealGate = 6.78e-10
            self.capFringe = 1.7e-10
            self.capJunction = 1.00e-3
            self.capOx = 3.77e-2
            self.effectiveElectronMobility = 297.70e-4
            self.effectiveHoleMobility = 95.27e-4
            self.pnSizeRatio = 2.41
            self.effectiveResistanceMultiplier = 1.51
            nmos_vals = [1823.8, 1808.2, 1792.6, 1777.0, 1761.4, 1745.8, 1730.3, 1714.7, 1699.1, 1683.2, 1666.6]
            pmos_vals = [1632.2, 1612.8, 1593.6, 1574.1, 1554.7, 1535.5, 1516.4, 1497.6, 1478.8, 1460.3, 1441.8]
            off_nmos = [2.80e-1, 3.28e-1, 3.81e-1, 4.39e-1, 5.02e-1, 5.69e-1, 6.42e-1, 7.20e-1, 8.03e-1, 8.91e-1, 9.84e-1]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = pmos_vals[i]
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_nmos[i]  # PMOS = NMOS
        elif _deviceRoadmap == DeviceRoadmap.LSTP:
            self.vdd = 1.0
            self.vth = 564.52e-3
            self.phyGateLength = 0.028e-6
            self.capIdealGate = 5.58e-10
            self.capFringe = 2.1e-10
            self.capJunction = 1.00e-3
            self.capOx = 1.99e-2
            self.effectiveElectronMobility = 456.14e-4
            self.effectiveHoleMobility = 96.98e-4
            self.pnSizeRatio = 2.23
            self.effectiveResistanceMultiplier = 1.99
            nmos_vals = [527.5, 520.2, 512.9, 505.8, 498.6, 491.4, 483.7, 474.4, 461.2, 442.6, 421.3]
            pmos_vals = [497.9, 489.5, 481.3, 473.2, 465.3, 457.6, 450.0, 442.5, 435.1, 427.5, 419.7]
            off_nmos = [1.01e-5, 1.65e-5, 2.62e-5, 4.06e-5, 6.12e-5, 9.02e-5, 1.30e-4, 1.83e-4, 2.51e-4, 3.29e-4, 4.10e-4]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = pmos_vals[i]
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_nmos[i]  # PMOS = NMOS
        elif _deviceRoadmap == DeviceRoadmap.LOP:
            self.vdd = 0.7
            self.vth = 288.94e-3
            self.phyGateLength = 0.022e-6
            self.capIdealGate = 6.13e-10
            self.capFringe = 2.0e-10
            self.capJunction = 1.00e-3
            self.capOx = 2.79e-2
            self.effectiveElectronMobility = 606.95e-4
            self.effectiveHoleMobility = 124.60e-4
            self.pnSizeRatio = 2.28
            self.effectiveResistanceMultiplier = 1.76
            nmos_vals = [682.1, 672.3, 662.5, 652.8, 643.0, 632.8, 620.9, 605.0, 583.6, 561.0, 542.7]
            pmos_vals = [772.4, 759.6, 746.9, 734.4, 722.1, 710.0, 698.1, 686.3, 674.4, 662.3, 650.2]
            off_nmos = [4.03e-3, 5.02e-3, 6.18e-3, 7.51e-3, 9.04e-3, 1.08e-2, 1.27e-2, 1.47e-2, 1.66e-2, 1.84e-2, 2.03e-2]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = pmos_vals[i]
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_nmos[i]  # PMOS = NMOS
        elif _deviceRoadmap == DeviceRoadmap.EDRAM:
            self.vdd = 1.1
            self.vpp = 1.5
            self.vth = 445.59e-3
            self.phyGateLength = 0.078e-6
            self.capIdealGate = 1.10e-9
            self.capFringe = 0.08e-9
            self.capJunction = 1e-3
            self.capOx = 1.41e-2
            self.effectiveElectronMobility = 426.30e-4
            self.effectiveHoleMobility = 426.30e-4
            self.pnSizeRatio = 2.05
            self.effectiveResistanceMultiplier = 1.65
            for i in range(0, 101, 10):
                self.currentOnNmos[i] = 456
                self.currentOnPmos[i] = 228
            off_nmos = [2.54e-5, 3.94e-5, 5.95e-5, 8.79e-5, 1.27e-4, 1.79e-4, 2.47e-4, 3.31e-4, 4.26e-4, 5.27e-4, 6.46e-4]
            off_pmos = [2.54e-5, 3.94e-5, 5.95e-5, 8.79e-5, 1.27e-4, 1.79e-4, 2.47e-4, 3.31e-4, 4.26e-4, 5.27e-4, 6.46e-4]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_pmos[i]
        else:
            print("Unknown device roadmap!")
            exit(1)

    def _init_32nm(self, _deviceRoadmap):
        """Initialize parameters for 32nm (36nm) technology node."""
        if _deviceRoadmap == DeviceRoadmap.HP:
            self.vdd = 0.9
            self.vth = 131.72e-3
            self.phyGateLength = 0.014e-6
            self.capIdealGate = 6.42e-10
            self.capFringe = 1.6e-10
            self.capJunction = 1.00e-3
            self.capOx = 4.59e-2
            self.effectiveElectronMobility = 257.73e-4
            self.effectiveHoleMobility = 89.92e-4
            self.pnSizeRatio = 2.41
            self.effectiveResistanceMultiplier = 1.49
            nmos_vals = [1785.8, 1771.8, 1757.8, 1743.8, 1729.8, 1715.7, 1701.7, 1687.6, 1673.5, 1659.4, 1645.0]
            pmos_vals = [1713.5, 1662.8, 1620.1, 1601.6, 1583.3, 1565.1, 1547.1, 1529.1, 1511.3, 1493.7, 1476.1]
            off_nmos = [8.34e-1, 9.00e-1, 9.68e-1, 1.04, 1.11, 1.18, 1.25, 1.32, 1.39, 1.46, 1.54]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = pmos_vals[i]
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_nmos[i]  # PMOS = NMOS
        elif _deviceRoadmap == DeviceRoadmap.LSTP:
            self.vdd = 1.0
            self.vth = 581.81e-3
            self.phyGateLength = 0.022e-6
            self.capIdealGate = 5.02e-10
            self.capFringe = 1.9e-10
            self.capJunction = 1.00e-3
            self.capOx = 2.19e-2
            self.effectiveElectronMobility = 395.20e-4
            self.effectiveHoleMobility = 88.67e-4
            self.pnSizeRatio = 2.23
            self.effectiveResistanceMultiplier = 1.99
            nmos_vals = [560.0, 553.0, 546.1, 539.3, 532.5, 525.8, 518.9, 511.5, 502.3, 489.2, 469.7]
            pmos_vals = [549.6, 541.1, 532.8, 524.6, 516.5, 508.7, 500.9, 493.3, 485.8, 478.3, 470.7]
            off_nmos = [3.02e-5, 4.51e-5, 6.57e-5, 9.35e-5, 1.31e-4, 1.79e-4, 2.41e-4, 3.19e-4, 4.15e-4, 5.29e-4, 6.58e-4]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = pmos_vals[i]
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_nmos[i]  # PMOS = NMOS
        elif _deviceRoadmap == DeviceRoadmap.LOP:
            self.vdd = 0.7
            self.vth = 278.52e-3
            self.phyGateLength = 0.018e-6
            self.capIdealGate = 5.54e-10
            self.capFringe = 2.0e-10
            self.capJunction = 1.00e-3
            self.capOx = 3.08e-2
            self.effectiveElectronMobility = 581.62e-4
            self.effectiveHoleMobility = 120.30e-4
            self.pnSizeRatio = 2.28
            self.effectiveResistanceMultiplier = 1.73
            nmos_vals = [760.3, 750.4, 740.5, 730.7, 720.8, 710.9, 700.3, 687.6, 670.5, 647.4, 623.6]
            pmos_vals = [878.6, 865.1, 851.8, 838.7, 825.7, 813.0, 800.3, 787.9, 775.5, 763.0, 750.3]
            off_nmos = [3.57e-2, 4.21e-2, 4.91e-2, 5.68e-2, 6.51e-2, 7.42e-2, 8.43e-2, 9.57e-2, 1.10e-1, 1.28e-1, 1.48e-1]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = pmos_vals[i]
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_nmos[i]  # PMOS = NMOS
        elif _deviceRoadmap == DeviceRoadmap.EDRAM:
            self.vdd = 1.0
            self.vpp = 1.5
            self.vth = 441.29e-3
            self.phyGateLength = 0.056e-6
            self.capIdealGate = 7.45e-10
            self.capFringe = 0.053e-9
            self.capJunction = 1e-3
            self.capOx = 1.48e-2
            self.effectiveElectronMobility = 408.12e-4
            self.effectiveHoleMobility = 408.12e-4
            self.pnSizeRatio = 2.05
            self.effectiveResistanceMultiplier = 1.65
            for i in range(0, 101, 10):
                self.currentOnNmos[i] = 1055.4
                self.currentOnPmos[i] = 527.7
            off_nmos = [3.57e-5, 5.51e-5, 8.27e-5, 1.21e-4, 1.74e-4, 2.45e-4, 3.38e-4, 4.53e-4, 5.87e-4, 7.29e-4, 8.87e-4]
            off_pmos = [3.57e-5, 5.51e-5, 8.27e-5, 1.21e-4, 1.74e-4, 2.45e-4, 3.38e-4, 4.53e-4, 5.87e-4, 7.29e-4, 8.87e-4]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_pmos[i]
        else:
            print("Unknown device roadmap!")
            exit(1)

    def _init_22nm(self, _deviceRoadmap):
        """Initialize parameters for 22nm (25nm) technology node."""
        if _deviceRoadmap == DeviceRoadmap.HP:
            self.vdd = 0.9
            self.vth = 128.72e-3
            self.phyGateLength = 0.010e-6
            self.capIdealGate = 3.83e-10
            self.capFringe = 1.6e-10
            self.capJunction = 0
            self.capOx = 3.83e-2
            self.effectiveElectronMobility = 397.26e-4
            self.effectiveHoleMobility = 83.60e-4
            self.pnSizeRatio = 2.0
            self.effectiveResistanceMultiplier = 1.45
            nmos_vals = [2029.9, 2009.8, 1989.6, 1969.6, 1949.8, 1930.7, 1910.5, 1891.0, 1871.7, 1852.5, 1834.4]
            off_nmos = [1.52e-7*3.93e6, 1.55e-7*3.93e6, 1.59e-7*3.93e6, 1.68e-7*3.93e6, 1.90e-7*3.93e6,
                       2.69e-7*3.93e6, 5.32e-7*3.93e6, 1.02e-6*3.93e6, 1.62e-6*3.93e6, 2.73e-6*3.93e6, 6.1e-6*3.93e6]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = nmos_vals[i] / 2  # PMOS = NMOS / 2
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_nmos[i]  # PMOS = NMOS
        elif _deviceRoadmap == DeviceRoadmap.LSTP:
            self.vdd = 0.8
            self.vth = 445.71e-3
            self.phyGateLength = 0.016e-6
            self.capIdealGate = 4.25e-10
            self.capFringe = 2e-10
            self.capJunction = 0
            self.capOx = 2.65e-2
            self.effectiveElectronMobility = 731.29e-4
            self.effectiveHoleMobility = 111.22e-4
            self.pnSizeRatio = 2.23
            self.effectiveResistanceMultiplier = 1.99
            nmos_vals = [745.5, 735.2, 725.1, 715.2, 705.4, 695.7, 686.2, 676.9, 667.7, 658.7, 649.8]
            off_nmos = [3.02e-5/1.86, 4.51e-5/1.86, 6.57e-5/1.86, 9.35e-5/1.86, 1.31e-4/1.86,
                       1.79e-4/1.86, 2.41e-4/1.86, 3.19e-4/1.86, 4.15e-4/1.86, 5.29e-4/1.86, 6.58e-4/1.86]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = nmos_vals[i] / 2  # PMOS = NMOS / 2
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_nmos[i]  # PMOS = NMOS
        elif _deviceRoadmap == DeviceRoadmap.LOP:
            self.vdd = 0.5
            self.vth = 217.39e-3
            self.phyGateLength = 0.011e-6
            self.capIdealGate = 3.45e-10
            self.capFringe = 1.7e-10
            self.capJunction = 0
            self.capOx = 3.14e-2
            self.effectiveElectronMobility = 747.37e-4
            self.effectiveHoleMobility = 118.35e-4
            self.pnSizeRatio = 2.28
            self.effectiveResistanceMultiplier = 1.73
            nmos_vals = [716.1, 704.3, 692.6, 681.2, 669.9, 658.8, 647.9, 637.1, 626.5, 616.0, 605.7]
            off_nmos = [3.57e-2/1.7, 4.21e-2/1.7, 4.91e-2/1.7, 5.68e-2/1.7, 6.51e-2/1.7,
                       7.42e-2/1.7, 8.43e-2/1.7, 9.57e-2/1.7, 1.10e-1/1.7, 1.28e-1/1.7, 1.48e-1/1.7]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOnNmos[idx] = nmos_vals[i]
                self.currentOnPmos[idx] = nmos_vals[i] / 2  # PMOS = NMOS / 2
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_nmos[i]  # PMOS = NMOS
        elif _deviceRoadmap == DeviceRoadmap.EDRAM:
            self.vdd = 0.9
            self.vpp = 1.4
            self.vth = 436.835e-3
            self.phyGateLength = 0.010e-6
            self.capIdealGate = 5.22e-10
            self.capFringe = 0.008e-9
            self.capJunction = 1e-3
            self.capOx = 1.58e-2
            self.effectiveElectronMobility = 459.295e-4
            self.effectiveHoleMobility = 459.295e-4
            self.pnSizeRatio = 2.10
            self.effectiveResistanceMultiplier = 1.65
            for i in range(0, 101, 10):
                self.currentOnNmos[i] = 1122.6
                self.currentOnPmos[i] = 540.05
            off_nmos = [4.66e-5, 6.36e-5, 9.52e-5, 1.39e-4, 2.00e-4, 2.81e-4, 3.86e-4, 5.18e-4, 6.72e-4, 8.36e-4, 1.02e-3]
            off_pmos = [4.66e-5, 6.36e-5, 9.52e-5, 1.39e-4, 2.00e-4, 2.81e-4, 3.86e-4, 5.18e-4, 6.72e-4, 8.36e-4, 1.03e-3]
            for i, idx in enumerate(range(0, 101, 10)):
                self.currentOffNmos[idx] = off_nmos[i]
                self.currentOffPmos[idx] = off_pmos[i]
        else:
            print("Unknown device roadmap!")
            exit(1)

    def _init_tsv_params(self, _featureSizeInNano):
        """Initialize TSV parameters based on feature size."""
        if _featureSizeInNano >= 180:
            # 180nm extrapolated from 22,32,45,65, and 90nm values
            # TSV aggressive, projected from ITRS
            self.tsv_pitch[0][0] = 5.6
            self.tsv_diameter[0][0] = 2.8
            self.tsv_length[0][0] = 10.0
            self.tsv_dielec_thickness[0][0] = 0.1
            self.tsv_contact_resistance[0][0] = 0.1
            self.tsv_depletion_width[0][0] = 0.6
            self.tsv_liner_dielectric_constant[0][0] = 3.3012
            # TSV conservative, projected from ITRS
            self.tsv_pitch[1][0] = 8.88
            self.tsv_diameter[1][0] = 4.46
            self.tsv_length[1][0] = 40.0
            self.tsv_dielec_thickness[1][0] = 0.1
            self.tsv_contact_resistance[1][0] = 0.1
            self.tsv_depletion_width[1][0] = 0.6
            self.tsv_liner_dielectric_constant[1][0] = 3.4652
            # TSV aggressive, industry reported
            self.tsv_pitch[0][1] = 58.6
            self.tsv_diameter[0][1] = 7.12
            self.tsv_length[0][1] = 77.0
            self.tsv_dielec_thickness[0][1] = 0.26
            self.tsv_contact_resistance[0][1] = 0.26
            self.tsv_depletion_width[0][1] = 0.6
            self.tsv_liner_dielectric_constant[0][1] = 3.3012
            # TSV conservative, industry reported
            self.tsv_pitch[1][1] = 104
            self.tsv_diameter[1][1] = 11.54
            self.tsv_length[1][1] = 85.0
            self.tsv_dielec_thickness[1][1] = 0.68
            self.tsv_contact_resistance[1][1] = 0.2
            self.tsv_depletion_width[1][1] = 0.6
            self.tsv_liner_dielectric_constant[1][1] = 3.4652
        elif _featureSizeInNano >= 120:
            # 130nm extrapolated
            self.tsv_pitch[0][0] = 4.78
            self.tsv_diameter[0][0] = 2.39
            self.tsv_length[0][0] = 9.0
            self.tsv_dielec_thickness[0][0] = 0.1
            self.tsv_contact_resistance[0][0] = 0.1
            self.tsv_depletion_width[0][0] = 0.6
            self.tsv_liner_dielectric_constant[0][0] = 2.9783
            self.tsv_pitch[1][0] = 7.7
            self.tsv_diameter[1][0] = 3.87
            self.tsv_length[1][0] = 35.0
            self.tsv_dielec_thickness[1][0] = 0.1
            self.tsv_contact_resistance[1][0] = 0.1
            self.tsv_depletion_width[1][0] = 0.6
            self.tsv_liner_dielectric_constant[1][0] = 3.2264
            self.tsv_pitch[0][1] = 49.9
            self.tsv_diameter[0][1] = 6.41
            self.tsv_length[0][1] = 68.0
            self.tsv_dielec_thickness[0][1] = 0.24
            self.tsv_contact_resistance[0][1] = 0.24
            self.tsv_depletion_width[0][1] = 0.6
            self.tsv_liner_dielectric_constant[0][1] = 2.9783
            self.tsv_pitch[1][1] = 91
            self.tsv_diameter[1][1] = 11.41
            self.tsv_length[1][1] = 77.5
            self.tsv_dielec_thickness[1][1] = 0.62
            self.tsv_contact_resistance[1][1] = 0.2
            self.tsv_depletion_width[1][1] = 0.6
            self.tsv_liner_dielectric_constant[1][1] = 3.2264
        elif _featureSizeInNano >= 90:
            self.tsv_pitch[0][0] = 4.0
            self.tsv_diameter[0][0] = 2.0
            self.tsv_length[0][0] = 8.0
            self.tsv_dielec_thickness[0][0] = 0.1
            self.tsv_contact_resistance[0][0] = 0.1
            self.tsv_depletion_width[0][0] = 0.6
            self.tsv_liner_dielectric_constant[0][0] = 2.709
            self.tsv_pitch[1][0] = 6.9
            self.tsv_diameter[1][0] = 3.5
            self.tsv_length[1][0] = 30.0
            self.tsv_dielec_thickness[1][0] = 0.1
            self.tsv_contact_resistance[1][0] = 0.1
            self.tsv_depletion_width[1][0] = 0.6
            self.tsv_liner_dielectric_constant[1][0] = 3.038
            self.tsv_pitch[0][1] = 45
            self.tsv_diameter[0][1] = 6.9
            self.tsv_length[0][1] = 60.0
            self.tsv_dielec_thickness[0][1] = 0.2
            self.tsv_contact_resistance[0][1] = 0.2
            self.tsv_depletion_width[0][1] = 0.6
            self.tsv_liner_dielectric_constant[0][1] = 2.709
            self.tsv_pitch[1][1] = 90
            self.tsv_diameter[1][1] = 11.3
            self.tsv_length[1][1] = 75.0
            self.tsv_dielec_thickness[1][1] = 0.5
            self.tsv_contact_resistance[1][1] = 0.2
            self.tsv_depletion_width[1][1] = 0.6
            self.tsv_liner_dielectric_constant[1][1] = 3.038
        elif _featureSizeInNano >= 65:
            self.tsv_pitch[0][0] = 3.2
            self.tsv_diameter[0][0] = 1.6
            self.tsv_length[0][0] = 7.0
            self.tsv_dielec_thickness[0][0] = 0.1
            self.tsv_contact_resistance[0][0] = 0.1
            self.tsv_depletion_width[0][0] = 0.6
            self.tsv_liner_dielectric_constant[0][0] = 2.303
            self.tsv_pitch[1][0] = 5
            self.tsv_diameter[1][0] = 2.5
            self.tsv_length[1][0] = 25.0
            self.tsv_dielec_thickness[1][0] = 0.1
            self.tsv_contact_resistance[1][0] = 0.1
            self.tsv_depletion_width[1][0] = 0.6
            self.tsv_liner_dielectric_constant[1][0] = 2.734
            self.tsv_pitch[0][1] = 30
            self.tsv_diameter[0][1] = 4.6
            self.tsv_length[0][1] = 50.0
            self.tsv_dielec_thickness[0][1] = 0.2
            self.tsv_contact_resistance[0][1] = 0.2
            self.tsv_depletion_width[0][1] = 0.6
            self.tsv_liner_dielectric_constant[0][1] = 2.303
            self.tsv_pitch[1][1] = 60
            self.tsv_diameter[1][1] = 7.5
            self.tsv_length[1][1] = 62.5
            self.tsv_dielec_thickness[1][1] = 0.5
            self.tsv_contact_resistance[1][1] = 0.2
            self.tsv_depletion_width[1][1] = 0.6
            self.tsv_liner_dielectric_constant[1][1] = 2.734
        elif _featureSizeInNano >= 45:
            self.tsv_pitch[0][0] = 2.2
            self.tsv_diameter[0][0] = 1.1
            self.tsv_length[0][0] = 6.0
            self.tsv_dielec_thickness[0][0] = 0.1
            self.tsv_contact_resistance[0][0] = 0.1
            self.tsv_depletion_width[0][0] = 0.6
            self.tsv_liner_dielectric_constant[0][0] = 1.958
            self.tsv_pitch[1][0] = 3.4
            self.tsv_diameter[1][0] = 1.7
            self.tsv_length[1][0] = 20.0
            self.tsv_dielec_thickness[1][0] = 0.1
            self.tsv_contact_resistance[1][0] = 0.1
            self.tsv_depletion_width[1][0] = 0.6
            self.tsv_liner_dielectric_constant[1][0] = 2.460
            self.tsv_pitch[0][1] = 20
            self.tsv_diameter[0][1] = 3.1
            self.tsv_length[0][1] = 40.0
            self.tsv_dielec_thickness[0][1] = 0.2
            self.tsv_contact_resistance[0][1] = 0.2
            self.tsv_depletion_width[0][1] = 0.6
            self.tsv_liner_dielectric_constant[0][1] = 1.958
            self.tsv_pitch[1][1] = 40
            self.tsv_diameter[1][1] = 5
            self.tsv_length[1][1] = 50.0
            self.tsv_dielec_thickness[1][1] = 0.5
            self.tsv_contact_resistance[1][1] = 0.2
            self.tsv_depletion_width[1][1] = 0.6
            self.tsv_liner_dielectric_constant[1][1] = 2.460
        elif _featureSizeInNano >= 32:
            self.tsv_pitch[0][0] = 1.4
            self.tsv_diameter[0][0] = 0.7
            self.tsv_length[0][0] = 5.0
            self.tsv_dielec_thickness[0][0] = 0.1
            self.tsv_contact_resistance[0][0] = 0.1
            self.tsv_depletion_width[0][0] = 0.6
            self.tsv_liner_dielectric_constant[0][0] = 1.664
            self.tsv_pitch[1][0] = 4
            self.tsv_diameter[1][0] = 2
            self.tsv_length[1][0] = 15.0
            self.tsv_dielec_thickness[1][0] = 0.1
            self.tsv_contact_resistance[1][0] = 0.1
            self.tsv_depletion_width[1][0] = 0.6
            self.tsv_liner_dielectric_constant[1][0] = 2.214
            self.tsv_pitch[0][1] = 15
            self.tsv_diameter[0][1] = 2.3
            self.tsv_length[0][1] = 30.0
            self.tsv_dielec_thickness[0][1] = 0.2
            self.tsv_contact_resistance[0][1] = 0.2
            self.tsv_depletion_width[0][1] = 0.6
            self.tsv_liner_dielectric_constant[0][1] = 1.664
            self.tsv_pitch[1][1] = 30
            self.tsv_diameter[1][1] = 3.8
            self.tsv_length[1][1] = 37.5
            self.tsv_dielec_thickness[1][1] = 0.5
            self.tsv_contact_resistance[1][1] = 0.2
            self.tsv_depletion_width[1][1] = 0.6
            self.tsv_liner_dielectric_constant[1][1] = 2.214
        elif _featureSizeInNano >= 22:
            self.tsv_pitch[0][0] = 0.8
            self.tsv_diameter[0][0] = 0.4
            self.tsv_length[0][0] = 4.0
            self.tsv_dielec_thickness[0][0] = 0.1
            self.tsv_contact_resistance[0][0] = 0.1
            self.tsv_depletion_width[0][0] = 0.6
            self.tsv_liner_dielectric_constant[0][0] = 1.414
            self.tsv_pitch[1][0] = 1.5
            self.tsv_diameter[1][0] = 0.8
            self.tsv_length[1][0] = 10.0
            self.tsv_dielec_thickness[1][0] = 0.1
            self.tsv_contact_resistance[1][0] = 0.1
            self.tsv_depletion_width[1][0] = 0.6
            self.tsv_liner_dielectric_constant[1][0] = 2.104
            self.tsv_pitch[0][1] = 9
            self.tsv_diameter[0][1] = 4.5
            self.tsv_length[0][1] = 25.0
            self.tsv_dielec_thickness[0][1] = 0.1
            self.tsv_contact_resistance[0][1] = 0.1
            self.tsv_depletion_width[0][1] = 0.6
            self.tsv_liner_dielectric_constant[0][1] = 1.414
            self.tsv_pitch[1][1] = 40
            self.tsv_diameter[1][1] = 7.5
            self.tsv_length[1][1] = 50.0
            self.tsv_dielec_thickness[1][1] = 0.2
            self.tsv_contact_resistance[1][1] = 0.2
            self.tsv_depletion_width[1][1] = 0.6
            self.tsv_liner_dielectric_constant[1][1] = 2.104

    def _interpolate_currents(self):
        """Interpolate current values for intermediate temperature indices."""
        for i in range(1, 100):
            if i % 10:
                idx_low = (i // 10) * 10
                idx_high = idx_low + 10
                a_nmos = self.currentOnNmos[idx_low]
                b_nmos = self.currentOnNmos[idx_high]
                self.currentOnNmos[i] = a_nmos + (b_nmos - a_nmos) * (i % 10) / 10

                a_pmos = self.currentOnPmos[idx_low]
                b_pmos = self.currentOnPmos[idx_high]
                self.currentOnPmos[i] = a_pmos + (b_pmos - a_pmos) * (i % 10) / 10

                a_off_nmos = self.currentOffNmos[idx_low]
                b_off_nmos = self.currentOffNmos[idx_high]
                self.currentOffNmos[i] = a_off_nmos + (b_off_nmos - a_off_nmos) * (i % 10) / 10

                a_off_pmos = self.currentOffPmos[idx_low]
                b_off_pmos = self.currentOffPmos[idx_high]
                self.currentOffPmos[i] = a_off_pmos + (b_off_pmos - a_off_pmos) * (i % 10) / 10

    def PrintProperty(self):
        """Print technology properties."""
        print("Fabrication Process Technology Node:")
        print(f"Feature Size: {self.featureSizeInNano} nm")
        print(f"Device Roadmap: {self.deviceRoadmap}")
        print(f"Vdd: {self.vdd} V")
        print(f"Vth: {self.vth} V")
        print(f"Physical Gate Length: {self.phyGateLength*1e9} nm")

    def InterpolateWith(self, rhs, _alpha):
        """
        Interpolate technology parameters with another Technology object.

        Args:
            rhs: Another Technology object to interpolate with
            _alpha: Interpolation coefficient (0 to 1)
        """
        if self.featureSizeInNano != rhs.featureSizeInNano:
            self.vdd = (1 - _alpha) * self.vdd + _alpha * rhs.vdd
            self.vth = (1 - _alpha) * self.vth + _alpha * rhs.vth
            self.phyGateLength = (1 - _alpha) * self.phyGateLength + _alpha * rhs.phyGateLength
            self.capIdealGate = (1 - _alpha) * self.capIdealGate + _alpha * rhs.capIdealGate
            self.capFringe = (1 - _alpha) * self.capFringe + _alpha * rhs.capFringe
            self.capJunction = (1 - _alpha) * self.capJunction + _alpha * rhs.capJunction
            self.capOx = (1 - _alpha) * self.capOx + _alpha * rhs.capOx
            self.effectiveElectronMobility = (1 - _alpha) * self.effectiveElectronMobility + _alpha * rhs.effectiveElectronMobility
            self.effectiveHoleMobility = (1 - _alpha) * self.effectiveHoleMobility + _alpha * rhs.effectiveHoleMobility
            self.pnSizeRatio = (1 - _alpha) * self.pnSizeRatio + _alpha * rhs.pnSizeRatio
            self.effectiveResistanceMultiplier = (1 - _alpha) * self.effectiveResistanceMultiplier + _alpha * rhs.effectiveResistanceMultiplier

            for i in range(101):
                self.currentOnNmos[i] = (1 - _alpha) * self.currentOnNmos[i] + _alpha * rhs.currentOnNmos[i]
                self.currentOnPmos[i] = (1 - _alpha) * self.currentOnPmos[i] + _alpha * rhs.currentOnPmos[i]
                self.currentOffNmos[i] = pow(self.currentOffNmos[i], 1 - _alpha) * pow(rhs.currentOffNmos[i], _alpha)
                self.currentOffPmos[i] = pow(self.currentOffPmos[i], 1 - _alpha) * pow(rhs.currentOffPmos[i], _alpha)

            # Recalculate derived values
            cjd = 1e-3
            cjswd = 2.5e-10
            cjswgd = 0.5e-10
            mjd = 0.5
            mjswd = 0.33
            mjswgd = 0.33
            self.buildInPotential = 0.9
            self.capJunction = cjd / pow(1 + self.vdd / self.buildInPotential, mjd)
            self.capSidewall = cjswd / pow(1 + self.vdd / self.buildInPotential, mjswd)
            self.capDrainToChannel = cjswgd / pow(1 + self.vdd / self.buildInPotential, mjswgd)

            self.vdsatNmos = self.phyGateLength * 1e5 / self.effectiveElectronMobility
            self.vdsatPmos = self.phyGateLength * 1e5 / self.effectiveHoleMobility

    def tsv_resistance(self, resistivity, tsv_len, tsv_diam, tsv_contact_resistance):
        """
        Calculate TSV resistance.

        Args:
            resistivity: Material resistivity (ohm-micron)
            tsv_len: TSV length (micron)
            tsv_diam: TSV diameter (micron)
            tsv_contact_resistance: Contact resistance (ohm)

        Returns:
            TSV resistance in ohms
        """
        resistance = resistivity * tsv_len / (3.1416 * (tsv_diam/2) * (tsv_diam/2)) + tsv_contact_resistance
        return resistance

    def tsv_capacitance(self, tsv_len, tsv_diam, tsv_pitch, dielec_thickness,
                       liner_dielectric_constant, depletion_width):
        """
        Calculate TSV capacitance.

        Args:
            tsv_len: TSV length (micron)
            tsv_diam: TSV diameter (micron)
            tsv_pitch: TSV pitch (micron)
            dielec_thickness: Dielectric thickness (micron)
            liner_dielectric_constant: Liner dielectric constant
            depletion_width: Depletion width (micron)

        Returns:
            TSV capacitance in Farads
        """
        e_si = PERMITTIVITY_FREE_SPACE * 11.9
        PI = 3.1416
        lateral_coupling_constant = 4.1
        diagonal_coupling_constant = 5.3

        liner_cap = 2 * PI * PERMITTIVITY_FREE_SPACE * liner_dielectric_constant * tsv_len / math.log(1 + dielec_thickness / (tsv_diam/2))
        depletion_cap = 2 * PI * e_si * tsv_len / math.log(1 + depletion_width / (dielec_thickness + tsv_diam/2))
        self_cap = 1 / (1/liner_cap + 1/depletion_cap)

        lateral_coupling_cap = 0.4 * (0.225 * math.log(0.97 * tsv_len / tsv_diam) + 0.53) * e_si / (tsv_pitch - tsv_diam) * PI * tsv_diam * tsv_len
        diagonal_coupling_cap = 0.4 * (0.225 * math.log(0.97 * tsv_len / tsv_diam) + 0.53) * e_si / (1.414 * tsv_pitch - tsv_diam) * PI * tsv_diam * tsv_len
        total_cap = self_cap + lateral_coupling_constant * lateral_coupling_cap + diagonal_coupling_constant * diagonal_coupling_cap

        return total_cap

    def tsv_area(self, tsv_pitch):
        """
        Calculate TSV area.

        Args:
            tsv_pitch: TSV pitch (micron)

        Returns:
            TSV area in square microns
        """
        return pow(tsv_pitch, 2)

    def WireTypeToTSVType(self, wiretype):
        """
        Convert wire type to TSV type.

        Args:
            wiretype: Wire type enum value

        Returns:
            TSV_type enum (Fine or Coarse)
        """
        rv = TSV_type.Fine

        if wiretype in [WireType.local_aggressive, WireType.semi_aggressive, WireType.global_aggressive]:
            rv = TSV_type.Fine
        elif wiretype in [WireType.local_conservative, WireType.semi_conservative,
                         WireType.global_conservative, WireType.dram_wordline]:
            rv = TSV_type.Coarse
        else:
            rv = TSV_type.Fine

        return rv

    def SetLayerCount(self, inputParameter, layers):
        """
        Recalculate TSV parameters based on layer count.

        Args:
            inputParameter: InputParameter object containing wire type configuration
            layers: Number of 3D layers
        """
        if layers == self.layerCount:
            return

        # TSV aggressive, projected from ITRS
        length_value = self.tsv_length[0][0] * layers
        self.tsv_parasitic_res[0][0] = self.tsv_resistance(BULK_CU_RESISTIVITY, length_value,
                                                           self.tsv_diameter[0][0],
                                                           self.tsv_contact_resistance[0][0])
        self.tsv_parasitic_cap[0][0] = self.tsv_capacitance(length_value, self.tsv_diameter[0][0],
                                                            self.tsv_pitch[0][0],
                                                            self.tsv_dielec_thickness[0][0],
                                                            self.tsv_liner_dielectric_constant[0][0],
                                                            self.tsv_depletion_width[0][0])
        self.tsv_occupation_area[0][0] = self.tsv_area(self.tsv_pitch[0][0])

        # TSV conservative, projected from ITRS
        length_value = self.tsv_length[1][0] * layers
        self.tsv_parasitic_res[1][0] = self.tsv_resistance(BULK_CU_RESISTIVITY, length_value,
                                                           self.tsv_diameter[1][0],
                                                           self.tsv_contact_resistance[1][0])
        self.tsv_parasitic_cap[1][0] = self.tsv_capacitance(length_value, self.tsv_diameter[1][0],
                                                            self.tsv_pitch[1][0],
                                                            self.tsv_dielec_thickness[1][0],
                                                            self.tsv_liner_dielectric_constant[1][0],
                                                            self.tsv_depletion_width[1][0])
        self.tsv_occupation_area[1][0] = self.tsv_area(self.tsv_pitch[1][0])

        # TSV aggressive, industry reported
        length_value = self.tsv_length[0][1] * layers
        self.tsv_parasitic_res[0][1] = self.tsv_resistance(BULK_CU_RESISTIVITY, length_value,
                                                           self.tsv_diameter[0][1],
                                                           self.tsv_contact_resistance[0][1])
        self.tsv_parasitic_cap[0][1] = self.tsv_capacitance(length_value, self.tsv_diameter[0][1],
                                                            self.tsv_pitch[0][1],
                                                            self.tsv_dielec_thickness[0][1],
                                                            self.tsv_liner_dielectric_constant[0][1],
                                                            self.tsv_depletion_width[0][1])
        self.tsv_occupation_area[0][1] = self.tsv_area(self.tsv_pitch[0][1])

        # TSV conservative, industry reported
        length_value = self.tsv_length[1][1] * layers
        self.tsv_parasitic_res[1][1] = self.tsv_resistance(BULK_CU_RESISTIVITY, length_value,
                                                           self.tsv_diameter[1][1],
                                                           self.tsv_contact_resistance[1][1])
        self.tsv_parasitic_cap[1][1] = self.tsv_capacitance(length_value, self.tsv_diameter[1][1],
                                                            self.tsv_pitch[1][1],
                                                            self.tsv_dielec_thickness[1][1],
                                                            self.tsv_liner_dielectric_constant[1][1],
                                                            self.tsv_depletion_width[1][1])
        self.tsv_occupation_area[1][1] = self.tsv_area(self.tsv_pitch[1][1])

        # Finalize TSV parameters in tech pointer
        local_ic_proj_type = self.WireTypeToTSVType(inputParameter.maxLocalWireType)
        global_ic_proj_type = self.WireTypeToTSVType(inputParameter.maxGlobalWireType)
        tsv_is_subarray_type = inputParameter.localTsvProjection
        tsv_os_bank_type = inputParameter.globalTsvProjection

        self.resTSV[TSV_type.Fine] = self.tsv_parasitic_res[local_ic_proj_type][tsv_is_subarray_type]
        self.capTSV[TSV_type.Fine] = self.tsv_parasitic_cap[local_ic_proj_type][tsv_is_subarray_type]
        self.areaTSV[TSV_type.Fine] = self.tsv_occupation_area[local_ic_proj_type][tsv_is_subarray_type]

        self.resTSV[TSV_type.Coarse] = self.tsv_parasitic_res[global_ic_proj_type][tsv_os_bank_type]
        self.capTSV[TSV_type.Coarse] = self.tsv_parasitic_cap[global_ic_proj_type][tsv_os_bank_type]
        self.areaTSV[TSV_type.Coarse] = self.tsv_occupation_area[global_ic_proj_type][tsv_os_bank_type]

        self.layerCount = layers
