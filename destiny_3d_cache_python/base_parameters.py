#!/usr/bin/env python3
#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

"""
Base parameters file for DESTINY symbolic modeling
Creates SymPy symbolic variables for all technology and memory cell parameters
"""

from sympy import symbols, sqrt, log, exp, ceiling, Abs, Min, Max
from typedef import DeviceRoadmap
import globals as g


class BaseParameters:
    """
    Base parameters class that creates symbolic variables for all DESTINY parameters.
    This enables symbolic computation and optimization.
    """

    def __init__(self):
        """Initialize all symbolic parameters"""
        self.tech_values = {}  # Dictionary to store concrete values
        self.symbol_table = {}  # Dictionary mapping string names to symbols

        # Initialize all symbolic variables
        self.init_technology_params()
        self.init_memcell_params()
        self.init_wire_params()
        self.init_tsv_params()
        self.init_input_params()

        # Build symbol table
        self.build_symbol_table()

    def init_technology_params(self):
        """Initialize technology parameters (from Technology.py)"""
        # Feature size
        self.featureSize = symbols("featureSize", positive=True, real=True)
        self.featureSizeInNano = symbols("featureSizeInNano", positive=True, real=True)

        # Voltage parameters
        self.vdd = symbols("vdd", positive=True, real=True)
        self.vpp = symbols("vpp", positive=True, real=True)
        self.vth = symbols("vth", positive=True, real=True)
        self.vdsatNmos = symbols("vdsatNmos", positive=True, real=True)
        self.vdsatPmos = symbols("vdsatPmos", positive=True, real=True)

        # Gate parameters
        self.phyGateLength = symbols("phyGateLength", positive=True, real=True)
        self.capIdealGate = symbols("capIdealGate", positive=True, real=True)
        self.capFringe = symbols("capFringe", positive=True, real=True)
        self.capJunction = symbols("capJunction", positive=True, real=True)
        self.capOverlap = symbols("capOverlap", positive=True, real=True)
        self.capSidewall = symbols("capSidewall", positive=True, real=True)
        self.capDrainToChannel = symbols("capDrainToChannel", positive=True, real=True)
        self.capOx = symbols("capOx", positive=True, real=True)
        self.buildInPotential = symbols("buildInPotential", positive=True, real=True)

        # Mobility parameters
        self.effectiveElectronMobility = symbols("effectiveElectronMobility", positive=True, real=True)
        self.effectiveHoleMobility = symbols("effectiveHoleMobility", positive=True, real=True)

        # Device sizing
        self.pnSizeRatio = symbols("pnSizeRatio", positive=True, real=True)
        self.effectiveResistanceMultiplier = symbols("effectiveResistanceMultiplier", positive=True, real=True)

        # Current parameters (at reference temperature 300K)
        # We'll use single values instead of temperature arrays for symbolic computation
        self.currentOnNmos = symbols("currentOnNmos", positive=True, real=True)
        self.currentOnPmos = symbols("currentOnPmos", positive=True, real=True)
        self.currentOffNmos = symbols("currentOffNmos", positive=True, real=True)
        self.currentOffPmos = symbols("currentOffPmos", positive=True, real=True)

        # Poly wire capacitance
        self.capPolywire = symbols("capPolywire", positive=True, real=True)

    def init_memcell_params(self):
        """Initialize memory cell parameters (from MemCell.py)"""
        # Core properties
        self.cellArea = symbols("cellArea", positive=True, real=True)
        self.cellAspectRatio = symbols("cellAspectRatio", positive=True, real=True)
        self.cellWidthInFeatureSize = symbols("cellWidthInFeatureSize", positive=True, real=True)
        self.cellHeightInFeatureSize = symbols("cellHeightInFeatureSize", positive=True, real=True)

        # Resistance properties
        self.resistanceOn = symbols("resistanceOn", positive=True, real=True)
        self.resistanceOff = symbols("resistanceOff", positive=True, real=True)
        self.capacitanceOn = symbols("capacitanceOn", positive=True, real=True)
        self.capacitanceOff = symbols("capacitanceOff", positive=True, real=True)

        # Read properties
        self.readVoltage = symbols("readVoltage", positive=True, real=True)
        self.readCurrent = symbols("readCurrent", positive=True, real=True)
        self.minSenseVoltage = symbols("minSenseVoltage", positive=True, real=True)
        self.wordlineBoostRatio = symbols("wordlineBoostRatio", positive=True, real=True)
        self.readPower = symbols("readPower", positive=True, real=True)

        # Write properties (reset)
        self.resetVoltage = symbols("resetVoltage", positive=True, real=True)
        self.resetCurrent = symbols("resetCurrent", positive=True, real=True)
        self.resetPulse = symbols("resetPulse", positive=True, real=True)
        self.resetEnergy = symbols("resetEnergy", positive=True, real=True)

        # Write properties (set)
        self.setVoltage = symbols("setVoltage", positive=True, real=True)
        self.setCurrent = symbols("setCurrent", positive=True, real=True)
        self.setPulse = symbols("setPulse", positive=True, real=True)
        self.setEnergy = symbols("setEnergy", positive=True, real=True)

        # Access device properties
        self.widthAccessCMOS = symbols("widthAccessCMOS", positive=True, real=True)
        self.widthSRAMCellNMOS = symbols("widthSRAMCellNMOS", positive=True, real=True)
        self.widthSRAMCellPMOS = symbols("widthSRAMCellPMOS", positive=True, real=True)
        self.voltageDropAccessDevice = symbols("voltageDropAccessDevice", positive=True, real=True)
        self.leakageCurrentAccessDevice = symbols("leakageCurrentAccessDevice", positive=True, real=True)

        # DRAM properties
        self.capDRAMCell = symbols("capDRAMCell", positive=True, real=True)

        # FBRAM properties
        self.gateOxThicknessFactor = symbols("gateOxThicknessFactor", positive=True, real=True)
        self.widthSOIDevice = symbols("widthSOIDevice", positive=True, real=True)

    def init_wire_params(self):
        """Initialize wire parameters (from Wire.py)"""
        # Wire dimensions
        self.wirePitch = symbols("wirePitch", positive=True, real=True)
        self.wireWidth = symbols("wireWidth", positive=True, real=True)
        self.wireThickness = symbols("wireThickness", positive=True, real=True)
        self.wireSpacing = symbols("wireSpacing", positive=True, real=True)

        # Wire material properties
        self.wireResistivity = symbols("wireResistivity", positive=True, real=True)
        self.barrierThickness = symbols("barrierThickness", positive=True, real=True)
        self.dishingThickness = symbols("dishingThickness", positive=True, real=True)
        self.alphaScatter = symbols("alphaScatter", positive=True, real=True)

        # Wire dielectric properties
        self.ildThickness = symbols("ildThickness", positive=True, real=True)
        self.millerValue = symbols("millerValue", positive=True, real=True)
        self.horizontalDielectric = symbols("horizontalDielectric", positive=True, real=True)
        self.verticalDielectric = symbols("verticalDielectric", positive=True, real=True)
        self.fringeCap = symbols("fringeCap", positive=True, real=True)

        # Wire parasitics (calculated)
        self.wireResistance = symbols("wireResistance", positive=True, real=True)
        self.wireCapacitance = symbols("wireCapacitance", positive=True, real=True)

    def init_tsv_params(self):
        """Initialize TSV (Through-Silicon Via) parameters for 3D stacking"""
        # TSV dimensions
        self.tsvPitch = symbols("tsvPitch", positive=True, real=True)
        self.tsvDiameter = symbols("tsvDiameter", positive=True, real=True)
        self.tsvLength = symbols("tsvLength", positive=True, real=True)

        # TSV dielectric properties
        self.tsvDielecThickness = symbols("tsvDielecThickness", positive=True, real=True)
        self.tsvContactResistance = symbols("tsvContactResistance", positive=True, real=True)
        self.tsvDepletionWidth = symbols("tsvDepletionWidth", positive=True, real=True)
        self.tsvLinerDielectricConstant = symbols("tsvLinerDielectricConstant", positive=True, real=True)

        # TSV parasitics (calculated)
        self.tsvResistance = symbols("tsvResistance", positive=True, real=True)
        self.tsvCapacitance = symbols("tsvCapacitance", positive=True, real=True)
        self.tsvArea = symbols("tsvArea", positive=True, real=True)

    def init_input_params(self):
        """Initialize simulation input parameters"""
        self.temperature = symbols("temperature", positive=True, real=True)
        self.processNode = symbols("processNode", positive=True, real=True)
        self.maxNmosSize = symbols("maxNmosSize", positive=True, real=True)
        self.maxDriverCurrent = symbols("maxDriverCurrent", positive=True, real=True)

        # Cache/memory parameters
        self.capacity = symbols("capacity", positive=True, real=True)
        self.wordWidth = symbols("wordWidth", positive=True, real=True)
        self.associativity = symbols("associativity", positive=True, real=True)

        # 3D stacking
        self.stackedDieCount = symbols("stackedDieCount", positive=True, integer=True)

    def build_symbol_table(self):
        """Build mapping from string names to symbol objects"""
        # Technology parameters
        self.symbol_table.update({
            'featureSize': self.featureSize,
            'featureSizeInNano': self.featureSizeInNano,
            'vdd': self.vdd,
            'vpp': self.vpp,
            'vth': self.vth,
            'vdsatNmos': self.vdsatNmos,
            'vdsatPmos': self.vdsatPmos,
            'phyGateLength': self.phyGateLength,
            'capIdealGate': self.capIdealGate,
            'capFringe': self.capFringe,
            'capJunction': self.capJunction,
            'capOverlap': self.capOverlap,
            'capSidewall': self.capSidewall,
            'capDrainToChannel': self.capDrainToChannel,
            'capOx': self.capOx,
            'buildInPotential': self.buildInPotential,
            'effectiveElectronMobility': self.effectiveElectronMobility,
            'effectiveHoleMobility': self.effectiveHoleMobility,
            'pnSizeRatio': self.pnSizeRatio,
            'effectiveResistanceMultiplier': self.effectiveResistanceMultiplier,
            'currentOnNmos': self.currentOnNmos,
            'currentOnPmos': self.currentOnPmos,
            'currentOffNmos': self.currentOffNmos,
            'currentOffPmos': self.currentOffPmos,
            'capPolywire': self.capPolywire,
        })

        # Memory cell parameters
        self.symbol_table.update({
            'cellArea': self.cellArea,
            'cellAspectRatio': self.cellAspectRatio,
            'cellWidthInFeatureSize': self.cellWidthInFeatureSize,
            'cellHeightInFeatureSize': self.cellHeightInFeatureSize,
            'resistanceOn': self.resistanceOn,
            'resistanceOff': self.resistanceOff,
            'capacitanceOn': self.capacitanceOn,
            'capacitanceOff': self.capacitanceOff,
            'readVoltage': self.readVoltage,
            'readCurrent': self.readCurrent,
            'minSenseVoltage': self.minSenseVoltage,
            'wordlineBoostRatio': self.wordlineBoostRatio,
            'readPower': self.readPower,
            'resetVoltage': self.resetVoltage,
            'resetCurrent': self.resetCurrent,
            'resetPulse': self.resetPulse,
            'resetEnergy': self.resetEnergy,
            'setVoltage': self.setVoltage,
            'setCurrent': self.setCurrent,
            'setPulse': self.setPulse,
            'setEnergy': self.setEnergy,
            'widthAccessCMOS': self.widthAccessCMOS,
            'widthSRAMCellNMOS': self.widthSRAMCellNMOS,
            'widthSRAMCellPMOS': self.widthSRAMCellPMOS,
            'voltageDropAccessDevice': self.voltageDropAccessDevice,
            'leakageCurrentAccessDevice': self.leakageCurrentAccessDevice,
            'capDRAMCell': self.capDRAMCell,
            'gateOxThicknessFactor': self.gateOxThicknessFactor,
            'widthSOIDevice': self.widthSOIDevice,
        })

        # Wire parameters
        self.symbol_table.update({
            'wirePitch': self.wirePitch,
            'wireWidth': self.wireWidth,
            'wireThickness': self.wireThickness,
            'wireSpacing': self.wireSpacing,
            'wireResistivity': self.wireResistivity,
            'barrierThickness': self.barrierThickness,
            'dishingThickness': self.dishingThickness,
            'alphaScatter': self.alphaScatter,
            'ildThickness': self.ildThickness,
            'millerValue': self.millerValue,
            'horizontalDielectric': self.horizontalDielectric,
            'verticalDielectric': self.verticalDielectric,
            'fringeCap': self.fringeCap,
            'wireResistance': self.wireResistance,
            'wireCapacitance': self.wireCapacitance,
        })

        # TSV parameters
        self.symbol_table.update({
            'tsvPitch': self.tsvPitch,
            'tsvDiameter': self.tsvDiameter,
            'tsvLength': self.tsvLength,
            'tsvDielecThickness': self.tsvDielecThickness,
            'tsvContactResistance': self.tsvContactResistance,
            'tsvDepletionWidth': self.tsvDepletionWidth,
            'tsvLinerDielectricConstant': self.tsvLinerDielectricConstant,
            'tsvResistance': self.tsvResistance,
            'tsvCapacitance': self.tsvCapacitance,
            'tsvArea': self.tsvArea,
        })

        # Input parameters
        self.symbol_table.update({
            'temperature': self.temperature,
            'processNode': self.processNode,
            'maxNmosSize': self.maxNmosSize,
            'maxDriverCurrent': self.maxDriverCurrent,
            'capacity': self.capacity,
            'wordWidth': self.wordWidth,
            'associativity': self.associativity,
            'stackedDieCount': self.stackedDieCount,
        })

    def populate_from_technology(self, tech):
        """
        Populate tech_values dictionary from a Technology object.
        This extracts concrete numerical values.

        Args:
            tech: Technology object from Technology.py
        """
        self.tech_values[self.featureSize] = tech.featureSize
        self.tech_values[self.featureSizeInNano] = tech.featureSizeInNano
        self.tech_values[self.vdd] = tech.vdd
        self.tech_values[self.vpp] = tech.vpp
        self.tech_values[self.vth] = tech.vth
        self.tech_values[self.vdsatNmos] = tech.vdsatNmos
        self.tech_values[self.vdsatPmos] = tech.vdsatPmos
        self.tech_values[self.phyGateLength] = tech.phyGateLength
        self.tech_values[self.capIdealGate] = tech.capIdealGate
        self.tech_values[self.capFringe] = tech.capFringe
        self.tech_values[self.capJunction] = tech.capJunction
        self.tech_values[self.capOverlap] = tech.capOverlap
        self.tech_values[self.capSidewall] = tech.capSidewall
        self.tech_values[self.capDrainToChannel] = tech.capDrainToChannel
        self.tech_values[self.capOx] = tech.capOx
        self.tech_values[self.buildInPotential] = tech.buildInPotential
        self.tech_values[self.effectiveElectronMobility] = tech.effectiveElectronMobility
        self.tech_values[self.effectiveHoleMobility] = tech.effectiveHoleMobility
        self.tech_values[self.pnSizeRatio] = tech.pnSizeRatio
        self.tech_values[self.effectiveResistanceMultiplier] = tech.effectiveResistanceMultiplier
        # Use 300K values as reference
        self.tech_values[self.currentOnNmos] = tech.currentOnNmos[0]
        self.tech_values[self.currentOnPmos] = tech.currentOnPmos[0]
        self.tech_values[self.currentOffNmos] = tech.currentOffNmos[0]
        self.tech_values[self.currentOffPmos] = tech.currentOffPmos[0]
        self.tech_values[self.capPolywire] = tech.capPolywire

    def populate_from_memcell(self, cell):
        """
        Populate tech_values dictionary from a MemCell object.

        Args:
            cell: MemCell object from MemCell.py
        """
        self.tech_values[self.cellArea] = cell.area
        self.tech_values[self.cellAspectRatio] = cell.aspectRatio
        self.tech_values[self.cellWidthInFeatureSize] = cell.widthInFeatureSize
        self.tech_values[self.cellHeightInFeatureSize] = cell.heightInFeatureSize
        self.tech_values[self.resistanceOn] = cell.resistanceOn
        self.tech_values[self.resistanceOff] = cell.resistanceOff
        self.tech_values[self.capacitanceOn] = cell.capacitanceOn
        self.tech_values[self.capacitanceOff] = cell.capacitanceOff
        self.tech_values[self.readVoltage] = cell.readVoltage
        self.tech_values[self.readCurrent] = cell.readCurrent
        self.tech_values[self.minSenseVoltage] = cell.minSenseVoltage
        self.tech_values[self.wordlineBoostRatio] = cell.wordlineBoostRatio
        self.tech_values[self.readPower] = cell.readPower
        self.tech_values[self.resetVoltage] = cell.resetVoltage
        self.tech_values[self.resetCurrent] = cell.resetCurrent
        self.tech_values[self.resetPulse] = cell.resetPulse
        self.tech_values[self.resetEnergy] = cell.resetEnergy
        self.tech_values[self.setVoltage] = cell.setVoltage
        self.tech_values[self.setCurrent] = cell.setCurrent
        self.tech_values[self.setPulse] = cell.setPulse
        self.tech_values[self.setEnergy] = cell.setEnergy
        self.tech_values[self.widthAccessCMOS] = cell.widthAccessCMOS
        self.tech_values[self.widthSRAMCellNMOS] = cell.widthSRAMCellNMOS
        self.tech_values[self.widthSRAMCellPMOS] = cell.widthSRAMCellPMOS
        self.tech_values[self.voltageDropAccessDevice] = cell.voltageDropAccessDevice
        self.tech_values[self.leakageCurrentAccessDevice] = cell.leakageCurrentAccessDevice
        self.tech_values[self.capDRAMCell] = cell.capDRAMCell
        self.tech_values[self.gateOxThicknessFactor] = cell.gateOxThicknessFactor
        self.tech_values[self.widthSOIDevice] = cell.widthSOIDevice

    def populate_from_input_parameter(self, inputParam):
        """
        Populate tech_values dictionary from InputParameter object.

        Args:
            inputParam: InputParameter object from InputParameter.py
        """
        self.tech_values[self.temperature] = inputParam.temperature
        self.tech_values[self.processNode] = inputParam.processNode
        self.tech_values[self.maxNmosSize] = inputParam.maxNmosSize
        self.tech_values[self.maxDriverCurrent] = inputParam.maxDriverCurrent
        self.tech_values[self.capacity] = inputParam.capacity
        self.tech_values[self.wordWidth] = inputParam.wordWidth
        self.tech_values[self.associativity] = inputParam.associativity
        self.tech_values[self.stackedDieCount] = inputParam.stackedDieCount

    def print_summary(self):
        """Print summary of symbolic parameters"""
        print("\n" + "=" * 80)
        print("DESTINY Base Parameters - Symbolic Variables")
        print("=" * 80)
        print(f"\nTotal symbolic variables: {len(self.symbol_table)}")
        print(f"Total concrete values populated: {len(self.tech_values)}")

        print("\nTechnology Parameters:")
        print(f"  vdd = {self.vdd}")
        print(f"  vth = {self.vth}")
        print(f"  featureSize = {self.featureSize}")
        print(f"  capIdealGate = {self.capIdealGate}")

        print("\nMemory Cell Parameters:")
        print(f"  cellArea = {self.cellArea}")
        print(f"  resistanceOn = {self.resistanceOn}")
        print(f"  resistanceOff = {self.resistanceOff}")

        if self.tech_values:
            print(f"\nSample concrete values:")
            print(f"  vdd = {self.tech_values.get(self.vdd, 'not set')}")
            print(f"  featureSize = {self.tech_values.get(self.featureSize, 'not set')}")

        print("=" * 80 + "\n")


# Global instance (can be initialized in main.py)
base_params = None


def initialize_base_params():
    """Initialize the global base parameters object"""
    global base_params
    base_params = BaseParameters()
    return base_params


def get_base_params():
    """Get the global base parameters object"""
    global base_params
    if base_params is None:
        base_params = initialize_base_params()
    return base_params
