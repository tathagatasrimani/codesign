#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

import math
from typedef import MemCellType, CellAccessType
import globals as g


class MemCell:
    """
    Memory cell class that encapsulates all properties and behaviors of different
    memory cell types including SRAM, DRAM, eDRAM, MRAM, PCRAM, Memristor, FBRAM, and NAND Flash.
    """

    def __init__(self):
        """Initialize memory cell with default values"""
        # Core properties
        self.memCellType = MemCellType.PCRAM
        self.processNode = 0  # Cell original process technology node, Unit: nm
        self.area = 0.0  # Cell area, Unit: F^2
        self.aspectRatio = 0.0  # Cell aspect ratio, H/W
        self.widthInFeatureSize = 0.0  # Cell width, Unit: F
        self.heightInFeatureSize = 0.0  # Cell height, Unit: F

        # Resistance properties
        self.resistanceOn = 0.0  # Turn-on resistance
        self.resistanceOff = 0.0  # Turn-off resistance
        self.capacitanceOn = 0.0  # Cell capacitance when memristor is on
        self.capacitanceOff = 0.0  # Cell capacitance when memristor is off

        # Read properties
        self.readMode = True  # True = voltage-mode, False = current-mode
        self.readVoltage = 0.0  # Read voltage
        self.readCurrent = 0.0  # Read current
        self.minSenseVoltage = 0.08  # Minimum sense voltage
        self.wordlineBoostRatio = 1.0  # Ratio of boost wordline voltage to vdd
        self.readPower = 0.0  # Read power per bitline (uW)

        # Reset properties
        self.resetMode = True  # True = voltage-mode, False = current-mode
        self.resetVoltage = 0.0  # Reset voltage
        self.resetCurrent = 0.0  # Reset current
        self.resetPulse = 0.0  # Reset pulse duration (ns)
        self.resetEnergy = 0.0  # Reset energy per cell (pJ)

        # Set properties
        self.setMode = True  # True = voltage-mode, False = current-mode
        self.setVoltage = 0.0  # Set voltage
        self.setCurrent = 0.0  # Set current
        self.setPulse = 0.0  # Set pulse duration (ns)
        self.setEnergy = 0.0  # Set energy per cell (pJ)

        # Access type
        self.accessType = CellAccessType.CMOS_access

        # Optional properties
        self.stitching = 0  # If non-zero, add stitching overhead for every x cells
        self.gateOxThicknessFactor = 2.0  # The oxide thickness of FBRAM could be larger than traditional SOI MOS
        self.widthSOIDevice = 0.0  # The gate width of SOI device as FBRAM element, Unit: F
        self.widthAccessCMOS = 0.0  # The gate width of CMOS access transistor, Unit: F
        self.voltageDropAccessDevice = 0.0  # The voltage drop on the access device, Unit: V
        self.leakageCurrentAccessDevice = 0.0  # Reverse current of access device, Unit: uA
        self.capDRAMCell = 0.0  # The DRAM cell capacitance if the memory cell is DRAM, Unit: F
        self.widthSRAMCellNMOS = 2.08  # Default NMOS width in SRAM cells is 2.08 (from CACTI)
        self.widthSRAMCellPMOS = 1.23  # Default PMOS width in SRAM cells is 1.23 (from CACTI)

        # For memristor
        self.readFloating = False  # If unselected wordlines/bitlines are floating to reduce total leakage
        self.resistanceOnAtSetVoltage = 0.0  # Low resistance state when set voltage is applied
        self.resistanceOffAtSetVoltage = 0.0  # High resistance state when set voltage is applied
        self.resistanceOnAtResetVoltage = 0.0  # Low resistance state when reset voltage is applied
        self.resistanceOffAtResetVoltage = 0.0  # High resistance state when reset voltage is applied
        self.resistanceOnAtReadVoltage = 0.0  # Low resistance state when read voltage is applied
        self.resistanceOffAtReadVoltage = 0.0  # High resistance state when read voltage is applied
        self.resistanceOnAtHalfReadVoltage = 0.0  # Low resistance state when 1/2 read voltage is applied
        self.resistanceOffAtHalfReadVoltage = 0.0  # High resistance state when 1/2 read voltage is applied
        self.resistanceOnAtHalfResetVoltage = 0.0  # Low resistance state when 1/2 reset voltage is applied

        # For NAND flash
        self.flashEraseVoltage = 0.0  # The erase voltage, Unit: V, highest W/E voltage in ITRS sheet
        self.flashPassVoltage = 0.0  # The voltage applied on the unselected wordline within the same block during programming, Unit: V
        self.flashProgramVoltage = 0.0  # The program voltage, Unit: V
        self.flashEraseTime = 0.0  # The flash erase time, Unit: s
        self.flashProgramTime = 0.0  # The SLC flash program time, Unit: s
        self.gateCouplingRatio = 0.0  # The ratio of control gate to total floating gate capacitance

        # For eDRAM
        self.retentionTime = g.invalid_value  # Cell time to data loss (us)
        self.temperature = 0.0  # Temperature for which the cell input values are valid

    def ReadCellFromFile(self, inputFile):
        """
        Read memory cell parameters from a .cell file

        Args:
            inputFile: Path to the .cell configuration file
        """
        try:
            with open(inputFile, 'r') as fp:
                for line in fp:
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue

                    # Parse MemCellType
                    if line.startswith("-MemCellType"):
                        cell_type = line.split(':')[1].strip()
                        if cell_type == "SRAM":
                            self.memCellType = MemCellType.SRAM
                        elif cell_type == "DRAM":
                            self.memCellType = MemCellType.DRAM
                        elif cell_type == "eDRAM":
                            self.memCellType = MemCellType.eDRAM
                        elif cell_type == "MRAM":
                            self.memCellType = MemCellType.MRAM
                        elif cell_type == "PCRAM":
                            self.memCellType = MemCellType.PCRAM
                        elif cell_type == "FBRAM":
                            self.memCellType = MemCellType.FBRAM
                        elif cell_type == "memristor":
                            self.memCellType = MemCellType.memristor
                        elif cell_type == "SLCNAND":
                            self.memCellType = MemCellType.SLCNAND
                        else:
                            self.memCellType = MemCellType.MLCNAND
                        continue

                    # Parse ProcessNode
                    if line.startswith("-ProcessNode"):
                        self.processNode = int(line.split(':')[1].strip())
                        continue

                    # Parse CellArea
                    if line.startswith("-CellArea"):
                        self.area = float(line.split(':')[1].strip())
                        continue

                    # Parse CellAspectRatio
                    if line.startswith("-CellAspectRatio"):
                        self.aspectRatio = float(line.split(':')[1].strip())
                        self.heightInFeatureSize = math.sqrt(self.area * self.aspectRatio)
                        self.widthInFeatureSize = math.sqrt(self.area / self.aspectRatio)
                        continue

                    # Parse memristor resistance values at different voltages
                    if line.startswith("-ResistanceOnAtSetVoltage"):
                        self.resistanceOnAtSetVoltage = float(line.split(':')[1].strip())
                        continue

                    if line.startswith("-ResistanceOffAtSetVoltage"):
                        self.resistanceOffAtSetVoltage = float(line.split(':')[1].strip())
                        continue

                    if line.startswith("-ResistanceOnAtResetVoltage"):
                        self.resistanceOnAtResetVoltage = float(line.split(':')[1].strip())
                        continue

                    if line.startswith("-ResistanceOffAtResetVoltage"):
                        self.resistanceOffAtResetVoltage = float(line.split(':')[1].strip())
                        continue

                    if line.startswith("-ResistanceOnAtReadVoltage"):
                        self.resistanceOnAtReadVoltage = float(line.split(':')[1].strip())
                        self.resistanceOn = self.resistanceOnAtReadVoltage
                        continue

                    if line.startswith("-ResistanceOffAtReadVoltage"):
                        self.resistanceOffAtReadVoltage = float(line.split(':')[1].strip())
                        self.resistanceOff = self.resistanceOffAtReadVoltage
                        continue

                    if line.startswith("-ResistanceOnAtHalfReadVoltage"):
                        self.resistanceOnAtHalfReadVoltage = float(line.split(':')[1].strip())
                        continue

                    if line.startswith("-ResistanceOffAtHalfReadVoltage"):
                        self.resistanceOffAtHalfReadVoltage = float(line.split(':')[1].strip())
                        continue

                    if line.startswith("-ResistanceOnAtHalfResetVoltage"):
                        self.resistanceOnAtHalfResetVoltage = float(line.split(':')[1].strip())
                        continue

                    # Parse basic resistance values
                    if line.startswith("-ResistanceOn") and "AtReadVoltage" not in line and "AtSetVoltage" not in line and "AtResetVoltage" not in line and "AtHalfReadVoltage" not in line and "AtHalfResetVoltage" not in line:
                        self.resistanceOn = float(line.split(':')[1].strip())
                        continue

                    if line.startswith("-ResistanceOff") and "AtReadVoltage" not in line and "AtSetVoltage" not in line and "AtResetVoltage" not in line and "AtHalfReadVoltage" not in line:
                        self.resistanceOff = float(line.split(':')[1].strip())
                        continue

                    # Parse capacitance
                    if line.startswith("-CapacitanceOn"):
                        self.capacitanceOn = float(line.split(':')[1].strip())
                        continue

                    if line.startswith("-CapacitanceOff"):
                        self.capacitanceOff = float(line.split(':')[1].strip())
                        continue

                    # Parse gate oxide thickness factor
                    if line.startswith("-GateOxThicknessFactor"):
                        self.gateOxThicknessFactor = float(line.split(':')[1].strip())
                        continue

                    # Parse SOI device width
                    if line.startswith("-SOIDeviceWidth"):
                        self.widthSOIDevice = float(line.split(':')[1].strip())
                        continue

                    # Parse ReadMode
                    if line.startswith("-ReadMode"):
                        mode = line.split(':')[1].strip()
                        self.readMode = (mode == "voltage")
                        continue

                    # Parse ReadVoltage
                    if line.startswith("-ReadVoltage"):
                        self.readVoltage = float(line.split(':')[1].strip())
                        continue

                    # Parse ReadCurrent
                    if line.startswith("-ReadCurrent"):
                        self.readCurrent = float(line.split(':')[1].strip())
                        self.readCurrent /= 1e6  # Convert from uA to A
                        continue

                    # Parse ReadPower
                    if line.startswith("-ReadPower") and "ReadEnergy" not in line:
                        self.readPower = float(line.split(':')[1].strip())
                        self.readPower /= 1e6  # Convert from uW to W
                        continue

                    # Parse WordlineBoostRatio
                    if line.startswith("-WordlineBoostRatio"):
                        self.wordlineBoostRatio = float(line.split(':')[1].strip())
                        continue

                    # Parse MinSenseVoltage
                    if line.startswith("-MinSenseVoltage"):
                        self.minSenseVoltage = float(line.split(':')[1].strip())
                        self.minSenseVoltage /= 1e3  # Convert from mV to V
                        continue

                    # Parse ResetMode
                    if line.startswith("-ResetMode"):
                        mode = line.split(':')[1].strip()
                        self.resetMode = (mode == "voltage")
                        continue

                    # Parse ResetVoltage
                    if line.startswith("-ResetVoltage"):
                        value = line.split(':')[1].strip()
                        if value == "vdd":
                            self.resetVoltage = g.tech.vdd if g.tech else 0.0
                        else:
                            self.resetVoltage = float(value)
                        continue

                    # Parse ResetCurrent
                    if line.startswith("-ResetCurrent"):
                        self.resetCurrent = float(line.split(':')[1].strip())
                        self.resetCurrent /= 1e6  # Convert from uA to A
                        continue

                    # Parse ResetPulse
                    if line.startswith("-ResetPulse"):
                        self.resetPulse = float(line.split(':')[1].strip())
                        self.resetPulse /= 1e9  # Convert from ns to s
                        continue

                    # Parse ResetEnergy
                    if line.startswith("-ResetEnergy"):
                        self.resetEnergy = float(line.split(':')[1].strip())
                        self.resetEnergy /= 1e12  # Convert from pJ to J
                        continue

                    # Parse SetMode
                    if line.startswith("-SetMode"):
                        mode = line.split(':')[1].strip()
                        self.setMode = (mode == "voltage")
                        continue

                    # Parse SetVoltage
                    if line.startswith("-SetVoltage"):
                        value = line.split(':')[1].strip()
                        if value == "vdd":
                            self.setVoltage = g.tech.vdd if g.tech else 0.0
                        else:
                            self.setVoltage = float(value)
                        continue

                    # Parse SetCurrent
                    if line.startswith("-SetCurrent"):
                        self.setCurrent = float(line.split(':')[1].strip())
                        self.setCurrent /= 1e6  # Convert from uA to A
                        continue

                    # Parse SetPulse
                    if line.startswith("-SetPulse"):
                        self.setPulse = float(line.split(':')[1].strip())
                        self.setPulse /= 1e9  # Convert from ns to s
                        continue

                    # Parse SetEnergy
                    if line.startswith("-SetEnergy"):
                        self.setEnergy = float(line.split(':')[1].strip())
                        self.setEnergy /= 1e12  # Convert from pJ to J
                        continue

                    # Parse Stitching
                    if line.startswith("-Stitching"):
                        self.stitching = int(line.split(':')[1].strip())
                        continue

                    # Parse AccessType
                    if line.startswith("-AccessType"):
                        access_type = line.split(':')[1].strip()
                        if access_type == "CMOS":
                            self.accessType = CellAccessType.CMOS_access
                        elif access_type == "BJT":
                            self.accessType = CellAccessType.BJT_access
                        elif access_type == "diode":
                            self.accessType = CellAccessType.diode_access
                        else:
                            self.accessType = CellAccessType.none_access
                        continue

                    # Parse AccessCMOSWidth
                    if line.startswith("-AccessCMOSWidth"):
                        if self.accessType != CellAccessType.CMOS_access:
                            print("Warning: The input of CMOS access transistor width is ignored because the cell is not CMOS-accessed.")
                        else:
                            self.widthAccessCMOS = float(line.split(':')[1].strip())
                        continue

                    # Parse VoltageDropAccessDevice
                    if line.startswith("-VoltageDropAccessDevice"):
                        self.voltageDropAccessDevice = float(line.split(':')[1].strip())
                        continue

                    # Parse LeakageCurrentAccessDevice
                    if line.startswith("-LeakageCurrentAccessDevice"):
                        self.leakageCurrentAccessDevice = float(line.split(':')[1].strip())
                        self.leakageCurrentAccessDevice /= 1e6  # Convert from uA to A
                        continue

                    # Parse DRAMCellCapacitance
                    if line.startswith("-DRAMCellCapacitance"):
                        if self.memCellType != MemCellType.DRAM and self.memCellType != MemCellType.eDRAM:
                            print("Warning: The input of DRAM cell capacitance is ignored because the memory cell is not DRAM.")
                        else:
                            self.capDRAMCell = float(line.split(':')[1].strip())
                        continue

                    # Parse SRAMCellNMOSWidth
                    if line.startswith("-SRAMCellNMOSWidth"):
                        if self.memCellType != MemCellType.SRAM:
                            print("Warning: The input of SRAM cell NMOS width is ignored because the memory cell is not SRAM.")
                        else:
                            self.widthSRAMCellNMOS = float(line.split(':')[1].strip())
                        continue

                    # Parse SRAMCellPMOSWidth
                    if line.startswith("-SRAMCellPMOSWidth"):
                        if self.memCellType != MemCellType.SRAM:
                            print("Warning: The input of SRAM cell PMOS width is ignored because the memory cell is not SRAM.")
                        else:
                            self.widthSRAMCellPMOS = float(line.split(':')[1].strip())
                        continue

                    # Parse ReadFloating
                    if line.startswith("-ReadFloating"):
                        value = line.split(':')[1].strip()
                        self.readFloating = (value == "true")
                        continue

                    # Parse FlashEraseVoltage
                    if line.startswith("-FlashEraseVoltage"):
                        if self.memCellType != MemCellType.SLCNAND and self.memCellType != MemCellType.MLCNAND:
                            print("Warning: The input of programming/erase voltage is ignored because the memory cell is not flash.")
                        else:
                            self.flashEraseVoltage = float(line.split(':')[1].strip())
                        continue

                    # Parse FlashProgramVoltage
                    if line.startswith("-FlashProgramVoltage"):
                        if self.memCellType != MemCellType.SLCNAND and self.memCellType != MemCellType.MLCNAND:
                            print("Warning: The input of programming/program voltage is ignored because the memory cell is not flash.")
                        else:
                            self.flashProgramVoltage = float(line.split(':')[1].strip())
                        continue

                    # Parse FlashPassVoltage
                    if line.startswith("-FlashPassVoltage"):
                        if self.memCellType != MemCellType.SLCNAND and self.memCellType != MemCellType.MLCNAND:
                            print("Warning: The input of pass voltage is ignored because the memory cell is not flash.")
                        else:
                            self.flashPassVoltage = float(line.split(':')[1].strip())
                        continue

                    # Parse FlashEraseTime
                    if line.startswith("-FlashEraseTime"):
                        if self.memCellType != MemCellType.SLCNAND and self.memCellType != MemCellType.MLCNAND:
                            print("Warning: The input of erase time is ignored because the memory cell is not flash.")
                        else:
                            self.flashEraseTime = float(line.split(':')[1].strip())
                            self.flashEraseTime /= 1e3  # Convert from ms to s
                        continue

                    # Parse FlashProgramTime
                    if line.startswith("-FlashProgramTime"):
                        if self.memCellType != MemCellType.SLCNAND and self.memCellType != MemCellType.MLCNAND:
                            print("Warning: The input of erase time is ignored because the memory cell is not flash.")
                        else:
                            self.flashProgramTime = float(line.split(':')[1].strip())
                            self.flashProgramTime /= 1e6  # Convert from us to s
                        continue

                    # Parse GateCouplingRatio
                    if line.startswith("-GateCouplingRatio"):
                        if self.memCellType != MemCellType.SLCNAND and self.memCellType != MemCellType.MLCNAND:
                            print("Warning: The input of gate coupling ratio (GCR) is ignored because the memory cell is not flash.")
                        else:
                            self.gateCouplingRatio = float(line.split(':')[1].strip())
                        continue

                    # Parse RetentionTime
                    if line.startswith("-RetentionTime"):
                        if self.memCellType != MemCellType.eDRAM:
                            print("Warning: The input of retention time is ignored because the cell is not eDRAM.")
                        else:
                            self.retentionTime = float(line.split(':')[1].strip())
                            self.retentionTime /= 1e6  # Convert from us to s
                        continue

                    # Parse Temperature
                    if line.startswith("-Temperature"):
                        if self.memCellType != MemCellType.eDRAM:
                            print("Warning: The input of temperature is ignored because the cell is not eDRAM.")
                        else:
                            self.temperature = float(line.split(':')[1].strip())
                        continue

        except FileNotFoundError:
            print(f"{inputFile} cannot be found!")
            print("This file may be present in \"config\" folder. If so, please run destiny from that folder, otherwise, change the file name to include folder location.")
            raise

    def ApplyPVT(self):
        """
        Apply Process-Voltage-Temperature (PVT) variations to the memory cell.
        Currently handles retention time scaling for eDRAM based on temperature.
        """
        if self.retentionTime == g.invalid_value:
            # TODO: No given retention time, we should calculate it
            return

        if self.memCellType == MemCellType.eDRAM:
            print(f"[Info] Retention time given at {self.temperature}K is {self.retentionTime * 1e6}us")
            exponent = -0.0268 * (g.inputParameter.temperature - self.temperature)
            self.retentionTime = self.retentionTime * math.exp(exponent)
            print(f"[Info] Retention time at {g.inputParameter.temperature}K is {self.retentionTime * 1e6}us")

    def CellScaling(self, _targetProcessNode):
        """
        Scale cell parameters to a new technology node

        Args:
            _targetProcessNode: Target process node in nm
        """
        if self.processNode > 0 and self.processNode != _targetProcessNode:
            scalingFactor = self.processNode / _targetProcessNode

            if self.memCellType == MemCellType.PCRAM:
                self.resistanceOn *= scalingFactor
                self.resistanceOff *= scalingFactor
                if not self.setMode:
                    self.setCurrent /= scalingFactor
                else:
                    self.setVoltage *= 1
                if not self.resetMode:
                    self.resetCurrent /= scalingFactor
                else:
                    self.resetVoltage *= 1
                if self.accessType == CellAccessType.diode_access:
                    self.capacitanceOn /= scalingFactor  # TO-DO
                    self.capacitanceOff /= scalingFactor  # TO-DO

            elif self.memCellType == MemCellType.MRAM:  # TO-DO: MRAM
                self.resistanceOn *= scalingFactor * scalingFactor
                self.resistanceOff *= scalingFactor * scalingFactor
                if not self.setMode:
                    self.setCurrent /= scalingFactor
                else:
                    self.setVoltage *= scalingFactor
                if not self.resetMode:
                    self.resetCurrent /= scalingFactor
                else:
                    self.resetVoltage *= scalingFactor
                if self.accessType == CellAccessType.diode_access:
                    self.capacitanceOn /= scalingFactor  # TO-DO
                    self.capacitanceOff /= scalingFactor  # TO-DO

            elif self.memCellType == MemCellType.memristor:  # TO-DO: memristor
                pass

            else:  # TO-DO: other RAMs
                pass

            self.processNode = _targetProcessNode

    def GetMemristance(self, _relativeReadVoltage):
        """
        Get the LRS resistance of memristor at log-linear region of I-V curve

        Args:
            _relativeReadVoltage: Relative read voltage (fraction of read voltage)

        Returns:
            Memristance at the specified voltage, or -1 if not a memristor
        """
        if self.memCellType == MemCellType.memristor:
            # x1: read voltage, x2: half voltage, x3: applied voltage
            if self.readVoltage == 0:
                x1 = self.readCurrent * self.resistanceOnAtReadVoltage
            else:
                x1 = self.readVoltage
            x2 = self.readVoltage / 2
            x3 = _relativeReadVoltage * self.readVoltage

            # y1:log(read current), y2: log(leakage current at half read voltage)
            y1 = math.log2(x1 / self.resistanceOnAtReadVoltage)
            y2 = math.log2(x2 / self.resistanceOnAtHalfReadVoltage)
            y3 = (y2 - y1) / (x2 - x1) * x3 + (x2 * y1 - x1 * y2) / (x2 - x1)  # Interpolation
            return x3 / math.pow(2, y3)
        else:
            print("Warning[MemCell] : Try to get memristance from a non-memristor memory cell")
            return -1

    def CalculateWriteEnergy(self):
        """Calculate write energy (reset and set) for the memory cell"""
        # Calculate reset energy
        if self.resetEnergy == 0:
            if self.resetMode:  # Voltage mode
                if self.memCellType == MemCellType.memristor:
                    if self.accessType == CellAccessType.none_access:
                        self.resetEnergy = abs(self.resetVoltage) * (abs(self.resetVoltage) - self.voltageDropAccessDevice) / self.resistanceOnAtResetVoltage * self.resetPulse
                    else:
                        self.resetEnergy = abs(self.resetVoltage) * (abs(self.resetVoltage) - self.voltageDropAccessDevice) / self.resistanceOn * self.resetPulse
                elif self.memCellType == MemCellType.PCRAM:
                    self.resetEnergy = abs(self.resetVoltage) * (abs(self.resetVoltage) - self.voltageDropAccessDevice) / self.resistanceOn * self.resetPulse
                elif self.memCellType == MemCellType.FBRAM:
                    self.resetEnergy = abs(self.resetVoltage) * abs(self.resetCurrent) * self.resetPulse
                else:
                    self.resetEnergy = abs(self.resetVoltage) * (abs(self.resetVoltage) - self.voltageDropAccessDevice) / self.resistanceOn * self.resetPulse
            else:  # Current mode
                if self.resetVoltage == 0:
                    self.resetEnergy = g.tech.vdd * abs(self.resetCurrent) * self.resetPulse if g.tech else 0
                else:
                    self.resetEnergy = abs(self.resetVoltage) * abs(self.resetCurrent) * self.resetPulse

        # Calculate set energy
        if self.setEnergy == 0:
            if self.setMode:  # Voltage mode
                if self.memCellType == MemCellType.memristor:
                    if self.accessType == CellAccessType.none_access:
                        self.setEnergy = abs(self.setVoltage) * (abs(self.setVoltage) - self.voltageDropAccessDevice) / self.resistanceOnAtSetVoltage * self.setPulse
                    else:
                        self.setEnergy = abs(self.setVoltage) * (abs(self.setVoltage) - self.voltageDropAccessDevice) / self.resistanceOn * self.setPulse
                elif self.memCellType == MemCellType.PCRAM:
                    self.setEnergy = abs(self.setVoltage) * (abs(self.setVoltage) - self.voltageDropAccessDevice) / self.resistanceOn * self.setPulse
                elif self.memCellType == MemCellType.FBRAM:
                    self.setEnergy = abs(self.setVoltage) * abs(self.setCurrent) * self.setPulse
                else:
                    self.setEnergy = abs(self.setVoltage) * (abs(self.setVoltage) - self.voltageDropAccessDevice) / self.resistanceOn * self.setPulse
            else:  # Current mode
                if self.resetVoltage == 0:
                    self.setEnergy = g.tech.vdd * abs(self.setCurrent) * self.setPulse if g.tech else 0
                else:
                    self.setEnergy = abs(self.setVoltage) * abs(self.setCurrent) * self.setPulse

    def CalculateReadPower(self):
        """
        Calculate read power for the memory cell

        Returns:
            Read power in W, or -1.0 if read power is already set or calculation fails
        """
        if self.readPower == 0:
            if self.readMode:  # Voltage-sensing
                if self.readVoltage == 0:  # Current-in voltage sensing
                    return g.tech.vdd * self.readCurrent if g.tech else 0
                if self.readCurrent == 0:  # Voltage-divider sensing
                    resInSerialForSenseAmp = math.sqrt(self.resistanceOn * self.resistanceOff)
                    maxBitlineCurrent = (self.readVoltage - self.voltageDropAccessDevice) / (self.resistanceOn + resInSerialForSenseAmp)
                    return g.tech.vdd * maxBitlineCurrent if g.tech else 0
            else:  # Current-sensing
                maxBitlineCurrent = (self.readVoltage - self.voltageDropAccessDevice) / self.resistanceOn
                return g.tech.vdd * maxBitlineCurrent if g.tech else 0
        else:
            return -1.0  # Should not call the function if read energy exists
        return -1.0

    def PrintCell(self, indent=0):
        """
        Print memory cell properties

        Args:
            indent: Number of spaces to indent the output
        """
        indent_str = ' ' * indent

        # Print cell type
        cell_type_names = {
            MemCellType.SRAM: "SRAM",
            MemCellType.DRAM: "DRAM",
            MemCellType.eDRAM: "Embedded DRAM",
            MemCellType.MRAM: "MRAM (Magnetoresistive)",
            MemCellType.PCRAM: "PCRAM (Phase-Change)",
            MemCellType.memristor: "RRAM (Memristor)",
            MemCellType.FBRAM: "FBRAM (Floating Body)",
            MemCellType.SLCNAND: "Single-Level Cell NAND Flash",
            MemCellType.MLCNAND: "Multi-Level Cell NAND Flash"
        }
        print(f"{indent_str}Memory Cell: {cell_type_names.get(self.memCellType, 'Unknown')}")
        print(f"{indent_str}Cell Area (F^2)    : {self.area} ({self.heightInFeatureSize}Fx{self.widthInFeatureSize}F)")
        print(f"{indent_str}Cell Aspect Ratio  : {self.aspectRatio}")

        # Print type-specific properties
        if self.memCellType in [MemCellType.PCRAM, MemCellType.MRAM, MemCellType.memristor, MemCellType.FBRAM]:
            # Print resistance
            if self.resistanceOn < 1e3:
                print(f"{indent_str}Cell Turned-On Resistance : {self.resistanceOn}ohm")
            elif self.resistanceOn < 1e6:
                print(f"{indent_str}Cell Turned-On Resistance : {self.resistanceOn / 1e3}Kohm")
            else:
                print(f"{indent_str}Cell Turned-On Resistance : {self.resistanceOn / 1e6}Mohm")

            if self.resistanceOff < 1e3:
                print(f"{indent_str}Cell Turned-Off Resistance: {self.resistanceOff}ohm")
            elif self.resistanceOff < 1e6:
                print(f"{indent_str}Cell Turned-Off Resistance: {self.resistanceOff / 1e3}Kohm")
            else:
                print(f"{indent_str}Cell Turned-Off Resistance: {self.resistanceOff / 1e6}Mohm")

            # Print read mode
            if self.readMode:
                print(f"{indent_str}Read Mode: Voltage-Sensing")
                if self.readCurrent > 0:
                    print(f"{indent_str}  - Read Current: {self.readCurrent * 1e6}uA")
                if self.readVoltage > 0:
                    print(f"{indent_str}  - Read Voltage: {self.readVoltage}V")
            else:
                print(f"{indent_str}Read Mode: Current-Sensing")
                if self.readCurrent > 0:
                    print(f"{indent_str}  - Read Current: {self.readCurrent * 1e6}uA")
                if self.readVoltage > 0:
                    print(f"{indent_str}  - Read Voltage: {self.readVoltage}V")

            # Print reset mode
            if self.resetMode:
                print(f"{indent_str}Reset Mode: Voltage")
                print(f"{indent_str}  - Reset Voltage: {self.resetVoltage}V")
            else:
                print(f"{indent_str}Reset Mode: Current")
                print(f"{indent_str}  - Reset Current: {self.resetCurrent * 1e6}uA")
            print(f"{indent_str}  - Reset Pulse: {self._to_second_str(self.resetPulse)}")

            # Print set mode
            if self.setMode:
                print(f"{indent_str}Set Mode: Voltage")
                print(f"{indent_str}  - Set Voltage: {self.setVoltage}V")
            else:
                print(f"{indent_str}Set Mode: Current")
                print(f"{indent_str}  - Set Current: {self.setCurrent * 1e6}uA")
            print(f"{indent_str}  - Set Pulse: {self._to_second_str(self.setPulse)}")

            # Print access type
            access_type_names = {
                CellAccessType.CMOS_access: "CMOS",
                CellAccessType.BJT_access: "BJT",
                CellAccessType.diode_access: "Diode",
                CellAccessType.none_access: "None Access Device"
            }
            print(f"{indent_str}Access Type: {access_type_names.get(self.accessType, 'Unknown')}")

        elif self.memCellType == MemCellType.SRAM:
            print(f"{indent_str}SRAM Cell Access Transistor Width: {self.widthAccessCMOS}F")
            print(f"{indent_str}SRAM Cell NMOS Width: {self.widthSRAMCellNMOS}F")
            print(f"{indent_str}SRAM Cell PMOS Width: {self.widthSRAMCellPMOS}F")

        elif self.memCellType == MemCellType.SLCNAND:
            print(f"{indent_str}Pass Voltage       : {self.flashPassVoltage}V")
            print(f"{indent_str}Programming Voltage: {self.flashProgramVoltage}V")
            print(f"{indent_str}Erase Voltage      : {self.flashEraseVoltage}V")
            print(f"{indent_str}Programming Time   : {self._to_second_str(self.flashProgramTime)}")
            print(f"{indent_str}Erase Time         : {self._to_second_str(self.flashEraseTime)}")
            print(f"{indent_str}Gate Coupling Ratio: {self.gateCouplingRatio}")

    def _to_second_str(self, time_seconds):
        """Helper function to convert time in seconds to human-readable string"""
        if time_seconds >= 1:
            return f"{time_seconds}s"
        elif time_seconds >= 1e-3:
            return f"{time_seconds * 1e3}ms"
        elif time_seconds >= 1e-6:
            return f"{time_seconds * 1e6}us"
        elif time_seconds >= 1e-9:
            return f"{time_seconds * 1e9}ns"
        elif time_seconds >= 1e-12:
            return f"{time_seconds * 1e12}ps"
        else:
            return f"{time_seconds}s"
