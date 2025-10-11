#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

from typedef import *


class InputParameter:
    def __init__(self):
        self.temperature = 300
        self.processNode = 0
        self.deviceRoadmap = DeviceRoadmap.HP
        self.maxNmosSize = 100.0
        self.maxDriverCurrent = 0.0

        # Design constraints
        self.designTarget = DesignTarget.cache
        self.optimizationTarget = OptimizationTarget.read_latency_optimized

        # Cache parameters
        self.cacheSize = 0
        self.lineSize = 0
        self.associativity = 1
        self.cacheAccessMode = CacheAccessMode.normal_access_mode

        # Memory parameters
        self.capacity = 0
        self.wordWidth = 0
        self.blockSize = 0

        # Technology parameters
        self.readLatencyConstraint = 0.0
        self.writeLatencyConstraint = 0.0
        self.readDynamicEnergyConstraint = 0.0
        self.writeDynamicEnergyConstraint = 0.0
        self.leakageConstraint = 0.0
        self.areaConstraint = 0.0
        self.readEdpConstraint = 0.0
        self.writeEdpConstraint = 0.0
        self.isConstraintApplied = False
        self.isPruningEnabled = False
        self.useCactiAssumption = False

        # Routing and sensing
        self.routingMode = RoutingMode.h_tree
        self.internalSensing = True

        # Write scheme
        self.writeScheme = WriteScheme.normal_write

        # NAND/DRAM parameters
        self.pageSize = 0
        self.flashBlockSize = 0

        # Memory cell files
        self.fileMemCell = []

        # Output file prefix
        self.outputFilePrefix = "output"

        # Design space exploration parameters (min/max ranges)
        self.minNumRowMat = 1
        self.maxNumRowMat = 512
        self.minNumColumnMat = 1
        self.maxNumColumnMat = 512
        self.minNumActiveMatPerRow = 1
        self.maxNumActiveMatPerRow = 512
        self.minNumActiveMatPerColumn = 1
        self.maxNumActiveMatPerColumn = 512
        self.minNumRowSubarray = 1
        self.maxNumRowSubarray = 2
        self.minNumColumnSubarray = 1
        self.maxNumColumnSubarray = 2
        self.minNumActiveSubarrayPerRow = 1
        self.maxNumActiveSubarrayPerRow = 2
        self.minNumActiveSubarrayPerColumn = 1
        self.maxNumActiveSubarrayPerColumn = 2
        self.minMuxSenseAmp = 1
        self.maxMuxSenseAmp = 256
        self.minMuxOutputLev1 = 1
        self.maxMuxOutputLev1 = 256
        self.minMuxOutputLev2 = 1
        self.maxMuxOutputLev2 = 256
        self.minNumRowPerSet = 1
        self.maxNumRowPerSet = 256
        self.minAreaOptimizationLevel = BufferDesignTarget.latency_first
        self.maxAreaOptimizationLevel = BufferDesignTarget.area_first
        self.minLocalWireType = WireType.local_aggressive
        self.maxLocalWireType = WireType.local_conservative
        self.minGlobalWireType = WireType.global_aggressive
        self.maxGlobalWireType = WireType.global_conservative
        self.minLocalWireRepeaterType = WireRepeaterType.repeated_none
        self.maxLocalWireRepeaterType = WireRepeaterType.repeated_50
        self.minGlobalWireRepeaterType = WireRepeaterType.repeated_none
        self.maxGlobalWireRepeaterType = WireRepeaterType.repeated_50
        self.minIsLocalWireLowSwing = 0
        self.maxIsLocalWireLowSwing = 1
        self.minIsGlobalWireLowSwing = 0
        self.maxIsGlobalWireLowSwing = 1

        # 3D stacking parameters
        self.stackedDieCount = 1
        self.partitionGranularity = 0
        self.localTSVProjection = TSV_type.Fine
        self.globalTSVProjection = TSV_type.Coarse
        self.globalTsvProjection = 1
        self.localTsvProjection = 1
        self.tsvRedundancy = 1.5
        self.monolithicStackCount = 1
        self.minStackLayer = 1
        self.maxStackLayer = 16
        self.forcedStackLayers = False

        # Additional flags
        self.doublePrune = False
        self.printAllOptimals = False
        self.allowDifferentTagTech = False
        self.printLevel = 1

    def ReadInputParameterFromFile(self, inputFile):
        """Read input parameters from configuration file"""
        try:
            self.forcedStackLayers = False

            with open(inputFile, 'r') as f:
                for line in f:
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith('//'):
                        continue

                    # Design Target
                    if line.startswith('-DesignTarget:'):
                        target = line.split(':')[1].strip()
                        if target.lower() == 'cache':
                            self.designTarget = DesignTarget.cache
                        elif target.lower() == 'ram':
                            self.designTarget = DesignTarget.RAM_chip
                            self.minNumRowPerSet = 1
                            self.maxNumRowPerSet = 1
                        else:
                            self.designTarget = DesignTarget.CAM_chip
                            self.minNumRowPerSet = 1
                            self.maxNumRowPerSet = 1

                    # Optimization Target
                    elif line.startswith('-OptimizationTarget:'):
                        target = line.split(':')[1].strip()
                        if target == 'ReadLatency':
                            self.optimizationTarget = OptimizationTarget.read_latency_optimized
                        elif target == 'WriteLatency':
                            self.optimizationTarget = OptimizationTarget.write_latency_optimized
                        elif target == 'ReadDynamicEnergy':
                            self.optimizationTarget = OptimizationTarget.read_energy_optimized
                        elif target == 'WriteDynamicEnergy':
                            self.optimizationTarget = OptimizationTarget.write_energy_optimized
                        elif target == 'ReadEDP':
                            self.optimizationTarget = OptimizationTarget.read_edp_optimized
                        elif target == 'WriteEDP':
                            self.optimizationTarget = OptimizationTarget.write_edp_optimized
                        elif target == 'LeakagePower':
                            self.optimizationTarget = OptimizationTarget.leakage_optimized
                        elif target == 'Area':
                            self.optimizationTarget = OptimizationTarget.area_optimized
                        else:
                            self.optimizationTarget = OptimizationTarget.full_exploration

                    # Output File Prefix
                    elif line.startswith('-OutputFilePrefix:'):
                        self.outputFilePrefix = line.split(':')[1].strip()

                    # Process Node
                    elif line.startswith('-ProcessNode:'):
                        self.processNode = int(line.split(':')[1].strip())

                    # Temperature
                    elif line.startswith('-Temperature'):
                        self.temperature = int(line.split(':')[1].strip().split()[0])

                    # Capacity (B)
                    elif line.startswith('-Capacity (B):'):
                        cap = int(line.split(':')[1].strip())
                        self.capacity = cap

                    # Capacity (KB)
                    elif line.startswith('-Capacity (KB):'):
                        cap = int(line.split(':')[1].strip())
                        self.capacity = cap * 1024

                    # Capacity (MB)
                    elif line.startswith('-Capacity (MB):'):
                        cap = int(line.split(':')[1].strip())
                        self.capacity = cap * 1024 * 1024
                        self.cacheSize = self.capacity

                    # Word Width
                    elif line.startswith('-WordWidth'):
                        self.wordWidth = int(line.split(':')[1].strip().split()[0])
                        self.lineSize = self.wordWidth // 8  # Convert bits to bytes

                    # Associativity
                    elif line.startswith('-Associativity'):
                        self.associativity = int(line.split(':')[1].strip())

                    # Max Driver Current
                    elif line.startswith('-MaxDriverCurrent'):
                        self.maxDriverCurrent = float(line.split(':')[1].strip().split()[0])

                    # Device Roadmap
                    elif line.startswith('-DeviceRoadmap:'):
                        roadmap = line.split(':')[1].strip()
                        if roadmap == 'HP':
                            self.deviceRoadmap = DeviceRoadmap.HP
                        elif roadmap == 'LSTP':
                            self.deviceRoadmap = DeviceRoadmap.LSTP
                        else:
                            self.deviceRoadmap = DeviceRoadmap.LOP

                    # Write Scheme
                    elif line.startswith('-WriteScheme:'):
                        scheme = line.split(':')[1].strip()
                        if scheme == 'SetBeforeReset':
                            self.writeScheme = WriteScheme.set_before_reset
                        elif scheme == 'ResetBeforeSet':
                            self.writeScheme = WriteScheme.reset_before_set
                        elif scheme == 'EraseBeforeSet':
                            self.writeScheme = WriteScheme.erase_before_set
                        elif scheme == 'EraseBeforeReset':
                            self.writeScheme = WriteScheme.erase_before_reset
                        elif scheme == 'WriteAndVerify':
                            self.writeScheme = WriteScheme.write_and_verify
                        else:
                            self.writeScheme = WriteScheme.normal_write

                    # Cache Access Mode
                    elif line.startswith('-CacheAccessMode:'):
                        mode = line.split(':')[1].strip()
                        if mode == 'Sequential':
                            self.cacheAccessMode = CacheAccessMode.sequential_access_mode
                        elif mode == 'Fast':
                            self.cacheAccessMode = CacheAccessMode.fast_access_mode
                        else:
                            self.cacheAccessMode = CacheAccessMode.normal_access_mode

                    # Local Wire Type
                    elif line.startswith('-LocalWireType:'):
                        wireType = line.split(':')[1].strip()
                        if wireType == 'LocalAggressive':
                            self.minLocalWireType = WireType.local_aggressive
                            self.maxLocalWireType = WireType.local_aggressive
                        elif wireType == 'LocalConservative':
                            self.minLocalWireType = WireType.local_conservative
                            self.maxLocalWireType = WireType.local_conservative
                        elif wireType == 'SemiAggressive':
                            self.minLocalWireType = WireType.semi_aggressive
                            self.maxLocalWireType = WireType.semi_aggressive
                        elif wireType == 'SemiConservative':
                            self.minLocalWireType = WireType.semi_conservative
                            self.maxLocalWireType = WireType.semi_conservative
                        elif wireType == 'GlobalAggressive':
                            self.minLocalWireType = WireType.global_aggressive
                            self.maxLocalWireType = WireType.global_aggressive
                        elif wireType == 'GlobalConservative':
                            self.minLocalWireType = WireType.global_conservative
                            self.maxLocalWireType = WireType.global_conservative
                        else:
                            self.minLocalWireType = WireType.dram_wordline
                            self.maxLocalWireType = WireType.dram_wordline

                    # Local Wire Repeater Type
                    elif line.startswith('-LocalWireRepeaterType:'):
                        repeaterType = line.split(':')[1].strip()
                        if repeaterType == 'RepeatedOpt':
                            self.minLocalWireRepeaterType = WireRepeaterType.repeated_opt
                            self.maxLocalWireRepeaterType = WireRepeaterType.repeated_opt
                        elif repeaterType == 'Repeated5%Penalty':
                            self.minLocalWireRepeaterType = WireRepeaterType.repeated_5
                            self.maxLocalWireRepeaterType = WireRepeaterType.repeated_5
                        elif repeaterType == 'Repeated10%Penalty':
                            self.minLocalWireRepeaterType = WireRepeaterType.repeated_10
                            self.maxLocalWireRepeaterType = WireRepeaterType.repeated_10
                        elif repeaterType == 'Repeated20%Penalty':
                            self.minLocalWireRepeaterType = WireRepeaterType.repeated_20
                            self.maxLocalWireRepeaterType = WireRepeaterType.repeated_20
                        elif repeaterType == 'Repeated30%Penalty':
                            self.minLocalWireRepeaterType = WireRepeaterType.repeated_30
                            self.maxLocalWireRepeaterType = WireRepeaterType.repeated_30
                        elif repeaterType == 'Repeated40%Penalty':
                            self.minLocalWireRepeaterType = WireRepeaterType.repeated_40
                            self.maxLocalWireRepeaterType = WireRepeaterType.repeated_40
                        elif repeaterType == 'Repeated50%Penalty':
                            self.minLocalWireRepeaterType = WireRepeaterType.repeated_50
                            self.maxLocalWireRepeaterType = WireRepeaterType.repeated_50
                        else:
                            self.minLocalWireRepeaterType = WireRepeaterType.repeated_none
                            self.maxLocalWireRepeaterType = WireRepeaterType.repeated_none

                    # Local Wire Use Low Swing
                    elif line.startswith('-LocalWireUseLowSwing:'):
                        lowSwing = line.split(':')[1].strip()
                        if lowSwing == 'Yes':
                            self.minIsLocalWireLowSwing = 1
                            self.maxIsLocalWireLowSwing = 1
                        else:
                            self.minIsLocalWireLowSwing = 0
                            self.maxIsLocalWireLowSwing = 0

                    # Global Wire Type
                    elif line.startswith('-GlobalWireType:'):
                        wireType = line.split(':')[1].strip()
                        if wireType == 'LocalAggressive':
                            self.minGlobalWireType = WireType.local_aggressive
                            self.maxGlobalWireType = WireType.local_aggressive
                        elif wireType == 'LocalConservative':
                            self.minGlobalWireType = WireType.local_conservative
                            self.maxGlobalWireType = WireType.local_conservative
                        elif wireType == 'SemiAggressive':
                            self.minGlobalWireType = WireType.semi_aggressive
                            self.maxGlobalWireType = WireType.semi_aggressive
                        elif wireType == 'SemiConservative':
                            self.minGlobalWireType = WireType.semi_conservative
                            self.maxGlobalWireType = WireType.semi_conservative
                        elif wireType == 'GlobalAggressive':
                            self.minGlobalWireType = WireType.global_aggressive
                            self.maxGlobalWireType = WireType.global_aggressive
                        elif wireType == 'GlobalConservative':
                            self.minGlobalWireType = WireType.global_conservative
                            self.maxGlobalWireType = WireType.global_conservative
                        else:
                            self.minGlobalWireType = WireType.dram_wordline
                            self.maxGlobalWireType = WireType.dram_wordline

                    # Global Wire Repeater Type
                    elif line.startswith('-GlobalWireRepeaterType:'):
                        repeaterType = line.split(':')[1].strip()
                        if repeaterType == 'RepeatedOpt':
                            self.minGlobalWireRepeaterType = WireRepeaterType.repeated_opt
                            self.maxGlobalWireRepeaterType = WireRepeaterType.repeated_opt
                        elif repeaterType == 'Repeated5%Penalty':
                            self.minGlobalWireRepeaterType = WireRepeaterType.repeated_5
                            self.maxGlobalWireRepeaterType = WireRepeaterType.repeated_5
                        elif repeaterType == 'Repeated10%Penalty':
                            self.minGlobalWireRepeaterType = WireRepeaterType.repeated_10
                            self.maxGlobalWireRepeaterType = WireRepeaterType.repeated_10
                        elif repeaterType == 'Repeated20%Penalty':
                            self.minGlobalWireRepeaterType = WireRepeaterType.repeated_20
                            self.maxGlobalWireRepeaterType = WireRepeaterType.repeated_20
                        elif repeaterType == 'Repeated30%Penalty':
                            self.minGlobalWireRepeaterType = WireRepeaterType.repeated_30
                            self.maxGlobalWireRepeaterType = WireRepeaterType.repeated_30
                        elif repeaterType == 'Repeated40%Penalty':
                            self.minGlobalWireRepeaterType = WireRepeaterType.repeated_40
                            self.maxGlobalWireRepeaterType = WireRepeaterType.repeated_40
                        elif repeaterType == 'Repeated50%Penalty':
                            self.minGlobalWireRepeaterType = WireRepeaterType.repeated_50
                            self.maxGlobalWireRepeaterType = WireRepeaterType.repeated_50
                        else:
                            self.minGlobalWireRepeaterType = WireRepeaterType.repeated_none
                            self.maxGlobalWireRepeaterType = WireRepeaterType.repeated_none

                    # Global Wire Use Low Swing
                    elif line.startswith('-GlobalWireUseLowSwing:'):
                        lowSwing = line.split(':')[1].strip()
                        if lowSwing == 'Yes':
                            self.minIsGlobalWireLowSwing = 1
                            self.maxIsGlobalWireLowSwing = 1
                        else:
                            self.minIsGlobalWireLowSwing = 0
                            self.maxIsGlobalWireLowSwing = 0

                    # Routing Mode
                    elif line.startswith('-Routing:'):
                        routing = line.split(':')[1].strip()
                        if routing == 'H-tree':
                            self.routingMode = RoutingMode.h_tree
                        else:
                            self.routingMode = RoutingMode.non_h_tree

                    # Internal Sensing
                    elif line.startswith('-InternalSensing:'):
                        sensing = line.split(':')[1].strip()
                        if sensing.lower() == 'true':
                            self.internalSensing = True
                        else:
                            self.internalSensing = False

                    # Memory Cell Input File
                    elif line.startswith('-MemoryCellInputFile:'):
                        fileName = line.split(':')[1].strip()
                        self.fileMemCell.append(fileName)

                    # Max NMOS Size
                    elif line.startswith('-MaxNmosSize'):
                        self.maxNmosSize = float(line.split(':')[1].strip().split()[0])

                    # Enable Pruning
                    elif line.startswith('-EnablePruning:'):
                        pruning = line.split(':')[1].strip()
                        if pruning == 'Yes':
                            self.isPruningEnabled = True
                        else:
                            self.isPruningEnabled = False

                    # Buffer Design Optimization
                    elif line.startswith('-BufferDesignOptimization:'):
                        optimization = line.split(':')[1].strip()
                        if optimization == 'latency':
                            self.minAreaOptimizationLevel = BufferDesignTarget.latency_first
                            self.maxAreaOptimizationLevel = BufferDesignTarget.latency_first
                        elif optimization == 'area':
                            self.minAreaOptimizationLevel = BufferDesignTarget.area_first
                            self.maxAreaOptimizationLevel = BufferDesignTarget.area_first
                        else:
                            self.minAreaOptimizationLevel = BufferDesignTarget.latency_area_trade_off
                            self.maxAreaOptimizationLevel = BufferDesignTarget.latency_area_trade_off

                    # Flash Page Size
                    elif line.startswith('-FlashPageSize'):
                        pageSize = int(line.split(':')[1].strip().split()[0])
                        self.pageSize = pageSize * 8  # Byte to bit

                    # Flash Block Size
                    elif line.startswith('-FlashBlockSize'):
                        blockSize = int(line.split(':')[1].strip().split()[0])
                        self.flashBlockSize = blockSize * 8 * 1024  # KB to bit

                    # Use CACTI Assumption
                    elif line.startswith('-UseCactiAssumption:'):
                        assumption = line.split(':')[1].strip()
                        if assumption == 'Yes':
                            self.useCactiAssumption = True
                            self.minNumActiveMatPerRow = self.maxNumColumnMat
                            self.maxNumActiveMatPerRow = self.maxNumColumnMat
                            self.minNumActiveMatPerColumn = 1
                            self.maxNumActiveMatPerColumn = 1
                            self.minNumRowSubarray = 2
                            self.maxNumRowSubarray = 2
                            self.minNumColumnSubarray = 2
                            self.maxNumColumnSubarray = 2
                            self.minNumActiveSubarrayPerRow = 2
                            self.maxNumActiveSubarrayPerRow = 2
                            self.minNumActiveSubarrayPerColumn = 2
                            self.maxNumActiveSubarrayPerColumn = 2
                        else:
                            self.useCactiAssumption = False

                    # Apply Constraints
                    elif line.startswith('-ApplyReadLatencyConstraint:'):
                        self.readLatencyConstraint = float(line.split(':')[1].strip())
                        self.isConstraintApplied = True

                    elif line.startswith('-ApplyWriteLatencyConstraint:'):
                        self.writeLatencyConstraint = float(line.split(':')[1].strip())
                        self.isConstraintApplied = True

                    elif line.startswith('-ApplyReadDynamicEnergyConstraint:'):
                        self.readDynamicEnergyConstraint = float(line.split(':')[1].strip())
                        self.isConstraintApplied = True

                    elif line.startswith('-ApplyWriteDynamicEnergyConstraint:'):
                        self.writeDynamicEnergyConstraint = float(line.split(':')[1].strip())
                        self.isConstraintApplied = True

                    elif line.startswith('-ApplyLeakageConstraint:'):
                        self.leakageConstraint = float(line.split(':')[1].strip())
                        self.isConstraintApplied = True

                    elif line.startswith('-ApplyAreaConstraint:'):
                        self.areaConstraint = float(line.split(':')[1].strip())
                        self.isConstraintApplied = True

                    elif line.startswith('-ApplyReadEdpConstraint:'):
                        self.readEdpConstraint = float(line.split(':')[1].strip())
                        self.isConstraintApplied = True

                    elif line.startswith('-ApplyWriteEdpConstraint:'):
                        self.writeEdpConstraint = float(line.split(':')[1].strip())
                        self.isConstraintApplied = True

                    # Forced configurations
                    elif line.startswith('-ForceBank3DA'):
                        parts = line.split(':')[1].strip().split(',')
                        total = parts[0].strip().split('x')
                        self.minNumRowMat = int(total[0])
                        self.minNumColumnMat = int(total[1])
                        self.minStackLayer = int(total[2])
                        self.maxNumRowMat = self.minNumRowMat
                        self.maxNumColumnMat = self.minNumColumnMat
                        self.maxStackLayer = self.minStackLayer
                        if self.forcedStackLayers:
                            print("Warning: Number of die stacked layers specified twice!")
                        self.forcedStackLayers = True

                    elif line.startswith('-ForceBank3D'):
                        parts = line.split(':')[1].strip().split(',')
                        total = parts[0].strip().split('x')
                        active = parts[1].strip().split('x')
                        self.minNumRowMat = int(total[0])
                        self.minNumColumnMat = int(total[1])
                        self.minStackLayer = int(total[2])
                        self.minNumActiveMatPerColumn = int(active[0])
                        self.minNumActiveMatPerRow = int(active[1])
                        self.maxNumRowMat = self.minNumRowMat
                        self.maxNumColumnMat = self.minNumColumnMat
                        self.maxStackLayer = self.minStackLayer
                        self.maxNumActiveMatPerColumn = self.minNumActiveMatPerColumn
                        self.maxNumActiveMatPerRow = self.minNumActiveMatPerRow
                        if self.forcedStackLayers:
                            print("Warning: Number of die stacked layers specified twice!")
                        self.forcedStackLayers = True

                    elif line.startswith('-ForceBankA'):
                        parts = line.split(':')[1].strip().split('x')
                        self.minNumRowMat = int(parts[0])
                        self.minNumColumnMat = int(parts[1])
                        self.maxNumRowMat = self.minNumRowMat
                        self.maxNumColumnMat = self.minNumColumnMat

                    elif line.startswith('-ForceBank'):
                        parts = line.split(':')[1].strip().split(',')
                        total = parts[0].strip().split('x')
                        active = parts[1].strip().split('x')
                        self.minNumRowMat = int(total[0])
                        self.minNumColumnMat = int(total[1])
                        self.minNumActiveMatPerColumn = int(active[0])
                        self.minNumActiveMatPerRow = int(active[1])
                        self.maxNumRowMat = self.minNumRowMat
                        self.maxNumColumnMat = self.minNumColumnMat
                        self.maxNumActiveMatPerColumn = self.minNumActiveMatPerColumn
                        self.maxNumActiveMatPerRow = self.minNumActiveMatPerRow

                    elif line.startswith('-ForceMatA'):
                        parts = line.split(':')[1].strip().split('x')
                        self.minNumRowSubarray = int(parts[0])
                        self.minNumColumnSubarray = int(parts[1])
                        self.maxNumRowSubarray = self.minNumRowSubarray
                        self.maxNumColumnSubarray = self.minNumColumnSubarray

                    elif line.startswith('-ForceMat'):
                        parts = line.split(':')[1].strip().split(',')
                        total = parts[0].strip().split('x')
                        active = parts[1].strip().split('x')
                        self.minNumRowSubarray = int(total[0])
                        self.minNumColumnSubarray = int(total[1])
                        self.minNumActiveSubarrayPerColumn = int(active[0])
                        self.minNumActiveSubarrayPerRow = int(active[1])
                        self.maxNumRowSubarray = self.minNumRowSubarray
                        self.maxNumColumnSubarray = self.minNumColumnSubarray
                        self.maxNumActiveSubarrayPerColumn = self.minNumActiveSubarrayPerColumn
                        self.maxNumActiveSubarrayPerRow = self.minNumActiveSubarrayPerRow

                    elif line.startswith('-ForceMuxSenseAmp:'):
                        self.minMuxSenseAmp = int(line.split(':')[1].strip())
                        self.maxMuxSenseAmp = self.minMuxSenseAmp

                    elif line.startswith('-ForceMuxOutputLev1:'):
                        self.minMuxOutputLev1 = int(line.split(':')[1].strip())
                        self.maxMuxOutputLev1 = self.minMuxOutputLev1

                    elif line.startswith('-ForceMuxOutputLev2:'):
                        self.minMuxOutputLev2 = int(line.split(':')[1].strip())
                        self.maxMuxOutputLev2 = self.minMuxOutputLev2

                    # Stacked Die Count
                    elif line.startswith('-StackedDieCount:'):
                        self.stackedDieCount = int(line.split(':')[1].strip())
                        self.minStackLayer = self.stackedDieCount
                        self.maxStackLayer = self.stackedDieCount
                        if self.forcedStackLayers:
                            print("Warning: Number of die stacked layers specified twice!")
                        self.forcedStackLayers = True

                    # Partition Granularity
                    elif line.startswith('-PartitionGranularity:'):
                        self.partitionGranularity = int(line.split(':')[1].strip())

                    # Local TSV Projection
                    elif line.startswith('-LocalTSVProjection:'):
                        val = int(line.split(':')[1].strip())
                        self.localTSVProjection = TSV_type.Fine if val == 0 else TSV_type.Coarse
                        self.localTsvProjection = val

                    # Global TSV Projection
                    elif line.startswith('-GlobalTSVProjection:'):
                        val = int(line.split(':')[1].strip())
                        self.globalTSVProjection = TSV_type.Fine if val == 0 else TSV_type.Coarse
                        self.globalTsvProjection = val

                    # TSV Redundancy
                    elif line.startswith('-TSVRedundancy:'):
                        self.tsvRedundancy = float(line.split(':')[1].strip())

                    # Monolithic Stack Count
                    elif line.startswith('-MonolithicStackCount:'):
                        self.monolithicStackCount = int(line.split(':')[1].strip())

                    # Print All Optimals
                    elif line.startswith('-PrintAllOptimals:'):
                        printAll = line.split(':')[1].strip()
                        if printAll.lower() == 'true':
                            self.printAllOptimals = True
                        else:
                            self.printAllOptimals = False

                    # Allow Different Tag Tech
                    elif line.startswith('-AllowDifferentTagTech:'):
                        allowDiff = line.split(':')[1].strip()
                        if allowDiff.lower() == 'true':
                            self.allowDifferentTagTech = True
                        else:
                            self.allowDifferentTagTech = False

                    # Print Level
                    elif line.startswith('-PrintLevel:'):
                        self.printLevel = int(line.split(':')[1].strip())

        except FileNotFoundError:
            print(f"Error: {inputFile} cannot be found!")
            import sys
            sys.exit(-1)

    def PrintInputParameter(self):
        """Print input parameters"""
        print(f"Temperature: {self.temperature} K")
        print(f"Process Node: {self.processNode} nm")
        print(f"Device Roadmap: {self.deviceRoadmap}")
        print(f"Cache Size: {self.cacheSize} bytes")
        print(f"Line Size: {self.lineSize} bytes")
        print(f"Associativity: {self.associativity}")
