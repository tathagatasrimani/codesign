# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

from enum import Enum, IntEnum


class MemCellType(IntEnum):
    SRAM = 0
    DRAM = 1
    eDRAM = 2
    MRAM = 3
    PCRAM = 4
    memristor = 5
    FBRAM = 6
    SLCNAND = 7
    MLCNAND = 8


class CellAccessType(IntEnum):
    CMOS_access = 0
    BJT_access = 1
    diode_access = 2
    none_access = 3


class DeviceRoadmap(IntEnum):
    HP = 0      # High performance
    LSTP = 1    # Low standby power
    LOP = 2     # Low operating power
    EDRAM = 3   # Embedded DRAM


class WireType(IntEnum):
    local_aggressive = 0      # Width = 2.5F
    local_conservative = 1
    semi_aggressive = 2       # Width = 4F
    semi_conservative = 3
    global_aggressive = 4     # Width = 8F
    global_conservative = 5
    dram_wordline = 6         # CACTI 6.5 has this one, but we don't plan to support it at this moment


class WireRepeaterType(IntEnum):
    repeated_none = 0    # No repeater
    repeated_opt = 1     # Add Repeater, optimal delay
    repeated_5 = 2       # Add Repeater, 5% delay overhead
    repeated_10 = 3      # Add Repeater, 10% delay overhead
    repeated_20 = 4      # Add Repeater, 20% delay overhead
    repeated_30 = 5      # Add Repeater, 30% delay overhead
    repeated_40 = 6      # Add Repeater, 40% delay overhead
    repeated_50 = 7      # Add Repeater, 50% delay overhead


class BufferDesignTarget(IntEnum):
    latency_first = 0           # The buffer will be optimized for latency
    latency_area_trade_off = 1  # the buffer will be fixed to 2-stage
    area_first = 2              # The buffer will be optimized for area


class MemoryType(IntEnum):
    data = 0
    tag = 1
    CAM = 2


class RoutingMode(IntEnum):
    h_tree = 0
    non_h_tree = 1


class WriteScheme(IntEnum):
    set_before_reset = 0
    reset_before_set = 1
    erase_before_set = 2
    erase_before_reset = 3
    write_and_verify = 4
    normal_write = 5


class DesignTarget(IntEnum):
    cache = 0
    RAM_chip = 1
    CAM_chip = 2


class OptimizationTarget(IntEnum):
    read_latency_optimized = 0
    write_latency_optimized = 1
    read_energy_optimized = 2
    write_energy_optimized = 3
    read_edp_optimized = 4
    write_edp_optimized = 5
    leakage_optimized = 6
    area_optimized = 7
    full_exploration = 8


class CacheAccessMode(IntEnum):
    normal_access_mode = 0      # data array lookup and tag access happen in parallel
                                # final data block is broadcasted in data array h-tree
                                # after getting the signal from the tag array
    sequential_access_mode = 1  # data array is accessed after accessing the tag array
    fast_access_mode = 2        # data and tag access happen in parallel


class TSV_type(IntEnum):
    Fine = 0           # ITRS high density
    Coarse = 1         # Industry reported in 2010
    NUM_TSV_TYPES = 2
