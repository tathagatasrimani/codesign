# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

import math
import sys
from constant import *
from Technology import Technology


def is_pow2(n):
    """Check if n is a power of 2"""
    if n < 1:
        return False
    return not (n & (n - 1))


def calculate_gate_cap(width, tech):
    """Calculate the gate capacitance"""
    return ((tech.capIdealGate + tech.capOverlap + 3 * tech.capFringe) * width +
            tech.phyGateLength * tech.capPolywire)


def calculate_fbram_gate_cap(width, thickness_factor, tech):
    """Calculate the capacitance of a FBRAM gate"""
    return ((tech.capIdealGate / thickness_factor + tech.capOverlap + 3 * tech.capFringe) * width +
            tech.phyGateLength * tech.capPolywire)


def calculate_fbram_drain_cap(width, tech):
    """Calculate the drain capacitance of a FBRAM"""
    return (3 * tech.capSidewall + tech.capDrainToChannel) * width


def calculate_gate_area(gate_type, num_input, width_nmos, width_pmos,
                        height_transistor_region, tech):
    """
    Calculate the gate area

    Returns:
        area: The area of the gate
        height: The height of the gate
        width: The width of the gate
    """
    ratio = width_pmos / (width_pmos + width_nmos)

    # Initialize variables
    max_width_pmos = 0
    max_width_nmos = 0
    unit_width_region_p = 0
    unit_width_region_n = 0
    width_region_p = 0
    width_region_n = 0
    height_region_p = 0
    height_region_n = 0

    if ratio == 0:  # no PMOS
        max_width_pmos = 0
        max_width_nmos = height_transistor_region
    elif ratio == 1:  # no NMOS
        max_width_pmos = height_transistor_region
        max_width_nmos = 0
    else:
        max_width_pmos = ratio * (height_transistor_region - MIN_GAP_BET_P_AND_N_DIFFS * tech.featureSize)
        max_width_nmos = max_width_pmos / ratio * (1 - ratio)

    if width_pmos > 0:
        if width_pmos < max_width_pmos:  # No folding
            unit_width_region_p = tech.featureSize
            height_region_p = width_pmos
        else:  # Folding
            num_folded_pmos = int(math.ceil(width_pmos / (max_width_pmos - 3 * tech.featureSize)))  # 3F for folding overhead
            unit_width_region_p = num_folded_pmos * tech.featureSize + (num_folded_pmos - 1) * tech.featureSize * MIN_GAP_BET_POLY
            height_region_p = max_width_pmos
    else:
        unit_width_region_p = 0
        height_region_p = 0

    if width_nmos > 0:
        if width_nmos < max_width_nmos:  # No folding
            unit_width_region_n = tech.featureSize
            height_region_n = width_nmos
        else:  # Folding
            num_folded_nmos = int(math.ceil(width_nmos / (max_width_nmos - 3 * tech.featureSize)))  # 3F for folding overhead
            unit_width_region_n = num_folded_nmos * tech.featureSize + (num_folded_nmos - 1) * tech.featureSize * MIN_GAP_BET_POLY
            height_region_n = max_width_nmos
    else:
        unit_width_region_n = 0
        height_region_n = 0

    if gate_type == INV:
        width_region_p = 2 * tech.featureSize * (CONTACT_SIZE + MIN_GAP_BET_CONTACT_POLY * 2) + unit_width_region_p
        width_region_n = 2 * tech.featureSize * (CONTACT_SIZE + MIN_GAP_BET_CONTACT_POLY * 2) + unit_width_region_n
    elif gate_type == NOR:
        width_region_p = (2 * tech.featureSize * (CONTACT_SIZE + MIN_GAP_BET_CONTACT_POLY * 2) +
                          unit_width_region_p * num_input + (num_input - 1) * tech.featureSize * MIN_GAP_BET_POLY)
        width_region_n = (2 * tech.featureSize * (CONTACT_SIZE + MIN_GAP_BET_CONTACT_POLY * 2) +
                          unit_width_region_n * num_input +
                          (num_input - 1) * tech.featureSize * (CONTACT_SIZE + MIN_GAP_BET_CONTACT_POLY * 2))
    elif gate_type == NAND:
        width_region_n = (2 * tech.featureSize * (CONTACT_SIZE + MIN_GAP_BET_CONTACT_POLY * 2) +
                          unit_width_region_n * num_input + (num_input - 1) * tech.featureSize * MIN_GAP_BET_POLY)
        width_region_p = (2 * tech.featureSize * (CONTACT_SIZE + MIN_GAP_BET_CONTACT_POLY * 2) +
                          unit_width_region_p * num_input +
                          (num_input - 1) * tech.featureSize * (CONTACT_SIZE + MIN_GAP_BET_CONTACT_POLY * 2))
    else:
        width_region_n = width_region_p = 0

    width = max(width_region_n, width_region_p)
    if width_pmos > 0 and width_nmos > 0:  # it is a gate
        height = (height_region_n + height_region_p + tech.featureSize * MIN_GAP_BET_P_AND_N_DIFFS +
                  2 * tech.featureSize * MIN_WIDTH_POWER_RAIL)
    else:  # it is a transistor
        height = height_region_n + height_region_p  # one of them is zero, and no power rail is added

    return width * height, height, width


def calculate_gate_capacitance(gate_type, num_input, width_nmos, width_pmos,
                                height_transistor_region, tech):
    """
    Calculate the capacitance of a gate

    Returns:
        cap_input: Input capacitance
        cap_output: Output capacitance
    """
    # TO-DO: most parts of this function is the same of calculate_gate_area,
    # perhaps they will be combined in future

    ratio = width_pmos / (width_pmos + width_nmos)

    max_width_pmos = 0
    max_width_nmos = 0
    unit_width_drain_p = 0
    unit_width_drain_n = 0
    width_drain_p = 0
    width_drain_n = 0
    height_drain_p = 0
    height_drain_n = 0
    num_folded_pmos = 1
    num_folded_nmos = 1

    if ratio == 0:  # no PMOS
        max_width_pmos = 0
        max_width_nmos = height_transistor_region
    elif ratio == 1:  # no NMOS
        max_width_pmos = height_transistor_region
        max_width_nmos = 0
    else:
        max_width_pmos = ratio * (height_transistor_region - MIN_GAP_BET_P_AND_N_DIFFS * tech.featureSize)
        max_width_nmos = max_width_pmos / ratio * (1 - ratio)

    if width_pmos > 0:
        if width_pmos < max_width_pmos:  # No folding
            unit_width_drain_p = 0
            height_drain_p = width_pmos
        else:  # Folding
            if max_width_pmos < 3 * tech.featureSize:
                print("Error: Unable to do PMOS folding because PMOS size limitation is less than 3F!")
                sys.exit(-1)
            num_folded_pmos = int(math.ceil(width_pmos / (max_width_pmos - 3 * tech.featureSize)))  # 3F for folding overhead
            unit_width_drain_p = (num_folded_pmos - 1) * tech.featureSize * MIN_GAP_BET_POLY
            height_drain_p = max_width_pmos
    else:
        unit_width_drain_p = 0
        height_drain_p = 0

    if width_nmos > 0:
        if width_nmos < max_width_nmos:  # No folding
            unit_width_drain_n = 0
            height_drain_n = width_nmos
        else:  # Folding
            if max_width_nmos < 3 * tech.featureSize:
                print("Error: Unable to do NMOS folding because NMOS size limitation is less than 3F!")
                sys.exit(-1)
            num_folded_nmos = int(math.ceil(width_nmos / (max_width_nmos - 3 * tech.featureSize)))  # 3F for folding overhead
            unit_width_drain_n = (num_folded_nmos - 1) * tech.featureSize * MIN_GAP_BET_POLY
            height_drain_n = max_width_nmos
    else:
        unit_width_drain_n = 0
        height_drain_n = 0

    if gate_type == INV:
        if width_pmos > 0:
            width_drain_p = tech.featureSize * (CONTACT_SIZE + MIN_GAP_BET_CONTACT_POLY * 2) + unit_width_drain_p
        if width_nmos > 0:
            width_drain_n = tech.featureSize * (CONTACT_SIZE + MIN_GAP_BET_CONTACT_POLY * 2) + unit_width_drain_n
    elif gate_type == NOR:
        # PMOS is in series, worst case capacitance is below
        if width_pmos > 0:
            width_drain_p = (tech.featureSize * (CONTACT_SIZE + MIN_GAP_BET_CONTACT_POLY * 2) +
                             unit_width_drain_p * num_input + (num_input - 1) * tech.featureSize * MIN_GAP_BET_POLY)
        # NMOS is parallel, capacitance is multiplied as below
        if width_nmos > 0:
            width_drain_n = ((tech.featureSize * (CONTACT_SIZE + MIN_GAP_BET_CONTACT_POLY * 2) +
                              unit_width_drain_n) * num_input)
    elif gate_type == NAND:
        # NMOS is in series, worst case capacitance is below
        if width_nmos > 0:
            width_drain_n = (tech.featureSize * (CONTACT_SIZE + MIN_GAP_BET_CONTACT_POLY * 2) +
                             unit_width_drain_n * num_input + (num_input - 1) * tech.featureSize * MIN_GAP_BET_POLY)
        # PMOS is parallel, capacitance is multiplied as below
        if width_pmos > 0:
            width_drain_p = ((tech.featureSize * (CONTACT_SIZE + MIN_GAP_BET_CONTACT_POLY * 2) +
                              unit_width_drain_p) * num_input)
    else:
        width_drain_n = width_drain_p = 0

    # Junction capacitance
    cap_drain_bottom_n = width_drain_n * height_drain_n * tech.capJunction
    cap_drain_bottom_p = width_drain_p * height_drain_p * tech.capJunction

    # Sidewall capacitance
    if num_folded_nmos % 2 == 0:
        cap_drain_sidewall_n = 2 * width_drain_n * tech.capSidewall
    else:
        cap_drain_sidewall_n = (2 * width_drain_n + height_drain_n) * tech.capSidewall

    if num_folded_pmos % 2 == 0:
        cap_drain_sidewall_p = 2 * width_drain_p * tech.capSidewall
    else:
        cap_drain_sidewall_p = (2 * width_drain_p + height_drain_p) * tech.capSidewall

    # Drain to channel capacitance
    cap_drain_to_channel_n = num_folded_nmos * height_drain_n * tech.capDrainToChannel
    cap_drain_to_channel_p = num_folded_pmos * height_drain_p * tech.capDrainToChannel

    cap_output = (cap_drain_bottom_n + cap_drain_bottom_p + cap_drain_sidewall_n +
                  cap_drain_sidewall_p + cap_drain_to_channel_n + cap_drain_to_channel_p)
    cap_input = calculate_gate_cap(width_nmos, tech) + calculate_gate_cap(width_pmos, tech)

    return cap_input, cap_output


def calculate_drain_cap(width, transistor_type, height_transistor_region, tech):
    """Calculate the drain capacitance"""
    if transistor_type == NMOS:
        _, drain_cap = calculate_gate_capacitance(INV, 1, width, 0, height_transistor_region, tech)
    else:
        _, drain_cap = calculate_gate_capacitance(INV, 1, 0, width, height_transistor_region, tech)
    return drain_cap


def calculate_gate_leakage(gate_type, num_input, width_nmos, width_pmos, temperature, tech):
    """Calculate the gate leakage"""
    temp_index = int(temperature) - 300
    if temp_index > 100 or temp_index < 0:
        print("Error: Temperature is out of range")
        sys.exit(-1)

    leak_n = tech.currentOffNmos
    leak_p = tech.currentOffPmos

    if gate_type == INV:
        leakage_n = width_nmos * leak_n[temp_index]
        leakage_p = width_pmos * leak_p[temp_index]
        return max(leakage_n, leakage_p)
    elif gate_type == NOR:
        leakage_n = width_nmos * leak_n[temp_index] * num_input
        if num_input == 2:
            return AVG_RATIO_LEAK_2INPUT_NOR * leakage_n
        else:
            return AVG_RATIO_LEAK_3INPUT_NOR * leakage_n
    elif gate_type == NAND:
        leakage_p = width_pmos * leak_p[temp_index] * num_input
        if num_input == 2:
            return AVG_RATIO_LEAK_2INPUT_NAND * leakage_p
        else:
            return AVG_RATIO_LEAK_3INPUT_NAND * leakage_p
    else:
        return 0.0


def calculate_on_resistance(width, transistor_type, temperature, tech):
    """Calculate the on resistance"""
    temp_index = int(temperature) - 300
    if temp_index > 100 or temp_index < 0:
        print("Error: Temperature is out of range")
        sys.exit(-1)

    if transistor_type == NMOS:
        r = tech.effectiveResistanceMultiplier * tech.vdd / (tech.currentOnNmos[temp_index] * width)
    else:
        r = tech.effectiveResistanceMultiplier * tech.vdd / (tech.currentOnPmos[temp_index] * width)
    return r


def calculate_transconductance(width, transistor_type, tech):
    """Calculate the transconductance"""
    if transistor_type == NMOS:
        vsat = min(tech.vdsatNmos, tech.vdd - tech.vth)
        gm = (tech.effectiveElectronMobility * tech.capOx) / 2 * width / tech.phyGateLength * vsat
    else:
        vsat = min(tech.vdsatPmos, tech.vdd - tech.vth)
        gm = (tech.effectiveHoleMobility * tech.capOx) / 2 * width / tech.phyGateLength * vsat
    return gm


def horowitz(tr, beta, ramp_input):
    """
    Horowitz timing model

    Returns:
        result: The delay
        ramp_output: The output ramp
    """
    alpha = 1 / ramp_input / tr
    vs = 0.5  # Normalized switching voltage
    result = tr * math.sqrt(math.log(vs) * math.log(vs) + 2 * alpha * beta * (1 - vs))
    ramp_output = (1 - vs) / result
    return result, ramp_output


def calculate_wire_resistance(resistivity, wire_width, wire_thickness,
                               barrier_thickness, dishing_thickness, alpha_scatter):
    """Calculate the wire resistance"""
    return (alpha_scatter * resistivity / (wire_thickness - barrier_thickness - dishing_thickness) /
            (wire_width - 2 * barrier_thickness))


def calculate_wire_capacitance(permittivity, wire_width, wire_thickness, wire_spacing,
                                ild_thickness, miller_value, horizontal_dielectric,
                                vertical_dielectric, fringe_cap):
    """Calculate the wire capacitance"""
    vertical_cap = 2 * permittivity * vertical_dielectric * wire_width / ild_thickness
    sidewall_cap = 2 * permittivity * miller_value * horizontal_dielectric * wire_thickness / wire_spacing
    return vertical_cap + sidewall_cap + fringe_cap


# PascalCase aliases for backward compatibility with C++ naming convention
CalculateGateCap = calculate_gate_cap
CalculateGateArea = calculate_gate_area
CalculateGateCapacitance = calculate_gate_capacitance
CalculateDrainCap = calculate_drain_cap
CalculateGateLeakage = calculate_gate_leakage
CalculateOnResistance = calculate_on_resistance
CalculateTransconductance = calculate_transconductance
MAX = max
MIN = min
