#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from CACTI-3DD, (c) 2012 Hewlett-Packard Development Company, L.P.
#See LICENSE_CACTI3DD file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

from FunctionUnit import FunctionUnit
from formula import *
from constant import *
import globals as g


class TSV(FunctionUnit):
    def __init__(self):
        super().__init__()
        self.w_TSV_n = [0.0] * MAX_NUMBER_GATES_STAGE
        self.w_TSV_p = [0.0] * MAX_NUMBER_GATES_STAGE

        self.numTotalBits = 0
        self.numAccessBits = 0
        self.numReadBits = 0
        self.numDataBits = 0

        self.res = 0.0
        self.cap = 0.0
        self.C_load_TSV = 0.0
        self.min_area = 0.0
        self.F = 0.0

        self.num_gates = 0
        self.TSV_metal_area = 0.0
        self.buffer_area = 0.0
        self.buffer_area_height = 0.0
        self.buffer_area_width = 0.0

        self.tsv_type = 0
        self.initialized = False
        self.invalid = True

    def Initialize(self, tsv_type, buffered=False):
        num_gates_min = 1
        min_w_pmos = g.tech.pnSizeRatio * MIN_NMOS_SIZE * g.tech.featureSize
        self.num_gates = 1

        self.cap = g.tech.capTSV[tsv_type]
        self.res = g.tech.resTSV[tsv_type]
        self.min_area = g.tech.areaTSV[tsv_type] * 1e-12

        if not buffered:
            self.num_gates = 0
        else:
            first_buf_stg_coef = 5  # To tune the total buffer delay
            self.w_TSV_n[0] = MIN_NMOS_SIZE * first_buf_stg_coef * g.tech.featureSize
            self.w_TSV_p[0] = self.w_TSV_n[0] * g.tech.pnSizeRatio

            self.C_load_TSV = self.cap + CalculateGateCap(MIN_NMOS_SIZE * g.tech.featureSize + min_w_pmos, g.tech)

            self.F = self.C_load_TSV / CalculateGateCap(self.w_TSV_n[0] + self.w_TSV_p[0], g.tech)

            # Obtain buffer chain stages using logic effort function
            self.num_gates = self.logical_effort(
                num_gates_min,
                1,
                self.F,
                self.w_TSV_n,
                self.w_TSV_p,
                self.C_load_TSV,
                g.tech.pnSizeRatio,
                MAX_NMOS_SIZE * g.tech.featureSize,
                g.tech
            )

        self.initialized = True
        if self.num_gates > MAX_NUMBER_GATES_STAGE:
            self.invalid = True

    def CalculateArea(self):
        Vdd = g.tech.vdd
        cumulative_area = 0
        cumulative_curr = 0
        cumulative_curr_Ig = 0
        self.buffer_area_height = 50 * g.tech.featureSize
        temperature = float(g.inputParameter.temperature)

        for i in range(self.num_gates):
            area, tempHeight, tempWidth = CalculateGateArea(INV, 1, self.w_TSV_n[i], self.w_TSV_p[i],
                                                             g.tech.featureSize * MAX_TRANSISTOR_HEIGHT,
                                                             g.tech)
            cumulative_area += area
            cumulative_curr += CalculateGateLeakage(INV, 1, self.w_TSV_n[i], self.w_TSV_p[i],
                                                    temperature, g.tech)
            cumulative_curr_Ig += CalculateGateLeakage(INV, 1, self.w_TSV_n[i], self.w_TSV_p[i],
                                                        temperature, g.tech)

        self.leakage = cumulative_curr_Ig * Vdd

        self.buffer_area = cumulative_area
        self.buffer_area_width = self.buffer_area / self.buffer_area_height

        self.TSV_metal_area = self.min_area * 3.1416 / 16

        if self.buffer_area < (self.min_area - self.TSV_metal_area):
            self.area = self.min_area
        else:
            self.area = self.buffer_area + self.TSV_metal_area

    def CalculateLatencyAndPower(self, _rampInputRead, _rampInputWrite):
        assert _rampInputRead != 0 and _rampInputWrite != 0

        # Assume we are using the same TSV type/size/etc. we are just driving in a different direction
        self.readDynamicEnergy, self.readLatency = self._CalculateLatencyAndPower(_rampInputRead)
        self.writeDynamicEnergy, self.writeLatency = self._CalculateLatencyAndPower(_rampInputWrite)

        # reset, set, erase are same driving direction as write
        self.resetDynamicEnergy = self.setDynamicEnergy = self.writeDynamicEnergy
        self.resetLatency = self.setLatency = self.writeLatency

    def _CalculateLatencyAndPower(self, _rampInput):
        delay = 0.0
        _dynamicEnergy = 0.0
        rampOutput = 0.0
        beta = 0.5

        if self.num_gates > 0:
            rd = CalculateOnResistance(self.w_TSV_n[0], NMOS, g.inputParameter.temperature, g.tech)
            capInput, capOutput = CalculateGateCapacitance(INV, 1, self.w_TSV_n[1], self.w_TSV_p[1],
                                                            g.tech.featureSize * MAX_TRANSISTOR_HEIGHT, g.tech)

            c_load = capInput + capOutput
            c_intrinsic = (CalculateDrainCap(self.w_TSV_p[0], PMOS,
                                            g.tech.featureSize * MAX_TRANSISTOR_HEIGHT, g.tech) +
                          CalculateDrainCap(self.w_TSV_n[0], NMOS,
                                           g.tech.featureSize * MAX_TRANSISTOR_HEIGHT, g.tech))
            tf = rd * (c_intrinsic + c_load)

            rampInput = _rampInput
            this_delay, rampOutput = horowitz(tf, beta, rampInput)
            delay += this_delay

            Vdd = g.tech.vdd
            _dynamicEnergy = (c_load + c_intrinsic) * Vdd * Vdd

            for i in range(1, self.num_gates - 1):
                rd = CalculateOnResistance(self.w_TSV_n[i], NMOS, g.inputParameter.temperature, g.tech)
                capInput, capOutput = CalculateGateCapacitance(INV, 1, self.w_TSV_n[i+1], self.w_TSV_p[i+1],
                                                                g.tech.featureSize * MAX_TRANSISTOR_HEIGHT, g.tech)

                c_load = capInput + capOutput
                c_intrinsic = (CalculateDrainCap(self.w_TSV_p[i], PMOS,
                                                g.tech.featureSize * MAX_TRANSISTOR_HEIGHT, g.tech) +
                              CalculateDrainCap(self.w_TSV_n[i], NMOS,
                                               g.tech.featureSize * MAX_TRANSISTOR_HEIGHT, g.tech))

                tf = rd * (c_intrinsic + c_load)
                rampInput = rampOutput
                this_delay, rampOutput = horowitz(tf, beta, rampInput)
                delay += this_delay

                _dynamicEnergy += (c_load + c_intrinsic) * Vdd * Vdd

            # add delay of final inverter that drives the TSV
            i = self.num_gates - 1
            c_load = self.C_load_TSV

            rd = CalculateOnResistance(self.w_TSV_n[i], NMOS, g.inputParameter.temperature, g.tech)
            c_intrinsic = (CalculateDrainCap(self.w_TSV_p[i], PMOS,
                                            g.tech.featureSize * MAX_TRANSISTOR_HEIGHT, g.tech) +
                          CalculateDrainCap(self.w_TSV_n[i], NMOS,
                                           g.tech.featureSize * MAX_TRANSISTOR_HEIGHT, g.tech))

            R_TSV_out = self.res
            tf = rd * (c_intrinsic + c_load) + R_TSV_out * c_load / 2

            rampInput = rampOutput
            this_delay, rampOutput = horowitz(tf, beta, rampInput)
            delay += this_delay

            _dynamicEnergy += (c_load + c_intrinsic) * Vdd * Vdd

            _latency = delay
        else:
            rampInput = _rampInput
            c_load = self.cap
            R_TSV_out = self.res
            tf = R_TSV_out * c_load / 2
            Vdd = g.tech.vdd

            this_delay, rampOutput = horowitz(tf, beta, rampInput)
            delay += this_delay

            _dynamicEnergy += c_load * Vdd * Vdd

            _latency = delay

        return _dynamicEnergy, _latency
