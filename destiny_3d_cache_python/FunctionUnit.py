#Copyright (c) 2015-2016, UT-Battelle, LLC. See LICENSE file in the top-level directory
# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

import math
import formula


class FunctionUnit:
    def __init__(self):
        self.height = 0.0
        self.width = 0.0
        self.area = 0.0
        self.readLatency = 0.0
        self.writeLatency = 0.0
        self.readDynamicEnergy = 0.0
        self.writeDynamicEnergy = 0.0
        self.leakage = 0.0

        # Optional properties (not valid for all the memory cells)
        self.resetLatency = 0.0
        self.setLatency = 0.0
        self.refreshLatency = 0.0
        self.resetDynamicEnergy = 0.0
        self.setDynamicEnergy = 0.0
        self.refreshDynamicEnergy = 0.0
        self.cellReadEnergy = 0.0
        self.cellSetEnergy = 0.0
        self.cellResetEnergy = 0.0

    def PrintProperty(self):
        print("Area =", self.height * 1e6, "um x", self.width * 1e6, "um =", self.area * 1e6, "mm^2")
        print("Timing:")
        print(" -  Read Latency =", self.readLatency * 1e9, "ns")
        print(" - Write Latency =", self.writeLatency * 1e9, "ns")
        print("Power:")
        print(" -  Read Dynamic Energy =", self.readDynamicEnergy * 1e12, "pJ")
        print(" - Write Dynamic Energy =", self.writeDynamicEnergy * 1e12, "pJ")
        print(" - Leakage Power =", self.leakage * 1e3, "mW")

    def logical_effort(self, num_gates_min, g, F, w_n, w_p, C_load,
                      p_to_n_sz_ratio, max_w_nmos, tech):
        num_gates = int(math.log(F) / math.log(formula.fopt))

        # check if num_gates is odd. if so, add 1 to make it even
        num_gates += 1 if (num_gates % 2) else 0
        num_gates = max(num_gates, num_gates_min)

        # recalculate the effective fanout of each stage
        f = pow(F, 1.0 / num_gates)
        i = num_gates - 1
        C_in = C_load / f
        w_n[i] = (1.0 / (1.0 + p_to_n_sz_ratio)) * C_in / formula.CalculateGateCap(1, tech)
        w_n[i] = max(w_n[i], formula.MIN_NMOS_SIZE * tech.featureSize)
        w_p[i] = p_to_n_sz_ratio * w_n[i]

        if w_n[i] > max_w_nmos:
            C_ld = formula.CalculateGateCap((1 + p_to_n_sz_ratio) * max_w_nmos, tech)
            F = g * C_ld / formula.CalculateGateCap(w_n[0] + w_p[0], tech)
            num_gates = int(math.log(F) / math.log(formula.fopt)) + 1
            num_gates += 1 if (num_gates % 2) else 0
            num_gates = max(num_gates, num_gates_min)
            f = pow(F, 1.0 / (num_gates - 1))
            i = num_gates - 1
            w_n[i] = max_w_nmos
            w_p[i] = p_to_n_sz_ratio * w_n[i]

        for i in range(num_gates - 2, 0, -1):
            w_n[i] = max(w_n[i+1] / f, formula.MIN_NMOS_SIZE * tech.featureSize)
            w_p[i] = p_to_n_sz_ratio * w_n[i]

        assert num_gates <= formula.MAX_NUMBER_GATES_STAGE
        return num_gates
