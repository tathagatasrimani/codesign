import logging
from abc import ABC, abstractmethod
import sympy as sp
import math
import numpy as np

from src.sim_util import symbolic_convex_max, symbolic_min, xreplace_safe

logger = logging.getLogger(__name__)

class TechModel(ABC):
    def __init__(self, model_cfg, base_params):
        self.model_cfg = model_cfg
        self.base_params = base_params
        self.constraints = []
        self.param_db = {}
        self.capped_delay_scale = 1
        self.capped_power_scale = 1
        self.init_physical_constants()
        self.init_tech_specific_constants()
        self.init_transistor_equations()

    def init_physical_constants(self):
        self.q = 1.6e-19  # electron charge (C)
        self.e_0 = 8.854e-12  # permittivity of free space (F/m)
        self.e_si = 11.9*self.e_0  # permittivity of silicon (F/m)
        self.e_sio2 = 3.9*self.e_0  # permittivity of silicon dioxide (F/m)
        self.n = 1.0
        self.K = 1.38e-23  # Boltzmann constant (J/K)
        self.T = 300  # Temperature (K)
        self.m_0 = 9.109e-31  # electron mass (kg)
        self.m_e = self.m_0
        self.h = 6.626e-34  # planck's constant (J*s)
        self.h_bar = self.h / (2*math.pi)
        self.V_T = self.K*self.T/self.q # thermal voltage (V)
        
    def init_scale_factors(self, max_speedup_factor, max_area_increase_factor):
        self.max_speedup_factor = max_speedup_factor
        self.max_area_increase_factor = max_area_increase_factor
        
        self.max_delay_scale = 1/max_speedup_factor
        self.max_power_scale = self.max_area_increase_factor
        assert max_area_increase_factor > 1, "max_area_increase_factor must be greater than 1"
        self.latency_scale_exp = np.log(max_speedup_factor) / np.log(self.max_area_increase_factor)
        # base_params.latency_scale and base_params.area_scale are set by the ratio of the starting cell area to current cell area
        #self.capped_delay_scale = symbolic_convex_max(self.max_delay_scale, 1 - self.latency_scale_slope * self.base_params.area_scale) # <= 1 (delay = delay_0 * capped_delay_scale)
        #self.capped_power_scale = symbolic_min(self.max_area_increase_factor, self.base_params.area_scale) # >= 1 (power = power_0 * capped_power_scale)
        self.capped_delay_scale_total = symbolic_convex_max(1/(self.base_params.area_scale**self.latency_scale_exp), self.max_delay_scale)
        self.capped_power_scale_total = symbolic_min(self.max_area_increase_factor, self.base_params.area_scale)

        area_scale_remaining = self.max_area_increase_factor / xreplace_safe(self.capped_power_scale_total, self.base_params.tech_values)
        cur_area_scale = xreplace_safe((self.base_params.W * self.base_params.L), self.base_params.tech_values)/(self.base_params.W * self.base_params.L)
        self.capped_power_scale = symbolic_min(area_scale_remaining, cur_area_scale)

        speedup_remaining = max_speedup_factor * xreplace_safe(self.capped_delay_scale_total, self.base_params.tech_values)
        if area_scale_remaining <= 1:
            cur_latency_scale_exp = 0
        else:
            cur_latency_scale_exp = np.log(speedup_remaining) / np.log(area_scale_remaining)
        self.capped_delay_scale = symbolic_convex_max(1/speedup_remaining, 1/(cur_area_scale**cur_latency_scale_exp))
        logger.info(f"max_speedup_factor: {self.max_speedup_factor}, max_area_increase_factor: {self.max_area_increase_factor}, area_scale_remaining: {area_scale_remaining}, speedup_remaining: {speedup_remaining}, cur_latency_scale_exp: {cur_latency_scale_exp}")
        self.capped_energy_scale = self.capped_delay_scale * self.capped_power_scale


    @abstractmethod
    def init_tech_specific_constants(self):
        pass

    @abstractmethod
    def init_transistor_equations(self):
        # set up generic stuff here

        # interconnect model
        # assume for now that wire width and thickness are both L (same as gate length)
        # and spacing between wire and dielectric is 2L
        self.dist = self.base_params.L
        self.wire_width = self.base_params.L
        self.wire_thick = self.base_params.L * 2

        layer_scale_width = [1.0, 1.2, 1.4, 1.8, 2.5, 3.5, 5.0, 7.0, 10.0, 15.0]
        layer_scale_thick = [1.0, 1.1, 1.2, 1.5, 1.9, 2.5, 3.5, 4.5, 5.5, 6.0]
        spacing_factor = [1.0, 1.0, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

        self.m1_Rsq = (self.base_params.m1_rho) / (self.wire_width*layer_scale_width[0]*self.wire_thick*layer_scale_thick[0]) # resistance per square (Ohm/m)
        self.m2_Rsq = (self.base_params.m2_rho) / (self.wire_width*layer_scale_width[1]*self.wire_thick*layer_scale_thick[1]) # resistance per square (Ohm/m)
        self.m3_Rsq = (self.base_params.m3_rho) / (self.wire_width*layer_scale_width[2]*self.wire_thick*layer_scale_thick[2]) # resistance per square (Ohm/m)
        self.m4_Rsq = (self.base_params.m4_rho) / (self.wire_width*layer_scale_width[3]*self.wire_thick*layer_scale_thick[3]) # resistance per square (Ohm/m)
        self.m5_Rsq = (self.base_params.m5_rho) / (self.wire_width*layer_scale_width[4]*self.wire_thick*layer_scale_thick[4]) # resistance per square (Ohm/m)
        self.m6_Rsq = (self.base_params.m6_rho) / (self.wire_width*layer_scale_width[5]*self.wire_thick*layer_scale_thick[5]) # resistance per square (Ohm/m)
        self.m7_Rsq = (self.base_params.m7_rho) / (self.wire_width*layer_scale_width[6]*self.wire_thick*layer_scale_thick[6]) # resistance per square (Ohm/m)
        self.m8_Rsq = (self.base_params.m8_rho) / (self.wire_width*layer_scale_width[7]*self.wire_thick*layer_scale_thick[7]) # resistance per square (Ohm/m)
        self.m9_Rsq = (self.base_params.m9_rho) / (self.wire_width*layer_scale_width[8]*self.wire_thick*layer_scale_thick[8]) # resistance per square (Ohm/m)
        self.m10_Rsq = (self.base_params.m10_rho) / (self.wire_width*layer_scale_width[9]*self.wire_thick*layer_scale_thick[9]) # resistance per square (Ohm/m)

        self.m1_Csq = (self.wire_thick*layer_scale_thick[0] * self.base_params.m1_k * self.e_0) / (self.dist*spacing_factor[0]) # capacitance per square (F/m)
        self.m2_Csq = (self.wire_thick*layer_scale_thick[1] * self.base_params.m2_k * self.e_0) / (self.dist*spacing_factor[1]) # capacitance per square (F/m)
        self.m3_Csq = (self.wire_thick*layer_scale_thick[2] * self.base_params.m3_k * self.e_0) / (self.dist*spacing_factor[2]) # capacitance per square (F/m)
        self.m4_Csq = (self.wire_thick*layer_scale_thick[3] * self.base_params.m4_k * self.e_0) / (self.dist*spacing_factor[3]) # capacitance per square (F/m)
        self.m5_Csq = (self.wire_thick*layer_scale_thick[4] * self.base_params.m5_k * self.e_0) / (self.dist*spacing_factor[4]) # capacitance per square (F/m)
        self.m6_Csq = (self.wire_thick*layer_scale_thick[5] * self.base_params.m6_k * self.e_0) / (self.dist*spacing_factor[5]) # capacitance per square (F/m)
        self.m7_Csq = (self.wire_thick*layer_scale_thick[6] * self.base_params.m7_k * self.e_0) / (self.dist*spacing_factor[6]) # capacitance per square (F/m)
        self.m8_Csq = (self.wire_thick*layer_scale_thick[7] * self.base_params.m8_k * self.e_0) / (self.dist*spacing_factor[7]) # capacitance per square (F/m)
        self.m9_Csq = (self.wire_thick*layer_scale_thick[8] * self.base_params.m9_k * self.e_0) / (self.dist*spacing_factor[8]) # capacitance per square (F/m)
        self.m10_Csq = (self.wire_thick*layer_scale_thick[9] * self.base_params.m10_k * self.e_0) / (self.dist*spacing_factor[9]) # capacitance per square (F/m)

        self.wire_parasitics = {
            "R": {
                "metal1": self.m1_Rsq,
                "metal2": self.m2_Rsq,
                "metal3": self.m3_Rsq,
                "metal4": self.m4_Rsq,
                "metal5": self.m5_Rsq,
                "metal6": self.m6_Rsq,
                "metal7": self.m7_Rsq,
                "metal8": self.m8_Rsq,
                "metal9": self.m9_Rsq,
                "metal10": self.m10_Rsq,
            },
            "C": {
                "metal1": self.m1_Csq,
                "metal2": self.m2_Csq,
                "metal3": self.m3_Csq,
                "metal4": self.m4_Csq,
                "metal5": self.m5_Csq,
                "metal6": self.m6_Csq,
                "metal7": self.m7_Csq,
                "metal8": self.m8_Csq,
                "metal9": self.m9_Csq,
                "metal10": self.m10_Csq,
            }
        }

        self.wire_len = 20*self.base_params.L

        self.C_wire = self.wire_parasitics["C"]["metal1"] * self.wire_len
        self.R_wire = self.wire_parasitics["R"]["metal1"] * self.wire_len

        self.V_th_eff = self.base_params.V_th
        self.u_n_eff = self.base_params.u_n

    @abstractmethod
    def config_param_db(self):
        self.param_db["tox"] = self.base_params.tox

    @abstractmethod
    def apply_base_parameter_effects(self):
        pass

    @abstractmethod
    def apply_additional_effects(self):
        self.delay_var = sp.symbols("delay_var")
        self.base_params.tech_values[self.delay_var] = xreplace_safe(self.delay, self.base_params.tech_values)
        """if self.model_cfg["effects"]["area_and_latency_scaling"]:
            if self.model_cfg["effects"]["max_parallel_en"]:
                MAX_PARALLEL = self.model_cfg["effects"]["max_parallel_val"]
                self.delay = self.delay * symbolic_convex_max(self.base_params.latency_scale, 1/MAX_PARALLEL)
                self.P_pass_inv = self.P_pass_inv * symbolic_min(self.base_params.area_scale, MAX_PARALLEL)
            else:
                self.delay = self.delay * self.base_params.latency_scale
                self.P_pass_inv = self.P_pass_inv * self.base_params.area_scale"""

    @abstractmethod
    def create_constraints(self, dennard_scaling_type="constant_field"):
        self.constraints = []
        # generic constraints
        self.constraints.append(self.delay_var >= self.delay)
        self.constraints.append(self.base_params.V_dd >= self.V_th_eff)
        self.constraints.append(self.base_params.V_dd >= self.base_params.V_th)
        if self.V_th_eff != self.base_params.V_th:
            self.constraints.append(self.V_th_eff >= 0)
        self.constraints.append(self.base_params.V_dd <= 5)

        self.constraints.append(self.I_off/(self.base_params.W) <= 100e-9 / (1e-6))


        #delta_V_th_process = 20e-3 # account for 20mV variation in Vth
        #V_th_min = self.V_th_eff - delta_V_th_process
        #V_th_max = self.V_th_eff + delta_V_th_process
        #max_I_off = self.I_off.subs({self.V_th_eff: V_th_min})
        #self.constraints.append(max_I_off/(self.base_params.W) <= 100e-9 / (1e-6))
        #self.constraints.append(V_th_min >= 0)
        #self.constraints.append(V_th_max <= self.base_params.V_dd)

        #self.constraints.append(self.base_params.W / self.base_params.L <= 50)
        self.constraints.append(self.base_params.W / self.base_params.L >= 1)

        # wire material constraints
        self.constraints.append(self.base_params.m1_rho >= 2e-8)
        self.constraints.append(self.base_params.m1_k >= 2)
        self.constraints.append(self.base_params.m2_rho >= 2e-8)
        self.constraints.append(self.base_params.m2_k >= 2)
        self.constraints.append(self.base_params.m3_rho >= 2e-8)
        self.constraints.append(self.base_params.m3_k >= 2)

        if self.model_cfg["effects"]["high_k_gate"]:
            self.constraints.append(self.base_params.k_gate >= 2)
            self.constraints.append(self.base_params.k_gate <= 20)
        
        self.constraints.append(self.base_params.u_n <= 0.1)