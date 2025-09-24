import logging
from abc import ABC, abstractmethod
import sympy as sp

from src.sim_util import symbolic_convex_max, symbolic_min

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
        
    def init_scale_factors(self, max_parallel_factor):
        self.max_delay_scale = 1/max_parallel_factor
        self.max_power_scale = max_parallel_factor
        self.capped_delay_scale = symbolic_convex_max(self.max_delay_scale, self.base_params.latency_scale)
        self.capped_power_scale = symbolic_min(self.max_power_scale, self.base_params.area_scale)

    @abstractmethod
    def init_tech_specific_constants(self):
        pass

    @abstractmethod
    def init_transistor_equations(self):
        # set up generic stuff here

        # interconnect model
        # assume for now that wire width and thickness are both L (same as gate length)
        # and spacing between wire and dielectric is 2L
        self.dist = 3*self.base_params.L
        self.wire_dim = 0.5*self.base_params.L

        self.m1_Rsq = (self.base_params.m1_rho) / (self.wire_dim**2) # resistance per square (Ohm/m)
        self.m2_Rsq = (self.base_params.m2_rho) / (self.wire_dim**2) # resistance per square (Ohm/m)
        self.m3_Rsq = (self.base_params.m3_rho) / (self.wire_dim**2) # resistance per square (Ohm/m)

        self.m1_Csq = (self.wire_dim * self.base_params.m1_k * self.e_0) / (self.dist) # capacitance per square (F/m)
        self.m2_Csq = (self.wire_dim * self.base_params.m2_k * self.e_0) / (self.dist) # capacitance per square (F/m)
        self.m3_Csq = (self.wire_dim * self.base_params.m3_k * self.e_0) / (self.dist) # capacitance per square (F/m)

        self.wire_parasitics = {
            "R": {
                "metal1": self.m1_Rsq,
                "metal2": self.m2_Rsq,
                "metal3": self.m3_Rsq,
            },
            "C": {
                "metal1": self.m1_Csq,
                "metal2": self.m2_Csq,
                "metal3": self.m3_Csq,
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
        if self.model_cfg["effects"]["area_and_latency_scaling"]:
            if self.model_cfg["effects"]["max_parallel_en"]:
                MAX_PARALLEL = self.model_cfg["effects"]["max_parallel_val"]
                self.delay = self.delay * symbolic_convex_max(self.base_params.latency_scale, 1/MAX_PARALLEL)
                self.P_pass_inv = self.P_pass_inv * symbolic_min(self.base_params.area_scale, MAX_PARALLEL)
            else:
                self.delay = self.base_params.latency_scale
                self.P_pass_inv = self.base_params.area_scale

    @abstractmethod
    def create_constraints(self, dennard_scaling_type="constant_field"):
        self.constraints = []
        # generic constraints
        self.constraints.append(self.base_params.V_dd >= self.V_th_eff)
        self.constraints.append(self.base_params.V_dd >= self.base_params.V_th)
        if self.V_th_eff != self.base_params.V_th:
            self.constraints.append(self.V_th_eff >= 0)
        self.constraints.append(self.base_params.V_dd <= 5)

        #if self.base_params.f in self.base_params.tech_values:
        #    self.constraints.append(self.delay <= 1e9/self.base_params.f)
        #self.constraints.append(self.base_params.f <= 5e9)
        self.constraints.append(self.I_off/(self.base_params.W) <= 100e-9 / (1e-6))


        delta_V_th_process = 20e-3 # account for 20mV variation in Vth
        V_th_min = self.V_th_eff - delta_V_th_process
        V_th_max = self.V_th_eff + delta_V_th_process
        max_I_off = self.I_off.subs({self.V_th_eff: V_th_min})
        self.constraints.append(max_I_off/(self.base_params.W) <= 100e-9 / (1e-6))
        self.constraints.append(V_th_min >= 0)
        self.constraints.append(V_th_max <= self.base_params.V_dd)

        self.constraints.append(self.base_params.W / self.base_params.L <= 50)
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


        if self.model_cfg["scaling_mode"] == "dennard":
            if dennard_scaling_type == "constant_field":
                self.constraints.append(sp.Eq(self.base_params.W/self.base_params.tech_values[self.base_params.W], 1/self.base_params.alpha_dennard))
                self.constraints.append(sp.Eq(self.base_params.L/self.base_params.tech_values[self.base_params.L], 1/self.base_params.alpha_dennard))
                self.constraints.append(sp.Eq(self.base_params.V_dd/self.base_params.tech_values[self.base_params.V_dd], 1/self.base_params.alpha_dennard))
                self.constraints.append(sp.Eq(self.V_th_eff/self.base_params.tech_values[self.V_th_eff], 1/self.base_params.alpha_dennard))
                self.constraints.append(sp.Eq(self.Cox/self.base_params.tech_values[self.Cox], self.base_params.alpha_dennard))
                if self.initial_alpha is None and self.base_params.tech_values[self.base_params.alpha_dennard] != 1:
                    self.initial_alpha = self.base_params.tech_values[self.base_params.alpha_dennard]
            else:
                self.constraints.append(sp.Eq(self.base_params.alpha_dennard, self.initial_alpha))
                self.constraints.append(sp.Eq(self.base_params.W/self.base_params.tech_values[self.base_params.W], 1/self.base_params.alpha_dennard))
                self.constraints.append(sp.Eq(self.base_params.L/self.base_params.tech_values[self.base_params.L], 1/self.base_params.alpha_dennard))
                self.constraints.append(sp.Eq(self.base_params.V_dd, self.base_params.tech_values[self.base_params.V_dd]))
                self.constraints.append(sp.Eq(self.V_th_eff, self.base_params.tech_values[self.V_th_eff]))
                #self.constraints.append(sp.Eq(self.base_params.V_dd/self.base_params.tech_values[self.base_params.V_dd], self.base_params.epsilon_dennard/self.base_params.alpha_dennard))
                #self.constraints.append(sp.Eq(self.V_th_eff/self.base_params.tech_values[self.V_th_eff], self.base_params.epsilon_dennard/self.base_params.alpha_dennard))
                self.constraints.append(sp.Eq(self.Cox/self.base_params.tech_values[self.Cox], self.base_params.alpha_dennard))

            #elif dennard_scaling_type == "generalized":
            #    self.constraints.append(sp.Eq(self.base_params.alpha_dennard, 1))
        elif self.model_cfg["scaling_mode"] == "dennard_implicit":
            self.constraints.append(self.base_params.tox <= self.base_params.tox.xreplace(self.base_params.tech_values))
            self.constraints.append(self.base_params.L <= self.base_params.L.xreplace(self.base_params.tech_values))
            self.constraints.append(self.base_params.W <= self.base_params.W.xreplace(self.base_params.tech_values))
            self.constraints.append(self.base_params.V_dd <= self.base_params.V_dd.xreplace(self.base_params.tech_values))
            self.constraints.append(self.V_th_eff <= self.V_th_eff.xreplace(self.base_params.tech_values))
            self.constraints.append(sp.Eq(self.base_params.W/self.base_params.W.xreplace(self.base_params.tech_values), self.base_params.L/self.base_params.L.xreplace(self.base_params.tech_values)))
            self.constraints.append(sp.Eq(self.base_params.V_dd/self.base_params.V_dd.xreplace(self.base_params.tech_values) , self.base_params.L/self.base_params.L.xreplace(self.base_params.tech_values))) # lateral electric field scaling
            self.constraints.append(sp.Eq(self.base_params.V_dd/self.base_params.V_dd.xreplace(self.base_params.tech_values) , self.base_params.tox/self.base_params.tox.xreplace(self.base_params.tech_values))) # lateral electric field scaling
        
