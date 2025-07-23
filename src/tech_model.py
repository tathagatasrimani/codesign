import logging
from abc import ABC, abstractmethod
import sympy as sp

logger = logging.getLogger(__name__)

class TechModel(ABC):
    def __init__(self, tech_type, model_cfg, base_params):
        self.tech_type = tech_type
        self.model_cfg = model_cfg
        self.base_params = base_params
        self.constraints = []
        self.init_physical_constants()
        self.init_tech_specific_constants()
        self.init_transistor_equations()

    def init_physical_constants(self):
        self.q = 1.6e-19  # electron charge (C)
        self.e_0 = 8.854e-12  # permittivity of free space (F/m)
        self.e_si = 11.9*8.854e-12  # permittivity of silicon (F/m)
        self.n = 1.0
        self.K = 1.38e-23  # Boltzmann constant (J/K)
        self.T = 300  # Temperature (K)
        self.m_0 = 9.109e-31  # electron mass (kg)
        self.h = 6.626e-34  # planck's constant (J*s)

    @abstractmethod
    def init_tech_specific_constants(self):
        pass

    @abstractmethod
    def init_transistor_equations(self):
        # set up generic stuff here

        # mock area and latency scaling for experimental purposes
        self.area_scale = (self.base_params.W * self.base_params.L).subs(self.base_params.tech_values) / (self.base_params.W * self.base_params.L)
        logger.info(f"area scale: {self.area_scale.subs(self.base_params.tech_values)}")
        self.latency_scale = 1/self.area_scale

        # interconnect model
        # assume for now that wire width and thickness are both L (same as gate length)
        # and spacing between wire and dielectric is 2L
        self.dist = 2*self.base_params.L
        self.wire_dim = 2*self.base_params.L

        self.m1_Rsq = (self.base_params.m1_rho * 1e-6) / (self.wire_dim**2) # resistance per square (Ohm*um)
        self.m2_Rsq = (self.base_params.m2_rho * 1e-6) / (self.wire_dim**2) # resistance per square (Ohm*um)
        self.m3_Rsq = (self.base_params.m3_rho * 1e-6) / (self.wire_dim**2) # resistance per square (Ohm*um)

        self.m1_Csq = (self.base_params.L * self.base_params.m1_k * self.e_0 * 1e-6) / (self.dist) # capacitance per square (F/um)
        self.m2_Csq = (self.base_params.L * self.base_params.m2_k * self.e_0 * 1e-6) / (self.dist) # capacitance per square (F/um)
        self.m3_Csq = (self.base_params.L * self.base_params.m3_k * self.e_0 * 1e-6) / (self.dist) # capacitance per square (F/um)

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

        self.wire_len = 100 #um

        self.C_wire = self.wire_parasitics["C"]["metal1"] * self.wire_len
        self.R_wire = self.wire_parasitics["R"]["metal1"] * self.wire_len

        self.V_th_eff = self.base_params.V_th
        self.u_n_eff = self.base_params.u_n

    @abstractmethod
    def apply_base_parameter_effects(self):
        pass

    @abstractmethod
    def apply_additional_effects(self):
        pass

    @abstractmethod
    def create_constraints(self, dennard_scaling_type="constant_field"):
        self.constraints = []
        # generic constraints
        self.constraints.append(self.base_params.V_dd >= self.V_th_eff)
        self.constraints.append(self.base_params.V_dd >= self.base_params.V_th)
        if self.V_th_eff != self.base_params.V_th:
            self.constraints.append(self.V_th_eff >= 0)
        self.constraints.append(self.base_params.V_dd <= 5)

        if self.base_params.f in self.base_params.tech_values:
            self.constraints.append(self.delay <= 1e9/self.base_params.f)
        self.constraints.append(self.I_off/(self.base_params.W*self.base_params.L) <= 100e-9 / (1e-6 * 1e-6))

        self.constraints.append(self.base_params.W / self.base_params.L <= 50)
        self.constraints.append(self.base_params.W / self.base_params.L >= 0.2)

        # wire material constraints
        self.constraints.append(self.base_params.m1_rho >= 2e-8)
        self.constraints.append(self.base_params.m1_k >= 2)
        self.constraints.append(self.base_params.m2_rho >= 2e-8)
        self.constraints.append(self.base_params.m2_k >= 2)
        self.constraints.append(self.base_params.m3_rho >= 2e-8)
        self.constraints.append(self.base_params.m3_k >= 2)

        if self.model_cfg["effects"]["high_k_gate"]:
            self.constraints.append(self.base_params.k_gate >= 2)
            self.constraints.append(self.base_params.k_gate <= 25)


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
            self.constraints.append(self.base_params.tox <= self.base_params.tox.subs(self.base_params.tech_values))
            self.constraints.append(self.base_params.L <= self.base_params.L.subs(self.base_params.tech_values))
            self.constraints.append(self.base_params.W <= self.base_params.W.subs(self.base_params.tech_values))
            self.constraints.append(self.base_params.V_dd <= self.base_params.V_dd.subs(self.base_params.tech_values))
            self.constraints.append(self.V_th_eff <= self.V_th_eff.subs(self.base_params.tech_values))
            self.constraints.append(sp.Eq(self.base_params.W/self.base_params.W.subs(self.base_params.tech_values), self.base_params.L/self.base_params.L.subs(self.base_params.tech_values)))
            self.constraints.append(sp.Eq(self.base_params.V_dd/self.base_params.V_dd.subs(self.base_params.tech_values) , self.base_params.L/self.base_params.L.subs(self.base_params.tech_values))) # lateral electric field scaling
            self.constraints.append(sp.Eq(self.base_params.V_dd/self.base_params.V_dd.subs(self.base_params.tech_values) , self.base_params.tox/self.base_params.tox.subs(self.base_params.tech_values))) # lateral electric field scaling
        
