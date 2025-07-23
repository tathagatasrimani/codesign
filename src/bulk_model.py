import logging
from src.tech_model import TechModel
from src.sim_util import symbolic_convex_max, symbolic_convex_min
import math
from sympy import symbols, ceiling, expand, exp, Abs


logger = logging.getLogger(__name__)

class BulkModel(TechModel):
    def __init__(self, tech_type, model_cfg, base_params):
        super().__init__(tech_type, model_cfg, base_params)

    def init_tech_specific_constants(self):
        self.phi_b = 3.1  # Schottky barrier height (eV)
        self.m_ox = 0.5*self.m_0  # effective mass of electron in oxide (g)
        self.L_critical = 0.5e-6
        self.m = 2

        # velocity saturation parameters
        self.alpha_L = 2

        # channel length modulation parameters
        self.lambda_L = 0

        # mobility degradation parameters
        self.theta_1 = 1
        self.theta_2 = 1

        # gate tunneling parameters
        self.A = ((self.q)**3) / (8*math.pi*self.h*self.phi_b*self.m_ox)
        self.B = (8*math.pi*(2*self.m_ox)**(1/2) * (self.phi_b*self.q)**(3/2)) / (3*self.q*self.h)

        # GIDL parameters
        self.A_GIDL = 1e-12
        self.B_GIDL = 2.3e9
        self.E_GIDL = 0.8

        # inverter diffusion capacitance constant
        self.gamma_diff = 1.0

    def init_transistor_equations(self):
        super().init_transistor_equations()

        # set up some generic variables
        self.e_ox = self.base_params.k_gate*self.e_0  if self.model_cfg["effects"]["high_k_gate"] else 3.9*self.e_0
        self.Cox = self.e_ox/self.base_params.tox
        self.u_p = self.base_params.u_n/2.5  # Hole mobility for PMOS, hardcoded for now.

        self.u_p_eff = self.u_p

        self.A_gate = self.base_params.W * self.base_params.L
        self.C_gate = self.Cox * self.A_gate

        self.apply_base_parameter_effects()

        # drive current equations
        self.I_d_nmos = self.u_n_eff*self.Cox*self.base_params.W*(Abs(self.base_params.V_dd-self.V_th_eff))**self.alpha_L * (1+self.lambda_L*self.base_params.V_dd) / (2*self.base_params.L)
        self.I_d_pmos = self.I_d_nmos * self.u_p_eff / self.u_n_eff

        self.R_avg_inv = self.base_params.V_dd / ((self.I_d_nmos + self.I_d_pmos)/2)

        # transistor delay equations
        self.C_diff = self.gamma_diff * self.C_gate
        self.C_load = self.C_gate # gate cap
        logger.info(f"C_load: {self.C_load}")
        if self.model_cfg["delay_parasitics"] == "all":
            self.delay = (self.R_avg_inv * (self.C_diff + self.C_wire/2) + (self.R_avg_inv + self.R_wire) * (self.C_wire/2 + self.C_load)) * 1e9  # ns
        elif self.model_cfg["delay_parasitics"] == "Csq only":
            self.delay = (self.R_avg_inv * (self.C_diff + self.C_wire/2) + (self.R_avg_inv) * (self.C_wire/2 + self.C_load)) * 1e9  # ns
        elif self.model_cfg["delay_parasitics"] == "const":
            self.delay = self.R_avg_inv * (self.C_diff + self.C_load + 0.3e-15 * 100) * 1e9  # ns
        else:
            self.delay = self.R_avg_inv * (self.C_load + self.C_diff) * 1e9

        # active energy
        self.E_act_inv = (0.5*self.C_load*self.base_params.V_dd*self.base_params.V_dd) * 1e9  # nJ

        # gate tunneling current (Fowler-Nordheim and WKB)
        # minimums are to avoid exponential explosion in solver. Normal values in exponent are negative.
        # gate tunneling
        self.V_ox = self.base_params.V_dd - self.V_th_eff
        self.E_ox = Abs(self.V_ox/self.base_params.tox)
        logger.info(f"B: {self.B}, A: {self.A}, t_ox: {self.base_params.tox.subs(self.base_params.tech_values)}, E_ox: {self.E_ox.subs(self.base_params.tech_values)}, intermediate: {(1-(1-self.V_ox/self.phi_b)**3/2).subs(self.base_params.tech_values)}")
        self.FN_term = self.A_gate * self.A * self.E_ox**2 * (exp(symbolic_convex_min(10, -self.B/self.E_ox)))
        self.WKB_term = self.A_gate * self.A * self.E_ox**2 * (exp(symbolic_convex_min(10, -self.B*(1-(1-self.V_ox/self.phi_b)**3/2)/self.E_ox)))
        self.I_tunnel = self.FN_term + self.WKB_term
        logger.info(f"I_tunnel: {self.I_tunnel.subs(self.base_params.tech_values)}")

        # GIDL current
        self.I_GIDL = self.A_GIDL * ((self.base_params.V_dd - self.E_GIDL)/(3*self.base_params.tox)) * exp(symbolic_convex_min(10, -3*self.base_params.tox*self.B_GIDL / (self.base_params.V_dd - self.E_GIDL))) # simplified from BSIM
        logger.info(f"I_GIDL: {self.I_GIDL.subs(self.base_params.tech_values)}")

        # subthreshold current
        self.V_T = self.K*self.T/self.q
        self.I_off_nmos = self.u_n_eff*self.Cox*(self.base_params.W/self.base_params.L)*(self.V_T)**2*exp(-self.V_th_eff/(self.n*self.V_T))
        self.I_off_pmos = self.I_off_nmos * self.u_p_eff / self.u_n_eff
        self.I_off = self.I_off_nmos + self.I_off_pmos # 2 for both NMOS and PMOS
        self.P_pass_inv = self.I_off*self.base_params.V_dd # base leakage power

        self.apply_additional_effects()

    def apply_base_parameter_effects(self):
        if self.model_cfg["effects"]["velocity_saturation"]:
            # Sakurai alpha-power law
            self.alpha_L =2 - 1/(1 + (self.base_params.L/self.L_critical)**self.m)
        if self.model_cfg["effects"]["channel_length_modulation"]:
            self.lambda_L = 0.1
        if self.model_cfg["effects"]["mobility_degradation"]:
            self.u_n_eff = self.base_params.u_n / (1+self.theta_1*(self.base_params.V_dd-self.V_th_eff)/(1+self.theta_2*(self.base_params.V_dd-self.V_th_eff)))
            self.u_p_eff = self.u_n_eff / 2.5
        if self.model_cfg["effects"]["subthreshold_slope_effect"]:
            self.n += (self.e_si/self.e_ox) * (self.base_params.tox/self.base_params.L)**(1/2)
        if self.model_cfg["effects"]["DIBL"]:
            self.V_th_eff -= self.base_params.V_dd * exp(-self.base_params.L/((self.e_si/self.e_ox) * self.base_params.tox)) # approximate DIBL effect

    def apply_additional_effects(self):
        if self.model_cfg["effects"]["gate_tunneling"]:
            self.I_off += self.I_tunnel*2 # 2 for both NMOS and PMOS
            self.P_pass_inv = self.I_off * self.base_params.V_dd
        if self.model_cfg["effects"]["GIDL"]:
            self.I_off += self.I_GIDL*2 # 2 for both NMOS and PMOS
            self.P_pass_inv = self.I_off * self.base_params.V_dd
        if self.model_cfg["effects"]["area_and_latency_scaling"]:
            self.delay = self.delay * self.latency_scale
            self.P_pass_inv = self.P_pass_inv * self.area_scale
            self.wire_len = self.wire_len * self.latency_scale

    def create_constraints(self, dennard_scaling_type="constant_field"):
        super().create_constraints(dennard_scaling_type)

        self.constraints.append(self.V_ox >= 0)