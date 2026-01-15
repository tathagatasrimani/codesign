import logging
from src.hardware_model.tech_models.tech_model_base import TechModel
from src.sim_util import symbolic_convex_max, symbolic_min, custom_cosh, custom_exp, xreplace_safe
import math
from sympy import symbols, ceiling, expand, exp, Abs, cosh, log, tan
import sympy as sp
from scipy.constants import k, e, epsilon_0, hbar, m_e
from src.inverse_pass.constraint import Constraint
from src.hardware_model.tech_models.tech_codesign_v0 import tech_codesign_v0

logger = logging.getLogger(__name__)

class MVSGeneralModel(TechModel):
    def __init__(self, model_cfg, base_params):
        super().__init__(model_cfg, base_params)

    def init_tech_specific_constants(self):
        super().init_tech_specific_constants()

        # gate tunneling parameters
        self.phi_b = 3.1  # Schottky barrier height (eV)
        self.m_ox = 0.5*self.m_0  # effective mass of electron in oxide (g)
        self.A = ((self.q)**3) / (8*math.pi*self.h*self.phi_b*self.m_ox)
        self.B = (8*math.pi*(2*self.m_ox)**(1/2) * (self.phi_b*self.q)**(3/2)) / (3*self.q*self.h)

    def get_gate_leakage_current(self, V_dd, V_th, tox, A_gate, tech_values):
        # gate tunneling current (Fowler-Nordheim and WKB)
        # minimums are to avoid exponential explosion in solver. Normal values in exponent are negative.
        # gate tunneling
        #self.V_ox = symbolic_convex_max(self.base_params.V_dd - self.V_th_eff, self.V_th_eff).xreplace(self.off_state)
        V_ox = V_dd - V_th
        E_ox = Abs(V_ox/tox)
        logger.info(f"B: {self.B}, A: {self.A}, t_ox: {tox.xreplace(tech_values)}, E_ox: {E_ox.xreplace(tech_values)}, intermediate: {(1-(1-V_ox/self.phi_b)**3/2).xreplace(tech_values)}")
        FN_term = A_gate * self.A * E_ox**2 * (custom_exp(-self.B/E_ox))
        WKB_term = A_gate * self.A * E_ox**2 * (custom_exp(-self.B*(1-(1-V_ox/self.phi_b)**3/2)/E_ox))
        I_tunnel = FN_term + WKB_term
        return I_tunnel

    def init_transistor_equations(self):
        super().init_transistor_equations()
        self.area, self.delay, self.Edynamic, self.Pstatic, self.Ieff_n, self.Ieff_p, self.Ioff_n, self.Ioff_p, self.C_load = tech_codesign_v0.final_symbolic_models(
            self.base_params.V_dd, 
            self.base_params.V_th, 
            self.base_params.L, 
            self.base_params.W, 
            self.base_params.beta_p_n, 
            self.base_params.mD_fac, 
            self.base_params.mu_eff_n, 
            self.base_params.mu_eff_p, 
            self.base_params.k_gate, 
            self.base_params.tox,
            self.base_params.eps_semi, 
            self.base_params.tsemi, 
            self.base_params.Lext, 
            self.base_params.Lc, 
            self.base_params.eps_cap, 
            self.base_params.rho_c_n, 
            self.base_params.rho_c_p, 
            self.base_params.Rsh_c_n, 
            self.base_params.Rsh_c_p, 
            self.base_params.Rsh_ext_n, 
            self.base_params.Rsh_ext_p,
            self.base_params.FO, 
            self.base_params.M, 
            self.base_params.a,
            self.R_wire,
            self.C_wire
        )
        logger.info(f"Area: {xreplace_safe(self.area, self.base_params.tech_values):.3e}")
        logger.info(f"Delay: {xreplace_safe(self.delay, self.base_params.tech_values):.3e}")
        logger.info(f"Edynamic: {xreplace_safe(self.Edynamic, self.base_params.tech_values):.3e}")
        logger.info(f"Pstatic: {xreplace_safe(self.Pstatic, self.base_params.tech_values):.3e}")
        logger.info(f"Ieff_n: {xreplace_safe(self.Ieff_n, self.base_params.tech_values):.3e}")
        logger.info(f"Ieff_p: {xreplace_safe(self.Ieff_p, self.base_params.tech_values):.3e}")
        logger.info(f"Ioff_n: {xreplace_safe(self.Ioff_n, self.base_params.tech_values):.3e}")
        logger.info(f"Ioff_p: {xreplace_safe(self.Ioff_p, self.base_params.tech_values):.3e}")
        logger.info(f"Cload: {xreplace_safe(self.C_load, self.base_params.tech_values):.3e}")

        self.E_act_inv = self.Edynamic
        self.P_pass_inv = self.Pstatic
        self.C_diff = self.C_load/self.base_params.FO
        self.R_avg_inv = 2*self.base_params.V_dd/(self.Ieff_n + self.Ieff_p)
        self.A_gate = self.base_params.W * self.base_params.L

        self.Lscale = tech_codesign_v0.get_Lscale(self.base_params.k_gate, self.base_params.eps_semi, self.base_params.tox, self.base_params.tsemi)
        #self.Lscale = self.base_params.Lscale
        logger.info(f"Lscale: {xreplace_safe(self.Lscale, self.base_params.tech_values):.3e}")
        self.n0, self.delta, self.dVt = tech_codesign_v0.symbolic_sce_model_cmg(self.base_params.L, self.base_params.V_th, self.Lscale)
        self.V_th_eff = self.base_params.V_th - self.dVt - self.delta * self.base_params.V_dd
        logger.info(f"n0: {xreplace_safe(self.n0, self.base_params.tech_values):.3e}")
        logger.info(f"delta: {xreplace_safe(self.delta, self.base_params.tech_values):.3e}")
        logger.info(f"dVt: {xreplace_safe(self.dVt, self.base_params.tech_values):.3e}")

        self.I_tunnel = self.get_gate_leakage_current(self.base_params.V_dd, self.V_th_eff, self.base_params.tox, self.A_gate, self.base_params.tech_values)
        logger.info(f"I_tunnel: {self.I_tunnel.xreplace(self.base_params.tech_values)}")
        self.I_sub = (self.Ioff_n + self.Ioff_p)/2
        self.I_off = self.I_sub + self.I_tunnel

        delta_Lgate = 0.15 * self.base_params.L
        worst_case_Lgate = self.base_params.L - delta_Lgate
        _, _, _, _, _, _, self.Ioff_n_worst_case, self.Ioff_p_worst_case, _ = tech_codesign_v0.final_symbolic_models(
            self.base_params.V_dd, 
            self.base_params.V_th, 
            worst_case_Lgate, 
            self.base_params.W, 
            self.base_params.beta_p_n, 
            self.base_params.mD_fac, 
            self.base_params.mu_eff_n, 
            self.base_params.mu_eff_p, 
            self.base_params.k_gate, 
            self.base_params.tox,
            self.base_params.eps_semi, 
            self.base_params.tsemi, 
            self.base_params.Lext, 
            self.base_params.Lc, 
            self.base_params.eps_cap, 
            self.base_params.rho_c_n, 
            self.base_params.rho_c_p, 
            self.base_params.Rsh_c_n, 
            self.base_params.Rsh_c_p, 
            self.base_params.Rsh_ext_n, 
            self.base_params.Rsh_ext_p,
            self.base_params.FO, 
            self.base_params.M, 
            self.base_params.a,
            self.R_wire,
            self.C_wire
        )
        self.I_tunnel_worst_case = self.get_gate_leakage_current(self.base_params.V_dd, self.base_params.V_th, self.base_params.tox, self.A_gate, self.base_params.tech_values)
        Lscale_worst_case = tech_codesign_v0.get_Lscale(self.base_params.k_gate, self.base_params.eps_semi, self.base_params.tox, self.base_params.tsemi)
        n0_worst_case, delta_worst_case, dVt_worst_case = tech_codesign_v0.symbolic_sce_model_cmg(worst_case_Lgate, self.base_params.V_th, Lscale_worst_case)
        self.V_th_eff_worst_case = self.base_params.V_th - dVt_worst_case - delta_worst_case * self.base_params.V_dd
        self.I_tunnel_worst_case = self.get_gate_leakage_current(self.base_params.V_dd, self.V_th_eff_worst_case, self.base_params.tox, self.A_gate, self.base_params.tech_values)
        self.I_sub_worst_case = (self.Ioff_n_worst_case + self.Ioff_p_worst_case)/2
        self.I_sub_worst_case_per_um = self.I_sub_worst_case / (self.base_params.W* 1e6)
        self.I_off_worst_case = self.I_sub_worst_case + self.I_tunnel_worst_case
        self.I_off_worst_case_per_um = self.I_off_worst_case / (self.base_params.W* 1e6)

        self.delay *= 1e+9 # convert to ns

        self.apply_additional_effects()

        self.config_param_db()


    def config_param_db(self):
        self.param_db["I_tunnel_per_um"] = self.I_tunnel / (self.base_params.W* 1e6)
        self.param_db["I_off_per_um"] = self.I_off / (self.base_params.W* 1e6)
        self.param_db["I_on_per_um"] = (self.Ieff_n + self.Ieff_p) / (2*self.base_params.W* 1e6)
        self.param_db["I_sub_per_um"] = self.I_sub / (self.base_params.W* 1e6)
        self.param_db["I_off_worst_case_per_um"] = self.I_off_worst_case_per_um
        self.param_db["I_sub_worst_case_per_um"] = self.I_sub_worst_case_per_um
        self.param_db["V_th_eff_worst_case"] = self.V_th_eff_worst_case
        self.param_db["A_gate"] = self.area
        self.param_db["C_wire"] = self.C_wire
        self.param_db["R_wire"] = self.R_wire
        self.param_db["C_load"] = self.C_load
        super().config_param_db()

    def apply_base_parameter_effects(self):
        super().apply_base_parameter_effects()

    def apply_additional_effects(self):
        super().apply_additional_effects()

    def create_constraints(self, dennard_scaling_type="constant_field"):
        super().create_constraints(dennard_scaling_type)
        self.constraints.append(Constraint(sp.Eq(self.base_params.FO, self.base_params.tech_values[self.base_params.FO], evaluate=False), "FO = tech_values[FO]"))
        self.constraints.append(Constraint(sp.Eq(self.base_params.M, self.base_params.tech_values[self.base_params.M], evaluate=False), "M = tech_values[M]"))
        self.constraints.append(Constraint(sp.Eq(self.base_params.beta_p_n, self.base_params.tech_values[self.base_params.beta_p_n], evaluate=False), "beta_p_n = tech_values[beta_p_n]"))
        self.constraints.append(Constraint(sp.Eq(self.base_params.a, self.base_params.tech_values[self.base_params.a], evaluate=False), "a = tech_values[a]"))
        self.constraints.append(Constraint(sp.Eq(self.base_params.mD_fac, self.base_params.tech_values[self.base_params.mD_fac], evaluate=False), "mD_fac = tech_values[mD_fac]"))
        self.constraints.append(Constraint(sp.Eq(self.base_params.mu_eff_n, self.base_params.tech_values[self.base_params.mu_eff_n], evaluate=False), "mu_eff_n = tech_values[mu_eff_n]"))
        self.constraints.append(Constraint(sp.Eq(self.base_params.mu_eff_p, self.base_params.tech_values[self.base_params.mu_eff_p], evaluate=False), "mu_eff_p = tech_values[mu_eff_p]"))
        self.constraints.append(Constraint(sp.Eq(self.base_params.eps_semi, self.base_params.tech_values[self.base_params.eps_semi], evaluate=False), "eps_semi = tech_values[eps_semi]"))
        self.constraints.append(Constraint(sp.Eq(self.base_params.tsemi, self.base_params.tech_values[self.base_params.tsemi], evaluate=False), "tsemi = tech_values[tsemi]"))
        #self.constraints.append(sp.Eq(self.base_params.Lext, self.base_params.tech_values[self.base_params.Lext], evaluate=False))
        #self.constraints.append(sp.Eq(self.base_params.Lc, self.base_params.tech_values[self.base_params.Lc], evaluate=False))
        self.constraints.append(Constraint(sp.Eq(self.base_params.eps_cap, self.base_params.tech_values[self.base_params.eps_cap], evaluate=False), "eps_cap = tech_values[eps_cap]"))
        self.constraints.append(Constraint(sp.Eq(self.base_params.rho_c_n, self.base_params.tech_values[self.base_params.rho_c_n], evaluate=False), "rho_c_n = tech_values[rho_c_n]"))
        self.constraints.append(Constraint(sp.Eq(self.base_params.rho_c_p, self.base_params.tech_values[self.base_params.rho_c_p], evaluate=False), "rho_c_p = tech_values[rho_c_p]"))
        self.constraints.append(Constraint(sp.Eq(self.base_params.Rsh_c_n, self.base_params.tech_values[self.base_params.Rsh_c_n], evaluate=False), "Rsh_c_n = tech_values[Rsh_c_n]"))
        self.constraints.append(Constraint(sp.Eq(self.base_params.Rsh_c_p, self.base_params.tech_values[self.base_params.Rsh_c_p], evaluate=False), "Rsh_c_p = tech_values[Rsh_c_p]"))
        self.constraints.append(Constraint(sp.Eq(self.base_params.Rsh_ext_n, self.base_params.tech_values[self.base_params.Rsh_ext_n], evaluate=False), "Rsh_ext_n = tech_values[Rsh_ext_n]"))
        self.constraints.append(Constraint(sp.Eq(self.base_params.Rsh_ext_p, self.base_params.tech_values[self.base_params.Rsh_ext_p], evaluate=False), "Rsh_ext_p = tech_values[Rsh_ext_p]"))

        self.constraints.append(Constraint(self.delta <= 0.15, "delta <= 0.15"))
        #self.constraints.append(Constraint(sp.Eq(self.e_si*tan(math.pi*self.base_params.tox/self.Lscale) + self.base_params.k_gate*self.e_0*tan(math.pi*self.base_params.tsemi/self.Lscale), 0, evaluate=False), "e_si*tan(pi*t_ox/scale_length) + k_gate*e_0*tan(pi*tsemi/scale_length) <= 1e-6"))
        #self.constraints.append(Constraint(self.I_off_worst_case_per_um <= 100e-9, "I_off_worst_case_per_um <= 100e-9"))
        #self.constraints.append(Constraint(self.V_th_eff_worst_case >= 0.05, "V_th_eff_worst_case >= 0.05"))