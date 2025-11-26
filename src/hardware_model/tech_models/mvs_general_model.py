import logging
from src.hardware_model.tech_models.tech_model_base import TechModel
from src.sim_util import symbolic_convex_max, symbolic_min, custom_cosh, custom_exp
import math
from sympy import symbols, ceiling, expand, exp, Abs, cosh, log
import sympy as sp
from scipy.constants import k, e, epsilon_0, hbar, m_e

from src.hardware_model.tech_models.tech_codesign_v0 import tech_codesign_v0

logger = logging.getLogger(__name__)

class MVSGeneralModel(TechModel):
    def __init__(self, model_cfg, base_params):
        super().__init__(model_cfg, base_params)

    def init_tech_specific_constants(self):
        super().init_tech_specific_constants()

    def init_transistor_equations(self):
        super().init_transistor_equations()
        self.area, self.delay, self.P_active, self.P_static, self.P_total, self.Ieff_n, self.Ieff_p, self.Ioff_n, self.Ioff_p, self.C_load = tech_codesign_v0.final_symbolic_models(
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
            self.base_params.f, 
            self.base_params.a
        )
        logger.info(f"Area: {self.area.xreplace(self.base_params.tech_values).evalf():.3e}")
        logger.info(f"Delay: {self.delay.xreplace(self.base_params.tech_values).evalf():.3e}")
        logger.info(f"Power_active: {self.P_active.xreplace(self.base_params.tech_values).evalf():.3e}")
        logger.info(f"Power_static: {self.P_static.xreplace(self.base_params.tech_values).evalf():.3e}")
        logger.info(f"Power_total: {self.P_total.xreplace(self.base_params.tech_values).evalf():.3e}")
        logger.info(f"Ieff_n: {self.Ieff_n.xreplace(self.base_params.tech_values).evalf():.3e}")
        logger.info(f"Ieff_p: {self.Ieff_p.xreplace(self.base_params.tech_values).evalf():.3e}")
        logger.info(f"Ioff_n: {self.Ioff_n.xreplace(self.base_params.tech_values).evalf():.3e}")
        logger.info(f"Ioff_p: {self.Ioff_p.xreplace(self.base_params.tech_values).evalf():.3e}")
        logger.info(f"Cload: {self.C_load.xreplace(self.base_params.tech_values).evalf():.3e}")

        self.E_act_inv = self.P_active * self.delay
        self.P_pass_inv = self.P_static
        self.C_diff = self.C_load/self.base_params.FO
        self.R_avg_inv = 2*self.base_params.V_dd/(self.Ieff_n + self.Ieff_p)
        self.A_gate = self.area
        self.I_off = (self.Ioff_n + self.Ioff_p)/2

        self.Lscale = tech_codesign_v0.get_Lscale(self.base_params.k_gate, self.base_params.eps_semi, self.base_params.tox, self.base_params.tsemi)
        logger.info(f"Lscale: {self.Lscale.xreplace(self.base_params.tech_values).evalf():.3e}")
        self.n0, self.delta, self.dVt = tech_codesign_v0.symbolic_sce_model_cmg(self.base_params.L, self.base_params.V_th, self.Lscale)
        self.V_th_eff = self.base_params.V_th - self.dVt - self.delta * self.base_params.V_dd
        logger.info(f"n0: {self.n0.xreplace(self.base_params.tech_values).evalf():.3e}")
        logger.info(f"delta: {self.delta.xreplace(self.base_params.tech_values).evalf():.3e}")
        logger.info(f"dVt: {self.dVt.xreplace(self.base_params.tech_values).evalf():.3e}")

        self.apply_additional_effects()

        self.config_param_db()


    def config_param_db(self):
        super().config_param_db()

    def apply_base_parameter_effects(self):
        super().apply_base_parameter_effects()

    def apply_additional_effects(self):
        super().apply_additional_effects()

    def create_constraints(self, dennard_scaling_type="constant_field"):
        super().create_constraints(dennard_scaling_type)
        self.constraints.append(sp.Eq(self.base_params.FO, self.base_params.tech_values[self.base_params.FO], evaluate=False))
        self.constraints.append(sp.Eq(self.base_params.M, self.base_params.tech_values[self.base_params.M], evaluate=False))
        self.constraints.append(sp.Eq(self.base_params.beta_p_n, self.base_params.tech_values[self.base_params.beta_p_n], evaluate=False))
        self.constraints.append(sp.Eq(self.base_params.a, self.base_params.tech_values[self.base_params.a], evaluate=False))
        self.constraints.append(sp.Eq(self.base_params.mD_fac, self.base_params.tech_values[self.base_params.mD_fac], evaluate=False))