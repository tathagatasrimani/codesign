import logging
from src.hardware_model.tech_models.tech_model_base import TechModel
from src.sim_util import symbolic_convex_max, symbolic_min, custom_cosh, custom_exp
import math
from sympy import symbols, ceiling, expand, exp, Abs, cosh, log
import sympy as sp
from src.inverse_pass.constraint import Constraint

logger = logging.getLogger(__name__)

class VSModel(TechModel):
    def __init__(self, model_cfg, base_params):
        super().__init__(model_cfg, base_params)

    def init_tech_specific_constants(self):
        super().init_tech_specific_constants()

    def init_transistor_equations(self):
        super().init_transistor_equations()
        
        

        self.config_param_db()

    def config_param_db(self):
        super().config_param_db()
        """
        self.param_db["L"] = self.base_params.L
        self.param_db["W"] = self.base_params.W
        self.param_db["I_sub"] = self.I_sub
        self.param_db["V_th"] = self.base_params.V_th
        self.param_db["V_th_eff"] = self.V_th_eff.xreplace(self.on_state).evalf()
        self.param_db["V_dd"] = self.base_params.V_dd
        self.param_db["wire RC"] = self.m1_Rsq * self.m1_Csq
        self.param_db["I_on"] = self.I_d_on
        self.param_db["I_on_per_um"] = self.I_d_on_per_um
        self.param_db["I_off_per_um"] = self.I_d_off_per_um
        self.param_db["I_tunnel_per_um"] = self.I_tunnel_per_um
        self.param_db["I_sub_per_um"] = self.I_sub_per_um
        self.param_db["DIBL factor"] = self.delta
        self.param_db["t_ox"] = self.base_params.tox
        self.param_db["eot"] = self.eot
        self.param_db["scale_length"] = self.scale_length
        self.param_db["C_load"] = self.C_load
        self.param_db["C_wire"] = self.C_wire
        self.param_db["R_wire"] = self.R_wire
        self.param_db["R_device"] = self.base_params.V_dd/self.I_d_on
        self.param_db["SS"] = self.S
        self.param_db["F_f"] = self.F_f_eval
        self.param_db["F_s"] = self.F_s_eval
        self.param_db["vx0"] = self.vx0
        self.param_db["v"] = self.v.xreplace(self.on_state).evalf()
        self.param_db["t_1"] = self.t_1
        self.param_db["R_s"] = self.R_s
        self.param_db["R_d"] = self.R_d
        self.param_db["parasitic capacitance"] = self.C_diff
        self.param_db["k_gate"] = self.base_params.k_gate

        self.param_db["A_gate"] = self.A_gate"""

    def apply_base_parameter_effects(self):
        pass

    def apply_additional_effects(self):
        super().apply_additional_effects()

    def create_constraints(self, dennard_scaling_type="constant_field"):
        super().create_constraints(dennard_scaling_type)
        