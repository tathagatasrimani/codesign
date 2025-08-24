import logging
from src.tech_model import TechModel
from src.sim_util import symbolic_convex_max, symbolic_convex_min, custom_cosh, custom_exp
import math
from sympy import symbols, ceiling, expand, exp, Abs, cosh, log
import sympy as sp


logger = logging.getLogger(__name__)

class VSModel(TechModel):
    def __init__(self, model_cfg, base_params):
        super().__init__(model_cfg, base_params)

    def init_tech_specific_constants(self):
        super().init_tech_specific_constants()

    def init_transistor_equations(self):
        super().init_transistor_equations()
        
        self.V_gsp = symbols('V_gsp')
        self.V_dsp = symbols('V_dsp')
        self.on_state = {
            self.V_gsp: self.base_params.V_dd,
            self.V_dsp: self.base_params.V_dd,
        }
        self.off_state = {
            self.V_gsp: 0,
            self.V_dsp: self.base_params.V_dd,
        }
        self.Cox = self.e_0 * self.base_params.k_gate / self.base_params.tox
        logger.info(f"Cox: {self.Cox.xreplace(self.base_params.tech_values).evalf()}")

        #self.C_inv = (self.Cox * self.base_params.Cs) / (self.Cox + self.base_params.Cs)
        self.C_inv = self.Cox

        self.delta_32n = 0.12 
        self.delta = self.delta_32n + 0.01 * (32e-9)/(self.base_params.L)
        logger.info(f"delta: {self.delta.xreplace(self.base_params.tech_values).evalf()}")
        self.V_th_eff = (self.base_params.V_th - self.delta * self.V_dsp).xreplace(self.on_state).evalf()
        self.alpha = 3.5
        self.phi_t = self.V_T
        self.S_32n = 0.098 # V/decade
        self.beta = 2.5 # for nmos
        self.L_ov = 0.15 * self.base_params.L # given as good typical value in paper
        self.vx0_32n = 1.35e5 # m/s (vs paper uses 35nm as reference but only data for 32nm)
        self.u = 250e-4 # m^2/V.s
        self.R_s = 75 # ohm*um, need to get rid of constant 75
        self.R_d = self.R_s
        self.C_g = self.C_inv # TODO: check if this is correct. GPT says Cg <= Cinv <= Cox

        self.vx0 = self.vx0_32n + 1e5 * (self.delta - self.delta_32n)
        self.S = self.S_32n + 0.1 * (self.delta * self.delta_32n)
        self.n = self.S / (self.phi_t * log(10))

        # note: we only care about ON state and saturation region for digital applications

        #self.F_f = 1/(1+custom_exp((self.V_gsp - (self.V_th_eff - self.alpha * self.phi_t / 2))/(self.alpha * self.phi_t)))
        self.F_f = 0 # for saturation region where Vgs sufficiently larger than Vth. Can look into near threshold region later
        #self.Q_ix0 = self.C_inv * self.n * self.phi_t * log(1 + custom_exp((self.V_gsp - (self.V_th_eff - self.alpha * self.phi_t * self.F_f))/(self.n*self.phi_t)))
        self.Q_ix0 = self.C_inv * self.n * self.phi_t * log(1 + exp((self.V_gsp - (self.V_th_eff - self.alpha * self.phi_t * self.F_f))/(self.n*self.phi_t)))
        self.Q_ix0_0 = self.Q_ix0.subs({self.V_dsp: 0})

        
        self.v = self.vx0 * (self.F_f + (1 - self.F_f) / (1 + self.base_params.W * self.R_s * self.C_g * (1 + 2*self.delta)*self.vx0))
        #self.L_c = self.base_params.L - 2 * self.L_ov
        #self.Vdsats = self.v * self.L_c / self.u + (self.R_s + self.R_d) * self.base_params.W * self.Q_ix0_0 * self.v
        #self.Vdsat = self.Vdsats * (1 - self.F_f) + self.phi_t * self.F_f
        #self.F_s = (self.V_dsp / self.Vdsat) / (1 + (self.V_dsp / self.Vdsat)**self.beta)**(1/self.beta)
        self.F_s = 1 # in saturation region

        #self.R_cmin = self.L_c / (self.Q_ix0_0 * self.u)
        self.I_d = self.base_params.W * self.Q_ix0 * self.v * self.F_s

        self.I_d_on = (self.I_d).subs(self.on_state)
        logger.info(f"I_d_on equation: {self.I_d_on.simplify()}")

        self.A_gate = self.base_params.W * self.base_params.L

        self.C_gate = self.Cox * self.A_gate

        logger.info(f"A_gate: {self.A_gate.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"area_scale: {self.base_params.area_scale.xreplace(self.base_params.tech_values).evalf()}")

        self.C_diff = self.C_gate
        self.C_load = self.C_gate
        self.R_avg_inv = self.base_params.V_dd / self.I_d_on

        logger.info(f"R_wire: {self.R_wire.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"C_wire: {self.C_wire.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"wire rc: {(self.R_wire * self.C_wire).xreplace(self.base_params.tech_values).evalf()}")

        if self.model_cfg["delay_parasitics"] == "all":
            self.delay = (self.R_avg_inv * (self.C_diff + self.C_wire/2) + (self.R_avg_inv + self.R_wire) * (self.C_wire/2 + self.C_load)) * 1e9  # ns
        elif self.model_cfg["delay_parasitics"] == "Csq only":
            self.delay = (self.R_avg_inv * (self.C_diff + self.C_wire/2) + (self.R_avg_inv) * (self.C_wire/2 + self.C_load)) * 1e9  # ns
        elif self.model_cfg["delay_parasitics"] == "const":
            self.delay = self.R_avg_inv * (self.C_diff + self.C_load + 0.3e-15 * 100) * 1e9  # ns
        else:
            self.delay = self.R_avg_inv * (self.C_load + self.C_diff) * 1e9
        #self.I_d_off = (self.base_params.W * self.Q_ix0_0 * self.vx0 * self.F_s).subs(self.off_state)
        self.I_off = self.u_n_eff * self.Cox * self.base_params.W / self.base_params.L * self.V_T**2 * custom_exp(-self.V_th_eff / (self.n * self.V_T))

        self.I_d_on_per_um = self.I_d_on / (self.base_params.W* 1e6)
        self.I_d_off_per_um = self.I_off / (self.base_params.W* 1e6)

        logger.info(f"I_d_on per um: {self.I_d_on_per_um.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"I_d_off per um: {self.I_d_off_per_um.xreplace(self.base_params.tech_values).evalf()}")

        self.E_act_inv = (0.5*(self.C_load + self.C_diff + self.C_wire)*self.base_params.V_dd*self.base_params.V_dd) * 1e9  # nJ
        logger.info(f"C_load: {self.C_load.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"C_diff: {self.C_diff.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"C_wire: {self.C_wire.xreplace(self.base_params.tech_values).evalf()}")

        self.P_pass_inv = self.I_off * self.base_params.V_dd


        self.apply_additional_effects()
        logger.info(f"E_act_inv: {self.E_act_inv.xreplace(self.base_params.tech_values).evalf()}")

        self.config_param_db()

    def config_param_db(self):
        super().config_param_db()
        self.param_db["I_d"] = self.I_d_on
        self.param_db["C_load"] = self.C_load
        self.param_db["delay"] = self.delay
        self.param_db["u_n_eff"] = self.u_n_eff
        self.param_db["V_th_eff"] = self.V_th_eff
        self.param_db["delta_vt_dibl"] = (-self.delta * self.V_dsp).xreplace(self.on_state).evalf()
        self.param_db["R_wire"] = self.R_wire
        self.param_db["C_wire"] = self.C_wire
        self.param_db["I_sub"] = self.I_off
        self.param_db["I_d_on_per_um"] = self.I_d_on_per_um
        self.param_db["I_sub_per_um"] = self.I_d_off_per_um

    def apply_base_parameter_effects(self):
        pass

    def apply_additional_effects(self):
        if self.model_cfg["effects"]["area_and_latency_scaling"]:
            self.delay = self.delay * self.base_params.latency_scale
            self.P_pass_inv = self.P_pass_inv * self.base_params.area_scale
            #self.wire_len = self.wire_len * self.base_params.latency_scale

    def create_constraints(self, dennard_scaling_type="constant_field"):
        super().create_constraints(dennard_scaling_type)
        