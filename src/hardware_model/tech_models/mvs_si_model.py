import logging
from src.hardware_model.tech_models.tech_model_base import TechModel
from src.sim_util import symbolic_convex_max, symbolic_min, custom_cosh, custom_exp, custom_pow
import math
from sympy import symbols, ceiling, expand, exp, Abs, cosh, log
import sympy as sp

logger = logging.getLogger(__name__)

class MVSSiModel(TechModel):
    def __init__(self, model_cfg, base_params):
        super().__init__(model_cfg, base_params)

    def init_tech_specific_constants(self):
        super().init_tech_specific_constants()

        # gate tunneling parameters
        self.phi_b = 3.1  # Schottky barrier height (eV)
        self.m_ox = 0.5*self.m_0  # effective mass of electron in oxide (g)

        self.etov = 1.3e-5 # equivalent thickness of dielectric at S/D-G overlap (m)
        self.Rs0 = 100.0e-6 # source resistance (ohm*m)
        self.Rd0 = 100.0e-6 # drain resistance (ohm*m)
        
        self.Cif = 1.0e-10 # inner fringing S or D capacitance (F/m)
        self.Cof = 2.0e-11 # outer fringing S or D capacitance (F/m)
        self.vx0 = 0.765e5 # saturated electron velocity (m/s)
        self.u_n = 200.0e-4 # electron mobility (m^2/V*s)
        self.beta = 1.7
        self.alpha = 3.5
        self.mc = 0.2
        self.me = self.mc * self.m_0  
        self.Vbs = 0.0
        self.gamma = 0.0
        self.phib = 1.2
        self.phi_t = self.V_T

        self.A = ((self.q)**3) / (8*math.pi*self.h*self.phi_b*self.m_ox)
        self.B = (8*math.pi*(2*self.m_ox)**(1/2) * (self.phi_b*self.q)**(3/2)) / (3*self.q*self.h)

    def init_transistor_equations(self):
        super().init_transistor_equations()
        
        self.V_gsp = symbols('V_gsp')
        self.V_dsp = symbols('V_dsp')
        self.on_state = {
            self.V_gsp: self.base_params.V_dd,
            self.V_dsp: self.base_params.V_dd,
        }
        self.NL_state = {
            self.V_gsp: self.base_params.V_dd/2,
            self.V_dsp: self.base_params.V_dd,
        }
        self.H_state = {
            self.V_gsp: self.base_params.V_dd,
            self.V_dsp: self.base_params.V_dd/2,
        }
        self.off_state = {
            self.V_gsp: 0,
            self.V_dsp: self.base_params.V_dd,
        }

        self.init_intrinsics()

        self.init_extrinsics()

        self.apply_additional_effects()

        self.config_param_db()

    def set_scale_length(self):
        if self.model_cfg["effects"]["t_1"]:
            self.t_1 = self.base_params.t_1
        else:
            self.t_1 = 10e-9 # TODO: come back to this
        self.scale_length = self.eot + self.t_1

    def set_sce_parameters(self):
        self.delta = exp(-math.pi*self.base_params.L/(2*self.scale_length))
        self.alpha_g = 1
        # value of denominator clamped to avoid solver issues
        self.S = self.phi_t * log(10) / symbolic_convex_max(1 - 2*self.alpha_g*exp(-math.pi*self.base_params.L/(2*self.scale_length)), 1e-5)
        assert (1 - 2*self.alpha_g*exp(-math.pi*self.base_params.L/(2*self.scale_length))).xreplace(self.base_params.tech_values).evalf() > 1e-5, f"SS denominator is too small, clamping for solver will cause inaccuracies"
        self.n = self.S / (self.phi_t * log(10))

    def set_smoothing_functions(self):
        # F_f and some charge related stuff
        self.eVgpre = custom_exp((self.V_gsp - self.Vtpcorr)/(self.alpha*self.phi_t * 1.5))
        self.FFpre = 1/(1+self.eVgpre)
        self.ab = 2*(1-0.99 * self.FFpre) * self.phi_t
        self.Vcorr = (1.0 + 2.0 * self.delta) * (self.ab/2) * (custom_exp(-self.V_dsp/self.ab))
        self.Vgscorr = self.V_gsp + self.Vcorr
        self.Vbscorr = self.Vbs + self.Vcorr
        self.Vt0bs = self.base_params.V_th + self.gamma * ((Abs(self.phib-self.Vbscorr)-self.phib**(1/2))**(1/2))
        self.Vt0bs0 = self.base_params.V_th + self.gamma * ((Abs(self.phib-self.Vbs)-self.phib**(1/2))**(1/2))
        self.Vtp = self.Vt0bs - self.V_dsp * self.delta - 0.5 * self.alpha * self.phi_t
        self.Vtp0 = self.Vt0bs0 - self.V_dsp * self.delta - 0.5 * self.alpha * self.phi_t
        self.eVg = custom_exp((self.Vgscorr - self.Vtp)/(self.alpha*self.phi_t))
        if self.model_cfg["effects"]["F_f"]:
            self.F_f = 1/(1+self.eVg)
            self.F_f_eval = self.F_f.xreplace(self.on_state)
        else:
            self.F_f = 0.0 # for saturation region where Vgs sufficiently larger than Vth. Can look into near threshold region later
            self.F_f_eval = 0.0
        self.eVg0 = custom_exp((self.V_gsp - self.Vtp0)/(self.alpha*self.phi_t))
        self.FF0 = 1/(1+self.eVg0)
        self.Qref = self.C_inv * self.n * self.phi_t
        self.eta = (self.Vgscorr - (self.Vt0bs - self.V_dsp * self.delta - self.F_f * self.alpha * self.phi_t)) / (self.n * self.phi_t)
        self.eta0 = (self.V_gsp - (self.Vt0bs0 - self.V_dsp * self.delta - self.FF0 * self.alpha * self.phi_t)) / (self.n * self.phi_t)

        # F_s
        self.Vdsats = self.vx0 * self.Leff / self.u_n_eff
        self.Vdsat = self.Vdsats * (1 - self.F_f) + self.phi_t * self.F_f
        self.Vdratio = self.V_dsp/self.Vdsat
        self.Vdbeta = custom_pow(self.Vdratio, self.beta, evaluate=False)
        self.Vdbetabeta = (1 + self.Vdbeta)**(1/self.beta)
        if self.model_cfg["effects"]["F_s"]:
            self.F_s = self.Vdratio / self.Vdbetabeta
            self.F_s_eval = self.F_s.xreplace(self.on_state)
        else:
            self.F_s = 1.0 # in saturation region
            self.F_s_eval = 1.0
        self.Vdsat_eval = (self.Vdsat.xreplace(self.NL_state) + self.Vdsat.xreplace(self.H_state)) / 2
        self.Vdratio_eval = (self.Vdratio.xreplace(self.NL_state) + self.Vdratio.xreplace(self.H_state)) / 2
        logger.info(f"Vdsat: {self.Vdsat_eval.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"Vdratio: {self.Vdratio_eval.xreplace(self.base_params.tech_values).evalf()}")


    def set_vs_charge(self):
        self.Q_ix0 = self.Qref * log(1+custom_exp(self.eta))
        # clamp result of softplus so it doesn't go to 0, causing issues in solver
        #self.Q_ix0 = self.C_inv * self.n * self.phi_t * symbolic_convex_max(1e-10,log(1 + exp((self.V_gsp - (self.V_th_eff - self.alpha * self.phi_t * self.F_f))/(self.n*self.phi_t))))
        self.Q_ix0_0 = self.Q_ix0.xreplace({self.V_th_eff: self.V_th_eff_general})
        self.Q_ix0_0 = self.Q_ix0_0.xreplace({self.V_dsp: 0})

    def set_mobility(self):
        self.u_n_eff = self.u_n # TODO: add mobility model

    def init_intrinsics(self):
        self.L_ov = self.base_params.L / 8 # TODO replace with proper value
        self.Leff = self.base_params.L - self.L_ov
        self.Cox = self.e_0 * self.base_params.k_gate / self.base_params.tox
        logger.info(f"Cox: {self.Cox.xreplace(self.base_params.tech_values).evalf()}")

        #self.C_inv = (self.Cox * self.base_params.Cs) / (self.Cox + self.base_params.Cs)
        self.C_inv = self.Cox

        self.eot = self.base_params.tox * 3.9/self.base_params.k_gate
        self.set_scale_length()

        self.set_sce_parameters()

        self.set_mobility()

        self.Vtpcorr = self.base_params.V_th + self.gamma * ((Abs(self.phib-self.Vbs)-self.phib**(1/2))**(1/2)) - self.V_dsp * self.delta

        self.V_th_eff_general = self.Vtpcorr
        self.V_th_eff = (self.V_th_eff_general.xreplace(self.NL_state) + self.V_th_eff_general.xreplace(self.H_state)) / 2

        self.set_smoothing_functions()

        self.set_vs_charge()

    def set_parasitic_resistances(self):
        self.R_s = self.Rs0/self.base_params.W # ohm*m
        self.R_d = self.R_s

    def set_parasitic_capacitances(self):
        self.Cofs = (0.345e-14/self.etov) * self.L_ov / 2.0 + self.Cof
        self.Cofd = (0.345e-14/self.etov) * self.L_ov / 2.0 + self.Cof


    def set_gate_tunneling(self):
        # gate tunneling current (Fowler-Nordheim and WKB)
        # minimums are to avoid exponential explosion in solver. Normal values in exponent are negative.
        # gate tunneling
        #self.V_ox = symbolic_convex_max(self.base_params.V_dd - self.V_th_eff, self.V_th_eff).xreplace(self.off_state)
        self.V_ox = self.base_params.V_dd - self.V_th_eff
        self.E_ox = Abs(self.V_ox/self.base_params.tox)
        logger.info(f"B: {self.B}, A: {self.A}, t_ox: {self.base_params.tox.xreplace(self.base_params.tech_values)}, E_ox: {self.E_ox.xreplace(self.base_params.tech_values)}, intermediate: {(1-(1-self.V_ox/self.phi_b)**3/2).xreplace(self.base_params.tech_values)}")
        self.FN_term = self.A_gate * self.A * self.E_ox**2 * (custom_exp(-self.B/self.E_ox))
        self.WKB_term = self.A_gate * self.A * self.E_ox**2 * (custom_exp(-self.B*(1-(1-self.V_ox/self.phi_b)**3/2)/self.E_ox))
        self.I_tunnel = self.FN_term + self.WKB_term
        logger.info(f"I_tunnel: {self.I_tunnel.xreplace(self.base_params.tech_values)}")

    def init_extrinsics(self):
        self.set_parasitic_resistances()
        self.set_parasitic_capacitances()
        self.v = self.vx0 * (self.F_f + (1 - self.F_f) / (1 + self.base_params.W * self.R_s * self.C_inv * (1 + 2*self.delta)*self.vx0))
        #self.R_cmin = self.L_c / (self.Q_ix0_0 * self.u_n_eff)
        self.I_d = self.base_params.W * self.Q_ix0 * self.v * self.F_s

        self.I_d_on = ((self.I_d).xreplace(self.NL_state) + (self.I_d).xreplace(self.H_state)) / 2
        #logger.info(f"I_d_on equation: {self.I_d_on.simplify()}")

        self.A_gate = self.base_params.W * self.base_params.L

        self.C_gate = self.Cox * self.A_gate

        logger.info(f"A_gate: {self.A_gate.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"area_scale: {self.base_params.area_scale.xreplace(self.base_params.tech_values).evalf()}")

        #self.C_diff = self.Cofs + self.Cofd
        self.C_diff = self.C_gate
        self.C_load = 4 * self.C_gate # FO4 load capacitance
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

        self.I_sub = self.I_d.xreplace(self.off_state) # max subthreshold leakage

        self.I_off = self.I_sub

        self.set_gate_tunneling()
        
        if self.model_cfg["effects"]["gate_tunneling"]:
            self.I_off = self.I_off + self.I_tunnel

        self.I_d_on_per_um = self.I_d_on / (self.base_params.W* 1e6)
        self.I_sub_per_um = self.I_sub / (self.base_params.W* 1e6)
        self.I_tunnel_per_um = self.I_tunnel / (self.base_params.W* 1e6)
        self.I_d_off_per_um = self.I_off / (self.base_params.W* 1e6)

        logger.info(f"I_d_on per um: {self.I_d_on_per_um.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"I_sub per um: {self.I_sub_per_um.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"I_tunnel per um: {self.I_tunnel_per_um.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"I_off per um: {self.I_d_off_per_um.xreplace(self.base_params.tech_values).evalf()}")
        self.E_act_inv = (0.5*(self.C_load + self.C_diff + self.C_wire)*self.base_params.V_dd*self.base_params.V_dd) * 1e9  # nJ
        logger.info(f"C_load: {self.C_load.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"C_diff: {self.C_diff.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"C_wire: {self.C_wire.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"E_act_inv: {self.E_act_inv.xreplace(self.base_params.tech_values).evalf()}")

        self.P_pass_inv = self.I_off * self.base_params.V_dd
        

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
        self.param_db["I_sub"] = self.I_sub
        self.param_db["I_tunnel"] = self.I_tunnel
        self.param_db["I_d_on_per_um"] = self.I_d_on_per_um
        self.param_db["I_sub_per_um"] = self.I_sub_per_um
        self.param_db["I_tunnel_per_um"] = self.I_tunnel_per_um
        self.param_db["I_d_off_per_um"] = self.I_d_off_per_um
        self.param_db["Eox"] = self.E_ox
        self.param_db["Vox"] = self.V_ox
        self.param_db["scale_length"] = self.scale_length
        self.param_db["A_gate"] = self.A_gate

    def apply_base_parameter_effects(self):
        pass

    def apply_additional_effects(self):
        super().apply_additional_effects()

    def create_constraints(self, dennard_scaling_type="constant_field"):
        super().create_constraints(dennard_scaling_type)
        self.constraints.append(self.delta <= 0.15)
        