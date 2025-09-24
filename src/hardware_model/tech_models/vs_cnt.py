import logging
from src.hardware_model.tech_models.tech_model_base import TechModel
from src.sim_util import symbolic_convex_max, symbolic_min, custom_cosh, custom_exp, custom_coth
import math
from sympy import symbols, ceiling, expand, exp, Abs, cosh, log, acosh
import sympy as sp


logger = logging.getLogger(__name__)

class VSPlanarModel(TechModel):
    def __init__(self, model_cfg, base_params):
        super().__init__(model_cfg, base_params)

    def init_tech_specific_constants(self):
        super().init_tech_specific_constants()

        # gate tunneling parameters
        self.phi_b = 3.1  # Schottky barrier height (eV)
        self.m_ox = 0.5*self.m_0  # effective mass of electron in oxide (g)
        self.A = ((self.q)**3) / (8*math.pi*self.h*self.phi_b*self.m_ox)
        self.B = (8*math.pi*(2*self.m_ox)**(1/2) * (self.phi_b*self.q)**(3/2)) / (3*self.q*self.h)
        #self.A = 4.97232 # for NMOS
        #self.B = 7.45669e11 # for NMOS

        self.cqa = 0.087e-15 / 1e-6
        self.cqb = 0.16e-15 / 1e-6
        self.Ep = 3*self.q
        self.acc = 0.142e-9
        self.d00 = 1.0e-9
        self.u_00 = 0.135 # m^2/V.s
        self.lam_00 = 66.2e-9
        self.c_u = 1.37
        self.z0 = 2.405
        self.Efsd = 0.3 # see figure 4 in paper
        self.vb0 = 4.1e9 # (m/s)
        self.d0 = 1.2e-9
        self.lam_v = 440.0e-9
        self.R_q = self.h/(4*self.q**2)
        self.g_c0 = 0.49e+3 # S/m
        self.E00 = 32e-3 * self.q # eV
        self.lam_c = 380.0e-9
        self.R_ext0 = 35
        self.alpha_d = 2.0
        self.alpha_n = 2.1
        self.kspa = 3.9 # spacer k


        # GIDL parameters
        self.A_GIDL = 1e-12
        self.B_GIDL = 2.3e9
        self.E_GIDL = 0.8

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

    def set_intrinsic_caps(self):
        self.Eg = 2*self.Ep*self.acc/self.base_params.d
        self.Cqeff = self.cqa*(self.q * self.Eg/(self.K*self.T)) + self.cqb # seems to be different models in paper versus technical manual, using paper here
        self.Cox = 2*math.pi*self.base_params.k_gate*self.e_0/(sp.log((2*self.base_params.tox + self.base_params.d) / self.base_params.d)) # assuming GAA structure
        self.C_inv = self.Cox * self.Cqeff / (self.Cox + self.Cqeff)
        logger.info(f"C_inv: {self.C_inv.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"Cox: {self.Cox.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"Cqeff: {self.Cqeff.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"Eg: {self.Eg.xreplace(self.base_params.tech_values).evalf()}")

    def set_mobility(self):
        self.u_0 = self.u_00
        self.lam_u = self.lam_00
        self.u_n_eff = (self.u_0 * self.base_params.L * (self.base_params.d/self.d00)**self.c_u) / (self.lam_u + self.base_params.L)

    def set_scale_length(self):
        self.eta_0 = self.z0*self.base_params.d/(self.base_params.d + 2*self.base_params.tox)
        self.b = 0.41*(self.eta_0/2 - (self.eta_0**3)/16)*(math.pi*self.eta_0/2)
        # assumes tox > d/2
        self.scale_length = (self.base_params.d + 2*self.base_params.tox) / (2*self.z0) * (1 + self.b * (self.gamma - 1))
        # if tox << d: self.scale_length = (self.base_params.d + 2*self.base_params.gamma + self.base_params.tox)/self.z0

    def set_sce_parameters(self):
        self.n = 1/(1-exp(-self.zeta))
        self.delta = exp(-self.zeta)
        self.Vth_rolloff = (2*self.Efsd + self.Eg)*exp(-self.zeta)

    def set_gate_tunneling_current(self):
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

    def set_smoothing_functions(self):
        if self.model_cfg["effects"]["F_f"]:
            self.F_f = 1/(1+custom_exp((self.V_gsp - (self.V_th_eff - self.alpha * self.phi_t / 2))/(self.alpha * self.phi_t)))
            self.F_f_eval = self.F_f.xreplace(self.on_state)
        else:
            self.F_f = 0.0 # for saturation region where Vgs sufficiently larger than Vth. Can look into near threshold region later
            self.F_f_eval = 0.0

        if self.model_cfg["effects"]["F_s"]:
            self.Vdsats = self.v * self.base_params.L / self.u_n_eff
            self.Vdsat = self.Vdsats * (1 - self.F_f) + self.phi_t * self.F_f
            self.F_s = (self.V_dsp / self.Vdsat) / (1 + (self.V_dsp / self.Vdsat)**self.beta)**(1/self.beta)
            self.F_s_eval = self.F_s.xreplace(self.on_state)
        else:
            self.F_s = 1.0 # in saturation region
            self.F_s_eval = 1.0

    def set_terminal_charge(self):
        self.Q_ix0 = self.C_inv * self.n * self.phi_t * log(1 + custom_exp((self.V_gsp - (self.V_th_eff - self.alpha * self.phi_t * self.F_f))/(self.n*self.phi_t)))
        # clamp result of softplus so it doesn't go to 0, causing issues in solver
        #self.Q_ix0 = self.C_inv * self.n * self.phi_t * symbolic_convex_max(1e-10,log(1 + exp((self.V_gsp - (self.V_th_eff - self.alpha * self.phi_t * self.F_f))/(self.n*self.phi_t))))
        self.Q_ix0_0 = self.Q_ix0.xreplace({self.V_th_eff: self.V_th_eff_general})
        self.Q_ix0_0 = self.Q_ix0_0.xreplace({self.V_dsp: 0})
    
    def init_intrinsics(self):
        self.set_intrinsic_caps()

        self.set_mobility()

        self.eot = self.base_params.tox * 3.9/self.base_params.k_gate

        self.set_scale_length()

        self.L_of = self.base_params.tox / 3
        self.zeta = (self.base_params.L + 2*self.L_of) / (2*self.scale_length)

        self.set_sce_parameters()

        self.vb = self.vb0 * (self.base_params.d/self.d0)**0.5
        self.vx0 = self.lam_v / (self.lam_v + self.base_params.L) * self.vb


        logger.info(f"delta: {self.delta.xreplace(self.base_params.tech_values).evalf()}")
        self.V_th_eff_general = self.base_params.V_th - self.delta * self.V_dsp - self.Vth_rolloff
        self.V_th_eff = (self.V_th_eff_general.xreplace(self.NL_state) + self.V_th_eff_general.xreplace(self.H_state)) / 2
        self.alpha = 3.5
        self.phi_t = self.V_T
        self.beta = 1.8

        # PICK UP HERE WITH CAPS
        self.C_g = self.C_inv

        self.set_smoothing_functions()

        self.set_terminal_charge()

    def set_parasitic_resistances(self):
        # phi_b has range of +/- phi_m, which is defined as 5.1eV
        self.phi_b = self.Eg/2
        self.g_c = self.g_c0*exp(-self.phi_b/self.E00)
        self.L_T = (self.g_c*self.R_q/self.lam_c + (self.g_c*self.R_q/2)**2)**-0.5
        self.R_c = self.R_q/2 * (1+4/(self.lam_c*self.g_c*self.R_q))**0.5 * custom_coth(self.base_params.L_c/self.L_T)
        self.L_ext = self.base_params.L/2 # TODO: check if this is correct
        self.nsd = 0.6e+9 # m^-1, come back to this but didn't want to get into doping concentration for the model
        self.R_ext = self.R_ext0 * self.L_ext / (self.d**self.alpha_d * self.nsd**self.alpha_n)
        self.R_s = self.R_c + self.R_ext # ohm*um
        self.R_d = self.R_s

    def set_Cof(self):
        tao1 = 2.5
        tao2 = 2.0
        h = self.base_params.tox + self.d/2
        sr = 0.4
        A = sr*2*math.pi*self.kspa*self.e_0*self.L_ext
        B = acosh(2*(h**2 + (0.28*self.L_ext)**2)**0.5 / self.d)
        self.C_of = A/B # for 1 CNT

    def set_Cgtc(self):
        # assuming backgate
        alphac = 2.76
        betac = 0.384
        tao = custom_exp(2-2*(1+ (2*(self.base_params.H_c + self.base_params.L_c)/self.L_ext)**0.5))
        tao2 = custom_exp(2-2*(1+ (2*(self.base_params.H_g + self.base_params.L)/self.base_params.tox)**0.5))
        con = self.e_0*self.base_params.W*self.kspa
        self.C_gtc = con*((alphac/log(2*math.pi * ((self.base_params.L_c + tao + self.base_params.H_c))) + betac/log(2*math.pi * ((self.base_params.L + self.base_params.tox) / (2*self.base_params.L + tao2*self.base_params.H_g)))))

    def set_parasitic_capacitances(self):
        self.set_Cof()
        self.set_Cgtc()

    def init_extrinsics(self):
        self.set_parasitic_resistances()

        self.set_parasitic_capacitances()

        self.v = self.vx0 * (self.F_f + (1 - self.F_f) / (1 + self.base_params.W * self.R_s * self.C_g * (1 + 2*self.delta)*self.vx0))

        #self.R_cmin = self.L_c / (self.Q_ix0_0 * self.u_n_eff)
        self.I_d = self.base_params.W * self.Q_ix0 * self.v * self.F_s

        #self.I_d_on = (self.I_d).xreplace(self.on_state)
        self.I_d_on = ((self.I_d).xreplace(self.NL_state) + (self.I_d).xreplace(self.H_state)) / 2
        #logger.info(f"I_d_on equation: {self.I_d_on.simplify()}")

        self.A_gate = self.base_params.W * self.base_params.L

        self.C_gate = self.Cox * self.A_gate

        logger.info(f"A_gate: {self.A_gate.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"area_scale: {self.base_params.area_scale.xreplace(self.base_params.tech_values).evalf()}")

        self.I_sub = self.I_d.xreplace(self.off_state) # max subthreshold leakage
        self.I_off = self.I_sub

        self.set_gate_tunneling_current()
        
        if self.model_cfg["effects"]["gate_tunneling"]:
            self.I_off = self.I_off + self.I_tunnel

        self.C_diff = self.C_of + self.C_gtc
        self.C_load = 4*self.C_gate # FO4 model
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

        self.I_d_on_per_um = self.I_d_on / (self.base_params.W* 1e6)
        self.I_sub_per_um = self.I_sub / (self.base_params.W* 1e6)
        self.I_tunnel_per_um = self.I_tunnel / (self.base_params.W* 1e6)
        self.I_d_off_per_um = self.I_off / (self.base_params.W* 1e6)
        self.I_GIDL_per_um = self.I_GIDL / (self.base_params.W* 1e6)

        logger.info(f"I_d_on per um: {self.I_d_on_per_um.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"I_sub per um: {self.I_sub_per_um.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"I_tunnel per um: {self.I_tunnel_per_um.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"I_off per um: {self.I_d_off_per_um.xreplace(self.base_params.tech_values).evalf()}")
        self.E_act_inv = (0.5*(self.C_load + self.C_diff + self.C_wire)*self.base_params.V_dd*self.base_params.V_dd) * 1e9  # nJ
        logger.info(f"C_load: {self.C_load.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"C_diff: {self.C_diff.xreplace(self.base_params.tech_values).evalf()}")
        logger.info(f"C_wire: {self.C_wire.xreplace(self.base_params.tech_values).evalf()}")

        self.P_pass_inv = self.I_off * self.base_params.V_dd

        logger.info(f"E_act_inv: {self.E_act_inv.xreplace(self.base_params.tech_values).evalf()}")

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
        self.param_db["I_GIDL_per_um"] = self.I_GIDL_per_um
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
        self.constraints.append(self.base_params.tox >= 2*self.base_params.d) # ensure scale length equation holds
        self.constraints.append(self.base_params.k_cnt <= 20)
        self.constraints.append(self.base_params.k_cnt >= 1)
        self.constraints.append(self.base_params.L_c == self.base_params.L/5) # from minimum values in technical manual