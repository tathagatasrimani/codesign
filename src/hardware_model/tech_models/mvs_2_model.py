import logging
from src.hardware_model.tech_models.tech_model_base import TechModel
from src.sim_util import symbolic_convex_max, symbolic_min, custom_cosh, custom_exp
import math
from sympy import symbols, ceiling, expand, exp, Abs, cosh, log
import sympy as sp

logger = logging.getLogger(__name__)

"""
From the technical manual:

MVS 2.0.0 is an improved physics-based virtual source (VS) model to describe transport in quasi-
ballistic transistors. The model is based on Landauer scattering theory and incorporates the eﬀects of (i)
degeneracy on thermal velocity and mean free path of carriers in the channel, (ii) drain-bias dependence
of gate capacitance and VS charge including the eﬀects of band non-parabolicity, and (iii) non-linear
resistance of the extrinsic device region on gm-degradation at high drain currents in the channel. The
improved charge model captures the phenomenon of reduction in VS charge under non-equilibrium
transport conditions in a quasi-ballistic transistor. We test the accuracy of the MVS 2.0.0 model by
comparing the model results with the measured I-V data of InGaAs HEMT devices with gate lengths
from 30 nm to 130 nm and Si ETSOI devices with gate lengths from 30 nm to 50 nm.
"""

class MVS2Model(TechModel):
    def __init__(self, model_cfg, base_params):
        super().__init__(model_cfg, base_params)

    def init_tech_specific_constants(self):
        super().init_tech_specific_constants()

        # gate tunneling parameters
        self.phi_b = 3.1  # Schottky barrier height (eV)
        self.m_ox = 0.5*self.m_0  # effective mass of electron in oxide (g)

        self.theta = 2.5
        self.beta = 1.55
        self.mt = 0.19 # relative transverse effective mass of carriers
        self.ml = 0.89 # relative longitudinal effective mass of carriers
        self.Rs0 = 160e-12 # source resistance (Ohm*m)
        self.ksee = 0.1 # parameter for VS velocity
        self.B_qm = 6.8e-9 # Stern QM correction numerator
        self.dqm0 = 4.6e-9 # QM correction factor at Qx0=0
        self.eps = 13.6 # relative permittivity of semiconductor channel material
        self.phi_t = self.V_T
        self.nu = 0.7

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

    def set_F_f(self):
        if self.model_cfg["effects"]["F_f"]:
            self.F_f = 1/(1+custom_exp((self.V_gsp - (self.V_th_eff - self.alpha * self.phi_t / 2))/(self.alpha * self.phi_t)))
            self.F_f_eval = self.F_f.xreplace(self.on_state)
        else:
            self.F_f = 0.0 # for saturation region where Vgs sufficiently larger than Vth. Can look into near threshold region later
            self.F_f_eval = 0.0

    def set_F_s(self):
        if self.model_cfg["effects"]["F_s"]:
            self.Vdsats = self.v * self.base_params.L / self.u_n_eff
            self.Vdsat = self.Vdsats * (1 - self.F_f) + self.phi_t * self.F_f
            self.F_s = (self.V_dsp / self.Vdsat) / (1 + (self.V_dsp / self.Vdsat)**self.beta)**(1/self.beta)
            self.F_s_eval = self.F_s.xreplace(self.on_state)
        else:
            self.F_s = 1.0 # in saturation region
            self.F_s_eval = 1.0

    def set_vs_charge(self):
        self.Q_ix0 = self.C_inv * self.n * self.phi_t * log(1 + custom_exp((self.V_gsp - (self.V_th_eff - self.alpha * self.phi_t * self.F_f))/(self.n*self.phi_t)))
        # clamp result of softplus so it doesn't go to 0, causing issues in solver
        #self.Q_ix0 = self.C_inv * self.n * self.phi_t * symbolic_convex_max(1e-10,log(1 + exp((self.V_gsp - (self.V_th_eff - self.alpha * self.phi_t * self.F_f))/(self.n*self.phi_t))))
        self.Q_ix0_0 = self.Q_ix0.xreplace({self.V_th_eff: self.V_th_eff_general})
        self.Q_ix0_0 = self.Q_ix0_0.xreplace({self.V_dsp: 0})

    def set_mobility(self):
        self.u_n_eff = self.u_n # TODO: add mobility model

    def init_intrinsics(self):
        self.Cox = self.e_0 * self.base_params.k_gate / self.base_params.tox
        logger.info(f"Cox: {self.Cox.xreplace(self.base_params.tech_values).evalf()}")

        #self.C_inv = (self.Cox * self.base_params.Cs) / (self.Cox + self.base_params.Cs)
        self.C_inv = self.Cox

        self.eot = self.base_params.tox * 3.9/self.base_params.k_gate
        self.set_scale_length()

        self.set_sce_parameters()

        self.set_mobility()

        logger.info(f"delta: {self.delta.xreplace(self.base_params.tech_values).evalf()}")
        self.V_th_eff_general = self.base_params.V_th - self.delta * self.V_dsp
        self.V_th_eff = (self.V_th_eff_general.xreplace(self.NL_state) + self.V_th_eff_general.xreplace(self.H_state)) / 2

        # Calculation for Delta_2 sub-band
        self.mD_delta2 = 2*self.mt*self.m_0
        self.mC_delta2 = 4*self.mt*self.m_0
        self.vT_delta2_int = (2.0*self.K*self.T/math.pi * self.mC_delta2/(self.mD_delta2**2))**0.5

        # Calculation for Delta_4 sub-band
        self.mD_delta4 = 4*(self.ml*self.mt)**0.5*self.m_0
        self.mC_delta4 = 4*(self.mt**0.5 + self.ml**0.5)**2 * self.m_0
        self.vT_delta4_int = (2.0*self.K*self.T/math.pi * self.mC_delta4/(self.mD_delta4**2))**0.5


        self.vT = self.nu * self.vT_delta2_int + (1 - self.nu) * self.vT_delta4_int # Average NDG thermal velocity of electrons by accounting for both the sub-bands
        self.lam = 2*self.phi_t *self.u_n_eff / self.vT # mean free path

        self.N2D = self.mD_delta2/(math.pi*self.h_bar**2)*self.K*self.T

        self.L_ov = self.base_params.L / 8 # TODO replace with proper value
        self.L_eff = self.base_params.L - self.L_ov

        #self.Vsf = self.V_gsp + self.delta * self.V_dsp + 


        self.vx0_32n = 1.35e5 # m/s (vs paper uses 35nm as reference but only data for 32nm)

        self.vx0 = self.vx0_32n + 1e5 * (self.delta - self.delta.xreplace({self.base_params.L: 32e-9}))

        self.set_F_f()

        self.set_vs_charge()

    def set_parasitic_resistances(self):
        self.R_s = self.Rs0/self.base_params.W
        self.R_d = self.R_s

    def set_parasitic_capacitances(self):
        pass

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
        self.set_F_s()
        #self.R_cmin = self.L_c / (self.Q_ix0_0 * self.u_n_eff)
        self.I_d = self.base_params.W * self.Q_ix0 * self.v * self.F_s

        self.I_d_on = ((self.I_d).xreplace(self.NL_state) + (self.I_d).xreplace(self.H_state)) / 2
        #logger.info(f"I_d_on equation: {self.I_d_on.simplify()}")

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
        self.param_db["L_ov"] = self.L_ov
        self.param_db["A_gate"] = self.A_gate
        self.param_db["k_gate"] = self.base_params.k_gate

    def apply_base_parameter_effects(self):
        pass

    def apply_additional_effects(self):
        super().apply_additional_effects()

    def create_constraints(self, dennard_scaling_type="constant_field"):
        super().create_constraints(dennard_scaling_type)
        self.constraints.append(self.delta <= 0.15)
        