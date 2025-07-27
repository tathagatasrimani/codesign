import logging
from src.tech_model import TechModel
from src.sim_util import symbolic_convex_max, symbolic_convex_min
import math
from sympy import symbols, ceiling, expand, exp, Abs, cosh, log
import sympy as sp


logger = logging.getLogger(__name__)

#########################
# BSIM4 Model
#########################
# DOCUMENT ASSUMPTIONS





class BulkBSIM4Model(TechModel):
    def __init__(self, tech_type, model_cfg, base_params):
        super().__init__(tech_type, model_cfg, base_params)

    def init_tech_specific_constants(self):
        self.phi_b = 3.1  # Schottky barrier height (eV)
        self.m_ox = 0.5*self.m_0  # effective mass of electron in oxide (g)
        self.L_critical = 0.5e-6
        self.m = 2

        #1.1
        self.EPSRSUB = self.e_si

        #2.1
        self.NDEP = 1.7e23 #m^-3, original value is 1.7e17 cm^-3

        #2.4
        self.DVT0 = 2.2
        self.DVT1 = 0.53
        self.Vbi = 0.9
        self.phi_s = 0.4
        self.ETA0 = 0.08
        self.DSUB = 0.56 # = DROUT
        self.K3 = 80 # narrow width coefficient
        self.W0 = 2.5e-6
        self.DVT0W = 0 # can experiment with this
        self.DVT1W = 5.3e6

        #3.1
        self.VOFF = -0.1
        self.VOFFL = 0
        self.ADOS = 1
        self.BDOS = 1
        self.VFB = -1.0

        #3.2
        self.NFACTOR = 1
        self.CDSCD = 0
        self.CDSC = 2.4e-4
        self.CIT = 0
        self.t_dep = 1 # TODO is this ok?
        # m_star = 0.5 + arctan(MINV)/pi but MINV = 0
        self.m_star = 0.5

        #4.1
        self.K1 = 0 # body bias coefficient, set to 0
        self.Vgb = 0
        self.K_lox = 1 # not defined anywhere

        #4.3
        self.A = 4.97232 # for NMOS
        self.B = 7.45669e11 # for NMOS
        self.AIGC = 1.36e-2 # for NMOS
        self.BIGC = 1.71e-3 # for NMOS
        self.CIGC = 0.75 # for NMOS
        self.NIGC = 1.0
        self.DELTA = 0.01

        #5.2
        self.UP = 0
        self.LP = 1.0e-8
        self.UA = 1.0e-15
        self.C0 = 2 # for nmos
        self.EU = 1.67 # NMOS
        self.UD = 0 # mobility scattering coeff, leaving this bit out cuz its 0

        #5.3
        self.PHIG = 4.5*self.q # metal gate workfunction, I think it doesnt scale much
        self.EASUB = 4.05*self.q
        self.Eg = 1.12*self.q # at 300K
        self.RDSWMIN = 0
        self.RDSW = 200 # ohm(μm)WR
        self.WR = 1
        self.PRWB = 0
        self.PRWG = 1

        #5.5
        self.VSAT = 8.0e4

        #5.6
        self.lam = 1
        self.XJ = 1.5e-7 # junction depth, apparently can't be determined very accurately so relies on PCLM to compensate

        # 5.7
        self.PCLM = 1.3
        self.FPROUT = 0
        self.PDIBLC1 = 0.39
        self.PDIBLC2 = 0.0086
        self.PDIBLCB = 0
        self.DROUT = 0.56
        self.Vbseff = 0
        self.PVAG = 0 

        # 5.8
        self.NF = 1

        # 6.1
        self.ALPHA0 = 0.1
        self.ALPHA1 = 0.01
        self.BETA0 = 2.0

        # 6.2
        self.A_GIDL = 1e-12
        self.B_GIDL = 2.3e9
        self.C_GIDL = 0.5
        self.E_GIDL = 0.8

        # 13.1
        self.KT1 = -0.11
        self.KT1L = 0
        self.KT2 = 0.022
        self.TNOM = 300

        # 13.2
        self.UTE = -1.5

        # 13.3
        self.AT = 3.3e4


    def init_transistor_equations(self):
        super().init_transistor_equations()

        # set up some generic variables
        self.e_ox = self.base_params.k_gate*self.e_0  if self.model_cfg["effects"]["high_k_gate"] else 3.9*self.e_0
        self.Cox = self.e_ox/self.base_params.tox

        # 1.1 gate dielectric. set mrtlMod=1
        self.EPSROX = self.base_params.k_gate
        self.TOXE = self.e_sio2/self.Cox
        self.TOXP = self.base_params.tox
        self.EOT = self.TOXE
        self.VDDEOT = self.base_params.V_dd

        # skip 1.2 (seemingly irrelevant) and 1.3 (just use W and L directly)

        # 2.1 long channel Vth = Vth0. ignore 2.2 and 2.3
        self.Coxe = self.e_ox/self.TOXE
        self.N_substrate = self.NDEP

        # 2.4 SCE and DIBL
        self.X_dep = (2*self.e_si*self.phi_s/(self.q*self.NDEP))**(1/2)
        self.lt = ((self.e_si*self.base_params.tox * self.X_dep)/(self.EPSROX))**(1/2)
        self.theta_SCE = (0.5*self.DVT0)/(cosh(symbolic_convex_min(10, self.DVT1*self.base_params.L/self.lt))-1)
        self.dVth_SCE = -self.theta_SCE * (self.Vbi - self.phi_s)
        self.theta_DIBL = 0.5/(cosh(symbolic_convex_min(10, self.DSUB*self.base_params.L/self.lt))-1)
        self.dVth_DIBL = -self.theta_DIBL * self.ETA0 * self.base_params.V_dd

        # 2.5 narrow width effect
        self.dVth_nw_1 = self.K3 * self.phi_s * (self.TOXE/(self.base_params.W - self.W0))
        self.dVth_nw_2 = -(0.5*self.DVT0W) / (cosh(symbolic_convex_min(10, self.DVT1W*self.base_params.L*self.base_params.W/self.lt))-1) * (self.Vbi - self.phi_s)

        # overall Vth
        self.V_th_eff = self.base_params.V_th + self.dVth_SCE + self.dVth_DIBL + self.dVth_nw_1 + self.dVth_nw_2

        # 13.1 temperature dependance of Vth
        self.dVth_T = (self.KT1 + self.KT1L/self.base_params.L + self.KT2*self.Vbseff) * (self.T / self.TNOM - 1)
        self.V_th_eff += self.dVth_T
        print(f"dVth_T: {self.dVth_T.xreplace(self.base_params.tech_values)}")

        self.dVfb_T = -self.KT1 * (self.T / self.TNOM - 1)
        self.VFB += self.dVfb_T
        if (type(self.dVfb_T) == sp.core.expr.Expr):
            print(f"dVfb_T: {self.dVfb_T.xreplace(self.base_params.tech_values)}")
        else:
            print(f"dVfb_T: {self.dVfb_T}")

        # 3.2 subthreshold swing
        self.Cdsc_term = self.CDSC*(0.5/(cosh(symbolic_convex_min(10, self.DVT1*self.base_params.L/self.lt))-1))
        self.Cdep = self.e_si/self.t_dep
        self.n = 1 + self.NFACTOR * (self.Cdep/self.Cox) * ((self.Cdsc_term + self.CIT)/self.Coxe)

        # 3.1 channel charge
        self.Vgse_on = self.base_params.V_dd
        self.Vgse_off = -self.VOFF
        self.V_off_prime = self.VOFF + self.VOFFL/self.base_params.L
        # can try also Vgsteff = Vgse_on - V_th_eff and same for off
        self.Vgsteff = self.n * self.V_T * log(1 + exp(symbolic_convex_min(10, (self.m_star*(self.Vgse_on - self.V_th_eff))/(self.n*self.V_T))))/ (self.m_star + self.n*self.Coxe * (2*self.phi_s/(self.q*self.NDEP*self.e_si))**(1/2) * exp(symbolic_convex_min(10, -((1-self.m_star)*(self.Vgse_on - self.V_th_eff) - self.V_off_prime)/(self.n*self.V_T))))
        self.Vgsteff_off = self.n * self.V_T * log(1 + exp(symbolic_convex_min(10, (self.m_star*(self.Vgse_off - self.V_th_eff))/(self.n*self.V_T))))/ (self.m_star + self.n*self.Coxe * (2*self.phi_s/(self.q*self.NDEP*self.e_si))**(1/2) * exp(symbolic_convex_min(10, -((1-self.m_star)*(self.Vgse_off - self.V_th_eff) - self.V_off_prime)/(self.n*self.V_T))))
        self.X_DC = (self.ADOS*1.9e-9)/(1+((self.Vgsteff + 4 * (self.base_params.V_th - self.VFB - self.phi_s))/(2*self.TOXP))**(0.7*self.BDOS))
        self.Ccen = self.e_si/self.X_DC
        self.Coxeff = (self.Coxe*self.Ccen)/(self.Coxe + self.Ccen)

        #5.1
        self.A_bulk = 0.1 # not touching that equation.

        self.Vb = (self.Vgsteff + 2*self.V_T)/self.A_bulk
        self.Qch0 = self.Coxeff * self.Vgsteff


        # gate leakage, we are modeling in OFF state
        # 4.2 Vox
        self.Vfbzb = self.base_params.V_th - self.phi_b - self.K1 * (self.phi_s**(1/2))
        self.VFBeff = self.Vfbzb - 0.5 * ((self.Vfbzb - self.Vgb - 0.02) + ((self.Vfbzb - self.Vgb - 0.02)**2 + 0.08*self.Vfbzb)**(1/2))
        self.Voxacc = self.Vfbzb - self.VFBeff
        self.Voxdepinv = self.K_lox * (self.phi_s)**(1/2) + self.Vgsteff
        self.V_ox = self.Voxacc + self.Voxdepinv

        # 5.2 unified mobility model
        # assume Vbs = 0
        self.f_L_eff = 1 - self.UP * exp(symbolic_convex_min(10, self.base_params.L/self.LP))
        self.u_n_eff = self.base_params.u_n * self.f_L_eff / (1 + self.UA*((self.Vgsteff + self.C0 * (self.base_params.V_th - self.VFB - self.phi_s))/ self.TOXE)**self.EU)
        self.u_p_eff = self.u_n_eff / 2.5

        self.E_eff = 3.9 * (self.Vgsteff + 2*self.V_th_eff - 2 * (self.PHIG - self.EASUB - self.Eg/2 + 0.45)) / (self.EOT * self.EPSRSUB)

        # 13.2 temperature dependance of mobility
        self.du_T = (self.T/self.TNOM)**(self.UTE)
        self.u_n_eff *= self.du_T
        self.u_p_eff *= self.du_T

        # 13.3 temperature dependance of vsat
        self.dvsat_T = -self.AT * (self.T/self.TNOM - 1)
        self.VSAT += self.dvsat_T
        if (type(self.dvsat_T) == sp.core.expr.Expr):
            print(f"dvsat_T: {self.dvsat_T.xreplace(self.base_params.tech_values)}")
        else:
            print(f"dvsat_T: {self.dvsat_T}")

        # 5.3 source drain resistance. ignore for now #TODO
        self.Rds = (self.RDSWMIN + self.RDSW * (1/(1+self.PRWG*self.Vgsteff))) / (1e6*self.base_params.W)**self.WR
        #5.5 velocity saturation
        self.E_sat = 2*self.VSAT/self.u_n_eff
        self.E_sat_p = 2*self.VSAT/self.u_p_eff
        #5.4 drain current triode
        self.I_ds0 = self.u_n_eff * self.base_params.W/self.base_params.L * self.Qch0 * self.base_params.V_dd * (1-self.base_params.V_dd/(2*self.Vb))/(1+self.base_params.V_dd/(self.E_sat*self.base_params.L))
        self.I_ds0_p = self.u_p_eff * self.base_params.W/self.base_params.L * self.Qch0 * self.base_params.V_dd * (1-self.base_params.V_dd/(2*self.Vb))/(1+self.base_params.V_dd/(self.E_sat_p*self.base_params.L))
        print(f"self.Vb: {self.Vb.xreplace(self.base_params.tech_values)}")
        
        # 5.6 Vdsat
        # external
        self.a = self.A_bulk**2 * self.base_params.W * self.VSAT * self.Coxe * self.Rds + self.A_bulk * (1/self.lam - 1)
        self.b = -((self.Vgsteff + 2*self.V_T)*(2/self.lam - 1) + self.A_bulk * self.E_sat * self.base_params.L + 3 * self.A_bulk * (self.Vgsteff + 2*self.V_T) * self.base_params.W * self.VSAT * self.Coxe * self.Rds)
        self.c = (self.Vgsteff + 2 * self.V_T)*self.E_sat * self.base_params.L + 2 * (self.Vgsteff + 2 * self.V_T)**2 * self.base_params.W * self.VSAT * self.Coxe * self.Rds
        self.b_p = -((self.Vgsteff + 2*self.V_T)*(2/self.lam - 1) + self.A_bulk * self.E_sat_p * self.base_params.L + 3 * self.A_bulk * (self.Vgsteff + 2*self.V_T) * self.base_params.W * self.VSAT * self.Coxe * self.Rds)
        self.c_p = (self.Vgsteff + 2 * self.V_T)*self.E_sat_p * self.base_params.L + 2 * (self.Vgsteff + 2 * self.V_T)**2 * self.base_params.W * self.VSAT * self.Coxe * self.Rds
        self.Vdsat = (-self.b - (self.b**2 - 4*self.a*self.c)**(1/2)) / (2*self.a)
        self.Vdsat_p = (-self.b_p - (self.b_p**2 - 4*self.a*self.c_p)**(1/2)) / (2*self.a)
        self.litl = (self.e_si*self.TOXE*self.XJ/(self.e_0*self.EPSROX))**(1/2) # UNITS SEEM WRONG IN MANUAL???
        print(f"litl calculation: e_si={self.e_si}, TOXE={self.TOXE}, XJ={self.XJ}, EPSROX={self.EPSROX}")
        print(f"litl: {self.litl.xreplace(self.base_params.tech_values)}")
        print(f"TOXE: {self.TOXE.xreplace(self.base_params.tech_values)}")
        print(f"esi/e0: {self.e_si/self.e_0}")
        print(f"intermediate: {self.e_si*self.TOXE/self.EPSROX}")
        print(f"intermediate sub: {(self.e_si*self.TOXE/self.EPSROX).xreplace(self.base_params.tech_values)}")

        # 4.3 tunneling (need to know Vdsat for Vdseff)
        # ignore 4.3.1 gate to substrate
        # 4.3.2 gate to channel, gate to s/d
        self.Toxratio = self.TOXP/self.TOXE
        # use igcMod = 2
        self.Vaux = self.NIGC * self.V_T * log(1 + exp((self.Vgse_off - self.V_th_eff)/(self.NIGC * self.V_T)), 10)
        self.Igc0 = self.base_params.W * self.base_params.L * self.A * self.Toxratio * self.Vgse_off * self.Vaux * exp(symbolic_convex_min(10, -self.B*self.TOXE*(self.AIGC - self.BIGC * self.Voxdepinv)*(1+self.CIGC*self.Voxdepinv)))
        self.Vdseff = self.Vdsat - 0.5*(self.Vdsat-self.base_params.V_dd-self.DELTA + ((self.Vdsat-self.base_params.V_dd-self.DELTA)**2 + 4*self.DELTA*self.Vdsat)**(1/2))
        self.PIGCD = (self.B * self.TOXE / (self.Vgsteff_off**2)) * (1 - self.Vdseff/(2*self.Vgsteff_off))
        self.Igcs = self.Igc0 * (self.PIGCD * self.Vdseff * exp(symbolic_convex_min(10, -self.PIGCD * self.Vdseff)) - 1 + 1e-4) / (self.PIGCD**2 * self.Vdseff**2 + 2e-4)
        self.Igcd = self.Igc0 * (1 - (self.PIGCD * self.Vdseff + 1) * exp(symbolic_convex_min(10, -self.PIGCD * self.Vdseff)) + 1e-4) / (self.PIGCD**2 * self.Vdseff**2 + 2e-4)
        self.Igc = self.Igcs + self.Igcd
        self.I_tunnel = self.Igc
        # for now, skipping Igs and Igd. Come back to it #TODO

        # internal
        #self.Vdsat = (self.E_sat * self.base_params.L * (self.Vgsteff + 2*self.V_T)) / (self.A_bulk * self.E_sat * self.base_params.L + self.Vgsteff + 2 * self.V_T)
        #self.Vdsat_p = (self.E_sat_p * self.base_params.L * (self.Vgsteff + 2*self.V_T)) / (self.A_bulk * self.E_sat_p * self.base_params.L + self.Vgsteff + 2 * self.V_T)
        # 5.7 output conductance model
        self.F = 1/(1+self.FPROUT * (self.base_params.L**(1/2))/(self.Vgsteff + 2*self.V_T))
        self.C_clm_n = 1/self.PCLM * self.F * (1 + self.PVAG * (self.Vgsteff/(self.base_params.L*self.E_sat))) * (1 + self.Rds*self.I_ds0/self.Vdseff)*(self.base_params.L + self.Vdsat/self.E_sat) / self.litl
        self.C_clm_p = 1/self.PCLM * self.F * (1 + self.PVAG * (self.Vgsteff/(self.base_params.L*self.E_sat_p))) * (1 + self.Rds*self.I_ds0_p/self.Vdseff)*(self.base_params.L + self.Vdsat_p/self.E_sat_p) / self.litl

        print(f"self.E_sat: {self.E_sat.xreplace(self.base_params.tech_values)}")
        print(f"self.Vgsteff: {self.Vgsteff.xreplace(self.base_params.tech_values)}")
        print(f"self.Vgsteff_off: {self.Vgsteff_off.xreplace(self.base_params.tech_values)}")
        print(f"self.Qch0: {self.Qch0.xreplace(self.base_params.tech_values)}")
        print(f"self.Voxdepinv: {self.Voxdepinv.xreplace(self.base_params.tech_values)}")

        # TODO: check value of vdd vs vdsat
        print(f"V_dd: {self.base_params.V_dd.xreplace(self.base_params.tech_values)}")
        print(f"Vdsat: {self.Vdsat.xreplace(self.base_params.tech_values)}")
        print(f"Vdseff: {self.Vdseff.xreplace(self.base_params.tech_values)}")
        self.V_aclm_n = self.C_clm_n * (self.base_params.V_dd - self.Vdsat)
        self.V_aclm_p = self.C_clm_p * (self.base_params.V_dd - self.Vdsat_p)

        print(f"self.V_aclm_n: {self.V_aclm_n.xreplace(self.base_params.tech_values)}")

        self.theta_rout = self.PDIBLC1/(2*cosh(symbolic_convex_min(10, self.DROUT*self.base_params.L/self.lt)) - 2) + self.PDIBLC2
        self.V_adibl_n = (self.Vgsteff + 2*self.V_T)/(self.theta_rout*(1+self.PDIBLCB*self.Vbseff)) * (1-(self.A_bulk*self.Vdsat/(self.A_bulk*self.Vdsat + self.Vgsteff + 2*self.V_T))) * (1 + self.PVAG*self.Vgsteff/(self.E_sat*self.base_params.L))
        self.V_adibl_p = (self.Vgsteff + 2*self.V_T)/(self.theta_rout*(1+self.PDIBLCB*self.Vbseff)) * (1-(self.A_bulk*self.Vdsat_p/(self.A_bulk*self.Vdsat_p + self.Vgsteff + 2*self.V_T))) * (1 + self.PVAG*self.Vgsteff/(self.E_sat_p*self.base_params.L))

        print(f"self.V_adibl_n: {self.V_adibl_n.xreplace(self.base_params.tech_values)}")

        # A and B may need to be empirically scaled (A down, B up)
        self.A_i = 0.5
        self.B_i = 1
        self.V_ascbe = self.B_i / self.A_i * exp(symbolic_convex_min(10, self.B_i * self.litl / (self.base_params.V_dd - self.Vdsat)))

        print(f"self.V_ascbe: {self.V_ascbe.xreplace(self.base_params.tech_values)}")

        # ignoring 5.7.4 pocket implant for now

        #5.8 single equation channel current model (I only care about saturation with VDD)
        self.V_asat_n = (self.E_sat*self.base_params.L + self.Vdsat + 2*self.Rds*self.VSAT*self.Coxe*self.base_params.W*self.Vgsteff*(1-self.A_bulk*self.Vdsat/(2*(self.Vgsteff+2*self.V_T)))) / (self.Rds*self.VSAT*self.Coxe*self.base_params.W*self.A_bulk - 1 + 2/self.lam)
        self.V_asat_p = (self.E_sat_p*self.base_params.L + self.Vdsat_p + 2*self.Rds*self.VSAT*self.Coxe*self.base_params.W*self.Vgsteff*(1-self.A_bulk*self.Vdsat_p/(2*(self.Vgsteff+2*self.V_T)))) / (self.Rds*self.VSAT*self.Coxe*self.base_params.W*self.A_bulk - 1 + 2/self.lam)

        self.V_a = self.V_asat_n + self.V_aclm_n
        self.V_a_p = self.V_asat_p + self.V_aclm_p

        self.I_d_n = (self.I_ds0 * self.NF)/(1+self.Rds*self.I_ds0/self.Vdseff)*(1+(1/self.C_clm_n)*log(self.V_a/self.V_asat_n)) * (1+(self.base_params.V_dd-self.Vdseff)/self.V_adibl_n) * (1 + (self.base_params.V_dd-self.Vdseff)/self.V_ascbe)
        self.I_d_p = (self.I_ds0_p * self.NF)/(1+self.Rds*self.I_ds0_p/self.Vdseff)*(1+(1/self.C_clm_p)*log(self.V_a_p/self.V_asat_p)) * (1+(self.base_params.V_dd-self.Vdseff)/self.V_adibl_p) * (1 + (self.base_params.V_dd-self.Vdseff)/self.V_ascbe)
        self.I_d = self.I_d_n + self.I_d_p

        print(f"I_ds0: {self.I_ds0.xreplace(self.base_params.tech_values)}")
        print(f"self.Rds: {self.Rds.xreplace(self.base_params.tech_values)}")
        print(f"self.Vdseff: {self.Vdseff.xreplace(self.base_params.tech_values)}")
        print(f"self.C_clm_n: {self.C_clm_n.xreplace(self.base_params.tech_values)}")
        print(f"self.V_adibl_n: {self.V_adibl_n.xreplace(self.base_params.tech_values)}")
        print(f"self.V_asat_n: {self.V_asat_n.xreplace(self.base_params.tech_values)}")
        print(f"self.V_a: {self.V_a.xreplace(self.base_params.tech_values)}")
        print(f"self.V_a_p: {self.V_a_p.xreplace(self.base_params.tech_values)}")
        print(f"I_d_n: {self.I_d_n.xreplace(self.base_params.tech_values)}")

        # coming back to subthreshold current from 3.2 now that we have mobility model
        self.I_0_n = self.u_n_eff * self.base_params.W/self.base_params.L * self.V_T**2 * (self.q*self.e_si*self.NDEP/(2*self.phi_s))**(1/2)
        self.I_0_p = self.I_0_n * self.u_p_eff / self.u_n_eff
        self.I_sub_n = self.I_0_n * (1-exp(symbolic_convex_min(10, -self.base_params.V_dd/self.V_T)))*exp(symbolic_convex_min(10, (-self.base_params.V_th - self.V_off_prime)/(self.n*self.V_T)))
        self.I_sub_p = self.I_sub_n * self.I_0_p / self.I_0_n
        self.I_sub = self.I_sub_n + self.I_sub_p

        # 6.1 I_ii impact ionization current.
        self.I_ii_n = (self.ALPHA0 + self.ALPHA1*self.base_params.L)/self.base_params.L * (self.base_params.V_dd-self.Vdseff)*exp(symbolic_convex_min(10, -self.BETA0*(self.base_params.V_dd-self.Vdseff)))*self.I_d_n
        self.I_ii_p = (self.ALPHA0 + self.ALPHA1*self.base_params.L)/self.base_params.L * (self.base_params.V_dd-self.Vdseff)*exp(symbolic_convex_min(10, -self.BETA0*(self.base_params.V_dd-self.Vdseff)))*self.I_d_p
        self.I_ii = self.I_ii_n + self.I_ii_p

        # 6.2 GIDL, ignore GISL
        self.Vdb = self.base_params.V_dd
        self.I_GIDL = self.A_GIDL * self.base_params.W * self.NF * (self.base_params.V_dd - self.Vgse_off - self.E_GIDL)/(3*self.TOXE)*exp(symbolic_convex_min(10, -self.B_GIDL * 3 * self.TOXE / (self.base_params.V_dd - self.Vgse_off - self.E_GIDL))) * (self.Vdb**3)/ (self.C_GIDL * self.Vdb**3)

        # 7.4 intrinsic capacitance
        self.A_gate = self.base_params.W * self.base_params.L
        self.C_gate = self.Coxe * self.A_gate

        # 7.5 Fringing and overlap capacitance
        self.C_fringe = (2*self.EPSROX * self.e_0/math.pi) * log(1+(4.0e-7)/self.TOXE, 10) * self.base_params.W # fringing capacitance
        self.CGSO = 0.6*self.XJ * self.Coxe
        self.C_overlap = self.CGSO * self.base_params.W # just considering gate to source overlap for now

        self.R_avg_inv = self.base_params.V_dd / ((self.I_d_n + self.I_d_p)/2)

        # transistor delay equations
        self.gamma_diff = 1 # for inverter
        self.C_diff = self.C_fringe + self.C_overlap + self.gamma_diff * self.C_gate
        print(f"C_diff: {self.C_diff.xreplace(self.base_params.tech_values)}")
        self.C_load = self.C_gate # gate cap
        print(f"C_load: {self.C_load.xreplace(self.base_params.tech_values)}")
        logger.info(f"C_load: {self.C_load}")
        if self.model_cfg["delay_parasitics"] == "all":
            self.delay = (self.R_avg_inv * (self.C_diff + self.C_wire/2) + (self.R_avg_inv + self.R_wire) * (self.C_wire/2 + self.C_load)) * 1e9  # ns
        elif self.model_cfg["delay_parasitics"] == "Csq only":
            self.delay = (self.R_avg_inv * (self.C_diff + self.C_wire/2) + (self.R_avg_inv) * (self.C_wire/2 + self.C_load)) * 1e9  # ns
        elif self.model_cfg["delay_parasitics"] == "const":
            self.delay = self.R_avg_inv * (self.C_diff + self.C_load + 0.3e-15 * 100) * 1e9  # ns
        else:
            self.delay = self.R_avg_inv * (self.C_load + self.C_diff) * 1e9

        print(f"delay: {self.delay.xreplace(self.base_params.tech_values).evalf()}")
        print(f"R_avg_inv: {self.R_avg_inv.xreplace(self.base_params.tech_values)}")
        print(f"C_diff: {self.C_diff.xreplace(self.base_params.tech_values)}")
        print(f"C_wire: {self.C_wire.xreplace(self.base_params.tech_values)}")
        print(f"C_load: {self.C_load.xreplace(self.base_params.tech_values)}")

        self.C_tot = self.C_diff + self.C_load + self.C_wire
        # active energy
        self.E_act_inv = (0.5*self.C_tot*self.base_params.V_dd*self.base_params.V_dd) * 1e9  # nJ

        self.I_off = self.I_sub + self.I_ii + self.I_GIDL + self.I_tunnel

        self.P_pass_inv = self.I_off * self.base_params.V_dd

    def apply_base_parameter_effects(self):
        return

    def apply_additional_effects(self):
        return

    def create_constraints(self, dennard_scaling_type="constant_field"):
        super().create_constraints(dennard_scaling_type)

        self.constraints.append(self.V_ox >= 0)