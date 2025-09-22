import logging
from src.hardware_model.tech_models.tech_model_base import TechModel
from src.sim_util import symbolic_convex_max, symbolic_convex_min, custom_cosh, custom_exp
import math
from sympy import symbols, ceiling, expand, exp, Abs, cosh, log
import sympy as sp

class BsimCMGModel(TechModel):    
    def __init__(self, model_cfg, base_params):
        super().__init__(model_cfg, base_params)

    def init_tech_specific_constants(self):
        # NOTES
        # using BULKMOD=1 (bulk substrate)
        # using GEOMOD=1 (tri-gate)
        # using TEMPMOD=0 (default)

        self.phi_b = 3.1  # Schottky barrier height (eV)
        self.m_ox = 0.5*self.m_0  # effective mass of electron in oxide (g)
        self.L_critical = 0.5e-6
        self.m = 2

        # 3.1.1
        self.EPSRSUB = self.e_si/self.e_0
        self.e_sub = self.EPSRSUB * self.e_0
        # my param
        self.L_FIN_RATIO = 1/3
        self.W_FIN_RATIO = 1000

        # 3.1.2
        self.EOTACC = 1e-10
        self.LINT = 0
        self.LLN = 1
        self.LL = 0
        self.XL = 0
        self.DLC = 0
        self.LLC = 0
        self.NF = 1
        self.NFIN = 4
        self.DELTAW = 0
        self.DELTAWCV = 0

        # 3.1.3
        self.m_x = 0.916*self.m_e
        self.m_x_prime = 0.190 * self.m_e
        self.m_d = 0.190 * self.m_e
        self.m_d_prime = 0.417 * self.m_e
        self.g_prime = 4.0
        self.g = 2.0

        # 3.1.4
        self.AQMTCEN = 0
        self.BQMTCEN = 1

        # 3.1.6
        self.PHIG = 4.61*self.q # metal gate workfunction, I think it doesnt scale much
        self.PHIGN1 = 0
        self.PHIGN2 = 1e5
        self.NFINNOM = self.NFIN
        self.ETA0 = 0.6
        self.PHIGLT = 0

        self.NBODY = 1.0e22 # m^-3

        self.DVT0 = 0
        self.DVT1 = 0.6
        # set Vbi and phi_s as ballpark values
        # self.NSD = 1e25 # m^-3
        # self.ni = 1e16 # m^-3 <- check this
        self.Vbi = 0.8
        self.phi_s = 0.7
        self.DSUB = 1.06

        self.CDSCD = 7e-3
        self.CDSC = 7e-3
        self.CIT = 0

        self.A = 3.75956e-7 # for NMOS
        self.B = 7.45669e11 # for NMOS
        self.AIGC = 1.36e-2 # for NMOS
        self.BIGC = 1.71e-3 # for NMOS
        self.CIGC = 0.075 # for NMOS
        self.DLCIGS = self.LINT
        self.TOXREF = 1.2e-9 # reference oxide thickness
        self.POXEDGE = 1
        self.NTOX = 1
        self.AIGS = 1.36e-2
        self.BIGS = 1.71e-3
        self.CIGS = 0.075

        self.UP = 0
        self.LPA = 1
        self.UA = 0.3 # (cm^2/MV)^EU
        self.EU = 2.5 # NMOS

        self.EASUB = 4.05*self.q
        self.Eg = 1.12*self.q # at 300K
        self.RDSWMIN = 0
        self.RDSW = 100 # ohm(μm)WR
        self.WR = 1
        self.PRWGS = 0

        self.VSAT = 85000

        self.lam = 1

        self.PCLM = 0.013
        self.PDIBL1 = 1.3
        self.PDIBL1R = self.PDIBL1
        self.PDIBL2 = 2e-4
        self.DROUT = 1.06
        self.Vbseff = 0
        self.PVAG = 1.0


        self.ALPHA0 = 0
        self.ALPHA1 = 0
        self.BETA0 = 0

        self.A_GIDL = 6.055e-12
        self.B_GIDL = 0.3e9
        self.C_GIDL = 0.2
        self.E_GIDL = 0.2

        self.KT1 = 0
        self.KT1L = 0
        self.KT11 = 0.01
        self.KT12 = 0.1
        self.TNOM = 300

        self.UTE = 0
        self.UTL = -1.5e-3
        self.UTE1 = -0.4

        self.AT = -0.00156
        self.AT2 = 2.0e-6
    
    def init_transistor_equations(self):
        super().init_transistor_equations()

        exps = []
        coshs = []
        logs = []

        # 3.1.1
        self.e_ox = self.base_params.k_gate*self.e_0  if self.model_cfg["effects"]["high_k_gate"] else 3.9*self.e_0

        self.EPSROX = self.base_params.k_gate
        self.TOXP = self.TOXE * self.EPSROX/3.9
        self.EOT = self.base_params.tox
        self.Cox = 3.9*self.e_0/self.EOT
        self.TFIN = self.base_params.L * self.L_FIN_RATIO
        self.HFIN = 1e-9 * self.W_FIN_RATIO/(self.base_params.L * 1e9) # convert to nano, then back to standard units
        self.Csi = self.e_sub/self.TFIN
        self.e_ratio = self.EPSRSUB/3.9

        # 3.1.2
        # bulkmod = 1
        self.delta_L = self.LINT + self.LL/(self.base_params.L + self.XL)**self.LLN
        self.Leff = self.base_params.L + self.XL - 2*self.delta_L
        self.delta_L_CV = self.LINT + self.LLC/(self.base_params.L + self.XL)**self.LLN
        self.Leff_CV = self.base_params.L + self.XL - 2*self.delta_L_CV
        self.COX_ACC = self.Cox * self.EOT/self.EOTACC
        self.NFIN_total = self.NF * self.NFIN
        # geomod=1
        self.WEFF_UFCM = 2*self.HFIN + self.TFIN
        self.ACH = self.HFIN * self.TFIN
        self.Weff0 = self.WEFF_UFCM - self.DELTAW
        self.WeffCV0 = self.WEFF_UFCM - self.DELTAWCV
        
        # 3.1.4 quantum mechanical effects
        #geomod=1
        self.MT_cen = 1 + self.AQMTCEN * custom_exp(-self.TFIN/self.BQMTCEN)
        exps.append(-self.TFIN/self.BQMTCEN)
        self.Tcen0 = self.TFIN * self.MT_cen

        # 3.1.5 binning calculations skip for now

        # 3.1.6 NFIN scaling equations
        self.PHIG_L_N = self.PHIG * (1 + self.PHIGN1/self.NFIN * log(1 + self.NFIN/self.PHIGN2)) * (1 + (self.NFIN - self.NFINNOM)*self.PHIGLT * self.Leff)


        # INCOMPLETE


        
        

    def apply_base_parameter_effects(self):
        return

    def apply_additional_effects(self):
        super().apply_additional_effects()

    def create_constraints(self, dennard_scaling_type="constant_field"):
        super().create_constraints(dennard_scaling_type)

        self.constraints.append(self.V_ox >= 0)
        self.constraints.append(self.dVth_SCE + self.dVth_DIBL >= -0.2)
        self.constraints.append(self.dVth_nw_1 + self.dVth_nw_2 <= 0.2)
        self.constraints.append(self.base_params.L <= self.base_params.W)
