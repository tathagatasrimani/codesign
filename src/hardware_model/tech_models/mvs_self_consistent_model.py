import logging
from src.hardware_model.tech_models.tech_model_base import TechModel
from src.sim_util import symbolic_convex_max, symbolic_min, custom_cosh, custom_exp, xreplace_safe
import math
from sympy import symbols, ceiling, expand, exp, Abs, cosh, log, tan
import sympy as sp
import numpy as np
from scipy.constants import k, e, epsilon_0, hbar, m_e
from joblib import Parallel, delayed
from src.inverse_pass.constraint import Constraint
from src.hardware_model.tech_models.tech_codesign_v0.Patrick_codesign_v0.cmos_inverter_model_kj import cmos_inv_vtc, mvs2_model
from src.hardware_model.tech_models.tech_codesign_v0.Patrick_codesign_v0.Rsd_model_kj import symbolic_Rsd_model_cmg
from src.hardware_model.tech_models.tech_codesign_v0.Patrick_codesign_v0.sce_model_kj import symbolic_sce_model_cmg
from src.hardware_model.tech_models.tech_codesign_v0.Patrick_codesign_v0.sce_model_kj import get_Lscale
from src.hardware_model.tech_models.tech_codesign_v0.Patrick_codesign_v0.delay_area_power_model_kj import symbolic_area_model
from src.hardware_model.tech_models.tech_codesign_v0.Patrick_codesign_v0.mvs_model_kj import get_mvs_model
from src.hardware_model.tech_models.tech_codesign_v0.Patrick_codesign_v0.Cpar_model_kj import symbolic_Cpar_model_cmg
logger = logging.getLogger(__name__)
debug = False

class MVSSelfConsistentModel(TechModel):
    def __init__(self, model_cfg, base_params):
        super().__init__(model_cfg, base_params)

    def init_tech_specific_constants(self):
        super().init_tech_specific_constants()

        # gate tunneling parameters
        self.phi_b = 3.1  # Schottky barrier height (eV)
        self.m_ox = 0.5*self.m_0  # effective mass of electron in oxide (g)
        self.A = ((self.q)**3) / (8*math.pi*self.h*self.phi_b*self.m_ox)
        self.B = (8*math.pi*(2*self.m_ox)**(1/2) * (self.phi_b*self.q)**(3/2)) / (3*self.q*self.h)

    def get_gate_leakage_current(self, V_dd, V_th, tox, A_gate):
        # gate tunneling current (Fowler-Nordheim and WKB)
        # minimums are to avoid exponential explosion in solver. Normal values in exponent are negative.
        # gate tunneling
        #self.V_ox = symbolic_convex_max(self.V_dd - self.V_th_eff, self.V_th_eff).xreplace(self.off_state)
        V_ox = V_dd - V_th
        E_ox = Abs(V_ox/tox)
        FN_term = A_gate * self.A * E_ox**2 * (custom_exp(-self.B/E_ox))
        WKB_term = A_gate * self.A * E_ox**2 * (custom_exp(-self.B*(1-(1-V_ox/self.phi_b)**3/2)/E_ox))
        I_tunnel = FN_term + WKB_term
        return I_tunnel

    def mvs2_wrapper(self, Vdd, Vt0, Lg, Wg, beta_p_n, mD_fac, mu_eff_n, mu_eff_p, eps_gox, tgox, eps_semi, tsemi, Lext, Lc, eps_cap, rho_c_n, rho_c_p, Rsh_c_n, Rsh_c_p, Rsh_ext_n, Rsh_ext_p):
        Wc_n = Wg
        Wext_n = 2 * Wg
        Rs_n = symbolic_Rsd_model_cmg(Lc, Lext, Wc_n, Wext_n, rho_c_n, Rsh_c_n, Rsh_ext_n)
        Rd_n = symbolic_Rsd_model_cmg(Lc, Lext, Wc_n, Wext_n, rho_c_n, Rsh_c_n, Rsh_ext_n)

        Wc_p = beta_p_n * Wg
        Wext_p = 2 * beta_p_n * Wg
        Rs_p = symbolic_Rsd_model_cmg(Lc, Lext, Wc_p, Wext_p, rho_c_p, Rsh_c_p, Rsh_ext_p)
        Rd_p = symbolic_Rsd_model_cmg(Lc, Lext, Wc_p, Wext_p, rho_c_p, Rsh_c_p, Rsh_ext_p)

        Lscale =  get_Lscale(eps_semi, eps_gox, tgox, tsemi)

        Leff = Lg
        Weff_Id_n = 2 * Wg
        Weff_Id_p = 2 * beta_p_n * Wg
        n0, delta, dVt = symbolic_sce_model_cmg(Leff, Vt0, Lscale)
        Cgc_on = eps_gox * self.e_0 / tgox
        mD = mD_fac * self.m_0
        vT = math.sqrt(2 * self.K*self.T * mD / (math.pi * mD**2))
        return get_mvs_model(Vt0, Leff, Weff_Id_n, mD, mu_eff_n, vT, Cgc_on, n0, delta, dVt, Rs_n, Rd_n), get_mvs_model(Vt0, Leff, Weff_Id_p, mD, mu_eff_p, vT, Cgc_on, n0, delta, dVt, Rs_p, Rd_p)

    def calculate_C(self, Lg, Wg, Lext, eps_cap, eps_gox, tgox, beta_p_n, FO, M):
        Cgc_on = eps_gox * self.e_0 / tgox
        tgate = 2 * Lg
        Weff_Cpar_n = 2 * Wg
        Cpar_n = symbolic_Cpar_model_cmg(Weff_Cpar_n, Lext, eps_cap, tgate)
        Weff_Cpar_p = 2 * beta_p_n * Wg
        Cpar_p = symbolic_Cpar_model_cmg(Weff_Cpar_p, Lext, eps_cap, tgate)

        # modified from original; non-FO4 Cpar terms moved to other variable
        Cload_n = FO * ( (2/3) * Cgc_on * Weff_Cpar_n * Lg + Cpar_n )
        Cload_p = FO * ( (2/3) * Cgc_on * Weff_Cpar_p * Lg + Cpar_p )
        Cload = Cload_n + Cload_p
        
        Cpar = M * (Cpar_n + Cpar_p)

        return Cload, Cpar

    def plot_vtc(self, Vin_vals, Vout_vals):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,5))
        plt.plot(Vin_vals, Vout_vals, label='Vout', lw=3.5)
        plt.plot(Vin_vals, Vin_vals, '--', color='gray', alpha=0.5, label='y=x')
        plt.xlabel('Vin (V)')
        plt.ylabel('Vout (V)')
        plt.title('CMOS Inverter VTC')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"cmos_inverter_vtc_L{self.L}_W{self.W}_Vdd{self.V_dd}_Vth{self.V_th}_tox{self.tox}.png", dpi=300)
        plt.close()

    def init_transistor_equations(self):
        super().init_transistor_equations()
        for value in self.base_params.tech_values:
            if hasattr(value, 'name'):
                setattr(self, value.name, self.base_params.tech_values[value])
        self.mvs2_model_n, self.mvs2_model_p = self.mvs2_wrapper(
            self.V_dd, 
            self.V_th, 
            self.L, 
            self.W, 
            self.beta_p_n, 
            self.mD_fac, 
            self.mu_eff_n, 
            self.mu_eff_p, 
            self.k_gate, 
            self.tox,
            self.eps_semi, 
            self.tsemi, 
            self.Lext, 
            self.Lc, 
            self.eps_cap, 
            self.rho_c_n, 
            self.rho_c_p, 
            self.Rsh_c_n, 
            self.Rsh_c_p, 
            self.Rsh_ext_n, 
            self.Rsh_ext_p,
        )
        self.area = symbolic_area_model(self.L, self.W, self.beta_p_n, self.Lext, self.Lc)

        self.Ieff_n, self.Ieff_p = self.mvs2_model_n.get_Ieff(self.V_dd), self.mvs2_model_p.get_Ieff(self.V_dd)
        self.Ioff_n, self.Ioff_p = self.mvs2_model_n.get_Ioff(self.V_dd), self.mvs2_model_p.get_Ioff(self.V_dd)

        # print("\nDebugging CMOS Inverter VTC Calculation:")
        # Vin_test = 0
        # vout_test = cmos_vtc_accurate_vout(Vin_test)
        # print(f"Vout at Vin = {Vin_test}:", vout_test)

        # Vin_vals setup
        Vin_vals = np.linspace(0.0, self.V_dd, 20)

        # Parallel execution
        # n_jobs=-1 uses all available CPU cores
        results = [cmos_inv_vtc(Vin, self.V_dd, self.V_th, self.L, self.W, self.beta_p_n, self.mD_fac, self.mu_eff_n, self.mu_eff_p, self.k_gate, self.tox, self.eps_semi, self.tsemi, self.Lext, self.Lc, self.eps_cap, self.rho_c_n, self.rho_c_p, self.Rsh_c_n, self.Rsh_c_p, self.Rsh_ext_n, self.Rsh_ext_p) for Vin in Vin_vals]

        #print(f"results: {results}")

        Vout_vals = np.array(results)

        self.crossing_idx = np.argmin(np.abs(Vin_vals - Vout_vals))

        # Compute slope using central difference (or np.gradient)
        slopes = np.gradient(Vout_vals, Vin_vals)
        self.slope_at_crossing = slopes[self.crossing_idx]
        log_info(f"slope_at_crossing: {self.slope_at_crossing}")
        log_info(f"slopes: {slopes}")
        log_info(f"Vin_vals: {Vin_vals}")
        log_info(f"Vout_vals: {Vout_vals}")

        # Calculate noise margin from VTC
        # V_IL: where slope crosses from above -1 to below -1 (magnitude crosses from < 1 to > 1)
        # V_IH: where slope crosses from below -1 to above -1 (magnitude crosses from > 1 to < 1)
        slope_above_neg1 = slopes > -1  # True where |slope| < 1

        # Find V_IL: first transition from slope > -1 to slope < -1
        V_IL = None
        V_IH = None
        for i in range(len(slopes) - 1):
            if slope_above_neg1[i] and not slope_above_neg1[i + 1]:
                # Interpolate to find exact crossing point
                t = (-1 - slopes[i]) / (slopes[i + 1] - slopes[i])
                V_IL = Vin_vals[i] + t * (Vin_vals[i + 1] - Vin_vals[i])
                log_info(f"V_IL: {V_IL}")
                break

        # Find V_IH: first transition from slope < -1 to slope > -1 (after V_IL)
        for i in range(len(slopes) - 1):
            if not slope_above_neg1[i] and slope_above_neg1[i + 1]:
                # Interpolate to find exact crossing point
                t = (-1 - slopes[i]) / (slopes[i + 1] - slopes[i])
                V_IH = Vin_vals[i] + t * (Vin_vals[i + 1] - Vin_vals[i])
                log_info(f"V_IH: {V_IH}")
                break

        # V_OH and V_OL: intersection points of VTC with the line Vout = Vdd - Vin
        # Find where (Vout - (Vdd - Vin)) crosses zero
        unity_gain_diff = Vout_vals - (self.V_dd - Vin_vals)
        log_info(f"unity_gain_diff: {unity_gain_diff}")

        V_OH = None
        V_OL = None
        crossings = []
        if unity_gain_diff[0] > 0: # special case if first value already above the line, otherwise we miss the first crossing
            crossings.append((Vin_vals[0], Vout_vals[0]))
        for i in range(len(unity_gain_diff) - 1):
            if unity_gain_diff[i] * unity_gain_diff[i + 1] <= 0:  # Sign change indicates crossing
                # Interpolate to find exact crossing point
                t = -unity_gain_diff[i] / (unity_gain_diff[i + 1] - unity_gain_diff[i])
                Vin_cross = Vin_vals[i] + t * (Vin_vals[i + 1] - Vin_vals[i])
                Vout_cross = Vout_vals[i] + t * (Vout_vals[i + 1] - Vout_vals[i])
                crossings.append((Vin_cross, Vout_cross))
        log_info(f"crossings: {crossings}")
        # First crossing (low Vin) gives V_OL, third crossing (high Vin) gives V_OH
        if len(crossings) == 3:
            V_OL = crossings[0][0]
            V_OH = crossings[2][0]
        else:
            logger.warning(f"Expected 3 crossings, found {len(crossings)}: {crossings}")

        # Calculate noise margins
        if V_IL is not None and V_IH is not None and V_OH is not None and V_OL is not None:
            self.V_IL = V_IL
            self.V_IH = V_IH
            self.V_OH = V_OH
            self.V_OL = V_OL
            self.NM_H = V_OH - V_IH  # High noise margin
            self.NM_L = V_IL - V_OL  # Low noise margin
            self.noise_margin = min(self.NM_H, self.NM_L)
        else:
            # If we couldn't find all points, set noise margin to a negative value to indicate invalid
            self.V_IL = V_IL if V_IL is not None else np.nan
            self.V_IH = V_IH if V_IH is not None else np.nan
            self.V_OH = V_OH if V_OH is not None else np.nan
            self.V_OL = V_OL if V_OL is not None else np.nan
            self.NM_H = np.nan
            self.NM_L = np.nan
            self.noise_margin = -1.0  # Invalid design

        if debug:
            self.plot_vtc(Vin_vals, Vout_vals)

        self.Lscale = get_Lscale(self.eps_semi, self.k_gate, self.tox, self.tsemi)
        self.n0, self.delta, self.dVt = symbolic_sce_model_cmg(self.L, self.V_th, self.Lscale)
        self.V_th_eff = self.V_th - self.dVt - self.delta * self.V_dd
        self.A_gate = self.L * self.W

        self.I_tunnel = self.get_gate_leakage_current(self.V_dd, self.V_th_eff, self.tox, self.A_gate)
        self.I_sub = (self.Ioff_n + self.Ioff_p)/2
        self.I_off = (self.I_sub + self.I_tunnel)/2
        self.Ieff = (self.Ieff_n + self.Ieff_p)/2

        self.C_load, self.C_par = self.calculate_C(self.L, self.W, self.Lext, self.eps_cap, self.k_gate, self.tox, self.beta_p_n, self.FO, self.M)

        self.E_act_inv = (0.5*(self.C_load + self.C_par + self.C_wire)*self.V_dd*2) * 1e9  # nJ
        self.P_pass_inv = self.I_off * self.V_dd
        self.R_avg_inv = 2*self.V_dd/(self.Ieff)

        self.delay = (self.R_avg_inv * (self.C_par + self.C_wire/2) + (self.R_avg_inv + self.R_wire) * (self.C_wire/2 + self.C_load)) * 1e9  # ns

        self.apply_additional_effects()

        self.config_param_db()

        self.config_sweep_output_db()

        self.create_constraints()

        self.config_pareto_metric_db()

    def config_param_db(self):
        self.param_db["I_tunnel_per_um"] = self.I_tunnel / (self.W* 1e6)
        self.param_db["I_off_per_um"] = self.I_off / (self.W* 1e6)
        self.param_db["I_on_per_um"] = (self.Ieff_n + self.Ieff_p) / (2*self.W* 1e6)
        self.param_db["I_sub_per_um"] = self.I_sub / (self.W* 1e6)
        self.param_db["A_gate"] = self.area
        self.param_db["C_wire"] = self.C_wire
        self.param_db["R_wire"] = self.R_wire
        self.param_db["C_load"] = self.C_load
        super().config_param_db()

    def config_sweep_output_db(self):
        self.sweep_output_db["L"] = self.L
        self.sweep_output_db["W"] = self.W
        self.sweep_output_db["V_dd"] = self.V_dd
        self.sweep_output_db["V_th"] = self.V_th
        self.sweep_output_db["tox"] = self.tox
        self.sweep_output_db["beta_p_n"] = self.beta_p_n
        self.sweep_output_db["mD_fac"] = self.mD_fac
        self.sweep_output_db["mu_eff_n"] = self.mu_eff_n
        self.sweep_output_db["mu_eff_p"] = self.mu_eff_p
        self.sweep_output_db["k_gate"] = self.k_gate
        self.sweep_output_db["eps_semi"] = self.eps_semi
        self.sweep_output_db["tsemi"] = self.tsemi
        self.sweep_output_db["Lext"] = self.Lext
        self.sweep_output_db["Lc"] = self.Lc
        self.sweep_output_db["eps_cap"] = self.eps_cap
        self.sweep_output_db["rho_c_n"] = self.rho_c_n
        self.sweep_output_db["rho_c_p"] = self.rho_c_p
        self.sweep_output_db["Rsh_c_n"] = self.Rsh_c_n
        self.sweep_output_db["Rsh_c_p"] = self.Rsh_c_p
        self.sweep_output_db["Rsh_ext_n"] = self.Rsh_ext_n
        self.sweep_output_db["Rsh_ext_p"] = self.Rsh_ext_p
        self.sweep_output_db["FO"] = self.FO
        self.sweep_output_db["M"] = self.M
        self.sweep_output_db["area"] = self.area
        self.sweep_output_db["delay"] = self.delay
        self.sweep_output_db["Edynamic"] = self.E_act_inv
        self.sweep_output_db["Pstatic"] = self.P_pass_inv
        self.sweep_output_db["Ieff"] = self.Ieff
        self.sweep_output_db["Ioff"] = self.I_off
        self.sweep_output_db["V_th_eff"] = self.V_th_eff
        self.sweep_output_db["C_load"] = self.C_load
        self.sweep_output_db["C_par"] = self.C_par
        self.sweep_output_db["C_wire"] = self.C_wire
        self.sweep_output_db["R_wire"] = self.R_wire
        self.sweep_output_db["R_avg_inv"] = self.R_avg_inv

        self.sweep_output_db["delta"] = self.delta
        self.sweep_output_db["dVt"] = self.dVt
        self.sweep_output_db["n0"] = self.n0
        self.sweep_output_db["Lscale"] = self.Lscale
        self.sweep_output_db["slope_at_crossing"] = self.slope_at_crossing
        self.sweep_output_db["V_IL"] = self.V_IL
        self.sweep_output_db["V_IH"] = self.V_IH
        self.sweep_output_db["V_OH"] = self.V_OH
        self.sweep_output_db["V_OL"] = self.V_OL
        self.sweep_output_db["NM_H"] = self.NM_H
        self.sweep_output_db["NM_L"] = self.NM_L
        self.sweep_output_db["noise_margin"] = self.noise_margin
        self.sweep_output_db["I_tunnel"] = self.I_tunnel
        self.sweep_output_db["I_sub"] = self.I_sub

    def config_pareto_metric_db(self):
        self.pareto_metric_db = {"area": "min", "delay": "min", "Edynamic": "min", "Pstatic": "min"}
        #self.pareto_metric_db = {"delay": "min"}
        self.input_metric_db = {"L", "W", "V_dd", "V_th", "tox"}

    def apply_base_parameter_effects(self):
        super().apply_base_parameter_effects()

    def apply_additional_effects(self):
        super().apply_additional_effects()

    def create_constraints(self, dennard_scaling_type="constant_field"):
        eps = 1e-6
        self.constraints = [
            Constraint(sp.Le(self.slope_at_crossing, -1.0, evaluate=False), label="slope_at_crossing"),
            Constraint(sp.Ge(self.noise_margin, eps, evaluate=False), label="noise_margin_positive"),
        ]
        #super().create_constraints(dennard_scaling_type)
        self.sweep_constraints_leq = {} # in sweep, we check for if these are <= 0. If not, then discard the design point.
        self.sweep_constraints_eq = {} # in sweep, we check for if these are == 0. If not, then discard the design point.
        #self.sweep_constraints_leq["W/L"] = self.base_params.W/self.base_params.L - 20
        #self.sweep_constraints_leq["0.1 - (W/L)"] = 0.1 - (self.base_params.W/self.base_params.L)
        #self.sweep_constraints_leq["Lc/L"] = self.base_params.Lc/self.base_params.L - 2
        #self.sweep_constraints_leq["Lext/L"] = self.base_params.Lext/self.base_params.L - 2
        #self.sweep_constraints_leq["0.1 - (Lext/L)"] = 0.1 - (self.base_params.Lext/self.base_params.L)
        #self.sweep_constraints_leq["0.1 - (Lc/L)"] = 0.1 - (self.base_params.Lc/self.base_params.L)
        #self.sweep_constraints_leq["tox/tsemi"] = self.base_params.tox/self.base_params.tsemi - 1/3 # the scale length equation kind of assumes this condition is met.
        #self.sweep_constraints_eq["rho_c_n - rho_c_p"] = self.base_params.rho_c_n - self.base_params.rho_c_p
        #self.sweep_constraints_eq["Rsh_c_n - Rsh_c_p"] = self.base_params.Rsh_c_n - self.base_params.Rsh_c_p
        #self.sweep_constraints_eq["Rsh_ext_n - Rsh_ext_p"] = self.base_params.Rsh_ext_n - self.base_params.Rsh_ext_p
        

