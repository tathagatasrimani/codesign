import os
import yaml
import logging
from src import global_symbol_table
from src import sim_util

import math
from src.inverse_pass.constraint import Constraint
from sympy import symbols, ceiling, expand, exp, Abs
import cvxpy as cp

logger = logging.getLogger(__name__)

TECH_NODE_FILE = "src/yaml/tech_nodes.yaml"

LARGE_VALUE = 1e15

# create sympy variables and initial tech values
class BaseParameters:
    def __init__(self, tech_node, symbol_type="sympy", tech_param_override=None):
        self.tech_node = tech_node
        self.constraints = []
        self.symbol_type = symbol_type
        self.output_parameters_initialized = False
        self.node_arrivals_end = self.symbol_init("node_arrivals_end")

        # placeholder symbols if using cvxpy pipeline
        self.delay = self.symbol_init("delay")
        self.E_act_inv = self.symbol_init("E_act_inv")
        self.P_pass_inv = self.symbol_init("P_pass_inv")
        self.C_gate = self.symbol_init("C_gate")

        # leaving both clk_period and f in here for flexibility, but only one of them will be used.
        self.clk_period = self.symbol_init("clk_period")
        self.f = self.symbol_init("f")

        # Logic parameters
        self.V_dd = self.symbol_init("V_dd")
        self.V_th = self.symbol_init("V_th")
        self.tox = self.symbol_init("tox")
        self.eot = self.symbol_init("eot")
        self.W = self.symbol_init("W")
        self.L = self.symbol_init("L")
        self.k_gate = self.symbol_init("k_gate")
        self.t_1 = self.symbol_init("t_1") # physical body thickness, used for scale length in vs model

        self.L_ov = self.symbol_init("L_ov")
        self.Lscale = self.symbol_init("Lscale")

        # dennard scaling factors, used for dennard scaling test
        self.alpha_dennard = self.symbol_init("alpha_dennard")
        self.epsilon_dennard = self.symbol_init("epsilon_dennard")
        
        self.area = self.symbol_init("area")

        # wire parameters
        self.m1_rho = self.symbol_init("m1_rho")
        self.m2_rho = self.symbol_init("m2_rho")
        self.m3_rho = self.symbol_init("m3_rho")
        self.m4_rho = self.symbol_init("m4_rho")
        self.m5_rho = self.symbol_init("m5_rho")
        self.m6_rho = self.symbol_init("m6_rho")
        self.m7_rho = self.symbol_init("m7_rho")
        self.m8_rho = self.symbol_init("m8_rho")
        self.m9_rho = self.symbol_init("m9_rho")
        self.m10_rho = self.symbol_init("m10_rho")

        self.m1_k = self.symbol_init("m1_k")
        self.m2_k = self.symbol_init("m2_k")
        self.m3_k = self.symbol_init("m3_k")
        self.m4_k = self.symbol_init("m4_k")
        self.m5_k = self.symbol_init("m5_k")
        self.m6_k = self.symbol_init("m6_k")
        self.m7_k = self.symbol_init("m7_k")
        self.m8_k = self.symbol_init("m8_k")
        self.m9_k = self.symbol_init("m9_k")
        self.m10_k = self.symbol_init("m10_k")

        # Electron mobility for NMOS
        self.u_n = self.symbol_init("u_n")

        # semiconductor capacitance for virtual source model
        self.Cs = self.symbol_init("Cs")

        # CNT parameters
        self.d = self.symbol_init("d")
        self.k_cnt = self.symbol_init("k_cnt")
        self.L_c = self.symbol_init("L_c")
        self.H_c = self.symbol_init("H_c")
        self.H_g = self.symbol_init("H_g")

        # MVS general model parameters
        self.beta_p_n = self.symbol_init("beta_p_n")
        self.mD_fac = self.symbol_init("mD_fac")
        self.mu_eff_n = self.symbol_init("mu_eff_n")
        self.mu_eff_p = self.symbol_init("mu_eff_p")
        self.eps_semi = self.symbol_init("eps_semi")
        self.tsemi = self.symbol_init("tsemi")
        self.Lext = self.symbol_init("Lext")
        self.Lc = self.symbol_init("Lc")
        self.eps_cap = self.symbol_init("eps_cap")
        self.rho_c_n = self.symbol_init("rho_c_n")
        self.rho_c_p = self.symbol_init("rho_c_p")
        self.Rsh_c_n = self.symbol_init("Rsh_c_n")
        self.Rsh_c_p = self.symbol_init("Rsh_c_p")
        self.Rsh_ext_n = self.symbol_init("Rsh_ext_n")
        self.Rsh_ext_p = self.symbol_init("Rsh_ext_p")
        self.FO = self.symbol_init("FO")
        self.M = self.symbol_init("M")
        self.a = self.symbol_init("a")
        self.GEO = self.symbol_init("GEO")
        self.MUL = self.symbol_init("MUL")

        self.logic_sensitivity = self.symbol_init("logic_sensitivity")
        self.logic_resource_sensitivity = self.symbol_init("logic_resource_sensitivity")
        self.logic_amdahl_limit = self.symbol_init("logic_amdahl_limit")
        self.logic_resource_amdahl_limit = self.symbol_init("logic_resource_amdahl_limit")

        self.interconnect_sensitivity = self.symbol_init("interconnect_sensitivity")
        self.interconnect_resource_sensitivity = self.symbol_init("interconnect_resource_sensitivity")
        self.interconnect_amdahl_limit = self.symbol_init("interconnect_amdahl_limit")
        self.interconnect_resource_amdahl_limit = self.symbol_init("interconnect_resource_amdahl_limit")

        self.memory_sensitivity = self.symbol_init("memory_sensitivity")
        self.memory_resource_sensitivity = self.symbol_init("memory_resource_sensitivity")
        self.memory_amdahl_limit = self.symbol_init("memory_amdahl_limit")
        self.memory_resource_amdahl_limit = self.symbol_init("memory_resource_amdahl_limit")

        # technology level parameter values
        self.tech_values = {}
        self.init_symbol_table()

        # set initial values for technology parameters based on tech node
        config = yaml.load(open(TECH_NODE_FILE), Loader=yaml.Loader)
        config_tech_node = config[self.tech_node] if tech_param_override is None else tech_param_override
        print(f"initial tech params: {config_tech_node}")
        for key in config["default"]:
            try:
                self.set_symbol_value(self.symbol_table[key], config_tech_node[key])
            except:
                logger.info(f"using default value for {key}")
                self.set_symbol_value(self.symbol_table[key], config["default"][key])
        # set initial values for dennard scaling factors (no actual meaning, they will be set by the optimizer)
        self.set_symbol_value(self.alpha_dennard, 1)
        self.set_symbol_value(self.epsilon_dennard, 1)

        # mock area and latency scaling for experimental purposes
        self.area_scale = sim_util.xreplace_safe(self.W * self.L, self.tech_values) / (self.W * self.L)
        self.latency_scale = 1/self.area_scale

    def set_symbol_value(self, symbol, value, dont_save_to_tech_values=False):
        if not dont_save_to_tech_values:
            self.tech_values[symbol] = value
        if self.symbol_type == "cvxpy":
            if type(value) == float and value == math.inf:
                symbol.value = LARGE_VALUE
            else:
                symbol.value = value

    def symbol_init(self, name):
        if self.symbol_type == "sympy":
            return symbols(name, real=True)
        elif self.symbol_type == "cvxpy":
            return cp.Variable(pos=True)
        else:
            raise ValueError(f"unsupported symbol type for symbol_init: {self.symbol_type}")

    def init_output_parameters_basic(self, output_metrics):
        self.output_parameters_initialized = True
        for metric in output_metrics:
            setattr(self, metric, symbols(metric, real=True))

    def create_constraints(self):
        # ensure that relevant parameters are positive
        self.constraints.append(Constraint(self.V_dd >= 0, "V_dd >= 0"))
        self.constraints.append(Constraint(self.V_th >= 0, "V_th >= 0"))
        self.constraints.append(Constraint(self.clk_period >= 0, "clk_period >= 0"))
        self.constraints.append(Constraint(self.f >= 0, "f >= 0"))
        self.constraints.append(Constraint(self.u_n >= 0, "u_n >= 0"))
        self.constraints.append(Constraint(self.tox >= 0, "tox >= 0"))
        self.constraints.append(Constraint(self.t_1 >= 0, "t_1 >= 0"))
        self.constraints.append(Constraint(self.L_ov >= 0, "L_ov >= 0"))
        self.constraints.append(Constraint(self.W >= 0, "W >= 0"))
        self.constraints.append(Constraint(self.L >= 0, "L >= 0"))
        self.constraints.append(Constraint(self.k_gate >= 0, "k_gate >= 0"))
        self.constraints.append(Constraint(self.Cs >= 0, "Cs >= 0"))
        self.constraints.append(Constraint(self.d >= 0, "d >= 0"))
        self.constraints.append(Constraint(self.k_cnt >= 0, "k_cnt >= 0"))
        self.constraints.append(Constraint(self.area >= 0, "area >= 0"))
        self.constraints.append(Constraint(self.L_c >= 0, "L_c >= 0"))
        self.constraints.append(Constraint(self.H_c >= 0, "H_c >= 0"))
        self.constraints.append(Constraint(self.H_g >= 0, "H_g >= 0"))
        self.constraints.append(Constraint(self.beta_p_n >= 0, "beta_p_n >= 0"))
        self.constraints.append(Constraint(self.mD_fac >= 0, "mD_fac >= 0"))
        self.constraints.append(Constraint(self.mu_eff_n >= 0, "mu_eff_n >= 0"))
        self.constraints.append(Constraint(self.mu_eff_p >= 0, "mu_eff_p >= 0"))
        self.constraints.append(Constraint(self.eps_semi >= 0, "eps_semi >= 0"))
        self.constraints.append(Constraint(self.tsemi >= 0, "tsemi >= 0"))
        self.constraints.append(Constraint(self.Lext >= 0, "Lext >= 0"))
        self.constraints.append(Constraint(self.Lc >= 0, "Lc >= 0"))
        self.constraints.append(Constraint(self.eps_cap >= 0, "eps_cap >= 0"))
        self.constraints.append(Constraint(self.rho_c_n >= 0, "rho_c_n >= 0"))
        self.constraints.append(Constraint(self.rho_c_p >= 0, "rho_c_p >= 0"))
        self.constraints.append(Constraint(self.Rsh_c_n >= 0, "Rsh_c_n >= 0"))
        self.constraints.append(Constraint(self.Rsh_c_p >= 0, "Rsh_c_p >= 0"))
        self.constraints.append(Constraint(self.Rsh_ext_n >= 0, "Rsh_ext_n >= 0"))
        self.constraints.append(Constraint(self.Rsh_ext_p >= 0, "Rsh_ext_p >= 0"))
        self.constraints.append(Constraint(self.FO >= 0, "FO >= 0"))
        self.constraints.append(Constraint(self.M >= 0, "M >= 0"))
        self.constraints.append(Constraint(self.f >= 0, "f >= 0"))
        self.constraints.append(Constraint(self.a >= 0, "a >= 0"))
        self.constraints.append(Constraint(self.GEO >= 0, "GEO >= 0"))
        self.constraints.append(Constraint(self.MUL >= 0, "MUL >= 0"))
        self.constraints.append(Constraint(self.eot >= 0, "eot >= 0"))

    def init_symbol_table(self):
        # initialize string to symbol mapping
        self.symbol_table = {
            "node_arrivals_end": self.node_arrivals_end,
            "delay": self.delay,
            "E_act_inv": self.E_act_inv,
            "P_pass_inv": self.P_pass_inv,
            "C_gate": self.C_gate,
            "V_dd": self.V_dd,
            "V_th": self.V_th,
            "eot": self.eot,
            "clk_period": self.clk_period,
            "f": self.f,
            "u_n": self.u_n,
            "tox": self.tox,
            "t_1": self.t_1,
            "L_ov": self.L_ov,
            "Lscale": self.Lscale,
            "W": self.W,
            "L": self.L,
            "k_gate": self.k_gate,
            "Cs": self.Cs,
            "d": self.d,
            "k_cnt": self.k_cnt,
            "area": self.area,
            "L_c": self.L_c,
            "H_c": self.H_c,
            "H_g": self.H_g,
            "beta_p_n": self.beta_p_n,
            "mD_fac": self.mD_fac,
            "mu_eff_n": self.mu_eff_n,
            "mu_eff_p": self.mu_eff_p,
            "eps_semi": self.eps_semi,
            "tsemi": self.tsemi,
            "Lext": self.Lext,
            "Lc": self.Lc,
            "eps_cap": self.eps_cap,
            "rho_c_n": self.rho_c_n,
            "rho_c_p": self.rho_c_p,
            "Rsh_c_n": self.Rsh_c_n,
            "Rsh_c_p": self.Rsh_c_p,
            "Rsh_ext_n": self.Rsh_ext_n,
            "Rsh_ext_p": self.Rsh_ext_p,
            "FO": self.FO,
            "M": self.M,
            "a": self.a,
            "GEO": self.GEO,
            "MUL": self.MUL,

            # wire parasitics
            "m1_rho": self.m1_rho,
            "m2_rho": self.m2_rho,
            "m3_rho": self.m3_rho,
            "m4_rho": self.m4_rho,
            "m5_rho": self.m5_rho,
            "m6_rho": self.m6_rho,
            "m7_rho": self.m7_rho,
            "m8_rho": self.m8_rho,
            "m9_rho": self.m9_rho,
            "m10_rho": self.m10_rho,
            "m1_k": self.m1_k,
            "m2_k": self.m2_k,
            "m3_k": self.m3_k,
            "m4_k": self.m4_k,
            "m5_k": self.m5_k,
            "m6_k": self.m6_k,
            "m7_k": self.m7_k,
            "m8_k": self.m8_k,
            "m9_k": self.m9_k,
            "m10_k": self.m10_k,
        }
        self.names = {}
        for name in self.symbol_table.keys():
            if not isinstance(self.symbol_table[name], dict):
                self.names[self.symbol_table[name]] = name
        global_symbol_table.global_symbol_table = self.symbol_table
