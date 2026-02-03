import os
import yaml
import logging
import src.cacti.cacti_python.get_dat as dat
import src.cacti.cacti_python.get_IO as IO
from src.cacti.cacti_python.parameter import InputParameter
from src import global_symbol_table
from src import sim_util

import math
from src.inverse_pass.constraint import Constraint
from sympy import symbols, ceiling, expand, exp, Abs
import cvxpy as cp

logger = logging.getLogger(__name__)

TECH_NODE_FILE = "src/yaml/tech_nodes.yaml"
WIRE_RC_FILE = "src/yaml/wire_rc.yaml"

LARGE_VALUE = 1e15

# create sympy variables and initial tech values
class BaseParameters:
    def __init__(self, tech_node, dat_file, symbol_type="sympy", tech_param_override=None):
        self.tech_node = tech_node
        self.dat_file = dat_file
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

        self.init_memory_params()

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
        self.init_memory_param_list()

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

    def init_output_parameters(self, output_metrics):
        self.output_parameters_initialized = True
        for metric in output_metrics:
            setattr(self, metric, cp.Parameter(pos=True))

    def init_output_parameters_basic(self, output_metrics):
        self.output_parameters_initialized = True
        for metric in output_metrics:
            setattr(self, metric, symbols(metric, real=True))

    def init_memory_params(self):
        # Memory parameters
        self.C_g_ideal = self.symbol_init("C_g_ideal")
        self.C_fringe = self.symbol_init("C_fringe")
        self.C_junc = self.symbol_init("C_junc")
        self.C_junc_sw = self.symbol_init("C_junc_sw")
        self.l_phy = self.symbol_init("l_phy")    
        self.l_elec = self.symbol_init("l_elec")
        self.nmos_effective_resistance_multiplier = self.symbol_init("nmos_effective_resistance_multiplier")
        self.Vdd = self.symbol_init("Vdd")
        self.Vth = self.symbol_init("Vth")
        self.Vdsat = self.symbol_init("Vdsat")
        self.I_on_n = self.symbol_init("I_on_n")  
        self.I_on_p = self.symbol_init("I_on_p")
        self.I_off_n = self.symbol_init("I_off_n")
        self.I_g_on_n = self.symbol_init("I_g_on_n")
        self.C_ox = self.symbol_init("C_ox")
        self.t_ox = self.symbol_init("t_ox")  
        self.n2p_drv_rt = self.symbol_init("n2p_drv_rt")
        self.lch_lk_rdc = self.symbol_init("lch_lk_rdc")
        self.Mobility_n = self.symbol_init("Mobility_n")
        self.gmp_to_gmn_multiplier = self.symbol_init("gmp_to_gmn_multiplier")
        self.vpp = self.symbol_init("vpp")    
        self.Wmemcella = self.symbol_init("Wmemcella")    
        self.Wmemcellpmos = self.symbol_init("Wmemcellpmos")    
        self.Wmemcellnmos = self.symbol_init("Wmemcellnmos")    
        self.area_cell = self.symbol_init("area_cell")    
        self.asp_ratio_cell = self.symbol_init("asp_ratio_cell")    
        self.vdd_cell = self.symbol_init("vdd_cell")      
        self.dram_cell_I_on = self.symbol_init("dram_cell_I_on")  
        self.dram_cell_Vdd = self.symbol_init("dram_cell_Vdd")  
        self.dram_cell_C = self.symbol_init("dram_cell_C")  
        self.dram_cell_I_off_worst_case_len_temp = self.symbol_init("dram_cell_I_off_worst_case_len_temp")  
        self.logic_scaling_co_eff = self.symbol_init("logic_scaling_co_eff")  
        self.core_tx_density = self.symbol_init("core_tx_density")      
        self.sckt_co_eff = self.symbol_init("sckt_co_eff")  
        self.chip_layout_overhead = self.symbol_init("chip_layout_overhead")  
        self.macro_layout_overhead = self.symbol_init("macro_layout_overhead")  
        self.sense_delay = self.symbol_init("sense_delay")  
        self.sense_dy_power = self.symbol_init("sense_dy_power")  
        self.wire_pitch = self.symbol_init("wire_pitch")    
        self.barrier_thickness = self.symbol_init("barrier_thickness")  
        self.dishing_thickness = self.symbol_init("dishing_thickness")  
        self.alpha_scatter = self.symbol_init("alpha_scatter")  
        self.aspect_ratio = self.symbol_init("aspect_ratio")  
        self.miller_value = self.symbol_init("miller_value")  
        self.horiz_dielectric_constant = self.symbol_init("horiz_dielectric_constant")      
        self.vert_dielectric_constant = self.symbol_init("vert_dielectric_constant")  
        self.ild_thickness = self.symbol_init("ild_thickness")  
        self.fringe_cap = self.symbol_init("fringe_cap")  
        self.resistivity = self.symbol_init("resistivity")  
        self.wire_r_per_micron = self.symbol_init("wire_r_per_micron")  
        self.wire_c_per_micron = self.symbol_init("wire_c_per_micron")      
        self.tsv_pitch = self.symbol_init("tsv_pitch")  
        self.tsv_diameter = self.symbol_init("tsv_diameter")  
        self.tsv_length = self.symbol_init("tsv_length")  
        self.tsv_dielec_thickness = self.symbol_init("tsv_dielec_thickness")  
        self.tsv_contact_resistance = self.symbol_init("tsv_contact_resistance")    
        self.tsv_depletion_width = self.symbol_init("tsv_depletion_width")  
        self.tsv_liner_dielectric_cons = self.symbol_init("tsv_liner_dielectric_cons")  

        # Memory I/O parameters
        self.vdd_io = self.symbol_init("vdd_io")  
        self.v_sw_clk = self.symbol_init("v_sw_clk")  
        self.c_int = self.symbol_init("c_int")  
        self.c_tx = self.symbol_init("c_tx")  
        self.c_data = self.symbol_init("c_data")  
        self.c_addr = self.symbol_init("c_addr")  
        self.i_bias = self.symbol_init("i_bias")  
        self.i_leak = self.symbol_init("i_leak")    
        self.ioarea_c = self.symbol_init("ioarea_c")  
        self.ioarea_k0 = self.symbol_init("ioarea_k0")  
        self.ioarea_k1 = self.symbol_init("ioarea_k1")  
        self.ioarea_k2 = self.symbol_init("ioarea_k2")  
        self.ioarea_k3 = self.symbol_init("ioarea_k3")      
        self.t_ds = self.symbol_init("t_ds")  
        self.t_is = self.symbol_init("t_is")  
        self.t_dh = self.symbol_init("t_dh")  
        self.t_ih = self.symbol_init("t_ih")  
        self.t_dcd_soc = self.symbol_init("t_dcd_soc")  
        self.t_dcd_dram = self.symbol_init("t_dcd_dram")    
        self.t_error_soc = self.symbol_init("t_error_soc")  
        self.t_skew_setup = self.symbol_init("t_skew_setup")  
        self.t_skew_hold = self.symbol_init("t_skew_hold")  
        self.t_dqsq = self.symbol_init("t_dqsq")  
        self.t_soc_setup = self.symbol_init("t_soc_setup")  
        self.t_soc_hold = self.symbol_init("t_soc_hold")    
        self.t_jitter_setup = self.symbol_init("t_jitter_setup")  
        self.t_jitter_hold = self.symbol_init("t_jitter_hold")  
        self.t_jitter_addr_setup = self.symbol_init("t_jitter_addr_setup")  
        self.t_jitter_addr_hold = self.symbol_init("t_jitter_addr_hold")  
        self.t_cor_margin = self.symbol_init("t_cor_margin")  
        self.r_diff_term = self.symbol_init("r_diff_term")      
        self.rtt1_dq_read = self.symbol_init("rtt1_dq_read")  
        self.rtt2_dq_read = self.symbol_init("rtt2_dq_read")  
        self.rtt1_dq_write = self.symbol_init("rtt1_dq_write")  
        self.rtt2_dq_write = self.symbol_init("rtt2_dq_write")  
        self.rtt_ca = self.symbol_init("rtt_ca")        
        self.rs1_dq = self.symbol_init("rs1_dq")  
        self.rs2_dq = self.symbol_init("rs2_dq")  
        self.r_stub_ca = self.symbol_init("r_stub_ca")  
        self.r_on = self.symbol_init("r_on")  
        self.r_on_ca = self.symbol_init("r_on_ca")  
        self.z0 = self.symbol_init("z0")    
        self.t_flight = self.symbol_init("t_flight")  
        self.t_flight_ca = self.symbol_init("t_flight_ca")  
        self.k_noise_write = self.symbol_init("k_noise_write")  
        self.k_noise_read = self.symbol_init("k_noise_read")  
        self.k_noise_addr = self.symbol_init("k_noise_addr")  
        self.v_noise_independent_write = self.symbol_init("v_noise_independent_write")      
        self.v_noise_independent_read = self.symbol_init("v_noise_independent_read")  
        self.v_noise_independent_addr = self.symbol_init("v_noise_independent_addr")  
        self.phy_datapath_s = self.symbol_init("phy_datapath_s")  
        self.phy_phase_rotator_s = self.symbol_init("phy_phase_rotator_s")  
        self.phy_clock_tree_s = self.symbol_init("phy_clock_tree_s")  
        self.phy_rx_s = self.symbol_init("phy_rx_s")        
        self.phy_dcc_s = self.symbol_init("phy_dcc_s")  
        self.phy_deskew_s = self.symbol_init("phy_deskew_s")  
        self.phy_leveling_s = self.symbol_init("phy_leveling_s")  
        self.phy_pll_s = self.symbol_init("phy_pll_s")  
        self.phy_datapath_d = self.symbol_init("phy_datapath_d")    
        self.phy_phase_rotator_d = self.symbol_init("phy_phase_rotator_d")  
        self.phy_clock_tree_d = self.symbol_init("phy_clock_tree_d")  
        self.phy_rx_d = self.symbol_init("phy_rx_d")  
        self.phy_dcc_d = self.symbol_init("phy_dcc_d")  
        self.phy_deskew_d = self.symbol_init("phy_deskew_d")    
        self.phy_leveling_d = self.symbol_init("phy_leveling_d")  
        self.phy_pll_d = self.symbol_init("phy_pll_d")  
        self.phy_pll_wtime = self.symbol_init("phy_pll_wtime")  
        self.phy_phase_rotator_wtime = self.symbol_init("phy_phase_rotator_wtime")  
        self.phy_rx_wtime = self.symbol_init("phy_rx_wtime")  
        self.phy_bandgap_wtime = self.symbol_init("phy_bandgap_wtime")  
        self.phy_deskew_wtime = self.symbol_init("phy_deskew_wtime")  
        self.phy_vrefgen_wtime = self.symbol_init("phy_vrefgen_wtime")  
        
        # not currently used
        self.BufPeriphAreaEff = self.symbol_init("buf_peripheral_area_proportion")  
        self.MemPeriphAreaEff = self.symbol_init("mem_peripheral_area_propportion")  

        # TODO: look into moving these to circuit model
        self.MemReadL = {}
        self.MemWriteL = {}
        self.MemReadEact = {}
        self.MemWriteEact = {}
        self.MemPpass = {}
        self.BufL = {}
        self.BufReadEact = {}
        self.BufWriteEact = {}
        self.BufPpass = {}
        self.OffChipIOL = {}
        self.OffChipIOPact = {} 

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
            "logic_sensitivity": self.logic_sensitivity,
            "logic_resource_sensitivity": self.logic_resource_sensitivity,
            "logic_amdahl_limit": self.logic_amdahl_limit,
            "logic_resource_amdahl_limit": self.logic_resource_amdahl_limit,
            "interconnect_sensitivity": self.interconnect_sensitivity,
            "interconnect_resource_sensitivity": self.interconnect_resource_sensitivity,
            "interconnect_amdahl_limit": self.interconnect_amdahl_limit,
            "interconnect_resource_amdahl_limit": self.interconnect_resource_amdahl_limit,
            "memory_sensitivity": self.memory_sensitivity,
            "memory_resource_sensitivity": self.memory_resource_sensitivity,
            "memory_amdahl_limit": self.memory_amdahl_limit,
            "memory_resource_amdahl_limit": self.memory_resource_amdahl_limit,
            "MemReadL": self.MemReadL,
            "MemWriteL": self.MemWriteL,
            "MemReadEact": self.MemReadEact,
            "MemWriteEact": self.MemWriteEact,
            "MemPpass": self.MemPpass,
            "BufL": self.BufL,
            "BufReadEact": self.BufReadEact,
            "BufWriteEact": self.BufWriteEact,
            "BufPpass": self.BufPpass,
            "OffChipIOL": self.OffChipIOL,
            "OffChipIOPact": self.OffChipIOPact,

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
            
            # Cacti .dat technology parameters
            'C_g_ideal': self.C_g_ideal,
            'C_fringe': self.C_fringe,
            'C_junc': self.C_junc,
            'C_junc_sw': self.C_junc_sw,
            'l_phy': self.l_phy,
            'l_elec': self.l_elec,
            'nmos_effective_resistance_multiplier': self.nmos_effective_resistance_multiplier,
            'Vdd': self.Vdd,
            'Vth': self.Vth,
            'Vdsat': self.Vdsat,
            'I_on_n': self.I_on_n,
            'I_on_p': self.I_on_p,
            'I_off_n': self.I_off_n,
            'I_g_on_n': self.I_g_on_n,
            'C_ox': self.C_ox,
            't_ox': self.t_ox,
            'n2p_drv_rt': self.n2p_drv_rt,
            'lch_lk_rdc': self.lch_lk_rdc,
            'Mobility_n': self.Mobility_n,
            'gmp_to_gmn_multiplier': self.gmp_to_gmn_multiplier,
            'vpp': self.vpp,
            'Wmemcella': self.Wmemcella,
            'Wmemcellpmos': self.Wmemcellpmos,
            'Wmemcellnmos': self.Wmemcellnmos,
            'area_cell': self.area_cell,
            'asp_ratio_cell': self.asp_ratio_cell,
            'vdd_cell': self.vdd_cell,
            'dram_cell_I_on': self.dram_cell_I_on,
            'dram_cell_Vdd': self.dram_cell_Vdd,
            'dram_cell_C': self.dram_cell_C,
            'dram_cell_I_off_worst_case_len_temp': self.dram_cell_I_off_worst_case_len_temp,
            'logic_scaling_co_eff': self.logic_scaling_co_eff,
            'core_tx_density': self.core_tx_density,
            'sckt_co_eff': self.sckt_co_eff,
            'chip_layout_overhead': self.chip_layout_overhead,
            'macro_layout_overhead': self.macro_layout_overhead,
            'sense_delay': self.sense_delay,
            'sense_dy_power': self.sense_dy_power,
            'wire_pitch': self.wire_pitch,
            'barrier_thickness': self.barrier_thickness,
            'dishing_thickness': self.dishing_thickness,
            'alpha_scatter': self.alpha_scatter,
            'aspect_ratio': self.aspect_ratio,
            'miller_value': self.miller_value,
            'horiz_dielectric_constant': self.horiz_dielectric_constant,
            'vert_dielectric_constant': self.vert_dielectric_constant,
            'ild_thickness': self.ild_thickness,
            'fringe_cap': self.fringe_cap,
            'resistivity': self.resistivity,
            'wire_r_per_micron': self.wire_r_per_micron,
            'wire_c_per_micron': self.wire_c_per_micron,
            'tsv_pitch': self.tsv_pitch,
            'tsv_diameter': self.tsv_diameter,
            'tsv_length': self.tsv_length,
            'tsv_dielec_thickness': self.tsv_dielec_thickness,
            'tsv_contact_resistance': self.tsv_contact_resistance,
            'tsv_depletion_width': self.tsv_depletion_width,
            'tsv_liner_dielectric_cons': self.tsv_liner_dielectric_cons,

            # Cacti IO technology parameters
            'vdd_io': self.vdd_io,
            'v_sw_clk': self.v_sw_clk,
            'c_int': self.c_int,
            'c_tx': self.c_tx,
            'c_data': self.c_data,
            'c_addr': self.c_addr,
            'i_bias': self.i_bias,
            'i_leak': self.i_leak,
            'ioarea_c': self.ioarea_c,
            'ioarea_k0': self.ioarea_k0,
            'ioarea_k1': self.ioarea_k1,
            'ioarea_k2': self.ioarea_k2,
            'ioarea_k3': self.ioarea_k3,
            't_ds': self.t_ds,
            't_is': self.t_is,
            't_dh': self.t_dh,
            't_ih': self.t_ih,
            't_dcd_soc': self.t_dcd_soc,
            't_dcd_dram': self.t_dcd_dram,
            't_error_soc': self.t_error_soc,
            't_skew_setup': self.t_skew_setup,
            't_skew_hold': self.t_skew_hold,
            't_dqsq': self.t_dqsq,
            't_soc_setup': self.t_soc_setup,
            't_soc_hold': self.t_soc_hold,
            't_jitter_setup': self.t_jitter_setup,
            't_jitter_hold': self.t_jitter_hold,
            't_jitter_addr_setup': self.t_jitter_addr_setup,
            't_jitter_addr_hold': self.t_jitter_addr_hold,
            't_cor_margin': self.t_cor_margin,
            'r_diff_term': self.r_diff_term,
            'rtt1_dq_read': self.rtt1_dq_read,
            'rtt2_dq_read': self.rtt2_dq_read,
            'rtt1_dq_write': self.rtt1_dq_write,
            'rtt2_dq_write': self.rtt2_dq_write,
            'rtt_ca': self.rtt_ca,
            'rs1_dq': self.rs1_dq,
            'rs2_dq': self.rs2_dq,
            'r_stub_ca': self.r_stub_ca,
            'r_on': self.r_on,
            'r_on_ca': self.r_on_ca,
            'z0': self.z0,
            't_flight': self.t_flight,
            't_flight_ca': self.t_flight_ca,
            'k_noise_write': self.k_noise_write,
            'k_noise_read': self.k_noise_read,
            'k_noise_addr': self.k_noise_addr,
            'v_noise_independent_write': self.v_noise_independent_write,
            'v_noise_independent_read': self.v_noise_independent_read,
            'v_noise_independent_addr': self.v_noise_independent_addr,
            'phy_datapath_s': self.phy_datapath_s,
            'phy_phase_rotator_s': self.phy_phase_rotator_s,
            'phy_clock_tree_s': self.phy_clock_tree_s,
            'phy_rx_s': self.phy_rx_s,
            'phy_dcc_s': self.phy_dcc_s,
            'phy_deskew_s': self.phy_deskew_s,
            'phy_leveling_s': self.phy_leveling_s,
            'phy_pll_s': self.phy_pll_s,
            'phy_datapath_d': self.phy_datapath_d,
            'phy_phase_rotator_d': self.phy_phase_rotator_d,
            'phy_clock_tree_d': self.phy_clock_tree_d,
            'phy_rx_d': self.phy_rx_d,
            'phy_dcc_d': self.phy_dcc_d,
            'phy_deskew_d': self.phy_deskew_d,
            'phy_leveling_d': self.phy_leveling_d,
            'phy_pll_d': self.phy_pll_d,
            'phy_pll_wtime': self.phy_pll_wtime,
            'phy_phase_rotator_wtime': self.phy_phase_rotator_wtime,
            'phy_rx_wtime': self.phy_rx_wtime,
            'phy_bandgap_wtime': self.phy_bandgap_wtime,
            'phy_deskew_wtime': self.phy_deskew_wtime,
            'phy_vrefgen_wtime': self.phy_vrefgen_wtime,

            # dennard scaling factors
            "alpha_dennard": self.alpha_dennard,
            "epsilon_dennard": self.epsilon_dennard,
        }
        self.names = {}
        for name in self.symbol_table.keys():
            if not isinstance(self.symbol_table[name], dict):
                self.names[self.symbol_table[name]] = name
        global_symbol_table.global_symbol_table = self.symbol_table
    
    def init_weights(self, cnt):
        self.w = cp.Variable(cnt, pos=True)
        self.symbol_table["w"] = self.w
    
    def init_memory_param_list(self):
        self.cacti_tech_params = [
            'C_g_ideal',
            'C_fringe',
            'C_junc',
            'C_junc_sw',
            'l_phy',
            'l_elec',
            'nmos_effective_resistance_multiplier',
            'Vdd',
            'Vth',
            'Vdsat',
            'I_on_n',
            'I_on_p',
            'I_off_n',
            'I_g_on_n',
            'C_ox',
            't_ox',
            'n2p_drv_rt',
            'lch_lk_rdc',
            'Mobility_n',
            'gmp_to_gmn_multiplier',
            'vpp',
            'Wmemcella',
            'Wmemcellpmos',
            'Wmemcellnmos',
            'area_cell',
            'asp_ratio_cell',
            'vdd_cell',
            'dram_cell_I_on',
            'dram_cell_Vdd',
            'dram_cell_C',
            'dram_cell_I_off_worst_case_len_temp',
            'logic_scaling_co_eff',
            'core_tx_density',
            'sckt_co_eff',
            'chip_layout_overhead',
            'macro_layout_overhead',
            'sense_delay',
            'sense_dy_power',
            'wire_pitch',
            'barrier_thickness',
            'dishing_thickness',
            'alpha_scatter',
            'aspect_ratio',
            'miller_value',
            'horiz_dielectric_constant',
            'vert_dielectric_constant',
            'ild_thickness',
            'fringe_cap',
            'resistivity',
            'wire_r_per_micron',
            'wire_c_per_micron',
            'tsv_pitch',
            'tsv_diameter',
            'tsv_length',
            'tsv_dielec_thickness',
            'tsv_contact_resistance',
            'tsv_depletion_width',
            'tsv_liner_dielectric_cons'
        ]

        self.cacti_io_tech_params = [
            'vdd_io',
            'v_sw_clk',
            'c_int',
            'c_tx',
            'c_data',
            'c_addr',
            'i_bias',
            'i_leak',
            'ioarea_c',
            'ioarea_k0',
            'ioarea_k1',
            'ioarea_k2',
            'ioarea_k3',
            't_ds',
            't_is',
            't_dh',
            't_ih',
            't_dcd_soc',
            't_dcd_dram',
            't_error_soc',
            't_skew_setup',
            't_skew_hold',
            't_dqsq',
            't_soc_setup',
            't_soc_hold',
            't_jitter_setup',
            't_jitter_hold',
            't_jitter_addr_setup',
            't_jitter_addr_hold',
            't_cor_margin',
            'r_diff_term',
            'rtt1_dq_read',
            'rtt2_dq_read',
            'rtt1_dq_write',
            'rtt2_dq_write',
            'rtt_ca',
            'rs1_dq',
            'rs2_dq',
            'r_stub_ca',
            'r_on',
            'r_on_ca',
            'z0',
            't_flight',
            't_flight_ca',
            'k_noise_write',
            'k_noise_read',
            'k_noise_addr',
            'v_noise_independent_write',
            'v_noise_independent_read',
            'v_noise_independent_addr',
            'phy_datapath_s',
            'phy_phase_rotator_s',
            'phy_clock_tree_s',
            'phy_rx_s',
            'phy_dcc_s',
            'phy_deskew_s',
            'phy_leveling_s',
            'phy_pll_s',
            'phy_datapath_d',
            'phy_phase_rotator_d',
            'phy_clock_tree_d',
            'phy_rx_d',
            'phy_dcc_d',
            'phy_deskew_d',
            'phy_leveling_d',
            'phy_pll_d',
            'phy_pll_wtime',
            'phy_phase_rotator_wtime',
            'phy_rx_wtime',
            'phy_bandgap_wtime',
            'phy_deskew_wtime',
            'phy_vrefgen_wtime'
        ]
