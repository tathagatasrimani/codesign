import os
import yaml
import logging
import src.cacti.cacti_python.get_dat as dat
import src.cacti.cacti_python.get_IO as IO
from src.cacti.cacti_python.parameter import InputParameter
from src import global_symbol_table

import math
from sympy import symbols, ceiling, expand, exp, Abs

logger = logging.getLogger(__name__)

TECH_NODE_FILE = "src/yaml/tech_nodes.yaml"
WIRE_RC_FILE = "src/yaml/wire_rc.yaml"

# create sympy variables and initial tech values
class BaseParameters:
    def __init__(self, tech_node, dat_file):
        self.tech_node = tech_node
        self.dat_file = dat_file
        self.constraints = []

        self.node_arrivals_end = symbols("node_arrivals_end")

        # leaving both clk_period and f in here for flexibility, but only one of them will be used.
        self.clk_period = symbols("clk_period", real=True)
        self.f = symbols("f", real=True)

        # Logic parameters
        self.V_dd = symbols("V_dd", real=True)
        self.V_th = symbols("V_th", real=True)
        self.tox = symbols("tox", real=True)
        self.W = symbols("W", real=True)
        self.L = symbols("L", real=True)
        self.k_gate = symbols("k_gate", real=True)
        self.t_1 = symbols("t_1", real=True) # physical body thickness, used for scale length in vs model

        self.L_ov = symbols("L_ov", real=True)

        # dennard scaling factors, used for dennard scaling test
        self.alpha_dennard = symbols("alpha_dennard", real=True)
        self.epsilon_dennard = symbols("epsilon_dennard", real=True)
        
        self.area = symbols("area", real=True)

        # wire parameters
        self.m1_rho = symbols("m1_rho", real=True)
        self.m2_rho = symbols("m2_rho", real=True)
        self.m3_rho = symbols("m3_rho", real=True)
        self.m4_rho = symbols("m4_rho", real=True)
        self.m5_rho = symbols("m5_rho", real=True)
        self.m6_rho = symbols("m6_rho", real=True)
        self.m7_rho = symbols("m7_rho", real=True)
        self.m8_rho = symbols("m8_rho", real=True)
        self.m9_rho = symbols("m9_rho", real=True)
        self.m10_rho = symbols("m10_rho", real=True)

        self.m1_k = symbols("m1_k", real=True)
        self.m2_k = symbols("m2_k", real=True)
        self.m3_k = symbols("m3_k", real=True)
        self.m4_k = symbols("m4_k", real=True)
        self.m5_k = symbols("m5_k", real=True)
        self.m6_k = symbols("m6_k", real=True)
        self.m7_k = symbols("m7_k", real=True)
        self.m8_k = symbols("m8_k", real=True)
        self.m9_k = symbols("m9_k", real=True)
        self.m10_k = symbols("m10_k", real=True)

        # Electron mobility for NMOS
        self.u_n = symbols("u_n", real=True)

        self.init_memory_params()

        # semiconductor capacitance for virtual source model
        self.Cs = symbols("Cs", real=True)

        # CNT parameters
        self.d = symbols("d", real=True)
        self.k_cnt = symbols("k_cnt", real=True)
        self.L_c = symbols("L_c", real=True)
        self.H_c = symbols("H_c", real=True)
        self.H_g = symbols("H_g", real=True)

        # MVS general model parameters
        self.beta_p_n = symbols("beta_p_n", real=True)
        self.mD_fac = symbols("mD_fac", real=True)
        self.mu_eff_n = symbols("mu_eff_n", real=True)
        self.mu_eff_p = symbols("mu_eff_p", real=True)
        self.eps_semi = symbols("eps_semi", real=True)
        self.tsemi = symbols("tsemi", real=True)
        self.Lext = symbols("Lext", real=True)
        self.Lc = symbols("Lc", real=True)
        self.eps_cap = symbols("eps_cap", real=True)
        self.rho_c_n = symbols("rho_c_n", real=True)
        self.rho_c_p = symbols("rho_c_p", real=True)
        self.Rsh_c_n = symbols("Rsh_c_n", real=True)
        self.Rsh_c_p = symbols("Rsh_c_p", real=True)
        self.Rsh_ext_n = symbols("Rsh_ext_n", real=True)
        self.Rsh_ext_p = symbols("Rsh_ext_p", real=True)
        self.FO = symbols("FO", real=True)
        self.M = symbols("M", real=True)
        self.a = symbols("a", real=True)

        self.logic_sensitivity = symbols("logic_sensitivity", real=True)
        self.logic_resource_sensitivity = symbols("logic_resource_sensitivity", real=True)
        self.logic_ahmdal_limit = symbols("logic_ahmdal_limit", real=True)
        self.logic_resource_ahmdal_limit = symbols("logic_resource_ahmdal_limit", real=True)

        self.interconnect_sensitivity = symbols("interconnect_sensitivity", real=True)
        self.interconnect_resource_sensitivity = symbols("interconnect_resource_sensitivity", real=True)
        self.interconnect_ahmdal_limit = symbols("interconnect_ahmdal_limit", real=True)
        self.interconnect_resource_ahmdal_limit = symbols("interconnect_resource_ahmdal_limit", real=True)

        self.memory_sensitivity = symbols("memory_sensitivity", real=True)
        self.memory_resource_sensitivity = symbols("memory_resource_sensitivity", real=True)
        self.memory_ahmdal_limit = symbols("memory_ahmdal_limit", real=True)
        self.memory_resource_ahmdal_limit = symbols("memory_resource_ahmdal_limit", real=True)

        # technology level parameter values
        self.tech_values = {}
        self.init_symbol_table()
        self.init_memory_param_list()

        # set initial values for technology parameters based on tech node
        config = yaml.load(open(TECH_NODE_FILE), Loader=yaml.Loader)
        print(config[self.tech_node])
        for key in config["default"]:
            try:
                self.tech_values[self.symbol_table[key]] = config[self.tech_node][key]
            except:
                logger.info(f"using default value for {key}")
                self.tech_values[self.symbol_table[key]] = config["default"][key]
        # set initial values for dennard scaling factors (no actual meaning, they will be set by the optimizer)
        self.tech_values[self.alpha_dennard] = 1
        self.tech_values[self.epsilon_dennard] = 1

        # mock area and latency scaling for experimental purposes
        self.area_scale = (self.W * self.L).xreplace(self.tech_values) / (self.W * self.L)
        self.latency_scale = 1/self.area_scale

    def init_memory_params(self):
        # Memory parameters
        self.C_g_ideal = symbols("C_g_ideal")
        self.C_fringe = symbols("C_fringe")
        self.C_junc = symbols("C_junc")
        self.C_junc_sw = symbols("C_junc_sw")
        self.l_phy = symbols("l_phy")    
        self.l_elec = symbols("l_elec")
        self.nmos_effective_resistance_multiplier = symbols("nmos_effective_resistance_multiplier")
        self.Vdd = symbols("Vdd")
        self.Vth = symbols("Vth")
        self.Vdsat = symbols("Vdsat")
        self.I_on_n = symbols("I_on_n")  
        self.I_on_p = symbols("I_on_p")
        self.I_off_n = symbols("I_off_n")
        self.I_g_on_n = symbols("I_g_on_n")
        self.C_ox = symbols("C_ox")
        self.t_ox = symbols("t_ox")  
        self.n2p_drv_rt = symbols("n2p_drv_rt")
        self.lch_lk_rdc = symbols("lch_lk_rdc")
        self.Mobility_n = symbols("Mobility_n")
        self.gmp_to_gmn_multiplier = symbols("gmp_to_gmn_multiplier")
        self.vpp = symbols("vpp")    
        self.Wmemcella = symbols("Wmemcella")    
        self.Wmemcellpmos = symbols("Wmemcellpmos")    
        self.Wmemcellnmos = symbols("Wmemcellnmos")    
        self.area_cell = symbols("area_cell")    
        self.asp_ratio_cell = symbols("asp_ratio_cell")    
        self.vdd_cell = symbols("vdd_cell")      
        self.dram_cell_I_on = symbols("dram_cell_I_on")  
        self.dram_cell_Vdd = symbols("dram_cell_Vdd")  
        self.dram_cell_C = symbols("dram_cell_C")  
        self.dram_cell_I_off_worst_case_len_temp = symbols("dram_cell_I_off_worst_case_len_temp")  
        self.logic_scaling_co_eff = symbols("logic_scaling_co_eff")  
        self.core_tx_density = symbols("core_tx_density")      
        self.sckt_co_eff = symbols("sckt_co_eff")  
        self.chip_layout_overhead = symbols("chip_layout_overhead")  
        self.macro_layout_overhead = symbols("macro_layout_overhead")  
        self.sense_delay = symbols("sense_delay")  
        self.sense_dy_power = symbols("sense_dy_power")  
        self.wire_pitch = symbols("wire_pitch")    
        self.barrier_thickness = symbols("barrier_thickness")  
        self.dishing_thickness = symbols("dishing_thickness")  
        self.alpha_scatter = symbols("alpha_scatter")  
        self.aspect_ratio = symbols("aspect_ratio")  
        self.miller_value = symbols("miller_value")  
        self.horiz_dielectric_constant = symbols("horiz_dielectric_constant")      
        self.vert_dielectric_constant = symbols("vert_dielectric_constant")  
        self.ild_thickness = symbols("ild_thickness")  
        self.fringe_cap = symbols("fringe_cap")  
        self.resistivity = symbols("resistivity")  
        self.wire_r_per_micron = symbols("wire_r_per_micron")  
        self.wire_c_per_micron = symbols("wire_c_per_micron")      
        self.tsv_pitch = symbols("tsv_pitch")  
        self.tsv_diameter = symbols("tsv_diameter")  
        self.tsv_length = symbols("tsv_length")  
        self.tsv_dielec_thickness = symbols("tsv_dielec_thickness")  
        self.tsv_contact_resistance = symbols("tsv_contact_resistance")    
        self.tsv_depletion_width = symbols("tsv_depletion_width")  
        self.tsv_liner_dielectric_cons = symbols("tsv_liner_dielectric_cons")  

        # Memory I/O parameters
        self.vdd_io = symbols("vdd_io")  
        self.v_sw_clk = symbols("v_sw_clk")  
        self.c_int = symbols("c_int")  
        self.c_tx = symbols("c_tx")  
        self.c_data = symbols("c_data")  
        self.c_addr = symbols("c_addr")  
        self.i_bias = symbols("i_bias")  
        self.i_leak = symbols("i_leak")    
        self.ioarea_c = symbols("ioarea_c")  
        self.ioarea_k0 = symbols("ioarea_k0")  
        self.ioarea_k1 = symbols("ioarea_k1")  
        self.ioarea_k2 = symbols("ioarea_k2")  
        self.ioarea_k3 = symbols("ioarea_k3")      
        self.t_ds = symbols("t_ds")  
        self.t_is = symbols("t_is")  
        self.t_dh = symbols("t_dh")  
        self.t_ih = symbols("t_ih")  
        self.t_dcd_soc = symbols("t_dcd_soc")  
        self.t_dcd_dram = symbols("t_dcd_dram")    
        self.t_error_soc = symbols("t_error_soc")  
        self.t_skew_setup = symbols("t_skew_setup")  
        self.t_skew_hold = symbols("t_skew_hold")  
        self.t_dqsq = symbols("t_dqsq")  
        self.t_soc_setup = symbols("t_soc_setup")  
        self.t_soc_hold = symbols("t_soc_hold")    
        self.t_jitter_setup = symbols("t_jitter_setup")  
        self.t_jitter_hold = symbols("t_jitter_hold")  
        self.t_jitter_addr_setup = symbols("t_jitter_addr_setup")  
        self.t_jitter_addr_hold = symbols("t_jitter_addr_hold")  
        self.t_cor_margin = symbols("t_cor_margin")  
        self.r_diff_term = symbols("r_diff_term")      
        self.rtt1_dq_read = symbols("rtt1_dq_read")  
        self.rtt2_dq_read = symbols("rtt2_dq_read")  
        self.rtt1_dq_write = symbols("rtt1_dq_write")  
        self.rtt2_dq_write = symbols("rtt2_dq_write")  
        self.rtt_ca = symbols("rtt_ca")        
        self.rs1_dq = symbols("rs1_dq")  
        self.rs2_dq = symbols("rs2_dq")  
        self.r_stub_ca = symbols("r_stub_ca")  
        self.r_on = symbols("r_on")  
        self.r_on_ca = symbols("r_on_ca")  
        self.z0 = symbols("z0")    
        self.t_flight = symbols("t_flight")  
        self.t_flight_ca = symbols("t_flight_ca")  
        self.k_noise_write = symbols("k_noise_write")  
        self.k_noise_read = symbols("k_noise_read")  
        self.k_noise_addr = symbols("k_noise_addr")  
        self.v_noise_independent_write = symbols("v_noise_independent_write")      
        self.v_noise_independent_read = symbols("v_noise_independent_read")  
        self.v_noise_independent_addr = symbols("v_noise_independent_addr")  
        self.phy_datapath_s = symbols("phy_datapath_s")  
        self.phy_phase_rotator_s = symbols("phy_phase_rotator_s")  
        self.phy_clock_tree_s = symbols("phy_clock_tree_s")  
        self.phy_rx_s = symbols("phy_rx_s")        
        self.phy_dcc_s = symbols("phy_dcc_s")  
        self.phy_deskew_s = symbols("phy_deskew_s")  
        self.phy_leveling_s = symbols("phy_leveling_s")  
        self.phy_pll_s = symbols("phy_pll_s")  
        self.phy_datapath_d = symbols("phy_datapath_d")    
        self.phy_phase_rotator_d = symbols("phy_phase_rotator_d")  
        self.phy_clock_tree_d = symbols("phy_clock_tree_d")  
        self.phy_rx_d = symbols("phy_rx_d")  
        self.phy_dcc_d = symbols("phy_dcc_d")  
        self.phy_deskew_d = symbols("phy_deskew_d")    
        self.phy_leveling_d = symbols("phy_leveling_d")  
        self.phy_pll_d = symbols("phy_pll_d")  
        self.phy_pll_wtime = symbols("phy_pll_wtime")  
        self.phy_phase_rotator_wtime = symbols("phy_phase_rotator_wtime")  
        self.phy_rx_wtime = symbols("phy_rx_wtime")  
        self.phy_bandgap_wtime = symbols("phy_bandgap_wtime")  
        self.phy_deskew_wtime = symbols("phy_deskew_wtime")  
        self.phy_vrefgen_wtime = symbols("phy_vrefgen_wtime")  
        
        # not currently used
        self.BufPeriphAreaEff = symbols("buf_peripheral_area_proportion")  
        self.MemPeriphAreaEff = symbols("mem_peripheral_area_propportion")  

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
        self.constraints.append(self.V_dd >= 0)
        self.constraints.append(self.V_th >= 0)
        self.constraints.append(self.clk_period >= 0)
        self.constraints.append(self.f >= 0)
        self.constraints.append(self.u_n >= 0)
        self.constraints.append(self.tox >= 0)
        self.constraints.append(self.t_1 >= 0)
        self.constraints.append(self.L_ov >= 0)
        self.constraints.append(self.W >= 0)
        self.constraints.append(self.L >= 0)
        self.constraints.append(self.k_gate >= 0)
        self.constraints.append(self.Cs >= 0)
        self.constraints.append(self.d >= 0)
        self.constraints.append(self.k_cnt >= 0)
        self.constraints.append(self.area >= 0)
        self.constraints.append(self.L_c >= 0)
        self.constraints.append(self.H_c >= 0)
        self.constraints.append(self.H_g >= 0)
        self.constraints.append(self.beta_p_n >= 0)
        self.constraints.append(self.mD_fac >= 0)
        self.constraints.append(self.mu_eff_n >= 0)
        self.constraints.append(self.mu_eff_p >= 0)
        self.constraints.append(self.eps_semi >= 0)
        self.constraints.append(self.tsemi >= 0)
        self.constraints.append(self.Lext >= 0)
        self.constraints.append(self.Lc >= 0)
        self.constraints.append(self.eps_cap >= 0)
        self.constraints.append(self.rho_c_n >= 0)
        self.constraints.append(self.rho_c_p >= 0)
        self.constraints.append(self.Rsh_c_n >= 0)
        self.constraints.append(self.Rsh_c_p >= 0)
        self.constraints.append(self.Rsh_ext_n >= 0)
        self.constraints.append(self.Rsh_ext_p >= 0)
        self.constraints.append(self.FO >= 0)
        self.constraints.append(self.M >= 0)
        self.constraints.append(self.f >= 0)
        self.constraints.append(self.a >= 0)

    def init_symbol_table(self):
        # initialize string to symbol mapping
        self.symbol_table = {
            "node_arrivals_end": self.node_arrivals_end,
            "V_dd": self.V_dd,
            "V_th": self.V_th,
            "clk_period": self.clk_period,
            "f": self.f,
            "u_n": self.u_n,
            "tox": self.tox,
            "t_1": self.t_1,
            "L_ov": self.L_ov,
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
            "logic_sensitivity": self.logic_sensitivity,
            "logic_resource_sensitivity": self.logic_resource_sensitivity,
            "logic_ahmdal_limit": self.logic_ahmdal_limit,
            "logic_resource_ahmdal_limit": self.logic_resource_ahmdal_limit,
            "interconnect_sensitivity": self.interconnect_sensitivity,
            "interconnect_resource_sensitivity": self.interconnect_resource_sensitivity,
            "interconnect_ahmdal_limit": self.interconnect_ahmdal_limit,
            "interconnect_resource_ahmdal_limit": self.interconnect_resource_ahmdal_limit,
            "memory_sensitivity": self.memory_sensitivity,
            "memory_resource_sensitivity": self.memory_resource_sensitivity,
            "memory_ahmdal_limit": self.memory_ahmdal_limit,
            "memory_resource_ahmdal_limit": self.memory_resource_ahmdal_limit,
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
        global_symbol_table.global_symbol_table = self.symbol_table
    
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
