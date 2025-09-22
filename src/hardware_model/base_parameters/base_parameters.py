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

        self.node_arrivals_end = symbols("node_arrivals_end", positive=True)

        self.f = symbols("f", positive=True)

        # Logic parameters
        self.V_dd = symbols("V_dd", positive=True)
        self.V_th = symbols("V_th", positive=True)
        self.tox = symbols("tox", positive=True)
        self.W = symbols("W", positive=True)
        self.L = symbols("L", positive=True)
        self.k_gate = symbols("k_gate", positive=True)
        self.t_1 = symbols("t_1", positive=True) # physical body thickness, used for scale length in vs model

        # dennard scaling factors, used for dennard scaling test
        self.alpha_dennard = symbols("alpha_dennard", positive=True)
        self.epsilon_dennard = symbols("epsilon_dennard", positive=True)
        
        self.area = symbols("area", positive=True)

        # wire parameters
        self.m1_rho = symbols("m1_rho", positive=True)
        self.m2_rho = symbols("m2_rho", positive=True)
        self.m3_rho = symbols("m3_rho", positive=True)

        self.m1_k = symbols("m1_k", positive=True)
        self.m2_k = symbols("m2_k", positive=True)
        self.m3_k = symbols("m3_k", positive=True)

        # Electron mobility for NMOS
        self.u_n = symbols("u_n", positive=True)

        self.init_memory_params()

        # semiconductor capacitance for virtual source model
        self.Cs = symbols("Cs", positive=True)

        # CNT parameters
        self.d = symbols("d", positive=True)
        self.k_cnt = symbols("k_cnt", positive=True)
        self.L_c = symbols("L_c", positive=True)

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
    
        # set initial values for memory parameters
        # CACTI
        cacti_params = {}
        # TODO, cell type, temp
        dat.scan_dat(cacti_params, dat_file, 0, 0, 360)
        cacti_params = {k: (1 if v is None or math.isnan(v) else (10**(-9) if v == 0 else v)) for k, v in cacti_params.items()}
        for key, value in cacti_params.items():
            self.tech_values[key] = value
        # set initial values for dennard scaling factors (no actual meaning, they will be set by the optimizer)
        self.tech_values[self.alpha_dennard] = 1
        self.tech_values[self.epsilon_dennard] = 1

        # CACTI IO
        cacti_IO_params = {}
        # TODO figure initial
        g_ip = InputParameter()
        g_ip.num_clk = 2
        g_ip.io_type = "DDR3"
        g_ip.num_mem_dq = 3
        g_ip.mem_data_width = 2
        g_ip.num_dq = 2
        g_ip.dram_dimm = "UDIMM"
        g_ip.bus_freq = 500
        
        IO.scan_IO(cacti_IO_params, g_ip, g_ip.io_type, g_ip.num_mem_dq, g_ip.mem_data_width, g_ip.num_dq, g_ip.dram_dimm, 1, g_ip.bus_freq)
        cacti_IO_params = {k: (1 if v is None or math.isnan(v) else (10**(-9) if v == 0 else v)) for k, v in cacti_IO_params.items()}
        for key, value in cacti_IO_params.items():
            self.tech_values[key] = value

        # mock area and latency scaling for experimental purposes
        self.area_scale = (self.W * self.L).xreplace(self.tech_values) / (self.W * self.L)
        self.latency_scale = 1/self.area_scale

    def init_memory_params(self):
        # Memory parameters
        self.C_g_ideal = symbols("C_g_ideal", positive=True)
        self.C_fringe = symbols("C_fringe", positive=True)
        self.C_junc = symbols("C_junc", positive=True)
        self.C_junc_sw = symbols("C_junc_sw", positive=True)
        self.l_phy = symbols("l_phy", positive=True)    
        self.l_elec = symbols("l_elec", positive=True)
        self.nmos_effective_resistance_multiplier = symbols("nmos_effective_resistance_multiplier", positive=True)
        self.Vdd = symbols("Vdd", positive=True)
        self.Vth = symbols("Vth", positive=True)
        self.Vdsat = symbols("Vdsat", positive=True)
        self.I_on_n = symbols("I_on_n", positive=True)  
        self.I_on_p = symbols("I_on_p", positive=True)
        self.I_off_n = symbols("I_off_n", positive=True)
        self.I_g_on_n = symbols("I_g_on_n", positive=True)
        self.C_ox = symbols("C_ox", positive=True)
        self.t_ox = symbols("t_ox", positive=True)  
        self.n2p_drv_rt = symbols("n2p_drv_rt", positive=True)
        self.lch_lk_rdc = symbols("lch_lk_rdc", positive=True)
        self.Mobility_n = symbols("Mobility_n", positive=True)
        self.gmp_to_gmn_multiplier = symbols("gmp_to_gmn_multiplier", positive=True)
        self.vpp = symbols("vpp", positive=True)    
        self.Wmemcella = symbols("Wmemcella", positive=True)    
        self.Wmemcellpmos = symbols("Wmemcellpmos", positive=True)    
        self.Wmemcellnmos = symbols("Wmemcellnmos", positive=True)    
        self.area_cell = symbols("area_cell", positive=True)    
        self.asp_ratio_cell = symbols("asp_ratio_cell", positive=True)    
        self.vdd_cell = symbols("vdd_cell", positive=True)      
        self.dram_cell_I_on = symbols("dram_cell_I_on", positive=True)  
        self.dram_cell_Vdd = symbols("dram_cell_Vdd", positive=True)  
        self.dram_cell_C = symbols("dram_cell_C", positive=True)  
        self.dram_cell_I_off_worst_case_len_temp = symbols("dram_cell_I_off_worst_case_len_temp", positive=True)  
        self.logic_scaling_co_eff = symbols("logic_scaling_co_eff", positive=True)  
        self.core_tx_density = symbols("core_tx_density", positive=True)      
        self.sckt_co_eff = symbols("sckt_co_eff", positive=True)  
        self.chip_layout_overhead = symbols("chip_layout_overhead", positive=True)  
        self.macro_layout_overhead = symbols("macro_layout_overhead", positive=True)  
        self.sense_delay = symbols("sense_delay", positive=True)  
        self.sense_dy_power = symbols("sense_dy_power", positive=True)  
        self.wire_pitch = symbols("wire_pitch", positive=True)    
        self.barrier_thickness = symbols("barrier_thickness", positive=True)  
        self.dishing_thickness = symbols("dishing_thickness", positive=True)  
        self.alpha_scatter = symbols("alpha_scatter", positive=True)  
        self.aspect_ratio = symbols("aspect_ratio", positive=True)  
        self.miller_value = symbols("miller_value", positive=True)  
        self.horiz_dielectric_constant = symbols("horiz_dielectric_constant", positive=True)      
        self.vert_dielectric_constant = symbols("vert_dielectric_constant", positive=True)  
        self.ild_thickness = symbols("ild_thickness", positive=True)  
        self.fringe_cap = symbols("fringe_cap", positive=True)  
        self.resistivity = symbols("resistivity", positive=True)  
        self.wire_r_per_micron = symbols("wire_r_per_micron", positive=True)  
        self.wire_c_per_micron = symbols("wire_c_per_micron", positive=True)      
        self.tsv_pitch = symbols("tsv_pitch", positive=True)  
        self.tsv_diameter = symbols("tsv_diameter", positive=True)  
        self.tsv_length = symbols("tsv_length", positive=True)  
        self.tsv_dielec_thickness = symbols("tsv_dielec_thickness", positive=True)  
        self.tsv_contact_resistance = symbols("tsv_contact_resistance", positive=True)    
        self.tsv_depletion_width = symbols("tsv_depletion_width", positive=True)  
        self.tsv_liner_dielectric_cons = symbols("tsv_liner_dielectric_cons", positive=True)  

        # Memory I/O parameters
        self.vdd_io = symbols("vdd_io", positive=True)  
        self.v_sw_clk = symbols("v_sw_clk", positive=True)  
        self.c_int = symbols("c_int", positive=True)  
        self.c_tx = symbols("c_tx", positive=True)  
        self.c_data = symbols("c_data", positive=True)  
        self.c_addr = symbols("c_addr", positive=True)  
        self.i_bias = symbols("i_bias", positive=True)  
        self.i_leak = symbols("i_leak", positive=True)    
        self.ioarea_c = symbols("ioarea_c", positive=True)  
        self.ioarea_k0 = symbols("ioarea_k0", positive=True)  
        self.ioarea_k1 = symbols("ioarea_k1", positive=True)  
        self.ioarea_k2 = symbols("ioarea_k2", positive=True)  
        self.ioarea_k3 = symbols("ioarea_k3", positive=True)      
        self.t_ds = symbols("t_ds", positive=True)  
        self.t_is = symbols("t_is", positive=True)  
        self.t_dh = symbols("t_dh", positive=True)  
        self.t_ih = symbols("t_ih", positive=True)  
        self.t_dcd_soc = symbols("t_dcd_soc", positive=True)  
        self.t_dcd_dram = symbols("t_dcd_dram", positive=True)    
        self.t_error_soc = symbols("t_error_soc", positive=True)  
        self.t_skew_setup = symbols("t_skew_setup", positive=True)  
        self.t_skew_hold = symbols("t_skew_hold", positive=True)  
        self.t_dqsq = symbols("t_dqsq", positive=True)  
        self.t_soc_setup = symbols("t_soc_setup", positive=True)  
        self.t_soc_hold = symbols("t_soc_hold", positive=True)    
        self.t_jitter_setup = symbols("t_jitter_setup", positive=True)  
        self.t_jitter_hold = symbols("t_jitter_hold", positive=True)  
        self.t_jitter_addr_setup = symbols("t_jitter_addr_setup", positive=True)  
        self.t_jitter_addr_hold = symbols("t_jitter_addr_hold", positive=True)  
        self.t_cor_margin = symbols("t_cor_margin", positive=True)  
        self.r_diff_term = symbols("r_diff_term", positive=True)      
        self.rtt1_dq_read = symbols("rtt1_dq_read", positive=True)  
        self.rtt2_dq_read = symbols("rtt2_dq_read", positive=True)  
        self.rtt1_dq_write = symbols("rtt1_dq_write", positive=True)  
        self.rtt2_dq_write = symbols("rtt2_dq_write", positive=True)  
        self.rtt_ca = symbols("rtt_ca", positive=True)        
        self.rs1_dq = symbols("rs1_dq", positive=True)  
        self.rs2_dq = symbols("rs2_dq", positive=True)  
        self.r_stub_ca = symbols("r_stub_ca", positive=True)  
        self.r_on = symbols("r_on", positive=True)  
        self.r_on_ca = symbols("r_on_ca", positive=True)  
        self.z0 = symbols("z0", positive=True)    
        self.t_flight = symbols("t_flight", positive=True)  
        self.t_flight_ca = symbols("t_flight_ca", positive=True)  
        self.k_noise_write = symbols("k_noise_write", positive=True)  
        self.k_noise_read = symbols("k_noise_read", positive=True)  
        self.k_noise_addr = symbols("k_noise_addr", positive=True)  
        self.v_noise_independent_write = symbols("v_noise_independent_write", positive=True)      
        self.v_noise_independent_read = symbols("v_noise_independent_read", positive=True)  
        self.v_noise_independent_addr = symbols("v_noise_independent_addr", positive=True)  
        self.phy_datapath_s = symbols("phy_datapath_s", positive=True)  
        self.phy_phase_rotator_s = symbols("phy_phase_rotator_s", positive=True)  
        self.phy_clock_tree_s = symbols("phy_clock_tree_s", positive=True)  
        self.phy_rx_s = symbols("phy_rx_s", positive=True)        
        self.phy_dcc_s = symbols("phy_dcc_s", positive=True)  
        self.phy_deskew_s = symbols("phy_deskew_s", positive=True)  
        self.phy_leveling_s = symbols("phy_leveling_s", positive=True)  
        self.phy_pll_s = symbols("phy_pll_s", positive=True)  
        self.phy_datapath_d = symbols("phy_datapath_d", positive=True)    
        self.phy_phase_rotator_d = symbols("phy_phase_rotator_d", positive=True)  
        self.phy_clock_tree_d = symbols("phy_clock_tree_d", positive=True)  
        self.phy_rx_d = symbols("phy_rx_d", positive=True)  
        self.phy_dcc_d = symbols("phy_dcc_d", positive=True)  
        self.phy_deskew_d = symbols("phy_deskew_d", positive=True)    
        self.phy_leveling_d = symbols("phy_leveling_d", positive=True)  
        self.phy_pll_d = symbols("phy_pll_d", positive=True)  
        self.phy_pll_wtime = symbols("phy_pll_wtime", positive=True)  
        self.phy_phase_rotator_wtime = symbols("phy_phase_rotator_wtime", positive=True)  
        self.phy_rx_wtime = symbols("phy_rx_wtime", positive=True)  
        self.phy_bandgap_wtime = symbols("phy_bandgap_wtime", positive=True)  
        self.phy_deskew_wtime = symbols("phy_deskew_wtime", positive=True)  
        self.phy_vrefgen_wtime = symbols("phy_vrefgen_wtime", positive=True)  
        
        # not currently used
        self.BufPeriphAreaEff = symbols("buf_peripheral_area_proportion", positive=True)  
        self.MemPeriphAreaEff = symbols("mem_peripheral_area_propportion", positive=True)  

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
        pass

    def init_symbol_table(self):
        # initialize string to symbol mapping
        self.symbol_table = {
            "node_arrivals_end": self.node_arrivals_end,
            "V_dd": self.V_dd,
            "V_th": self.V_th,
            "f": self.f,
            "u_n": self.u_n,
            "tox": self.tox,
            "t_1": self.t_1,
            "W": self.W,
            "L": self.L,
            "k_gate": self.k_gate,
            "Cs": self.Cs,
            "d": self.d,
            "k_cnt": self.k_cnt,
            "area": self.area,
            "L_c": self.L_c,
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
            "m1_k": self.m1_k,
            "m2_k": self.m2_k,
            "m3_k": self.m3_k,

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
