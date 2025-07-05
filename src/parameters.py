import os
import yaml
import logging
from src import coefficients
import src.cacti.cacti_python.get_dat as dat
import src.cacti.cacti_python.get_IO as IO
from src.cacti.cacti_python.parameter import InputParameter
from src import global_symbol_table

import math
from sympy import symbols, ceiling, expand, exp, Abs

logger = logging.getLogger(__name__)

TECH_NODE_FILE = "src/params/tech_nodes.yaml"
WIRE_RC_FILE = "src/params/wire_rc.yaml"

def symbolic_convex_max(a, b, evaluate=True):
    """
    Max(a, b) in a format which ipopt accepts.
    """
    return 0.5 * (a + b + Abs(a - b, evaluate=evaluate))

def symbolic_convex_min(a, b, evaluate=True):
    """
    Min(a, b) in a format which ipopt accepts.
    """
    return 0.5 * (a + b - Abs(a - b, evaluate=evaluate))

class Parameters:
    def __init__(self, tech_node, dat_file, model_cfg):
        self.tech_node = tech_node if tech_node else "default"
        print(f"tech node: {self.tech_node}")
        # hardcoded tech node to reference for logical effort coefficients
        self.coeffs = coefficients.create_and_save_coefficients([7])
        # initialize symbolic variables and equations
        self.set_coefficients(self.coeffs)

        self.effects = model_cfg["effects"]

        self.f = symbols("f", positive=True)

        # Logic parameters
        self.V_dd = symbols("V_dd", positive=True)
        self.V_th = symbols("V_th", positive=True)
        self.Cox = symbols("Cox", positive=True)
        self.W = symbols("W", positive=True)
        self.L = symbols("L", positive=True)
        self.V_offset = 0.1 #symbols("V_offset", positive=True)
        self.q = 1.6e-19  # electron charge (C)
        self.e_si = 11.9*8.854e-12  # permittivity of silicon (F/m)
        self.e_ox = 3.9*8.854e-12  # permittivity of oxide (F/m)
        self.tox = self.e_ox/self.Cox
        self.n = 1.0
        self.K = 1.38e-23  # Boltzmann constant (J/K)
        self.T = 300  # Temperature (K)
        self.alpha_bg = 0.473
        self.beta_bg = 636
        self.Eg_0 = 1.17 # bandgap at 0K (eV)
        self.phi_b = 3.1  # Schottky barrier height (eV)
        self.m_0 = 9.109e-31  # electron mass (kg)
        self.m_ox = 0.5*self.m_0  # effective mass of electron in oxide (g)
        self.t_ox_ = symbols("t_ox_", positive=True)

        # dennard scaling factors, used for dennard scaling test
        self.alpha_dennard = symbols("alpha_dennard", positive=True)
        self.epsilon_dennard = symbols("epsilon_dennard", positive=True)
        
        self.area = symbols("area", positive=True)

        self.m1_Rsq = symbols("m1_Rsq", positive=True)
        self.m2_Rsq = symbols("m2_Rsq", positive=True)
        self.m3_Rsq = symbols("m3_Rsq", positive=True)

        self.m1_Csq = symbols("m1_Csq", positive=True)
        self.m2_Csq = symbols("m2_Csq", positive=True)
        self.m3_Csq = symbols("m3_Csq", positive=True)
        self.wire_parasitics = {
            "R": {
                "metal1": self.m1_Rsq,
                "metal2": self.m2_Rsq,
                "metal3": self.m3_Rsq,
            },
            "C": {
                "metal1": self.m1_Csq,
                "metal2": self.m2_Csq,
                "metal3": self.m3_Csq,
            }
        }

        self.A_gate = self.W*self.L
        self.C_gate = self.Cox*self.A_gate

        self.L_critical = 0.5e-6
        self.m = 2

        self.alpha_L = 2

        self.lambda_L = 0

        self.eta_L = 0.2/(1+(self.L/15e-9)**self.m)

        self.V_ref = 1

        self.V_th_eff = self.V_th

         # Electron mobility for NMOS
        self.u_n = symbols("u_n", positive=True)
        self.u_p = self.u_n/2.5  # Hole mobility for PMOS, hardcoded for now.

        self.init_memory_params()

        # technology level parameter values
        self.tech_values = {}
        self.init_symbol_table()
        self.init_memory_param_list()

        wire_rc_init = "" # can set later
        wire_rc_config = yaml.load(open(WIRE_RC_FILE), Loader=yaml.Loader)
        for key in wire_rc_config["default"]:
            try:
                self.tech_values[self.symbol_table[key]] = wire_rc_config[wire_rc_init][key]
            except:
                logger.info(f"using default value for {key}")
                self.tech_values[self.symbol_table[key]] = wire_rc_config["default"][key]

        # set initial values for technology parameters based on tech node
        config = yaml.load(open(TECH_NODE_FILE), Loader=yaml.Loader)
        print(config[self.tech_node])
        for key in config["default"]:
            try:
                self.tech_values[self.symbol_table[key]] = config[self.tech_node][key]
            except:
                logger.info(f"using default value for {key}")
                self.tech_values[self.symbol_table[key]] = config["default"][key]

        self.area_scale = (self.W * self.L).subs(self.tech_values) / (self.W * self.L)

        print(f"area scale: {self.area_scale.subs(self.tech_values)}")

        self.latency_scale = 1/self.area_scale

        self.apply_base_parameter_effects()

        # drive current, including alpha power law and channel length modulation
        self.I_d_nmos = self.u_n*self.Cox*self.W*(Abs(self.V_dd-self.V_th_eff))**self.alpha_L * (1+self.lambda_L*self.V_dd) / (2*self.L)
        self.I_d_pmos = self.I_d_nmos * self.u_p / self.u_n

        self.R_avg_inv = self.V_dd / ((self.I_d_nmos + self.I_d_pmos)/2)

        self.wire_len = 100 #um
        #print(f"C_load_wire: {C_load_wire}")
        print(f"C_gate: {self.C_gate}")

        self.C_wire = self.wire_parasitics["C"]["metal1"] * self.wire_len
        self.R_wire = self.wire_parasitics["R"]["metal1"] * self.wire_len

        self.gamma_diff = 1.0
        self.C_diff = self.gamma_diff * self.C_gate
        self.C_load = self.C_gate # gate cap
        print(f"C_load: {self.C_load}")
        if model_cfg["delay_parasitics"] == "all":
            self.delay = (self.R_avg_inv * (self.C_diff + self.C_wire/2) + (self.R_avg_inv + self.R_wire) * (self.C_wire/2 + self.C_load)) * 1e9  # ns
        elif model_cfg["delay_parasitics"] == "Csq only":
            self.delay = (self.R_avg_inv * (self.C_diff + self.C_wire/2) + (self.R_avg_inv) * (self.C_wire/2 + self.C_load)) * 1e9  # ns
        elif model_cfg["delay_parasitics"] == "const":
            self.delay = self.R_avg_inv * (self.C_diff + self.C_load + 0.3e-15 * 100) * 1e9  # ns
        else:
            self.delay = self.R_avg_inv * (self.C_load + self.C_diff) * 1e9

        self.E_act_inv = (0.5*self.C_load*self.V_dd*self.V_dd) * 1e9  # nJ

        self.h = 6.626e-34  # planck's constant (J*s)
        self.V_ox = self.V_dd - self.V_th_eff
        self.E_ox = Abs(self.V_ox/self.tox)
        self.A = ((self.q)**3) / (8*math.pi*self.h*self.phi_b*self.q)
        self.B = (8*math.pi*(2*self.m_ox)**(1/2) * (self.phi_b*self.q)**(3/2)) / (3*self.q*self.h)
        print(f"B: {self.B}, A: {self.A}, t_ox: {self.tox.subs(self.tech_values)}, E_ox: {self.E_ox.subs(self.tech_values)}, intermediate: {(1-(1-self.V_ox/self.phi_b)**3/2).subs(self.tech_values)}")

        # gate tunneling current (Fowler-Nordheim and WKB)
        # minimums are to avoid exponential explosion in solver. Normal values in exponent are negative.
        self.FN_term = self.A_gate * self.A * self.E_ox**2 * (exp(symbolic_convex_min(10, -self.B/self.E_ox)))
        self.WKB_term = self.A_gate * self.A * self.E_ox**2 * (exp(symbolic_convex_min(10, -self.B*(1-(1-self.V_ox/self.phi_b)**3/2)/self.E_ox)))
        self.I_tunnel = self.FN_term + self.WKB_term
        print(f"I_tunnel: {self.I_tunnel.subs(self.tech_values)}")


        # subthreshold current
        self.I_off_nmos = self.u_n*self.Cox*(self.W/self.L)*(self.K*self.T/self.q)**2*exp(-self.V_th_eff*self.q/(self.n*self.K*self.T))
        self.I_off_pmos = self.u_p*self.Cox*(self.W/self.L)*(self.K*self.T/self.q)**2*exp(-self.V_th_eff*self.q/(self.n*self.K*self.T))
        
        self.I_off = self.I_off_nmos + self.I_off_pmos # 2 for both NMOS and PMOS
        self.P_pass_inv = self.I_off*self.V_dd  

        self.apply_additional_effects()

        print(f"I_d: {self.I_d_nmos}")
        print(f"I_off: {self.I_off}")

        # UNITS: ns
        self.symbolic_latency_wc = {
            "And": lambda: self.make_sym_lat_wc(self.gamma["And"]),
            "Or": lambda: self.make_sym_lat_wc(self.gamma["Or"]),
            "Add": lambda: self.make_sym_lat_wc(self.gamma["Add"]),
            "Sub": lambda: self.make_sym_lat_wc(self.gamma["Sub"]),
            "Mult": lambda: self.make_sym_lat_wc(self.gamma["Mult"]),
            "FloorDiv": lambda: self.make_sym_lat_wc(self.gamma["FloorDiv"]),
            "Mod": lambda: self.make_sym_lat_wc(self.gamma["Mod"]),
            "LShift": lambda: self.make_sym_lat_wc(self.gamma["LShift"]),
            "RShift": lambda: self.make_sym_lat_wc(self.gamma["RShift"]),
            "BitOr": lambda: self.make_sym_lat_wc(self.gamma["BitOr"]),
            "BitXor": lambda: self.make_sym_lat_wc(self.gamma["BitXor"]),
            "BitAnd": lambda: self.make_sym_lat_wc(self.gamma["BitAnd"]),
            "Eq": lambda: self.make_sym_lat_wc(self.gamma["Eq"]),
            "NotEq": lambda: self.make_sym_lat_wc(self.gamma["NotEq"]),
            "Lt": lambda: self.make_sym_lat_wc(self.gamma["Lt"]),
            "LtE": lambda: self.make_sym_lat_wc(self.gamma["LtE"]),
            "Gt": lambda: self.make_sym_lat_wc(self.gamma["Gt"]),
            "GtE": lambda: self.make_sym_lat_wc(self.gamma["GtE"]), 
            "USub": lambda: self.make_sym_lat_wc(self.gamma["USub"]),   
            "UAdd": lambda: self.make_sym_lat_wc(self.gamma["UAdd"]),
            "IsNot": lambda: self.make_sym_lat_wc(self.gamma["IsNot"]),
            "Not": lambda: self.make_sym_lat_wc(self.gamma["Not"]),
            "Invert": lambda: self.make_sym_lat_wc(self.gamma["Invert"]),
            "Regs": lambda: self.make_sym_lat_wc(self.gamma["Regs"]),   
            "Buf": lambda: self.make_buf_lat_dict(),    
            "MainMem": lambda: self.make_mem_lat_dict(),
            "OffChipIO": lambda: self.make_io_lat_dict(),
        }

        # UNITS: nJ
        self.symbolic_energy_active = {
            "And": lambda: self.make_sym_energy_act(self.alpha["And"]),
            "Or": lambda: self.make_sym_energy_act(self.alpha["Or"]),
            "Add": lambda: self.make_sym_energy_act(self.alpha["Add"]),
            "Sub": lambda: self.make_sym_energy_act(self.alpha["Sub"]),
            "Mult": lambda: self.make_sym_energy_act(self.alpha["Mult"]),
            "FloorDiv": lambda: self.make_sym_energy_act(self.alpha["FloorDiv"]),
            "Mod": lambda: self.make_sym_energy_act(self.alpha["Mod"]),
            "LShift": lambda: self.make_sym_energy_act(self.alpha["LShift"]),
            "RShift": lambda: self.make_sym_energy_act(self.alpha["RShift"]),
            "BitOr": lambda: self.make_sym_energy_act(self.alpha["BitOr"]),
            "BitXor": lambda: self.make_sym_energy_act(self.alpha["BitXor"]),
            "BitAnd": lambda: self.make_sym_energy_act(self.alpha["BitAnd"]),
            "Eq": lambda: self.make_sym_energy_act(self.alpha["Eq"]),
            "NotEq": lambda: self.make_sym_energy_act(self.alpha["NotEq"]),
            "Lt": lambda: self.make_sym_energy_act(self.alpha["Lt"]),
            "LtE": lambda: self.make_sym_energy_act(self.alpha["LtE"]),
            "Gt": lambda: self.make_sym_energy_act(self.alpha["Gt"]),
            "GtE": lambda: self.make_sym_energy_act(self.alpha["GtE"]),
            "USub": lambda: self.make_sym_energy_act(self.alpha["USub"]),
            "UAdd": lambda: self.make_sym_energy_act(self.alpha["UAdd"]),
            "IsNot": lambda: self.make_sym_energy_act(self.alpha["IsNot"]),
            "Not": lambda: self.make_sym_energy_act(self.alpha["Not"]),
            "Invert": lambda: self.make_sym_energy_act(self.alpha["Invert"]),   
            "Regs": lambda: self.make_sym_energy_act(self.alpha["Regs"]),
            "Buf": lambda: self.make_buf_energy_active_dict(),
            "MainMem": lambda: self.make_mainmem_energy_active_dict(),
            "OffChipIO": lambda: self.make_io_energy_active_dict(),
        }

        # UNITS: W
        self.symbolic_power_passive = {
            "And": lambda: self.make_sym_power_pass(self.beta["And"]),
            "Or": lambda: self.make_sym_power_pass(self.beta["Or"]),
            "Add": lambda: self.make_sym_power_pass(self.beta["Add"]),
            "Sub": lambda: self.make_sym_power_pass(self.beta["Sub"]),
            "Mult": lambda: self.make_sym_power_pass(self.beta["Mult"]),
            "FloorDiv": lambda: self.make_sym_power_pass(self.beta["FloorDiv"]),
            "Mod": lambda: self.make_sym_power_pass(self.beta["Mod"]),
            "LShift": lambda: self.make_sym_power_pass(self.beta["LShift"]),
            "RShift": lambda: self.make_sym_power_pass(self.beta["RShift"]),
            "BitOr": lambda: self.make_sym_power_pass(self.beta["BitOr"]),
            "BitXor": lambda: self.make_sym_power_pass(self.beta["BitXor"]),
            "BitAnd": lambda: self.make_sym_power_pass(self.beta["BitAnd"]),
            "Eq": lambda: self.make_sym_power_pass(self.beta["Eq"]),
            "NotEq": lambda: self.make_sym_power_pass(self.beta["NotEq"]),
            "Lt": lambda: self.make_sym_power_pass(self.beta["Lt"]),
            "LtE": lambda: self.make_sym_power_pass(self.beta["LtE"]),
            "Gt": lambda: self.make_sym_power_pass(self.beta["Gt"]),
            "GtE": lambda: self.make_sym_power_pass(self.beta["GtE"]),
            "USub": lambda: self.make_sym_power_pass(self.beta["USub"]),
            "UAdd": lambda: self.make_sym_power_pass(self.beta["UAdd"]),
            "IsNot": lambda: self.make_sym_power_pass(self.beta["IsNot"]),
            "Not": lambda: self.make_sym_power_pass(self.beta["Not"]),
            "Invert": lambda: self.make_sym_power_pass(self.beta["Invert"]),
            "Regs": lambda: self.make_sym_power_pass(self.beta["Regs"]),
            "MainMem": lambda: self.make_mainmem_power_passive_dict(),
            "Buf": lambda: self.make_buf_power_passive_dict(),
        }

        # UNITS: um^2
        self.symbolic_area = {
            "And": lambda: self.make_sym_area(self.area_coeffs["And"]),
            "Or": lambda: self.make_sym_area(self.area_coeffs["Or"]),
            "Add": lambda: self.make_sym_area(self.area_coeffs["Add"]),
            "Sub": lambda: self.make_sym_area(self.area_coeffs["Sub"]),
            "Mult": lambda: self.make_sym_area(self.area_coeffs["Mult"]),
            "FloorDiv": lambda: self.make_sym_area(self.area_coeffs["FloorDiv"]),
            "Mod": lambda: self.make_sym_area(self.area_coeffs["Mod"]), 
            "LShift": lambda: self.make_sym_area(self.area_coeffs["LShift"]),
            "RShift": lambda: self.make_sym_area(self.area_coeffs["RShift"]),
            "BitOr": lambda: self.make_sym_area(self.area_coeffs["BitOr"]),
            "BitXor": lambda: self.make_sym_area(self.area_coeffs["BitXor"]),
            "BitAnd": lambda: self.make_sym_area(self.area_coeffs["BitAnd"]),
            "Eq": lambda: self.make_sym_area(self.area_coeffs["Eq"]),
            "NotEq": lambda: self.make_sym_area(self.area_coeffs["NotEq"]),
            "Lt": lambda: self.make_sym_area(self.area_coeffs["Lt"]),
            "LtE": lambda: self.make_sym_area(self.area_coeffs["LtE"]),
            "Gt": lambda: self.make_sym_area(self.area_coeffs["Gt"]),
            "GtE": lambda: self.make_sym_area(self.area_coeffs["GtE"]),
            "USub": lambda: self.make_sym_area(self.area_coeffs["USub"]),
            "UAdd": lambda: self.make_sym_area(self.area_coeffs["UAdd"]),
            "IsNot": lambda: self.make_sym_area(self.area_coeffs["IsNot"]),
            "Not": lambda: self.make_sym_area(self.area_coeffs["Not"]),
            "Invert": lambda: self.make_sym_area(self.area_coeffs["Invert"]),
            "Regs": lambda: self.make_sym_area(self.area_coeffs["Regs"]),
        }

        # memories output from forward pass
        self.memories = {}

        # main mem from inverse pass
        self.symbolic_mem = {}

        # buffers from inverse pass
        self.symbolic_buf = {}

        # symbolic expressions for resource attributes (i.e. Buf latency) from inverse pass
        self.symbolic_rsc_exprs = {}

        # circuit level parameter values
        self.circuit_values = {}

        # wire length by edge
        self.wire_length_by_edge = {}

        self.metal_layers = ["metal1", "metal2", "metal3"]


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
        self.tech_values[self.t_ox_] = self.e_ox / self.tech_values[self.Cox]

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

        self.update_circuit_values()

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

    def apply_base_parameter_effects(self):
        if self.effects["velocity_saturation"]:
            # Sakurai alpha-power law
            self.alpha_L =1 + 1/(1 + (self.L/self.L_critical)**self.m)
        if self.effects["channel_length_modulation"]:
            self.lambda_L = 0.1
        if self.effects["mobility_degradation"]:
            theta_1 = 1
            theta_2 = 1
            self.u_n = self.u_n / (1+theta_1*(self.V_dd-self.V_th_eff)/(1+theta_2*(self.V_dd-self.V_th_eff)))
            self.u_p = self.u_n / 2.5
        if self.effects["subthreshold_slope_effect"]:
            self.n += (self.e_si/self.e_ox) * (self.tox/self.L)**(1/2)
        if self.effects["DIBL"]:
            self.V_th_eff -= self.eta_L * (symbolic_convex_max(self.V_dd - self.V_ref, 0))

    def apply_additional_effects(self):
        print(self.I_tunnel)
        tunnel = self.I_tunnel.subs(self.tech_values)
        print(f"i tunnel: {tunnel}")

        #print(f"u_n: {self.u_n.subs(self.tech_values)}")
        if self.effects["gate_tunneling"]:
            self.I_off += self.I_tunnel*2 # 2 for both NMOS and PMOS
            self.P_pass_inv = self.I_off * self.V_dd
        if self.effects["area_and_latency_scaling"]:
            self.delay = self.delay * self.latency_scale
            self.P_pass_inv = self.P_pass_inv * self.area_scale
            self.wire_len = self.wire_len * self.latency_scale
            
    def set_memories(self, memories):
        self.memories = memories
        self.update_circuit_values()

    def compare_symbolic_mem(self):
        for key in self.symbolic_mem:
            assert key in self.memories, f"symbolic memory {key} not found in memories"       

    def update_circuit_values(self):
        # derive curcuit level values from technology values
        self.circuit_values["latency"] = {
            key: float(self.symbolic_latency_wc[key]().subs(self.tech_values)) for key in self.symbolic_latency_wc if key not in ["Buf", "MainMem", "OffChipIO"]
        }
        self.circuit_values["dynamic_energy"] = {
            key: float(self.symbolic_energy_active[key]().subs(self.tech_values)) for key in self.symbolic_energy_active if key not in ["Buf", "MainMem", "OffChipIO"]
        }
        self.circuit_values["passive_power"] = {
            key: float(self.symbolic_power_passive[key]().subs(self.tech_values)) for key in self.symbolic_power_passive if key not in ["Buf", "MainMem"]
        }
        self.circuit_values["area"] = {
            key: float(self.symbolic_area[key]().subs(self.tech_values)) for key in self.symbolic_area
        }

        # memory values
        self.circuit_values["latency"]["rsc"] = {
            key: self.memories[key]["Access time (ns)"] for key in self.memories
        }
        self.circuit_values["dynamic_energy"]["rsc"] = {
            "Read": {
                key: self.memories[key]["Dynamic read energy (nJ)"] for key in self.memories
            },
            "Write": {
                key: self.memories[key]["Dynamic write energy (nJ)"] for key in self.memories
            }
        }
        self.circuit_values["passive_power"]["rsc"] = {
            key: self.memories[key]["Standby leakage per bank(mW)"] * 1e-3 for key in self.memories
        }
        self.circuit_values["area"]["rsc"] = {
            key: self.memories[key]["Area (mm2)"] * 1e6 for key in self.memories
        }

    def wire_delay(self, edge, symbolic=False):
        # wire delay = R * C * length^2 (ns)
        if symbolic:
            return sum([self.wire_length_by_edge[edge][layer]**2 * 
                        self.wire_parasitics["R"][layer] * 
                        self.wire_parasitics["C"][layer] 
                        if layer in self.wire_length_by_edge[edge]
                        else 0
                        for layer in self.metal_layers]) * 1e9
        else:
            return sum([self.wire_length_by_edge[edge][layer]**2 * 
                        self.tech_values[self.wire_parasitics["R"][layer]] * 
                        self.tech_values[self.wire_parasitics["C"][layer]] 
                        if layer in self.wire_length_by_edge[edge]
                        else 0
                        for layer in self.metal_layers]) * 1e9
        
    def wire_energy(self, edge, symbolic=False):
        # wire energy = 0.5 * C * V_dd^2 * length
        if symbolic:
            return 0.5*sum([self.wire_length_by_edge[edge][layer] * 
                        self.wire_parasitics["C"][layer] * self.V_dd**2 
                        if layer in self.wire_length_by_edge[edge]
                        else 0
                        for layer in self.metal_layers]) * 1e9
        else:
            return 0.5*sum([self.wire_length_by_edge[edge][layer] * 
                        self.tech_values[self.wire_parasitics["C"][layer]] * self.tech_values[self.V_dd]**2 
                        if layer in self.wire_length_by_edge[edge]
                        else 0
                        for layer in self.metal_layers]) * 1e9

    def set_coefficients(self, coeffs):
        self.alpha = coeffs["alpha"]
        self.beta = coeffs["beta"]
        self.gamma = coeffs["gamma"]
        self.area_coeffs = coeffs["area"]

    def make_sym_lat_wc(self, gamma):
        return gamma * self.delay

    def make_buf_lat_dict(self):
        return self.BufL

    def make_mem_lat_dict(self):    
        d = {}
        for mem in self.MemReadL:
            d[mem] = (self.MemReadL[mem] + self.MemWriteL[mem]) / 2
        return d

    def make_io_lat_dict(self):
        return self.OffChipIOL
    
    def make_buf_energy_active_dict(self):
        d = {}
        for mem in self.BufReadEact:
            d[mem] = ((self.BufReadEact[mem] + self.BufWriteEact[mem]) / 2)
        return d

    def make_mainmem_energy_active_dict(self):
        d = {}
        for mem in self.MemWriteEact:
            d[mem] = ((self.MemWriteEact[mem] + self.MemReadEact[mem]) / 2)
        return d    

    def make_io_energy_active_dict(self):
        return self.OffChipIOPact * self.OffChipIOL
    
    def make_sym_energy_act(self, alpha):
        return alpha * self.E_act_inv

    def make_mainmem_power_passive_dict(self):
        return self.MemPpass
    
    def make_buf_power_passive_dict(self):
        return self.BufPpass
    
    def make_sym_power_pass(self, beta):
        return beta * self.P_pass_inv

    def make_sym_area(self, area_coeff):
        return area_coeff * self.area
    
    def init_symbol_table(self):
        # initialize string to symbol mapping
        self.symbol_table = {
            "V_dd": self.V_dd,
            "V_th": self.V_th,
            "f": self.f,
            "u_n": self.u_n,
            "u_p": self.u_p,
            "Cox": self.Cox,
            "t_ox_": self.t_ox_,
            "W": self.W,
            "L": self.L,
            "q": self.q,
            "V_offset": self.V_offset,
            "n": self.n,
            "K": self.K,
            "T": self.T,
            "area": self.area,
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
            "m1_Rsq": self.m1_Rsq,
            "m2_Rsq": self.m2_Rsq,
            "m3_Rsq": self.m3_Rsq,
            "m1_Csq": self.m1_Csq,
            "m2_Csq": self.m2_Csq,
            "m3_Csq": self.m3_Csq,
            
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
    

# def make_sym_lat_cyc(f, lat_wc): # bad name, output is not in units of cycles, its in units of time.
#     return ceiling(f*lat_wc)/f

# symbolic_latency_cyc = {
#     "And": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["And"]),
#     "Or": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["Or"]),
#     "Add": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["Add"]),
#     "Sub": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["Sub"]),
#     "Mult": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["Mult"]),
#     "FloorDiv": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["FloorDiv"]),
#     "Mod": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["Mod"]),
#     "LShift": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["LShift"]),
#     "RShift": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["RShift"]),
#     "BitOr": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["BitOr"]),
#     "BitXor": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["BitXor"]),
#     "BitAnd": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["BitAnd"]),
#     "Eq": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["Eq"]),
#     "NotEq": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["NotEq"]),
#     "Lt": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["Lt"]),
#     "LtE": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["LtE"]),
#     "Gt": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["Gt"]),
#     "GtE": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["GtE"]),
#     "USub": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["USub"]),
#     "UAdd": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["UAdd"]),
#     "IsNot": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["IsNot"]),
#     "Not": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["Not"]),
#     "Invert": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["Invert"]),
#     "Regs": lambda: make_sym_lat_cyc(f, symbolic_latency_wc["Regs"]),
#     "Buf": BufL,
#     "MainMem": (MemReadL + MemWriteL) / 2,
#     "OffChipIO": OffChipIOL,
# }


if __name__ == "__main__":
    Param = Parameters("DEFAULT")