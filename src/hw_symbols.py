import os
import yaml

from sympy import symbols, ceiling, expand, exp

V_dd = symbols("V_dd", positive=True)
f = symbols("f", positive=True)
MemReadL = {}
MemWriteL = {}
MemReadEact = {}
MemWriteEact = {}
MemPpass = {}
BufL = {}
BufReadEact = {}
BufWriteEact = {}
BufPpass = {}
OffChipIOL = {}
OffChipIOPact = {}

# Cacti .dat technology parameters
C_g_ideal = symbols('C_g_ideal', positive=True)
C_fringe = symbols('C_fringe', positive=True)
C_junc = symbols('C_junc', positive=True)
C_junc_sw = symbols('C_junc_sw', positive=True)
l_phy = symbols('l_phy', positive=True)
l_elec = symbols('l_elec', positive=True)
nmos_effective_resistance_multiplier = symbols('nmos_effective_resistance_multiplier', positive=True)
Vdd = symbols('Vdd', positive=True)
Vth = symbols('Vth', positive=True)
Vdsat = symbols('Vdsat', positive=True)
I_on_n = symbols('I_on_n', positive=True)
I_on_p = symbols('I_on_p', positive=True)
I_off_n = symbols('I_off_n', positive=True)
I_g_on_n = symbols('I_g_on_n', positive=True)
C_ox = symbols('C_ox', positive=True)
t_ox = symbols('t_ox', positive=True)
n2p_drv_rt = symbols('n2p_drv_rt', positive=True)
lch_lk_rdc = symbols('lch_lk_rdc', positive=True)
Mobility_n = symbols('Mobility_n', positive=True)
gmp_to_gmn_multiplier = symbols('gmp_to_gmn_multiplier', positive=True)
vpp = symbols('vpp', positive=True)
Wmemcella = symbols('Wmemcella', positive=True)
Wmemcellpmos = symbols('Wmemcellpmos', positive=True)
Wmemcellnmos = symbols('Wmemcellnmos', positive=True)
area_cell = symbols('area_cell', positive=True)
asp_ratio_cell = symbols('asp_ratio_cell', positive=True)
vdd_cell = symbols('vdd_cell', positive=True)
dram_cell_I_on = symbols('dram_cell_I_on', positive=True)
dram_cell_Vdd = symbols('dram_cell_Vdd', positive=True)
dram_cell_C = symbols('dram_cell_C', positive=True)
dram_cell_I_off_worst_case_len_temp = symbols('dram_cell_I_off_worst_case_len_temp', positive=True)
logic_scaling_co_eff = symbols('logic_scaling_co_eff', positive=True)
core_tx_density = symbols('core_tx_density', positive=True)
sckt_co_eff = symbols('sckt_co_eff', positive=True)
chip_layout_overhead = symbols('chip_layout_overhead', positive=True)
macro_layout_overhead = symbols('macro_layout_overhead', positive=True)
sense_delay = symbols('sense_delay', positive=True)
sense_dy_power = symbols('sense_dy_power', positive=True)
wire_pitch = symbols('wire_pitch', positive=True)
barrier_thickness = symbols('barrier_thickness', positive=True)
dishing_thickness = symbols('dishing_thickness', positive=True)
alpha_scatter = symbols('alpha_scatter', positive=True)
aspect_ratio = symbols('aspect_ratio', positive=True)
miller_value = symbols('miller_value', positive=True)
horiz_dielectric_constant = symbols('horiz_dielectric_constant', positive=True)
vert_dielectric_constant = symbols('vert_dielectric_constant', positive=True)
ild_thickness = symbols('ild_thickness', positive=True)
fringe_cap = symbols('fringe_cap', positive=True)
resistivity = symbols('resistivity', positive=True)
wire_r_per_micron = symbols('wire_r_per_micron', positive=True)
wire_c_per_micron = symbols('wire_c_per_micron', positive=True)
tsv_pitch = symbols('tsv_pitch', positive=True)
tsv_diameter = symbols('tsv_diameter', positive=True)
tsv_length = symbols('tsv_length', positive=True)
tsv_dielec_thickness = symbols('tsv_dielec_thickness', positive=True)
tsv_contact_resistance = symbols('tsv_contact_resistance', positive=True)
tsv_depletion_width = symbols('tsv_depletion_width', positive=True)
tsv_liner_dielectric_cons = symbols('tsv_liner_dielectric_cons', positive=True)

# CACTI I/O
vdd_io = symbols('vdd_io', positive=True)
v_sw_clk = symbols('v_sw_clk', positive=True)
c_int = symbols('c_int', positive=True)
c_tx = symbols('c_tx', positive=True)
c_data = symbols('c_data', positive=True)
c_addr = symbols('c_addr', positive=True)
i_bias = symbols('i_bias', positive=True)
i_leak = symbols('i_leak', positive=True)
ioarea_c = symbols('ioarea_c', positive=True)
ioarea_k0 = symbols('ioarea_k0', positive=True)
ioarea_k1 = symbols('ioarea_k1', positive=True)
ioarea_k2 = symbols('ioarea_k2', positive=True)
ioarea_k3 = symbols('ioarea_k3', positive=True)
t_ds = symbols('t_ds', positive=True)
t_is = symbols('t_is', positive=True)
t_dh = symbols('t_dh', positive=True)
t_ih = symbols('t_ih', positive=True)
t_dcd_soc = symbols('t_dcd_soc', positive=True)
t_dcd_dram = symbols('t_dcd_dram', positive=True)
t_error_soc = symbols('t_error_soc', positive=True)
t_skew_setup = symbols('t_skew_setup', positive=True)
t_skew_hold = symbols('t_skew_hold', positive=True)
t_dqsq = symbols('t_dqsq', positive=True)
t_soc_setup = symbols('t_soc_setup', positive=True)
t_soc_hold = symbols('t_soc_hold', positive=True)
t_jitter_setup = symbols('t_jitter_setup', positive=True)
t_jitter_hold = symbols('t_jitter_hold', positive=True)
t_jitter_addr_setup = symbols('t_jitter_addr_setup', positive=True)
t_jitter_addr_hold = symbols('t_jitter_addr_hold', positive=True)
t_cor_margin = symbols('t_cor_margin', positive=True)
r_diff_term = symbols('r_diff_term', positive=True)
rtt1_dq_read = symbols('rtt1_dq_read', positive=True)
rtt2_dq_read = symbols('rtt2_dq_read', positive=True)
rtt1_dq_write = symbols('rtt1_dq_write', positive=True)
rtt2_dq_write = symbols('rtt2_dq_write', positive=True)
rtt_ca = symbols('rtt_ca', positive=True)
rs1_dq = symbols('rs1_dq', positive=True)
rs2_dq = symbols('rs2_dq', positive=True)
r_stub_ca = symbols('r_stub_ca', positive=True)
r_on = symbols('r_on', positive=True)
r_on_ca = symbols('r_on_ca', positive=True)
z0 = symbols('z0', positive=True)
t_flight = symbols('t_flight', positive=True)
t_flight_ca = symbols('t_flight_ca', positive=True)
k_noise_write = symbols('k_noise_write', positive=True)
k_noise_read = symbols('k_noise_read', positive=True)
k_noise_addr = symbols('k_noise_addr', positive=True)
v_noise_independent_write = symbols('v_noise_independent_write', positive=True)
v_noise_independent_read = symbols('v_noise_independent_read', positive=True)
v_noise_independent_addr = symbols('v_noise_independent_addr', positive=True)
phy_datapath_s = symbols('phy_datapath_s', positive=True)
phy_phase_rotator_s = symbols('phy_phase_rotator_s', positive=True)
phy_clock_tree_s = symbols('phy_clock_tree_s', positive=True)
phy_rx_s = symbols('phy_rx_s', positive=True)
phy_dcc_s = symbols('phy_dcc_s', positive=True)
phy_deskew_s = symbols('phy_deskew_s', positive=True)
phy_leveling_s = symbols('phy_leveling_s', positive=True)
phy_pll_s = symbols('phy_pll_s', positive=True)
phy_datapath_d = symbols('phy_datapath_d', positive=True)
phy_phase_rotator_d = symbols('phy_phase_rotator_d', positive=True)
phy_clock_tree_d = symbols('phy_clock_tree_d', positive=True)
phy_rx_d = symbols('phy_rx_d', positive=True)
phy_dcc_d = symbols('phy_dcc_d', positive=True)
phy_deskew_d = symbols('phy_deskew_d', positive=True)
phy_leveling_d = symbols('phy_leveling_d', positive=True)
phy_pll_d = symbols('phy_pll_d', positive=True)
phy_pll_wtime = symbols('phy_pll_wtime', positive=True)
phy_phase_rotator_wtime = symbols('phy_phase_rotator_wtime', positive=True)
phy_rx_wtime = symbols('phy_rx_wtime', positive=True)
phy_bandgap_wtime = symbols('phy_bandgap_wtime', positive=True)
phy_deskew_wtime = symbols('phy_deskew_wtime', positive=True)
phy_vrefgen_wtime = symbols('phy_vrefgen_wtime', positive=True)


# not currently used
BufPeriphAreaEff = symbols("buf_peripheral_area_proportion", positive=True)
MemPeriphAreaEff = symbols("mem_peripheral_area_propportion", positive=True)

# UNITS: Ohms
Reff = {
    "And": symbols("Reff_And", positive=True),
    "Or": symbols("Reff_Or", positive=True),
    "Add": symbols("Reff_Add", positive=True),
    "Sub": symbols("Reff_Sub", positive=True),
    "Mult": symbols("Reff_Mult", positive=True),
    "FloorDiv": symbols("Reff_FloorDiv", positive=True),
    "Mod": symbols("Reff_Mod", positive=True),
    "LShift": symbols("Reff_LShift", positive=True),
    "RShift": symbols("Reff_RShift", positive=True),
    "BitOr": symbols("Reff_BitOr", positive=True),
    "BitXor": symbols("Reff_BitXor", positive=True),
    "BitAnd": symbols("Reff_BitAnd", positive=True),
    "Eq": symbols("Reff_Eq", positive=True),
    "NotEq": symbols("Reff_NotEq", positive=True),
    "Lt": symbols("Reff_Lt", positive=True),
    "LtE": symbols("Reff_LtE", positive=True),
    "Gt": symbols("Reff_Gt", positive=True),
    "GtE": symbols("Reff_GtE", positive=True),
    "USub": symbols("Reff_USub", positive=True),
    "UAdd": symbols("Reff_UAdd", positive=True),
    "IsNot": symbols("Reff_IsNot", positive=True),
    "Not": symbols("Reff_Not", positive=True),
    "Invert": symbols("Reff_Invert", positive=True),
    "Regs": symbols("Reff_Regs", positive=True)
}

# UNITS: nF
Ceff = {
    "And": symbols("Ceff_And", positive=True),
    "Or": symbols("Ceff_Or", positive=True),
    "Add": symbols("Ceff_Add", positive=True),
    "Sub": symbols("Ceff_Sub", positive=True),
    "Mult": symbols("Ceff_Mult", positive=True),
    "FloorDiv": symbols("Ceff_FloorDiv", positive=True),
    "Mod": symbols("Ceff_Mod", positive=True),
    "LShift": symbols("Ceff_LShift", positive=True),
    "RShift": symbols("Ceff_RShift", positive=True),
    "BitOr": symbols("Ceff_BitOr", positive=True),
    "BitXor": symbols("Ceff_BitXor", positive=True),
    "BitAnd": symbols("Ceff_BitAnd", positive=True),
    "Eq": symbols("Ceff_Eq", positive=True),
    "NotEq": symbols("Ceff_NotEq", positive=True),
    "Lt": symbols("Ceff_Lt", positive=True),
    "LtE": symbols("Ceff_LtE", positive=True),
    "Gt": symbols("Ceff_Gt", positive=True),
    "GtE": symbols("Ceff_GtE", positive=True),
    "USub": symbols("Ceff_USub", positive=True),
    "UAdd": symbols("Ceff_UAdd", positive=True),
    "IsNot": symbols("Ceff_IsNot", positive=True),
    "Not": symbols("Ceff_Not", positive=True),
    "Invert": symbols("Ceff_Invert", positive=True),
    "Regs": symbols("Ceff_Regs", positive=True),
}

symbol_table = {
    "V_dd": V_dd,
    "MemReadL": MemReadL,
    "MemWriteL": MemWriteL,
    "MemReadEact": MemReadEact,
    "MemWriteEact": MemWriteEact,
    "MemPpass": MemPpass,
    "BufL": BufL,
    "BufReadEact": BufReadEact,
    "BufWriteEact": BufWriteEact,
    "BufPpass": BufPpass,
    "OffChipIOL": OffChipIOL,
    "OffChipIOPact": OffChipIOPact,
    "Reff_And": Reff["And"],
    "Reff_Or": Reff["Or"],
    "Reff_Add": Reff["Add"],
    "Reff_Sub": Reff["Sub"],
    "Reff_Mult": Reff["Mult"],
    "Reff_FloorDiv": Reff["FloorDiv"],
    "Reff_Mod": Reff["Mod"],
    "Reff_LShift": Reff["LShift"],
    "Reff_RShift": Reff["RShift"],
    "Reff_BitOr": Reff["BitOr"],
    "Reff_BitXor": Reff["BitXor"],
    "Reff_BitAnd": Reff["BitAnd"],
    "Reff_Eq": Reff["Eq"],
    "Reff_NotEq": Reff["NotEq"],
    "Reff_Lt": Reff["Lt"],
    "Reff_LtE": Reff["LtE"],
    "Reff_Gt": Reff["Gt"],
    "Reff_GtE": Reff["GtE"],
    "Reff_USub": Reff["USub"],
    "Reff_UAdd": Reff["UAdd"],
    "Reff_IsNot": Reff["IsNot"],
    "Reff_Not": Reff["Not"],
    "Reff_Invert": Reff["Invert"],
    "Reff_Regs": Reff["Regs"],
    "Ceff_And": Ceff["And"],
    "Ceff_Or": Ceff["Or"],
    "Ceff_Add": Ceff["Add"],
    "Ceff_Sub": Ceff["Sub"],
    "Ceff_Mult": Ceff["Mult"],
    "Ceff_FloorDiv": Ceff["FloorDiv"],
    "Ceff_Mod": Ceff["Mod"],
    "Ceff_LShift": Ceff["LShift"],
    "Ceff_RShift": Ceff["RShift"],
    "Ceff_BitOr": Ceff["BitOr"],
    "Ceff_BitXor": Ceff["BitXor"],
    "Ceff_BitAnd": Ceff["BitAnd"],
    "Ceff_Eq": Ceff["Eq"],
    "Ceff_NotEq": Ceff["NotEq"],
    "Ceff_Lt": Ceff["Lt"],
    "Ceff_LtE": Ceff["LtE"],
    "Ceff_Gt": Ceff["Gt"],
    "Ceff_GtE": Ceff["GtE"],
    "Ceff_USub": Ceff["USub"],
    "Ceff_UAdd": Ceff["UAdd"],
    "Ceff_IsNot": Ceff["IsNot"],
    "Ceff_Not": Ceff["Not"],
    "Ceff_Invert": Ceff["Invert"],
    "Ceff_Regs": Ceff["Regs"],
    
    # Cacti .dat technology parameters
    'C_g_ideal': C_g_ideal,
    'C_fringe': C_fringe,
    'C_junc': C_junc,
    'C_junc_sw': C_junc_sw,
    'l_phy': l_phy,
    'l_elec': l_elec,
    'nmos_effective_resistance_multiplier': nmos_effective_resistance_multiplier,
    'Vdd': Vdd,
    'Vth': Vth,
    'Vdsat': Vdsat,
    'I_on_n': I_on_n,
    'I_on_p': I_on_p,
    'I_off_n': I_off_n,
    'I_g_on_n': I_g_on_n,
    'C_ox': C_ox,
    't_ox': t_ox,
    'n2p_drv_rt': n2p_drv_rt,
    'lch_lk_rdc': lch_lk_rdc,
    'Mobility_n': Mobility_n,
    'gmp_to_gmn_multiplier': gmp_to_gmn_multiplier,
    'vpp': vpp,
    'Wmemcella': Wmemcella,
    'Wmemcellpmos': Wmemcellpmos,
    'Wmemcellnmos': Wmemcellnmos,
    'area_cell': area_cell,
    'asp_ratio_cell': asp_ratio_cell,
    'vdd_cell': vdd_cell,
    'dram_cell_I_on': dram_cell_I_on,
    'dram_cell_Vdd': dram_cell_Vdd,
    'dram_cell_C': dram_cell_C,
    'dram_cell_I_off_worst_case_len_temp': dram_cell_I_off_worst_case_len_temp,
    'logic_scaling_co_eff': logic_scaling_co_eff,
    'core_tx_density': core_tx_density,
    'sckt_co_eff': sckt_co_eff,
    'chip_layout_overhead': chip_layout_overhead,
    'macro_layout_overhead': macro_layout_overhead,
    'sense_delay': sense_delay,
    'sense_dy_power': sense_dy_power,
    'wire_pitch': wire_pitch,
    'barrier_thickness': barrier_thickness,
    'dishing_thickness': dishing_thickness,
    'alpha_scatter': alpha_scatter,
    'aspect_ratio': aspect_ratio,
    'miller_value': miller_value,
    'horiz_dielectric_constant': horiz_dielectric_constant,
    'vert_dielectric_constant': vert_dielectric_constant,
    'ild_thickness': ild_thickness,
    'fringe_cap': fringe_cap,
    'resistivity': resistivity,
    'wire_r_per_micron': wire_r_per_micron,
    'wire_c_per_micron': wire_c_per_micron,
    'tsv_pitch': tsv_pitch,
    'tsv_diameter': tsv_diameter,
    'tsv_length': tsv_length,
    'tsv_dielec_thickness': tsv_dielec_thickness,
    'tsv_contact_resistance': tsv_contact_resistance,
    'tsv_depletion_width': tsv_depletion_width,
    'tsv_liner_dielectric_cons': tsv_liner_dielectric_cons,

    # Cacti IO technology parameters
    'vdd_io': vdd_io,
    'v_sw_clk': v_sw_clk,
    'c_int': c_int,
    'c_tx': c_tx,
    'c_data': c_data,
    'c_addr': c_addr,
    'i_bias': i_bias,
    'i_leak': i_leak,
    'ioarea_c': ioarea_c,
    'ioarea_k0': ioarea_k0,
    'ioarea_k1': ioarea_k1,
    'ioarea_k2': ioarea_k2,
    'ioarea_k3': ioarea_k3,
    't_ds': t_ds,
    't_is': t_is,
    't_dh': t_dh,
    't_ih': t_ih,
    't_dcd_soc': t_dcd_soc,
    't_dcd_dram': t_dcd_dram,
    't_error_soc': t_error_soc,
    't_skew_setup': t_skew_setup,
    't_skew_hold': t_skew_hold,
    't_dqsq': t_dqsq,
    't_soc_setup': t_soc_setup,
    't_soc_hold': t_soc_hold,
    't_jitter_setup': t_jitter_setup,
    't_jitter_hold': t_jitter_hold,
    't_jitter_addr_setup': t_jitter_addr_setup,
    't_jitter_addr_hold': t_jitter_addr_hold,
    't_cor_margin': t_cor_margin,
    'r_diff_term': r_diff_term,
    'rtt1_dq_read': rtt1_dq_read,
    'rtt2_dq_read': rtt2_dq_read,
    'rtt1_dq_write': rtt1_dq_write,
    'rtt2_dq_write': rtt2_dq_write,
    'rtt_ca': rtt_ca,
    'rs1_dq': rs1_dq,
    'rs2_dq': rs2_dq,
    'r_stub_ca': r_stub_ca,
    'r_on': r_on,
    'r_on_ca': r_on_ca,
    'z0': z0,
    't_flight': t_flight,
    't_flight_ca': t_flight_ca,
    'k_noise_write': k_noise_write,
    'k_noise_read': k_noise_read,
    'k_noise_addr': k_noise_addr,
    'v_noise_independent_write': v_noise_independent_write,
    'v_noise_independent_read': v_noise_independent_read,
    'v_noise_independent_addr': v_noise_independent_addr,
    'phy_datapath_s': phy_datapath_s,
    'phy_phase_rotator_s': phy_phase_rotator_s,
    'phy_clock_tree_s': phy_clock_tree_s,
    'phy_rx_s': phy_rx_s,
    'phy_dcc_s': phy_dcc_s,
    'phy_deskew_s': phy_deskew_s,
    'phy_leveling_s': phy_leveling_s,
    'phy_pll_s': phy_pll_s,
    'phy_datapath_d': phy_datapath_d,
    'phy_phase_rotator_d': phy_phase_rotator_d,
    'phy_clock_tree_d': phy_clock_tree_d,
    'phy_rx_d': phy_rx_d,
    'phy_dcc_d': phy_dcc_d,
    'phy_deskew_d': phy_deskew_d,
    'phy_leveling_d': phy_leveling_d,
    'phy_pll_d': phy_pll_d,
    'phy_pll_wtime': phy_pll_wtime,
    'phy_phase_rotator_wtime': phy_phase_rotator_wtime,
    'phy_rx_wtime': phy_rx_wtime,
    'phy_bandgap_wtime': phy_bandgap_wtime,
    'phy_deskew_wtime': phy_deskew_wtime,
    'phy_vrefgen_wtime': phy_vrefgen_wtime,
}


# passive power
beta = yaml.load(
    open(os.path.join(os.path.dirname(__file__), "params/coefficients.yaml"), "r"),
    Loader=yaml.Loader,
)["beta"]

def make_sym_lat_wc(elem):
    return Reff[elem] * Ceff[elem]

def make_buf_lat_dict():
    return BufL

def make_mem_lat_dict():
    d = {}
    for mem in MemReadL:
        d[mem] = (MemReadL[mem] + MemWriteL[mem]) / 2
    return d

def make_io_lat_dict():
    return OffChipIOL

# UNITS: ns
symbolic_latency_wc = {
    "And": lambda: make_sym_lat_wc("And"),
    "Or": lambda: make_sym_lat_wc("Or"),
    "Add": lambda: make_sym_lat_wc("Add"),
    "Sub": lambda: make_sym_lat_wc("Sub"),
    "Mult": lambda: make_sym_lat_wc("Mult"),
    "FloorDiv": lambda: make_sym_lat_wc("FloorDiv"),
    "Mod": lambda: make_sym_lat_wc("Mod"),
    "LShift": lambda: make_sym_lat_wc("LShift"),
    "RShift": lambda: make_sym_lat_wc("RShift"),
    "BitOr": lambda: make_sym_lat_wc("BitOr"),
    "BitXor": lambda: make_sym_lat_wc("BitXor"),
    "BitAnd": lambda: make_sym_lat_wc("BitAnd"),
    "Eq": lambda: make_sym_lat_wc("Eq"),
    "NotEq": lambda: make_sym_lat_wc("NotEq"),
    "Lt": lambda: make_sym_lat_wc("Lt"),
    "LtE": lambda: make_sym_lat_wc("LtE"),
    "Gt": lambda: make_sym_lat_wc("Gt"),
    "GtE": lambda: make_sym_lat_wc("GtE"),
    "USub": lambda: make_sym_lat_wc("USub"),
    "UAdd": lambda: make_sym_lat_wc("UAdd"),
    "IsNot": lambda: make_sym_lat_wc("IsNot"),
    "Not": lambda: make_sym_lat_wc("Not"),
    "Invert": lambda: make_sym_lat_wc("Invert"),
    "Regs": lambda: make_sym_lat_wc("Regs"),
    "Buf": lambda: make_buf_lat_dict(),
    "MainMem": lambda: make_mem_lat_dict(), # this needs to change later to sep the two.
    "OffChipIO": lambda: make_io_lat_dict(),
}

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

def make_sym_power_act(elem):
    return 0.5 * V_dd * V_dd  / Reff[elem] # C dependence will reappear explicitly when we go back to Energy from power.

def make_buf_power_active_dict():
    d = {}
    print("making buf power active")
    for mem in BufReadEact:
        d[mem] = ((BufReadEact[mem] + BufWriteEact[mem]) / 2) / symbolic_latency_wc["Buf"]()[mem]
        print("adding to buf power active")
    return d

def make_mainmem_power_active_dict():
    d = {}
    for mem in MemWriteEact:
        d[mem] = ((MemWriteEact[mem] + MemReadEact[mem]) / 2) / symbolic_latency_wc["MainMem"]()[mem]
    return d

def make_io_power_active_dict():
    return OffChipIOPact

# UNITS: W
symbolic_power_active = {
    "And": lambda: make_sym_power_act("And"),
    "Or": lambda: make_sym_power_act("Or"),
    "Add": lambda: make_sym_power_act("Add"),
    "Sub": lambda: make_sym_power_act("Sub"),
    "Mult": lambda: make_sym_power_act("Mult"),
    "FloorDiv": lambda: make_sym_power_act("FloorDiv"),
    "Mod": lambda: make_sym_power_act("Mod"),
    "LShift": lambda: make_sym_power_act("LShift"),
    "RShift": lambda: make_sym_power_act("RShift"),
    "BitOr": lambda: make_sym_power_act("BitOr"),
    "BitXor": lambda: make_sym_power_act("BitXor"),
    "BitAnd": lambda: make_sym_power_act("BitAnd"),
    "Eq": lambda: make_sym_power_act("Eq"),
    "NotEq": lambda: make_sym_power_act("NotEq"),
    "Lt": lambda: make_sym_power_act("Lt"),
    "LtE": lambda: make_sym_power_act("LtE"),
    "Gt": lambda: make_sym_power_act("Gt"),
    "GtE": lambda: make_sym_power_act("GtE"),
    "USub": lambda: make_sym_power_act("USub"),
    "UAdd": lambda: make_sym_power_act("UAdd"),
    "IsNot": lambda: make_sym_power_act("IsNot"),
    "Not": lambda: make_sym_power_act("Not"),
    "Invert": lambda: make_sym_power_act("Invert"),
    "Regs": lambda: make_sym_power_act("Regs"),
    "Buf": lambda: make_buf_power_active_dict(),
    "MainMem": lambda: make_mainmem_power_active_dict(),
    "OffChipIO": lambda: make_io_power_active_dict(),
}

def make_sym_power_pass(beta, P_pass_inv=V_dd**2 / (Reff["Not"] * 100)):
    return beta * P_pass_inv

def make_mainmem_power_passive_dict():
    return MemPpass

def make_buf_power_passive_dict():
    return BufPpass

# UNITS: W
symbolic_power_passive = {
    "And": lambda: make_sym_power_pass(beta["And"]),
    "Or": lambda: make_sym_power_pass(beta["Or"]),
    "Add": lambda: make_sym_power_pass(beta["Add"]),
    "Sub": lambda: make_sym_power_pass(beta["Sub"]),
    "Mult": lambda: make_sym_power_pass(beta["Mult"]),
    "FloorDiv": lambda: make_sym_power_pass(beta["FloorDiv"]),
    "Mod": lambda: make_sym_power_pass(beta["Mod"]),
    "LShift": lambda: make_sym_power_pass(beta["LShift"]),
    "RShift": lambda: make_sym_power_pass(beta["RShift"]),
    "BitOr": lambda: make_sym_power_pass(beta["BitOr"]),
    "BitXor": lambda: make_sym_power_pass(beta["BitXor"]),
    "BitAnd": lambda: make_sym_power_pass(beta["BitAnd"]),
    "Eq": lambda: make_sym_power_pass(beta["Eq"]),
    "NotEq": lambda: make_sym_power_pass(beta["NotEq"]),
    "Lt": lambda: make_sym_power_pass(beta["Lt"]),
    "LtE": lambda: make_sym_power_pass(beta["LtE"]),
    "Gt": lambda: make_sym_power_pass(beta["Gt"]),
    "GtE": lambda: make_sym_power_pass(beta["GtE"]),
    "USub": lambda: make_sym_power_pass(beta["USub"]),
    "UAdd": lambda: make_sym_power_pass(beta["UAdd"]),
    "IsNot": lambda: make_sym_power_pass(beta["IsNot"]),
    "Not": lambda: make_sym_power_pass(beta["Not"]),
    "Invert": lambda: make_sym_power_pass(beta["Invert"]),
    "Regs": lambda: make_sym_power_pass(beta["Regs"]),
    "MainMem": lambda: make_mainmem_power_passive_dict(),
    "Buf": lambda: make_buf_power_passive_dict(),
}

def update_symbolic_passive_power(R_off_on_ratio):
    for elem in symbolic_power_passive:
        if elem != "MainMem" and elem != "Buf":
            symbolic_power_passive[elem] = make_sym_power_pass(beta[elem], V_dd**2 / (R_off_on_ratio * Reff["Not"]))

cacti_tech_params = [
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

cacti_io_tech_params = [
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
