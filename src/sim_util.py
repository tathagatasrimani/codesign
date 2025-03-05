import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from . import hw_symbols

def generate_init_params_from_rcs_as_symbols(rcs):
    """
    Just some format conversion
    keys are strings
    """
    initial_params = {}
    for elem in rcs["Reff"]:
        initial_params[hw_symbols.symbol_table["Reff_" + elem]] = rcs["Reff"][elem]
        initial_params[hw_symbols.symbol_table["Ceff_" + elem]] = rcs["Ceff"][elem]
    initial_params[hw_symbols.V_dd] = rcs["other"]["V_dd"]
    """initial_params[hw_symbols.MemReadEact] = rcs["other"]["MemReadEact"]
    initial_params[hw_symbols.MemWriteEact] = rcs["other"]["MemWriteEact"]
    initial_params[hw_symbols.MemPpass] = rcs["other"]["MemPpass"]
    initial_params[hw_symbols.BufReadEact] = rcs["other"]["BufReadEact"]
    initial_params[hw_symbols.BufWriteEact] = rcs["other"]["BufWriteEact"]
    initial_params[hw_symbols.BufPpass] = rcs["other"]["BufPpass"]
    initial_params[hw_symbols.OffChipIOL] = rcs["other"]["OffChipIOL"]
    initial_params[hw_symbols.OffChipIOPact] = rcs["other"]["OffChipIOPact"]"""

    # CACTI
    initial_params[hw_symbols.C_g_ideal] = rcs["Cacti"]["C_g_ideal"]
    initial_params[hw_symbols.C_fringe] = rcs["Cacti"]["C_fringe"]
    initial_params[hw_symbols.C_junc] = rcs["Cacti"]["C_junc"]
    initial_params[hw_symbols.C_junc_sw] = rcs["Cacti"]["C_junc_sw"]
    initial_params[hw_symbols.l_phy] = rcs["Cacti"]["l_phy"]
    initial_params[hw_symbols.l_elec] = rcs["Cacti"]["l_elec"]
    initial_params[hw_symbols.nmos_effective_resistance_multiplier] = rcs["Cacti"]["nmos_effective_resistance_multiplier"]
    initial_params[hw_symbols.Vdd] = rcs["Cacti"]["Vdd"]
    initial_params[hw_symbols.Vth] = rcs["Cacti"]["Vth"]
    initial_params[hw_symbols.Vdsat] = rcs["Cacti"]["Vdsat"]
    initial_params[hw_symbols.I_on_n] = rcs["Cacti"]["I_on_n"]
    initial_params[hw_symbols.I_on_p] = rcs["Cacti"]["I_on_p"]
    initial_params[hw_symbols.I_off_n] = rcs["Cacti"]["I_off_n"]
    initial_params[hw_symbols.I_g_on_n] = rcs["Cacti"]["I_g_on_n"]
    initial_params[hw_symbols.C_ox] = rcs["Cacti"]["C_ox"]
    initial_params[hw_symbols.t_ox] = rcs["Cacti"]["t_ox"]
    initial_params[hw_symbols.n2p_drv_rt] = rcs["Cacti"]["n2p_drv_rt"]
    initial_params[hw_symbols.lch_lk_rdc] = rcs["Cacti"]["lch_lk_rdc"]
    initial_params[hw_symbols.Mobility_n] = rcs["Cacti"]["Mobility_n"]
    initial_params[hw_symbols.gmp_to_gmn_multiplier] = rcs["Cacti"]["gmp_to_gmn_multiplier"]
    initial_params[hw_symbols.vpp] = rcs["Cacti"]["vpp"]
    initial_params[hw_symbols.Wmemcella] = rcs["Cacti"]["Wmemcella"]
    initial_params[hw_symbols.Wmemcellpmos] = rcs["Cacti"]["Wmemcellpmos"]
    initial_params[hw_symbols.Wmemcellnmos] = rcs["Cacti"]["Wmemcellnmos"]
    initial_params[hw_symbols.area_cell] = rcs["Cacti"]["area_cell"]
    initial_params[hw_symbols.asp_ratio_cell] = rcs["Cacti"]["asp_ratio_cell"]
    # initial_params[hw_symbols.vdd_cell] = rcs["Cacti"]["vdd_cell"]    # TODO check use of vdd_cell
    initial_params[hw_symbols.dram_cell_I_on] = rcs["Cacti"]["dram_cell_I_on"]
    initial_params[hw_symbols.dram_cell_Vdd] = rcs["Cacti"]["dram_cell_Vdd"]
    initial_params[hw_symbols.dram_cell_C] = rcs["Cacti"]["dram_cell_C"]
    initial_params[hw_symbols.dram_cell_I_off_worst_case_len_temp] = rcs["Cacti"]["dram_cell_I_off_worst_case_len_temp"]
    initial_params[hw_symbols.logic_scaling_co_eff] = rcs["Cacti"]["logic_scaling_co_eff"]
    initial_params[hw_symbols.core_tx_density] = rcs["Cacti"]["core_tx_density"]
    initial_params[hw_symbols.sckt_co_eff] = rcs["Cacti"]["sckt_co_eff"]
    initial_params[hw_symbols.chip_layout_overhead] = rcs["Cacti"]["chip_layout_overhead"]
    initial_params[hw_symbols.macro_layout_overhead] = rcs["Cacti"]["macro_layout_overhead"]
    initial_params[hw_symbols.sense_delay] = rcs["Cacti"]["sense_delay"]
    initial_params[hw_symbols.sense_dy_power] = rcs["Cacti"]["sense_dy_power"]
    initial_params[hw_symbols.wire_pitch] = rcs["Cacti"]["wire_pitch"]
    initial_params[hw_symbols.barrier_thickness] = rcs["Cacti"]["barrier_thickness"]
    initial_params[hw_symbols.dishing_thickness] = rcs["Cacti"]["dishing_thickness"]
    initial_params[hw_symbols.alpha_scatter] = rcs["Cacti"]["alpha_scatter"]
    initial_params[hw_symbols.aspect_ratio] = rcs["Cacti"]["aspect_ratio"]
    initial_params[hw_symbols.miller_value] = rcs["Cacti"]["miller_value"]
    initial_params[hw_symbols.horiz_dielectric_constant] = rcs["Cacti"]["horiz_dielectric_constant"]
    initial_params[hw_symbols.vert_dielectric_constant] = rcs["Cacti"]["vert_dielectric_constant"]
    initial_params[hw_symbols.ild_thickness] = rcs["Cacti"]["ild_thickness"]
    initial_params[hw_symbols.fringe_cap] = rcs["Cacti"]["fringe_cap"]
    initial_params[hw_symbols.resistivity] = rcs["Cacti"]["resistivity"]
    initial_params[hw_symbols.wire_r_per_micron] = rcs["Cacti"]["wire_r_per_micron"]
    initial_params[hw_symbols.wire_c_per_micron] = rcs["Cacti"]["wire_c_per_micron"]
    initial_params[hw_symbols.tsv_pitch] = rcs["Cacti"]["tsv_pitch"]
    initial_params[hw_symbols.tsv_diameter] = rcs["Cacti"]["tsv_diameter"]
    initial_params[hw_symbols.tsv_length] = rcs["Cacti"]["tsv_length"]
    initial_params[hw_symbols.tsv_dielec_thickness] = rcs["Cacti"]["tsv_dielec_thickness"]
    initial_params[hw_symbols.tsv_contact_resistance] = rcs["Cacti"]["tsv_contact_resistance"]
    initial_params[hw_symbols.tsv_depletion_width] = rcs["Cacti"]["tsv_depletion_width"]
    initial_params[hw_symbols.tsv_liner_dielectric_cons] = rcs["Cacti"]["tsv_liner_dielectric_cons"]

    # CACTI IO
    initial_params[hw_symbols.vdd_io] = rcs["Cacti_IO"]["vdd_io"]
    initial_params[hw_symbols.v_sw_clk] = rcs["Cacti_IO"]["v_sw_clk"]
    initial_params[hw_symbols.c_int] = rcs["Cacti_IO"]["c_int"]
    initial_params[hw_symbols.c_tx] = rcs["Cacti_IO"]["c_tx"]
    initial_params[hw_symbols.c_data] = rcs["Cacti_IO"]["c_data"]
    initial_params[hw_symbols.c_addr] = rcs["Cacti_IO"]["c_addr"]
    initial_params[hw_symbols.i_bias] = rcs["Cacti_IO"]["i_bias"]
    initial_params[hw_symbols.i_leak] = rcs["Cacti_IO"]["i_leak"]
    initial_params[hw_symbols.ioarea_c] = rcs["Cacti_IO"]["ioarea_c"]
    initial_params[hw_symbols.ioarea_k0] = rcs["Cacti_IO"]["ioarea_k0"]
    initial_params[hw_symbols.ioarea_k1] = rcs["Cacti_IO"]["ioarea_k1"]
    initial_params[hw_symbols.ioarea_k2] = rcs["Cacti_IO"]["ioarea_k2"]
    initial_params[hw_symbols.ioarea_k3] = rcs["Cacti_IO"]["ioarea_k3"]
    initial_params[hw_symbols.t_ds] = rcs["Cacti_IO"]["t_ds"]
    initial_params[hw_symbols.t_is] = rcs["Cacti_IO"]["t_is"]
    initial_params[hw_symbols.t_dh] = rcs["Cacti_IO"]["t_dh"]
    initial_params[hw_symbols.t_ih] = rcs["Cacti_IO"]["t_ih"]
    initial_params[hw_symbols.t_dcd_soc] = rcs["Cacti_IO"]["t_dcd_soc"]
    initial_params[hw_symbols.t_dcd_dram] = rcs["Cacti_IO"]["t_dcd_dram"]
    initial_params[hw_symbols.t_error_soc] = rcs["Cacti_IO"]["t_error_soc"]
    initial_params[hw_symbols.t_skew_setup] = rcs["Cacti_IO"]["t_skew_setup"]
    initial_params[hw_symbols.t_skew_hold] = rcs["Cacti_IO"]["t_skew_hold"]
    initial_params[hw_symbols.t_dqsq] = rcs["Cacti_IO"]["t_dqsq"]
    initial_params[hw_symbols.t_soc_setup] = rcs["Cacti_IO"]["t_soc_setup"]
    initial_params[hw_symbols.t_soc_hold] = rcs["Cacti_IO"]["t_soc_hold"]
    initial_params[hw_symbols.t_jitter_setup] = rcs["Cacti_IO"]["t_jitter_setup"]
    initial_params[hw_symbols.t_jitter_hold] = rcs["Cacti_IO"]["t_jitter_hold"]
    initial_params[hw_symbols.t_jitter_addr_setup] = rcs["Cacti_IO"]["t_jitter_addr_setup"]
    initial_params[hw_symbols.t_jitter_addr_hold] = rcs["Cacti_IO"]["t_jitter_addr_hold"]
    initial_params[hw_symbols.t_cor_margin] = rcs["Cacti_IO"]["t_cor_margin"]
    initial_params[hw_symbols.r_diff_term] = rcs["Cacti_IO"]["r_diff_term"]
    initial_params[hw_symbols.rtt1_dq_read] = rcs["Cacti_IO"]["rtt1_dq_read"]
    initial_params[hw_symbols.rtt2_dq_read] = rcs["Cacti_IO"]["rtt2_dq_read"]
    initial_params[hw_symbols.rtt1_dq_write] = rcs["Cacti_IO"]["rtt1_dq_write"]
    initial_params[hw_symbols.rtt2_dq_write] = rcs["Cacti_IO"]["rtt2_dq_write"]
    initial_params[hw_symbols.rtt_ca] = rcs["Cacti_IO"]["rtt_ca"]
    initial_params[hw_symbols.rs1_dq] = rcs["Cacti_IO"]["rs1_dq"]
    initial_params[hw_symbols.rs2_dq] = rcs["Cacti_IO"]["rs2_dq"]
    initial_params[hw_symbols.r_stub_ca] = rcs["Cacti_IO"]["r_stub_ca"]
    initial_params[hw_symbols.r_on] = rcs["Cacti_IO"]["r_on"]
    initial_params[hw_symbols.r_on_ca] = rcs["Cacti_IO"]["r_on_ca"]
    initial_params[hw_symbols.z0] = rcs["Cacti_IO"]["z0"]
    initial_params[hw_symbols.t_flight] = rcs["Cacti_IO"]["t_flight"]
    initial_params[hw_symbols.t_flight_ca] = rcs["Cacti_IO"]["t_flight_ca"]
    initial_params[hw_symbols.k_noise_write] = rcs["Cacti_IO"]["k_noise_write"]
    initial_params[hw_symbols.k_noise_read] = rcs["Cacti_IO"]["k_noise_read"]
    initial_params[hw_symbols.k_noise_addr] = rcs["Cacti_IO"]["k_noise_addr"]
    initial_params[hw_symbols.v_noise_independent_write] = rcs["Cacti_IO"]["v_noise_independent_write"]
    initial_params[hw_symbols.v_noise_independent_read] = rcs["Cacti_IO"]["v_noise_independent_read"]
    initial_params[hw_symbols.v_noise_independent_addr] = rcs["Cacti_IO"]["v_noise_independent_addr"]
    initial_params[hw_symbols.phy_datapath_s] = rcs["Cacti_IO"]["phy_datapath_s"]
    initial_params[hw_symbols.phy_phase_rotator_s] = rcs["Cacti_IO"]["phy_phase_rotator_s"]
    initial_params[hw_symbols.phy_clock_tree_s] = rcs["Cacti_IO"]["phy_clock_tree_s"]
    initial_params[hw_symbols.phy_rx_s] = rcs["Cacti_IO"]["phy_rx_s"]
    initial_params[hw_symbols.phy_dcc_s] = rcs["Cacti_IO"]["phy_dcc_s"]
    initial_params[hw_symbols.phy_deskew_s] = rcs["Cacti_IO"]["phy_deskew_s"]
    initial_params[hw_symbols.phy_leveling_s] = rcs["Cacti_IO"]["phy_leveling_s"]
    initial_params[hw_symbols.phy_pll_s] = rcs["Cacti_IO"]["phy_pll_s"]
    initial_params[hw_symbols.phy_datapath_d] = rcs["Cacti_IO"]["phy_datapath_d"]
    initial_params[hw_symbols.phy_phase_rotator_d] = rcs["Cacti_IO"]["phy_phase_rotator_d"]
    initial_params[hw_symbols.phy_clock_tree_d] = rcs["Cacti_IO"]["phy_clock_tree_d"]
    initial_params[hw_symbols.phy_rx_d] = rcs["Cacti_IO"]["phy_rx_d"]
    initial_params[hw_symbols.phy_dcc_d] = rcs["Cacti_IO"]["phy_dcc_d"]
    initial_params[hw_symbols.phy_deskew_d] = rcs["Cacti_IO"]["phy_deskew_d"]
    initial_params[hw_symbols.phy_leveling_d] = rcs["Cacti_IO"]["phy_leveling_d"]
    initial_params[hw_symbols.phy_pll_d] = rcs["Cacti_IO"]["phy_pll_d"]
    initial_params[hw_symbols.phy_pll_wtime] = rcs["Cacti_IO"]["phy_pll_wtime"]
    initial_params[hw_symbols.phy_phase_rotator_wtime] = rcs["Cacti_IO"]["phy_phase_rotator_wtime"]
    initial_params[hw_symbols.phy_rx_wtime] = rcs["Cacti_IO"]["phy_rx_wtime"]
    initial_params[hw_symbols.phy_bandgap_wtime] = rcs["Cacti_IO"]["phy_bandgap_wtime"]
    initial_params[hw_symbols.phy_deskew_wtime] = rcs["Cacti_IO"]["phy_deskew_wtime"]
    initial_params[hw_symbols.phy_vrefgen_wtime] = rcs["Cacti_IO"]["phy_vrefgen_wtime"]

    return initial_params

def change_clk_period_in_script(filename, new_period):
    new_lines = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            new_line= line
            if line.find("set clk_period") != -1:
                new_line = line.replace(line.split()[-1], str(new_period))
            new_lines.append(new_line)
    with open(filename, "w") as f:
        f.writelines(new_lines)

def topological_layout_plot(graph, reverse=False, extra_edges=None):
    # Compute the topological order of the nodes
    if nx.is_directed_acyclic_graph(graph):
        topological_order = list(nx.topological_sort(graph))
    else:
        cycle = nx.find_cycle(graph)
        raise ValueError(f"Graph is not a Directed Acyclic Graph (DAG), topological sorting is not possible. Cycle is {cycle}")
    
    # Group nodes by level in topological order
    levels = defaultdict(int)
    in_degrees = {node: graph.in_degree(node) for node in graph.nodes()}
    
    for node in topological_order:
        level = 0 if in_degrees[node] == 0 else max(levels[parent] + 1 for parent in graph.predecessors(node))
        levels[node] = level
    
    # Arrange nodes in horizontal groups based on level
    level_nodes = defaultdict(list)
    for node, level in levels.items():
        level_nodes[level].append(node)
    
    # Assign positions: group nodes by levels from top to bottom
    pos = {}
    for level, nodes in level_nodes.items():
        x_positions = np.linspace(-len(nodes)/2, len(nodes)/2, num=len(nodes))
        for x, node in zip(x_positions, nodes):
            pos[node] = (x, -level)

    if extra_edges:
        edge_colors = ['red' if (u, v) in extra_edges else 'gray' for (u, v) in graph.edges()]
    else:
        edge_colors = ['gray' for (u, v) in graph.edges()]
    
    # Draw the graph with curved edges to avoid overlap
    plt.figure(figsize=(10, 6))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color=edge_colors, node_size=700, font_size=10, connectionstyle="arc3,rad=0.2")
    
    # Draw dashed lines between topological levels
    max_level = max(level_nodes.keys())
    for level in range(max_level):
        plt.axhline(y=-(level + 0.5), color='gray', linestyle='dashed', linewidth=0.5)

    # Show the graph
    plt.show()
