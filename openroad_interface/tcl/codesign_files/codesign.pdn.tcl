# taken from Nangate45.pdn.tcl
####################################
# global connections
####################################
add_global_connection -defer_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDD$} -power
add_global_connection -defer_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDDPE$}
add_global_connection -defer_connection -net {VDD} -inst_pattern {.*} -pin_pattern {^VDDCE$}
add_global_connection -defer_connection -net {VSS} -inst_pattern {.*} -pin_pattern {^VSS$} -ground
add_global_connection -defer_connection -net {VSS} -inst_pattern {.*} -pin_pattern {^VSSE$}
global_connect
####################################
# voltage domains
####################################
set_voltage_domain -name {CORE} -power {VDD} -ground {VSS}
####################################
# standard cell grid
####################################
define_pdn_grid -name {grid} -voltage_domains {CORE}
add_pdn_stripe -grid {grid} -layer {metal1} -width {0.17} -pitch {2.4} -offset {0} -followpins
add_pdn_stripe -grid {grid} -layer {metal4} -width {0.48} -pitch {56.0} -offset {2}
add_pdn_stripe -grid {grid} -layer {metal7} -width {1.40} -pitch {40.0} -offset {2}
add_pdn_connect -grid {grid} -layers {metal1 metal4}
add_pdn_connect -grid {grid} -layers {metal4 metal7}


####################################
# Minimal (dummy) macro PDN grids
####################################

# For macros with no rotation (R0 / R180 / MX / MY)
define_pdn_grid -name CORE_macro_grid_1 \
    -macro \
    -default \
    -voltage_domains {CORE} \
    -orient {R0 R180 MX MY} \
    -halo {0.1 0.1 0.1 0.1} \
    -grid_over_boundary

add_pdn_stripe -grid CORE_macro_grid_1 -layer metal1 \
    -width {0.05} \
    -pitch {1000.0} \
    -offset {0.0} \
    -followpins

# For macros rotated by 90° (R90/R270)
define_pdn_grid -name CORE_macro_grid_2 \
    -macro \
    -default \
    -voltage_domains {CORE} \
    -orient {R90 R270 MXR90 MYR90} \
    -halo {0.1 0.1 0.1 0.1} \
    -grid_over_boundary

add_pdn_stripe -grid CORE_macro_grid_2 -layer metal1 \
    -width {0.05} \
    -pitch {1000.0} \
    -offset {0.0} \
    -followpins
####################################
