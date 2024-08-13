# gcd flow pipe cleaner
source "helpers.tcl"
source "flow_helpers.tcl"
source "sky130hs/sky130hs.vars"

set design "gcd"
set top_module "gcd"
set sdc_file "gcd_sky130hs.sdc"
set die_area {0 0 299.96 300.128}
set core_area {9.996 10.08 289.964 290.048}

# Assumes flow_helpers.tcl has been read.
read_libraries
read_def results/first_generated.def
read_sdc $sdc_file

################################################################
# IO Placement (random)
place_pins -random -hor_layers $io_placer_hor_layer -ver_layers $io_placer_ver_layer

################################################################
# Macro Placement
if { [have_macros] } {
  global_placement -density $global_place_density
  macro_placement -halo $macro_place_halo -channel $macro_place_channel
}

################################################################
# Tapcell insertion
eval tapcell $tapcell_args

################################################################
# Power distribution network insertion
source $pdn_cfg
pdngen

################################################################
# Global placement

foreach layer_adjustment $global_routing_layer_adjustments {
  lassign $layer_adjustment layer adjustment
  set_global_routing_layer_adjustment $layer $adjustment
}
set_routing_layers -signal $global_routing_layers \
  -clock $global_routing_clock_layers
set_macro_extension 2

global_placement -routability_driven -density $global_place_density \
  -pad_left $global_place_pad -pad_right $global_place_pad

# set thread count for all tools with support for multithreading.
# set after global placement because it uses omp but generates
# different results when using multiple threads.
set_thread_count [exec getconf _NPROCESSORS_ONLN]

# IO Placement
place_pins -hor_layers $io_placer_hor_layer -ver_layers $io_placer_ver_layer

# checkpoint
set global_place_db [make_result_file ${design}_${platform}_global_place.db]
write_db $global_place_db

################################################################
# Repair max slew/cap/fanout violations and normalize slews
source $layer_rc_file
set_wire_rc -signal -layer $wire_rc_layer
set_wire_rc -clock  -layer $wire_rc_layer_clk
set_dont_use $dont_use

estimate_parasitics -placement

repair_design -slew_margin $slew_margin -cap_margin $cap_margin

repair_tie_fanout -separation $tie_separation $tielo_port
repair_tie_fanout -separation $tie_separation $tiehi_port

set_placement_padding -global -left $detail_place_pad -right $detail_place_pad
detailed_placement

# post resize timing report (ideal clocks)
report_worst_slack -min -digits 3
report_worst_slack -max -digits 3
report_tns -digits 3
# Check slew repair
report_check_types -max_slew -max_capacitance -max_fanout -violators

utl::metric "RSZ::repair_design_buffer_count" [rsz::repair_design_buffer_count]
utl::metric "RSZ::max_slew_slack" [expr [sta::max_slew_check_slack_limit] * 100]
utl::metric "RSZ::max_fanout_slack" [expr [sta::max_fanout_check_slack_limit] * 100]
utl::metric "RSZ::max_capacitance_slack" [expr [sta::max_capacitance_check_slack_limit] * 100]

################################################################
# Clock Tree Synthesis

# Clone clock tree inverters next to register loads
# so cts does not try to buffer the inverted clocks.
repair_clock_inverters

clock_tree_synthesis -root_buf $cts_buffer -buf_list $cts_buffer \
  -sink_clustering_enable \
  -sink_clustering_max_diameter $cts_cluster_diameter

# CTS leaves a long wire from the pad to the clock tree root.
repair_clock_nets

# place clock buffers
detailed_placement

################################################################
# Setup/hold timing repair

set_propagated_clock [all_clocks]

# Global routing is fast enough for the flow regressions.
# It is NOT FAST ENOUGH FOR PRODUCTION USE.
set repair_timing_use_grt_parasitics 0
if { $repair_timing_use_grt_parasitics } {
  # Global route for parasitics - no guide file requied
  global_route -congestion_iterations 100
  estimate_parasitics -global_routing
} else {
  estimate_parasitics -placement
}

repair_timing -skip_gate_cloning

# Post timing repair.
report_worst_slack -min -digits 3
report_worst_slack -max -digits 3
report_tns -digits 3
report_check_types -max_slew -max_capacitance -max_fanout -violators -digits 3

utl::metric "RSZ::worst_slack_min" [sta::worst_slack -min]
utl::metric "RSZ::worst_slack_max" [sta::worst_slack -max]
utl::metric "RSZ::tns_max" [sta::total_negative_slack -max]
utl::metric "RSZ::hold_buffer_count" [rsz::hold_buffer_count]

################################################################
# Detailed Placement

detailed_placement

# Capture utilization before fillers make it 100%
utl::metric "DPL::utilization" [format %.1f [expr [rsz::utilization] * 100]]
utl::metric "DPL::design_area" [sta::format_area [rsz::design_area] 0]

# checkpoint
set dpl_db [make_result_file ${design}_${platform}_dpl.db]
write_db $dpl_db

set verilog_file [make_result_file ${design}_${platform}.v]
write_verilog $verilog_file

################################################################
# Global routing

pin_access -bottom_routing_layer $min_routing_layer \
           -top_routing_layer $max_routing_layer

set route_guide [make_result_file ${design}_${platform}.route_guide]
global_route -guide_file $route_guide \
  -congestion_iterations 100 -verbose

set verilog_file [make_result_file ${design}_${platform}.v]
write_verilog -remove_cells $filler_cells $verilog_file

################################################################
# Antenna repair

repair_antennas -iterations 5

check_antennas
utl::metric "GRT::ANT::errors" [ant::antenna_violation_count]

################################################################
# Filler placement

filler_placement $filler_cells
check_placement -verbose

################################################################
# Detailed routing

# set multithreading only for detailed route due to instability
# in repair_antennas
set_thread_count [exec getconf _NPROCESSORS_ONLN]

# Run pin access again after inserting diodes and moving cells
pin_access -bottom_routing_layer $min_routing_layer \
           -top_routing_layer $max_routing_layer

detailed_route -output_drc [make_result_file "${design}_${platform}_route_drc.rpt"] \
               -output_maze [make_result_file "${design}_${platform}_maze.log"] \
               -no_pin_access \
               -save_guide_updates \
               -bottom_routing_layer $min_routing_layer \
               -top_routing_layer $max_routing_layer \
               -verbose 0

write_guides [make_result_file "${design}_${platform}_output_guide.mod"]
set drv_count [detailed_route_num_drvs]
utl::metric "DRT::drv" $drv_count

check_antennas
utl::metric "DRT::ANT::errors" [ant::antenna_violation_count]


set routed_def [make_result_file final_generated.def]
write_def $routed_def

################################################################
# Extraction

if { $rcx_rules_file != "" } {
  define_process_corner -ext_model_index 0 X
  extract_parasitics -ext_model_file $rcx_rules_file -debug_net_id metal1

  set spef_file [make_result_file generated.spef]
  write_spef $spef_file

  read_spef $spef_file
} else {
  # Use global routing based parasitics inlieu of rc extraction
  estimate_parasitics -global_routing
}

exit
