# set_debug_level DPL place 1
#set_debug_level MPL multilevel_autoclustering 2
#set_debug_level MPL coarse_shaping 2
#set_debug_level MPL fine_shaping 2
#set_debug_level MPL hierarchical_macro_placement 3
#set_debug_level MPL flipping 1
#set_debug_level MPL boundary_push 1
set_thread_count [expr [cpu_count] - 1]
################################################################
# IO Placement (random)
place_pins -random -hor_layers $io_placer_hor_layer -ver_layers $io_placer_ver_layer

################################################################
# Macro Placement
# global_placement -density $global_place_density



################################################################
# Macro Placement — use hierarchical placer if available, else rtl_macro_placer

set flow_dir [file dirname [info script]]
set hier_placement_tcl [file join $flow_dir "hierarchical_placement.tcl"]

if {[file exists $hier_placement_tcl]} {
  puts "Using hierarchical macro placement"
  source $hier_placement_tcl
} else {
  rtl_macro_placer \
      -target_util 0.25 \
      -target_dead_space 0.05 \
      -min_ar 0.33 \
      -area_weight 0.1 \
      -outline_weight 100.0 \
      -wirelength_weight 100.0 \
      -guidance_weight 10.0 \
      -fence_weight 10.0 \
      -boundary_weight 50.0 \
      -notch_weight 10.0 \
      -macro_blockage_weight 10.0 \
      -halo_width 10 \
      -halo_height 10 \
      -report_directory reports \
      -write_macro_placement macro_place.tcl
}

# Lock macro positions by sourcing the generated macro placement file
#source macro_place.tcl

################################################################
# Tapcell insertion
eval tapcell $tapcell_args

################################################################
# Power distribution network insertion
#source $pdn_cfg
#pdngen

################################################################
# Global placement

foreach layer_adjustment $global_routing_layer_adjustments {
  lassign $layer_adjustment layer adjustment
  set_global_routing_layer_adjustment $layer $adjustment
}
set_routing_layers -signal $global_routing_layers \
  -clock $global_routing_clock_layers
set_macro_extension 2

set ::env(REPLACE_SEED) 42

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

puts "INFO: starting repair design"

# Set debug level for repair_design to see detailed buffer insertion logic
# Level 1-3: higher number = more detailed output
#set_debug_level RSZ repair_design 3
#set_debug_level RSZ repair_net 3
#set_debug_level RSZ early_sizing 3
#set_debug_level RSZ memory 3
#set_debug_level RSZ buffer_under_slew 3
#set_debug_level RSZ resizer 3

repair_design -slew_margin $slew_margin -cap_margin $cap_margin

repair_tie_fanout -separation $tie_separation $tielo_port
repair_tie_fanout -separation $tie_separation $tiehi_port

set_placement_padding -global -left $detail_place_pad -right $detail_place_pad
detailed_placement -max_displacement 500

################################################################
# Clock Tree Synthesis

# Clone clock tree inverters next to register loads
# so cts does not try to buffer the inverted clocks.
#repair_clock_inverters

#clock_tree_synthesis -root_buf $cts_buffer -buf_list $cts_buffer \
#  -sink_clustering_enable \
#  -sink_clustering_max_diameter $cts_cluster_diameter

# CTS leaves a long wire from the pad to the clock tree root.
#repair_clock_nets

# place clock buffers
#detailed_placement

################################################################
# Setup/hold timing repair

#set_propagated_clock [all_clocks]

# Global routing is fast enough for the flow regressions.
# It is NOT FAST ENOUGH FOR PRODUCTION USE.
#set repair_timing_use_grt_parasitics 0
#if { $repair_timing_use_grt_parasitics } {
#  # Global route for parasitics - no guide file requied
#  global_route -congestion_iterations 100
#  estimate_parasitics -global_routing
#} else {
#  estimate_parasitics -placement
#}

#repair_timing -skip_gate_cloning

# Post timing repair.
#report_worst_slack -min -digits 3
#report_worst_slack -max -digits 3
#report_tns -digits 3
#report_check_types -max_slew -max_capacitance -max_fanout -violators -digits 3

#utl::metric "RSZ::worst_slack_min" [sta::worst_slack -min]
#utl::metric "RSZ::worst_slack_max" [sta::worst_slack -max]
#utl::metric "RSZ::tns_max" [sta::total_negative_slack -max]
#utl::metric "RSZ::hold_buffer_count" [rsz::hold_buffer_count]

################################################################
# Detailed Placement

#detailed_placement

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

# Define the routing window once
set_routing_layers \
  -signal ${min_routing_layer}-${max_routing_layer} \
  -clock  $global_routing_clock_layers

# Now pin access uses the above
pin_access

set base_layer_adjustments $global_routing_layer_adjustments
set route_attempt_scales {1.00 0.85 0.72 0.62}
set route_success 0
set route_error ""
set route_guide [make_result_file ${design}_${platform}.route_guide]
set route_guide_attempt ""

set attempt_idx 0
foreach scale $route_attempt_scales {
  incr attempt_idx

  foreach layer_adjustment $base_layer_adjustments {
    lassign $layer_adjustment layer adjustment
    set scaled_adjustment [expr {$adjustment * $scale}]
    if {$scaled_adjustment < 0.15} {
      set scaled_adjustment 0.15
    }
    set_global_routing_layer_adjustment $layer $scaled_adjustment
  }

  set route_guide_attempt [make_result_file ${design}_${platform}.route_guide.a${attempt_idx}]
  puts "Global route attempt $attempt_idx with layer-adjustment scale $scale"

  set rc [catch {
    global_route -guide_file $route_guide_attempt \
      -congestion_iterations 100 -verbose
  } route_result]

  if {$rc == 0} {
    if {$route_guide_attempt ne $route_guide} {
      file copy -force $route_guide_attempt $route_guide
    }
    set route_success 1
    break
  }

  set route_error $route_result
  puts "Global route attempt $attempt_idx failed: $route_error"
}

if {!$route_success} {
  error "Global route failed after [llength $route_attempt_scales] attempts. Last error: $route_error"
}

set verilog_file [make_result_file ${design}_${platform}.v]
write_verilog -remove_cells $filler_cells $verilog_file

report_wire_length -net * -file "../results/wire_length_global.txt" -global_route

set routed_def [make_result_file final_generated.def]
write_def $routed_def

# Use CODESIGN_SNAPSHOT_PATH if set (Python may use /tmp then copy to results)
if { [info exists ::env(CODESIGN_SNAPSHOT_PATH)] } {
  set snapshot_path $::env(CODESIGN_SNAPSHOT_PATH)
} else {
  set snapshot_path [make_result_file design_snapshot.png]
}
# In headless/offscreen runs, the layout widget's "visibleRegion()" can be empty,
# leading to a zero-size QImage and a hard failure in `QImage::save()`.
# Force a deterministic, non-empty render region using die_area/core_area.
set save_area ""
if { [info exists die_area] } {
  set save_area $die_area
} elseif { [info exists core_area] } {
  set save_area $core_area
}

if { $save_area ne "" } {
  # Force a deterministic pixel scale so the rendered QImage isn't zero-sized in headless/offscreen runs.
  # -width is in pixels.
  # Keep within OpenROAD/Qt internal max image size (kMaxImageSize=7200) to avoid clamping/illegal sizes.
  set save_width_px 4000
  save_image $snapshot_path -area $save_area -width $save_width_px -display_option {Tracks/Pref true}
} else {
  save_image $snapshot_path -display_option {Tracks/Pref true}
}

exit