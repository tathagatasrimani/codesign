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
# Macro Placement (using rtl_macro_placer)

# Updated Tiered Placement flow

puts "Step 1: Initial Placement of Macros"

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
    -write_macro_placement initial_macro_place.tcl

# Configuring for Custom Placement of Muxes
set mux_prefix "Mux"
set final_file "macro_place.tcl"
set halo_x 10
set halo_y 10

set db [ord::get_db]
set block [[$db getChip] getBlock]
set tech [$db getTech]
set dbu [$tech getDbUnitsPerMicron]
set core_box [$block getCoreArea]

# Helper for microns/dbu conversion
proc mic2db {val dbu} { return [expr {round($val * $dbu)}] }
proc db2mic {val dbu} { return [expr {$val / double($dbu)}] }

# List to track Mux placements to avoid Mux-Mux overlaps
set placed_mux_bboxes {}

# Helper functions

# Check if a candidate box overlaps with any existing Mux boxes (including Halo)
proc check_overlap {candidate_box placed_list halo_x_db halo_y_db} {
    set c_xlo [$candidate_box xMin]
    set c_ylo [$candidate_box yMin]
    set c_xhi [$candidate_box xMax]
    set c_yhi [$candidate_box yMax]

    # Inflate candidate by halo for the check
    set check_box [odb::Rect \
        [expr {$c_xlo - $halo_x_db}] \
        [expr {$c_ylo - $halo_y_db}] \
        [expr {$c_xhi + $halo_x_db}] \
        [expr {$c_yhi + $halo_y_db}] \
    ]

    foreach other_box $placed_list {
        if { [$check_box intersection $other_box] } {
            return 1 ;# Overlap detected
        }
    }
    return 0
}

# Clamp a point (x, y) to be inside a given Box (Region/Die)
proc clamp_to_box {x y box} {
    set b_xlo [$box xMin]
    set b_ylo [$box yMin]
    set b_xhi [$box xMax]
    set b_yhi [$box yMax]

    if {$x < $b_xlo} {set x $b_xlo}
    if {$x > $b_xhi} {set x $b_xhi}
    if {$y < $b_ylo} {set y $b_ylo}
    if {$y > $b_yhi} {set y $b_yhi}
    
    return [list $x $y]
}

# Time to start Mux placement
puts "Step 2: Relocating ${mux_prefix} Macros to allow overlaps with other Macros"

# Open final macro placement file for writing
set fp [open $final_file "w"]

set insts [$block getInsts]

foreach inst $insts {
  set inst_name [$inst getName]

  # If instance is a Mux, attempt to place it with overlap allowance
  if {[string match "${mux_prefix}*" $inst_name]} {

    # Calculate centroid of connectivity
    set sum_x 0; set sum_y 0; set count 0
    set iterms [$inst getITerms]

    foreach iterm $iterms {
      set net [$iterm getNet]
      if {$net == "NULL"} {continue}

      foreach other_iterm [$net getITerms] {
        set other_inst [$other_iterm getInst]

        # Don't consider self
        if {$other_inst == $inst} {continue}

        set loc [$other_inst getLocation]
        set sum_x [expr {$sum_x + [lindex $loc 0]}]
        set sum_y [expr {$sum_y + [lindex $loc 1]}]
        incr count
      }
    }

    # If no connections, use original location
    if {$count == 0} {
      set current_loc [$inst getLocation]
      set target_x [lindex $current_loc 0]
      set target_y [lindex $current_loc 1]
    } else {
      set target_x [expr {$sum_x / $count}]
      set target_y [expr {$sum_y / $count}]
    }

    # Apply fence & region constraints
    set region [$inst getRegion]
    if {$region != "NULL"} {
      # Get the boundaries
      set region_box [lindex [$region getBoundaries] 0]
      set clamped [clamp_to_box $target_x $target_y $region_box]
      set target_x [lindex $clamped 0]
      set target_y [lindex $clamped 1]
    } else {
      # Clamp to core area
      set clamped [clamp_to_box $target_x $target_y $core_box]
      set target_x [lindex $clamped 0]
      set target_y [lindex $clamped 1]
    }

    # Resolve Mux-Mux overlaps with Halo consideration
    set master [$inst getMaster]
    set w [$master getWidth]
    set h [$master getHeight]
    
    set halo_x_db [mic2db $halo_x $dbu]
    set halo_y_db [mic2db $halo_y $dbu]

    set attempt 0
    set max_attempts 10
    set placed_legal 0

    while {$attempt < $max_attempts} {
      set x_end [expr {$target_x + $w}]
      set y_end [expr {$target_y + $h}]
      set candidate_bbox [odb::Rect $target_x $target_y $x_end $y_end]

      if {![check_overlap $candidate_bbox $placed_mux_bboxes $halo_x_db $halo_y_db]} {
        set placed_legal 1
        lappend placed_mux_bboxes $candidate_bbox
        break
      }

      # Simple spiral/shift strategy if overlap found
      set shift_amount [expr {$w + $halo_x_db}]
      set target_x [expr {$target_x + $shift_amount}]
      incr attempt
    }
    
    if {!$placed_legal} {
      puts "Warning: Could not find overlap-free spot for $inst_name. Placing at last attempt."
    }

    # Time to place the Mux
    set final_x_db [db2mic $target_x $dbu]
    set final_y_db [db2mic $target_y $dbu]

    $inst setLocation $target_x $target_y
    $inst setPlacementStatus "PLACED"

    # Write placement command to final macro placement file
    puts $fp "place_macro -macro_name $inst_name -location \"$final_mic_x $final_mic_y\" -allow_overlap"
  } else {
    # Non-Mux macros retain their original placement
    set loc [$inst getLocation]
    set mic_x [db2mic [lindex $loc 0] $dbu]
    set mic_y [db2mic [lindex $loc 1] $dbu]
    puts $fp "place_macro -macro_name $inst_name -location \"$mic_x $mic_y\""
  }
}
close $fp
puts "Macro placement with overlap handling completed. See $final_file for details."

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

set route_guide [make_result_file ${design}_${platform}.route_guide]
global_route -guide_file $route_guide \
  -congestion_iterations 100 -verbose

set verilog_file [make_result_file ${design}_${platform}.v]
write_verilog -remove_cells $filler_cells $verilog_file

report_wire_length -net * -file "../results/wire_length_global.txt" -global_route

set routed_def [make_result_file final_generated.def]
write_def $routed_def

save_image [make_result_file design_snapshot.png] -display_option {Tracks/Pref true} 

exit