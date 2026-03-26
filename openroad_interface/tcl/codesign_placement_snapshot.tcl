# Placement-time layout image: macros, instance pins (incl. I/O), and logical
# flywires between blocks. Written after detailed placement / before global
# routing so it remains useful when global_route fails (no routed shapes yet).

proc codesign_write_placement_snapshot {} {
  if { ![info exists design] } {
    puts stderr "codesign_write_placement_snapshot: design not set; skipping."
    return
  }

  set snapshot_path [make_result_file design_placement_snapshot.png]

  set save_area ""
  if { [info exists die_area] } {
    set save_area $die_area
  } elseif { [info exists core_area] } {
    set save_area $core_area
  }

  # Match post-route snapshot sharpness; OpenROAD clamps to ~7200 px.
  set save_width_px 4000

  # Grids off. Routed shapes off (usually empty / misleading pre-global-route).
  # Pins + pin names show macro/port locations. Flywires show logical connectivity.
  if { [catch {
    if { $save_area ne "" } {
      save_image $snapshot_path -area $save_area -width $save_width_px \
        -display_option [list "Tracks/Pref" false] \
        -display_option [list "Tracks/Non Pref" false] \
        -display_option [list "Misc/Manufacturing grid" false] \
        -display_option [list "Misc/GCell grid" false] \
        -display_option [list "Shape Types/Routing/Segments" false] \
        -display_option [list "Shape Types/Routing/Vias" false] \
        -display_option [list "Shape Types/Special Routing/Segments" false] \
        -display_option [list "Shape Types/Special Routing/Vias" false] \
        -display_option [list "Shape Types/Pins" true] \
        -display_option [list "Shape Types/Pin Names" true] \
        -display_option [list "Misc/Flywires only" true] \
        -display_option [list "Instances/Macro" true]
    } else {
      save_image $snapshot_path -width $save_width_px \
        -display_option [list "Tracks/Pref" false] \
        -display_option [list "Tracks/Non Pref" false] \
        -display_option [list "Misc/Manufacturing grid" false] \
        -display_option [list "Misc/GCell grid" false] \
        -display_option [list "Shape Types/Routing/Segments" false] \
        -display_option [list "Shape Types/Routing/Vias" false] \
        -display_option [list "Shape Types/Special Routing/Segments" false] \
        -display_option [list "Shape Types/Special Routing/Vias" false] \
        -display_option [list "Shape Types/Pins" true] \
        -display_option [list "Shape Types/Pin Names" true] \
        -display_option [list "Misc/Flywires only" true] \
        -display_option [list "Instances/Macro" true]
    }
  } err] } {
    puts stderr "codesign_write_placement_snapshot: save_image failed: $err"
  } else {
    puts "INFO: wrote placement snapshot (macros / I/O / flywires): $snapshot_path"
  }
}
