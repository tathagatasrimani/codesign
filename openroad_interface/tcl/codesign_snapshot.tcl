# Layout snapshot for visual inspection (success or partial/failed flow).
# Sourced from codesign_top.tcl after codesign_flow.tcl so it always runs
# when the design is loaded (e.g. after global-route congestion failure).

if { ![info exists design] } {
  puts stderr "codesign_snapshot.tcl: variable design not set; skipping snapshot."
  return
}

if { [info exists ::env(CODESIGN_SNAPSHOT_PATH)] } {
  set snapshot_path $::env(CODESIGN_SNAPSHOT_PATH)
} else {
  set snapshot_path [make_result_file design_snapshot.png]
}

# Region in microns (same convention as codesign_top.tcl die_area / core_area).
set save_area ""
if { [info exists die_area] } {
  set save_area $die_area
} elseif { [info exists core_area] } {
  set save_area $core_area
}

# Render width in pixels (OpenROAD clamps to internal max ~7200).
set save_width_px 4000

# Turn off track / manufacturing grids (overlay). Keep routing shape types and
# nets visible so connectivity (wires/vias) still shows.
# Paths use OpenROAD DisplayControls "path/path" naming (see displayControls.cpp).

if { [catch {
  if { $save_area ne "" } {
    save_image $snapshot_path -area $save_area -width $save_width_px \
      -display_option [list "Tracks/Pref" false] \
      -display_option [list "Tracks/Non Pref" false] \
      -display_option [list "Misc/Manufacturing grid" false] \
      -display_option [list "Misc/GCell grid" false]
  } else {
    save_image $snapshot_path -width $save_width_px \
      -display_option [list "Tracks/Pref" false] \
      -display_option [list "Tracks/Non Pref" false] \
      -display_option [list "Misc/Manufacturing grid" false] \
      -display_option [list "Misc/GCell grid" false]
  }
} err] } {
  puts stderr "codesign_snapshot.tcl: save_image failed: $err"
}
