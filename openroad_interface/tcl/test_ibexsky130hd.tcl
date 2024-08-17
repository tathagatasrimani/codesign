# ibex sky130hd
source "helpers.tcl"
source "flow_helpers.tcl"
source "sky130hd/sky130hd.vars"

set design "ibex"
set top_module "ibex_core"
set sdc_file "ibex_sky130hd.sdc"
set die_area {0 0 800 800}
set core_area {10 10 790 790}

set slew_margin 30
set cap_margin 25

set global_place_density 0.5

# Assumes flow_helpers.tcl has been read.
read_libraries
read_def results/first_generated.def
read_sdc $sdc_file

source -echo "codesign_flow.tcl"