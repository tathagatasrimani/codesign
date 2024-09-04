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

source -echo "codesign_flow.tcl"
