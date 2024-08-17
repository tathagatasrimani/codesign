# gcd flow pipe cleaner
source "helpers.tcl"
source "flow_helpers.tcl"
source "Nangate45/Nangate45.vars"

set design "gcd"
set top_module "gcd"
set sdc_file "gcd_nangate45.sdc"

set die_area {0 0 100.13 100.8}
set core_area {10.07 11.2 90.25 91}

read_libraries
read_def results/first_generated.def
read_sdc $sdc_file

source -echo "codesign_flow.tcl"