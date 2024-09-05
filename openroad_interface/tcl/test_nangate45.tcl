# gcd flow pipe cleaner
source "helpers.tcl"
source "flow_helpers.tcl"
source "Nangate45/Nangate45.vars"

set design "gcd"
set top_module "gcd"
set sdc_file "gcd_nangate45.sdc"
set image_name "results/gcd_nangate45.jpeg"

set die_area {0 0 60.13 60.8}
set core_area {10.07 11.2 50.25 51}

read_libraries
read_def results/first_generated.def
read_sdc $sdc_file

source -echo "codesign_flow.tcl"