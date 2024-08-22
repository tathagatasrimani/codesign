# gcd flow pipe cleaner
source "helpers.tcl"
source "flow_helpers.tcl"
source "Nangate45/Nangate45.vars"

set design "gcd"
set top_module "gcd"
set sdc_file "gcd_nangate45.sdc"
set image_name "results/gcd_nangate45_bigger.jpeg"

set die_area {0 0 610 610}
set core_area {10 10 600 600}

read_libraries
read_def results/first_generated.def
read_sdc $sdc_file

source -echo "codesign_flow.tcl"