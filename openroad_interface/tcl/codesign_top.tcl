source "codesign_files/helpers.tcl"
source "codesign_files/codesign_flow_helpers.tcl"
source "codesign_files/codesign.vars"

set design "codesign"
set top_module "codesign"
set sdc_file "codesign_files/codesign.sdc"
#set image_name "results/codesign.jpeg"

set die_area {0 0 2000 2000}
set core_area {10 10 1990 1990}

read_libraries
read_def ../results/first_generated.def
read_sdc $sdc_file

source -echo "codesign_flow.tcl"