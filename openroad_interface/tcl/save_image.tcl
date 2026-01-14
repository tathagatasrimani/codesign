# Script to load a completed OpenROAD design and save an image
# Usage: cd /scratch/patrick/codesign/src/tmp/tmp_gemm_edp_27/pd_1632_dsp && openroad -gui save_image.tcl
# Or: openroad -gui -exit save_image.tcl  (to save and exit automatically)

source "codesign_files/helpers.tcl"
source "codesign_files/codesign_flow_helpers.tcl"
source "codesign_files/codesign.vars"

set design "codesign"

# Read libraries
read_libraries

# Go back to parent directory to read DEF file
cd ..

# Read the final routed DEF file
read_def results/final_generated-tcl.def

draw_route_guides [get_nets] -show_segments

save_image $image_path -display_option {Tracks/Pref true}


