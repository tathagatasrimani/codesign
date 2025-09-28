source "codesign_files/helpers.tcl"
source "codesign_files/codesign_flow_helpers.tcl"
source "codesign_files/codesign.vars"

## uncomment the following line to see all valid openroad commands
## help


set design "codesign"
set top_module "codesign"
set sdc_file "codesign_files/codesign.sdc"
#set image_name "results/codesign.jpeg"

## TODO: these need to scale appropriatley based on the area constraints
## will be updated in a script
set die_area {0 0 2800 2800}
set core_area {50 50 2750 2750}

read_libraries
read_def ../results/first_generated.def
read_sdc $sdc_file

# Place I/O pins on legal edges/tracks (no deprecated -random)
# Pick layers that exist in your tech; metal2/metal3 are examples.
place_pins -hor_layers {metal3} -ver_layers {metal2} -min_distance 1

# Optional: confirm pins exist / are named
puts "Ports: [get_ports *]"

source "codesign_flow.tcl"