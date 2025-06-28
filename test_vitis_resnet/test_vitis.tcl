open_project resnet_18
set_top forward
add_files resnet18.cpp
open_solution resnet18_sol -flow_target vitis
set_part {xcvu9p-flga2104-2L-e} 
create_clock -period 10 -name default
csynth_design
export_design
