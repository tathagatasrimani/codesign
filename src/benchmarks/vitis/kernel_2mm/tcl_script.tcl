open_project kernel_2mm
set_top kernel_2mm
add_files kernel_2mm.cpp
open_solution "solution1" -flow_target vitis
set_part {xcvu11p-flga2577-1-e}
create_clock -period 10 -name default
csynth_design
export_design 
exit