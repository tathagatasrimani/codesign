open_project sgd_sw
set_top sgd_sw
add_files sgd_sw.cpp
open_solution "solution1" -flow_target vitis
set_part {xcvu11p-flga2577-1-e}
create_clock -period 10 -name default
csynth_design
export_design 
exit