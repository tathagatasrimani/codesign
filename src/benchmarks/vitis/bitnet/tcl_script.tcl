open_project bitnet
set_top bitnet
add_files bitnet.cpp
open_solution "solution1" -flow_target vitis
set_part {xcvu11p-flga2577-1-e}
create_clock -period 10 -name default
csynth_design
#export_design ## we do not need the vivado IP block
exit