open_project VJEPA2
set_top VJEPA2
add_files VJEPA2.c
open_solution "solution1" -flow_target vitis
set_partc {xcvu11p-flga2577-1-e}
create_clock -period 10 -name default
csynth_design
export_design 
exit