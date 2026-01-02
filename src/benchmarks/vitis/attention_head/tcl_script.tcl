open_project attention_head
set_top attention_head
add_files attention_head.cpp
open_solution "solution1" -flow_target vitis
set_part {xcvu11p-flga2577-1-e}
create_clock -period 10 -name default
csynth_design
#export_design ## we do not need the vivado IP block
exit