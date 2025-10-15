open_project attention_project
set_top attn_step
add_files attention.h
add_files attention.cpp
add_files -tb test.cpp
open_solution "solution1"
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
export_design -format ip_catalog
exit

