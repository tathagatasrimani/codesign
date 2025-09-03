open_project test_gemm
set_top test_gemm
add_files test_gemm_dse.cpp
open_solution "solution1" -flow_target vitis
set_part {xcvu11p-flga2577-1-e}
create_clock -period 10 -name default
csynth_design
export_design 
exit