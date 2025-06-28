open_project test_gemm
set_top test_gemm
add_files test_gemm_dse.cpp
open_solution “solution1” -flow_target vitis
set_part {xcvu9p-flga2104-2L-e} 
create_clock -period 10 -name default
csynth_design
export_design
