import argparse
from src.forward_pass.vitis_parse_verbose_rpt import parse_verbose_rpt
from src.forward_pass.vitis_create_netlist import create_vitis_netlist
from src.forward_pass.vitis_create_cdfg import create_cdfg_vitis
from src.forward_pass.vitis_merge_netlists import merge_netlists_vitis
from src.forward_pass import schedule_vitis

def main(benchmark_dir, benchmark_name, just_schedule=False, pytorch=False):
    parse_results_dir = f"{benchmark_dir}/parse_results"
    if pytorch:
        top_level_module_name = "forward"
    else:
        top_level_module_name = benchmark_name

    allowed_functions = {"fmul", "mul", "add", "call"}

    if not just_schedule:
        ## Do preprocessing to the vitis data for the next scripts
        parse_verbose_rpt(f"{benchmark_dir}/{benchmark_name}/solution1/.autopilot/db", parse_results_dir)

        ## Create the netlist
        create_vitis_netlist(parse_results_dir)

        ## Create the CDFGs for each FSM
        create_cdfg_vitis(parse_results_dir)

        merge_netlists_vitis(parse_results_dir, top_level_module_name, allowed_functions)
    
    schedule_parser = schedule_vitis.vitis_schedule_parser(f"{benchmark_dir}", benchmark_name, top_level_module_name, 250, allowed_functions)
    schedule_parser.create_dfgs()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdir", type=str, required=True) # src/tmp_for_test/benchmark
    parser.add_argument("--bname", type=str, required=True) # test_gemm
    parser.add_argument("--sched_only", type=bool, required=False, default=False) # True
    parser.add_argument("--pytorch", type=bool, required=False, default=False) # True
    args = parser.parse_args()
    main(args.bdir, args.bname, args.sched_only, args.pytorch)