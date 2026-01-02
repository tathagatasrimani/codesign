import os
import argparse
import subprocess
import json

CODESIGN_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

SCALEHLS_DIR = os.path.join(os.path.dirname(__file__), "../ScaleHLS-HIDA")

C_MLIR_FOLDER = os.path.join(os.path.dirname(__file__), "c_mlir")
if not os.path.exists(C_MLIR_FOLDER):
    os.makedirs(C_MLIR_FOLDER)

C_TEST_LOG_FOLDER = os.path.join(os.path.dirname(__file__), "c_test_log")
if not os.path.exists(C_TEST_LOG_FOLDER):
    os.makedirs(C_TEST_LOG_FOLDER)

C_INPUT_FOLDER = os.path.join(os.path.dirname(__file__), "c_input")
if not os.path.exists(C_INPUT_FOLDER):
    os.makedirs(C_INPUT_FOLDER)

C_DEBUG_LOG_FOLDER = os.path.join(os.path.dirname(__file__), "c_debug_log")
if not os.path.exists(C_DEBUG_LOG_FOLDER):
    os.makedirs(C_DEBUG_LOG_FOLDER)

SCALEHLS_DESIGN_SPACE_FOLDER = os.path.join(os.path.dirname(__file__), "c_design_space")
if not os.path.exists(SCALEHLS_DESIGN_SPACE_FOLDER):
    os.makedirs(SCALEHLS_DESIGN_SPACE_FOLDER)

CPP_OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "cpp_output")
if not os.path.exists(CPP_OUTPUT_FOLDER):
    os.makedirs(CPP_OUTPUT_FOLDER)

def set_config(num_dsp, sample_sub_funcs, sample_iter_num, no_unroll_top_func):
    with open(f"{SCALEHLS_DIR}/test/Transforms/Directive/config.json", "r") as f:
        config = json.load(f)
    config["dsp"] = num_dsp
    config["bram"] = num_dsp
    config["sample-sub-funcs"] = sample_sub_funcs
    config["sample-iter-num"] = sample_iter_num
    config["no-unroll-top-func"] = no_unroll_top_func
    with open(f"{SCALEHLS_DIR}/test/Transforms/Directive/config.json", "w") as f:
        json.dump(config, f)

def run_c_file(input_file, debug_point, no_dse, num_dsp, sample_sub_funcs, sample_iter_num, no_unroll_top_func):
    set_config(num_dsp, sample_sub_funcs, sample_iter_num, no_unroll_top_func)
    log_index = 0
    debug_point_txt = f" debug-point={debug_point}" if debug_point != 0 else ""
    log_path = f"{C_TEST_LOG_FOLDER}/{input_file}_{log_index}.log" if debug_point == 0 else f"{C_DEBUG_LOG_FOLDER}/{input_file}/{input_file}_debug_{debug_point}_{log_index}.log"
    while os.path.exists(log_path):
        log_index += 1
        log_path = f"{C_TEST_LOG_FOLDER}/{input_file}_{log_index}.log" if debug_point == 0 else f"{C_DEBUG_LOG_FOLDER}/{input_file}/{input_file}_debug_{debug_point}_{log_index}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    pipeline = "scalehls-no-dse-pipeline" if no_dse else "scalehls-dse-pipeline"
    design_space_index = 0
    design_space_path = f"{SCALEHLS_DESIGN_SPACE_FOLDER}/{input_file}_{design_space_index}"
    while os.path.exists(design_space_path):
        design_space_index += 1
        design_space_path = f"{SCALEHLS_DESIGN_SPACE_FOLDER}/{input_file}_{design_space_index}"
    os.makedirs(design_space_path)

    translate_txt = f"scalehls-translate -scalehls-emit-hlscpp -emit-vitis-directives > {CPP_OUTPUT_FOLDER}/{input_file}.cpp" if debug_point == 0 else f"2>&1 tee {C_TEST_LOG_FOLDER}/{input_file}_{log_index}_debug_{debug_point}.log"

    scalehls_cmd = [
        "bash", "-c",
        f'''
        cd {CODESIGN_ROOT_DIR}
        source full_env_start.sh
        cd {SCALEHLS_DIR}
        source scalehls_env.sh
        cd {design_space_path}
        cgeist {C_INPUT_FOLDER}/{input_file}.c -function='*' --O0 -S -memref-fullrank -raise-scf-to-affine -std=c11 -I{SCALEHLS_DIR}/polygeist/tools/cgeist/Test/polybench/utilities -I/usr/include -I/usr/lib/gcc/x86_64-linux-gnu/13/include -I/usr/local/include -resource-dir $(clang -print-resource-dir) > {C_MLIR_FOLDER}/{input_file}.mlir
        scalehls-opt {C_MLIR_FOLDER}/{input_file}.mlir -{pipeline}="top-func={input_file} target-spec={SCALEHLS_DIR}/test/Transforms/Directive/config.json{debug_point_txt}" -debug-only=scalehls | {translate_txt}
        cd {CODESIGN_ROOT_DIR}
        deactivate
        conda deactivate
        '''
    ]

    with open(f"{C_TEST_LOG_FOLDER}/{input_file}_{log_index}.log", "w") as f:
        p = subprocess.Popen(
            scalehls_cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={}  # clean environment
        )
        p.wait()
    if p.returncode != 0:
        print(f"Error: scalehls failed for {input_file}, log index: {log_index}")
        return
    
    print(f"ScaleHLS completed for {input_file}, log index: {log_index}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--debug_point", type=int, default=0)
    parser.add_argument("--no_dse", action="store_true", default=False)
    parser.add_argument("--num_dsp", type=int, default=150)
    parser.add_argument("--sample_sub_funcs", action="store_true", default=False)
    parser.add_argument("--sample_iter_num", type=int, default=100)
    parser.add_argument("--no_unroll_top_func", action="store_true", default=False)
    args = parser.parse_args()

    input_file = args.input_file
    debug_point = args.debug_point
    no_dse = args.no_dse
    num_dsp = args.num_dsp
    sample_sub_funcs = args.sample_sub_funcs    
    sample_iter_num = args.sample_iter_num
    no_unroll_top_func = args.no_unroll_top_func
    run_c_file(input_file, debug_point, no_dse, num_dsp, sample_sub_funcs, sample_iter_num, no_unroll_top_func)