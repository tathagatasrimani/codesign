import os
import argparse
import subprocess

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

def run_c_file(input_file, debug_point):
    debug_point_txt = f" debug-point={debug_point}" if debug_point != 0 else ""
    log_path = f"{C_TEST_LOG_FOLDER}/{input_file}.log" if debug_point == 0 else f"{C_DEBUG_LOG_FOLDER}/{input_file}/{input_file}_debug_{debug_point}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    design_space_path = f"{SCALEHLS_DESIGN_SPACE_FOLDER}/{input_file}"
    if not os.path.exists(design_space_path):
        os.makedirs(design_space_path)
    scalehls_cmd = [
        "bash", "-c",
        f'''
        cd {CODESIGN_ROOT_DIR}
        source full_env_start.sh
        cd {SCALEHLS_DIR}
        source scalehls_env.sh
        cd {design_space_path}
        cgeist {C_INPUT_FOLDER}/{input_file}.c -function={input_file} -S -memref-fullrank -raise-scf-to-affine -std=c11 -I/scratch_disks/scratch0/patrick_codesign/codesign/ScaleHLS-HIDA/polygeist/tools/cgeist/Test/polybench/utilities -I/usr/include -I/usr/lib/gcc/x86_64-linux-gnu/13/include -I/usr/local/include -resource-dir $(clang -print-resource-dir) > {C_MLIR_FOLDER}/{input_file}.mlir
        scalehls-opt {C_MLIR_FOLDER}/{input_file}.mlir -scalehls-dse-pipeline="top-func={input_file} target-spec={SCALEHLS_DIR}/test/Transforms/Directive/config.json{debug_point_txt}" -debug-only=scalehls 2>&1 | tee {log_path}
        cd {CODESIGN_ROOT_DIR}
        deactivate
        conda deactivate
        '''
    ]

    with open(f"{C_TEST_LOG_FOLDER}/{input_file}.log", "w") as f:
        p = subprocess.Popen(
            scalehls_cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={}  # clean environment
        )
        p.wait()
    if p.returncode != 0:
        print(f"Error: scalehls failed for {input_file}")
        return
    
    print(f"ScaleHLS completed for {input_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--debug_point", type=int, default=0)
    args = parser.parse_args()

    input_file = args.input_file
    debug_point = args.debug_point

    run_c_file(input_file, debug_point)