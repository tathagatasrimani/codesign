import os
import argparse
import subprocess

CODESIGN_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

SCALEHLS_DIR = os.path.join(os.path.dirname(__file__), "../ScaleHLS-HIDA")

INPUT_FOLDER = os.path.join(os.path.dirname(__file__), "inputs")
INITIAL_MLIR_FOLDER = os.path.join(os.path.dirname(__file__), "initial_mlir")
SCALEHLS_OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "scalehls_out_file")
SCALEHLS_OUTPUT_LOG_FOLDER = os.path.join(os.path.dirname(__file__), "scalehls_out_log")
TEST_LOG_FOLDER_SCALEHLS = os.path.join(os.path.dirname(__file__), "test_log_scalehls")
TEST_LOG_FOLDER_DEBUG_SCALEHLS = os.path.join(os.path.dirname(__file__), "test_log_debug_scalehls")
CPP_OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "cpp_output")
SCALEHLS_DESIGN_SPACE_FOLDER = os.path.join(os.path.dirname(__file__), "pytorch_design_space")
if not os.path.exists(SCALEHLS_DESIGN_SPACE_FOLDER):
    os.makedirs(SCALEHLS_DESIGN_SPACE_FOLDER)
if not os.path.exists(INITIAL_MLIR_FOLDER):
    os.makedirs(INITIAL_MLIR_FOLDER)
if not os.path.exists(SCALEHLS_OUTPUT_FOLDER):
    os.makedirs(SCALEHLS_OUTPUT_FOLDER)
if not os.path.exists(SCALEHLS_OUTPUT_LOG_FOLDER):
    os.makedirs(SCALEHLS_OUTPUT_LOG_FOLDER)
if not os.path.exists(CPP_OUTPUT_FOLDER):
    os.makedirs(CPP_OUTPUT_FOLDER)
if not os.path.exists(TEST_LOG_FOLDER_SCALEHLS):
    os.makedirs(TEST_LOG_FOLDER_SCALEHLS)
if not os.path.exists(TEST_LOG_FOLDER_DEBUG_SCALEHLS):
    os.makedirs(TEST_LOG_FOLDER_DEBUG_SCALEHLS)


def run_pytorch_file(input_file_name, debug_point):

    log_path = f"{TEST_LOG_FOLDER_SCALEHLS}/{input_file_name}.log" if debug_point == 0 else f"{TEST_LOG_FOLDER_DEBUG_SCALEHLS}/{input_file_name}/{input_file_name}_debug_{debug_point}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    debug_point_txt = f" debug-point={debug_point}" if debug_point != 0 else ""
    log_index = 0
    design_space_path = f"{SCALEHLS_DESIGN_SPACE_FOLDER}/{input_file_name}_{log_index}"
    while os.path.exists(design_space_path):
        log_index += 1
        design_space_path = f"{SCALEHLS_DESIGN_SPACE_FOLDER}/{input_file_name}_{log_index}"
    os.makedirs(design_space_path)
    scalehls_cmd = [
        "bash", "-c",
        f'''
        cd {CODESIGN_ROOT_DIR}
        source full_env_start.sh
        cd {SCALEHLS_DIR}
        source scalehls_env.sh
        cd {design_space_path}
        python {INPUT_FOLDER}/{input_file_name}.py > {INITIAL_MLIR_FOLDER}/{input_file_name}.mlir
        scalehls-opt {INITIAL_MLIR_FOLDER}/{input_file_name}.mlir -hida-pytorch-dse-pipeline="top-func=forward target-spec={SCALEHLS_DIR}/test/Transforms/Directive/config.json{debug_point_txt}" -debug-only=scalehls | scalehls-translate -scalehls-emit-hlscpp -emit-vitis-directives > {CPP_OUTPUT_FOLDER}/{input_file_name}.cpp
        deactivate
        conda deactivate
        '''
    ]

    with open(f"{TEST_LOG_FOLDER_SCALEHLS}/{input_file_name}_{log_index}.log", "w") as f:
        p = subprocess.Popen(
            scalehls_cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={}  # clean environment
        )
        p.wait()
    if p.returncode != 0:
        print(f"Error: scalehls failed for {input_file_name}")
        return

    print(f"scalehls completed for {input_file_name}. Check logs for possible issues")

if __name__ == "__main__":
    """
    This script is used to run the pytorch file through the scalehls pipeline.
    It will then print the success message.

    Example Usage:
    python run_pytorch_file.py bitnet_small --debug_point 2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--debug_point", type=str, default=0)
    args = parser.parse_args()

    input_file = args.input_file
    debug_point = args.debug_point
    run_pytorch_file(input_file, debug_point)