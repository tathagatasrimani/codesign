import os
import argparse
import subprocess

CODESIGN_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

SCALEHLS_DIR = os.path.join(os.path.dirname(__file__), "../ScaleHLS-HIDA")
CODESIGN_OPT_DIR = os.path.join(os.path.dirname(__file__), "../codesign-opt")

INPUT_FOLDER = os.path.join(os.path.dirname(__file__), "inputs")
INITIAL_MLIR_FOLDER = os.path.join(os.path.dirname(__file__), "initial_mlir")
AFTER_CODESIGN_OPT_FOLDER = os.path.join(os.path.dirname(__file__), "after_codesign_opt")
CODESIGN_OPT_TRUNC_FOLDER = os.path.join(os.path.dirname(__file__), "codesign_opt_trunc")
SCALEHLS_OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "scalehls_out_file")
SCALEHLS_OUTPUT_LOG_FOLDER = os.path.join(os.path.dirname(__file__), "scalehls_out_log")
TEST_LOG_FOLDER_CODESIGN_OPT = os.path.join(os.path.dirname(__file__), "test_log_codesign_opt")
TEST_LOG_FOLDER_SCALEHLS = os.path.join(os.path.dirname(__file__), "test_log_scalehls")


def run_pytorch_file(input_file_name):
    
    codesign_opt_cmd = [
        "bash", "-c",
        f'''
        cd {CODESIGN_ROOT_DIR}
        source full_env_start.sh
        cd {CODESIGN_OPT_DIR}
        source codesign-opt-env.sh
        echo "Running codesign-opt"
        echo $PATH
        python {INPUT_FOLDER}/{input_file_name}.py > {INITIAL_MLIR_FOLDER}/{input_file_name}.mlir
        python {CODESIGN_OPT_DIR}/test/replace_maximumf.py {INITIAL_MLIR_FOLDER}/{input_file_name}.mlir {AFTER_CODESIGN_OPT_FOLDER}/{input_file_name}.mlir
        python {CODESIGN_OPT_DIR}/test/truncate_mlir.py {AFTER_CODESIGN_OPT_FOLDER}/{input_file_name}.mlir -o {CODESIGN_OPT_TRUNC_FOLDER}/{input_file_name}.mlir
        deactivate
        conda deactivate
        '''
    ]

    with open(f"{TEST_LOG_FOLDER_CODESIGN_OPT}/{input_file_name}.log", "w") as f:
        p = subprocess.Popen(
            codesign_opt_cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={}  # clean environment
        )
        p.wait()
    if p.returncode != 0:
        print(f"Error: codesign-opt failed for {input_file_name}")
        return

    scalehls_cmd = [
        "bash", "-c",
        f'''
        cd {CODESIGN_ROOT_DIR}
        source full_env_start.sh
        cd {SCALEHLS_DIR}
        source scalehls_env.sh
        scalehls-opt {AFTER_CODESIGN_OPT_FOLDER}/{input_file_name}.mlir -hida-pytorch-pipeline="top-func=forward loop-tile-size=1 loop-unroll-factor=1" -debug-only=scalehls 2>&1 | tee {SCALEHLS_OUTPUT_LOG_FOLDER}/{input_file_name}.log
        deactivate
        conda deactivate
        '''
    ]
    
    truncate_log_cmd = [
        "bash", "-c",
        f'''
        cd {CODESIGN_ROOT_DIR}
        source full_env_start.sh
        cd {CODESIGN_OPT_DIR}
        source codesign-opt-env.sh
        python {CODESIGN_OPT_DIR}/test/truncate_mlir.py {SCALEHLS_OUTPUT_LOG_FOLDER}/{input_file_name}.log -o {SCALEHLS_OUTPUT_LOG_FOLDER}/{input_file_name}.log
        deactivate
        conda deactivate
        '''
    ]

    with open(f"{TEST_LOG_FOLDER_SCALEHLS}/{input_file_name}.log", "w") as f:
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

    with open(f"{TEST_LOG_FOLDER_SCALEHLS}/{input_file_name}.log", "w") as f:
        p = subprocess.Popen(
            truncate_log_cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={}  # clean environment
        )
        p.wait()
    if p.returncode != 0:
        print(f"Error: truncate log failed for {input_file_name}")
        return

    print(f"codesign-opt and scalehls completed for {input_file_name}. Check logs for possible issues")

if __name__ == "__main__":
    """
    This script is used to run the pytorch file through the codesign-opt and scalehls pipeline.
    It will run the pytorch file through the codesign-opt pipeline and then the scalehls pipeline.
    It will then truncate the log file and save it to the test_log_folder.
    It will then print the success message.
    MLIR files are truncated at various points so that they can be viewed, because often the tensors take up too much space.
    But they are not truncated in the actual run of scalehls.

    Example Usage:
    python run_pytorch_file.py bitnet_small
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    input_file = args.input_file
    run_pytorch_file(input_file)