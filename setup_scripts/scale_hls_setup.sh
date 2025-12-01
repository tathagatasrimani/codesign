#!/bin/bash

# Parse command line options
FORCE_FULL=0
for arg in "$@"; do
    if [[ "$arg" == "--full" || "$arg" == "1" ]]; then
        FORCE_FULL=1
        break
    fi
done

# Save original PYTHONPATH
PREV_PYTHONPATH="$PYTHONPATH"

if [[ $FORCE_FULL -eq 1 ]]; then
    echo "[setup] FULL BUILD of ScaleHLS + Polygeist LLVM"

    # --- Top-level ---
    git submodule init
    sed -i "s|git@github.com:|https://github.com/|g" .gitmodules
    git submodule sync
    git submodule update --init ScaleHLS-HIDA

    # --- Inside ScaleHLS-HIDA ---
    cd ScaleHLS-HIDA
    sed -i "s|git@github.com:|https://github.com/|g" .gitmodules
    git submodule sync
    git submodule update --init --recursive

    # --- Ensure lld is available on RHEL8/9 ---
    if ! command -v lld >/dev/null 2>&1; then
        echo "[setup] lld not found, installing with yum..."
        sudo yum install -y lld
    else
        echo "[setup] lld already installed."
    fi

    ###############################################
    # Ensure no stale LLVM/MLIR env interferes
    # And force a clean rebuild
    ###############################################
    echo "[setup] Resetting LLVM/MLIR environment to force local build..."

    unset LLVM_HOME
    unset MLIR_HOME
    unset CC
    unset CXX

    # Remove old builds that may have been corrupted
    rm -rf build llvm-project

    echo "[setup] Cleaned old build directories."
    echo "[setup] System-level LLVM ignored (build-scalehls will use its own llvm-project)."

    # Run the unified Polygeist+LLVM+ScaleHLS build
    ./build-scalehls.sh || echo "[setup] WARNING: build-scalehls.sh returned non-zero (tests may have failed)"

else
    echo "[setup] Skipping ScaleHLS build (no --full)."
    cd ScaleHLS-HIDA
fi

###############################################
# REQUIRE LOCAL LLVM (Polygeist-style layout)
###############################################
if [[ -d "$PWD/llvm-project/build/bin" ]]; then
    export LLVM_HOME="$PWD/llvm-project/build"
    export MLIR_HOME="$LLVM_HOME"
    export PATH="$LLVM_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$LLVM_HOME/lib:$LD_LIBRARY_PATH"
    echo "[setup] Using LOCAL LLVM: $LLVM_HOME"
else
    echo "**************************************************************"
    echo "[ERROR] Local LLVM NOT FOUND!"
    echo "[ERROR] Expected: ScaleHLS-HIDA/llvm-project/build/bin/llc"
    echo "[ERROR] System LLVM must NEVER be used."
    echo "[ERROR] Run again with: ./scale_hls_setup.sh --full"
    echo "**************************************************************"
    exit 1
fi

###############################################
# Add ScaleHLS tool binaries to PATH
###############################################
export PATH="$PWD/build/bin:$PATH"
export PATH="$LLVM_HOME/bin:$PATH"  # ensure local clang/opt/mlir-opt override system

remove_pythonpath=0

## python path might not be set, so check first
if [[ -z "${PYTHONPATH+x}" ]]; then
    export PYTHONPATH="$PWD/build/tools/scalehls/python_packages/scalehls_core"
    remove_pythonpath=1
else
    export PYTHONPATH="$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core"
fi

echo "current directory: $(pwd)"

## add mlir_venv/ to the .gitignore if not already present
if ! grep -q "mlir_venv/" .gitignore 2>/dev/null; then
    echo "Adding mlir_venv/ to .gitignore"
    echo "mlir_venv/" >> .gitignore
fi

# Create venv only if it does not exist
if [[ ! -d "mlir_venv" ]]; then
    echo "Creating new Torch-MLIR venv..."
    python3.11 -m venv mlir_venv
    source mlir_venv/bin/activate

    python3.11 -m pip install --upgrade pip
    if [[ -f requirements.txt ]]; then
        pip
