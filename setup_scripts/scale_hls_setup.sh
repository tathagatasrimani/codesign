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

    ###########################################################
    # CRITICAL: ENSURE POLYGEIST LLVM SUBMODULE EXISTS
    ###########################################################
    if [[ ! -d "polygeist/llvm-project" ]]; then
        echo "[setup] ERROR: polygeist/llvm-project submodule missing!"
        exit 1
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

    ###########################################################
    # Clean ONLY the polygeist llvm-project + scalehls build
    ###########################################################
    echo "[setup] Cleaning old Polygeist + LLVM builds..."
    rm -rf polygeist/build
    rm -rf llvm-project
    rm -rf build

    echo "[setup] Cleaned old build directories."
    echo "[setup] System-level LLVM ignored (build-scalehls will use its own llvm-project)."

    # Run the unified Polygeist+LLVM+ScaleHLS build
    ./build-scalehls.sh || echo "[setup] WARNING: build-scalehls.sh returned non-zero (tests may have failed)"

else
    echo "[setup] Skipping ScaleHLS build (no --full)."
    cd ScaleHLS-HIDA
fi

#######################################################################
# REQUIRE LOCAL LLVM IN POLYGEIST DIRECTORY (fast mode)
#######################################################################
if [[ -x "$PWD/llvm-project/build/bin/llc" ]]; then
    LLVM_HOME="$PWD/llvm-project/build"
    echo "[setup] Local LLVM FOUND at $LLVM_HOME"
else
    echo "**************************************************************"
    echo "[ERROR] Local LLVM NOT FOUND in Polygeist directory!"
    echo "[ERROR] Expected: ScaleHLS-HIDA/llvm-project/build/bin/llc"
    echo "[ERROR] You are in the SLOW MODE (unified build)."
    echo "[ERROR] Run again with: ./scale_hls_setup.sh --full"
    echo "**************************************************************"
    exit 1
fi

export LLVM_HOME
export MLIR_HOME="$LLVM_HOME"
export PATH="$LLVM_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM_HOME/lib:$LD_LIBRARY_PATH"

#######################################################################
# Add ScaleHLS + Polygeist tools
#######################################################################
export PATH="$PWD/build/bin:$PATH"
export PATH="$PWD/polygeist/build/bin:$PATH"

#######################################################################
# Setup PYTHONPATH
#######################################################################
remove_pythonpath=0
if [[ -z "${PYTHONPATH+x}" ]]; then
    export PYTHONPATH="$PWD/build/tools/scalehls/python_packages/scalehls_core"
    remove_pythonpath=1
else
    export PYTHONPATH="$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core"
fi

#######################################################################
# Setup venv
#######################################################################
if [[ ! -d "mlir_venv" ]]; then
    python3.11 -m venv mlir_venv
    source mlir_venv/bin/activate
    python3.11 -m pip install --upgrade pip
    pip install --no-deps -r requirements.txt
    deactivate
fi

#######################################################################
# Optional Torch test
#######################################################################
if [[ $FORCE_FULL -eq 1 ]]; then
    source mlir_venv/bin/activate
    python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
    deactivate
fi

#######################################################################
# Restore PYTHONPATH
#######################################################################
cd ..
if [[ $remove_pythonpath -eq 1 ]]; then
    unset PYTHONPATH
else
    export PYTHONPATH="$PREV_PYTHONPATH"
fi

echo "[setup] DONE — using FAST Polygeist LLVM mode"