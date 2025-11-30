#!/bin/bash

########################################################################
# Parse command line options
########################################################################
FORCE_FULL=0
for arg in "$@"; do
    if [[ "$arg" == "--full" || "$arg" == "1" ]]; then
        FORCE_FULL=1
        break
    fi
done

echo "[setup] FORCE_FULL = ${FORCE_FULL}"

########################################################################
# Save previous PYTHONPATH for restoration
########################################################################
PREV_PYTHONPATH="$PYTHONPATH"

########################################################################
# FULL BUILD MODE
########################################################################
if [[ $FORCE_FULL -eq 1 ]]; then
    echo "[setup] Running FULL build of ScaleHLS + local LLVM"

    # --- Initialize submodules ---
    git submodule init
    sed -i "s|git@github.com:|https://github.com/|g" .gitmodules
    git submodule sync
    git submodule update --init ScaleHLS-HIDA

    cd ScaleHLS-HIDA
    sed -i "s|git@github.com:|https://github.com/|g" .gitmodules
    git submodule sync
    git submodule update --init --recursive

    # --- Ensure lld exists ---
    if ! command -v lld >/dev/null 2>&1; then
        echo "[setup] Installing lld..."
        sudo yum install -y lld
    else
        echo "[setup] lld already installed."
    fi

    ####################################################################
    # Clean old builds
    ####################################################################
    echo "[setup] Removing previous build directories..."
    rm -rf build llvm-project polygeist/build polygeist/llvm-project/build

    ####################################################################
    # Run Polygeist + LLVM + ScaleHLS unified build
    ####################################################################
    echo "[setup] Running build-scalehls.sh..."
    ./build-scalehls.sh || echo "[setup] WARNING: build-scalehls.sh returned non-zero (test failures likely)."

else
    ####################################################################
    # NON-FULL MODE: only set environment
    ####################################################################
    echo "[setup] Full build NOT requested; setting environment only."
    cd ScaleHLS-HIDA
fi


########################################################################
# LOCAL LLVM IS REQUIRED — NO SYSTEM LLVM ALLOWED
########################################################################
echo "[setup] Checking for local LLVM..."

# Polygeist llvm-project should contain the local LLVM
if [[ -x "$PWD/polygeist/llvm-project/build/bin/llc" ]]; then
    LLVM_HOME="$PWD/polygeist/llvm-project/build"
elif [[ -x "$PWD/llvm-project/build/bin/llc" ]]; then
    LLVM_HOME="$PWD/llvm-project/build"
else
    echo "**************************************************************"
    echo "[ERROR] Local LLVM NOT FOUND!"
    echo "[ERROR] Expected llc at:"
    echo "    ScaleHLS-HIDA/polygeist/llvm-project/build/bin/llc"
    echo " or ScaleHLS-HIDA/llvm-project/build/bin/llc"
    echo "[ERROR] The system LLVM must NEVER be used."
    echo "[ERROR] Run again with: ./scale_hls_setup.sh --full"
    echo "**************************************************************"
    exit 1
fi

echo "[setup] Using LOCAL LLVM at: $LLVM_HOME"
export LLVM_HOME
export MLIR_HOME="$LLVM_HOME"
export PATH="$LLVM_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM_HOME/lib:$LD_LIBRARY_PATH"


########################################################################
# Add ScaleHLS + Polygeist binaries to PATH
########################################################################
echo "[setup] Adding ScaleHLS + Polygeist tools to PATH"

if [[ -d "$PWD/build/bin" ]]; then
    export PATH="$PWD/build/bin:$PATH"
else
    echo "[setup] ERROR: build/bin missing — ScaleHLS did not build."
    exit 1
fi

if [[ -d "$PWD/polygeist/build/bin" ]]; then
    export PATH="$PWD/polygeist/build/bin:$PATH"
fi

echo "[setup] PATH updated."


########################################################################
# Set up PYTHONPATH for ScaleHLS python packages
########################################################################
remove_pythonpath=0

if [[ -z "${PYTHONPATH+x}" ]]; then
    export PYTHONPATH="$PWD/build/tools/scalehls/python_packages/scalehls_core"
    remove_pythonpath=1
else
    export PYTHONPATH="$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core"
fi


########################################################################
# Setup Torch-MLIR venv if missing
########################################################################
if [[ ! -d "mlir_venv" ]]; then
    echo "[setup] Creating Torch-MLIR venv..."
    python3.11 -m venv mlir_venv
    source mlir_venv/bin/activate
    python3.11 -m pip install --upgrade pip

    if [[ -f requirements.txt ]]; then
        pip install --no-deps -r requirements.txt
    else
        echo "[setup] ERROR: requirements.txt missing!"
        deactivate
        exit 1
    fi
    deactivate
else
    echo "[setup] Torch-MLIR environment already exists."
fi


########################################################################
# Optional PyTorch import test
########################################################################
if [[ $FORCE_FULL -eq 1 ]]; then
    echo "[setup] Testing Torch installation..."
    source mlir_venv/bin/activate
    python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
    deactivate
fi

########################################################################
# Restore previous PYTHONPATH
########################################################################
cd ..

if [[ $remove_pythonpath -eq 1 ]]; then
    unset PYTHONPATH
    echo "PYTHONPATH was empty before; unset again."
else
    export PYTHONPATH="$PREV_PYTHONPATH"
    echo "PYTHONPATH restored."
fi

echo "[setup] ScaleHLS setup complete."
