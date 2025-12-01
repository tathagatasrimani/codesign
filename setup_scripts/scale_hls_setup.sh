#!/bin/bash

############################################################
# Parse args
############################################################
FORCE_FULL=0
for arg in "$@"; do
    if [[ "$arg" == "--full" || "$arg" == "1" ]]; then
        FORCE_FULL=1
        break
    fi
done

echo "[setup] FORCE_FULL = ${FORCE_FULL}"

############################################################
# Save PYTHONPATH
############################################################
PREV_PYTHONPATH="$PYTHONPATH"

############################################################
# FULL BUILD: update submodules + run build script
############################################################
if [[ $FORCE_FULL -eq 1 ]]; then
    echo "[setup] Performing FULL unified ScaleHLS + LLVM build"

    git submodule init
    sed -i "s|git@github.com:|https://github.com/|g" .gitmodules
    git submodule sync

    git submodule update --init ScaleHLS-HIDA

    cd ScaleHLS-HIDA
    sed -i "s|git@github.com:|https://github.com/|g" .gitmodules
    git submodule sync
    git submodule update --init --recursive

    # Ensure lld
    if ! command -v lld >/dev/null 2>&1; then
        sudo yum install -y lld
    fi

    echo "[setup] Removing old build/"
    rm -rf build

    echo "[setup] Running unified build (tests may fail)..."
    ./build-scalehls.sh || echo "[setup] WARNING: test failures expected"

else
    echo "[setup] Not a full build; setting environment only"
    cd ScaleHLS-HIDA
fi

############################################################
# Require unified local LLVM: ScaleHLS-HIDA/build/bin/llc
############################################################
echo "[setup] Checking for local unified LLVM..."

if [[ ! -x "$PWD/build/bin/llc" ]]; then
    echo "**************************************************************"
    echo "[ERROR] Local unified LLVM NOT FOUND!"
    echo "[ERROR] Expected: ScaleHLS-HIDA/build/bin/llc"
    echo "[ERROR] System LLVM must NEVER be used."
    echo "[ERROR] Run: ./scale_hls_setup.sh --full"
    echo "**************************************************************"
    exit 1
fi

echo "[setup] Found local LLVM: $PWD/build/bin/llc"

export LLVM_HOME="$PWD/build"
export MLIR_HOME="$LLVM_HOME"
export PATH="$LLVM_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM_HOME/lib:$LD_LIBRARY_PATH"


############################################################
# Add ScaleHLS + Polygeist tools
############################################################
export PATH="$PWD/build/bin:$PATH"           # scalehls-opt, mlir-opt, llc, clang, etc.
export PATH="$PWD/polygeist/build/bin:$PATH" # if exists

echo "[setup] PATH updated"


############################################################
# Setup PYTHONPATH
############################################################
remove_pythonpath=0
if [[ -z "${PYTHONPATH+x}" ]]; then
    export PYTHONPATH="$PWD/build/tools/scalehls/python_packages/scalehls_core"
    remove_pythonpath=1
else
    export PYTHONPATH="$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core"
fi


############################################################
# Torch-MLIR virtualenv
############################################################
if [[ ! -d "mlir_venv" ]]; then
    echo "[setup] Creating Torch-MLIR venv..."
    python3.11 -m venv mlir_venv
    source mlir_venv/bin/activate
    python3.11 -m pip install --upgrade pip
    pip install --no-deps -r requirements.txt
    deactivate
else
    echo "[setup] Using existing venv"
fi


############################################################
# Optional: torch test
############################################################
if [[ $FORCE_FULL -eq 1 ]]; then
    source mlir_venv/bin/activate
    python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
    deactivate
fi


############################################################
# Restore PYTHONPATH
############################################################
cd ..

if [[ $remove_pythonpath -eq 1 ]]; then
    unset PYTHONPATH
else
    export PYTHONPATH="$PREV_PYTHONPATH"
fi

echo "[setup] ScaleHLS setup complete."
