#!/bin/bash

###############################################
# Parse command line options
###############################################
FORCE_FULL=0
for arg in "$@"; do
    if [[ "$arg" == "1" || "$arg" == "--full" ]]; then
        FORCE_FULL=1
        break
    fi
done

###############################################
# Save previous PYTHONPATH
###############################################
PREV_PYTHONPATH="$PYTHONPATH"

echo "[setup] FORCE_FULL = $FORCE_FULL"

###############################################
# FULL BUILD (local LLVM + Polygeist + ScaleHLS)
###############################################
if [[ $FORCE_FULL -eq 1 ]]; then
    echo "[setup] Performing FULL clean build of ScaleHLS + local LLVM"

    # --- Top-level repo ---
    git submodule init
    sed -i "s|git@github.com:|https://github.com/|g" .gitmodules
    git submodule sync
    git submodule update --init ScaleHLS-HIDA

    # --- Inside ScaleHLS-HIDA ---
    cd ScaleHLS-HIDA

    sed -i "s|git@github.com:|https://github.com/|g" .gitmodules
    git submodule sync
    git submodule update --init --recursive

    ###############################################
    # Force NO system LLVM leakage
    ###############################################
    echo "[setup] Blocking system LLVM/MLIR from CMake"

    # Remove LLVM detection environment
    unset LLVM_DIR LLVM_ROOT LLVM_INSTALL_PREFIX
    unset MLIR_DIR MLIR_ROOT

    # Remove system LLVM binaries from PATH
    PATH=$(printf "%s" "$PATH" | tr ':' '\n' | grep -vE '^/usr/bin$' | grep -vE '/usr/bin' | paste -sd: -)
    export PATH

    # Force CMake to use GCC for bootstrapping (not system clang)
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++

    echo "[setup] PATH cleaned. CC=$CC CXX=$CXX"

    ###############################################
    # Remove all old builds (important!)
    ###############################################
    echo "[setup] Removing previous builds..."
    rm -rf build llvm-project

    ###############################################
    # Run build-scalehls.sh
    ###############################################
    echo "[setup] Running build-scalehls.sh..."
    ./build-scalehls.sh
    BUILD_EXIT_CODE=$?

    ###############################################
    # Validate build
    ###############################################
    LLVM_BIN_DIR="$PWD/llvm-project/build/bin"
    SCALEHLS_BIN_DIR="$PWD/build/bin"

    # ScaleHLS must exist
    if [[ ! -x "$SCALEHLS_BIN_DIR/scalehls-opt" ]]; then
        echo "[setup] ERROR: scalehls-opt missing. ScaleHLS did NOT build."
        exit 1
    fi

    # Local LLVM must exist
    if [[ ! -x "$LLVM_BIN_DIR/llc" ]]; then
        echo "[setup] ERROR: Local LLVM was NOT built!"
        echo "[setup] CMake improperly detected system LLVM."
        echo "[setup] FIX: Ensure PATH is cleaned, and remove entire build/"
        exit 1
    fi

    # Tests may fail — this is OK
    if [[ $BUILD_EXIT_CODE -ne 0 ]]; then
        echo "[setup] WARNING: ScaleHLS tests failed (expected with DSE modifications). Continuing..."
    else
        echo "[setup] Build passed with no errors."
    fi

    ###############################################
    # Export local LLVM paths
    ###############################################
    export LLVM_HOME="$PWD/llvm-project/build"
    export MLIR_HOME="$LLVM_HOME"
    export PATH="$LLVM_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$LLVM_HOME/lib:$LD_LIBRARY_PATH"

    echo "[setup] Using LOCAL LLVM: $LLVM_HOME"

else
###############################################
# NON-FULL BUILD (reuse existing local LLVM)
###############################################
    echo "[setup] Full build NOT requested. Initializing environment..."

    cd ScaleHLS-HIDA

    if [[ ! -d "llvm-project/build/bin" ]]; then
        echo "[setup] ERROR: No local LLVM found. Must run with --full first."
        exit 1
    fi

    export LLVM_HOME="$PWD/llvm-project/build"
    export MLIR_HOME="$LLVM_HOME"
    export PATH="$LLVM_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$LLVM_HOME/lib:$LD_LIBRARY_PATH"

    echo "[setup] Using existing LOCAL LLVM: $LLVM_HOME"
fi

###############################################
# Add ScaleHLS + Polygeist tools to PATH
###############################################
echo "[setup] Updating PATH with ScaleHLS and Polygeist tools..."

if [[ -d "$PWD/build/bin" ]]; then
    export PATH="$PWD/build/bin:$PATH"
fi
if [[ -d "$PWD/polygeist/build/bin" ]]; then
    export PATH="$PWD/polygeist/build/bin:$PATH"
fi

###############################################
# Setup Python path for ScaleHLS Python backend
###############################################
remove_pythonpath=0
if [[ -z "${PYTHONPATH+x}" ]]; then
    export PYTHONPATH="$PWD/build/tools/scalehls/python_packages/scalehls_core"
    remove_pythonpath=1
else
    export PYTHONPATH="$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core"
fi

echo "[setup] PYTHONPATH = $PYTHONPATH"

###############################################
# Add mlir_venv to .gitignore
###############################################
if ! grep -q "mlir_venv/" .gitignore 2>/dev/null; then
    echo "mlir_venv/" >> .gitignore
fi

###############################################
# Torch-MLIR Virtualenv Setup
###############################################
if [[ ! -d "mlir_venv" ]]; then
    echo "[setup] Creating virtualenv..."
    python3.11 -m venv mlir_venv
    source mlir_venv/bin/activate
    python3.11 -m pip install --upgrade pip
    pip install --no-deps -r requirements.txt
    deactivate
else
    echo "[setup] Virtualenv already exists."
fi

###############################################
# Optional: Validate PyTorch env
###############################################
if [[ $FORCE_FULL -eq 1 ]]; then
    source mlir_venv/bin/activate
    python - <<'PY'
import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
PY
    deactivate
fi

###############################################
# Restore PYTHONPATH
###############################################
cd ..
if [[ $remove_pythonpath -eq 1 ]]; then
    unset PYTHONPATH
    echo "[setup] PYTHONPATH unset."
else
    export PYTHONPATH="$PREV_PYTHONPATH"
    echo "[setup] PYTHONPATH restored."
fi

echo "[setup] Setup complete."
