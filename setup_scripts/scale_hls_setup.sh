#!/bin/bash

############################################################
# Parse command line options
############################################################
FORCE_FULL=0
for arg in "$@"; do
    if [[ "$arg" == "1" || "$arg" == "--full" ]]; then
        FORCE_FULL=1
        break
    fi
done

echo "[setup] FORCE_FULL = $FORCE_FULL"

############################################################
# Preserve PYTHONPATH
############################################################
PREV_PYTHONPATH="$PYTHONPATH"


############################################################
# FULL BUILD (local LLVM + Polygeist + ScaleHLS)
############################################################
if [[ $FORCE_FULL -eq 1 ]]; then
    echo "[setup] Performing FULL clean build of ScaleHLS + llvm-project"

    # --- Top-level updates ---
    git submodule init
    sed -i "s|git@github.com:|https://github.com/|g" .gitmodules
    git submodule sync
    git submodule update --init ScaleHLS-HIDA

    cd ScaleHLS-HIDA
    sed -i "s|git@github.com:|https://github.com/|g" .gitmodules
    git submodule sync
    git submodule update --init --recursive

    ############################################################
    # Ensure lld exists
    ############################################################
    if ! command -v lld >/dev/null 2>&1; then
        echo "[setup] Installing lld..."
        sudo yum install -y lld
    else
        echo "[setup] lld already installed."
    fi

    ############################################################
    # BLOCK SYSTEM LLVM — but KEEP system utilities
    ############################################################
    echo "[setup] Blocking system LLVM/MLIR while keeping core utilities"

    # Unset LLVM-related CMake variables
    unset LLVM_DIR LLVM_ROOT LLVM_INSTALL_PREFIX
    unset MLIR_DIR MLIR_ROOT

    # Prevent CMake from locating system LLVMConfig.cmake
    export LLVM_DIR=NO_SYSTEM_LLVM
    export MLIR_DIR=NO_SYSTEM_MLIR
    export LLVM_ROOT=NO_SYSTEM_LLVM
    export MLIR_ROOT=NO_SYSTEM_MLIR

    # Remove ONLY LLVM-related binaries from PATH
    PATH=$(printf "%s" "$PATH" \
        | tr ':' '\n' \
        | grep -vE '/usr/bin/(clang|clang\+\+|opt|llc|mlir|lld)' \
        | paste -sd: -)
    export PATH

    # Force GCC for bootstrapping the llvm-project build
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++

    ############################################################
    # CLEAN OLD BUILDS
    ############################################################
    echo "[setup] Removing previous build directories..."
    rm -rf build llvm-project

    ############################################################
    # RUN BUILD
    ############################################################
    echo "[setup] Running build-scalehls.sh..."
    ./build-scalehls.sh
    BUILD_EXIT_CODE=$?


    ############################################################
    # VERIFY BUILD RESULTS
    ############################################################
    LLVM_BIN_DIR="$PWD/llvm-project/build/bin"
    SCALEHLS_BIN_DIR="$PWD/build/bin"

    # ScaleHLS must exist
    if [[ ! -x "$SCALEHLS_BIN_DIR/scalehls-opt" ]]; then
        echo "[setup] ERROR: scalehls-opt missing — ScaleHLS FAILED to build."
        exit 1
    fi

    # Local LLVM must exist
    if [[ ! -x "$LLVM_BIN_DIR/llc" ]]; then
        echo "[setup] ERROR: Local LLVM NOT built! System LLVM was mistakenly used."
        echo "[setup] Fix: ensure PATH filtering is correct and start from clean repo."
        exit 1
    fi

    # ScaleHLS tests can fail safely
    if [[ $BUILD_EXIT_CODE -ne 0 ]]; then
        echo "[setup] WARNING: ScaleHLS test failures detected (expected). Continuing..."
    else
        echo "[setup] SUCCESS: Build completed with no errors."
    fi

    ############################################################
    # Export local LLVM environment
    ############################################################
    export LLVM_HOME="$PWD/llvm-project/build"
    export MLIR_HOME="$LLVM_HOME"
    export PATH="$LLVM_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$LLVM_HOME/lib:$LD_LIBRARY_PATH"

    echo "[setup] Using LOCAL LLVM at $LLVM_HOME"

else
############################################################
# NON-FULL BUILD — reinitialize environment
############################################################
    echo "[setup] Full build NOT requested — setting up environment."

    cd ScaleHLS-HIDA

    if [[ ! -d "llvm-project/build/bin" ]]; then
        echo "[setup] ERROR: Local LLVM not found. Must run with --full first."
        exit 1
    fi

    export LLVM_HOME="$PWD/llvm-project/build"
    export MLIR_HOME="$LLVM_HOME"
    export PATH="$LLVM_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$LLVM_HOME/lib:$LD_LIBRARY_PATH"

    echo "[setup] Using existing LOCAL LLVM at $LLVM_HOME"
fi


############################################################
# Add ScaleHLS + Polygeist to PATH
############################################################
echo "[setup] Adding ScaleHLS + Polygeist binaries to PATH"

if [[ -d "$PWD/build/bin" ]]; then
    export PATH="$PWD/build/bin:$PATH"
fi
if [[ -d "$PWD/polygeist/build/bin" ]]; then
    export PATH="$PWD/polygeist/build/bin:$PATH"
fi


############################################################
# Python path setup
############################################################
remove_pythonpath=0

if [[ -z "${PYTHONPATH+x}" ]]; then
    export PYTHONPATH="$PWD/build/tools/scalehls/python_packages/scalehls_core"
    remove_pythonpath=1
else
    export PYTHONPATH="$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core"
fi

echo "[setup] PYTHONPATH = $PYTHONPATH"


############################################################
# .gitignore update
############################################################
if ! grep -q "mlir_venv/" .gitignore 2>/dev/null; then
    echo "mlir_venv/" >> .gitignore
fi


############################################################
# Virtualenv setup
############################################################
if [[ ! -d "mlir_venv" ]]; then
    echo "[setup] Creating Torch-MLIR virtualenv..."
    python3.11 -m venv mlir_venv
    source mlir_venv/bin/activate
    python3.11 -m pip install --upgrade pip
    pip install --no-deps -r requirements.txt
    deactivate
else
    echo "[setup] Virtualenv already exists."
fi


############################################################
# Optional import test
############################################################
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


############################################################
# Restore PYTHONPATH
############################################################
cd ..

if [[ $remove_pythonpath -eq 1 ]]; then
    unset PYTHONPATH
    echo "[setup] PYTHONPATH unset (was empty before)."
else
    export PYTHONPATH="$PREV_PYTHONPATH"
    echo "[setup] PYTHONPATH restored."
fi

echo "[setup] Setup complete."
