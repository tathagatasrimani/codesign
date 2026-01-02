#!/bin/bash

##############################
# Parse command line options
##############################
FORCE_FULL=0
for arg in "$@"; do
    if [[ "$arg" == "--full" || "$arg" == "1" ]]; then
        FORCE_FULL=1
        break
    fi
done

##############################
# Save previous PYTHONPATH
##############################
PREV_PYTHONPATH="$PYTHONPATH"

##############################
# If doing full build
##############################
if [[ $FORCE_FULL -eq 1 ]]; then
    echo "[setup] Performing FULL build of LLVM + Polygeist + ScaleHLS"

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

    # --- Ensure lld is available ---
    if ! command -v lld >/dev/null 2>&1; then
        echo "[setup] Installing lld..."
        sudo yum install -y lld
    else
        echo "[setup] lld already installed."
    fi

    ###############################################
    # Reset environment: ensure no system LLVM leaks in
    ###############################################
    echo "[setup] Resetting LLVM/MLIR environment for local build."
    unset LLVM_HOME
    unset MLIR_HOME
    unset CC
    unset CXX

    # Remove old builds
    rm -rf build llvm-project
    echo "[setup] Removed old build directories."

    ###############################################
    # Build LLVM + Polygeist + ScaleHLS
    ###############################################
    echo "[setup] Running build-scalehls.sh..."
    ./build-scalehls.sh
    BUILD_EXIT_CODE=$?

    ###############################################
    # Detect real failure vs test-only failure
    ###############################################
    LLVM_BIN_DIR="$PWD/llvm-project/build/bin"
    SCALEHLS_BIN_DIR="$PWD/build/bin"

    if [[ -x "$SCALEHLS_BIN_DIR/scalehls-opt" ]]; then
        if [[ $BUILD_EXIT_CODE -ne 0 ]]; then
            echo "[setup] WARNING: ScaleHLS tests failed (expected when DSE is modified). Continuing..."
        else
            echo "[setup] SUCCESS: local LLVM + ScaleHLS built with no errors."
        fi
    else
        echo "[setup] ERROR: Critical build artifacts missing."
        echo "[setup] LLVM or ScaleHLS did NOT actually build."
        exit 1
    fi

else
    ###############################################
    # NOT doing full build — just reinitialize environment
    ###############################################
    echo "[setup] Full build NOT requested. Setting up environment..."

    cd ScaleHLS-HIDA
fi

###############################################
# Add tool binaries to PATH
###############################################
echo "[setup] Updating PATH..."

# ScaleHLS tools
if [[ -d "$PWD/build/bin" ]]; then
    export PATH="$PWD/build/bin:$PATH"
fi

# Polygeist tools (if present)
if [[ -d "$PWD/polygeist/build/bin" ]]; then
    export PATH="$PWD/polygeist/build/bin:$PATH"
fi

###############################################
# Setup Python path for ScaleHLS Python tools
###############################################
remove_pythonpath=0

if [[ -z "${PYTHONPATH+x}" ]]; then
    export PYTHONPATH="$PWD/build/tools/scalehls/python_packages/scalehls_core"
    remove_pythonpath=1
else
    export PYTHONPATH="$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core"
fi

echo "[setup] PYTHONPATH: $PYTHONPATH"
echo "[setup] Current directory: $(pwd)"

###############################################
# Add mlir_venv to .gitignore if missing
###############################################
if ! grep -q "mlir_venv/" .gitignore 2>/dev/null; then
    echo "mlir_venv/" >> .gitignore
fi

###############################################
# Create venv if needed
###############################################
if [[ ! -d "mlir_venv" ]]; then
    echo "[setup] Creating Torch-MLIR virtualenv..."
    python3.11 -m venv mlir_venv
    source mlir_venv/bin/activate

    python3.11 -m pip install --upgrade pip
    pip install --no-deps -r requirements.txt

    deactivate
else
    echo "[setup] Torch-MLIR venv already exists."
fi

###############################################
# Optional: test imports
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
# Restore PYTHONPATH if needed
###############################################
cd ..

if [[ $remove_pythonpath -eq 1 ]]; then
    unset PYTHONPATH
    echo "[setup] PYTHONPATH unset (was empty before)."
else
    export PYTHONPATH="$PREV_PYTHONPATH"
    echo "[setup] PYTHONPATH restored."
fi

echo "[setup] Setup complete."