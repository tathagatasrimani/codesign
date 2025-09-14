#!/bin/bash

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

./build-scalehls.sh

export PATH=$PATH:$PWD/build/bin:$PWD/polygeist/build/bin

## python path might not be set, so check first
if [ -z "${PYTHONPATH+x}" ]; then
    export PYTHONPATH="$PWD/build/tools/scalehls/python_packages/scalehls_core"
else
    export PYTHONPATH="$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core"
fi


echo "current directory: $(pwd)"

# Create venv only if it does not exist
if [ ! -d "mlir_venv" ]; then
    echo "Creating new Torch-MLIR venv..."
    python3.11 -m venv mlir_venv
    source mlir_venv/bin/activate

    python3.11 -m pip install --upgrade pip
    if [ -f requirements.txt ]; then
        pip install --no-deps -r requirements.txt
    else
        echo "ERROR: requirements.txt not found in $(pwd)."
        deactivate
        exit 1
    fi

    echo "Torch-MLIR venv setup complete."
    deactivate
else
    echo "Torch-MLIR venv already exists. Skipping creation."
fi

# Optional: test that torch + torchvision import cleanly
source mlir_venv/bin/activate
python - <<'PY'
import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
PY
deactivate

cd ..