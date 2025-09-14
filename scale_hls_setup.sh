#!/bin/bash
set -euo pipefail

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

./build-scalehls.sh

export PATH=$PATH:$PWD/build/bin:$PWD/polygeist/build/bin
export PYTHONPATH=$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core

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