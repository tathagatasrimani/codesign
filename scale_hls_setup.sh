#!/bin/bash

git submodule update --init scalehls-hida
git clone https://github.com/UIUC-ChenLab/ScaleHLS-HIDA.git
cd scalehls-hida

sed -i "s|git@github\.com:|https://github.com/|g" .gitmodules
git submodule update --init polygeist
cd polygeist
git submodule update --init llvm-project
cd ..

./build-scalehls.sh

export PATH=$PATH:$PWD/build/bin:$PWD/polygeist/build/bin
export PYTHONPATH=$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core

echo "current directory: $(pwd)"

## check if the torch-mlir conda environment already exists
if conda env list | grep -q "torch-mlir"; then
    echo "torch-mlir environment already exists."
else
    echo "Creating torch-mlir environment..."
    # Create env (add pip so we can install wheels/reqs)
    conda create -y -n torch-mlir python=3.11 numpy=1.24 pip

    # Ensure 'conda activate' works inside a non-interactive script
    conda activate torch-mlir

    python -m pip install --upgrade pip

    # We should be in the ScaleHLS-HIDA repo root when running this block
    # (requirements.txt lives there, per README)
    if [ -f requirements.txt ]; then
        # Install prebuilt Torch-MLIR stack as specified by the repo
        pip install --no-deps -r requirements.txt
    else
        echo "ERROR: requirements.txt not found in $(pwd). Please run this from the scalehls-hida repo root."
        conda deactivate
        exit 1
    fi

conda deactivate
fi

cd ..