#!/bin/bash

## Install scaleHLS in a clean environment
env -i bash -c '
  git submodule update --init scalehls
  git clone https://github.com/UIUC-ChenLab/scalehls.git
  cd scalehls

  sed -i "s|git@github\.com:|https://github.com/|g" .gitmodules
  git submodule update --init polygeist
  cd polygeist
  git submodule update --init llvm-project
  cd ..

  ./build-scalehls.sh
'

export PATH=$PATH:$PWD/build/bin:$PWD/polygeist/build/bin
export PYTHONPATH=$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core

conda create -n torch-mlir python=3.11
conda activate torch-mlir
python -m pip install --upgrade pip

pip install --pre torch-mlir torchvision \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels

conda deactivate torch-mlir


