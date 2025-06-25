#!/bin/bash

git submodule update --init scalehls
git clone https://github.com/UIUC-ChenLab/scalehls.git
cd scalehls

sed -i "s|git@github\.com:|https://github.com/|g" .gitmodules
git submodule update --init polygeist
cd polygeist
git submodule update --init llvm-project
cd ..

./build-scalehls.sh

export PATH=$PATH:$PWD/build/bin:$PWD/polygeist/build/bin
export PYTHONPATH=$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core

echo "current directory: $(pwd)"

conda create -n torch-mlir python=3.11 numpy=1.24
conda activate torch-mlir
python -m pip install --upgrade pip

wget https://cmu.box.com/shared/static/8hz00av1wm93pttfz7212xagtv0nkd6x -O torch-2.2.0.dev20231204+cpu-cp311-cp311-linux_x86_64.whl

wget https://cmu.box.com/shared/static/ag5ofnldjtrkr2uw6h4af6sew6f3cw6h -O torch_mlir-20231229.1067-cp311-cp311-linux_x86_64.whl

pip install torch-2.2.0.dev20231204+cpu-cp311-cp311-linux_x86_64.whl
pip install torch_mlir-20231229.1067-cp311-cp311-linux_x86_64.whl

rm -rf torch-2.2.0.dev20231204+cpu-cp311-cp311-linux_x86_64.whl torch_mlir-20231229.1067-cp311-cp311-linux_x86_64.whl

conda deactivate

cd ..


