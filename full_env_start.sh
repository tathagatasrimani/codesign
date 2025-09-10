#!/bin/bash

## update submodules
git submodule update --init --recursive

# check if the openroad executable exists
if [ -f "openroad_interface/OpenROAD/build/src/openroad" ]; then
    echo "OpenROAD executable already exists."
else
    echo "OpenROAD executable not found. Running openroad_install.sh..."
    # Check OS, run openroad install script
    if [ -f /etc/redhat-release ] then
        OS_VERSION=$(cat /etc/redhat-release)
        case "$OS_VERSION" in 
            *"Rocky Linux release 8"*)
                bash openroad_install_rhel8.sh
            ;;
            *"Rocky Linux release 9"*)
                bash openroad_install.sh
            ;;
        esac    
    else
        echo "Unsupported OS"
        exit 1
    fi
fi

# Ensure that the OpenROAD executable was created
if [ -f "openroad_interface/OpenROAD/build/src/openroad" ]; then
    echo "OpenROAD installation completed successfully."
else
    echo "OpenROAD installation failed."
    exit 1
fi

# Check if the directory miniconda3 exists
if [ -d "miniconda3" ]; then
    export PATH="$(pwd)/miniconda3/bin:$PATH"
    source miniconda3/etc/profile.d/conda.sh
else
    # Install and set up environment
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$(pwd)/miniconda3"
    export PATH="$(pwd)/miniconda3/bin:$PATH"
    source miniconda3/etc/profile.d/conda.sh
    conda env create -f environment_simplified.yml

    # create symlinks for g++-13 needed by cacti
    cd miniconda3/envs/codesign/bin
    ln -s x86_64-conda-linux-gnu-gcc gcc-13
    ln -s x86_64-conda-linux-gnu-g++ g++-13
    cd ../../../..
fi


## update conda packages
conda update -n base -c defaults conda # update conda itself
conda env update -f environment_simplified.yml --prune # update the environment
conda activate codesign # activate the environment

## build ScaleHLS-HIDA and export env
if [ -d "scalehls-hida" ]; then
    echo "ScaleHLS-HIDA directory exists. Skipping clone."
else
    echo "Cloning ScaleHLS-HIDA..."
    git clone --recursive git@github.com:UIUC-ChenLab/ScaleHLS-HIDA.git scalehls-hida
fi

cd scalehls-hida
echo "Building ScaleHLS-HIDA (with Python bindings)..."
./build-scalehls.sh -p ON -j $(nproc)

# Export PATH and PYTHONPATH as suggested by README
export PATH=$PATH:$(pwd)/build/bin:$(pwd)/polygeist/build/bin
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/tools/scalehls/python_packages/scalehls_core
# Set up Torch-MLIR venv for PyTorch model compilation (from README)
if [ ! -d "mlir_venv" ]; then
    python3 -m venv mlir_venv
fi
source mlir_venv/bin/activate
python -m pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install --no-deps -r requirements.txt
fi
deactivate
cd ..


## make cacti 
cd src/cacti
make
cd ../..

## make verilator
source verilator_install.sh


## Change for the catapult environment name you want to use
source stanford_catapult_env.sh