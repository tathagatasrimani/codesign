#!/bin/bash

################## INSTALL OPENROAD ##################
git submodule update --init --recursive openroad_interface/OpenROAD

# check if the openroad executable exists
if [ -f "openroad_interface/OpenROAD/build/src/openroad" ]; then
    echo "OpenROAD executable already exists."
else
    echo "OpenROAD executable not found. Running openroad_install.sh..."
    # Run the OpenROAD installation script
    bash openroad_install.sh
fi

# Ensure that the OpenROAD executable was created
if [ -f "openroad_interface/OpenROAD/build/src/openroad" ]; then
    echo "OpenROAD installation completed successfully."
else
    echo "OpenROAD installation failed."
    exit 1
fi

################### SET UP CONDA ENVIRONMENT ##################
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


################ SET UP SCALEHLS ##################
source scale_hls_setup.sh # setup scalehls

conda activate codesign # activate the codesign environment


## update the rest of the submodules
git submodule update --init --recursive

###############  BUILD CACTI #################3
cd src/cacti
make
cd ../..

## make verilator
source verilator_install.sh


############### LOAD VITIS HLS #################
module load vitis/2022.1

## Change for the catapult environment name you want to use
source stanford_catapult_env.sh