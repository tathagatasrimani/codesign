#!/bin/bash

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

## update submodules
git submodule update --init --recursive

## update conda packages
conda update -n base -c defaults conda # update conda itself
conda env update -f environment_simplified.yml --prune # update the environment
conda activate codesign # activate the environment


## make cacti 
cd src/cacti
make
cd ../..


## Change for the catapult environment name you want to use
source stanford_catapult_env.sh