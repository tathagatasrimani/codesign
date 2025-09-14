#!/bin/bash

################## PARSE UNIVERSITY ARGUMENT ##################

host=$(hostname)

if [[ "$host" == *stanford* ]]; then
    export UNIVERSITY="stanford"
elif [[ "$host" == *cmu* ]]; then
    export UNIVERSITY="cmu"
else
    echo "Hostname is '$host' — does not contain 'stanford' or 'cmu'."
    read -p "Please pick your university (stanford/cmu): " choice
    case "$choice" in
        stanford|Stanford|STANFORD)
            export UNIVERSITY="stanford"
            ;;
        cmu|CMU|Cmu)
            export UNIVERSITY="cmu"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

echo "UNIVERSITY set to: $UNIVERSITY"

################## INSTALL OPENROAD ##################
git submodule update --init --recursive openroad_interface/OpenROAD

# check if the openroad executable exists
if [ -f "openroad_interface/OpenROAD/build/src/openroad" ]; then
    echo "OpenROAD executable already exists."
else
    echo "OpenROAD executable not found. Running openroad_install.sh..."
    # Check OS, run openroad install script
    if [ -f /etc/redhat-release ]; then
        OS_VERSION=$(cat /etc/redhat-release)
        case "$OS_VERSION" in 
            *"Rocky Linux release 8"*|*"Red Hat Enterprise Linux release 8"*)
                bash openroad_install_rhel8.sh
            ;;
            *"Rocky Linux release 9"*|*"Red Hat Enterprise Linux release 9"*)
                bash openroad_install.sh
            ;;
            *)
                echo "Unsupported Rocky Linux version: $OS_VERSION"
                exit 1
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

################ SET UP SCALEHLS ##################
## we want this to operate outside of conda, so do this first
source scale_hls_setup.sh # setup scalehls

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

conda activate codesign # activate the codesign environment


## update the rest of the submodules
git submodule update --init --recursive

###############  BUILD CACTI #################3
cd src/cacti
make
cd ../..

## make verilator
source verilator_install.sh

## Load cad tools
if [ "$UNIVERSITY" = "stanford" ]; then
    echo "Setting up Stanford CAD tools..."
    source stanford_cad_tool_setup.sh
elif [ "$UNIVERSITY" = "cmu" ]; then
    echo "Setting up CMU CAD tools..."
    source cmu_cad_tool_setup.sh
else
    echo "Unsupported university for licensed cad tool setup: $UNIVERSITY"
    exit 1
fi