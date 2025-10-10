#!/bin/bash

################## CHECK BUILD LOG / FORCE FULL ##################
BUILD_LOG="build.log"
FORCE_FULL=0

# Parse command line options
for arg in "$@"; do
    if [[ "$arg" == "--full" ]]; then
        FORCE_FULL=1
        break
    fi
done

if [[ $FORCE_FULL -eq 0 ]]; then
    if [[ ! -f "$BUILD_LOG" ]]; then
        echo "No build log found — forcing full build."
        FORCE_FULL=1
    else
        last_epoch=$(date -r "$BUILD_LOG" +%s)
        now_epoch=$(date +%s)
        diff_days=$(( (now_epoch - last_epoch) / 86400 ))
        if [[ $diff_days -ge 7 ]]; then
            echo "Last build was $diff_days days ago — forcing full build."
            FORCE_FULL=1
        fi
    fi
fi

if [[ $FORCE_FULL -eq 1 ]]; then
    echo ">>> Performing FULL build"
else
    echo ">>> Performing incremental build"
fi

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

OPENROAD_PRE_INSTALLED_BIN_PATH="../../../../deps/OpenROAD/build/bin/openroad"
OPENROAD_PREINSTALLED_SRC_PATH="openroad_interface/OpenROAD/build/src/openroad"


printf '>>> SCRIPT START %s\n' "$(date)"
printf 'PWD: %s\n' "$(pwd)"
printf 'Contents:\n'
ls -la

echo " -> Looking for BIN_PATH: $OPENROAD_PRE_INSTALLED_BIN_PATH"
if [ -f "$OPENROAD_PRE_INSTALLED_BIN_PATH" ]; then
    echo "!!!!! Found: $OPENROAD_PRE_INSTALLED_BIN_PATH"
else
    echo "XXXXX Not found: $OPENROAD_PRE_INSTALLED_BIN_PATH"
    echo " -> Looking for Open road pre installed SRC_PATH: $OPENROAD_PREINSTALLED_SRC_PATH"
    if [ -f "$OPENROAD_PREINSTALLED_SRC_PATH" ]; then
        echo "!!!!! Found: $OPENROAD_PREINSTALLED_SRC_PATH"
    else
        echo "XXXXX Not found: $OPENROAD_PREINSTALLED_SRC_PATH"
        echo "OpenROAD executable not found. Running openroad_install.sh..."
        # bash openroad_install.sh
    fi
fi

echo "UNIVERSITY set to: $UNIVERSITY"

## set home directory to codesign home directory
export HOME="$(pwd)"
export PATH="$HOME/.local/bin:$(echo "$PATH")"
export CMAKE_PREFIX_PATH="$HOME/.local"

## for cmu setup, set tmp directory to local directory to avoid filling system tmp
if [ "$UNIVERSITY" = "cmu" ]; then
    export TMPDIR="$HOME/.tmp"
    export TEMP="$TMPDIR"
    export TEMPDIR="$TMPDIR"
    export TMP="$TMPDIR"
    mkdir -p "$TMPDIR"
    echo "Set TMPDIR to $TMPDIR"
fi

################## INSTALL OPENROAD ##################
git submodule update --init --recursive openroad_interface/OpenROAD

if  [[ "${GITHUB_ACTIONS:-}" == "true" && -f $OPENROAD_PRE_INSTALLED_BIN_PATH ]]; then
    echo "OpenROAD executable already exists."
else
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
fi

if [[ "${GITHUB_ACTIONS:-}" == "true" && -f $OPENROAD_PRE_INSTALLED_BIN_PATH ]]; then
    echo "OpenROAD installation completed successfully."
else
    # Ensure that the OpenROAD executable was created
    if [ -f "openroad_interface/OpenROAD/build/src/openroad" ]; then
        echo "OpenROAD installation completed successfully."
    else
        echo "OpenROAD installation failed."
        exit 1
    fi
fi

################ SET UP SCALEHLS ##################
## we want this to operate outside of conda, so do this first
source scale_hls_setup.sh $FORCE_FULL # setup scalehls

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


if [[ $FORCE_FULL -eq 1 ]]; then
    ## update conda packages
    conda update -n base -c defaults conda # update conda itself
    conda config --set channel_priority strict
    conda env update -f environment_simplified.yml --prune # update the environment
fi

conda activate codesign # activate the codesign environment


## update the rest of the submodules
if [[ $FORCE_FULL -eq 1 ]]; then
    git submodule update --init --recursive
fi

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

############### Add useful alisas ###############
alias create_checkpoint="python3 -m test.checkpoint_controller"
alias run_codesign="python3 -m src.codesign"

################## SUCCESSFUL BUILD LOG ##################
if [[ $FORCE_FULL -eq 1 ]]; then
    date "+%Y-%m-%d %H:%M:%S" > "$BUILD_LOG"
fi

echo "Last full build completed successfully on $(cat $BUILD_LOG)"which

echo "Environment setup complete."
