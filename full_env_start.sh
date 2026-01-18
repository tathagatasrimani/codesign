#!/bin/bash

################## CHECK BUILD LOG / FORCE FULL ##################

SETUP_SCRIPTS_FOLDER="$(pwd)/setup_scripts"

BUILD_LOG="$SETUP_SCRIPTS_FOLDER/build.log"
FORCE_FULL=0
SKIP_OPENROAD=0

# Parse command line options
for arg in "$@"; do
    if [[ "$arg" == "--full" ]]; then
        FORCE_FULL=1
    elif [[ "$arg" == "--skip-openroad" ]]; then
        SKIP_OPENROAD=1
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

if [[ $SKIP_OPENROAD -eq 1 ]] || [[ "${GITHUB_ACTIONS:-}" == "true" && "${OPENROAD_PRE_INSTALLED:-0}" == "1" ]] || [[ -f "openroad_interface/OpenROAD/build/src/openroad" ]]; then
    echo "We likely will not need SUDO permissions for this build."
else
    echo "SUDO permissions may be required for this build. Enter SUDO password if prompted."
    sudo -v
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

## print the university name to a file in the setup scripts folder
echo "$UNIVERSITY" > "$SETUP_SCRIPTS_FOLDER"/university_name.txt


printf '>>> SCRIPT START %s\n' "$(date)"
printf 'Current directory: %s\n' "$(pwd)"

echo "UNIVERSITY set to: $UNIVERSITY"

## set home directory to codesign home directory
export OLD_HOME="$HOME"
export HOME="$(pwd)"
export PATH="$HOME/.local/bin:$(echo "$PATH")"
export CMAKE_PREFIX_PATH="$HOME/.local"

## for cmu setup, set tmp directory to local directory to avoid filling system tmp
if [ "$UNIVERSITY" = "cmu" ]; then
    export TMPDIR="$HOME/.tmp"
    export TEMP="$TMPDIR"
    export TEMPDIR="$TMPDIR"
    export TMP="$TMPDIR"
    export PYTHONPYCACHEPREFIX="$TMPDIR/__pycache__"
    export CONDA_PKGS_DIRS="$TMPDIR/conda_pkgs"
    export PIP_CACHE_DIR="$TMPDIR/pip_cache"
    mkdir -p "$TMPDIR"
    echo "Set TMPDIR to $TMPDIR"
fi

################## INSTALL OPENROAD ##################
if [[ $SKIP_OPENROAD -eq 1 ]]; then
    echo "Skipping OpenROAD installation (--skip-openroad flag set)."
else
    git submodule update --init --recursive openroad_interface/OpenROAD

    if  [[ "${GITHUB_ACTIONS:-}" == "true" && "${OPENROAD_PRE_INSTALLED:-0}" == "1" ]]; then
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
                        bash "$SETUP_SCRIPTS_FOLDER"/openroad_install_rhel8.sh
                    ;;
                    *"Rocky Linux release 9"*|*"Red Hat Enterprise Linux release 9"*)
                        bash "$SETUP_SCRIPTS_FOLDER"/openroad_install.sh
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

    if [[ "${GITHUB_ACTIONS:-}" == "true" && "${OPENROAD_PRE_INSTALLED:-0}" == "1"  ]]; then
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
fi

################ SET UP SCALEHLS ##################
## we want this to operate outside of conda, so do this first
source "$SETUP_SCRIPTS_FOLDER"/scale_hls_setup.sh $FORCE_FULL # setup scalehls

################### SET UP CONDA ENVIRONMENT ##################
# Check if the directory miniconda3 exists
if [ -d "miniconda3" ]; then
    export PATH="$(pwd):$PATH"
    source miniconda3/etc/profile.d/conda.sh
else   
    # Install and set up environment
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$(pwd)/miniconda3"
    export PATH="$(pwd):$PATH"
    source miniconda3/etc/profile.d/conda.sh

    ## Accept conda TOS
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

    conda env create -f "$SETUP_SCRIPTS_FOLDER"/environment_simplified.yml -y

    # create symlinks for g++-13 needed by cacti
    cd miniconda3/envs/codesign/bin
    ln -sf x86_64-conda-linux-gnu-gcc gcc-13
    ln -sf x86_64-conda-linux-gnu-g++ g++-13
    cd ../../../..
fi


if [[ $FORCE_FULL -eq 1 ]]; then
    ## update conda packages
    conda update -n base -c defaults conda -y # update conda itself
    conda config --set channel_priority strict
    conda env update -f "$SETUP_SCRIPTS_FOLDER"/environment_simplified.yml --prune # update the environment
fi

conda activate codesign # activate the codesign environment

################ SET UP STREAMHLS ##################
## StreamHLS setup needs conda to be available (setup-env.sh uses conda)
source "$SETUP_SCRIPTS_FOLDER"/streamhls_setup.sh $FORCE_FULL # setup stream hls

## update the rest of the submodules
if [[ $FORCE_FULL -eq 1 ]]; then
    git submodule update --init --recursive
fi

###############  BUILD CACTI #################3
cd src/cacti
make
cd ../..

## make verilator
source "$SETUP_SCRIPTS_FOLDER"/verilator_install.sh

## Load cad tools
if [ "$UNIVERSITY" = "stanford" ]; then
    echo "Setting up Stanford CAD tools..."
    source "$SETUP_SCRIPTS_FOLDER"/stanford_cad_tool_setup.sh
elif [ "$UNIVERSITY" = "cmu" ]; then
    echo "Setting up CMU CAD tools..."
    source "$SETUP_SCRIPTS_FOLDER"/cmu_cad_tool_setup.sh
else
    echo "Unsupported university for licensed cad tool setup: $UNIVERSITY"
    exit 1
fi

# Only copy Xauthority if we're in a different directory than the old home
if [ "$HOME" != "$OLD_HOME" ]; then
    echo "Copying Xauthority from $OLD_HOME to $HOME"
    if [ -f .Xauthority ]; then
        rm .Xauthority
        echo "Removed existing .Xauthority"
    fi
    cp "$OLD_HOME"/.Xauthority .Xauthority
    echo "Copied Xauthority from $OLD_HOME to $HOME"
fi

############### Add useful alisas ###############
alias create_checkpoint="python3 -m test.checkpoint_controller"
alias run_codesign="python3 -m src.codesign"
alias run_tech_test="python3 -m test.experiments.dennard_multi_core"

alias clean_checkpoints="rm -rf ~/test/saved_checkpoints/*"
alias clean_logs="rm -rf ~/logs/*"
alias clean_tmp="rm -rf ~/src/tmp/*"
alias clean_codesign="clean_checkpoints; clean_logs; clean_tmp"
alias run_regression="python3 -m test.regression_run"

################## SUCCESSFUL BUILD LOG ##################
if [[ $FORCE_FULL -eq 1 ]]; then
    date "+%Y-%m-%d %H:%M:%S" > "$BUILD_LOG"
fi

echo "Last full build completed successfully on $(cat $BUILD_LOG)"which

echo "Environment setup complete."