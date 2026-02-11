#!/bin/bash

################## CHECK BUILD LOG / FORCE FULL ##################

SETUP_SCRIPTS_FOLDER="$(pwd)/setup_scripts"

BUILD_LOG="$SETUP_SCRIPTS_FOLDER/build.log"
FORCE_FULL=0
SKIP_OPENROAD=0

# Start timer
start_time=$(date +%s)

record_full_build_metadata() {
    local build_time root_commit
    build_time=$(date "+%Y-%m-%d %H:%M:%S")
    root_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

    {
        echo "build_time: $build_time"
        echo "root_commit: $root_commit"
        echo "submodules:"

        if git config --file .gitmodules --get-regexp 'submodule\..*\.path' >/dev/null 2>&1; then
            git submodule foreach --recursive --quiet '
                sub_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
                printf "  - %s: %s\n" "$path" "$sub_commit"
            '
        else
            echo "  - none"
        fi
    } > "$BUILD_LOG"
}

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
    # Prompt for AMPL license UUID (using /dev/tty to work with gui_install.py)
    # echo "Enter AMPL license UUID (press Enter to skip): " > /dev/tty
    # read AMPL_LICENSE_UUID </dev/tty
    
    # if [[ -n "$AMPL_LICENSE_UUID" ]]; then
    #     echo "AMPL license UUID received. Activation will occur after setup completes."
    # else
    #     echo "Skipping AMPL license activation."
    # fi
    
    echo "SUDO permissions may be required for this build. Enter SUDO password if prompted."
    sudo -v
fi
echo "Thank you for entering your sudo password if prompted."
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

## ensure that git is set to fetch submodules in parallel (faster)
git config --global fetch.parallel $(nproc)
git config --global submodule.fetchJobs $(nproc)

################## INSTALL OPENROAD ##################
echo "STARTING STEP 1: OPENROAD INSTALLATION"
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
echo "COMPLETED STEP 1: OPENROAD INSTALLATION"

################ SET UP SCALEHLS ##################
echo "STARTING STEP 2: SCALEHLS SETUP"
## we want this to operate outside of conda, so do this first
source "$SETUP_SCRIPTS_FOLDER"/scale_hls_setup.sh $FORCE_FULL # setup scalehls

echo "COMPLETED STEP 2: SCALEHLS SETUP"

################### SET UP CONDA ENVIRONMENT ##################
echo "STARTING STEP 3: CONDA ENVIRONMENT SETUP"
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
fi


if [[ $FORCE_FULL -eq 1 ]]; then
    ## update conda packages
    conda update -n base -c defaults conda -y # update conda itself
    conda config --set channel_priority strict
    conda env update -f "$SETUP_SCRIPTS_FOLDER"/environment_simplified.yml --prune # update the environment
fi

conda activate codesign # activate the codesign environment

echo "COMPLETED STEP 3: CONDA ENVIRONMENT SETUP"

################ SET UP STREAMHLS ##################
echo "STARTING STEP 4: STREAMHLS SETUP"
## StreamHLS setup needs conda to be available (setup-env.sh uses conda). 
## Note that we run this script with 'bash' since we will source the streamhls environment separately when we need it.
bash "$SETUP_SCRIPTS_FOLDER"/streamhls_setup.sh $FORCE_FULL # setup stream hls

echo "COMPLETED STEP 4: STREAMHLS SETUP"

echo "STARTING STEP 5: SUBMODULE UPDATE"
## update the rest of the submodules
if [[ $FORCE_FULL -eq 1 ]]; then
    git submodule update --init --recursive
fi
echo "COMPLETED STEP 5: SUBMODULE UPDATE"

###############  BUILD VERILATOR #################
echo "STARTING STEP 6: VERILATOR BUILD"
source "$SETUP_SCRIPTS_FOLDER"/verilator_install.sh
echo "COMPLETED STEP 6: VERILATOR BUILD"

############### HANDLE XAUTHORITY #################
echo "STARTING STEP 7: XAUTHORITY HANDLING"
# Only copy Xauthority if we're in a different directory than the old home
if [ "$HOME" != "$OLD_HOME" ]; then
    echo "Copying Xauthority from $OLD_HOME to $HOME"
    if [ -f .Xauthority ]; then
        rm .Xauthority
        echo "Removed existing .Xauthority"
    fi
    if [ -f "$OLD_HOME"/.Xauthority ]; then
        cp "$OLD_HOME"/.Xauthority .Xauthority
        echo "Copied Xauthority from $OLD_HOME to $HOME"
    else
        echo "No .Xauthority file found in $OLD_HOME"
    fi
    
fi
echo "COMPLETED STEP 7: XAUTHORITY HANDLING"

############### Add useful aliases ###############
echo "STARTING STEP 8: ADDING USEFUL ALIASES"
alias create_checkpoint="python3 -m test.checkpoint_controller"
alias run_codesign="python3 -m src.codesign"

alias clean_checkpoints="rm -rf ~/test/saved_checkpoints/*"
alias clean_logs="rm -rf ~/logs/*"
alias clean_tmp="rm -rf ~/src/tmp/*"
alias clean_codesign="clean_checkpoints; clean_logs; clean_tmp"
alias run_regression="python3 -m test.regression_run"
alias run_sweep="python3 -m src.hardware_model.tech_models.tech_library.sweep_tech_codesign"
echo "COMPLETED STEP 8: ADDING USEFUL ALIASES"
################## SUCCESSFUL BUILD LOG ##################
if [[ $FORCE_FULL -eq 1 ]]; then
    record_full_build_metadata
fi

if [[ -f "$BUILD_LOG" ]]; then
    echo "Last full build metadata:"
    cat "$BUILD_LOG"
fi

echo "ENVIRONMENT SETUP COMPLETE"

############### ACTIVATE AMPL LICENSE ###############
# # Activate AMPL license if UUID was provided at the beginning
# if [[ -n "$AMPL_LICENSE_UUID" ]]; then
#     echo "STARTING AMPL LICENSE ACTIVATION"
#     AMPL_DIR="$(pwd)/Stream-HLS/ampl.linux-intel64"
    
#     # Create directory if it doesn't exist
#     mkdir -p "$AMPL_DIR"
    
#     echo "Saving AMPL license..."
#     echo "$AMPL_LICENSE_UUID" > "$AMPL_DIR/ampl.lic"
#     echo "AMPL license saved to $AMPL_DIR/ampl.lic"
    
#     # Activate the AMPL license
#     if [ -f "$AMPL_DIR/ampl" ]; then
#         echo "Activating AMPL license..."
#         cd "$AMPL_DIR"
#         ./ampl <<EOF
# shell "amplkey activate --uuid $AMPL_LICENSE_UUID";
# exit;
# EOF
#         cd - > /dev/null
#         echo "AMPL license activated successfully."
#     else
#         echo "Warning: AMPL executable not found at $AMPL_DIR/ampl"
#         echo "License saved but activation skipped."
#     fi
#     echo "COMPLETED AMPL LICENSE ACTIVATION"
# fi

# End timer
end_time=$(date +%s)

# Calculate duration
duration=$((end_time - start_time))

# Convert to minutes and seconds
minutes=$((duration / 60))
seconds=$((duration % 60))

# Print duration
printf "\nElapsed time: %d minutes and %d seconds\n" $minutes $seconds