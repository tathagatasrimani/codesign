#!/bin/bash

################## EXPORT-IP ONLY SETUP ##################
# This setup intentionally installs only what is required for:
#   Kernel -> ScaleHLS/StreamHLS DSE -> Vitis HLS export_design (IP export)
# It skips OpenROAD, CACTI build, Verilator build, and end-of-build auto-tests.

SETUP_SCRIPTS_FOLDER="$(pwd)/setup_scripts"

BUILD_LOG="$SETUP_SCRIPTS_FOLDER/build_export_ip.log"
FORCE_FULL=0
USE_MAX_PARALLEL=0
MAX_PARALLEL_CORES=24

# Start timer
start_time=$(date +%s)

record_export_ip_build_metadata() {
    local build_time root_commit
    build_time=$(date "+%Y-%m-%d %H:%M:%S")
    root_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

    {
        echo "build_mode: export_ip_only"
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
    elif [[ "$arg" == "--max_parallel_install" ]]; then
        USE_MAX_PARALLEL=1
    elif [[ "$arg" == "--skip-openroad" ]]; then
        echo "--skip-openroad ignored: OpenROAD is always skipped in export-IP setup."
    fi
done

if [[ $FORCE_FULL -eq 0 ]]; then
    if [[ ! -f "$BUILD_LOG" ]]; then
        echo "No export-IP build log found, forcing full export-IP build."
        FORCE_FULL=1
    fi
fi

if [[ $FORCE_FULL -eq 1 ]]; then
    echo ">>> Performing FULL export-IP setup"
else
    echo ">>> Performing incremental export-IP setup"
fi

TOTAL_CORES=$(nproc 2>/dev/null || echo 0)
if [[ $TOTAL_CORES -le 0 ]]; then
    TOTAL_CORES=1
fi

if [[ $USE_MAX_PARALLEL -eq 1 ]]; then
    TARGET_CORES=$TOTAL_CORES
else
    TARGET_CORES=$((TOTAL_CORES / 2))
    if [[ $TARGET_CORES -lt 1 ]]; then
        TARGET_CORES=1
    fi
fi

if [[ $TARGET_CORES -gt $MAX_PARALLEL_CORES ]]; then
    TARGET_CORES=$MAX_PARALLEL_CORES
fi

################## AMPL UUID LICENSE PROMPT ##################

AMPL_UUID_FILE="$(pwd)/ampl_uuid.txt"

prompt_and_store_ampl_uuid() {
    local ampl_uuid
    echo -n "Please enter your AMPL UUID license: " > /dev/tty
    read ampl_uuid < /dev/tty
    echo "$ampl_uuid" > "$AMPL_UUID_FILE"
    echo "AMPL UUID saved to: $AMPL_UUID_FILE"
    echo ""
}

# Keep same behavior as original: ask during full install.
if [[ $FORCE_FULL -eq 1 ]]; then
    if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
        echo "Using $TARGET_CORES cores for this build."
    fi
    prompt_and_store_ampl_uuid
fi

################## NO-SUDO POLICY FOR FPGA SETUP ##################

# FPGA export-IP setup is intentionally user-space only.
# Helper scripts must not request or depend on sudo in this flow.
export FPGA_NO_SUDO_INSTALL=1
echo "FPGA export-IP setup will run without sudo."

# Guard against accidental sudo calls in downstream sourced scripts.
sudo() {
    echo "[no-sudo] sudo is disabled for FPGA export-IP setup."
    return 1
}
export -f sudo

################## PARSE UNIVERSITY ARGUMENT ##################

host=$(hostname)

if [[ "$host" == *stanford* ]]; then
    export UNIVERSITY="stanford"
elif [[ "$host" == *cmu* ]]; then
    export UNIVERSITY="cmu"
else
    echo "Hostname is '$host' and does not contain 'stanford' or 'cmu'."
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

# Persist university selection for run-time setup script selection in codesign
# (used by src/codesign.py when sourcing per-university Vitis setup scripts).
echo "$UNIVERSITY" > "$SETUP_SCRIPTS_FOLDER/university_name.txt"

printf '>>> SCRIPT START %s\n' "$(date)"
printf 'Current directory: %s\n' "$(pwd)"
echo "UNIVERSITY set to: $UNIVERSITY"

## set home directory to codesign home directory
export OLD_HOME="$HOME"
export HOME="$(pwd)"
export PATH="$HOME/.local/bin:$(echo "$PATH")"
export CMAKE_PREFIX_PATH="$HOME/.local"

## for cmu setup, set tmp directory to local directory to avoid filling system tmp
if [[ "$UNIVERSITY" == "cmu" ]]; then
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
git config --global fetch.parallel "$TARGET_CORES"
git config --global submodule.fetchJobs "$TARGET_CORES"

################## SKIP OPENROAD ##################
echo "STARTING STEP 1: OPENROAD INSTALLATION"
echo "Skipping OpenROAD installation in export-IP setup."
echo "COMPLETED STEP 1: OPENROAD INSTALLATION"

################### SET UP CONDA ENVIRONMENT ##################
echo "STARTING STEP 2: CONDA ENVIRONMENT SETUP"
if [[ -d "miniconda3" ]]; then
    export PATH="$(pwd):$PATH"
    source miniconda3/etc/profile.d/conda.sh
else
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$(pwd)/miniconda3"
    export PATH="$(pwd):$PATH"
    source miniconda3/etc/profile.d/conda.sh

    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

    conda env create -f "$SETUP_SCRIPTS_FOLDER/environment_simplified.yml" -y

    cd miniconda3/envs/codesign/bin
    ln -sf x86_64-conda-linux-gnu-gcc gcc-13
    ln -sf x86_64-conda-linux-gnu-g++ g++-13
    cd ../../../..
fi

if [[ $FORCE_FULL -eq 1 ]]; then
    conda update -n base -c defaults conda -y
    conda config --set channel_priority strict
    conda env update -f "$SETUP_SCRIPTS_FOLDER/environment_simplified.yml" --prune -y
fi

conda activate codesign

# Install lld in user-space for FPGA flow when not already available.
if ! command -v lld >/dev/null 2>&1; then
    echo "lld not found in PATH; installing into conda env 'codesign' (no sudo)."
    conda install -n codesign -c conda-forge lld -y
fi
echo "COMPLETED STEP 2: CONDA ENVIRONMENT SETUP"

################## SET UP SCALEHLS ##################
echo "STARTING STEP 3: SCALEHLS SETUP"
export SCALEHLS_SKIP_SYSTEM_DEPS=1
source "$SETUP_SCRIPTS_FOLDER/scale_hls_setup.sh" "$FORCE_FULL"
echo "COMPLETED STEP 3: SCALEHLS SETUP"

################ SET UP STREAMHLS ##################
echo "STARTING STEP 4: STREAMHLS SETUP"
bash "$SETUP_SCRIPTS_FOLDER/streamhls_setup.sh" "$FORCE_FULL"

# Ensure AMPL UUID is available for activation when Stream-HLS AMPL package exists.
if [[ -d "Stream-HLS/ampl.linux-intel64" ]]; then
    if [[ ! -f "$AMPL_UUID_FILE" ]] || [[ -z "$(cat "$AMPL_UUID_FILE" 2>/dev/null)" ]]; then
        echo "AMPL UUID not found; prompting now for Stream-HLS AMPL activation."
        prompt_and_store_ampl_uuid
    fi

    UUID=$(cat "$AMPL_UUID_FILE")
    if [[ -n "$UUID" ]]; then
        cd Stream-HLS/ampl.linux-intel64
        ./ampl <<EOF
shell "amplkey activate --uuid $UUID";
exit;
EOF
        cd ../..
        echo "AMPL activation completed."
    else
        echo "AMPL UUID is empty, skipping activation."
    fi
else
    echo "Stream-HLS AMPL directory not found; skipping AMPL activation."
fi

echo "COMPLETED STEP 4: STREAMHLS SETUP"

################## SKIP EXTRA SUBMODULES ##################
echo "STARTING STEP 5: SUBMODULE UPDATE"
echo "Skipping full recursive submodule update in export-IP setup."
echo "COMPLETED STEP 5: SUBMODULE UPDATE"

################## SKIP CACTI BUILD ##################
echo "STARTING STEP 6: CACTI BUILD"
echo "Skipping CACTI build in export-IP setup."
echo "COMPLETED STEP 6: CACTI BUILD"

################## SKIP VERILATOR BUILD ##################
echo "STARTING STEP 7: VERILATOR BUILD"
echo "Skipping Verilator build in export-IP setup."
echo "COMPLETED STEP 7: VERILATOR BUILD"

################## HANDLE XAUTHORITY ##################
echo "STARTING STEP 8: XAUTHORITY HANDLING"
if [[ "$HOME" != "$OLD_HOME" ]]; then
    echo "Copying Xauthority from $OLD_HOME to $HOME"
    if [[ -f .Xauthority ]]; then
        rm .Xauthority
        echo "Removed existing .Xauthority"
    fi
    if [[ -f "$OLD_HOME/.Xauthority" ]]; then
        cp "$OLD_HOME/.Xauthority" .Xauthority
        echo "Copied Xauthority from $OLD_HOME to $HOME"
    else
        echo "No .Xauthority file found in $OLD_HOME"
    fi
fi
echo "COMPLETED STEP 8: XAUTHORITY HANDLING"

############### Add useful aliases ###############
echo "STARTING STEP 9: ADDING USEFUL ALIASES"
alias create_checkpoint="python3 -m test.checkpoint_controller"
alias run_codesign="python3 -m src.codesign"
alias run_tech_test="python3 -m test.experiments.dennard_multi_core"

alias clean_checkpoints="rm -rf ~/test/saved_checkpoints/*"
alias clean_logs="rm -rf ~/logs/*"
alias clean_tmp="rm -rf ~/src/tmp/*"
alias clean_codesign="clean_checkpoints; clean_logs; clean_tmp"
alias run_regression="python3 -m test.regression_run"
alias run_sweep="python3 -m src.hardware_model.tech_models.tech_library.sweep_tech_codesign"
echo "COMPLETED STEP 9: ADDING USEFUL ALIASES"

################## SUCCESSFUL BUILD LOG ##################
if [[ $FORCE_FULL -eq 1 ]]; then
    record_export_ip_build_metadata
fi

if [[ -f "$BUILD_LOG" ]]; then
    echo "Last export-IP build metadata:"
    cat "$BUILD_LOG"
fi

echo "EXPORT-IP ENVIRONMENT SETUP COMPLETE"
echo "BUILD COMPLETED SUCCESSFULLY."

# End timer
end_time=$(date +%s)

# Calculate duration
duration=$((end_time - start_time))

# Convert to minutes and seconds
minutes=$((duration / 60))
seconds=$((duration % 60))

# Print duration
printf "\nElapsed time: %d minutes and %d seconds\n" "$minutes" "$seconds"

# Intentionally do NOT run end-of-build auto-tests in export-IP setup mode.