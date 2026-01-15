#!/bin/bash

# Parse command line options
for arg in "$@"; do
    if [[ "$arg" == "--full" ]]; then
        FORCE_FULL=1
        break
    fi
done

if [[ $FORCE_FULL -eq 1 ]]; then
    # --- Top-level ---
    git submodule init
    sed -i "s|git@github.com:|https://github.com/|g" .gitmodules
    git submodule sync

    git submodule update --init Stream-HLS

    # --- Inside Stream-HLS ---
    cd Stream-HLS
    sed -i "s|git@github.com:|https://github.com/|g" .gitmodules
    git submodule sync
    git submodule update --init --recursive

    if [ -d "ampl.linux-intel64" ]; then
        echo "[setup] AMPL package found."
    else
        echo "[setup] AMPL package not found."
        echo "Please obtain it from https://ampl.com/ampl-in-academia/ (or your local copy),"
        echo "then paste or move the directory 'ampl.linux-intel64' into this directory:"
        pwd
        echo
        read -p "Press ENTER after you have placed 'ampl.linux-intel64' here to continue..." _

        if [ -d "ampl.linux-intel64" ]; then
            echo "[setup] AMPL package found after user copy."
        else
            echo "[setup] AMPL package still not found in $(pwd). Exiting."
            exit 1
        fi
    fi

    source setup-env.sh

    source build-llvm.sh

    source build-streamhls.sh

    conda deactivate
    cd ..
else
    echo "[setup] Skipping Stream-HLS build is not set."
fi