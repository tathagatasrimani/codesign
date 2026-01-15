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

    if [ -f "ampl.linux-intel64" ]; then
        echo "[setup] AMPL package found."
    else
        echo "[setup] AMPL package not found, please get a license from https://ampl.com/ampl-in-academia/"
        echo "and place it in the Stream-HLS directory."
        exit 1
    fi

    ./setup-env.sh

    ./build-llvm.sh

    ./build-streamhls.sh

    conda deactivate
    cd ..
else
    echo "[setup] Skipping Stream-HLS build is not set."
fi