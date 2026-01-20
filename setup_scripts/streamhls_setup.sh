#!/bin/bash

# Parse command line options
for arg in "$@"; do
    if [[ "$arg" == "--full" || "$arg" == "1" ]]; then
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
        echo "[setup] AMPL package not found. Downloading..."
        wget -O ampl_package.tar.gz "https://cmu.box.com/shared/static/n6c6f147vefdrrsedammfqhfhteg7vyt" || {
            echo "[setup] Failed to download AMPL package. Exiting."
            exit 1
        }
        
        echo "[setup] Extracting AMPL package..."
        tar -xzf ampl_package.tar.gz || {
            echo "[setup] Failed to extract AMPL package. Exiting."
            exit 1
        }
        
        rm -f ampl_package.tar.gz
        
        if [ -d "ampl.linux-intel64" ]; then
            echo "[setup] AMPL package downloaded and extracted successfully."
        else
            echo "[setup] AMPL package directory not found after extraction. Exiting."
            exit 1
        fi
    fi

    ## automatically accept conda confirmations and Terms of Service
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
    
    sed -i 's|conda create -n streamhls python=3.11|conda create -n streamhls python=3.11 -y|' setup-env.sh

    source setup-env.sh

    ## ensure that the build uses all available CPU cores
    sed -i 's|cmake --build \. --target check-mlir|cmake --build . --target check-mlir -- -j$(nproc)|' build-llvm.sh
    source build-llvm.sh

    source build-streamhls.sh

    cd ..
else
    echo "[setup] Skipping Stream-HLS build is not set."
fi